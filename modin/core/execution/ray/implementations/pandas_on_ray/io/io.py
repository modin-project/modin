# Licensed to Modin Development Team under one or more contributor license agreements.
# See the NOTICE file distributed with this work for additional information regarding
# copyright ownership.  The Modin Development Team licenses this file to you under the
# Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

"""The module holds the factory which performs I/O using pandas on Ray."""

import os

import pandas

from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.core.execution.ray.generic.io import RayIO
from modin.core.io import (
    CSVDispatcher,
    FWFDispatcher,
    JSONDispatcher,
    ParquetDispatcher,
    FeatherDispatcher,
    SQLDispatcher,
    ExcelDispatcher,
)
from modin.core.storage_formats.pandas.parsers import (
    PandasCSVParser,
    PandasFWFParser,
    PandasJSONParser,
    PandasParquetParser,
    PandasFeatherParser,
    PandasSQLParser,
    PandasExcelParser,
)
from modin.core.execution.ray.common.task_wrapper import RayWrapper
from modin.core.execution.ray.implementations.pandas_on_ray.partitioning.partition import (
    PandasOnRayDataframePartition,
)
from modin.core.execution.ray.implementations.pandas_on_ray.dataframe.dataframe import (
    PandasOnRayDataframe,
)


class PandasOnRayIO(RayIO):
    """Factory providing methods for performing I/O operations using pandas as storage format on Ray as engine."""

    frame_cls = PandasOnRayDataframe
    query_compiler_cls = PandasQueryCompiler
    build_args = dict(
        frame_partition_cls=PandasOnRayDataframePartition,
        query_compiler_cls=PandasQueryCompiler,
        frame_cls=PandasOnRayDataframe,
    )
    read_csv = type("", (RayWrapper, PandasCSVParser, CSVDispatcher), build_args).read
    read_fwf = type("", (RayWrapper, PandasFWFParser, FWFDispatcher), build_args).read
    read_json = type(
        "", (RayWrapper, PandasJSONParser, JSONDispatcher), build_args
    ).read
    read_parquet = type(
        "", (RayWrapper, PandasParquetParser, ParquetDispatcher), build_args
    ).read
    # Blocked on pandas-dev/pandas#12236. It is faster to default to pandas.
    # read_hdf = type("", (RayWrapper, PandasHDFParser, HDFReader), build_args).read
    read_feather = type(
        "", (RayWrapper, PandasFeatherParser, FeatherDispatcher), build_args
    ).read
    read_sql = type("", (RayWrapper, PandasSQLParser, SQLDispatcher), build_args).read
    read_excel = type(
        "", (RayWrapper, PandasExcelParser, ExcelDispatcher), build_args
    ).read

    to_csv = type("", (RayWrapper, CSVDispatcher), {}).to_csv

    @classmethod
    def to_sql(cls, qc, **kwargs):
        """
        Write records stored in the `qc` to a SQL database.

        Parameters
        ----------
        qc : BaseQueryCompiler
            The query compiler of the Modin dataframe that we want to run ``to_sql`` on.
        **kwargs : dict
            Parameters for ``pandas.to_sql(**kwargs)``.
        """
        # we first insert an empty DF in order to create the full table in the database
        # This also helps to validate the input against pandas
        # we would like to_sql() to complete only when all rows have been inserted into the database
        # since the mapping operation is non-blocking, each partition will return an empty DF
        # so at the end, the blocking operation will be this empty DF to_pandas

        empty_df = qc.getitem_row_array([0]).to_pandas().head(0)
        empty_df.to_sql(**kwargs)
        # so each partition will append its respective DF
        kwargs["if_exists"] = "append"
        columns = qc.columns

        def func(df):
            """
            Override column names in the wrapped dataframe and convert it to SQL.

            Notes
            -----
            This function returns an empty ``pandas.DataFrame`` because ``apply_full_axis``
            expects a Frame object as a result of operation (and ``to_sql`` has no dataframe result).
            """
            df.columns = columns
            df.to_sql(**kwargs)
            return pandas.DataFrame()

        result = qc._modin_frame.apply_full_axis(1, func, new_index=[], new_columns=[])
        # FIXME: we should be waiting for completion less expensievely, maybe use _modin_frame.materialize()?
        result.to_pandas()  # blocking operation

    @staticmethod
    def _to_parquet_check_support(kwargs):
        """
        Check if parallel version of `to_parquet` could be used.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed to `.to_parquet()`.

        Returns
        -------
        bool
            Whether parallel version of `to_parquet` is applicable.
        """
        path = kwargs["path"]
        compression = kwargs["compression"]
        if not isinstance(path, str):
            return False
        if any((path.endswith(ext) for ext in [".gz", ".bz2", ".zip", ".xz"])):
            return False
        if compression is None or not compression == "snappy":
            return False
        return True

    @classmethod
    def to_parquet(cls, qc, **kwargs):
        """
        Write a ``DataFrame`` to the binary parquet format.

        Parameters
        ----------
        qc : BaseQueryCompiler
            The query compiler of the Modin dataframe that we want to run `to_parquet` on.
        **kwargs : dict
            Parameters for `pandas.to_parquet(**kwargs)`.
        """
        if not cls._to_parquet_check_support(kwargs):
            return RayIO.to_parquet(qc, **kwargs)

        def func(df, **kw):
            """
            Dump a chunk of rows as parquet, then save them to target maintaining order.

            Parameters
            ----------
            df : pandas.DataFrame
                A chunk of rows to write to a parquet file.
            **kw : dict
                Arguments to pass to ``pandas.to_parquet(**kwargs)`` plus an extra argument
                `partition_idx` serving as chunk index to maintain rows order.
            """
            output_path = kwargs["path"]
            compression = kwargs["compression"]
            partition_idx = kw["partition_idx"]
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            kwargs[
                "path"
            ] = f"{output_path}/part-{partition_idx:04d}.{compression}.parquet"
            df.to_parquet(**kwargs)
            return pandas.DataFrame()

        result = qc._modin_frame._partition_mgr_cls.map_axis_partitions(
            axis=1,
            partitions=qc._modin_frame._partitions,
            map_func=func,
            keep_partitioning=True,
            lengths=None,
            enumerate_partitions=True,
        )
        RayWrapper.materialize([partition.future for partition in result.flatten()])
