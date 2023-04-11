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

"""Module houses class that implements ``BaseIO`` using Dask as an execution engine."""

import fsspec
import pandas

from modin.core.io import BaseIO
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.core.execution.dask.implementations.pandas_on_dask.dataframe import (
    PandasOnDaskDataframe,
)
from modin.core.execution.dask.implementations.pandas_on_dask.partitioning import (
    PandasOnDaskDataframePartition,
)
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
from modin.core.execution.dask.common import DaskWrapper


class PandasOnDaskIO(BaseIO):
    """The class implements interface in ``BaseIO`` using Dask as an execution engine."""

    frame_cls = PandasOnDaskDataframe
    query_compiler_cls = PandasQueryCompiler
    build_args = dict(
        frame_cls=PandasOnDaskDataframe,
        frame_partition_cls=PandasOnDaskDataframePartition,
        query_compiler_cls=PandasQueryCompiler,
    )

    def __make_read(*classes, build_args=build_args):  # noqa: GL08
        # used to reduce code duplication
        return type("", (DaskWrapper, *classes), build_args).read

    read_csv = __make_read(PandasCSVParser, CSVDispatcher)
    read_fwf = __make_read(PandasFWFParser, FWFDispatcher)
    read_json = __make_read(PandasJSONParser, JSONDispatcher)
    read_parquet = __make_read(PandasParquetParser, ParquetDispatcher)
    # Blocked on pandas-dev/pandas#12236. It is faster to default to pandas.
    # read_hdf = __make_read(PandasHDFParser, HDFReader)
    read_feather = __make_read(PandasFeatherParser, FeatherDispatcher)
    read_sql = __make_read(PandasSQLParser, SQLDispatcher)
    read_excel = __make_read(PandasExcelParser, ExcelDispatcher)

    del __make_read  # to not pollute class namespace

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
            return BaseIO.to_parquet(qc, **kwargs)

        output_path = kwargs["path"]
        client_kwargs = (kwargs.get("storage_options") or {}).get("client_kwargs", {})
        fs, url = fsspec.core.url_to_fs(output_path, client_kwargs=client_kwargs)
        fs.mkdirs(url, exist_ok=True)

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
            compression = kwargs["compression"]
            partition_idx = kw["partition_idx"]
            kwargs[
                "path"
            ] = f"{output_path}/part-{partition_idx:04d}.{compression}.parquet"
            df.to_parquet(**kwargs)
            return pandas.DataFrame()

        # Ensure that the metadata is synchronized
        qc._modin_frame._propagate_index_objs(axis=None)
        result = qc._modin_frame._partition_mgr_cls.map_axis_partitions(
            axis=1,
            partitions=qc._modin_frame._partitions,
            map_func=func,
            keep_partitioning=True,
            lengths=None,
            enumerate_partitions=True,
        )
        DaskWrapper.materialize(
            [part.list_of_blocks[0] for row in result for part in row]
        )

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

        def func(df):  # pragma: no cover
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

        # Ensure that the metadata is synchronized
        qc._modin_frame._propagate_index_objs(axis=None)
        result = qc._modin_frame.apply_full_axis(1, func, new_index=[], new_columns=[])
        result._partition_mgr_cls.wait_partitions(result._partitions.flatten())
