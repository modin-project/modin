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

import numpy as np
from distributed.client import default_client

from modin.core.execution.dask.common import DaskWrapper
from modin.core.execution.dask.implementations.pandas_on_dask.dataframe import (
    PandasOnDaskDataframe,
)
from modin.core.execution.dask.implementations.pandas_on_dask.partitioning import (
    PandasOnDaskDataframePartition,
)
from modin.core.io import (
    BaseIO,
    CSVDispatcher,
    ExcelDispatcher,
    FeatherDispatcher,
    FWFDispatcher,
    JSONDispatcher,
    ParquetDispatcher,
    SQLDispatcher,
)
from modin.core.storage_formats.pandas.parsers import (
    PandasCSVParser,
    PandasExcelParser,
    PandasFeatherParser,
    PandasFWFParser,
    PandasJSONParser,
    PandasParquetParser,
    PandasSQLParser,
)
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.distributed.dataframe.pandas.partitions import (
    from_partitions,
    unwrap_partitions,
)
from modin.experimental.core.io import (
    ExperimentalCSVGlobDispatcher,
    ExperimentalCustomTextDispatcher,
    ExperimentalGlobDispatcher,
    ExperimentalSQLDispatcher,
)
from modin.experimental.core.storage_formats.pandas.parsers import (
    ExperimentalCustomTextParser,
    ExperimentalPandasCSVGlobParser,
    ExperimentalPandasJsonParser,
    ExperimentalPandasParquetParser,
    ExperimentalPandasPickleParser,
    ExperimentalPandasXmlParser,
)
from modin.pandas.series import Series
from modin.utils import MODIN_UNNAMED_SERIES_LABEL


class PandasOnDaskIO(BaseIO):
    """The class implements interface in ``BaseIO`` using Dask as an execution engine."""

    frame_cls = PandasOnDaskDataframe
    frame_partition_cls = PandasOnDaskDataframePartition
    query_compiler_cls = PandasQueryCompiler
    build_args = dict(
        frame_cls=PandasOnDaskDataframe,
        frame_partition_cls=PandasOnDaskDataframePartition,
        query_compiler_cls=PandasQueryCompiler,
        base_io=BaseIO,
    )

    def __make_read(*classes, build_args=build_args):
        # used to reduce code duplication
        return type("", (DaskWrapper, *classes), build_args).read

    def __make_write(*classes, build_args=build_args):
        # used to reduce code duplication
        return type("", (DaskWrapper, *classes), build_args).write

    read_csv = __make_read(PandasCSVParser, CSVDispatcher)
    read_fwf = __make_read(PandasFWFParser, FWFDispatcher)
    read_json = __make_read(PandasJSONParser, JSONDispatcher)
    read_parquet = __make_read(PandasParquetParser, ParquetDispatcher)
    to_parquet = __make_write(ParquetDispatcher)
    # Blocked on pandas-dev/pandas#12236. It is faster to default to pandas.
    # read_hdf = __make_read(PandasHDFParser, HDFReader)
    read_feather = __make_read(PandasFeatherParser, FeatherDispatcher)
    read_sql = __make_read(PandasSQLParser, SQLDispatcher)
    to_sql = __make_write(SQLDispatcher)
    read_excel = __make_read(PandasExcelParser, ExcelDispatcher)

    # experimental methods that don't exist in pandas
    read_csv_glob = __make_read(
        ExperimentalPandasCSVGlobParser, ExperimentalCSVGlobDispatcher
    )
    read_parquet_glob = __make_read(
        ExperimentalPandasParquetParser, ExperimentalGlobDispatcher
    )
    to_parquet_glob = __make_write(
        ExperimentalGlobDispatcher,
        build_args={**build_args, "base_write": BaseIO.to_parquet},
    )
    read_json_glob = __make_read(
        ExperimentalPandasJsonParser, ExperimentalGlobDispatcher
    )
    to_json_glob = __make_write(
        ExperimentalGlobDispatcher,
        build_args={**build_args, "base_write": BaseIO.to_json},
    )
    read_xml_glob = __make_read(ExperimentalPandasXmlParser, ExperimentalGlobDispatcher)
    to_xml_glob = __make_write(
        ExperimentalGlobDispatcher,
        build_args={**build_args, "base_write": BaseIO.to_xml},
    )
    read_pickle_glob = __make_read(
        ExperimentalPandasPickleParser, ExperimentalGlobDispatcher
    )
    to_pickle_glob = __make_write(
        ExperimentalGlobDispatcher,
        build_args={**build_args, "base_write": BaseIO.to_pickle},
    )
    read_custom_text = __make_read(
        ExperimentalCustomTextParser, ExperimentalCustomTextDispatcher
    )
    read_sql_distributed = __make_read(
        ExperimentalSQLDispatcher, build_args={**build_args, "base_read": read_sql}
    )

    del __make_read  # to not pollute class namespace
    del __make_write  # to not pollute class namespace

    @classmethod
    def from_dask(cls, dask_obj):
        """
        Create a Modin `query_compiler` from a Dask DataFrame.

        Parameters
        ----------
        dask_obj : dask.dataframe.DataFrame
            The Dask DataFrame to convert from.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler containing data from the Dask DataFrame.
        """
        client = default_client()
        dask_fututures = client.compute(dask_obj.to_delayed())
        modin_df = from_partitions(dask_fututures, axis=0)._query_compiler
        return modin_df

    @classmethod
    def to_dask(cls, modin_obj):
        """
        Convert a Modin DataFrame/Series to a Dask DataFrame/Series.

        Parameters
        ----------
        modin_obj : modin.pandas.DataFrame, modin.pandas.Series
            The Modin DataFrame/Series to convert.

        Returns
        -------
        dask.dataframe.DataFrame or dask.dataframe.Series
            Converted object with type depending on input.
        """
        from dask.dataframe import from_delayed

        partitions = unwrap_partitions(modin_obj, axis=0)

        # partiotions must be converted to pandas Series
        if isinstance(modin_obj, Series):
            client = default_client()

            def df_to_series(df):
                series = df[df.columns[0]]
                if df.columns[0] == MODIN_UNNAMED_SERIES_LABEL:
                    series.name = None
                return series

            partitions = [client.submit(df_to_series, part) for part in partitions]

        return from_delayed(partitions)

    @classmethod
    def from_map(cls, func, iterable, *args, **kwargs):
        """
        Create a Modin `query_compiler` from a map function.

        This method will construct a Modin `query_compiler` split by row partitions.
        The number of row partitions matches the number of elements in the iterable object.

        Parameters
        ----------
        func : callable
            Function to map across the iterable object.
        iterable : Iterable
            An iterable object.
        *args : tuple
            Positional arguments to pass in `func`.
        **kwargs : dict
            Keyword arguments to pass in `func`.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler containing data returned by map function.
        """
        func = cls.frame_cls._partition_mgr_cls.preprocess_func(func)
        partitions = np.array(
            [
                [
                    cls.frame_partition_cls(
                        DaskWrapper.deploy(
                            func,
                            f_args=(obj,) + args,
                            f_kwargs=kwargs,
                            return_pandas_df=True,
                        )
                    )
                ]
                for obj in iterable
            ]
        )
        return cls.query_compiler_cls(cls.frame_cls(partitions))
