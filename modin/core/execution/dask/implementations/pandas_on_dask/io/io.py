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


class PandasOnDaskIO(BaseIO):
    """The class implements interface in ``BaseIO`` using Dask as an execution engine."""

    frame_cls = PandasOnDaskDataframe
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
