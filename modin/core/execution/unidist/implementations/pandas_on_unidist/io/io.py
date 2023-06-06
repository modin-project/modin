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

"""The module holds the factory which performs I/O using pandas on unidist."""

from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.core.execution.unidist.generic.io import UnidistIO
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
from modin.core.execution.unidist.common import UnidistWrapper, SignalActor
from ..dataframe import PandasOnUnidistDataframe
from ..partitioning import PandasOnUnidistDataframePartition


class PandasOnUnidistIO(UnidistIO):
    """Factory providing methods for performing I/O operations using pandas as storage format on unidist as engine."""

    frame_cls = PandasOnUnidistDataframe
    query_compiler_cls = PandasQueryCompiler
    build_args = dict(
        frame_partition_cls=PandasOnUnidistDataframePartition,
        query_compiler_cls=PandasQueryCompiler,
        frame_cls=PandasOnUnidistDataframe,
        base_io=UnidistIO,
        signal_actor=SignalActor,
    )

    def __make_read(*classes, build_args=build_args):
        # used to reduce code duplication
        return type("", (UnidistWrapper, *classes), build_args).read

    def __make_write(*classes, build_args=build_args):
        # used to reduce code duplication
        return type("", (UnidistWrapper, *classes), build_args).write

    read_csv = __make_read(PandasCSVParser, CSVDispatcher)
    to_csv = __make_write(CSVDispatcher)
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

    del __make_read  # to not pollute class namespace
    del __make_write  # to not pollute class namespace
