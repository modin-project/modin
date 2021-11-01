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
from modin.experimental.core.execution.unidist.generic.io import UnidistIO
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
from modin.experimental.core.execution.unidist.common.task_wrapper import UnidistTask
from modin.experimental.core.execution.unidist.implementations.pandas_on_unidist.partitioning.partition import (
    PandasOnUnidistDataframePartition,
)
from modin.experimental.core.execution.unidist.implementations.pandas_on_unidist.dataframe.dataframe import (
    PandasOnUnidistDataframe,
)


class PandasOnUnidistIO(UnidistIO):
    """Factory providing methods for performing I/O operations using pandas as storage format on unidist as engine."""

    frame_cls = PandasOnUnidistDataframe
    query_compiler_cls = PandasQueryCompiler
    build_args = dict(
        frame_partition_cls=PandasOnUnidistDataframePartition,
        query_compiler_cls=PandasQueryCompiler,
        frame_cls=PandasOnUnidistDataframe,
    )
    read_csv = type("", (UnidistTask, PandasCSVParser, CSVDispatcher), build_args).read
    read_fwf = type("", (UnidistTask, PandasFWFParser, FWFDispatcher), build_args).read
    read_json = type(
        "", (UnidistTask, PandasJSONParser, JSONDispatcher), build_args
    ).read
    read_parquet = type(
        "", (UnidistTask, PandasParquetParser, ParquetDispatcher), build_args
    ).read
    # Blocked on pandas-dev/pandas#12236. It is faster to default to pandas.
    # read_hdf = type("", (UnidistTask, PandasHDFParser, HDFReader), build_args).read
    read_feather = type(
        "", (UnidistTask, PandasFeatherParser, FeatherDispatcher), build_args
    ).read
    read_sql = type("", (UnidistTask, PandasSQLParser, SQLDispatcher), build_args).read
    read_excel = type(
        "", (UnidistTask, PandasExcelParser, ExcelDispatcher), build_args
    ).read
