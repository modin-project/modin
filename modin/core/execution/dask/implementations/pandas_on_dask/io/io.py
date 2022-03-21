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

from modin.core.io import BaseIO
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.core.execution.dask.implementations.pandas_on_dask.dataframe.dataframe import (
    PandasOnDaskDataframe,
)
from modin.core.execution.dask.implementations.pandas_on_dask.partitioning.partition import (
    PandasOnDaskDataframePartition,
)
from modin.core.io import (
    CSVDispatcher,
    JSONDispatcher,
    ParquetDispatcher,
    FeatherDispatcher,
    SQLDispatcher,
    ExcelDispatcher,
)
from modin.core.storage_formats.pandas.parsers import (
    PandasCSVParser,
    PandasJSONParser,
    PandasParquetParser,
    PandasFeatherParser,
    PandasSQLParser,
    PandasExcelParser,
)
from modin.core.execution.dask.common.engine_wrapper import DaskWrapper


class PandasOnDaskIO(BaseIO):
    """The class implements interface in ``BaseIO`` using Dask as an execution engine."""

    frame_cls = PandasOnDaskDataframe
    query_compiler_cls = PandasQueryCompiler
    build_args = dict(
        frame_cls=PandasOnDaskDataframe,
        frame_partition_cls=PandasOnDaskDataframePartition,
        query_compiler_cls=PandasQueryCompiler,
    )

    read_csv = type("", (DaskWrapper, PandasCSVParser, CSVDispatcher), build_args).read
    read_json = type(
        "", (DaskWrapper, PandasJSONParser, JSONDispatcher), build_args
    ).read
    read_parquet = type(
        "", (DaskWrapper, PandasParquetParser, ParquetDispatcher), build_args
    ).read
    # Blocked on pandas-dev/pandas#12236. It is faster to default to pandas.
    # read_hdf = type("", (DaskWrapper, PandasHDFParser, HDFReader), build_args).read
    read_feather = type(
        "", (DaskWrapper, PandasFeatherParser, FeatherDispatcher), build_args
    ).read
    read_sql = type("", (DaskWrapper, PandasSQLParser, SQLDispatcher), build_args).read
    read_excel = type(
        "", (DaskWrapper, PandasExcelParser, ExcelDispatcher), build_args
    ).read
