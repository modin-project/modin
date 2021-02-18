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

from modin.engines.base.io import BaseIO
from modin.backends.pandas.query_compiler import PandasQueryCompiler
from modin.engines.dask.pandas_on_dask.frame.data import PandasOnDaskFrame
from modin.engines.dask.pandas_on_dask.frame.partition import PandasOnDaskFramePartition
from modin.engines.base.io import (
    CSVDispatcher,
    JSONDispatcher,
    ParquetDispatcher,
    FeatherDispatcher,
    SQLDispatcher,
    ExcelDispatcher,
)
from modin.backends.pandas.parsers import (
    PandasCSVParser,
    PandasJSONParser,
    PandasParquetParser,
    PandasFeatherParser,
    PandasSQLParser,
    PandasExcelParser,
)
from modin.engines.dask.task_wrapper import DaskTask


class PandasOnDaskIO(BaseIO):

    frame_cls = PandasOnDaskFrame
    query_compiler_cls = PandasQueryCompiler
    build_args = dict(
        frame_cls=PandasOnDaskFrame,
        frame_partition_cls=PandasOnDaskFramePartition,
        query_compiler_cls=PandasQueryCompiler,
    )

    read_csv = type("", (DaskTask, PandasCSVParser, CSVDispatcher), build_args).read
    read_json = type("", (DaskTask, PandasJSONParser, JSONDispatcher), build_args).read
    read_parquet = type(
        "", (DaskTask, PandasParquetParser, ParquetDispatcher), build_args
    ).read
    # Blocked on pandas-dev/pandas#12236. It is faster to default to pandas.
    # read_hdf = type("", (DaskTask, PandasHDFParser, HDFReader), build_args).read
    read_feather = type(
        "", (DaskTask, PandasFeatherParser, FeatherDispatcher), build_args
    ).read
    read_sql = type("", (DaskTask, PandasSQLParser, SQLDispatcher), build_args).read
    read_excel = type(
        "", (DaskTask, PandasExcelParser, ExcelDispatcher), build_args
    ).read
