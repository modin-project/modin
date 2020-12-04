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

from modin.backends.pandas.query_compiler import PandasQueryCompiler
from modin.engines.ray.generic.io import RayIO
from modin.engines.base.io import (
    CSVDispatcher,
    FWFDispatcher,
    JSONDispatcher,
    ParquetDispatcher,
    FeatherDispatcher,
    SQLDispatcher,
    ExcelDispatcher,
)
from modin.backends.pandas.parsers import (
    PandasCSVParser,
    PandasFWFParser,
    PandasJSONParser,
    PandasParquetParser,
    PandasFeatherParser,
    PandasSQLParser,
    PandasExcelParser,
)
from modin.engines.ray.task_wrapper import RayTask
from modin.engines.ray.pandas_on_ray.frame.partition import PandasOnRayFramePartition
from modin.engines.ray.pandas_on_ray.frame.data import PandasOnRayFrame


class PandasOnRayIO(RayIO):

    frame_cls = PandasOnRayFrame
    query_compiler_cls = PandasQueryCompiler
    build_args = dict(
        frame_partition_cls=PandasOnRayFramePartition,
        query_compiler_cls=PandasQueryCompiler,
        frame_cls=PandasOnRayFrame,
    )
    read_csv = type("", (RayTask, PandasCSVParser, CSVDispatcher), build_args).read
    read_fwf = type("", (RayTask, PandasFWFParser, FWFDispatcher), build_args).read
    read_json = type("", (RayTask, PandasJSONParser, JSONDispatcher), build_args).read
    read_parquet = type(
        "", (RayTask, PandasParquetParser, ParquetDispatcher), build_args
    ).read
    # Blocked on pandas-dev/pandas#12236. It is faster to default to pandas.
    # read_hdf = type("", (RayTask, PandasHDFParser, HDFReader), build_args).read
    read_feather = type(
        "", (RayTask, PandasFeatherParser, FeatherDispatcher), build_args
    ).read
    read_sql = type("", (RayTask, PandasSQLParser, SQLDispatcher), build_args).read
    read_excel = type(
        "", (RayTask, PandasExcelParser, ExcelDispatcher), build_args
    ).read
