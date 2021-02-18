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

from modin.backends.pyarrow.query_compiler import PyarrowQueryCompiler
from modin.engines.ray.generic.io import RayIO
from modin.experimental.engines.pyarrow_on_ray.frame.data import PyarrowOnRayFrame
from modin.experimental.engines.pyarrow_on_ray.frame.partition import (
    PyarrowOnRayFramePartition,
)
from modin.backends.pyarrow.parsers import PyarrowCSVParser
from modin.engines.ray.task_wrapper import RayTask
from modin.engines.base.io import CSVDispatcher


class PyarrowOnRayCSVDispatcher(RayTask, PyarrowCSVParser, CSVDispatcher):
    frame_cls = PyarrowOnRayFrame
    frame_partition_cls = PyarrowOnRayFramePartition
    query_compiler_cls = PyarrowQueryCompiler


class PyarrowOnRayIO(RayIO):
    frame_cls = PyarrowOnRayFrame
    frame_partition_cls = PyarrowOnRayFramePartition
    query_compiler_cls = PyarrowQueryCompiler
    csv_reader = PyarrowOnRayCSVDispatcher

    read_parquet_remote_task = None
    read_hdf_remote_task = None
    read_feather_remote_task = None
    read_sql_remote_task = None
