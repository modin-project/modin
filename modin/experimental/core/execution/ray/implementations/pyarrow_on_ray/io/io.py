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

"""Module for housing IO classes with PyArrow storage format and Ray engine."""

from modin.experimental.core.storage_formats.pyarrow import (
    PyarrowQueryCompiler,
    PyarrowCSVParser,
)
from modin.core.execution.ray.generic.io import RayIO
from modin.experimental.core.execution.ray.implementations.pyarrow_on_ray.dataframe.dataframe import (
    PyarrowOnRayDataframe,
)
from modin.experimental.core.execution.ray.implementations.pyarrow_on_ray.partitioning.partition import (
    PyarrowOnRayDataframePartition,
)
from modin.core.execution.ray.common import RayTask
from modin.core.io import CSVDispatcher


class PyarrowOnRayCSVDispatcher(RayTask, PyarrowCSVParser, CSVDispatcher):
    """Class handles utils for reading `.csv` files with PyArrow storage format and Ray engine."""

    frame_cls = PyarrowOnRayDataframe
    frame_partition_cls = PyarrowOnRayDataframePartition
    query_compiler_cls = PyarrowQueryCompiler


class PyarrowOnRayIO(RayIO):
    """Class for storing IO functions operated on PyArrow storage format and Ray engine."""

    frame_cls = PyarrowOnRayDataframe
    frame_partition_cls = PyarrowOnRayDataframePartition
    query_compiler_cls = PyarrowQueryCompiler
    csv_reader = PyarrowOnRayCSVDispatcher

    read_parquet_remote_task = None
    read_hdf_remote_task = None
    read_feather_remote_task = None
    read_sql_remote_task = None
