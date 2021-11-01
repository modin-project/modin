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

"""Module for housing IO classes with PyArrow storage format and unidist engine."""

from modin.core.storage_formats.pyarrow.query_compiler import PyarrowQueryCompiler
from modin.experimental.core.execution.unidist.generic.io import UnidistIO
from modin.experimental.core.execution.unidist.implementations.pyarrow_on_unidist.dataframe.dataframe import (
    PyarrowOnUnidistDataframe,
)
from modin.experimental.core.execution.unidist.implementations.pyarrow_on_unidist.partitioning.partition import (
    PyarrowOnUnidistDataframePartition,
)
from modin.core.storage_formats.pyarrow.parsers import PyarrowCSVParser
from modin.experimental.core.execution.unidist.common.task_wrapper import UnidistTask
from modin.core.io import CSVDispatcher


class PyarrowOnUnidistCSVDispatcher(UnidistTask, PyarrowCSVParser, CSVDispatcher):
    """Class handles utils for reading `.csv` files with PyArrow storage format and unidist engine."""

    frame_cls = PyarrowOnUnidistDataframe
    frame_partition_cls = PyarrowOnUnidistDataframePartition
    query_compiler_cls = PyarrowQueryCompiler


class PyarrowOnUnidistIO(UnidistIO):
    """Class for storing IO functions operated on PyArrow storage format and unidist engine."""

    frame_cls = PyarrowOnUnidistDataframe
    frame_partition_cls = PyarrowOnUnidistDataframePartition
    query_compiler_cls = PyarrowQueryCompiler
    csv_reader = PyarrowOnUnidistCSVDispatcher

    read_parquet_remote_task = None
    read_hdf_remote_task = None
    read_feather_remote_task = None
    read_sql_remote_task = None
