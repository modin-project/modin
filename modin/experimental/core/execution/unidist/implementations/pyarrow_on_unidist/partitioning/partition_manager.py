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

"""Module houses class for tracking partitions with PyArrow storage format and unidist engine."""

from modin.experimental.core.execution.unidist.generic.partitioning.partition_manager import (
    GenericUnidistDataframePartitionManager,
)
from .axis_partition import (
    PyarrowOnUnidistDataframeColumnPartition,
    PyarrowOnUnidistDataframeRowPartition,
)
from .partition import PyarrowOnUnidistFramePartition


class PyarrowOnUnidistDataframePartitionManager(
    GenericUnidistDataframePartitionManager
):
    """
    Class for tracking partitions with PyArrow storage format and unidist engine.

    Inherits all functionality from ``GenericUnidistDataframePartitionManager`` and ``PandasDataframePartitionManager`` base
    classes.
    """

    # This object uses PyarrowOnUnidistFramePartition objects as the underlying store.
    _partition_class = PyarrowOnUnidistFramePartition
    _column_partitions_class = PyarrowOnUnidistDataframeColumnPartition
    _row_partition_class = PyarrowOnUnidistDataframeRowPartition
