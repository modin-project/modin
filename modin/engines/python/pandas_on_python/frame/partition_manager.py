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

from modin.engines.base.frame.partition_manager import BaseFrameManager
from .axis_partition import (
    PandasOnPythonFrameColumnPartition,
    PandasOnPythonFrameRowPartition,
)
from .partition import PandasOnPythonFramePartition


class PythonFrameManager(BaseFrameManager):
    """This method implements the interface in `BaseFrameManager`."""

    # This object uses RayRemotePartition objects as the underlying store.
    _partition_class = PandasOnPythonFramePartition
    _column_partitions_class = PandasOnPythonFrameColumnPartition
    _row_partition_class = PandasOnPythonFrameRowPartition
