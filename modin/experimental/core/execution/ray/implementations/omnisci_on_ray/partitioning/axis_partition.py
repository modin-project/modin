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

"""
Module houses axis partition classes with OmniSci backend and Ray engine.

These classes are intended to perform operations on the data rows and columns.
"""

from modin.core.dataframe.pandas.partitioning.axis_partition import (
    PandasFrameAxisPartition,
)
from .partition import OmnisciOnRayFramePartition

import ray


class OmnisciOnRayFrameAxisPartition(PandasFrameAxisPartition):
    """
    Base class for axis partition classes with OmniSci backend and Ray engine.

    Inherits functionality from ``PandasFrameAxisPartition`` class, that
    contains all of the implementation for these backend and engine.

    Parameters
    ----------
    list_of_blocks : list-like
        Partitions of the axis.
    """

    def __init__(self, list_of_blocks):
        for obj in list_of_blocks:
            obj.drain_call_queue()
        # Unwrap from PandasFramePartition object for ease of use
        self.list_of_blocks = [obj.oid for obj in list_of_blocks]

    partition_type = OmnisciOnRayFramePartition
    instance_type = ray.ObjectRef


class OmnisciOnRayFrameColumnPartition(OmnisciOnRayFrameAxisPartition):
    """
    Column partition class with OmniSci backend and Ray engine.

    Inherits functionality from ``OmnisciOnRayFrameAxisPartition`` class, that
    contains all of the implementation for this class, and this class defines
    the axis to perform the computation over.

    Parameters
    ----------
    list_of_blocks : list-like
        Partitions of the axis.
    """

    axis = 0


class OmnisciOnRayFrameRowPartition(OmnisciOnRayFrameAxisPartition):
    """
    Row partition class with OmniSci backend and Ray engine.

    Inherits functionality from ``OmnisciOnRayFrameAxisPartition`` class, that
    contains all of the implementation for this class, and this class defines
    the axis to perform the computation over.

    Parameters
    ----------
    list_of_blocks : list-like
        Partitions of the axis.
    """

    axis = 1
