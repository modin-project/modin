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

"""The module defines interface for an axis partition with pandas storage format and python engine."""

import pandas

from modin.core.dataframe.pandas.partitioning.axis_partition import (
    PandasDataframeAxisPartition,
)
from .partition import PandasOnPythonDataframePartition


class PandasOnPythonDataframeAxisPartition(PandasDataframeAxisPartition):
    """
    Class defines axis partition interface with pandas storage format and Python engine.

    Inherits functionality from ``PandasDataframeAxisPartition`` class.

    Parameters
    ----------
    list_of_blocks : list
        List with partition objects to create common axis partition from.
    axis : {0, 1}
        The axis on which the partitions are located.
    """

    def __init__(self, list_of_blocks, axis):
        for obj in list_of_blocks:
            obj.drain_call_queue()
        # Unwrap from PandasDataframePartition object for ease of use
        self.list_of_blocks = [obj.data for obj in list_of_blocks]
        self.axis = axis

    partition_type = PandasOnPythonDataframePartition
    instance_type = pandas.DataFrame
