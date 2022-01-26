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

"""The module defines interface for a virtual partition with pandas storage format and python engine."""

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
    full_axis : bool, default: True
        Whether or not the virtual partition encompasses the whole axis.
    """

    def __init__(self, list_of_blocks, full_axis: bool = True):
        if not full_axis:
            raise NotImplementedError(
                "Pandas on Python execution requires full-axis partitions."
            )
        for obj in list_of_blocks:
            obj.drain_call_queue()
        # Unwrap from PandasDataframePartition object for ease of use
        self.list_of_blocks = [obj.data for obj in list_of_blocks]

    partition_type = PandasOnPythonDataframePartition
    instance_type = pandas.DataFrame


class PandasOnPythonDataframeColumnPartition(PandasOnPythonDataframeAxisPartition):
    """
    The column partition implementation for pandas storage format and Python engine.

    All of the implementation for this class is in the ``PandasOnPythonDataframeAxisPartition``
    parent class, and this class defines the axis to perform the computation over.

    Parameters
    ----------
    list_of_blocks : list
        List with partition objects to create common axis partition from.
    full_axis : bool, default: True
        Whether or not the virtual partition encompasses the whole axis.
    """

    axis = 0


class PandasOnPythonDataframeRowPartition(PandasOnPythonDataframeAxisPartition):
    """
    The row partition implementation for pandas storage format and Python engine.

    All of the implementation for this class is in the ``PandasOnPythonDataframeAxisPartition``
    parent class, and this class defines the axis to perform the computation over.

    Parameters
    ----------
    list_of_blocks : list
        List with partition objects to create common axis partition from.
    full_axis : bool, default: True
        Whether or not the virtual partition encompasses the whole axis.
    """

    axis = 1
