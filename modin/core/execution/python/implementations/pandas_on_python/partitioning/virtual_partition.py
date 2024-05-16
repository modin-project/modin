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

from modin.core.dataframe.pandas.partitioning.axis_partition import (
    PandasDataframeAxisPartition,
)
from modin.utils import _inherit_docstrings

from .partition import PandasOnPythonDataframePartition


class PandasOnPythonDataframeAxisPartition(PandasDataframeAxisPartition):
    """
    Class defines axis partition interface with pandas storage format and Python engine.

    Inherits functionality from ``PandasDataframeAxisPartition`` class.

    Parameters
    ----------
    list_of_partitions : Union[list, PandasOnPythonDataframePartition]
        List of ``PandasOnPythonDataframePartition`` and
        ``PandasOnPythonDataframeVirtualPartition`` objects, or a single
        ``PandasOnPythonDataframePartition``.
    get_ip : bool, default: False
        Whether to get node IP addresses to conforming partitions or not.
    full_axis : bool, default: True
        Whether or not the virtual partition encompasses the whole axis.
    call_queue : list, optional
        A list of tuples (callable, args, kwargs) that contains deferred calls.
    length : int, optional
        Length, or reference to length, of wrapped ``pandas.DataFrame``.
    width : int, optional
        Width, or reference to width, of wrapped ``pandas.DataFrame``.
    """

    partition_type = PandasOnPythonDataframePartition


@_inherit_docstrings(PandasOnPythonDataframeAxisPartition)
class PandasOnPythonDataframeColumnPartition(PandasOnPythonDataframeAxisPartition):
    axis = 0


@_inherit_docstrings(PandasOnPythonDataframeAxisPartition)
class PandasOnPythonDataframeRowPartition(PandasOnPythonDataframeAxisPartition):
    axis = 1
