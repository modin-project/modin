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
Function module provides template for a query compiler methods for a set of common operations.

This interface is written in a general way, therefore, if the function being implemented
requires details when processing parameters (for example, fallback to pandas case) or an
additional level of processing the created frame, then `Function.register` call is no
longer enough and the usual function creation is required (via `def`).
"""

from .function import Function
from .mapfunction import MapFunction
from .mapreducefunction import MapReduceFunction
from .reductionfunction import ReductionFunction
from .foldfunction import FoldFunction
from .binary_function import BinaryFunction
from .groupby_function import GroupbyReduceFunction, GROUPBY_REDUCE_FUNCTIONS

__all__ = [
    "Function",
    "MapFunction",
    "MapReduceFunction",
    "ReductionFunction",
    "FoldFunction",
    "BinaryFunction",
    "GroupbyReduceFunction",
    "GROUPBY_REDUCE_FUNCTIONS",
]
