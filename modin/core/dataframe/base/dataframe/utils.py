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
Module contains useful enums for Modin.

Axis is an enum that represents the `axis` argument for dataframe operations.
JoinType is an enum that represents the `join_type` or `how` argument for the join algebra operator.
"""

from enum import Enum


class Axis(Enum):  # noqa: PR01
    """
    An enum that represents the `axis` argument provided to the algebra operators.

    The enum has 3 values - ROW_WISE to represent the row axis, COL_WISE to represent the
    column axis, and CELL_WISE to represent no axis. ROW_WISE operations iterate over the rows
    COL_WISE operations over the columns, and CELL_WISE operations over any of the partitioning
    schemes that are supported in Modin (row-wise, column-wise, or block-wise).
    """

    ROW_WISE = 0
    COL_WISE = 1
    CELL_WISE = None


class JoinType(Enum):  # noqa: PR01
    """
    An enum that represents the `join_type` argument provided to the algebra operators.

    The enum has 4 values - INNER to represent inner joins, LEFT to represent left joins, RIGHT to
    represent right joins, and OUTER to represent outer joins.
    """

    INNER = "inner"
    LEFT = "left"
    RIGHT = "right"
    OUTER = "outer"
