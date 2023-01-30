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

import pandas

from enum import Enum
from pandas.api.types import is_scalar


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


def join_columns(left, right, left_on, right_on, suffixes):
    """
    Compute resulting columns for the two dataframes being merged.

    Parameters
    ----------
    left : pandas.Index
        Columns of the left frame to join.
    right : pandas.Index
        Columns of the right frame to join.
    left_on : list-like or scalar
        Column names on which the frames are joined in the left DataFrame.
    right_on : list-like or scalar
        Column names on which the frames are joined in the right DataFrame.
    suffixes : tuple(str, str)
        A 2-length sequence containing suffixes to append to the intersected columns.

    Returns
    -------
    pandas.Index, dict[IndexLabel -> IndexLabel], dict[IndexLabel -> IndexLabel]
        Returns columns for the resulting frame and mappings of old to new column
        names for `left` and `right` accordingly.
    """
    if is_scalar(left_on):
        left_on = [left_on]
    if is_scalar(right_on):
        right_on = [right_on]

    left_conflicts = set(left) & (set(right) - set(right_on))
    right_conflicts = set(right) & (set(left) - set(left_on))
    conflicting_cols = left_conflicts | right_conflicts

    def _get_new_name(col, suffix):
        if col in conflicting_cols:
            return (
                (f"{col[0]}{suffix}", *col[1:])
                if isinstance(col, tuple)
                else f"{col}{suffix}"
            )
        else:
            return col

    left_renamer = {}
    right_renamer = {}
    new_left = []
    new_right = []

    for col in left:
        new_name = _get_new_name(col, suffixes[0])
        new_left.append(new_name)
        left_renamer[col] = new_name

    for col in right:
        if col not in left_on or col not in right_on:
            new_name = _get_new_name(col, suffixes[1])
            new_right.append(new_name)
            right_renamer[col] = new_name

    new_columns = pandas.Index(new_left).append(pandas.Index(new_right))
    return new_columns, left_renamer, right_renamer
