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
from typing import Dict, List, Sequence, Tuple, cast

import pandas
from pandas._typing import IndexLabel
from pandas.api.types import is_scalar
from pandas.core.dtypes.common import is_integer_dtype


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


def join_columns(
    left: pandas.Index,
    right: pandas.Index,
    left_on: IndexLabel,
    right_on: IndexLabel,
    suffixes: Tuple[str, str],
) -> Tuple[pandas.Index, Dict[IndexLabel, IndexLabel], Dict[IndexLabel, IndexLabel]]:
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
    suffixes : tuple[str, str]
        A 2-length sequence containing suffixes to append to the intersected columns.

    Returns
    -------
    pandas.Index, dict[IndexLabel -> IndexLabel], dict[IndexLabel -> IndexLabel]
        Returns columns for the resulting frame and mappings of old to new column
        names for `left` and `right` accordingly.

    Raises
    ------
    NotImplementedError
        Raised when one of the keys to join is an index level, pandas behaviour is really
        complicated in this case, so we're not supporting this case for now.
    """
    # using `cast` to make `mypy` acknowledged that the variable now ensured to be `Sequence[IndexLabel]`
    left_on = cast(Sequence[IndexLabel], [left_on] if is_scalar(left_on) else left_on)
    right_on = cast(
        Sequence[IndexLabel], [right_on] if is_scalar(right_on) else right_on
    )

    # handling a simple case of merging on one column and when the column is located in an index
    if len(left_on) == 1 and len(right_on) == 1 and left_on[0] == right_on[0]:
        if left_on[0] not in left and right_on[0] not in right:
            # in this case the 'on' column will stay in the index, so we can simply
            # drop the 'left/right_on' values and proceed as normal
            left_on = []
            right_on = []
        # in other cases, we can simply add the index name to columns and proceed as normal
        elif left_on[0] not in left:
            left = left.insert(loc=0, item=left_on[0])  # type: ignore
        elif right_on[0] not in right:
            right = right.insert(loc=0, item=right_on[0])  # type: ignore

    if any(col not in left for col in left_on) or any(
        col not in right for col in right_on
    ):
        raise NotImplementedError(
            "Cases, where one of the keys to join is an index level, are not yet supported."
        )

    left_conflicts = set(left) & (set(right) - set(right_on))
    right_conflicts = set(right) & (set(left) - set(left_on))
    conflicting_cols = left_conflicts | right_conflicts

    def _get_new_name(col: IndexLabel, suffix: str) -> IndexLabel:
        if col in conflicting_cols:
            return (
                (f"{col[0]}{suffix}", *col[1:])
                if isinstance(col, tuple)
                else f"{col}{suffix}"
            )
        else:
            return col

    left_renamer: Dict[IndexLabel, IndexLabel] = {}
    right_renamer: Dict[IndexLabel, IndexLabel] = {}
    new_left: List = []
    new_right: List = []

    for col in left:
        new_name = _get_new_name(col, suffixes[0])
        new_left.append(new_name)
        left_renamer[col] = new_name

    for col in right:
        # If we're joining on the column that exists in both frames then it was already
        # taken from the 'left', don't want to take it again from the 'right'.
        if not (col in left_on and col in right_on):
            new_name = _get_new_name(col, suffixes[1])
            new_right.append(new_name)
            right_renamer[col] = new_name

    new_columns = pandas.Index(new_left + new_right)
    return new_columns, left_renamer, right_renamer


def is_trivial_index(index: pandas.Index) -> bool:
    """
    Check if the index is a trivial index, i.e. a sequence [0..n].

    Parameters
    ----------
    index : pandas.Index
        An index to check.

    Returns
    -------
    bool
    """
    if len(index) == 0:
        return True
    if isinstance(index, pandas.RangeIndex):
        return index.start == 0 and index.step == 1
    if not (isinstance(index, pandas.Index) and is_integer_dtype(index)):
        return False
    return (
        index.is_monotonic_increasing
        and index.is_unique
        and index.min() == 0
        and index.max() == len(index) - 1
    )
