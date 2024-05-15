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

"""Implement utils for pandas component."""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np
import pandas
from pandas._typing import AggFuncType, AggFuncTypeBase, AggFuncTypeDict, IndexLabel
from pandas.util._decorators import doc

from modin.utils import hashable

_doc_binary_operation = """
Return {operation} of {left} and `{right}` (binary operator `{bin_op}`).

Parameters
----------
{right} : {right_type}
    The second operand to perform computation.

Returns
-------
{returns}
"""

SET_DATAFRAME_ATTRIBUTE_WARNING = (
    "Modin doesn't allow columns to be created via a new attribute name - see "
    + "https://pandas.pydata.org/pandas-docs/stable/indexing.html#attribute-access"
)


def cast_function_modin2pandas(func):
    """
    Replace Modin functions with pandas functions if `func` is callable.

    Parameters
    ----------
    func : object

    Returns
    -------
    object
    """
    if callable(func):
        if func.__module__ == "modin.pandas.series":
            func = getattr(pandas.Series, func.__name__)
        elif func.__module__ in ("modin.pandas.dataframe", "modin.pandas.base"):
            # FIXME: when the method is defined in `modin.pandas.base` file, then the
            # type cannot be determined, in general there may be an error, but at the
            # moment it is better.
            func = getattr(pandas.DataFrame, func.__name__)
    return func


def is_scalar(obj):
    """
    Return True if given object is scalar.

    This method works the same as is_scalar method from pandas but
    it is optimized for Modin frames. For BasePandasDataset objects
    pandas version of is_scalar tries to access missing attribute
    causing index scan. This triggers execution for lazy frames and
    we avoid it by handling BasePandasDataset objects separately.

    Parameters
    ----------
    obj : object
        Object to check.

    Returns
    -------
    bool
        True if given object is scalar and False otherwise.
    """
    from pandas.api.types import is_scalar as pandas_is_scalar

    from .base import BasePandasDataset

    return not isinstance(obj, BasePandasDataset) and pandas_is_scalar(obj)


def get_pandas_backend(dtypes: pandas.Series) -> str | None:
    """
    Determine the backend based on the `dtypes`.

    Parameters
    ----------
    dtypes : pandas.Series
        DataFrame dtypes.

    Returns
    -------
    str | None
        Backend name.
    """
    backend = None
    if any(isinstance(x, pandas.ArrowDtype) for x in dtypes):
        backend = "pyarrow"
    return backend


def is_full_grab_slice(slc, sequence_len=None):
    """
    Check that the passed slice grabs the whole sequence.

    Parameters
    ----------
    slc : slice
        Slice object to check.
    sequence_len : int, optional
        Length of the sequence to index with the passed `slc`.
        If not specified the function won't be able to check whether
        ``slc.stop`` is equal or greater than the sequence length to
        consider `slc` to be a full-grab, and so, only slices with
        ``.stop is None`` are considered to be a full-grab.

    Returns
    -------
    bool
    """
    assert isinstance(slc, slice), "slice object required"
    return (
        slc.start in (None, 0)
        and slc.step in (None, 1)
        and (
            slc.stop is None or (sequence_len is not None and slc.stop >= sequence_len)
        )
    )


def from_modin_frame_to_mi(df, sortorder=None, names=None):
    """
    Make a pandas.MultiIndex from a DataFrame.

    Parameters
    ----------
    df : DataFrame
        DataFrame to be converted to pandas.MultiIndex.
    sortorder : int, default: None
        Level of sortedness (must be lexicographically sorted by that
        level).
    names : list-like, optional
        If no names are provided, use the column names, or tuple of column
        names if the columns is a MultiIndex. If a sequence, overwrite
        names with the given sequence.

    Returns
    -------
    pandas.MultiIndex
        The pandas.MultiIndex representation of the given DataFrame.
    """
    from .dataframe import DataFrame

    if isinstance(df, DataFrame):
        from modin.error_message import ErrorMessage

        ErrorMessage.default_to_pandas("`MultiIndex.from_frame`")
        df = df._to_pandas()
    return _original_pandas_MultiIndex_from_frame(df, sortorder, names)


def is_label(obj, label, axis=0):
    """
    Check whether or not 'obj' contain column or index level with name 'label'.

    Parameters
    ----------
    obj : modin.pandas.DataFrame, modin.pandas.Series or modin.core.storage_formats.base.BaseQueryCompiler
        Object to check.
    label : object
        Label name to check.
    axis : {0, 1}, default: 0
        Axis to search for `label` along.

    Returns
    -------
    bool
        True if check is successful, False otherwise.
    """
    qc = getattr(obj, "_query_compiler", obj)
    return hashable(label) and (
        label in qc.get_axis(axis ^ 1) or label in qc.get_index_names(axis)
    )


def check_both_not_none(option1, option2):
    """
    Check that both `option1` and `option2` are not None.

    Parameters
    ----------
    option1 : Any
        First object to check if not None.
    option2 : Any
        Second object to check if not None.

    Returns
    -------
    bool
        True if both option1 and option2 are not None, False otherwise.
    """
    return not (option1 is None or option2 is None)


def broadcast_item(
    obj,
    row_lookup,
    col_lookup,
    item,
    need_columns_reindex=True,
):
    """
    Use NumPy to broadcast or reshape item with reindexing.

    Parameters
    ----------
    obj : DataFrame or Series or query compiler
        The object containing the necessary information about the axes.
    row_lookup : slice or scalar
        The global row index to locate inside of `item`.
    col_lookup : range, array, list, slice or scalar
        The global col index to locate inside of `item`.
    item : DataFrame, Series, or query_compiler
        Value that should be broadcast to a new shape of `to_shape`.
    need_columns_reindex : bool, default: True
        In the case of assigning columns to a dataframe (broadcasting is
        part of the flow), reindexing is not needed.

    Returns
    -------
    (np.ndarray, Optional[Series])
        * np.ndarray - `item` after it was broadcasted to `to_shape`.
        * Series - item's dtypes.

    Raises
    ------
    ValueError
        1) If `row_lookup` or `col_lookup` contains values missing in
        DataFrame/Series index or columns correspondingly.
        2) If `item` cannot be broadcast from its own shape to `to_shape`.

    Notes
    -----
    NumPy is memory efficient, there shouldn't be performance issue.
    """
    # It is valid to pass a DataFrame or Series to __setitem__ that is larger than
    # the target the user is trying to overwrite.
    from .dataframe import DataFrame
    from .series import Series

    new_row_len = (
        len(obj.index[row_lookup]) if isinstance(row_lookup, slice) else len(row_lookup)
    )
    new_col_len = (
        len(obj.columns[col_lookup])
        if isinstance(col_lookup, slice)
        else len(col_lookup)
    )
    to_shape = new_row_len, new_col_len

    dtypes = None
    if isinstance(item, (pandas.Series, pandas.DataFrame, Series, DataFrame)):
        # convert indices in lookups to names, as pandas reindex expects them to be so
        axes_to_reindex = {}
        index_values = obj.index[row_lookup]
        if not index_values.equals(item.index):
            axes_to_reindex["index"] = index_values
        if need_columns_reindex and isinstance(item, (pandas.DataFrame, DataFrame)):
            column_values = obj.columns[col_lookup]
            if not column_values.equals(item.columns):
                axes_to_reindex["columns"] = column_values
        # New value for columns/index make that reindex add NaN values
        if axes_to_reindex:
            item = item.reindex(**axes_to_reindex)

        dtypes = item.dtypes
        if not isinstance(dtypes, pandas.Series):
            dtypes = pandas.Series([dtypes])

    try:
        # Cast to numpy drop information about heterogeneous types (cast to common)
        # TODO: we shouldn't do that, maybe there should be the if branch
        item = np.array(item)
        if dtypes is None:
            dtypes = pandas.Series([item.dtype] * len(col_lookup))
        if np.prod(to_shape) == np.prod(item.shape):
            return item.reshape(to_shape), dtypes
        else:
            return np.broadcast_to(item, to_shape), dtypes
    except ValueError:
        from_shape = np.array(item).shape
        raise ValueError(
            f"could not broadcast input array from shape {from_shape} into shape "
            + f"{to_shape}"
        )


def _walk_aggregation_func(
    key: IndexLabel, value: AggFuncType, depth: int = 0
) -> Iterator[Tuple[IndexLabel, AggFuncTypeBase, Optional[str], bool]]:
    """
    Walk over a function from a dictionary-specified aggregation.

    Note: this function is not supposed to be called directly and
    is used by ``walk_aggregation_dict``.

    Parameters
    ----------
    key : IndexLabel
        A key in a dictionary-specified aggregation for the passed `value`.
        This means an index label to apply the `value` functions against.
    value : AggFuncType
        An aggregation function matching the `key`.
    depth : int, default: 0
        Specifies a nesting level for the `value` where ``depth=0`` is when
        you call the function on a raw dictionary value.

    Yields
    ------
    (col: IndexLabel, func: AggFuncTypeBase, func_name: Optional[str], col_renaming_required: bool)
        Yield an aggregation function with its metadata:
            - `col`: column name to apply the function.
            - `func`: aggregation function to apply to the column.
            - `func_name`: custom function name that was specified in the dict.
            - `col_renaming_required`: whether it's required to rename the
                `col` into ``(col, func_name)``.
    """
    col_renaming_required = bool(depth)

    if isinstance(value, (list, tuple)):
        if depth == 0:
            for val in value:
                yield from _walk_aggregation_func(key, val, depth + 1)
        elif depth == 1:
            if len(value) != 2:
                raise ValueError(
                    f"Incorrect rename format. Renamer must consist of exactly two elements, got: {len(value)}."
                )
            func_name, func = value
            yield key, func, func_name, col_renaming_required
        else:
            # pandas doesn't support this as well
            raise NotImplementedError("Nested renaming is not supported.")
    else:
        yield key, value, None, col_renaming_required


def walk_aggregation_dict(
    agg_dict: AggFuncTypeDict,
) -> Iterator[Tuple[IndexLabel, AggFuncTypeBase, Optional[str], bool]]:
    """
    Walk over an aggregation dictionary.

    Parameters
    ----------
    agg_dict : AggFuncTypeDict

    Yields
    ------
    (col: IndexLabel, func: AggFuncTypeBase, func_name: Optional[str], col_renaming_required: bool)
        Yield an aggregation function with its metadata:
            - `col`: column name to apply the function.
            - `func`: aggregation function to apply to the column.
            - `func_name`: custom function name that was specified in the dict.
            - `col_renaming_required`: whether it's required to rename the
                `col` into ``(col, func_name)``.
    """
    for key, value in agg_dict.items():
        yield from _walk_aggregation_func(key, value)


def _doc_binary_op(operation, bin_op, left="Series", right="right", returns="Series"):
    """
    Return callable documenting `Series` or `DataFrame` binary operator.

    Parameters
    ----------
    operation : str
        Operation name.
    bin_op : str
        Binary operation name.
    left : str, default: 'Series'
        The left object to document.
    right : str, default: 'right'
        The right operand name.
    returns : str, default: 'Series'
        Type of returns.

    Returns
    -------
    callable
    """
    if left == "Series":
        right_type = "Series or scalar value"
    elif left == "DataFrame":
        right_type = "DataFrame, Series or scalar value"
    elif left == "BasePandasDataset":
        right_type = "BasePandasDataset or scalar value"
    else:
        raise NotImplementedError(
            f"Only 'BasePandasDataset', `DataFrame` and 'Series' `left` are allowed, actually passed: {left}"
        )
    doc_op = doc(
        _doc_binary_operation,
        operation=operation,
        right=right,
        right_type=right_type,
        bin_op=bin_op,
        returns=returns,
        left=left,
    )

    return doc_op


_original_pandas_MultiIndex_from_frame = pandas.MultiIndex.from_frame
pandas.MultiIndex.from_frame = from_modin_frame_to_mi
