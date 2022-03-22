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

from pandas import MultiIndex
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


def from_non_pandas(df, index, columns, dtype):
    """
    Convert a non-pandas DataFrame into Modin DataFrame.

    Parameters
    ----------
    df : object
        Non-pandas DataFrame.
    index : object
        Index for non-pandas DataFrame.
    columns : object
        Columns for non-pandas DataFrame.
    dtype : type
        Data type to force.

    Returns
    -------
    modin.pandas.DataFrame
        Converted DataFrame.
    """
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    new_qc = FactoryDispatcher.from_non_pandas(df, index, columns, dtype)
    if new_qc is not None:
        from .dataframe import DataFrame

        return DataFrame(query_compiler=new_qc)
    return new_qc


def from_pandas(df):
    """
    Convert a pandas DataFrame to a Modin DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The pandas DataFrame to convert.

    Returns
    -------
    modin.pandas.DataFrame
        A new Modin DataFrame object.
    """
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher
    from .dataframe import DataFrame

    return DataFrame(query_compiler=FactoryDispatcher.from_pandas(df))


def from_arrow(at):
    """
    Convert an Arrow Table to a Modin DataFrame.

    Parameters
    ----------
    at : Arrow Table
        The Arrow Table to convert from.

    Returns
    -------
    DataFrame
        A new Modin DataFrame object.
    """
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher
    from .dataframe import DataFrame

    return DataFrame(query_compiler=FactoryDispatcher.from_arrow(at))


def from_dataframe(df):
    """
    Convert a DataFrame implementing the dataframe exchange protocol to a Modin DataFrame.

    See more about the protocol in https://data-apis.org/dataframe-protocol/latest/index.html.

    Parameters
    ----------
    df : DataFrame
        The DataFrame object supporting the dataframe exchange protocol.

    Returns
    -------
    DataFrame
        A new Modin DataFrame object.
    """
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher
    from .dataframe import DataFrame

    return DataFrame(query_compiler=FactoryDispatcher.from_dataframe(df))


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


_original_pandas_MultiIndex_from_frame = MultiIndex.from_frame
MultiIndex.from_frame = from_modin_frame_to_mi
