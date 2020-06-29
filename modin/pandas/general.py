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

import pandas

from modin.error_message import ErrorMessage
from .base import BasePandasDataset
from .dataframe import DataFrame
from .series import Series
from .utils import to_pandas


def isna(obj):
    """
    Detect missing values for an array-like object.
    Args:
        obj: Object to check for null or missing values.

    Returns:
        bool or array-like of bool
    """
    if isinstance(obj, BasePandasDataset):
        return obj.isna()
    else:
        return pandas.isna(obj)


isnull = isna


def notna(obj):
    if isinstance(obj, BasePandasDataset):
        return obj.notna()
    else:
        return pandas.notna(obj)


notnull = notna


def merge(
    left,
    right,
    how: str = "inner",
    on=None,
    left_on=None,
    right_on=None,
    left_index: bool = False,
    right_index: bool = False,
    sort: bool = False,
    suffixes=("_x", "_y"),
    copy: bool = True,
    indicator: bool = False,
    validate=None,
) -> DataFrame:
    """Database style join, where common columns in "on" are merged.

    Args:
        left: DataFrame.
        right: DataFrame.
        how: What type of join to use.
        on: The common column name(s) to join on. If None, and left_on and
            right_on  are also None, will default to all commonly named
            columns.
        left_on: The column(s) on the left to use for the join.
        right_on: The column(s) on the right to use for the join.
        left_index: Use the index from the left as the join keys.
        right_index: Use the index from the right as the join keys.
        sort: Sort the join keys lexicographically in the result.
        suffixes: Add this suffix to the common names not in the "on".
        copy: Does nothing in our implementation
        indicator: Adds a column named _merge to the DataFrame with
            metadata from the merge about each row.
        validate: Checks if merge is a specific type.

    Returns:
         A merged Dataframe
        """
    if not isinstance(left, DataFrame):
        raise ValueError(
            "can not merge DataFrame with instance of type {}".format(type(right))
        )

    return left.merge(
        right,
        how=how,
        on=on,
        left_on=left_on,
        right_on=right_on,
        left_index=left_index,
        right_index=right_index,
        sort=sort,
        suffixes=suffixes,
        copy=copy,
        indicator=indicator,
        validate=validate,
    )


def merge_ordered(
    left,
    right,
    on=None,
    left_on=None,
    right_on=None,
    left_by=None,
    right_by=None,
    fill_method=None,
    suffixes=("_x", "_y"),
    how: str = "outer",
) -> DataFrame:
    if not isinstance(left, DataFrame):
        raise ValueError(
            "can not merge DataFrame with instance of type {}".format(type(right))
        )
    ErrorMessage.default_to_pandas("`merge_ordered`")
    if isinstance(right, DataFrame):
        right = to_pandas(right)
    return DataFrame(
        pandas.merge_ordered(
            to_pandas(left),
            right,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_by=left_by,
            right_by=right_by,
            fill_method=fill_method,
            suffixes=suffixes,
            how=how,
        )
    )


def merge_asof(
    left,
    right,
    on=None,
    left_on=None,
    right_on=None,
    left_index: bool = False,
    right_index: bool = False,
    by=None,
    left_by=None,
    right_by=None,
    suffixes=("_x", "_y"),
    tolerance=None,
    allow_exact_matches: bool = True,
    direction: str = "backward",
) -> DataFrame:
    if not isinstance(left, DataFrame):
        raise ValueError(
            "can not merge DataFrame with instance of type {}".format(type(right))
        )
    ErrorMessage.default_to_pandas("`merge_asof`")
    if isinstance(right, DataFrame):
        right = to_pandas(right)
    return DataFrame(
        pandas.merge_asof(
            to_pandas(left),
            right,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            by=by,
            left_by=left_by,
            right_by=right_by,
            suffixes=suffixes,
            tolerance=tolerance,
            allow_exact_matches=allow_exact_matches,
            direction=direction,
        )
    )


def pivot_table(
    data,
    values=None,
    index=None,
    columns=None,
    aggfunc="mean",
    fill_value=None,
    margins=False,
    dropna=True,
    margins_name="All",
    observed=False,
):
    if not isinstance(data, DataFrame):
        raise ValueError(
            "can not create pivot table with instance of type {}".format(type(data))
        )

    return data.pivot_table(
        values=values,
        index=index,
        columns=columns,
        aggfunc=aggfunc,
        fill_value=fill_value,
        margins=margins,
        dropna=dropna,
        margins_name=margins_name,
    )


def pivot(data, index=None, columns=None, values=None):
    if not isinstance(data, DataFrame):
        raise ValueError("can not pivot with instance of type {}".format(type(data)))
    return data.pivot(index=index, columns=columns, values=values)


def to_numeric(arg, errors="raise", downcast=None):
    """
    Convert argument to a numeric type.

    The default return dtype is `float64` or `int64`
    depending on the data supplied. Use the `downcast` parameter
    to obtain other dtypes.

    Please note that precision loss may occur if really large numbers
    are passed in. Due to the internal limitations of `ndarray`, if
    numbers smaller than `-9223372036854775808` (np.iinfo(np.int64).min)
    or larger than `18446744073709551615` (np.iinfo(np.uint64).max) are
    passed in, it is very likely they will be converted to float so that
    they can stored in an `ndarray`. These warnings apply similarly to
    `Series` since it internally leverages `ndarray`.

    Parameters
    ----------
    arg : scalar, list, tuple, 1-d array, or Series
        Argument to be converted.
    errors : {'ignore', 'raise', 'coerce'}, default 'raise'
        - If 'raise', then invalid parsing will raise an exception.
        - If 'coerce', then invalid parsing will be set as NaN.
        - If 'ignore', then invalid parsing will return the input.
    downcast : {'int', 'signed', 'unsigned', 'float'}, default None
        If not None, and if the data has been successfully cast to a
        numerical dtype (or if the data was numeric to begin with),
        downcast that resulting data to the smallest numerical dtype
        possible according to the following rules:

        - 'int' or 'signed': smallest signed int dtype (min.: np.int8)
        - 'unsigned': smallest unsigned int dtype (min.: np.uint8)
        - 'float': smallest float dtype (min.: np.float32)

        As this behaviour is separate from the core conversion to
        numeric values, any errors raised during the downcasting
        will be surfaced regardless of the value of the 'errors' input.

        In addition, downcasting will only occur if the size
        of the resulting data's dtype is strictly larger than
        the dtype it is to be cast to, so if none of the dtypes
        checked satisfy that specification, no downcasting will be
        performed on the data.

    Returns
    -------
    ret
        Numeric if parsing succeeded.
        Return type depends on input.  Series if Series, otherwise ndarray.
    """
    if not isinstance(arg, Series):
        return pandas.to_numeric(arg, errors=errors, downcast=downcast)
    return arg._to_numeric(errors=errors, downcast=downcast)


def unique(values):
    """
    Return unique values of input data.

    Uniques are returned in order of appearance. Hash table-based unique,
    therefore does NOT sort.

    Returns
    -------
    ndarray
        The unique values returned as a NumPy array.
    """
    return Series(values).unique()


def value_counts(
    values, sort=True, ascending=False, normalize=False, bins=None, dropna=True
):
    """
    Compute a histogram of the counts of non-null values.

    Parameters
    ----------
    values : ndarray (1-d)
    sort : bool, default True
        Sort by values
    ascending : bool, default False
        Sort in ascending order
    normalize: bool, default False
        If True then compute a relative histogram
    bins : integer, optional
        Rather than count values, group them into half-open bins,
        convenience for pd.cut, only works with numeric data
    dropna : bool, default True
        Don't include counts of NaN

    Returns
    -------
    Series

    Notes
    -----
    The indices of resulting object will be in descending
    (ascending, if ascending=True) order for equal values.
    It slightly differ from pandas where indices are located in random order.
    """
    return Series(values).value_counts(
        sort=sort, ascending=ascending, normalize=normalize, bins=bins, dropna=dropna,
    )
