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
import numpy as np

from typing import Hashable, Iterable, Mapping, Optional, Union
from pandas._typing import FrameOrSeriesUnion
from pandas.core.dtypes.common import is_list_like

from modin.error_message import ErrorMessage
from .base import BasePandasDataset
from .dataframe import DataFrame
from .series import Series
from modin.utils import to_pandas
from modin.backends.base.query_compiler import BaseQueryCompiler


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
):
    """
    Merge DataFrame or named Series objects with a database-style join.

    The join is done on columns or indexes. If joining columns on columns,
    the DataFrame indexes will be ignored. Otherwise if joining indexes on indexes or
    indexes on a column or columns, the index will be passed on.

    Parameters
    ----------
    right : DataFrame or named Series
        Object to merge with.
    how : {'left', 'right', 'outer', 'inner'}, default 'inner'
        Type of merge to be performed.
        - left: use only keys from left frame,
            similar to a SQL left outer join; preserve key order.
        - right: use only keys from right frame,
            similar to a SQL right outer join; preserve key order.
        - outer: use union of keys from both frames,
            similar to a SQL full outer join; sort keys lexicographically.
        - inner: use intersection of keys from both frames,
            similar to a SQL inner join; preserve the order of the left keys.
    on : label or list
        Column or index level names to join on.
        These must be found in both DataFrames. If on is None and not merging on indexes
        then this defaults to the intersection of the columns in both DataFrames.
    left_on : label or list, or array-like
        Column or index level names to join on in the left DataFrame.
        Can also be an array or list of arrays of the length of the left DataFrame.
        These arrays are treated as if they are columns.
    right_on : label or list, or array-like
        Column or index level names to join on in the right DataFrame.
        Can also be an array or list of arrays of the length of the right DataFrame.
        These arrays are treated as if they are columns.
    left_index : bool, default False
        Use the index from the left DataFrame as the join key(s).
        If it is a MultiIndex, the number of keys in the other DataFrame
        (either the index or a number of columns) must match the number of levels.
    right_index : bool, default False
        Use the index from the right DataFrame as the join key. Same caveats as left_index.
    sort : bool, default False
        Sort the join keys lexicographically in the result DataFrame.
        If False, the order of the join keys depends on the join type (how keyword).
    suffixes : tuple of (str, str), default ('_x', '_y')
        Suffix to apply to overlapping column names in the left and right side, respectively.
        To raise an exception on overlapping columns use (False, False).
    copy : bool, default True
        If False, avoid copy if possible.
    indicator : bool or str, default False
        If True, adds a column to output DataFrame called "_merge" with information
        on the source of each row. If string, column with information on source of each row
        will be added to output DataFrame, and column will be named value of string.
        Information column is Categorical-type and takes on a value of "left_only"
        for observations whose merge key only appears in 'left' DataFrame,
        "right_only" for observations whose merge key only appears in 'right' DataFrame,
        and "both" if the observation’s merge key is found in both.
    validate : str, optional
        If specified, checks if merge is of specified type.
        - 'one_to_one' or '1:1': check if merge keys are unique in both left and right datasets.
        - 'one_to_many' or '1:m': check if merge keys are unique in left dataset.
        - 'many_to_one' or 'm:1': check if merge keys are unique in right dataset.
        - 'many_to_many' or 'm:m': allowed, but does not result in checks.

    Returns
    -------
    DataFrame
        A DataFrame of the two merged objects.
    """
    if isinstance(left, Series):
        if left.name is None:
            raise ValueError("Cannot merge a Series without a name")
        else:
            left = left.to_frame()

    if not isinstance(left, DataFrame):
        raise TypeError(
            f"Can only merge Series or DataFrame objects, a {type(left)} was passed"
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
        sort=sort,
        ascending=ascending,
        normalize=normalize,
        bins=bins,
        dropna=dropna,
    )


def concat(
    objs: Union[
        Iterable[FrameOrSeriesUnion], Mapping[Optional[Hashable], FrameOrSeriesUnion]
    ],
    axis=0,
    join="outer",
    ignore_index: bool = False,
    keys=None,
    levels=None,
    names=None,
    verify_integrity: bool = False,
    sort: bool = False,
    copy: bool = True,
) -> FrameOrSeriesUnion:
    if isinstance(objs, (pandas.Series, Series, DataFrame, str, pandas.DataFrame)):
        raise TypeError(
            "first argument must be an iterable of pandas "
            "objects, you passed an object of type "
            '"{name}"'.format(name=type(objs).__name__)
        )
    axis = pandas.DataFrame()._get_axis_number(axis)
    if isinstance(objs, dict):
        list_of_objs = list(objs.values())
    else:
        list_of_objs = list(objs)
    if len(list_of_objs) == 0:
        raise ValueError("No objects to concatenate")

    list_of_objs = [obj for obj in list_of_objs if obj is not None]

    if len(list_of_objs) == 0:
        raise ValueError("All objects passed were None")
    try:
        type_check = next(
            obj
            for obj in list_of_objs
            if not isinstance(obj, (pandas.Series, Series, pandas.DataFrame, DataFrame))
        )
    except StopIteration:
        type_check = None
    if type_check is not None:
        raise ValueError(
            'cannot concatenate object of type "{0}"; only '
            "modin.pandas.Series "
            "and modin.pandas.DataFrame objs are "
            "valid",
            type(type_check),
        )
    all_series = all(isinstance(obj, Series) for obj in list_of_objs)
    if all_series and axis == 0:
        return Series(
            query_compiler=list_of_objs[0]._query_compiler.concat(
                axis,
                [o._query_compiler for o in list_of_objs[1:]],
                join=join,
                join_axes=None,
                ignore_index=ignore_index,
                keys=None,
                levels=None,
                names=None,
                verify_integrity=False,
                copy=True,
                sort=sort,
            )
        )
    if join not in ["inner", "outer"]:
        raise ValueError(
            "Only can inner (intersect) or outer (union) join the other axis"
        )
    # We have the weird Series and axis check because, when concatenating a
    # dataframe to a series on axis=0, pandas ignores the name of the series,
    # and this check aims to mirror that (possibly buggy) functionality
    list_of_objs = [
        obj
        if isinstance(obj, DataFrame)
        else DataFrame(obj.rename())
        if isinstance(obj, (pandas.Series, Series)) and axis == 0
        else DataFrame(obj)
        for obj in list_of_objs
    ]
    list_of_objs = [
        obj._query_compiler
        for obj in list_of_objs
        if (not obj._query_compiler.lazy_execution and len(obj.index))
        or len(obj.columns)
    ]
    if keys is not None:
        if all_series:
            new_idx = keys
        else:
            list_of_objs = [
                list_of_objs[i] for i in range(min(len(list_of_objs), len(keys)))
            ]
            new_idx_labels = {
                k: v.index if axis == 0 else v.columns
                for k, v in zip(keys, list_of_objs)
            }
            tuples = [
                (k, *o) if isinstance(o, tuple) else (k, o)
                for k, obj in new_idx_labels.items()
                for o in obj
            ]
            new_idx = pandas.MultiIndex.from_tuples(tuples)
            if names is not None:
                new_idx.names = names
            else:
                old_name = _determine_name(list_of_objs, axis)
                if old_name is not None:
                    new_idx.names = [None] + old_name
    elif isinstance(objs, dict):
        new_idx = pandas.concat(
            {k: pandas.Series(index=obj.axes[axis]) for k, obj in objs.items()}
        ).index
    else:
        new_idx = None
    new_query_compiler = list_of_objs[0].concat(
        axis,
        list_of_objs[1:],
        join=join,
        join_axes=None,
        ignore_index=ignore_index,
        keys=None,
        levels=None,
        names=None,
        verify_integrity=False,
        copy=True,
        sort=sort,
    )
    result_df = DataFrame(query_compiler=new_query_compiler)
    if new_idx is not None:
        if axis == 0:
            result_df.index = new_idx
        else:
            result_df.columns = new_idx
    return result_df


def _determine_name(objs: Iterable[BaseQueryCompiler], axis: Union[int, str]):
    """
    Determine names of index after concatenation along passed axis

    Parameters
    ----------
    objs : iterable of QueryCompilers
        objects to concatenate

    axis : int or str
        the axis to concatenate along

    Returns
    -------
        `list` with single element - computed index name, `None` if it could not
        be determined
    """
    axis = pandas.DataFrame()._get_axis_number(axis)

    def get_names(obj):
        return obj.columns.names if axis else obj.index.names

    names = np.array([get_names(obj) for obj in objs])

    # saving old name, only if index names of all objs are the same
    if np.all(names == names[0]):
        # we must do this check to avoid this calls `list(str_like_name)`
        return list(names[0]) if is_list_like(names[0]) else [names[0]]
    else:
        return None


def to_datetime(
    arg,
    errors="raise",
    dayfirst=False,
    yearfirst=False,
    utc=None,
    format=None,
    exact=True,
    unit=None,
    infer_datetime_format=False,
    origin="unix",
    cache=True,
):
    """
    Convert argument to datetime.

    Parameters
    ----------
    arg : int, float, str, datetime, list, tuple, 1-d array, Series, DataFrame/dict-like
        The object to convert to a datetime.
    errors : {'ignore', 'raise', 'coerce'}, default 'raise'
        - If 'raise', then invalid parsing will raise an exception.
        - If 'coerce', then invalid parsing will be set as NaT.
        - If 'ignore', then invalid parsing will return the input.
    dayfirst : bool, default False
        Specify a date parse order if `arg` is str or its list-likes.
        If True, parses dates with the day first, eg 10/11/12 is parsed as
        2012-11-10.
        Warning: dayfirst=True is not strict, but will prefer to parse
        with day first (this is a known bug, based on dateutil behavior).
    yearfirst : bool, default False
        Specify a date parse order if `arg` is str or its list-likes.

        - If True parses dates with the year first, eg 10/11/12 is parsed as
          2010-11-12.
        - If both dayfirst and yearfirst are True, yearfirst is preceded (same
          as dateutil).

        Warning: yearfirst=True is not strict, but will prefer to parse
        with year first (this is a known bug, based on dateutil behavior).
    utc : bool, default None
        Return UTC DatetimeIndex if True (converting any tz-aware
        datetime.datetime objects as well).
    format : str, default None
        The strftime to parse time, eg "%d/%m/%Y", note that "%f" will parse
        all the way up to nanoseconds.
        See strftime documentation for more information on choices:
        https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior.
    exact : bool, True by default
        Behaves as:
        - If True, require an exact format match.
        - If False, allow the format to match anywhere in the target string.

    unit : str, default 'ns'
        The unit of the arg (D,s,ms,us,ns) denote the unit, which is an
        integer or float number. This will be based off the origin.
        Example, with unit='ms' and origin='unix' (the default), this
        would calculate the number of milliseconds to the unix epoch start.
    infer_datetime_format : bool, default False
        If True and no `format` is given, attempt to infer the format of the
        datetime strings based on the first non-NaN element,
        and if it can be inferred, switch to a faster method of parsing them.
        In some cases this can increase the parsing speed by ~5-10x.
    origin : scalar, default 'unix'
        Define the reference date. The numeric values would be parsed as number
        of units (defined by `unit`) since this reference date.

        - If 'unix' (or POSIX) time; origin is set to 1970-01-01.
        - If 'julian', unit must be 'D', and origin is set to beginning of
          Julian Calendar. Julian day number 0 is assigned to the day starting
          at noon on January 1, 4713 BC.
        - If Timestamp convertible, origin is set to Timestamp identified by
          origin.
    cache : bool, default True
        If True, use a cache of unique, converted dates to apply the datetime
        conversion. May produce significant speed-up when parsing duplicate
        date strings, especially ones with timezone offsets. The cache is only
        used when there are at least 50 values. The presence of out-of-bounds
        values will render the cache unusable and may slow down parsing.

    Returns
    -------
    datetime
        If parsing succeeded.
        Return type depends on input:

        - list-like: DatetimeIndex
        - Series: Series of datetime64 dtype
        - scalar: Timestamp

        In case when it is not possible to return designated types (e.g. when
        any element of input is before Timestamp.min or after Timestamp.max)
        return will have datetime.datetime type (or corresponding
        array/Series).
    """
    if not isinstance(arg, (DataFrame, Series)):
        return pandas.to_datetime(
            arg,
            errors=errors,
            dayfirst=dayfirst,
            yearfirst=yearfirst,
            utc=utc,
            format=format,
            exact=exact,
            unit=unit,
            infer_datetime_format=infer_datetime_format,
            origin=origin,
            cache=cache,
        )
    return arg._to_datetime(
        errors=errors,
        dayfirst=dayfirst,
        yearfirst=yearfirst,
        utc=utc,
        format=format,
        exact=exact,
        unit=unit,
        infer_datetime_format=infer_datetime_format,
        origin=origin,
        cache=cache,
    )
