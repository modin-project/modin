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

"""Implement pandas general API."""

import pandas
import numpy as np

from typing import Hashable, Iterable, Mapping, Union
from pandas.core.dtypes.common import is_list_like

from modin.error_message import ErrorMessage
from .base import BasePandasDataset
from .dataframe import DataFrame
from .series import Series
from modin.utils import to_pandas
from modin.core.storage_formats.base.query_compiler import BaseQueryCompiler
from modin.utils import _inherit_docstrings
from modin.logging import enable_logging


@_inherit_docstrings(pandas.isna, apilink="pandas.isna")
@enable_logging
def isna(obj):  # noqa: PR01, RT01, D200
    """
    Detect missing values for an array-like object.
    """
    if isinstance(obj, BasePandasDataset):
        return obj.isna()
    else:
        return pandas.isna(obj)


isnull = isna


@_inherit_docstrings(pandas.notna, apilink="pandas.notna")
@enable_logging
def notna(obj):  # noqa: PR01, RT01, D200
    """
    Detect non-missing values for an array-like object.
    """
    if isinstance(obj, BasePandasDataset):
        return obj.notna()
    else:
        return pandas.notna(obj)


notnull = notna


@_inherit_docstrings(pandas.merge, apilink="pandas.merge")
@enable_logging
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
):  # noqa: PR01, RT01, D200
    """
    Merge DataFrame or named Series objects with a database-style join.
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


@_inherit_docstrings(pandas.merge_ordered, apilink="pandas.merge_ordered")
@enable_logging
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
) -> DataFrame:  # noqa: PR01, RT01, D200
    """
    Perform a merge for ordered data with optional filling/interpolation.
    """
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


@_inherit_docstrings(pandas.merge_asof, apilink="pandas.merge_asof")
@enable_logging
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
) -> DataFrame:  # noqa: PR01, RT01, D200
    """
    Perform a merge by key distance.
    """
    if not isinstance(left, DataFrame):
        raise ValueError(
            "can not merge DataFrame with instance of type {}".format(type(right))
        )
    ErrorMessage.default_to_pandas("`merge_asof`")

    # As of Pandas 1.2 these should raise an error; before that it did
    # something likely random:
    if (
        (on and (left_index or right_index))
        or (left_on and left_index)
        or (right_on and right_index)
    ):
        raise ValueError("Can't combine left/right_index with left/right_on or on.")

    # Pandas fallbacks for tricky cases:
    if (
        # No idea how this works or why it does what it does; and in fact
        # there's a Pandas bug suggesting it's wrong:
        # https://github.com/pandas-dev/pandas/issues/33463
        (left_index and right_on is not None)
        # This is the case where by is a list of columns. If we're copying lots
        # of columns out of Pandas, maybe not worth trying our path, it's not
        # clear it's any better:
        or not isinstance(by, (str, type(None)))
        or not isinstance(left_by, (str, type(None)))
        or not isinstance(right_by, (str, type(None)))
        # The implementation below assumes that the right index is unique
        # because it uses merge_asof to map each position in the merged
        # index to the label of the one right row that should be merged
        # at that row position.
        or not right.index.is_unique
    ):
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

    left_column = None
    right_column = None

    if on is not None:
        if left_on is not None or right_on is not None:
            raise ValueError("If 'on' is set, 'left_on' and 'right_on' can't be set.")
        left_on = on
        right_on = on

    if left_on is not None:
        left_column = to_pandas(left[left_on])
    elif left_index:
        left_column = left.index
    else:
        raise ValueError("Need some sort of 'on' spec")

    if right_on is not None:
        right_column = to_pandas(right[right_on])
    elif right_index:
        right_column = right.index
    else:
        raise ValueError("Need some sort of 'on' spec")

    # If we haven't set these by now, there's a bug in this function.
    assert left_column is not None
    assert right_column is not None

    if by is not None:
        if left_by is not None or right_by is not None:
            raise ValueError("Can't have both 'by' and 'left_by' or 'right_by'")
        left_by = right_by = by

    # List of columns case should have been handled by direct Pandas fallback
    # earlier:
    assert isinstance(left_by, (str, type(None)))
    assert isinstance(right_by, (str, type(None)))

    left_pandas_limited = {"on": left_column}
    right_pandas_limited = {"on": right_column, "right_labels": right.index}
    extra_kwargs = {}  # extra arguments to Pandas merge_asof

    if left_by is not None or right_by is not None:
        extra_kwargs["by"] = "by"
        left_pandas_limited["by"] = to_pandas(left[left_by])
        right_pandas_limited["by"] = to_pandas(right[right_by])

    # 1. Construct Pandas DataFrames with just the 'on' and optional 'by'
    # columns, and the index as another column.
    left_pandas_limited = pandas.DataFrame(left_pandas_limited, index=left.index)
    right_pandas_limited = pandas.DataFrame(right_pandas_limited)

    # 2. Use Pandas' merge_asof to figure out how to map labels on left to
    # labels on the right.
    merged = pandas.merge_asof(
        left_pandas_limited,
        right_pandas_limited,
        on="on",
        direction=direction,
        allow_exact_matches=allow_exact_matches,
        tolerance=tolerance,
        **extra_kwargs,
    )
    # Now merged["right_labels"] shows which labels from right map to left's index.

    # 3. Re-index right using the merged["right_labels"]; at this point right
    # should be same length and (semantically) same order as left:
    right_subset = right.reindex(index=pandas.Index(merged["right_labels"]))
    if not right_index:
        right_subset.drop(columns=[right_on], inplace=True)
    if right_by is not None and left_by == right_by:
        right_subset.drop(columns=[right_by], inplace=True)
    right_subset.index = left.index

    # 4. Merge left and the new shrunken right:
    result = merge(
        left,
        right_subset,
        left_index=True,
        right_index=True,
        suffixes=suffixes,
        how="left",
    )

    # 5. Clean up to match Pandas output:
    if left_on is not None and right_index:
        result.insert(
            # In theory this could use get_indexer_for(), but that causes an error:
            list(result.columns).index(left_on + suffixes[0]),
            left_on,
            result[left_on + suffixes[0]],
        )
    if not left_index and not right_index:
        result.index = pandas.RangeIndex(start=0, stop=len(result))

    return result


@_inherit_docstrings(pandas.pivot, apilink="pandas.pivot")
@enable_logging
def pivot(data, index=None, columns=None, values=None):  # noqa: PR01, RT01, D200
    """
    Return reshaped DataFrame organized by given index / column values.
    """
    if not isinstance(data, DataFrame):
        raise ValueError("can not pivot with instance of type {}".format(type(data)))
    return data.pivot(index=index, columns=columns, values=values)


@_inherit_docstrings(pandas.to_numeric, apilink="pandas.to_numeric")
@enable_logging
def to_numeric(arg, errors="raise", downcast=None):  # noqa: PR01, RT01, D200
    """
    Convert argument to a numeric type.
    """
    if not isinstance(arg, Series):
        return pandas.to_numeric(arg, errors=errors, downcast=downcast)
    return arg._to_numeric(errors=errors, downcast=downcast)


@_inherit_docstrings(pandas.unique, apilink="pandas.unique")
@enable_logging
def unique(values):  # noqa: PR01, RT01, D200
    """
    Return unique values based on a hash table.
    """
    return Series(values).unique()


# Adding docstring since pandas docs don't have web section for this function.
@enable_logging
def value_counts(
    values, sort=True, ascending=False, normalize=False, bins=None, dropna=True
):
    """
    Compute a histogram of the counts of non-null values.

    Parameters
    ----------
    values : ndarray (1-d)
        Values to perform computation.
    sort : bool, default: True
        Sort by values.
    ascending : bool, default: False
        Sort in ascending order.
    normalize : bool, default: False
        If True then compute a relative histogram.
    bins : integer, optional
        Rather than count values, group them into half-open bins,
        convenience for pd.cut, only works with numeric data.
    dropna : bool, default: True
        Don't include counts of NaN.

    Returns
    -------
    Series
    """
    return Series(values).value_counts(
        sort=sort,
        ascending=ascending,
        normalize=normalize,
        bins=bins,
        dropna=dropna,
    )


@_inherit_docstrings(pandas.concat, apilink="pandas.concat")
@enable_logging
def concat(
    objs: "Iterable[DataFrame | Series] | Mapping[Hashable, DataFrame | Series]",
    axis=0,
    join="outer",
    ignore_index: bool = False,
    keys=None,
    levels=None,
    names=None,
    verify_integrity: bool = False,
    sort: bool = False,
    copy: bool = True,
) -> "DataFrame | Series":  # noqa: PR01, RT01, D200
    """
    Concatenate Modin objects along a particular axis.
    """
    if isinstance(objs, (pandas.Series, Series, DataFrame, str, pandas.DataFrame)):
        raise TypeError(
            "first argument must be an iterable of pandas "
            + "objects, you passed an object of type "
            + f'"{type(objs).__name__}"'
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
            + "modin.pandas.Series "
            + "and modin.pandas.DataFrame objs are "
            + "valid",
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


@_inherit_docstrings(pandas.to_datetime, apilink="pandas.to_datetime")
@enable_logging
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
):  # noqa: PR01, RT01, D200
    """
    Convert argument to datetime.
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


@_inherit_docstrings(pandas.get_dummies, apilink="pandas.get_dummies")
@enable_logging
def get_dummies(
    data,
    prefix=None,
    prefix_sep="_",
    dummy_na=False,
    columns=None,
    sparse=False,
    drop_first=False,
    dtype=None,
):  # noqa: PR01, RT01, D200
    """
    Convert categorical variable into dummy/indicator variables.
    """
    if sparse:
        raise NotImplementedError(
            "SparseDataFrame is not implemented. "
            + "To contribute to Modin, please visit "
            + "github.com/modin-project/modin."
        )
    if not isinstance(data, DataFrame):
        ErrorMessage.default_to_pandas("`get_dummies` on non-DataFrame")
        if isinstance(data, Series):
            data = data._to_pandas()
        return DataFrame(
            pandas.get_dummies(
                data,
                prefix=prefix,
                prefix_sep=prefix_sep,
                dummy_na=dummy_na,
                columns=columns,
                sparse=sparse,
                drop_first=drop_first,
                dtype=dtype,
            )
        )
    else:
        new_manager = data._query_compiler.get_dummies(
            columns,
            prefix=prefix,
            prefix_sep=prefix_sep,
            dummy_na=dummy_na,
            drop_first=drop_first,
            dtype=dtype,
        )
        return DataFrame(query_compiler=new_manager)


@_inherit_docstrings(pandas.melt, apilink="pandas.melt")
@enable_logging
def melt(
    frame,
    id_vars=None,
    value_vars=None,
    var_name=None,
    value_name="value",
    col_level=None,
    ignore_index: bool = True,
):  # noqa: PR01, RT01, D200
    """
    Unpivot a DataFrame from wide to long format, optionally leaving identifiers set.
    """
    return frame.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name=var_name,
        value_name=value_name,
        col_level=col_level,
        ignore_index=ignore_index,
    )


@_inherit_docstrings(pandas.crosstab, apilink="pandas.crosstab")
@enable_logging
def crosstab(
    index,
    columns,
    values=None,
    rownames=None,
    colnames=None,
    aggfunc=None,
    margins=False,
    margins_name: str = "All",
    dropna: bool = True,
    normalize=False,
) -> DataFrame:  # noqa: PR01, RT01, D200
    """
    Compute a simple cross tabulation of two (or more) factors.
    """
    ErrorMessage.default_to_pandas("`crosstab`")
    pandas_crosstab = pandas.crosstab(
        index,
        columns,
        values,
        rownames,
        colnames,
        aggfunc,
        margins,
        margins_name,
        dropna,
        normalize,
    )
    return DataFrame(pandas_crosstab)


# Adding docstring since pandas docs don't have web section for this function.
@enable_logging
def lreshape(data: DataFrame, groups, dropna=True, label=None):
    """
    Reshape wide-format data to long. Generalized inverse of ``DataFrame.pivot``.

    Accepts a dictionary, `groups`, in which each key is a new column name
    and each value is a list of old column names that will be "melted" under
    the new column name as part of the reshape.

    Parameters
    ----------
    data : DataFrame
        The wide-format DataFrame.
    groups : dict
        Dictionary in the form: `{new_name : list_of_columns}`.
    dropna : bool, default: True
        Whether include columns whose entries are all NaN or not.
    label : optional
        Deprecated parameter.

    Returns
    -------
    DataFrame
        Reshaped DataFrame.
    """
    if not isinstance(data, DataFrame):
        raise ValueError("can not lreshape with instance of type {}".format(type(data)))
    ErrorMessage.default_to_pandas("`lreshape`")
    return DataFrame(
        pandas.lreshape(to_pandas(data), groups, dropna=dropna, label=label)
    )


@_inherit_docstrings(pandas.wide_to_long, apilink="pandas.wide_to_long")
@enable_logging
def wide_to_long(
    df: DataFrame, stubnames, i, j, sep: str = "", suffix: str = r"\d+"
) -> DataFrame:  # noqa: PR01, RT01, D200
    """
    Unpivot a DataFrame from wide to long format.
    """
    if not isinstance(df, DataFrame):
        raise ValueError(
            "can not wide_to_long with instance of type {}".format(type(df))
        )
    ErrorMessage.default_to_pandas("`wide_to_long`")
    return DataFrame(
        pandas.wide_to_long(to_pandas(df), stubnames, i, j, sep=sep, suffix=suffix)
    )


def _determine_name(objs: Iterable[BaseQueryCompiler], axis: Union[int, str]):
    """
    Determine names of index after concatenation along passed axis.

    Parameters
    ----------
    objs : iterable of QueryCompilers
        Objects to concatenate.
    axis : int or str
        The axis to concatenate along.

    Returns
    -------
    list with single element
        Computed index name, `None` if it could not be determined.
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


@_inherit_docstrings(pandas.to_datetime, apilink="pandas.to_timedelta")
@enable_logging
def to_timedelta(arg, unit=None, errors="raise"):  # noqa: PR01, RT01, D200
    """
    Convert argument to timedelta.

    Accepts str, timedelta, list-like or Series for arg parameter.
    Returns a Series if and only if arg is provided as a Series.
    """
    if isinstance(arg, Series):
        query_compiler = arg._query_compiler.to_timedelta(unit=unit, errors=errors)
        return Series(query_compiler=query_compiler)
    return pandas.to_timedelta(arg, unit=unit, errors=errors)
