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

from __future__ import annotations

import warnings
from typing import Hashable, Iterable, Mapping, Optional, Union

import numpy as np
import pandas
from pandas._libs.lib import NoDefault, no_default
from pandas._typing import ArrayLike, DtypeBackend, Scalar, npt
from pandas.core.dtypes.common import is_list_like

from modin.core.storage_formats import BaseQueryCompiler
from modin.error_message import ErrorMessage
from modin.logging import enable_logging
from modin.pandas.io import to_pandas
from modin.utils import _inherit_docstrings

from .base import BasePandasDataset
from .dataframe import DataFrame
from .series import Series


@_inherit_docstrings(pandas.isna, apilink="pandas.isna")
@enable_logging
def isna(
    obj,
) -> bool | npt.NDArray[np.bool_] | Series | DataFrame:  # noqa: PR01, RT01, D200
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
def notna(
    obj,
) -> bool | npt.NDArray[np.bool_] | Series | DataFrame:  # noqa: PR01, RT01, D200
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
    copy: Optional[bool] = None,
    indicator: bool = False,
    validate=None,
) -> DataFrame:  # noqa: PR01, RT01, D200
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
    for operand in (left, right):
        if not isinstance(operand, (Series, DataFrame)):
            raise TypeError(
                f"Can only merge Series or DataFrame objects, a {type(operand)} was passed"
            )

    return DataFrame(
        query_compiler=left._query_compiler.merge_ordered(
            right._query_compiler,
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

    if on is not None:
        if left_on is not None or right_on is not None:
            raise ValueError("If 'on' is set, 'left_on' and 'right_on' can't be set.")
        left_on = on
        right_on = on

    if by is not None:
        if left_by is not None or right_by is not None:
            raise ValueError("Can't have both 'by' and 'left_by' or 'right_by'")
        left_by = right_by = by

    if left_on is None and not left_index:
        raise ValueError("Must pass on, left_on, or left_index=True")

    if right_on is None and not right_index:
        raise ValueError("Must pass on, right_on, or right_index=True")

    return DataFrame(
        query_compiler=left._query_compiler.merge_asof(
            right._query_compiler,
            left_on,
            right_on,
            left_index,
            right_index,
            left_by,
            right_by,
            suffixes,
            tolerance,
            allow_exact_matches,
            direction,
        )
    )


@_inherit_docstrings(pandas.pivot_table, apilink="pandas.pivot_table")
@enable_logging
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
    observed=no_default,
    sort=True,
) -> DataFrame:
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
        observed=observed,
        sort=sort,
    )


@_inherit_docstrings(pandas.pivot, apilink="pandas.pivot")
@enable_logging
def pivot(
    data, *, columns, index=no_default, values=no_default
) -> DataFrame:  # noqa: PR01, RT01, D200
    """
    Return reshaped DataFrame organized by given index / column values.
    """
    if not isinstance(data, DataFrame):
        raise ValueError("can not pivot with instance of type {}".format(type(data)))
    return data.pivot(index=index, columns=columns, values=values)


@_inherit_docstrings(pandas.to_numeric, apilink="pandas.to_numeric")
@enable_logging
def to_numeric(
    arg,
    errors="raise",
    downcast=None,
    dtype_backend: Union[DtypeBackend, NoDefault] = no_default,
) -> Scalar | np.ndarray | Series:  # noqa: PR01, RT01, D200
    """
    Convert argument to a numeric type.
    """
    if not isinstance(arg, Series):
        return pandas.to_numeric(
            arg, errors=errors, downcast=downcast, dtype_backend=dtype_backend
        )
    return arg._to_numeric(
        errors=errors, downcast=downcast, dtype_backend=dtype_backend
    )


@_inherit_docstrings(pandas.qcut, apilink="pandas.qcut")
@enable_logging
def qcut(
    x, q, labels=None, retbins=False, precision=3, duplicates="raise"
):  # noqa: PR01, RT01, D200
    """
    Quantile-based discretization function.
    """
    kwargs = {
        "labels": labels,
        "retbins": retbins,
        "precision": precision,
        "duplicates": duplicates,
    }
    if not isinstance(x, Series):
        return pandas.qcut(x, q, **kwargs)
    return x._qcut(q, **kwargs)


@_inherit_docstrings(pandas.cut, apilink="pandas.cut")
@enable_logging
def cut(
    x,
    bins,
    right: bool = True,
    labels=None,
    retbins: bool = False,
    precision: int = 3,
    include_lowest: bool = False,
    duplicates: str = "raise",
    ordered: bool = True,
):
    if isinstance(x, DataFrame):
        raise ValueError("Input array must be 1 dimensional")
    if not isinstance(x, Series):
        ErrorMessage.default_to_pandas(
            reason=f"pd.cut is not supported on objects of type {type(x)}"
        )
        import pandas

        return pandas.cut(
            x,
            bins,
            right=right,
            labels=labels,
            retbins=retbins,
            precision=precision,
            include_lowest=include_lowest,
            duplicates=duplicates,
            ordered=ordered,
        )

    def _wrap_in_series_object(qc_result):
        if isinstance(qc_result, type(x._query_compiler)):
            return Series(query_compiler=qc_result)
        if isinstance(qc_result, (tuple, list)):
            return tuple([_wrap_in_series_object(result) for result in qc_result])
        return qc_result

    return _wrap_in_series_object(
        x._query_compiler.cut(
            bins,
            right=right,
            labels=labels,
            retbins=retbins,
            precision=precision,
            include_lowest=include_lowest,
            duplicates=duplicates,
            ordered=ordered,
        )
    )


@_inherit_docstrings(pandas.unique, apilink="pandas.unique")
@enable_logging
def unique(values) -> ArrayLike:  # noqa: PR01, RT01, D200
    """
    Return unique values based on a hash table.
    """
    return Series(values).unique()


# Adding docstring since pandas docs don't have web section for this function.
@enable_logging
def value_counts(
    values, sort=True, ascending=False, normalize=False, bins=None, dropna=True
) -> Series:
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
    warnings.warn(
        "pandas.value_counts is deprecated and will be removed in a "
        + "future version. Use pd.Series(obj).value_counts() instead.",
        FutureWarning,
    )
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
    *,
    axis=0,
    join="outer",
    ignore_index: bool = False,
    keys=None,
    levels=None,
    names=None,
    verify_integrity: bool = False,
    sort: bool = False,
    copy: Optional[bool] = None,
) -> DataFrame | Series:  # noqa: PR01, RT01, D200
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
        input_list_of_objs = list(objs.values())
    else:
        input_list_of_objs = list(objs)
    if len(input_list_of_objs) == 0:
        raise ValueError("No objects to concatenate")

    list_of_objs = [obj for obj in input_list_of_objs if obj is not None]

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
    if join == "outer":
        # Filter out empties
        list_of_objs = [
            obj
            for obj in list_of_objs
            if (
                isinstance(obj, (Series, pandas.Series))
                or (isinstance(obj, DataFrame) and obj._query_compiler.lazy_shape)
                or sum(obj.shape) > 0
            )
        ]
    elif join != "inner":
        raise ValueError(
            "Only can inner (intersect) or outer (union) join the other axis"
        )
    list_of_objs = [
        (
            obj._query_compiler
            if isinstance(obj, DataFrame)
            else DataFrame(obj)._query_compiler
        )
        for obj in list_of_objs
    ]
    if keys is None and isinstance(objs, dict):
        keys = list(objs.keys())
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
    else:
        new_idx = None

    if len(list_of_objs) == 0:
        return DataFrame(
            index=input_list_of_objs[0].index.append(
                [f.index for f in input_list_of_objs[1:]]
            )
        )

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
    utc=False,
    format=None,
    exact=no_default,
    unit=None,
    infer_datetime_format=no_default,
    origin="unix",
    cache=True,
) -> Scalar | ArrayLike | Series | DataFrame:  # noqa: PR01, RT01, D200
    """
    Convert argument to datetime.
    """
    if not hasattr(arg, "_to_datetime"):
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
) -> DataFrame:  # noqa: PR01, RT01, D200
    """
    Convert categorical variable into dummy/indicator variables.
    """
    if sparse:
        raise NotImplementedError(
            "SparseArray is not implemented. "
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
) -> DataFrame:  # noqa: PR01, RT01, D200
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
def lreshape(data: DataFrame, groups, dropna=True) -> DataFrame:
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

    Returns
    -------
    DataFrame
        Reshaped DataFrame.
    """
    if not isinstance(data, DataFrame):
        raise ValueError("can not lreshape with instance of type {}".format(type(data)))
    ErrorMessage.default_to_pandas("`lreshape`")
    return DataFrame(pandas.lreshape(to_pandas(data), groups, dropna=dropna))


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
    return DataFrame(
        query_compiler=df._query_compiler.wide_to_long(
            stubnames=stubnames,
            i=i,
            j=j,
            sep=sep,
            suffix=suffix,
        )
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
def to_timedelta(
    arg, unit=None, errors="raise"
) -> Scalar | pandas.Index | Series:  # noqa: PR01, RT01, D200
    """
    Convert argument to timedelta.

    Accepts str, timedelta, list-like or Series for arg parameter.
    Returns a Series if and only if arg is provided as a Series.
    """
    if isinstance(arg, Series):
        query_compiler = arg._query_compiler.to_timedelta(unit=unit, errors=errors)
        return Series(query_compiler=query_compiler)
    return pandas.to_timedelta(arg, unit=unit, errors=errors)
