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
Module contains ``NativeQueryCompiler`` class.

``NativeQueryCompiler`` is responsible for compiling efficient DataFrame algebra
queries for small data and empty ``PandasDataFrame``.
"""

from typing import Optional

import numpy as np
import pandas
from pandas.core.dtypes.common import is_list_like, is_scalar

from modin.core.storage_formats.base.query_compiler import BaseQueryCompiler
from modin.core.storage_formats.pandas.query_compiler_caster import QueryCompilerCaster
from modin.utils import (
    MODIN_UNNAMED_SERIES_LABEL,
    _inherit_docstrings,
    try_cast_to_pandas,
)


def _get_axis(axis):
    """
    Build index labels getter of the specified axis.

    Parameters
    ----------
    axis : {0, 1}
        Axis to get labels from. 0 is for index and 1 is for column.

    Returns
    -------
    callable(NativeQueryCompiler) -> pandas.Index
    """
    if axis == 0:
        return lambda self: self._modin_frame.index
    else:
        return lambda self: self._modin_frame.columns


def _set_axis(axis):
    """
    Build index labels setter of the specified axis.

    Parameters
    ----------
    axis : {0, 1}
        Axis to set labels on. 0 is for index and 1 is for column.

    Returns
    -------
    callable(NativeQueryCompiler)
    """
    if axis == 0:

        def set_axis(self, idx):
            self._modin_frame.index = idx

    else:

        def set_axis(self, cols):
            self._modin_frame.columns = cols

    return set_axis


def _str_map(func_name):
    """
    Build function that calls specified string function on frames ``str`` accessor.

    Parameters
    ----------
    func_name : str
        String function name to execute on ``str`` accessor.

    Returns
    -------
    callable(pandas.DataFrame, *args, **kwargs) -> pandas.DataFrame
    """

    def str_op_builder(df, *args, **kwargs):
        """Apply specified function against `str` accessor of the passed frame."""
        str_s = df.squeeze(axis=1).str
        return getattr(pandas.Series.str, func_name)(str_s, *args, **kwargs).to_frame()

    return str_op_builder


def _dt_prop_map(property_name):
    """
    Build function that access specified property of the ``dt`` property of the passed frame.

    Parameters
    ----------
    property_name : str
        Date-time property name to access.

    Returns
    -------
    callable(pandas.DataFrame, *args, **kwargs) -> pandas.DataFrame
        Function to be applied in the frame.

    Notes
    -----
    This applies non-callable properties of ``Series.dt``.
    """

    def dt_op_builder(df, *args, **kwargs):
        """Access specified date-time property of the passed frame."""
        prop_val = getattr(df.squeeze(axis=1).dt, property_name)
        if isinstance(prop_val, pandas.Series):
            return prop_val.to_frame()
        elif isinstance(prop_val, pandas.DataFrame):
            return prop_val
        else:
            return pandas.DataFrame([prop_val])

    return dt_op_builder


def _dt_func_map(func_name):
    """
    Build function that apply specified method against ``dt`` property of the passed frame.

    Parameters
    ----------
    func_name : str
        Date-time function name to apply.

    Returns
    -------
    callable(pandas.DataFrame, *args, **kwargs) -> pandas.DataFrame
        Function to be applied in the frame.

    Notes
    -----
    This applies callable methods of ``Series.dt``.
    """

    def dt_op_builder(df, *args, **kwargs):
        """Apply specified function against ``dt`` accessor of the passed frame."""
        dt_s = df.squeeze(axis=1).dt
        dt_func_result = getattr(pandas.Series.dt, func_name)(dt_s, *args, **kwargs)
        # If we don't specify the dtype for the frame, the frame might get the
        # wrong dtype, e.g. for to_pydatetime in https://github.com/modin-project/modin/issues/4436
        return pandas.DataFrame(dt_func_result, dtype=dt_func_result.dtype)

    return dt_op_builder


def _rolling_func(func):
    """
    Build function that apply specified rolling method of the passed frame.

    Parameters
    ----------
    func : str
        Rolling function name to apply.

    Returns
    -------
    callable(pandas.DataFrame, *args, **kwargs) -> pandas.DataFrame
        Function to be applied to the frame.
    """

    def rolling_builder(df, fold_axis, rolling_args, *args, **kwargs):
        rolling_result = df.rolling(*rolling_args)
        rolling_op = getattr(rolling_result, func)
        return rolling_op(*args, **kwargs)

    return rolling_builder


def _reindex(df, axis, labels, **kwargs):  # noqa: GL08
    return df.reindex(labels=labels, axis=axis, **kwargs)


def _concat(df, axis, other, join_axes=None, **kwargs):  # noqa: GL08
    if not isinstance(other, list):
        other = [other]
    if (
        isinstance(df, pandas.DataFrame)
        and len(df.columns) == 1
        and df.columns[0] == MODIN_UNNAMED_SERIES_LABEL
    ):
        df = df[df.columns[0]]

    ignore_index = kwargs.get("ignore_index", False)
    concat_join = ["outer", "inner"]
    if kwargs.get("join", "outer") in concat_join:
        if not isinstance(other, list):
            other = [other]
        other = [df] + other
        result = pandas.concat(other, axis=axis, **kwargs)
    else:
        if isinstance(other, (list, np.ndarray)) and len(other) == 1:
            other = other[0]
        ignore_index = kwargs.pop("ignore_index", None)
        kwargs["how"] = kwargs.pop("join", None)
        if isinstance(other, (pandas.DataFrame, pandas.Series)):
            result = df.join(other, rsuffix="r_", **kwargs)
        else:
            result = df.join(other, **kwargs)
    if ignore_index:
        if axis == 0:
            result = result.reset_index(drop=True)
        else:
            result.columns = pandas.RangeIndex(len(result.columns))
    return result


def _to_datetime(df, *args, **kwargs):  # noqa: GL08
    return pandas.to_datetime(df.squeeze(axis=1), *args, **kwargs)


def _to_numeric(df, *args, **kwargs):  # noqa: GL08
    return pandas.to_numeric(df.squeeze(axis=1), *args, **kwargs)


def _groupby(agg_name):
    """
    Build function that apply specified groupby method of the passed frame.

    Parameters
    ----------
    agg_name : str
        GroupBy aggregate function name to apply.

    Returns
    -------
    callable(pandas.DataFrame, *args, **kwargs) -> pandas.DataFrame
        Function to be applied to the frame.
    """
    __aggregation_methods_dict = {
        "axis_wise": pandas.core.groupby.DataFrameGroupBy.aggregate,
        "group_wise": pandas.core.groupby.DataFrameGroupBy.apply,
        "transform": pandas.core.groupby.DataFrameGroupBy.transform,
    }

    def groupby_callable(
        df,
        by,
        axis,
        groupby_kwargs,
        agg_args,
        agg_kwargs,
        agg_func=None,
        how="axis_wise",
        drop=False,
        **kwargs,
    ):
        by_names = []
        if isinstance(by, pandas.DataFrame):
            by = by.squeeze(axis=1)
        if isinstance(by, list):
            for i in range(len(by)):
                if isinstance(by[i], pandas.DataFrame):
                    by[i] = by[i].squeeze(axis=1)
                if isinstance(by[i], pandas.Series):
                    if isinstance(df.index, pandas.MultiIndex):
                        by[i].name = pandas.MultiIndex.from_tuples(by[i].name)
                    by_names.append(by[i].name)
                elif isinstance(by[i], str):
                    by_names.append(by[i])
        if isinstance(by, pandas.DataFrame):
            by_names = list(by.columns)
            to_append = by.columns[[name not in df.columns for name in by_names]]
            if len(to_append) > 0:
                df = pandas.concat([df, by[to_append]], axis=1)
            by = by_names
        if isinstance(by, pandas.Series) and drop:
            by_names = [by.name]
        if (
            is_list_like(by)
            and drop
            and not any([is_list_like(curr_by) for curr_by in by])
        ):
            by = by_names

        groupby_obj = df.groupby(by=by, axis=axis, **groupby_kwargs)
        if agg_name == "agg":
            if isinstance(agg_func, dict):
                agg_func = {
                    k: v[0] if isinstance(v, list) and len(v) == 1 else v
                    for k, v in agg_func.items()
                }
            groupby_agg = __aggregation_methods_dict[how]
            result = groupby_agg(groupby_obj, agg_func, *agg_args, **agg_kwargs)
        else:
            groupby_agg = getattr(groupby_obj, agg_name)
            if callable(groupby_agg):
                result = groupby_agg(*agg_args, **agg_kwargs)
            else:
                result = groupby_agg

        return result

    return groupby_callable


def _register_binary(op):
    """
    Build function that apply specified binary method of the passed frame.

    Parameters
    ----------
    op : str
        Binary function name to apply.

    Returns
    -------
    callable(pandas.DataFrame, *args, **kwargs) -> pandas.DataFrame
        Function to be applied to the frame.
    """

    def binary_operator(df, other, **kwargs):
        squeeze_other = kwargs.pop("broadcast", False) or kwargs.pop(
            "squeeze_other", False
        )
        squeeze_self = kwargs.pop("squeeze_self", False)

        if squeeze_other:
            other = other.squeeze(axis=1)

        if squeeze_self:
            df = df.squeeze(axis=1)
        result = getattr(df, op)(other, **kwargs)
        if (
            not isinstance(result, pandas.Series)
            and not isinstance(result, pandas.DataFrame)
            and is_list_like(result)
        ):
            result = pandas.DataFrame(result)

        return result

    return binary_operator


def _register_expanding(func):
    """
    Build function that apply specified expanding window functions.

    Parameters
    ----------
    func : str
        Expanding window functionname to apply.

    Returns
    -------
    callable(pandas.DataFrame, *args, **kwargs) -> pandas.DataFrame
        Function to be applied to the frame.
    """

    def expanding_operator(df, fold_axis, rolling_args, *args, **kwargs):
        squeeze_self = kwargs.pop("squeeze_self", False)

        if squeeze_self:
            df = df.squeeze(axis=1)
        roller = df.expanding(*rolling_args)
        if type(func) is property:
            return func.fget(roller)

        return func(roller, *args, **kwargs)

    return expanding_operator


def _register_resample(op):
    """
    Build function that apply specified resample method of the passed frame.

    Parameters
    ----------
    op : str
        Resample function name to apply.

    Returns
    -------
    callable(pandas.DataFrame, *args, **kwargs) -> pandas.DataFrame
        Function to be applied to the frame.
    """

    def resample_operator(df, resample_kwargs, *args, **kwargs):
        resampler = df.resample(**resample_kwargs)
        result = getattr(resampler, op)(*args, **kwargs)
        return result

    return resample_operator


def _drop(df, **kwargs):  # noqa: GL08
    if (
        kwargs.get("labels", None) is not None
        or kwargs.get("index", None) is not None
        or kwargs.get("columns", None) is not None
    ):
        return df.drop(**kwargs)
    return df


def _fillna(df, value, **kwargs):  # noqa: GL08
    squeeze_self = kwargs.pop("squeeze_self", False)
    squeeze_value = kwargs.pop("squeeze_value", False)
    if squeeze_self and isinstance(df, pandas.DataFrame):
        df = df.squeeze(axis=1)
    if squeeze_value and isinstance(value, pandas.DataFrame):
        value = value.squeeze(axis=1)
    return df.fillna(value, **kwargs)


def _is_monotonic(monotonic_type):  # noqa: GL08
    def is_monotonic_caller(ser):
        return pandas.DataFrame([getattr(ser, monotonic_type)])

    return is_monotonic_caller


def _sort_index(df, inplace=False, **kwargs):  # noqa: GL08
    if inplace:
        df.sort_index(inplace=inplace, **kwargs)
    else:
        df = df.sort_index(inplace=inplace, **kwargs)
    return df


def _combine(df, other, func, **kwargs):  # noqa: GL08
    if isinstance(df, pandas.Series):
        return func(df, other)
    return df.combine(other, func)


def _getitem_array(df, key):  # noqa: GL08
    if isinstance(key, pandas.DataFrame):
        key = key.squeeze(axis=1)
    return df[key]


def _getitem_row_array(df, key):  # noqa: GL08
    if isinstance(key, pandas.DataFrame):
        key = key.squeeze(axis=1)
    return df.iloc[key]


def _write_items(
    df,
    row_numeric_index,
    col_numeric_index,
    item,
    need_columns_reindex=True,
):  # noqa: GL08
    from modin.pandas.utils import broadcast_item, is_scalar

    if not isinstance(row_numeric_index, slice):
        row_numeric_index = list(row_numeric_index)
    if not isinstance(col_numeric_index, slice):
        col_numeric_index = list(col_numeric_index)
    if not is_scalar(item):
        broadcasted_items, _ = broadcast_item(
            df,
            row_numeric_index,
            col_numeric_index,
            item,
            need_columns_reindex=need_columns_reindex,
        )
    else:
        broadcasted_items = item

    if isinstance(df.iloc[row_numeric_index, col_numeric_index], pandas.Series):
        broadcasted_items = broadcasted_items.squeeze()
    df.iloc[row_numeric_index, col_numeric_index] = broadcasted_items
    return df


def _setitem(df, axis, key, value):  # noqa: GL08
    if is_scalar(key) and isinstance(value, pandas.DataFrame):
        value = value.squeeze()
    if not axis:
        df[key] = value
    else:
        df.loc[key] = value
    return df


def _delitem(df, key):  # noqa: GL08
    return df.drop(columns=[key])


def _get_dummies(df, columns, **kwargs):  # noqa: GL08
    return pandas.get_dummies(df, columns=columns, **kwargs)


def _register_default_pandas(
    func,
    is_series=False,
    squeeze_args=False,
    squeeze_kwargs=False,
    return_raw=False,
    in_place=False,
):
    """
    Build function that apply specified method of the passed frame.

    Parameters
    ----------
    func : callable
        Function to apply.
    is_series : bool, default: False
        If True, the passed frame will always be squeezed to a series.
    squeeze_args : bool, default: False
        If True, all passed arguments will be squeezed.
    squeeze_kwargs : bool, default: False
        If True, all passed key word arguments will be squeezed.
    return_raw : bool, default: False
        If True, and the result not DataFrame or Series it is returned as is without wrapping in query compiler.
    in_place : bool, default: False
        If True, the specified function will be applied on the passed frame in place.

    Returns
    -------
    callable(pandas.DataFrame, *args, **kwargs) -> pandas.DataFrame
        Function to be applied to the frame.
    """

    def caller(query_compiler, *args, **kwargs):
        df = query_compiler._modin_frame
        if is_series:
            df = df.squeeze(axis=1)
        exclude_names = ["fold_axis", "dtypes"]
        kwargs = kwargs.copy()
        for name in exclude_names:
            kwargs.pop(name, None)
        args = try_cast_to_pandas(args, squeeze=squeeze_args)
        kwargs = try_cast_to_pandas(kwargs, squeeze=squeeze_kwargs)
        result = func(df, *args, **kwargs)
        inplace_method = kwargs.get("inplace", False)
        if in_place:
            inplace_method = in_place
        if inplace_method:
            result = df
        if return_raw and not isinstance(result, (pandas.Series, pandas.DataFrame)):
            return result
        if isinstance(result, pandas.Series):
            if result.name is None:
                result.name = MODIN_UNNAMED_SERIES_LABEL
            result = result.to_frame()

        return query_compiler.__constructor__(result)

    return caller


@_inherit_docstrings(BaseQueryCompiler)
class NativeQueryCompiler(BaseQueryCompiler, QueryCompilerCaster):
    """
    Query compiler for the pandas storage format.

    This class translates common query compiler API into
    native library functions (e.g., pandas) to execute operations
    on small data depending on the threshold.

    Parameters
    ----------
    pandas_frame : pandas.DataFrame
        The pandas frame to query with the compiled queries.
    shape_hint : {"row", "column", None}, default: None
        Shape hint for frames known to be a column or a row, otherwise None.
    """

    _modin_frame: pandas.DataFrame
    _shape_hint: Optional[str]

    def __init__(self, pandas_frame, shape_hint: Optional[str] = None):
        if hasattr(pandas_frame, "_to_pandas"):
            pandas_frame = pandas_frame._to_pandas()
        if is_scalar(pandas_frame):
            pandas_frame = pandas.DataFrame([pandas_frame])
        elif not isinstance(pandas_frame, pandas.DataFrame):
            pandas_frame = pandas.DataFrame(pandas_frame)

        self._modin_frame = pandas_frame
        self._shape_hint = shape_hint

    def execute(self):
        pass

    @property
    def frame_has_materialized_dtypes(self) -> bool:
        """
        Check if the undelying dataframe has materialized dtypes.

        Returns
        -------
        bool
        """
        return True

    def set_frame_dtypes_cache(self, dtypes):
        """
        Set dtypes cache for the underlying dataframe frame.

        Parameters
        ----------
        dtypes : pandas.Series, ModinDtypes, callable or None

        Notes
        -----
        This function is for consistency with other QCs,
        dtypes should be assigned directly on the frame.
        """
        pass

    def set_frame_index_cache(self, index):
        """
        Set index cache for underlying dataframe.

        Parameters
        ----------
        index : sequence, callable or None

        Notes
        -----
        This function is for consistency with other QCs,
        index should be assigned directly on the frame.
        """
        pass

    @property
    def frame_has_index_cache(self):
        """
        Check if the index cache exists for underlying dataframe.

        Returns
        -------
        bool
        """
        return True

    @property
    def frame_has_dtypes_cache(self) -> bool:
        """
        Check if the dtypes cache exists for the underlying dataframe.

        Returns
        -------
        bool
        """
        return True

    def take_2d_positional(self, index=None, columns=None):
        index = slice(None) if index is None else index
        columns = slice(None) if columns is None else columns
        return self.__constructor__(self._modin_frame.iloc[index, columns])

    def copy(self):
        return self.__constructor__(self._modin_frame.copy())

    def setitem_bool(self, row_loc, col_loc, item):

        self._modin_frame.loc[row_loc._modin_frame.squeeze(axis=1), col_loc] = item
        return self.__constructor__(self._modin_frame)

    __and__ = _register_default_pandas(pandas.DataFrame.__and__)
    __dir__ = _register_default_pandas(pandas.DataFrame.__dir__)
    __eq__ = _register_default_pandas(pandas.DataFrame.__eq__)
    __format__ = _register_default_pandas(pandas.DataFrame.__format__)
    __ge__ = _register_default_pandas(pandas.DataFrame.__ge__)
    __gt__ = _register_default_pandas(pandas.DataFrame.__gt__)
    __le__ = _register_default_pandas(pandas.DataFrame.__le__)
    __lt__ = _register_default_pandas(pandas.DataFrame.__lt__)
    __ne__ = _register_default_pandas(pandas.DataFrame.__ne__)
    __or__ = _register_default_pandas(pandas.DataFrame.__or__)
    __rand__ = _register_default_pandas(pandas.DataFrame.__rand__)
    __reduce__ = _register_default_pandas(pandas.DataFrame.__reduce__, return_raw=True)
    __reduce_ex__ = _register_default_pandas(
        pandas.DataFrame.__reduce_ex__, return_raw=True
    )
    __ror__ = _register_default_pandas(pandas.DataFrame.__ror__)
    __rxor__ = _register_default_pandas(pandas.DataFrame.__rxor__)
    __sizeof__ = _register_default_pandas(pandas.DataFrame.__sizeof__)
    __xor__ = _register_default_pandas(pandas.DataFrame.__xor__)
    abs = _register_default_pandas(pandas.DataFrame.abs)
    add = _register_default_pandas(_register_binary("add"))
    all = _register_default_pandas(pandas.DataFrame.all)
    any = _register_default_pandas(pandas.DataFrame.any)
    apply = _register_default_pandas(pandas.DataFrame.apply)
    apply_on_series = _register_default_pandas(pandas.Series.apply, is_series=True)
    applymap = _register_default_pandas(pandas.DataFrame.applymap)
    astype = _register_default_pandas(pandas.DataFrame.astype)
    case_when = _register_default_pandas(pandas.Series.case_when)
    cat_codes = _register_default_pandas(lambda ser: ser.cat.codes, is_series=True)
    combine = _register_default_pandas(_combine)
    combine_first = _register_default_pandas(lambda df, other: df.combine_first(other))
    compare = _register_default_pandas(pandas.DataFrame.compare)
    concat = _register_default_pandas(_concat)
    conj = _register_default_pandas(
        lambda df, *args, **kwargs: pandas.DataFrame(np.conj(df))
    )
    convert_dtypes = _register_default_pandas(pandas.DataFrame.convert_dtypes)
    count = _register_default_pandas(pandas.DataFrame.count)
    corr = _register_default_pandas(pandas.DataFrame.corr)
    cov = _register_default_pandas(pandas.DataFrame.cov)
    cummax = _register_default_pandas(pandas.DataFrame.cummax)
    cummin = _register_default_pandas(pandas.DataFrame.cummin)
    cumprod = _register_default_pandas(pandas.DataFrame.cumprod)
    cumsum = _register_default_pandas(pandas.DataFrame.cumsum)
    delitem = _register_default_pandas(_delitem)
    df_update = _register_default_pandas(pandas.DataFrame.update, in_place=True)
    diff = _register_default_pandas(pandas.DataFrame.diff)
    dot = _register_default_pandas(_register_binary("dot"))
    drop = _register_default_pandas(_drop)
    dropna = _register_default_pandas(pandas.DataFrame.dropna)  # axis values switched?
    dt_ceil = _register_default_pandas(_dt_func_map("ceil"))
    dt_components = _register_default_pandas(_dt_prop_map("components"))
    dt_date = _register_default_pandas(_dt_prop_map("date"))
    dt_day = _register_default_pandas(_dt_prop_map("day"))
    dt_day_name = _register_default_pandas(_dt_func_map("day_name"))
    dt_dayofweek = _register_default_pandas(_dt_prop_map("dayofweek"))
    dt_dayofyear = _register_default_pandas(_dt_prop_map("dayofyear"))
    dt_days = _register_default_pandas(_dt_prop_map("days"))
    dt_days_in_month = _register_default_pandas(_dt_prop_map("days_in_month"))
    dt_daysinmonth = _register_default_pandas(_dt_prop_map("daysinmonth"))
    dt_end_time = _register_default_pandas(_dt_prop_map("end_time"))
    dt_floor = _register_default_pandas(_dt_func_map("floor"))
    dt_freq = _register_default_pandas(
        lambda df: pandas.DataFrame([df.squeeze(axis=1).dt.freq])
    )
    dt_hour = _register_default_pandas(_dt_prop_map("hour"))
    dt_is_leap_year = _register_default_pandas(_dt_prop_map("is_leap_year"))
    dt_is_month_end = _register_default_pandas(_dt_prop_map("is_month_end"))
    dt_is_month_start = _register_default_pandas(_dt_prop_map("is_month_start"))
    dt_is_quarter_end = _register_default_pandas(_dt_prop_map("is_quarter_end"))
    dt_is_quarter_start = _register_default_pandas(_dt_prop_map("is_quarter_start"))
    dt_is_year_end = _register_default_pandas(_dt_prop_map("is_year_end"))
    dt_is_year_start = _register_default_pandas(_dt_prop_map("is_year_start"))
    dt_microsecond = _register_default_pandas(_dt_prop_map("microsecond"))
    dt_microseconds = _register_default_pandas(_dt_prop_map("microseconds"))
    dt_minute = _register_default_pandas(_dt_prop_map("minute"))
    dt_month = _register_default_pandas(_dt_prop_map("month"))
    dt_month_name = _register_default_pandas(_dt_func_map("month_name"))
    dt_nanosecond = _register_default_pandas(_dt_prop_map("nanosecond"))
    dt_nanoseconds = _register_default_pandas(_dt_prop_map("nanoseconds"))
    dt_normalize = _register_default_pandas(_dt_func_map("normalize"))
    dt_quarter = _register_default_pandas(_dt_prop_map("quarter"))
    dt_qyear = _register_default_pandas(_dt_prop_map("qyear"))
    dt_round = _register_default_pandas(_dt_func_map("round"))
    dt_second = _register_default_pandas(_dt_prop_map("second"))
    dt_seconds = _register_default_pandas(_dt_prop_map("seconds"))
    dt_start_time = _register_default_pandas(_dt_prop_map("start_time"))
    dt_strftime = _register_default_pandas(_dt_func_map("strftime"))
    dt_time = _register_default_pandas(_dt_prop_map("time"))
    dt_timetz = _register_default_pandas(_dt_prop_map("timetz"))
    dt_to_period = _register_default_pandas(_dt_func_map("to_period"))
    dt_to_pydatetime = _register_default_pandas(_dt_func_map("to_pydatetime"))
    dt_to_pytimedelta = _register_default_pandas(_dt_func_map("to_pytimedelta"))
    dt_to_timestamp = _register_default_pandas(_dt_func_map("to_timestamp"))
    dt_total_seconds = _register_default_pandas(_dt_func_map("total_seconds"))
    dt_tz = _register_default_pandas(
        lambda df: pandas.DataFrame([df.squeeze(axis=1).dt.tz])
    )
    dt_tz_convert = _register_default_pandas(_dt_func_map("tz_convert"))
    dt_tz_localize = _register_default_pandas(_dt_func_map("tz_localize"))
    dt_week = _register_default_pandas(_dt_prop_map("week"))
    dt_weekday = _register_default_pandas(_dt_prop_map("weekday"))
    dt_weekofyear = _register_default_pandas(_dt_prop_map("weekofyear"))
    dt_year = _register_default_pandas(_dt_prop_map("year"))
    duplicated = _register_default_pandas(pandas.DataFrame.duplicated)
    eq = _register_default_pandas(_register_binary("eq"))
    equals = _register_default_pandas(_register_binary("equals"))
    eval = _register_default_pandas(pandas.DataFrame.eval)
    explode = _register_default_pandas(pandas.DataFrame.explode)
    expanding_count = _register_default_pandas(
        _register_expanding(pandas.core.window.expanding.Expanding.count)
    )
    expanding_sum = _register_default_pandas(
        _register_expanding(pandas.core.window.expanding.Expanding.sum)
    )
    expanding_mean = _register_default_pandas(
        _register_expanding(pandas.core.window.expanding.Expanding.mean)
    )
    expanding_median = _register_default_pandas(
        _register_expanding(pandas.core.window.expanding.Expanding.median)
    )
    expanding_std = _register_default_pandas(
        _register_expanding(pandas.core.window.expanding.Expanding.std)
    )
    expanding_min = _register_default_pandas(
        _register_expanding(pandas.core.window.expanding.Expanding.min)
    )
    expanding_max = _register_default_pandas(
        _register_expanding(pandas.core.window.expanding.Expanding.max)
    )
    expanding_skew = _register_default_pandas(
        _register_expanding(pandas.core.window.expanding.Expanding.skew)
    )
    expanding_kurt = _register_default_pandas(
        _register_expanding(pandas.core.window.expanding.Expanding.kurt)
    )
    expanding_sem = _register_default_pandas(
        _register_expanding(pandas.core.window.expanding.Expanding.sem)
    )
    expanding_quantile = _register_default_pandas(
        _register_expanding(pandas.core.window.expanding.Expanding.quantile)
    )
    expanding_aggregate = _register_default_pandas(
        _register_expanding(pandas.core.window.expanding.Expanding.aggregate)
    )
    expanding_var = _register_default_pandas(
        _register_expanding(pandas.core.window.expanding.Expanding.var)
    )
    expanding_rank = _register_default_pandas(
        _register_expanding(pandas.core.window.expanding.Expanding.rank)
    )

    fillna = _register_default_pandas(_fillna)
    first_valid_index = _register_default_pandas(
        pandas.DataFrame.first_valid_index, return_raw=True
    )
    floordiv = _register_default_pandas(_register_binary("floordiv"))
    ge = _register_default_pandas(_register_binary("ge"))
    get_dummies = _register_default_pandas(_get_dummies)
    getitem_array = _register_default_pandas(_getitem_array)
    getitem_row_array = _register_default_pandas(_getitem_row_array)
    groupby_agg = _register_default_pandas(_groupby("agg"))
    groupby_all = _register_default_pandas(_groupby("all"))
    groupby_any = _register_default_pandas(_groupby("any"))
    groupby_count = _register_default_pandas(_groupby("count"))
    groupby_cummax = _register_default_pandas(_groupby("cummax"))
    groupby_cummin = _register_default_pandas(_groupby("cummin"))
    groupby_cumprod = _register_default_pandas(_groupby("cumprod"))
    groupby_cumsum = _register_default_pandas(_groupby("cumsum"))
    groupby_dtypes = _register_default_pandas(_groupby("dtypes"))
    groupby_fillna = _register_default_pandas(_groupby("fillna"))
    groupby_max = _register_default_pandas(_groupby("max"))
    groupby_mean = _register_default_pandas(_groupby("mean"))
    groupby_median = _register_default_pandas(_groupby("median"))
    groupby_min = _register_default_pandas(_groupby("min"))
    groupby_nunique = _register_default_pandas(_groupby("nunique"))
    groupby_prod = _register_default_pandas(_groupby("prod"))
    groupby_quantile = _register_default_pandas(_groupby("quantile"))
    groupby_rank = _register_default_pandas(_groupby("rank"))
    groupby_shift = _register_default_pandas(_groupby("shift"))
    groupby_skew = _register_default_pandas(_groupby("skew"))
    groupby_std = _register_default_pandas(_groupby("std"))
    groupby_sum = _register_default_pandas(_groupby("sum"))
    groupby_var = _register_default_pandas(_groupby("var"))
    gt = _register_default_pandas(_register_binary("gt"))
    idxmax = _register_default_pandas(pandas.DataFrame.idxmax)
    idxmin = _register_default_pandas(pandas.DataFrame.idxmin)
    infer_objects = _register_default_pandas(
        pandas.DataFrame.infer_objects, return_raw=True
    )
    insert = _register_default_pandas(
        pandas.DataFrame.insert, in_place=True, squeeze_args=True
    )
    invert = _register_default_pandas(pandas.DataFrame.__invert__)
    is_monotonic = _register_default_pandas(
        _is_monotonic("is_monotonic"), is_series=True
    )
    is_monotonic_decreasing = _register_default_pandas(
        _is_monotonic("is_monotonic_decreasing"), is_series=True
    )
    is_monotonic_increasing = _register_default_pandas(
        _is_monotonic("is_monotonic_increasing"), is_series=True
    )
    isna = _register_default_pandas(pandas.DataFrame.isna)
    join = _register_default_pandas(pandas.DataFrame.join)
    kurt = _register_default_pandas(pandas.DataFrame.kurt, return_raw=True)
    last_valid_index = _register_default_pandas(
        pandas.DataFrame.last_valid_index, return_raw=True
    )
    le = _register_default_pandas(_register_binary("le"))
    lt = _register_default_pandas(_register_binary("lt"))
    # mad = _register_default_pandas(pandas.DataFrame.mad)
    mask = _register_default_pandas(pandas.DataFrame.mask)
    max = _register_default_pandas(pandas.DataFrame.max)
    map = _register_default_pandas(pandas.DataFrame.map)
    mean = _register_default_pandas(pandas.DataFrame.mean, return_raw=True)
    median = _register_default_pandas(pandas.DataFrame.median, return_raw=True)
    melt = _register_default_pandas(pandas.DataFrame.melt)
    memory_usage = _register_default_pandas(pandas.DataFrame.memory_usage)
    merge = _register_default_pandas(pandas.DataFrame.merge)
    min = _register_default_pandas(pandas.DataFrame.min)
    mod = _register_default_pandas(_register_binary("mod"))
    mode = _register_default_pandas(pandas.DataFrame.mode)
    mul = _register_default_pandas(_register_binary("mul"))
    ne = _register_default_pandas(_register_binary("ne"))
    negative = _register_default_pandas(pandas.DataFrame.__neg__)
    nlargest = _register_default_pandas(pandas.DataFrame.nlargest)
    notna = _register_default_pandas(pandas.DataFrame.notna)
    nsmallest = _register_default_pandas(lambda df, **kwargs: df.nsmallest(**kwargs))
    nunique = _register_default_pandas(pandas.DataFrame.nunique)
    pivot = _register_default_pandas(pandas.DataFrame.pivot)
    pivot_table = _register_default_pandas(pandas.DataFrame.pivot_table)
    pow = _register_default_pandas(_register_binary("pow"))
    prod = _register_default_pandas(pandas.DataFrame.prod)
    prod_min_count = _register_default_pandas(pandas.DataFrame.prod)
    quantile_for_list_of_values = _register_default_pandas(pandas.DataFrame.quantile)
    quantile_for_single_value = _register_default_pandas(pandas.DataFrame.quantile)
    query = _register_default_pandas(pandas.DataFrame.query)
    radd = _register_default_pandas(_register_binary("radd"))
    rank = _register_default_pandas(pandas.DataFrame.rank)
    reindex = _register_default_pandas(_reindex)
    repeat = _register_default_pandas(pandas.Series.repeat, is_series=True)
    replace = _register_default_pandas(pandas.DataFrame.replace)
    resample_agg_df = _register_default_pandas(_register_resample("agg"))
    resample_agg_ser = _register_default_pandas(
        _register_resample("agg"), is_series=True
    )
    resample_app_df = _register_default_pandas(_register_resample("apply"))
    resample_app_ser = _register_default_pandas(
        _register_resample("apply"), is_series=True
    )
    resample_asfreq = _register_default_pandas(_register_resample("asfreq"))
    resample_backfill = _register_default_pandas(_register_resample("backfill"))
    resample_bfill = _register_default_pandas(_register_resample("bfill"))
    resample_count = _register_default_pandas(_register_resample("count"))
    resample_ffill = _register_default_pandas(_register_resample("ffill"))
    resample_fillna = _register_default_pandas(_register_resample("fillna"))
    resample_first = _register_default_pandas(_register_resample("first"))
    resample_get_group = _register_default_pandas(_register_resample("get_group"))
    resample_interpolate = _register_default_pandas(_register_resample("interpolate"))
    resample_last = _register_default_pandas(_register_resample("last"))
    resample_max = _register_default_pandas(_register_resample("max"))
    resample_mean = _register_default_pandas(_register_resample("mean"))
    resample_median = _register_default_pandas(_register_resample("median"))
    resample_min = _register_default_pandas(_register_resample("min"))
    resample_nearest = _register_default_pandas(_register_resample("nearest"))
    resample_nunique = _register_default_pandas(_register_resample("nunique"))
    resample_ohlc_df = _register_default_pandas(_register_resample("ohlc"))
    resample_ohlc_ser = _register_default_pandas(
        _register_resample("ohlc"), is_series=True
    )
    resample_pad = _register_default_pandas(_register_resample("pad"))
    resample_pipe = _register_default_pandas(_register_resample("pipe"))
    resample_prod = _register_default_pandas(_register_resample("prod"))
    resample_quantile = _register_default_pandas(_register_resample("quantile"))
    resample_sem = _register_default_pandas(_register_resample("sem"))
    resample_size = _register_default_pandas(_register_resample("size"))
    resample_std = _register_default_pandas(_register_resample("std"))
    resample_sum = _register_default_pandas(_register_resample("sum"))
    resample_transform = _register_default_pandas(_register_resample("transform"))
    resample_var = _register_default_pandas(_register_resample("var"))
    reset_index = _register_default_pandas(pandas.DataFrame.reset_index)
    rfloordiv = _register_default_pandas(_register_binary("rfloordiv"))
    rmod = _register_default_pandas(_register_binary("rmod"))
    rolling_aggregate = _register_default_pandas(_rolling_func("aggregate"))
    rolling_apply = _register_default_pandas(_rolling_func("apply"))
    rolling_corr = _register_default_pandas(_rolling_func("corr"))
    rolling_count = _register_default_pandas(_rolling_func("count"))
    rolling_cov = _register_default_pandas(_rolling_func("cov"))
    rolling_kurt = _register_default_pandas(_rolling_func("kurt"))
    rolling_max = _register_default_pandas(_rolling_func("max"))
    rolling_mean = _register_default_pandas(_rolling_func("mean"))
    rolling_median = _register_default_pandas(_rolling_func("median"))
    rolling_min = _register_default_pandas(_rolling_func("min"))
    rolling_quantile = _register_default_pandas(_rolling_func("quantile"))
    rolling_skew = _register_default_pandas(_rolling_func("skew"))
    rolling_std = _register_default_pandas(_rolling_func("std"))
    rolling_sum = _register_default_pandas(_rolling_func("sum"))
    rolling_var = _register_default_pandas(_rolling_func("var"))
    round = _register_default_pandas(pandas.DataFrame.round)
    rmul = _register_default_pandas(_register_binary("rmul"))
    rpow = _register_default_pandas(_register_binary("rpow"))
    rsub = _register_default_pandas(_register_binary("rsub"))
    rtruediv = _register_default_pandas(_register_binary("rtruediv"))
    searchsorted = _register_default_pandas(pandas.Series.searchsorted, is_series=True)
    sem = _register_default_pandas(pandas.DataFrame.sem)
    series_view = _register_default_pandas(pandas.Series.view, is_series=True)
    set_index_from_columns = _register_default_pandas(pandas.DataFrame.set_index)
    setitem = _register_default_pandas(_setitem)
    skew = _register_default_pandas(pandas.DataFrame.skew, return_raw=True)
    sort_index = _register_default_pandas(_sort_index)
    sort_columns_by_row_values = _register_default_pandas(
        lambda df, columns, **kwargs: df.sort_values(by=columns, axis=1, **kwargs)
    )
    sort_rows_by_column_values = _register_default_pandas(
        lambda df, columns, **kwargs: df.sort_values(by=columns, axis=0, **kwargs)
    )
    stack = _register_default_pandas(pandas.DataFrame.stack)
    std = _register_default_pandas(pandas.DataFrame.std)
    str___getitem__ = _register_default_pandas(_str_map("__getitem__"))
    str_capitalize = _register_default_pandas(_str_map("capitalize"))
    str_center = _register_default_pandas(_str_map("center"))
    str_contains = _register_default_pandas(_str_map("contains"))
    str_count = _register_default_pandas(_str_map("count"))
    str_endswith = _register_default_pandas(_str_map("endswith"))
    str_find = _register_default_pandas(_str_map("find"))
    str_findall = _register_default_pandas(_str_map("findall"))
    str_get = _register_default_pandas(_str_map("get"))
    str_index = _register_default_pandas(_str_map("index"))
    str_isalnum = _register_default_pandas(_str_map("isalnum"))
    str_isalpha = _register_default_pandas(_str_map("isalpha"))
    str_isdecimal = _register_default_pandas(_str_map("isdecimal"))
    str_isdigit = _register_default_pandas(_str_map("isdigit"))
    str_islower = _register_default_pandas(_str_map("islower"))
    str_isnumeric = _register_default_pandas(_str_map("isnumeric"))
    str_isspace = _register_default_pandas(_str_map("isspace"))
    str_istitle = _register_default_pandas(_str_map("istitle"))
    str_isupper = _register_default_pandas(_str_map("isupper"))
    str_join = _register_default_pandas(_str_map("join"))
    str_len = _register_default_pandas(_str_map("len"))
    str_ljust = _register_default_pandas(_str_map("ljust"))
    str_lower = _register_default_pandas(_str_map("lower"))
    str_lstrip = _register_default_pandas(_str_map("lstrip"))
    str_match = _register_default_pandas(_str_map("match"))
    str_normalize = _register_default_pandas(_str_map("normalize"))
    str_pad = _register_default_pandas(_str_map("pad"))
    str_partition = _register_default_pandas(_str_map("partition"))
    str_repeat = _register_default_pandas(_str_map("repeat"))
    str_replace = _register_default_pandas(_str_map("replace"))
    str_rfind = _register_default_pandas(_str_map("rfind"))
    str_rindex = _register_default_pandas(_str_map("rindex"))
    str_rjust = _register_default_pandas(_str_map("rjust"))
    str_rpartition = _register_default_pandas(_str_map("rpartition"))
    str_rsplit = _register_default_pandas(_str_map("rsplit"))
    str_rstrip = _register_default_pandas(_str_map("rstrip"))
    str_slice = _register_default_pandas(_str_map("slice"))
    str_slice_replace = _register_default_pandas(_str_map("slice_replace"))
    str_split = _register_default_pandas(_str_map("split"))
    str_startswith = _register_default_pandas(_str_map("startswith"))
    str_strip = _register_default_pandas(_str_map("strip"))
    str_swapcase = _register_default_pandas(_str_map("swapcase"))
    str_title = _register_default_pandas(_str_map("title"))
    str_translate = _register_default_pandas(_str_map("translate"))
    str_upper = _register_default_pandas(_str_map("upper"))
    str_wrap = _register_default_pandas(_str_map("wrap"))
    str_zfill = _register_default_pandas(_str_map("zfill"))
    sub = _register_default_pandas(_register_binary("sub"))
    sum = _register_default_pandas(pandas.DataFrame.sum)
    sum_min_count = _register_default_pandas(pandas.DataFrame.sum)
    to_datetime = _register_default_pandas(_to_datetime)
    to_numeric = _register_default_pandas(_to_numeric)
    to_numpy = _register_default_pandas(pandas.DataFrame.to_numpy, return_raw=True)
    to_timedelta = _register_default_pandas(
        lambda ser, *args, **kwargs: pandas.to_timedelta(ser, *args, **kwargs),
        is_series=True,
    )
    transpose = _register_default_pandas(pandas.DataFrame.transpose)
    truediv = _register_default_pandas(_register_binary("truediv"))
    unstack = _register_default_pandas(pandas.DataFrame.unstack)
    var = _register_default_pandas(pandas.DataFrame.var)
    where = _register_default_pandas(pandas.DataFrame.where)
    window_mean = _register_default_pandas(_rolling_func("mean"))
    window_std = _register_default_pandas(_rolling_func("std"))
    window_sum = _register_default_pandas(_rolling_func("sum"))
    window_var = _register_default_pandas(_rolling_func("var"))
    write_items = _register_default_pandas(_write_items)

    T = property(transpose)

    add_prefix = _register_default_pandas(pandas.DataFrame.add_prefix)
    add_suffix = _register_default_pandas(pandas.DataFrame.add_suffix)

    def clip(self, lower, upper, **kwargs):
        if isinstance(lower, BaseQueryCompiler):
            lower = lower.to_pandas().squeeze(1)
        if isinstance(upper, BaseQueryCompiler):
            upper = upper.to_pandas().squeeze(1)
        return _register_default_pandas(pandas.DataFrame.clip)(
            self, lower, upper, **kwargs
        )

    def describe(self, percentiles: np.ndarray):
        return _register_default_pandas(pandas.DataFrame.describe)(
            self,
            percentiles=percentiles,
            include="all",
        )

    def series_update(self, other, **kwargs):
        return _register_default_pandas(_register_binary("update"), in_place=True)(
            self,
            other=other,
            squeeze_self=True,
            squeeze_other=True,
            **kwargs,
        )

    def expanding_cov(
        self,
        fold_axis,
        expanding_args,
        squeeze_self,
        squeeze_other,
        other=None,
        pairwise=None,
        ddof=1,
        numeric_only=False,
        **kwargs,
    ):
        other_for_default = (
            other
            if other is None
            else (
                other.to_pandas().squeeze(axis=1)
                if squeeze_other
                else other.to_pandas()
            )
        )
        return _register_default_pandas(
            _register_expanding(pandas.core.window.expanding.Expanding.cov)
        )(
            self,
            fold_axis,
            expanding_args,
            other=other_for_default,
            pairwise=pairwise,
            ddof=ddof,
            numeric_only=numeric_only,
            squeeze_self=squeeze_self,
            **kwargs,
        )

    def expanding_corr(
        self,
        fold_axis,
        expanding_args,
        squeeze_self,
        squeeze_other,
        other=None,
        pairwise=None,
        ddof=1,
        numeric_only=False,
        **kwargs,
    ):
        other_for_default = (
            other
            if other is None
            else (
                other.to_pandas().squeeze(axis=1)
                if squeeze_other
                else other.to_pandas()
            )
        )
        return _register_default_pandas(
            _register_expanding(pandas.core.window.expanding.Expanding.corr)
        )(
            self,
            fold_axis,
            expanding_args,
            other=other_for_default,
            pairwise=pairwise,
            ddof=ddof,
            numeric_only=numeric_only,
            squeeze_self=squeeze_self,
            **kwargs,
        )

    def groupby_size(
        self,
        by,
        axis,
        groupby_kwargs,
        agg_args,
        agg_kwargs,
        drop=False,
    ):
        result = _register_default_pandas(_groupby("size"))(
            self,
            by=by,
            axis=axis,
            groupby_kwargs=groupby_kwargs,
            agg_args=agg_args,
            agg_kwargs=agg_kwargs,
            drop=drop,
            method="size",
        )
        if not groupby_kwargs.get("as_index", False):
            # Renaming 'MODIN_UNNAMED_SERIES_LABEL' to a proper name

            result.columns = result.columns[:-1].append(pandas.Index(["size"]))
        return result

    def get_axis(self, axis):
        return self._modin_frame.index if axis == 0 else self._modin_frame.columns

    def get_index_name(self, axis=0):
        return self.get_axis(axis).name

    def get_index_names(self, axis=0):
        return self.get_axis(axis).names

    def set_index_name(self, name, axis=0):
        self.get_axis(axis).name = name

    def has_multiindex(self, axis=0):
        if axis == 0:
            return isinstance(self._modin_frame.index, pandas.MultiIndex)
        assert axis == 1
        return isinstance(self._modin_frame.columns, pandas.MultiIndex)

    def isin(self, values, ignore_indices=False, **kwargs):
        if isinstance(values, type(self)) and ignore_indices:
            # Pandas logic is that it ignores indexing if 'values' is a 1D object
            values = values.to_pandas().squeeze(axis=1)
        if self._shape_hint == "column":
            return _register_default_pandas(pandas.Series.isin, is_series=True)(
                self, values, **kwargs
            )
        else:
            return _register_default_pandas(pandas.DataFrame.isin)(
                self, values, **kwargs
            )

    def to_pandas(self):
        return self._modin_frame

    @classmethod
    def from_pandas(cls, df, data_cls):
        return cls(df)

    @classmethod
    def from_arrow(cls, at, data_cls):
        return cls(at.to_pandas())

    def free(self):
        return

    def finalize(self):
        return

    # Dataframe exchange protocol

    def to_dataframe(self, nan_as_null: bool = False, allow_copy: bool = True):
        return self._modin_frame.__dataframe__(
            nan_as_null=nan_as_null, allow_copy=allow_copy
        )

    @classmethod
    def from_dataframe(cls, df, data_cls):
        return cls(pandas.api.interchange.from_dataframe(df))

    # END Dataframe exchange protocol

    index = property(_get_axis(0), _set_axis(0))
    columns = property(_get_axis(1), _set_axis(1))

    @property
    def dtypes(self):
        return self._modin_frame.dtypes

    def getitem_column_array(self, key, numeric=False, ignore_order=False):
        if numeric:
            return self.__constructor__(self._modin_frame.iloc[:, key])
        return self.__constructor__(self._modin_frame.loc[:, key])

    def is_series_like(self):
        return len(self._modin_frame.columns) == 1 or len(self._modin_frame.index) == 1

    def support_materialization_in_worker_process(self) -> bool:
        """
        Whether it's possible to call function `to_pandas` during the pickling process, at the moment of recreating the object.

        Returns
        -------
        bool
        """
        return False

    def get_pandas_backend(self) -> Optional[str]:
        """
        Get backend stored in `_modin_frame`.

        Returns
        -------
        str | None
            Backend name.
        """
        return None
