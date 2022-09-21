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
Module contains ``SmallQueryCompiler`` class.

``SmallQueryCompiler`` is responsible for compiling efficient DataFrame algebra
queries for small data and empty ``PandasDataFrame``.
"""

import pandas
from pandas.core.indexes.api import ensure_index_from_sequences
from typing import List, Hashable

from modin.core.storage_formats.base.query_compiler import BaseQueryCompiler
from modin.utils import MODIN_UNNAMED_SERIES_LABEL
from modin.utils import (
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
    callable(PandasQueryCompiler) -> pandas.Index
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
    callable(PandasQueryCompiler)
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
        Function to be applied in the partitions.

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
        Function to be applied in the partitions.

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


def _resample_func(
    df, resample_kwargs, func_name, new_columns=None, df_op=None, *args, **kwargs
):
    """
    Resample underlying time-series data and apply aggregation on it.

    Parameters
    ----------
    resample_kwargs : dict
        Resample parameters in the format of ``modin.pandas.DataFrame.resample`` signature.
    func_name : str
        Aggregation function name to apply on resampler object.
    new_columns : list of labels, optional
        Actual column labels of the resulted frame, supposed to be a hint for the
        Modin frame. If not specified will be computed automaticly.
    df_op : callable(pandas.DataFrame) -> [pandas.DataFrame, pandas.Series], optional
        Preprocessor function to apply to the passed frame before resampling.
    *args : args
        Arguments to pass to the aggregation function.
    **kwargs : kwargs
        Arguments to pass to the aggregation function.

    Returns
    -------
    PandasQueryCompiler
        New QueryCompiler containing the result of resample aggregation.
    """

    """Resample time-series data of the passed frame and apply aggregation function on it."""
    if df_op is not None:
        df = df_op(df)
    resampled_val = df.resample(**resample_kwargs)
    op = getattr(pandas.core.resample.Resampler, func_name)
    if callable(op):
        try:
            # This will happen with Arrow buffer read-only errors. We don't want to copy
            # all the time, so this will try to fast-path the code first.
            return op(df, resampled_val, *args, **kwargs)
        except (ValueError):
            resampled_val = df.copy().resample(**resample_kwargs)
            return op(df, resampled_val, *args, **kwargs)
    else:
        return getattr(df, resampled_val, func_name)


def _resample_get_group(df, resample_kwargs, name, obj):
    return _resample_func(df, resample_kwargs, "get_group", name=name, obj=obj)


def _resample_app_ser(df, resample_kwargs, func, *args, **kwargs):
    return _resample_func(
        resample_kwargs,
        "apply",
        df_op=lambda df: df.squeeze(axis=1),
        func=func,
        *args,
        **kwargs,
    )


def _resample_app_df(df, resample_kwargs, func, *args, **kwargs):
    return _resample_func(df, resample_kwargs, "apply", func=func, *args, **kwargs)


def _resample_agg_ser(df, resample_kwargs, func, *args, **kwargs):
    return _resample_func(
        resample_kwargs,
        "aggregate",
        df_op=lambda df: df.squeeze(axis=1),
        func=func,
        *args,
        **kwargs,
    )


def _resample_agg_df(df, resample_kwargs, func, *args, **kwargs):
    return _resample_func(df, resample_kwargs, "aggregate", func=func, *args, **kwargs)


def _resample_transform(df, resample_kwargs, arg, *args, **kwargs):
    return _resample_func(df, resample_kwargs, "transform", arg=arg, *args, **kwargs)


def _resample_pipe(df, resample_kwargs, func, *args, **kwargs):
    return _resample_func(df, resample_kwargs, "pipe", func=func, *args, **kwargs)


def _resample_ffill(df, resample_kwargs, limit):
    return _resample_func(df, resample_kwargs, "ffill", limit=limit)


def _resample_backfill(df, resample_kwargs, limit):
    return _resample_func(df, resample_kwargs, "backfill", limit=limit)


def _resample_bfill(df, resample_kwargs, limit):
    return _resample_func(df, resample_kwargs, "bfill", limit=limit)


def _resample_pad(df, resample_kwargs, limit):
    return _resample_func(df, resample_kwargs, "pad", limit=limit)


def _resample_nearest(df, resample_kwargs, limit):
    return _resample_func(df, resample_kwargs, "nearest", limit=limit)


def _resample_fillna(df, resample_kwargs, method, limit):
    return _resample_func(df, resample_kwargs, "fillna", method=method, limit=limit)


def _resample_asfreq(df, resample_kwargs, fill_value):
    return _resample_func(df, resample_kwargs, "asfreq", fill_value=fill_value)


def _resample_interpolate(
    self,
    resample_kwargs,
    method,
    axis,
    limit,
    inplace,
    limit_direction,
    limit_area,
    downcast,
    **kwargs,
):
    return _resample_func(
        resample_kwargs,
        "interpolate",
        axis=axis,
        limit=limit,
        inplace=inplace,
        limit_direction=limit_direction,
        limit_area=limit_area,
        downcast=downcast,
        **kwargs,
    )


def _resample_count(df, resample_kwargs):
    return _resample_func(df, resample_kwargs, "count")


def _resample_nunique(df, resample_kwargs, _method, *args, **kwargs):
    return _resample_func(
        df, resample_kwargs, "nunique", _method=_method, *args, **kwargs
    )


def _resample_first(df, resample_kwargs, _method, *args, **kwargs):
    return _resample_func(
        df, resample_kwargs, "first", _method=_method, *args, **kwargs
    )


def _resample_last(df, resample_kwargs, _method, *args, **kwargs):
    return _resample_func(df, resample_kwargs, "last", _method=_method, *args, **kwargs)


def _resample_max(df, resample_kwargs, _method, *args, **kwargs):
    return _resample_func(df, resample_kwargs, "max", _method=_method, *args, **kwargs)


def _resample_mean(df, resample_kwargs, _method, *args, **kwargs):
    return _resample_func(
        df, resample_kwargs, "median", _method=_method, *args, **kwargs
    )


def _resample_median(df, resample_kwargs, _method, *args, **kwargs):
    return _resample_func(
        df, resample_kwargs, "median", _method=_method, *args, **kwargs
    )


def _resample_min(df, resample_kwargs, _method, *args, **kwargs):
    return _resample_func(df, resample_kwargs, "min", _method=_method, *args, **kwargs)


def _resample_ohlc_ser(df, resample_kwargs, _method, *args, **kwargs):
    return _resample_func(
        resample_kwargs,
        "ohlc",
        df_op=lambda df: df.squeeze(axis=1),
        _method=_method,
        *args,
        **kwargs,
    )


def _resample_ohlc_df(df, resample_kwargs, _method, *args, **kwargs):
    return _resample_func(df, resample_kwargs, "ohlc", _method=_method, *args, **kwargs)


def _resample_prod(df, resample_kwargs, _method, min_count, *args, **kwargs):
    return _resample_func(
        resample_kwargs,
        "prod",
        _method=_method,
        min_count=min_count,
        *args,
        **kwargs,
    )


def _resample_size(df, resample_kwargs):
    return _resample_func(
        resample_kwargs, "size", new_columns=[MODIN_UNNAMED_SERIES_LABEL]
    )


def _resample_sem(df, resample_kwargs, _method, *args, **kwargs):
    return _resample_func(df, resample_kwargs, "sem", _method=_method, *args, **kwargs)


def _resample_std(df, resample_kwargs, ddof, *args, **kwargs):
    return _resample_func(df, resample_kwargs, "std", ddof=ddof, *args, **kwargs)


def _resample_sum(df, resample_kwargs, _method, min_count, *args, **kwargs):
    return _resample_func(
        resample_kwargs,
        "sum",
        _method=_method,
        min_count=min_count,
        *args,
        **kwargs,
    )


def _resample_var(df, resample_kwargs, ddof, *args, **kwargs):
    return _resample_func(df, resample_kwargs, "var", ddof=ddof, *args, **kwargs)


def _resample_quantile(df, resample_kwargs, q, **kwargs):
    return _resample_func(df, resample_kwargs, "quantile", q=q, **kwargs)


def _rolling_func(func):
    def rolling_builder(df, rolling_args, *args, **kwargs):
        rolling_result = df.rolling(*rolling_args)
        rolling_op = getattr(rolling_result, func)
        return rolling_op(rolling_result, *args, **kwargs)

    return rolling_builder


@_inherit_docstrings(BaseQueryCompiler)
class SmallQueryCompiler(BaseQueryCompiler):
    """
    Query compiler for the pandas storage format.

    This class translates common query compiler API to default all methods
    to pandas.

    Parameters
    ----------
    modin_frame : pandas.DataFrame
        Modin Frame to query with the compiled queries.
    """

    def __init__(self, modin_frame):
        self._modin_frame = modin_frame
        print("THERES SOMETHING HERE")

    def default_to_pandas(self, pandas_op, *args, **kwargs):
        # type(self) might not work
        # SmallQueryCompiler and PandasQueryCompiler are not the same
        args = (a.to_pandas() if isinstance(a, type(self)) else a for a in args)
        kwargs = {
            k: v.to_pandas if isinstance(v, type(self)) else v
            for k, v in kwargs.items()
        }

        result = pandas_op(self._modin_frame, *args, **kwargs)
        if isinstance(result, pandas.Series):
            if result.name is None:
                result.name = "__reduced__"
            result = result.to_frame()

        return result
        # if isinstance(result, pandas.DataFrame):
        #     return self.from_pandas(result, type(self._modin_frame))
        # else:
        #     return result

    def _register_default_pandas(func):
        def caller(query_compiler, broadcast=None, fold_axis=None, *args, **kwargs):
            print(func.__name__)
            # exclude_names = ["broadcast", "fold_axis"]
            args = try_cast_to_pandas(args)
            # if "fold_axis" in kwargs:
            #     kwargs["axis"] = kwargs["fold_axis"]
            # kwargs = {k: v for k, v in kwargs.items() if k not in exclude_names}
            kwargs = try_cast_to_pandas(kwargs)
            result = func(query_compiler._modin_frame, *args, **kwargs)
            if isinstance(result, pandas.Series):
                if result.name is None:
                    result.name = "__reduced__"
                result = result.to_frame()
            # Add check if need to turn into regular query compiler here
            return query_compiler.__constructor__(result)

        return caller

    abs = _register_default_pandas(pandas.DataFrame.abs)
    add = _register_default_pandas(pandas.DataFrame.add)
    all = _register_default_pandas(pandas.DataFrame.all)
    any = _register_default_pandas(pandas.DataFrame.any)
    applymap = _register_default_pandas(pandas.DataFrame.applymap)
    astype = _register_default_pandas(pandas.DataFrame.astype)
    combine = _register_default_pandas(pandas.DataFrame.combine)
    combine_first = _register_default_pandas(pandas.DataFrame.combine_first)
    # conj = _register_default_pandas(pandas.DataFrame.conj)
    convert_dtypes = _register_default_pandas(pandas.DataFrame.convert_dtypes)
    copy = _register_default_pandas(pandas.DataFrame.copy)
    count = _register_default_pandas(pandas.DataFrame.count)
    cummax = _register_default_pandas(pandas.DataFrame.cummax)
    cummin = _register_default_pandas(pandas.DataFrame.cummin)
    cumprod = _register_default_pandas(pandas.DataFrame.cumprod)
    cumsum = _register_default_pandas(pandas.DataFrame.cumsum)
    # df_update = _register_default_pandas(pandas.DataFrame.df_update)
    diff = _register_default_pandas(pandas.DataFrame.diff)
    dt_ceil = _register_default_pandas(_dt_func_map("ceil"))
    # dt_components ?
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
    # dt_freq ?
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
    # dt_tz ?
    dt_tz_convert = _register_default_pandas(_dt_func_map("tz_convert"))
    dt_tz_localize = _register_default_pandas(_dt_func_map("tz_localize"))
    dt_week = _register_default_pandas(_dt_prop_map("week"))
    dt_weekday = _register_default_pandas(_dt_prop_map("weekday"))
    dt_weekofyear = _register_default_pandas(_dt_prop_map("weekofyear"))
    dt_year = _register_default_pandas(_dt_prop_map("year"))
    eq = _register_default_pandas(pandas.DataFrame.eq)
    explode = _register_default_pandas(pandas.DataFrame.explode)
    floordiv = _register_default_pandas(pandas.DataFrame.floordiv)
    ge = _register_default_pandas(pandas.DataFrame.ge)
    # groupby_agg
    # groupby_all
    # groupby_any
    # groupby_count
    # groupby_cummax
    # groupby_cummin
    # groupby_cumprod
    # groupby_cumsum
    # groupby_dtypes
    # groupby_fillna
    # groupby_max
    # groupby_mean
    # groupby_median
    # groupby_min
    # groupby_nunique
    # groupby_prod
    # groupby_quantile
    # groupby_rank
    # groupby_shift
    # groupby_size
    # groupby_skew
    # groupby_std
    # groupby_sum
    # groupby_var
    gt = _register_default_pandas(pandas.DataFrame.gt)
    idxmax = _register_default_pandas(pandas.DataFrame.idxmax)
    idxmin = _register_default_pandas(pandas.DataFrame.idxmin)
    invert = _register_default_pandas(pandas.DataFrame.__invert__)
    # is_monotonic_decreasing = _register_default_pandas(
    #     pandas.DataFrame.is_monotonic_decreasing
    # )
    # is_monotonic_increasing = _register_default_pandas(
    #     pandas.DataFrame.is_monotonic_increasing
    # )
    isin = _register_default_pandas(pandas.DataFrame.isin)
    isna = _register_default_pandas(pandas.DataFrame.isna)
    kurt = _register_default_pandas(pandas.DataFrame.kurt)
    le = _register_default_pandas(pandas.DataFrame.le)
    lt = _register_default_pandas(pandas.DataFrame.lt)
    mad = _register_default_pandas(pandas.DataFrame.mad)
    max = _register_default_pandas(pandas.DataFrame.max)
    mean = _register_default_pandas(pandas.DataFrame.mean)
    median = _register_default_pandas(pandas.DataFrame.median)
    memory_usage = _register_default_pandas(pandas.DataFrame.memory_usage)
    min = _register_default_pandas(pandas.DataFrame.min)
    mod = _register_default_pandas(pandas.DataFrame.mod)
    mode = _register_default_pandas(pandas.DataFrame.mode)
    mul = _register_default_pandas(pandas.DataFrame.mul)
    ne = _register_default_pandas(pandas.DataFrame.ne)
    negative = _register_default_pandas(pandas.DataFrame.__neg__)
    notna = _register_default_pandas(pandas.DataFrame.notna)
    pow = _register_default_pandas(pandas.DataFrame.pow)
    prod = _register_default_pandas(pandas.DataFrame.prod)
    prod_min_count = _register_default_pandas(pandas.DataFrame.prod)
    radd = _register_default_pandas(pandas.DataFrame.radd)
    replace = _register_default_pandas(pandas.DataFrame.replace)
    resample_agg_df = _register_default_pandas(_resample_agg_df)
    resample_agg_ser = _register_default_pandas(_resample_agg_ser)
    resample_app_df = _register_default_pandas(_resample_app_df)
    resample_app_ser = _register_default_pandas(_resample_app_ser)
    resample_asfreq = _register_default_pandas(_resample_asfreq)
    resample_backfill = _register_default_pandas(_resample_backfill)
    resample_bfill = _register_default_pandas(_resample_bfill)
    resample_count = _register_default_pandas(_resample_count)
    resample_ffill = _register_default_pandas(_resample_ffill)
    resample_fillna = _register_default_pandas(_resample_fillna)
    resample_first = _register_default_pandas(_resample_first)
    resample_get_group = _register_default_pandas(_resample_get_group)
    resample_interpolate = _register_default_pandas(_resample_interpolate)
    resample_last = _register_default_pandas(_resample_last)
    resample_max = _register_default_pandas(_resample_max)
    resample_mean = _register_default_pandas(_resample_mean)
    resample_median = _register_default_pandas(_resample_median)
    resample_min = _register_default_pandas(_resample_min)
    resample_nearest = _register_default_pandas(_resample_nearest)
    resample_nunique = _register_default_pandas(_resample_nunique)
    resample_ohlc_df = _register_default_pandas(_resample_ohlc_df)
    resample_ohlc_ser = _register_default_pandas(_resample_ohlc_ser)
    resample_pad = _register_default_pandas(_resample_pad)
    resample_pipe = _register_default_pandas(_resample_pipe)
    resample_prod = _register_default_pandas(_resample_prod)
    resample_quantile = _register_default_pandas(_resample_quantile)
    resample_sem = _register_default_pandas(_resample_sem)
    resample_size = _register_default_pandas(_resample_size)
    resample_std = _register_default_pandas(_resample_std)
    resample_sum = _register_default_pandas(_resample_sum)
    resample_transform = _register_default_pandas(_resample_transform)
    resample_var = _register_default_pandas(_resample_var)
    rfloordiv = _register_default_pandas(pandas.DataFrame.rfloordiv)
    rmod = _register_default_pandas(pandas.DataFrame.rmod)
    # rolling_aggregate
    rolling_apply = _register_default_pandas(_rolling_func("apply"))
    # rolling_corr
    rolling_count = _register_default_pandas(_rolling_func("count"))
    # rolling_cov
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
    rsub = _register_default_pandas(pandas.DataFrame.rsub)
    rtruediv = _register_default_pandas(pandas.DataFrame.rtruediv)
    sem = _register_default_pandas(pandas.DataFrame.sem)
    skew = _register_default_pandas(pandas.DataFrame.skew)
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
    sub = _register_default_pandas(pandas.DataFrame.sub)
    sum = _register_default_pandas(pandas.DataFrame.sum)
    sum_min_count = _register_default_pandas(pandas.DataFrame.sum)
    truediv = _register_default_pandas(pandas.DataFrame.truediv)
    var = _register_default_pandas(pandas.DataFrame.var)

    def to_pandas(self):
        return self._modin_frame

    @classmethod
    def from_pandas(cls, df, data_cls):
        return cls(data_cls.from_pandas(df))

    @classmethod
    def from_arrow(cls, at, data_cls):
        return

    def free(self):
        return

    def finalize(self):
        self._modin_frame.finalize()

    # Dataframe exchange protocol

    def to_dataframe(self, nan_as_null: bool = False, allow_copy: bool = True):
        return self._modin_frame.__dataframe__(
            nan_as_null=nan_as_null, allow_copy=allow_copy
        )

    @classmethod
    def from_dataframe(cls, df, data_cls):
        return cls(data_cls.from_dataframe(df))

    # END Dataframe exchange protocol

    index = property(_get_axis(0), _set_axis(0))
    columns = property(_get_axis(1), _set_axis(1))

    @property
    def dtypes(self):
        return self._modin_frame.dtypes

    # def __getattribute__(self, item):
    #     print("Getting attribute:", item)
    #     return super()._modin_frame.__getattribute__(item)

    # END Index, columns, and dtypes objects

    # Metadata modification methods
    def add_prefix(self, prefix, axis=1):
        return self.__constructor__(self._modin_frame.add_prefix(prefix, axis))

    def add_suffix(self, suffix, axis=1):
        return self.__constructor__(self._modin_frame.add_suffix(suffix, axis))

    # END Metadata modification methods
    def getitem_column_array(self, key, numeric=False):
        # Convert to list for type checking
        # if numeric:
        #     new_modin_frame = self._modin_frame.take_2d_labels_or_positional(
        #         col_positions=key
        #     )
        # else:
        #     new_modin_frame = self._modin_frame.take_2d_labels_or_positional(
        #         col_labels=key
        #     )
        return self.__constructor__(self._modin_frame[key])

    # Reindex/reset_index (may shuffle data)
    def reindex(self, axis, labels, **kwargs):
        new_index = self.index if axis else labels
        new_columns = labels if axis else self.columns
        new_modin_frame = self._modin_frame.apply_full_axis(
            axis,
            lambda df: df.reindex(labels=labels, axis=axis, **kwargs),
            new_index=new_index,
            new_columns=new_columns,
        )
        return self.__constructor__(new_modin_frame)

    def columnarize(self):
        if len(self._modin_frame.columns) != 1 or (
            len(self._modin_frame.index) == 1
            and self._modin_frame.index[0] == MODIN_UNNAMED_SERIES_LABEL
        ):
            return SmallQueryCompiler(self._modin_frame.transpose())
        return self

    def reset_index(self, **kwargs):
        drop = kwargs.get("drop", False)
        level = kwargs.get("level", None)
        new_index = None
        if level is not None:
            if not isinstance(level, (tuple, list)):
                level = [level]
            level = [self.index._get_level_number(lev) for lev in level]
            uniq_sorted_level = sorted(set(level))
            if len(uniq_sorted_level) < self.index.nlevels:
                # We handle this by separately computing the index. We could just
                # put the labels into the data and pull them back out, but that is
                # expensive.
                new_index = (
                    self.index.droplevel(uniq_sorted_level)
                    if len(level) < self.index.nlevels
                    else pandas.RangeIndex(len(self.index))
                )
        else:
            uniq_sorted_level = list(range(self.index.nlevels))

        if not drop:
            if len(uniq_sorted_level) < self.index.nlevels:
                # These are the index levels that will remain after the reset_index
                keep_levels = [
                    i for i in range(self.index.nlevels) if i not in uniq_sorted_level
                ]
                new_copy = self.copy()
                # Change the index to have only the levels that will be inserted
                # into the data. We will replace the old levels later.
                new_copy.index = self.index.droplevel(keep_levels)
                new_copy.index.names = [
                    "level_{}".format(level_value)
                    if new_copy.index.names[level_index] is None
                    else new_copy.index.names[level_index]
                    for level_index, level_value in enumerate(uniq_sorted_level)
                ]
                new_modin_frame = new_copy._modin_frame.from_labels()
                # Replace the levels that will remain as a part of the index.
                new_modin_frame.index = new_index
            else:
                new_modin_frame = self._modin_frame.from_labels()
            if isinstance(new_modin_frame.columns, pandas.MultiIndex):
                # Fix col_level and col_fill in generated column names because from_labels works with assumption
                # that col_level and col_fill are not specified but it expands tuples in level names.
                col_level = kwargs.get("col_level", 0)
                col_fill = kwargs.get("col_fill", "")
                if col_level != 0 or col_fill != "":
                    # Modify generated column names if col_level and col_fil have values different from default.
                    levels_names_list = [
                        f"level_{level_index}" if level_name is None else level_name
                        for level_index, level_name in enumerate(self.index.names)
                    ]
                    if col_fill is None:
                        # Initialize col_fill if it is None.
                        # This is some weird undocumented Pandas behavior to take first
                        # element of the last column name.
                        last_col_name = levels_names_list[uniq_sorted_level[-1]]
                        last_col_name = (
                            list(last_col_name)
                            if isinstance(last_col_name, tuple)
                            else [last_col_name]
                        )
                        if len(last_col_name) not in (1, self.columns.nlevels):
                            raise ValueError(
                                "col_fill=None is incompatible "
                                + f"with incomplete column name {last_col_name}"
                            )
                        col_fill = last_col_name[0]
                    columns_list = new_modin_frame.columns.tolist()
                    for level_index, level_value in enumerate(uniq_sorted_level):
                        level_name = levels_names_list[level_value]
                        # Expand tuples into separate items and fill the rest with col_fill
                        top_level = [col_fill] * col_level
                        middle_level = (
                            list(level_name)
                            if isinstance(level_name, tuple)
                            else [level_name]
                        )
                        bottom_level = [col_fill] * (
                            self.columns.nlevels - (col_level + len(middle_level))
                        )
                        item = tuple(top_level + middle_level + bottom_level)
                        if len(item) > self.columns.nlevels:
                            raise ValueError(
                                "Item must have length equal to number of levels."
                            )
                        columns_list[level_index] = item
                    new_modin_frame.columns = pandas.MultiIndex.from_tuples(
                        columns_list, names=self.columns.names
                    )
            new_self = self.__constructor__(new_modin_frame)
        else:
            new_self = self.copy()
            new_self.index = (
                pandas.RangeIndex(len(new_self.index))
                if new_index is None
                else new_index
            )
        return new_self

    def set_index_from_columns(
        self, keys: List[Hashable], drop: bool = True, append: bool = False
    ):
        new_modin_frame = self._modin_frame.to_labels(keys)
        if append:
            arrays = []
            # Appending keeps the original order of the index levels, then appends the
            # new index objects.
            names = list(self.index.names)
            if isinstance(self.index, pandas.MultiIndex):
                for i in range(self.index.nlevels):
                    arrays.append(self.index._get_level_values(i))
            else:
                arrays.append(self.index)

            # Add the names in the correct order.
            names.extend(new_modin_frame.index.names)
            if isinstance(new_modin_frame.index, pandas.MultiIndex):
                for i in range(new_modin_frame.index.nlevels):
                    arrays.append(new_modin_frame.index._get_level_values(i))
            else:
                arrays.append(new_modin_frame.index)
            new_modin_frame.index = ensure_index_from_sequences(arrays, names)
        if not drop:
            # The algebraic operator for this operation always drops the column, but we
            # can copy the data in this object and just use the index from the result of
            # the query compiler call.
            result = self._modin_frame.copy()
            result.index = new_modin_frame.index
        else:
            result = new_modin_frame
        return self.__constructor__(result)

    # END Reindex/reset_index
