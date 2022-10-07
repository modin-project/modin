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

import numpy as np
import pandas
from pandas.core.indexes.api import ensure_index_from_sequences
from pandas.core.dtypes.common import (
    is_list_like,
    is_scalar,
)
from typing import List, Hashable

from modin.core.storage_formats.base.query_compiler import BaseQueryCompiler
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.utils import MODIN_UNNAMED_SERIES_LABEL
from modin.utils import (
    _inherit_docstrings,
    try_cast_to_pandas_sqc,
)


MODIN_UNNAMED_SERIES_LABEL = "__reduced__"


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


def _rolling_func(func):
    def rolling_builder(df, rolling_args, *args, **kwargs):
        rolling_result = df.rolling(*rolling_args)
        rolling_op = getattr(rolling_result, func)
        return rolling_op(rolling_result, *args, **kwargs)

    return rolling_builder


def _reindex(df, axis, labels, **kwargs):
    return df.reindex(labels=labels, axis=axis, **kwargs)


def _concat(df, axis, other, join=None, join_axes=None, **kwargs):
    if not isinstance(other, list):
        other = [other]
    return pandas.concat([df] + other, axis=axis, **kwargs)


def _unique(values, **kwargs):
    # print(type(df))
    if isinstance(values, pandas.DataFrame):
        values = values.squeeze(axis=1)
    return pandas.unique(values, **kwargs)


def _to_datetime(df, *args, **kwargs):
    return pandas.to_datetime(df.squeeze(axis=1), *args, **kwargs)


def _to_numeric(df, *args, **kwargs):
    return pandas.to_numeric(df.squeeze(axis=1), *args, **kwargs)


def _groupby(agg_name):
    def groupby_callable(
        df, by, axis, groupby_kwargs, agg_args, agg_kwargs, drop=False
    ):
        groupby_obj = df.groupby(by=by, axis=axis, **groupby_kwargs)
        agg_func = getattr(groupby_obj, agg_name)
        return agg_func(*agg_args, **agg_kwargs)

    return groupby_callable


def _take_2d(df, index=None, columns=None):
    columns = columns if columns is not None else slice(None)
    index = index if index is not None else slice(None)
    print(df.iloc[index, columns])
    print(type(df.iloc[index, columns]))
    return df.iloc[index, columns]


def _register_binary(op):
    def binary_operator(df, other, **kwargs):
        if isinstance(other, pandas.DataFrame) and (
            not df.empty or (len(other.columns) > 1 and other.columns[0] == MODIN_UNNAMED_SERIES_LABEL)
        ):
            print("BEFORE SQUEEZE:", other)
            something = other.squeeze()
            print("AFTER SQUEEZE:", something)
        return getattr(df, op)(other, **kwargs)

    return binary_operator


def _register_resample(op):
    def resample_operator(df, resample_kwargs, *args, **kwargs):
        resampler = df.resample(**resample_kwargs)
        result = getattr(resampler, op)(*args, **kwargs)
        return result

    return resample_operator


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
        if not isinstance(modin_frame, pandas.DataFrame):
            modin_frame = pandas.DataFrame(modin_frame)
        self._modin_frame = modin_frame

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
                result.name = MODIN_UNNAMED_SERIES_LABEL
            result = result.to_frame()

        return result
        # if isinstance(result, pandas.DataFrame):
        #     return self.from_pandas(result, type(self._modin_frame))
        # else:
        #     return result

    def _register_default_pandas(
        func,
        is_series=False,
        return_modin=True,
        in_place=False,
        df_copy=False,
        filter_kwargs=[],
    ):
        def caller(query_compiler, *args, **kwargs):
            print(func.__name__)
            df = query_compiler._modin_frame
            if df_copy:
                df = df.copy()
            if is_series:
                df = df.squeeze(axis=1)
            exclude_names = [
                "broadcast",
                "fold_axis",
                "squeeze_self",
                "squeeze_value",
            ] + filter_kwargs
            for name in exclude_names:
                kwargs.pop(name, None)
            print("BEFORE ARGS:", args)
            args = try_cast_to_pandas_sqc(args)
            kwargs = try_cast_to_pandas_sqc(kwargs)
            print("ARGS:", args)
            print("KWARGS:", kwargs)
            if func.__name__ == "transform":
                assert False
            result = func(df, *args, **kwargs)
            if in_place:
                result = df
            if not return_modin:
                return result
            if isinstance(result, pandas.Series):
                if result.name is None:
                    result.name = MODIN_UNNAMED_SERIES_LABEL
                result = result.to_frame()
            return query_compiler.__constructor__(result)

        return caller

    __and__ = _register_default_pandas(pandas.DataFrame.__and__)
    # __class__ = _register_default_pandas(pandas.DataFrame.__class__)
    # __delattr__ = _register_default_pandas(pandas.DataFrame.__delattr__)
    __dir__ = _register_default_pandas(pandas.DataFrame.__dir__)
    __eq__ = _register_default_pandas(pandas.DataFrame.__eq__)
    __format__ = _register_default_pandas(pandas.DataFrame.__format__)
    __ge__ = _register_default_pandas(pandas.DataFrame.__ge__)
    # __getattribute__ = _register_default_pandas(pandas.DataFrame.__getattribute__)
    __gt__ = _register_default_pandas(pandas.DataFrame.__gt__)
    # __hash__ = _register_default_pandas(pandas.DataFrame.__hash__)
    # __init__ = _register_default_pandas(pandas.DataFrame.__init__)
    # __init_subclass__ = _register_default_pandas(pandas.DataFrame.__init_subclass__)
    __le__ = _register_default_pandas(pandas.DataFrame.__le__)
    __lt__ = _register_default_pandas(pandas.DataFrame.__lt__)
    __ne__ = _register_default_pandas(pandas.DataFrame.__ne__)
    # __new__ = _register_default_pandas(pandas.DataFrame.__new__)
    __or__ = _register_default_pandas(pandas.DataFrame.__or__)
    __rand__ = _register_default_pandas(pandas.DataFrame.__rand__)
    __reduce__ = _register_default_pandas(
        pandas.DataFrame.__reduce__, return_modin=False
    )
    __reduce_ex__ = _register_default_pandas(
        pandas.DataFrame.__reduce_ex__, return_modin=False
    )
    # __repr__ = _register_default_pandas(pandas.DataFrame.__repr__)
    __ror__ = _register_default_pandas(pandas.DataFrame.__ror__)
    __rxor__ = _register_default_pandas(pandas.DataFrame.__rxor__)
    # __setattr__ = _register_default_pandas(pandas.DataFrame.__setattr__)
    __sizeof__ = _register_default_pandas(pandas.DataFrame.__sizeof__)
    # __str__ = _register_default_pandas(pandas.DataFrame.__str__)
    # __subclasshook__ = _register_default_pandas(pandas.DataFrame.__subclasshook__)
    __xor__ = _register_default_pandas(pandas.DataFrame.__xor__)
    abs = _register_default_pandas(pandas.DataFrame.abs)
    add = _register_default_pandas(_register_binary("add"))
    all = _register_default_pandas(pandas.DataFrame.all)
    any = _register_default_pandas(pandas.DataFrame.any)
    apply = _register_default_pandas(pandas.DataFrame.apply)
    apply_on_series = _register_default_pandas(pandas.Series.apply, is_series=True)
    applymap = _register_default_pandas(pandas.DataFrame.applymap)
    astype = _register_default_pandas(pandas.DataFrame.astype)
    cat_codes = _register_default_pandas(pandas.Series.cat.codes, is_series=True)
    clip = _register_default_pandas(pandas.DataFrame.clip)
    combine = _register_default_pandas(pandas.DataFrame.combine)
    combine_first = _register_default_pandas(pandas.DataFrame.combine_first)
    # conj = _register_default_pandas(pandas.DataFrame.conj)
    compare = _register_default_pandas(pandas.DataFrame.compare)
    concat = _register_default_pandas(_concat)
    conj = _register_default_pandas(
        lambda df, *args, **kwargs: pandas.DataFrame(np.conj(df))
    )
    convert_dtypes = _register_default_pandas(pandas.DataFrame.convert_dtypes)
    copy = _register_default_pandas(pandas.DataFrame.copy)
    count = _register_default_pandas(pandas.DataFrame.count)
    corr = _register_default_pandas(pandas.DataFrame.corr)
    cov = _register_default_pandas(pandas.DataFrame.cov)
    cummax = _register_default_pandas(pandas.DataFrame.cummax)
    cummin = _register_default_pandas(pandas.DataFrame.cummin)
    cumprod = _register_default_pandas(pandas.DataFrame.cumprod)
    cumsum = _register_default_pandas(pandas.DataFrame.cumsum)
    describe = _register_default_pandas(pandas.DataFrame.describe)
    df_update = _register_default_pandas(
        pandas.DataFrame.update, in_place=True, df_copy=True
    )
    diff = _register_default_pandas(pandas.DataFrame.diff)
    drop = _register_default_pandas(pandas.DataFrame.drop)
    dropna = _register_default_pandas(pandas.DataFrame.dropna)  # axis values switched?
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
    eq = _register_default_pandas(_register_binary("eq"), filter_kwargs=["dtypes"])
    eval = _register_default_pandas(pandas.DataFrame.eval)
    explode = _register_default_pandas(pandas.DataFrame.explode)
    fillna = _register_default_pandas(pandas.DataFrame.fillna)
    first_valid_index = _register_default_pandas(
        pandas.DataFrame.first_valid_index, return_modin=False
    )
    floordiv = _register_default_pandas(_register_binary("floordiv"))
    ge = _register_default_pandas(pandas.DataFrame.ge, filter_kwargs=["dtypes"])
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
    groupby_size = _register_default_pandas(_groupby("size"))
    groupby_skew = _register_default_pandas(_groupby("skew"))
    groupby_std = _register_default_pandas(_groupby("std"))
    groupby_sum = _register_default_pandas(_groupby("sum"))
    groupby_var = _register_default_pandas(_groupby("var"))
    gt = _register_default_pandas(pandas.DataFrame.gt, filter_kwargs=["dtypes"])
    idxmax = _register_default_pandas(pandas.DataFrame.idxmax)
    idxmin = _register_default_pandas(pandas.DataFrame.idxmin)
    insert = _register_default_pandas(pandas.DataFrame.insert, in_place=True)
    invert = _register_default_pandas(pandas.DataFrame.__invert__)
    # is_monotonic_decreasing = _register_default_pandas(
    #     pandas.DataFrame.is_monotonic_decreasing
    # )
    # is_monotonic_increasing = _register_default_pandas(
    #     pandas.DataFrame.is_monotonic_increasing
    # )
    isin = _register_default_pandas(pandas.DataFrame.isin)
    isna = _register_default_pandas(pandas.DataFrame.isna)
    # join = _register_default_pandas(pandas.DataFrame.join)
    def _join(*args, **kwargs):
        print("ALFJSLKJDF", kwargs)
        return pandas.DataFrame.join(*args, **kwargs)

    join = _register_default_pandas(_join)
    kurt = _register_default_pandas(pandas.DataFrame.kurt)
    last_valid_index = _register_default_pandas(
        pandas.DataFrame.last_valid_index, return_modin=False
    )
    le = _register_default_pandas(pandas.DataFrame.le, filter_kwargs=["dtypes"])
    lt = _register_default_pandas(pandas.DataFrame.lt, filter_kwargs=["dtypes"])
    mad = _register_default_pandas(pandas.DataFrame.mad)
    max = _register_default_pandas(pandas.DataFrame.max)
    mean = _register_default_pandas(pandas.DataFrame.mean)
    median = _register_default_pandas(pandas.DataFrame.median)
    melt = _register_default_pandas(pandas.DataFrame.melt)
    memory_usage = _register_default_pandas(pandas.DataFrame.memory_usage)
    merge = _register_default_pandas(pandas.DataFrame.merge)
    min = _register_default_pandas(pandas.DataFrame.min)
    mod = _register_default_pandas(_register_binary("mod"))
    mode = _register_default_pandas(pandas.DataFrame.mode)
    mul = _register_default_pandas(_register_binary("mul"))
    ne = _register_default_pandas(pandas.DataFrame.ne, filter_kwargs=["dtypes"])
    negative = _register_default_pandas(pandas.DataFrame.__neg__)
    nlargest = _register_default_pandas(pandas.DataFrame.nlargest)
    notna = _register_default_pandas(pandas.DataFrame.notna)
    nsmallest = _register_default_pandas(pandas.DataFrame.nsmallest)
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
    rmul = _register_default_pandas(_register_binary("rmul"))
    rpow = _register_default_pandas(_register_binary("rpow"))
    rsub = _register_default_pandas(_register_binary("rsub"))
    rtruediv = _register_default_pandas(_register_binary("rtruediv"))
    searchsorted = _register_default_pandas(pandas.Series.searchsorted, is_series=True)
    sem = _register_default_pandas(pandas.DataFrame.sem)
    series_update = _register_default_pandas(
        pandas.Series.update, is_series=True, in_place=True, df_copy=True
    )
    series_view = _register_default_pandas(pandas.Series.view, is_series=True)
    set_index_from_columns = _register_default_pandas(pandas.DataFrame.set_index)
    skew = _register_default_pandas(pandas.DataFrame.skew)
    sort_index = _register_default_pandas(pandas.DataFrame.sort_index)
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
    take_2d = _register_default_pandas(_take_2d)
    to_datetime = _register_default_pandas(_to_datetime)
    to_numeric = _register_default_pandas(_to_numeric)
    to_numpy = _register_default_pandas(pandas.DataFrame.to_numpy, return_modin=False)
    transpose = _register_default_pandas(pandas.DataFrame.transpose)
    truediv = _register_default_pandas(_register_binary("truediv"))
    # unique = _register_default_pandas(_unique, return_modin=False)
    unique = _register_default_pandas(pandas.Series.unique, is_series=True)
    unstack = _register_default_pandas(pandas.DataFrame.unstack)
    var = _register_default_pandas(pandas.DataFrame.var)
    where = _register_default_pandas(pandas.DataFrame.where)

    T = property(transpose)

    _add_prefix_df = _register_default_pandas(pandas.DataFrame.add_prefix)
    _add_prefix_series = _register_default_pandas(
        pandas.Series.add_prefix, is_series=True
    )

    def add_prefix(self, prefix, axis=1):
        if axis:
            return self._add_prefix_df(prefix=prefix)
        return self._add_prefix_series(prefix=prefix)

    _add_suffix_df = _register_default_pandas(pandas.DataFrame.add_suffix)
    _add_suffix_series = _register_default_pandas(
        pandas.Series.add_suffix, is_series=True
    )

    def add_suffix(self, suffix, axis=1):
        if axis:
            return self._add_suffix_df(suffix=suffix)
        return self._add_suffix_series(suffix=suffix)

    def dot(self, other, squeeze_self=None, squeeze_other=None):
        if isinstance(other, PandasQueryCompiler) or isinstance(
            other, SmallQueryCompiler
        ):
            other = (
                other.to_pandas().squeeze(axis=1)
                if squeeze_other
                else other.to_pandas()
            )
        if squeeze_self:
            result = self._modin_frame.squeeze(axis=1).dot(other)
        else:
            result = self._modin_frame.dot(other)
        if isinstance(result, pandas.Series):
            if result.name is None:
                result.name = "__reduced__"
            result = result.to_frame()
        if is_list_like(result):
            result = pandas.DataFrame(result)
        else:
            result = pandas.DataFrame([result])

        return self.__constructor__(result)

    # def drop(self, index=None, columns=None):
    #     result = self._modin_frame.drop(index=index, columns=columns)
    #     return self.__constructor__(result)

    def finalize(self):
        pass

    def get_axis(self, axis):
        return self._modin_frame.index if axis == 0 else self._modin_frame.columns

    def _get_dummies(df, columns, **kwargs):
        return pandas.get_dummies(df, columns=columns, **kwargs)

    def get_index_name(self, axis=0):
        return self.get_axis(axis).name

    def get_index_names(self, axis=0):
        return self.get_axis(axis).names

    def has_multiindex(self, axis=0):
        if axis == 0:
            return isinstance(self._modin_frame.index, pandas.MultiIndex)
        assert axis == 1
        return isinstance(self._modin_frame.columns, pandas.MultiIndex)

    def infer_objects(self):
        return self

    def insert_item(self, *args, **kwargs):
        print("Not implemented")
        return

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

    def _getitem_array(df, key):
        if isinstance(key, pandas.DataFrame):
            key = key.squeeze(axis=1)
        return df[key]

    def _getitem_row_array(df, key):
        if isinstance(key, pandas.DataFrame):
            key = key.squeeze(axis=1)
        return df.iloc[key]

    def columnarize(self):
        if len(self._modin_frame.columns) != 1 or (
            len(self._modin_frame.index) == 1
            and self._modin_frame.index[0] == MODIN_UNNAMED_SERIES_LABEL
        ):
            return SmallQueryCompiler(self._modin_frame.transpose())
        return self

    def is_series_like(self):
        return len(self._modin_frame.columns) == 1 or len(self._modin_frame.index) == 1

    def _write_items(df, row_numeric_index, col_numeric_index, broadcasted_items):
        if not isinstance(row_numeric_index, slice):
            row_numeric_index = list(row_numeric_index)
        if not isinstance(col_numeric_index, slice):
            col_numeric_index = list(col_numeric_index)

        if isinstance(df.iloc[row_numeric_index, col_numeric_index], pandas.Series):
            broadcasted_items = broadcasted_items.squeeze()
        df.iloc[row_numeric_index, col_numeric_index] = broadcasted_items
        return df

    def _setitem(df, axis, key, value):
        if is_scalar(key) and isinstance(value, pandas.DataFrame):
            value = value.squeeze()
        if not axis:
            df[key] = value
        else:
            df.loc[key] = value
        return df

    def _delitem(df, key):
        return df.drop(columns=[key])

    get_dummies = _register_default_pandas(_get_dummies)
    getitem_array = _register_default_pandas(_getitem_array)
    getitem_row_array = _register_default_pandas(_getitem_row_array)
    delitem = _register_default_pandas(_delitem)
    write_items = _register_default_pandas(_write_items)
    setitem = _register_default_pandas(_setitem)
