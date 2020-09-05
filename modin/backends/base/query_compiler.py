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
import abc

from modin.data_management.functions.default_methods import (
    DataFrameDefault,
    SeriesDefault,
    DateTimeDefault,
    StrDefault,
    BinaryDefault,
    AnyDefault,
    ResampleDefault,
    RollingDefault,
    CatDefault,
    GroupByDefault,
)

from pandas.core.dtypes.common import is_scalar
import pandas
import numpy as np


def _get_axis(axis):
    def axis_getter(self):
        return self.to_pandas().axes[axis]

    return axis_getter


def _set_axis(axis):
    def axis_setter(self, labels):
        new_qc = DataFrameDefault.register("set_axis")(self, axis=axis, labels=labels)
        self.__dict__.update(new_qc.__dict__)

    return axis_setter


class BaseQueryCompiler(abc.ABC):
    """Abstract Class that handles the queries to Modin dataframes.

    Note: See the Abstract Methods and Fields section immediately below this
        for a list of requirements for subclassing this object.
    """

    @property
    def __constructor__(self):
        """By default, constructor method will invoke an init."""
        return type(self)

    @abc.abstractmethod
    def default_to_pandas(self, pandas_op, *args, **kwargs):
        """Default to pandas behavior.

        Parameters
        ----------
        pandas_op : callable
            The operation to apply, must be compatible pandas DataFrame call
        args
            The arguments for the `pandas_op`
        kwargs
            The keyword arguments for the `pandas_op`

        Returns
        -------
        BaseQueryCompiler
            The result of the `pandas_op`, converted back to BaseQueryCompiler
        """
        pass

    # Abstract Methods and Fields: Must implement in children classes
    # In some cases, there you may be able to use the same implementation for
    # some of these abstract methods, but for the sake of generality they are
    # treated differently.

    lazy_execution = False

    # Data Management Methods
    @abc.abstractmethod
    def free(self):
        """In the future, this will hopefully trigger a cleanup of this object."""
        # TODO create a way to clean up this object.
        pass

    # END Data Management Methods

    # To/From Pandas
    @abc.abstractmethod
    def to_pandas(self):
        """Converts Modin DataFrame to Pandas DataFrame.

        Returns:
            Pandas DataFrame of the QueryCompiler.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def from_pandas(cls, df, data_cls):
        """Improve simple Pandas DataFrame to an advanced and superior Modin DataFrame.

        Parameters
        ----------
        df: pandas.DataFrame
            The pandas DataFrame to convert from.
        data_cls :
            Modin DataFrame object to convert to.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler containing data from the Pandas DataFrame.
        """
        pass

    # END To/From Pandas

    # From Arrow
    @classmethod
    @abc.abstractmethod
    def from_arrow(cls, at, data_cls):
        """Improve simple Arrow Table to an advanced and superior Modin DataFrame.

        Parameters
        ----------
        at : Arrow Table
            The Arrow Table to convert from.
        data_cls :
            Modin DataFrame object to convert to.

        Returns
        -------
        BaseQueryCompiler
            QueryCompiler containing data from the Pandas DataFrame.
        """
        pass

    # END From Arrow

    index = property(_get_axis(0), _set_axis(0))
    columns = property(_get_axis(1), _set_axis(1))

    # DataFrame methods

    def drop(self, index=None, columns=None):
        if index is None and columns is None:
            return self
        else:
            return DataFrameDefault.register("drop")(self, index=index, columns=columns)

    def nlargest(self, n=5, columns=None, keep="first"):
        if columns is None:
            return SeriesDefault.register("nlargest")(self, n=n, keep=keep)
        else:
            return DataFrameDefault.register("nlargest")(
                self, n=n, columns=columns, keep=keep
            )

    def nsmallest(self, n=5, columns=None, keep="first"):
        if columns is None:
            return SeriesDefault.register("nsmallest")(self, n=n, keep=keep)
        else:
            return DataFrameDefault.register("nsmallest")(
                self, n=n, columns=columns, keep=keep
            )

    def last_valid_index(self):
        return AnyDefault.register("last_valid_index")(self).to_pandas().squeeze()

    def first_valid_index(self):
        return AnyDefault.register("first_valid_index")(self).to_pandas().squeeze()

    def add_suffix(self, suffix=None, axis=1):
        if axis:
            return DataFrameDefault.register("add_suffix")(self, suffix=suffix)
        else:
            return SeriesDefault.register("add_suffix")(self, suffix=suffix)

    def add_prefix(self, prefix=None, axis=1):
        if axis:
            return DataFrameDefault.register("add_prefix")(self, prefix=prefix)
        else:
            return SeriesDefault.register("add_prefix")(self, prefix=prefix)

    def concat(self, axis, other, **kwargs):
        concat_join = ["inner", "outer"]

        def concat(df, axis, other, **kwargs):
            kwargs.pop("join_axes", None)
            ignore_index = kwargs.get("ignore_index", False)
            if kwargs.get("join", "outer") in concat_join:
                if not isinstance(other, list):
                    other = [other]
                other = [df] + other
                result = pandas.concat(other, axis=axis, **kwargs)
            else:
                if isinstance(other, (list, np.ndarray)) and len(other) == 1:
                    other = other[0]
                how = kwargs.pop("join", None)
                ignore_index = kwargs.pop("ignore_index", None)
                kwargs["how"] = how
                result = df.join(other, **kwargs)
            if ignore_index:
                if axis == 0:
                    result = result.reset_index(drop=True)
                else:
                    result.columns = pandas.RangeIndex(len(result.columns))
            return result

        return DataFrameDefault.register(concat)(self, axis=axis, other=other, **kwargs)

    @property
    def dtypes(self):
        return self.to_pandas().dtypes

    def view(self, index=None, columns=None):
        index = [] if index is None else index
        columns = [] if columns is None else columns

        def applyier(df):
            return df.iloc[index, columns]

        return DataFrameDefault.register(applyier)(self)

    def sort_columns_by_row_values(self, rows, ascending=True, **kwargs):
        return DataFrameDefault.register("sort_values")(
            self, by=rows, axis=1, ascending=ascending, **kwargs
        )

    def sort_rows_by_column_values(self, rows, ascending=True, **kwargs):
        return DataFrameDefault.register("sort_values")(
            self, by=rows, axis=0, ascending=ascending, **kwargs
        )

    abs = DataFrameDefault.register("abs")
    all = DataFrameDefault.register("all")
    any = DataFrameDefault.register("any")
    apply = DataFrameDefault.register("apply")
    applymap = DataFrameDefault.register("applymap")
    astype = DataFrameDefault.register("astype")
    clip = DataFrameDefault.register("clip")
    copy = DataFrameDefault.register("copy")
    count = DataFrameDefault.register("count")
    cummax = DataFrameDefault.register("cummax")
    cummin = DataFrameDefault.register("cummin")
    cumprod = DataFrameDefault.register("cumprod")
    cumsum = DataFrameDefault.register("cumsum")
    describe = DataFrameDefault.register("describe")
    diff = DataFrameDefault.register("diff")
    dropna = DataFrameDefault.register("dropna")
    eval = DataFrameDefault.register("eval")
    fillna = DataFrameDefault.register("fillna")
    idxmax = DataFrameDefault.register("idxmax")
    idxmin = DataFrameDefault.register("idxmin")
    insert = DataFrameDefault.register("insert", inplace=True)
    isin = DataFrameDefault.register("isin")
    isna = DataFrameDefault.register("isna")
    join = DataFrameDefault.register("join")
    kurt = DataFrameDefault.register("kurt")
    mad = DataFrameDefault.register("mad")
    max = DataFrameDefault.register("max")
    mean = DataFrameDefault.register("mean")
    median = DataFrameDefault.register("median")
    melt = DataFrameDefault.register("melt")
    memory_usage = DataFrameDefault.register("memory_usage")
    merge = DataFrameDefault.register("merge")
    min = DataFrameDefault.register("min")
    mode = DataFrameDefault.register("mode")
    notna = DataFrameDefault.register("notna")
    nunique = DataFrameDefault.register("nunique")
    pivot = DataFrameDefault.register("pivot")
    prod = DataFrameDefault.register("prod")
    query = DataFrameDefault.register("query")
    rank = DataFrameDefault.register("rank")
    reindex = DataFrameDefault.register("reindex")
    replace = DataFrameDefault.register("replace")
    reset_index = DataFrameDefault.register("reset_index")
    round = DataFrameDefault.register("round")
    skew = DataFrameDefault.register("skew")
    sort_index = DataFrameDefault.register("sort_index")
    stack = DataFrameDefault.register("stack")
    std = DataFrameDefault.register("std")
    sum = DataFrameDefault.register("sum")
    to_numpy = DataFrameDefault.register("to_numpy")
    transpose = DataFrameDefault.register("transpose")
    unstack = DataFrameDefault.register("unstack")
    var = DataFrameDefault.register("var")
    where = DataFrameDefault.register("where")

    sum_min_count = DataFrameDefault.register("sum")
    prod_min_count = DataFrameDefault.register("prod")
    quantile_for_single_value = DataFrameDefault.register("quantile")
    quantile_for_list_of_values = quantile_for_single_value

    def columnarize(self):
        """
        Transposes this QueryCompiler if it has a single row but multiple columns.

        This method should be called for QueryCompilers representing a Series object,
        i.e. self.is_series_like() should be True.

        Returns
        -------
        PandasQueryCompiler
            Transposed new QueryCompiler or self.
        """
        if len(self.columns) != 1 or (
            len(self.index) == 1 and self.index[0] == "__reduced__"
        ):
            return self.transpose()
        return self

    def is_series_like(self):
        """Return True if QueryCompiler has a single column or row"""
        return len(self.columns) == 1 or len(self.index) == 1

    def has_multiindex(self, axis=0):
        """
        Check if specified axis is indexed by MultiIndex.

        Parameters
        ----------
        axis : 0 or 1, default 0
            The axis to check (0 - index, 1 - columns).

        Returns
        -------
        bool
            True if index at specified axis is MultiIndex and False otherwise.
        """
        if axis == 0:
            return isinstance(self.index, pandas.MultiIndex)
        assert axis == 1
        return isinstance(self.columns, pandas.MultiIndex)

    # End of DataFrame methods

    # Series methods

    is_monotonic = AnyDefault.register("is_monotonic")
    is_monotonic_decreasing = AnyDefault.register("is_monotonic_decreasing")
    repeat = AnyDefault.register("repeat")
    searchsorted = AnyDefault.register("searchsorted")
    unique = AnyDefault.register("unique")
    value_counts = AnyDefault.register("value_counts")

    # End of Series methods

    # GroupBy methods

    groupby_agg = GroupByDefault.register("groupby_agg")
    groupby_all = GroupByDefault.register("groupby_all")
    groupby_any = GroupByDefault.register("groupby_any")
    groupby_count = GroupByDefault.register("groupby_count")
    groupby_dict_agg = GroupByDefault.register("groupby_dict_agg")
    groupby_max = GroupByDefault.register("groupby_max")
    groupby_min = GroupByDefault.register("groupby_min")
    groupby_prod = GroupByDefault.register("groupby_prod")
    groupby_size = GroupByDefault.register("groupby_size")
    groupby_sum = GroupByDefault.register("groupby_sum")

    # End of GroupBy methods

    # DateTime methods

    dt_ceil = DateTimeDefault.register("dt_ceil")
    dt_components = DateTimeDefault.register("dt_components")
    dt_date = DateTimeDefault.register("dt_date")
    dt_day = DateTimeDefault.register("dt_day")
    dt_day_name = DateTimeDefault.register("dt_day_name")
    dt_dayofweek = DateTimeDefault.register("dt_dayofweek")
    dt_dayofyear = DateTimeDefault.register("dt_dayofyear")
    dt_days = DateTimeDefault.register("dt_days")
    dt_days_in_month = DateTimeDefault.register("dt_days_in_month")
    dt_daysinmonth = DateTimeDefault.register("dt_daysinmonth")
    dt_end_time = DateTimeDefault.register("dt_end_time")
    dt_floor = DateTimeDefault.register("dt_floor")
    dt_freq = DateTimeDefault.register("dt_freq")
    dt_hour = DateTimeDefault.register("dt_hour")
    dt_is_leap_year = DateTimeDefault.register("dt_is_leap_year")
    dt_is_month_end = DateTimeDefault.register("dt_is_month_end")
    dt_is_month_start = DateTimeDefault.register("dt_is_month_start")
    dt_is_quarter_end = DateTimeDefault.register("dt_is_quarter_end")
    dt_is_quarter_start = DateTimeDefault.register("dt_is_quarter_start")
    dt_is_year_end = DateTimeDefault.register("dt_is_year_end")
    dt_is_year_start = DateTimeDefault.register("dt_is_year_start")
    dt_microsecond = DateTimeDefault.register("dt_microsecond")
    dt_microseconds = DateTimeDefault.register("dt_microseconds")
    dt_minute = DateTimeDefault.register("dt_minute")
    dt_month = DateTimeDefault.register("dt_month")
    dt_month_name = DateTimeDefault.register("dt_month_name")
    dt_nanosecond = DateTimeDefault.register("dt_nanosecond")
    dt_nanoseconds = DateTimeDefault.register("dt_nanoseconds")
    dt_normalize = DateTimeDefault.register("dt_normalize")
    dt_quarter = DateTimeDefault.register("dt_quarter")
    dt_qyear = DateTimeDefault.register("dt_qyear")
    dt_round = DateTimeDefault.register("dt_round")
    dt_second = DateTimeDefault.register("dt_second")
    dt_seconds = DateTimeDefault.register("dt_seconds")
    dt_start_time = DateTimeDefault.register("dt_start_time")
    dt_strftime = DateTimeDefault.register("dt_strftime")
    dt_time = DateTimeDefault.register("dt_time")
    dt_timetz = DateTimeDefault.register("dt_timetz")
    dt_to_period = DateTimeDefault.register("dt_to_period")
    dt_to_pydatetime = DateTimeDefault.register("dt_to_pydatetime")
    dt_to_pytimedelta = DateTimeDefault.register("dt_to_pytimedelta")
    dt_to_timestamp = DateTimeDefault.register("dt_to_timestamp")
    dt_total_seconds = DateTimeDefault.register("dt_total_seconds")
    dt_tz = DateTimeDefault.register("dt_tz")
    dt_tz_convert = DateTimeDefault.register("dt_tz_convert")
    dt_tz_localize = DateTimeDefault.register("dt_tz_localize")
    dt_week = DateTimeDefault.register("dt_week")
    dt_weekday = DateTimeDefault.register("dt_weekday")
    dt_weekofyear = DateTimeDefault.register("dt_weekofyear")
    dt_year = DateTimeDefault.register("dt_year")

    # End of DateTime methods

    # Resample methods

    resample_agg_df = ResampleDefault.register("resample_agg_df")
    resample_agg_ser = ResampleDefault.register("resample_agg_ser")
    resample_app_df = ResampleDefault.register("resample_app_df")
    resample_app_ser = ResampleDefault.register("resample_app_ser")
    resample_asfreq = ResampleDefault.register("resample_asfreq")
    resample_backfill = ResampleDefault.register("resample_backfill")
    resample_bfill = ResampleDefault.register("resample_bfill")
    resample_count = ResampleDefault.register("resample_count")
    resample_ffill = ResampleDefault.register("resample_ffill")
    resample_fillna = ResampleDefault.register("resample_fillna")
    resample_first = ResampleDefault.register("resample_first")
    resample_get_group = ResampleDefault.register("resample_get_group")
    resample_interpolate = ResampleDefault.register("resample_interpolate")
    resample_last = ResampleDefault.register("resample_last")
    resample_max = ResampleDefault.register("resample_max")
    resample_mean = ResampleDefault.register("resample_mean")
    resample_median = ResampleDefault.register("resample_median")
    resample_min = ResampleDefault.register("resample_min")
    resample_nearest = ResampleDefault.register("resample_nearest")
    resample_nunique = ResampleDefault.register("resample_nunique")
    resample_ohlc_df = ResampleDefault.register("resample_ohlc_df")
    resample_ohlc_ser = ResampleDefault.register("resample_ohlc_ser")
    resample_pad = ResampleDefault.register("resample_pad")
    resample_pipe = ResampleDefault.register("resample_pipe")
    resample_prod = ResampleDefault.register("resample_prod")
    resample_quantile = ResampleDefault.register("resample_quantile")
    resample_sem = ResampleDefault.register("resample_sem")
    resample_size = ResampleDefault.register("resample_size")
    resample_std = ResampleDefault.register("resample_std")
    resample_sum = ResampleDefault.register("resample_sum")
    resample_transform = ResampleDefault.register("resample_transform")
    resample_var = ResampleDefault.register("resample_var")

    # End of Resample methods

    # Str methods

    str_capitalize = StrDefault.register("str_capitalize")
    str_center = StrDefault.register("str_center")
    str_contains = StrDefault.register("str_contains")
    str_count = StrDefault.register("str_count")
    str_endswith = StrDefault.register("str_endswith")
    str_find = StrDefault.register("str_find")
    str_findall = StrDefault.register("str_findall")
    str_get = StrDefault.register("str_get")
    str_index = StrDefault.register("str_index")
    str_isalnum = StrDefault.register("str_isalnum")
    str_isalpha = StrDefault.register("str_isalpha")
    str_isdecimal = StrDefault.register("str_isdecimal")
    str_isdigit = StrDefault.register("str_isdigit")
    str_islower = StrDefault.register("str_islower")
    str_isnumeric = StrDefault.register("str_isnumeric")
    str_isspace = StrDefault.register("str_isspace")
    str_istitle = StrDefault.register("str_istitle")
    str_isupper = StrDefault.register("str_isupper")
    str_join = StrDefault.register("str_join")
    str_len = StrDefault.register("str_len")
    str_ljust = StrDefault.register("str_ljust")
    str_lower = StrDefault.register("str_lower")
    str_lstrip = StrDefault.register("str_lstrip")
    str_match = StrDefault.register("str_match")
    str_normalize = StrDefault.register("str_normalize")
    str_pad = StrDefault.register("str_pad")
    str_partition = StrDefault.register("str_partition")
    str_repeat = StrDefault.register("str_repeat")
    str_replace = StrDefault.register("str_replace")
    str_rfind = StrDefault.register("str_rfind")
    str_rindex = StrDefault.register("str_rindex")
    str_rjust = StrDefault.register("str_rjust")
    str_rpartition = StrDefault.register("str_rpartition")
    str_rsplit = StrDefault.register("str_rsplit")
    str_rstrip = StrDefault.register("str_rstrip")
    str_slice = StrDefault.register("str_slice")
    str_slice_replace = StrDefault.register("str_slice_replace")
    str_split = StrDefault.register("str_split")
    str_startswith = StrDefault.register("str_startswith")
    str_strip = StrDefault.register("str_strip")
    str_swapcase = StrDefault.register("str_swapcase")
    str_title = StrDefault.register("str_title")
    str_translate = StrDefault.register("str_translate")
    str_upper = StrDefault.register("str_upper")
    str_wrap = StrDefault.register("str_wrap")
    str_zfill = StrDefault.register("str_zfill")

    # End of Str methods

    # Rolling methods

    rolling_aggregate = RollingDefault.register("rolling_aggregate")
    rolling_apply = RollingDefault.register("rolling_apply")
    rolling_corr = RollingDefault.register("rolling_corr")
    rolling_count = RollingDefault.register("rolling_count")
    rolling_cov = RollingDefault.register("rolling_cov")
    rolling_kurt = RollingDefault.register("rolling_kurt")
    rolling_max = RollingDefault.register("rolling_max")
    rolling_mean = RollingDefault.register("rolling_mean")
    rolling_median = RollingDefault.register("rolling_median")
    rolling_min = RollingDefault.register("rolling_min")
    rolling_quantile = RollingDefault.register("rolling_quantile")
    rolling_skew = RollingDefault.register("rolling_skew")
    rolling_std = RollingDefault.register("rolling_std")
    rolling_sum = RollingDefault.register("rolling_sum")
    rolling_var = RollingDefault.register("rolling_var")

    # End of Rolling methods

    # Window methods

    window_mean = RollingDefault.register("window_mean")
    window_std = RollingDefault.register("window_std")
    window_sum = RollingDefault.register("window_sum")
    window_var = RollingDefault.register("window_var")

    # End of Window methods

    # Binary Operations methods

    __and__ = AnyDefault.register("__and__")
    __or__ = AnyDefault.register("__or__")
    __rand__ = AnyDefault.register("__rand__")
    __ror__ = AnyDefault.register("__ror__")
    __rxor__ = AnyDefault.register("__rxor__")
    __xor__ = AnyDefault.register("__xor__")
    add = AnyDefault.register("add")
    combine = AnyDefault.register("combine")
    combine_first = AnyDefault.register("combine_first")
    dot = AnyDefault.register("dot")
    eq = AnyDefault.register("eq")
    floordiv = AnyDefault.register("floordiv")
    ge = AnyDefault.register("ge")
    gt = AnyDefault.register("gt")
    le = AnyDefault.register("le")
    lt = AnyDefault.register("lt")
    mod = AnyDefault.register("mod")
    mul = AnyDefault.register("mul")
    ne = AnyDefault.register("ne")
    pow = AnyDefault.register("pow")
    rfloordiv = AnyDefault.register("rfloordiv")
    rmod = AnyDefault.register("rmod")
    rpow = AnyDefault.register("rpow")
    rsub = AnyDefault.register("rsub")
    rtruediv = AnyDefault.register("rtruediv")
    sub = AnyDefault.register("sub")
    truediv = AnyDefault.register("truediv")

    # End of Binary Operations methods

    # General methods

    def get_dummies(self, columns, **kwargs):
        def get_dummies(df, columns, **kwargs):
            return pandas.get_dummies(df, columns=columns, **kwargs)

        return DataFrameDefault.register(get_dummies)(self, columns=columns, **kwargs)

    def conj(self, *args, **kwargs):
        def conj(df, *args, **kwargs):
            return pandas.DataFrame(np.conj(df))

        return DataFrameDefault.register(conj)(self, *args, **kwargs)

    df_update = BinaryDefault.register("update", inplace=True)

    def series_update(*args, **kwargs):
        return BinaryDefault.register("update", inplace=True)(
            *args, squeeze_self=True, squeeze_other=True, **kwargs
        )

    series_view = SeriesDefault.register("view")

    def getitem_row_array(self, key):
        def get_row(df, key):
            return df.iloc[key]

        return DataFrameDefault.register(get_row)(self, key=key)

    def getitem_column_array(self, key, numeric=False):
        def get_column(df, key):
            if numeric:
                return df.iloc[:, key]
            else:
                return df[key]

        return DataFrameDefault.register(get_column)(self, key=key)

    invert = DataFrameDefault.register("__invert__")
    negative = DataFrameDefault.register("__neg__")
    to_numeric = SeriesDefault.register("to_numeric", obj_type=pandas)
    to_datetime = SeriesDefault.register("to_datetime", obj_type=pandas)

    def setitem(self, axis, key, value):
        def setitem(df, axis, key, value):
            if is_scalar(key) and isinstance(value, pandas.DataFrame):
                value = value.squeeze()
            if not axis:
                df[key] = value
            else:
                df.loc[key] = value
            return df

        return DataFrameDefault.register(setitem)(self, axis=axis, key=key, value=value)

    def write_items(self, row_numeric_index, col_numeric_index, broadcasted_items):
        def write_items(df, broadcasted_items):
            if isinstance(df.iloc[row_numeric_index, col_numeric_index], pandas.Series):
                broadcasted_items = broadcasted_items.squeeze()
            df.iloc[
                list(row_numeric_index), list(col_numeric_index)
            ] = broadcasted_items
            return df

        return DataFrameDefault.register(write_items)(
            self, broadcasted_items=broadcasted_items
        )

    # End of General methods

    # Categories methods

    cat_codes = CatDefault.register("cat_codes")

    # End of Categories methods

    # __delitem__
    # This will change the shape of the resulting data.
    def delitem(self, key):
        return self.drop(columns=[key])

    # END __delitem__
