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

from modin.core.storage_formats.base.query_compiler import BaseQueryCompiler
import numpy as np
import inspect
from pandas._libs.lib import no_default, NoDefault
from pandas.api.types import is_list_like
from pandas.core.computation.parsing import tokenize_string

from typing import Any


class ClientQueryCompiler(BaseQueryCompiler):
    @classmethod
    def set_server_connection(cls, conn):
        cls._service = conn

    def __init__(self, id):
        assert (
            id is not None
        ), "Make sure the client is properly connected and returns and ID"
        if isinstance(id, Exception):
            raise id
        self._id = id

    def _set_columns(self, new_columns):
        self._id = self._service.rename(self._id, new_col_labels=new_columns)
        self._columns_cache = self._service.columns(self._id)

    def _get_columns(self):
        if self._columns_cache is None:
            self._columns_cache = self._service.columns(self._id)
        return self._columns_cache

    def _set_index(self, new_index):
        self._id = self._service.rename(self._id, new_row_labels=new_index)

    def _get_index(self):
        return self._service.index(self._id)

    columns = property(_get_columns, _set_columns)
    _columns_cache = None
    index = property(_get_index, _set_index)
    _dtypes_cache = None

    @property
    def dtypes(self):
        if self._dtypes_cache is None:
            self._dtypes_cache = self._service.dtypes(self._id)
        return self._dtypes_cache

    @classmethod
    def from_pandas(cls, df, data_cls):
        raise NotImplementedError

    def to_sql(
        self,
        name,
        con,
        schema=None,
        if_exists="fail",
        index=True,
        index_label=None,
        chunksize=None,
        dtype=None,
        method=None,
    ):
        return self._service.to_sql(
            self._id,
            name,
            con,
            schema,
            if_exists,
            index,
            index_label,
            chunksize,
            dtype,
            method,
        )

    def to_pandas(self):
        value = self._service.to_pandas(self._id)
        if isinstance(value, Exception):
            raise value
        return value

    def to_numpy(self, **kwargs):
        value = self._service.to_numpy(self._id, **kwargs)
        if isinstance(value, Exception):
            raise value
        return value

    def default_to_pandas(self, pandas_op, *args, **kwargs):
        raise NotImplementedError

    def copy(self):
        return self.__constructor__(self._id)

    def insert(self, loc, column, value):
        if isinstance(value, ClientQueryCompiler):
            value = value._id
            is_qc = True
        else:
            is_qc = False
        return self.__constructor__(
            self._service.insert(self._id, loc, column, value, is_qc)
        )

    def insert_item(self, axis, loc, value, how="inner", replace=False):
        value_is_qc = isinstance(value, ClientQueryCompiler)
        if value_is_qc:
            value = value._id
        return self._service.insert_item(
            self._id, axis, loc, value, how, replace, value_is_qc
        )

    def setitem(self, axis, key, value):
        if isinstance(value, ClientQueryCompiler):
            value = value._id
            is_qc = True
        else:
            is_qc = False
        return self.__constructor__(
            self._service.setitem(self._id, axis, key, value, is_qc)
        )

    def getitem_array(self, key):
        if isinstance(key, ClientQueryCompiler):
            key = key._id
            is_qc = True
        else:
            is_qc = False
        return self.__constructor__(self._service.getitem_array(self._id, key, is_qc))

    def replace(
        self,
        to_replace=None,
        value=no_default,
        inplace=False,
        limit=None,
        regex=False,
        method: "str | NoDefault" = no_default,
    ):
        if isinstance(to_replace, ClientQueryCompiler):
            is_to_replace_qc = True
        else:
            is_to_replace_qc = False
        if isinstance(regex, ClientQueryCompiler):
            is_regex_qc = True
        else:
            is_regex_qc = False
        return self.__constructor__(
            self._service.replace(
                self._id,
                to_replace,
                value,
                inplace,
                limit,
                regex,
                method,
                is_to_replace_qc,
                is_regex_qc,
            )
        )

    def fillna(
        self,
        squeeze_self,
        squeeze_value,
        value=None,
        method=None,
        axis=None,
        inplace=False,
        limit=None,
        downcast=None,
    ):
        if isinstance(value, ClientQueryCompiler):
            is_qc = True
        else:
            is_qc = False
        return self.__constructor__(
            self._service.fillna(
                self._id,
                squeeze_self,
                squeeze_value,
                value,
                method,
                axis,
                inplace,
                limit,
                downcast,
                is_qc,
            )
        )

    def concat(self, axis, other, **kwargs):
        if is_list_like(other):
            other = [o._id for o in other]
        else:
            other = [other._id]
        return self.__constructor__(
            self._service.concat(self._id, axis, other, **kwargs)
        )

    def sort_rows_by_column_values(self, columns, ascending=True, **kwargs):
        return self.__constructor__(
            self._service.sort_rows_by_column_values(
                self._id, columns, ascending=ascending, **kwargs
            )
        )

    def merge(self, right, **kwargs):
        return self.__constructor__(self._service.merge(self._id, right._id, **kwargs))

    def merge_asof(self, right, **kwargs):
        return self.__constructor__(
            self._service.merge_asof(self._id, right._id, **kwargs)
        )

    def groupby_mean(
        self,
        by,
        axis,
        groupby_kwargs,
        agg_args,
        agg_kwargs,
        drop=False,
    ):
        return self.__constructor__(
            self._service.groupby_mean(
                self._id, by._id, axis, groupby_kwargs, agg_args, agg_kwargs, drop
            )
        )

    def groupby_count(
        self,
        by,
        axis,
        groupby_kwargs,
        agg_args,
        agg_kwargs,
        drop=False,
    ):
        return self.__constructor__(
            self._service.groupby_count(
                self._id, by._id, axis, groupby_kwargs, agg_args, agg_kwargs, drop
            )
        )

    def groupby_max(
        self,
        by,
        axis,
        groupby_kwargs,
        agg_args,
        agg_kwargs,
        drop=False,
    ):
        return self.__constructor__(
            self._service.groupby_max(
                self._id, by._id, axis, groupby_kwargs, agg_args, agg_kwargs, drop
            )
        )

    def groupby_min(
        self,
        by,
        axis,
        groupby_kwargs,
        agg_args,
        agg_kwargs,
        drop=False,
    ):
        return self.__constructor__(
            self._service.groupby_min(
                self._id, by._id, axis, groupby_kwargs, agg_args, agg_kwargs, drop
            )
        )

    def groupby_sum(
        self,
        by,
        axis,
        groupby_kwargs,
        agg_args,
        agg_kwargs,
        drop=False,
    ):
        return self.__constructor__(
            self._service.groupby_sum(
                self._id, by._id, axis, groupby_kwargs, agg_args, agg_kwargs, drop
            )
        )

    def groupby_agg(
        self,
        by,
        agg_func,
        axis,
        groupby_kwargs,
        agg_args,
        agg_kwargs,
        how="axis_wise",
        drop=False,
    ):
        return self.__constructor__(
            self._service.groupby_agg(
                self._id,
                by._id,
                agg_func,
                axis,
                groupby_kwargs,
                agg_args,
                agg_kwargs,
                how,
                drop,
            )
        )

    def to_datetime(self, *args, **kwargs):
        return self.__constructor__(
            self._service.to_datetime(self._id, *args, **kwargs)
        )

    def finalize(self):
        raise NotImplementedError

    def free(self):
        return

    @classmethod
    def from_arrow(cls, at, data_cls):
        raise NotImplementedError

    @classmethod
    def from_dataframe(cls, df, data_cls):
        raise NotImplementedError

    def to_dataframe(self, nan_as_null: bool = False, allow_copy: bool = True):
        raise NotImplementedError

    def clip(self, lower, upper, **kwargs):
        lower_is_qc = isinstance(lower, type(self))
        upper_is_qc = isinstance(upper, type(self))
        if lower_is_qc:
            lower = lower._id

    def isin(self, values):
        # isin is unusal because it passes API layer objects to query compiler
        # instead of converting them to query compiler objects (Modin issue #3106)
        from modin.pandas import DataFrame, Series

        is_qc = isinstance(values, (DataFrame, Series))
        if is_qc:
            values = values._query_compiler._id
        return self.__constructor__(self._service.isin(self._id, values, is_qc))

    def where(self, cond, other, **kwargs):
        cond_is_qc = isinstance(cond, type(self))
        other_is_qc = isinstance(other, type(self))
        if cond_is_qc:
            cond = cond._id
        if other_is_qc:
            other = other._id
        return self.__constructor__(
            self._service.where(
                self._id, cond, cond_is_qc, other, other_is_qc, **kwargs
            )
        )

    # take_2d is special because service still uses `view`, but modin calls `take_2d`
    def take_2d(self, index=None, columns=None):
        return self.__constructor__(self._service.view(self._id, index, columns))

    # The service should define the same default of numeric=False, but it doesn't,
    # so we do it here. If we don't define the default here, the service complains
    # because it never gets the `numeric` param.
    def getitem_column_array(self, key, numeric=False):
        return self.__constructor__(
            self._service.getitem_column_array(self._id, key, numeric)
        )

    # BUG: cumulative functions are wrong in service. need special treatment here.
    # service signature is def exposed_cumsum(self, id, axis, skipna, *args, **kwargs):
    # and query compiler container signature is
    #     def cumsum(self, id, fold_axis, skipna, *args, **kwargs):
    # whereas this can take both fold_axis and axis.
    # I think we're actually passing axis=fold_axis and skipna=axis and skipna
    # as a *arg to the service.
    def cummax(self, fold_axis, axis, skipna, *args, **kwargs):
        return self.__constructor__(
            self._service.cummax(self._id, fold_axis, axis, skipna, *args, **kwargs)
        )

    def cummin(self, fold_axis, axis, skipna, *args, **kwargs):
        return self.__constructor__(
            self._service.cummin(self._id, fold_axis, axis, skipna, *args, **kwargs)
        )

    def cumsum(self, fold_axis, axis, skipna, *args, **kwargs):
        return self.__constructor__(
            self._service.cumsum(self._id, fold_axis, axis, skipna, *args, **kwargs)
        )

    def cumprod(self, fold_axis, axis, skipna, *args, **kwargs):
        return self.__constructor__(
            self._service.cumprod(self._id, fold_axis, axis, skipna, *args, **kwargs)
        )

    # Use this buggy sub which calls rsub because the service expects the bug:
    # https://github.com/ponder-org/soda/blob/5aca5483ec24b0fc0bb00a3dcab410da297598b1/pushdown_service/test/snowflake/arithmetic/test_numeric.py#L26-L37
    # need to fix the test in the service.
    def sub(self, other, **kwargs):
        if isinstance(other, ClientQueryCompiler):
            other = other._id
            is_qc = True
        else:
            is_qc = False
        return self.__constructor__(
            self._service.rsub(self._id, other, is_qc, **kwargs)
        )

    def query(self, expr, **kwargs):
        # TODO: Don't need all this; API layer passes local and global vars
        # in local_dict and global_dict. But the service is buggy
        # and dodesn't use those dicts. So wwe need to keep this for now.
        is_variable = False
        variable_list = []
        for k, v in tokenize_string(expr):
            if v == "" or v == " ":
                continue
            if is_variable:
                frame = inspect.currentframe()
                identified = False
                while frame:
                    if v in frame.f_locals:
                        value = frame.f_locals[v]
                        if isinstance(value, list):
                            value = tuple(value)
                        variable_list.append(str(value))
                        identified = True
                        break
                    frame = frame.f_back
                if not identified:
                    # TODO this error does not quite match pandas
                    raise ValueError(f"{v} not found")
                is_variable = False
            elif v == "@":
                is_variable = True
                continue
            else:
                if v in self.columns:
                    v = f"`{v}`"
                variable_list.append(v)
        expr = " ".join(variable_list)
        return self.__constructor__(self._service.query(self._id, expr, **kwargs))

    def ewm_cov(self, other=None, *args, **kwargs):
        other_is_qc = isinstance(other, type(self))
        if other_is_qc:
            other = other._id
        return self.__constructor__(
            self._service.ewm_cov(self._id, other, other_is_qc, *args, **kwargs)
        )

    def ewm_corr(self, other=None, *args, **kwargs):
        other_is_qc = isinstance(other, type(self))
        if other_is_qc:
            other = other._id
        return self.__constructor__(
            self._service.ewm_corr(self._id, other, other_is_qc, *args, **kwargs)
        )

    def expanding_cov(self, other=None, *args, **kwargs):
        other_is_qc = isinstance(other, type(self))
        if other_is_qc:
            other = other._id
        return self.__constructor__(
            self._service.expanding_cov(self._id, other, other_is_qc, *args, **kwargs)
        )

    def expanding_corr(self, other=None, *args, **kwargs):
        other_is_qc = isinstance(other, type(self))
        if other_is_qc:
            other = other._id
        return self.__constructor__(
            self._service.expanding_corr(self._id, other, other_is_qc, *args, **kwargs)
        )

    def mask(self, cond, other=np.nan, *args, **kwargs):
        cond_is_qc = isinstance(cond, type(self))
        if cond_is_qc:
            cond = cond._id
        other_is_qc = isinstance(other, type(self))
        if other_is_qc:
            other = other._id
        return self.__constructor__(
            self._service.mask(
                self._id, cond, cond_is_qc, other, other_is_qc, *args, **kwargs
            )
        )


def _set_forwarding_method_for_binary_function(method_name: str) -> None:
    """
    Define a binary method that forwards arguments to the service.
    Parameters
    ----------
    method_name : str
    """

    def forwarding_method(
        self: ClientQueryCompiler,
        other: Any,
        **kwargs,
    ):
        other_is_qc = isinstance(other, type(self))
        if other_is_qc:
            other = other._id
        return self.__constructor__(
            getattr(self._service, method_name)(self._id, other, other_is_qc, **kwargs)
        )

    setattr(ClientQueryCompiler, method_name, forwarding_method)


def _set_forwarding_method_for_single_id(method_name: str) -> None:
    """
    Define a method that forwards arguments to the service.
    Parameters
    ----------
    method_name : str
    """

    def forwarding_method(
        self: ClientQueryCompiler,
        *args,
        **kwargs,
    ):
        return self.__constructor__(
            getattr(self._service, method_name)(self._id, *args, **kwargs)
        )

    setattr(ClientQueryCompiler, method_name, forwarding_method)


def _set_forwarding_groupby_method(method_name: str):
    """
    Define a groupby method that forwards arguments to the service.
    Parameters
    ----------
    method_name : str
    """

    def forwarding_method(self, by, *args, **kwargs):
        if not isinstance(by, type(self)):
            raise NotImplementedError("Must always GroupBy another modin.pandas object")
        return self.__constructor__(
            getattr(self._service, method_name)(self._id, by._id, *args, **kwargs)
        )

    setattr(ClientQueryCompiler, method_name, forwarding_method)


_BINARY_FORWARDING_METHODS = frozenset(
    {
        "eq",
        "lt",
        "le",
        "gt",
        "ge",
        "ne",
        "__and__",
        "__or__",
        "add",
        "radd",
        "truediv",
        "rtruediv",
        "mod",
        "rmod",
        "rsub",
        "mul",
        "rmul",
        "floordiv",
        "rfloordiv",
        "__rand__",
        "__ror__",
        "__xor__",
        "__rxor__",
        "pow",
        "rpow",
        "combine",
        "combine_first",
        "compare",
        "df_update",
        "dot",
        "join",
        "series_update",
        "align",
        "series_corr",
        "divmod",
        "reindex_like",
        "rdivmod",
        "corrwith" "merge_ordered",
    }
)

_SINGLE_ID_FORWARDING_METHODS = frozenset(
    {
        "abs",
        "asfreq",
        "columnarize",
        "transpose",
        "getitem_row_array",
        "getitem_row_labels_array",
        "pivot",
        "get_dummies",
        "drop",
        "isna",
        "notna",
        "add_prefix",
        "add_suffix",
        "astype",
        "dropna",
        "sum",
        "prod",
        "count",
        "mean",
        "median",
        "std",
        "min",
        "max",
        "any",
        "all",
        "quantile_for_single_value",
        "quantile_for_list_of_values",
        "describe",
        "set_index_from_columns",
        "reset_index",
        "sort_rows_by_column_values",
        "sort_index",
        "dt_nanosecond",
        "dt_microsecond",
        "dt_second",
        "dt_minute",
        "dt_hour",
        "dt_day",
        "dt_dayofweek",
        "dt_weekday",
        "dt_day_name",
        "dt_dayofyear",
        "dt_week",
        "dt_weekofyear",
        "dt_month",
        "dt_month_name",
        "dt_quarter",
        "dt_year",
        "dt_ceil",
        "dt_components",
        "dt_date",
        "dt_days",
        "dt_days_in_month",
        "dt_daysinmonth",
        "dt_end_time",
        "dt_floor",
        "dt_freq",
        "dt_is_leap_year",
        "dt_is_month_end",
        "dt_is_month_start",
        "dt_is_quarter_end",
        "dt_is_quarter_start",
        "dt_is_year_end",
        "dt_is_year_start",
        "dt_microseconds",
        "dt_nanoseconds",
        "dt_normalize",
        "dt_qyear",
        "dt_round",
        "dt_seconds",
        "dt_start_time",
        "dt_strftime",
        "dt_time",
        "dt_timetz",
        "dt_to_period",
        "dt_to_pydatetime",
        "dt_to_pytimedelta",
        "dt_to_timestamp",
        "dt_total_seconds",
        "dt_tz",
        "dt_tz_convert",
        "dt_tz_localize",
        "str_capitalize",
        "str_isalnum",
        "str_isalpha",
        "str_isdecimal",
        "str_isdigit",
        "str_islower",
        "str_isnumeric",
        "str_isspace",
        "str_istitle",
        "str_isupper",
        "str_len",
        "str_lower",
        "str_title",
        "str_upper",
        "str_center",
        "str_contains",
        "str_count",
        "str_endswith",
        "str_find",
        "str_index",
        "str_rfind",
        "str_findall",
        "str_get",
        "str_join",
        "str_lstrip",
        "str_ljust",
        "str_rjust",
        "str_match",
        "str_pad",
        "str_repeat",
        "str_split",
        "str_rsplit",
        "str_rstrip",
        "str_slice",
        "str_slice_replace",
        "str_startswith",
        "str_strip",
        "str_zfill",
        "str_casefold",
        "str_getdummies",
        "str_extract",
        "str_extractall",
        "is_monotonic_increasing",
        "is_monotonic_decreasing",
        "idxmax",
        "idxmin",
        "apply",
        "apply_on_series",
        "applymap",
        "cat_codes",
        "convert_dtypes",
        "corr",
        "cov",
        "diff",
        "eval",
        "expanding_sum",
        "expanding_min",
        "expanding_max",
        "expanding_mean",
        "expanding_var",
        "expanding_std",
        "expanding_count",
        "expanding_sem",
        "expanding_count",
        "expanding_median",
        "expanding_var",
        "expanding_skew",
        "expanding_kurt",
        "expanding_apply",
        "expanding_aggregate",
        "expanding_quantile",
        "expanding_rank",
        "explode",
        "first_valid_index",
        "infer_objects",
        "invert",
        "kurt",
        "last_valid_index",
        "mad",
        "melt",
        "memory_usage",
        "mode",
        "negative",
        "nlargest",
        "nsmallest",
        "nunique",
        "pivot_table",
        "rank",
        "reindex",
        "repeat",
        "resample_agg_df",
        "resample_agg_ser",
        "resample_app_df",
        "resample_app_ser",
        "resample_asfreq",
        "resample_backfill",
        "resample_bfill",
        "resample_count",
        "resample_ffill",
        "resample_fillna",
        "resample_first",
        "resample_get_group",
        "resample_interpolate",
        "resample_last",
        "resample_max",
        "resample_mean",
        "resample_median",
        "resample_min",
        "resample_nearest",
        "resample_nunique",
        "resample_ohlc_df",
        "resample_ohlc_ser",
        "resample_pad",
        "resample_pipe",
        "resample_prod",
        "resample_quantile",
        "resample_sem",
        "resample_size",
        "resample_std",
        "resample_sum",
        "resample_transform",
        "resample_var",
        "rolling_aggregate",
        "rolling_apply",
        "rolling_corr",
        "rolling_count",
        "rolling_cov",
        "rolling_kurt",
        "rolling_max",
        "rolling_mean",
        "rolling_median",
        "rolling_min",
        "rolling_quantile",
        "rolling_skew",
        "rolling_std",
        "rolling_sem",
        "rolling_sum",
        "rolling_var",
        "window_mean",
        "window_std",
        "window_sum",
        "window_var",
        "round",
        "searchsorted",
        "series_view",
        "sem",
        "skew",
        "sort_columns_by_row_values",
        "stack",
        "str___getitem__",
        "str_normalize",
        "str_partition",
        "str_replace",
        "str_rindex",
        "str_rpartition",
        "str_swapcase",
        "str_translate",
        "str_wrap",
        "to_numeric",
        "unique",
        "unstack",
        "var",
        "write_items",
        "set_index_name",
        "set_index_names",
        "ewm_mean",
        "ewm_sum",
        "ewm_std",
        "ewm_var",
        "pct_change",
        "sizeof",
        "argsort",
        "between",
        "factorize",
        "dataframe_hist",
        "series_hist",
        "interpolate",
        "nlargest",
        "nsmallest",
        "swaplevel",
        "dataframe_to_dict",
        "series_to_dict",
        "to_list",
        "truncate",
        "lookup",
        "wide_to_long",
    }
)

_GROUPBY_FORWARDING_METHODS = frozenset(
    {
        "mean",
        "count",
        "max",
        "min",
        "sum",
        "agg",
        "all",
        "any",
        "size",
        "skew",
        "cumsum",
        "cummax",
        "cummin",
        "cumprod",
        "std",
        "rank",
        "nunique",
        "median",
        "quantile",
        "fillna",
        "dtypes",
        "shift",
        "prod",
        "var",
    }
)

for method in _BINARY_FORWARDING_METHODS:
    _set_forwarding_method_for_binary_function(method)

for method in _SINGLE_ID_FORWARDING_METHODS:
    _set_forwarding_method_for_single_id(method)

for method in _GROUPBY_FORWARDING_METHODS:
    _set_forwarding_groupby_method("groupby_" + method)

ClientQueryCompiler.prod_min_count = ClientQueryCompiler.prod
ClientQueryCompiler.sum_min_count = ClientQueryCompiler.sum
