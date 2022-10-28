import numpy as np
import pickle
from typing import Any, NamedTuple, Optional
from uuid import UUID, uuid4
from modin.core.io.io import BaseIO

from modin.core.storage_formats.base.query_compiler import BaseQueryCompiler


class ForwardingQueryCompilerService:
    def __init__(self, query_compiler_type: BaseQueryCompiler, io_type: BaseIO):
        self._qc = {}
        self._qc_type = query_compiler_type
        self._io_type = io_type

    def _generate_id(self) -> UUID:
        id = uuid4()
        while id in self._qc:
            id = uuid4()
        return id

    def add_query_compiler(self, qc) -> UUID:
        id = self._generate_id()
        self._qc[id] = qc
        return id

    def to_pandas(self, id):
        return self._qc[id].to_pandas()

    class DefaultToPandasResult(NamedTuple):
        result: Optional[Any]
        result_is_qc_id: bool

    def default_to_pandas(
        self, id: UUID, pandas_op, *args, **kwargs
    ) -> DefaultToPandasResult:
        result = self._qc[id].default_to_pandas(pandas_op, *args, **kwargs)
        result_is_qc_id = isinstance(result, self._qc_type)
        if result_is_qc_id:
            new_id = self._generate_id()
            self._qc[new_id] = result
            result = new_id
        return self.DefaultToPandasResult(
            result=result, result_is_qc_id=result_is_qc_id
        )

    def rename(self, id, new_col_labels=None, new_row_labels=None):
        new_id = self._generate_id()
        new_qc = self._qc[new_id] = self._qc[id].copy()
        if new_col_labels is not None:
            new_qc.columns = new_col_labels
        if new_row_labels is not None:
            new_qc.index = new_row_labels
        return new_id

    def columns(self, id):
        return self._qc[id].columns

    def index(self, id):
        return self._qc[id].index

    def dtypes(self, id):
        return self._qc[id].dtypes

    def insert(self, id, loc, column, value, is_qc):
        if is_qc:
            value = self._qc[value]
        new_id = self._generate_id()
        self._qc[new_id] = self._qc[id].insert(loc, column, value)
        return new_id

    def setitem(self, id, axis, key, value, is_qc):
        if is_qc:
            value = self._qc[value]
        new_id = self._generate_id()
        self._qc[new_id] = self._qc[id].setitem(axis, key, value)
        return new_id

    def getitem_array(self, id, key, is_qc):
        if is_qc:
            key = self._qc[key]
        new_id = self._generate_id()
        self._qc[new_id] = self._qc[id].getitem_array(key)
        return new_id

    def replace(
        self,
        id,
        to_replace,
        value,
        inplace,
        limit,
        regex,
        method,
        is_to_replace_qc,
        is_regex_qc,
    ):
        if is_to_replace_qc:
            to_replace = self._qc[to_replace]
        if is_regex_qc:
            regex = self._qc[regex]
        new_id = self._generate_id()
        # TODO(GH#3108): Use positional arguments instead of keyword arguments
        # in the query compilers so we don't have to name all the arguments
        # here.
        self._qc[new_id] = self._qc[id].replace(
            to_replace=to_replace,
            value=value,
            inplace=inplace,
            limit=limit,
            regex=regex,
            method=method,
        )
        return new_id

    def fillna(
        self,
        id,
        squeeze_self,
        squeeze_value,
        value,
        method,
        axis,
        inplace,
        limit,
        downcast,
        is_qc,
    ):
        if is_qc:
            value = self._qc[value]
        new_id = self._generate_id()
        # TODO(GH#3108): Use positional arguments instead of keyword arguments
        # in the query compilers so we don't have to name all the
        # arguments here.
        self._qc[new_id] = self._qc[id].fillna(
            squeeze_self=squeeze_self,
            squeeze_value=squeeze_value,
            value=value,
            method=method,
            axis=axis,
            inplace=inplace,
            limit=limit,
            downcast=downcast,
        )
        return new_id

    def concat(self, id, axis, other, **kwargs):
        # convert id to query compiler
        other = [self._qc[o] for o in other]
        new_id = self._generate_id()
        self._qc[new_id] = self._qc[id].concat(axis, other, **kwargs)
        return new_id

    def eq(self, id, other, is_qc, **kwargs):
        if is_qc:
            other = self._qc[other]
        new_id = self._generate_id()
        self._qc[new_id] = self._qc[id].eq(other, **kwargs)
        return new_id

    def lt(self, id, other, is_qc, **kwargs):
        if is_qc:
            other = self._qc[other]
        new_id = self._generate_id()
        self._qc[new_id] = self._qc[id].lt(other, **kwargs)
        return new_id

    def le(self, id, other, is_qc, **kwargs):
        if is_qc:
            other = self._qc[other]
        new_id = self._generate_id()
        self._qc[new_id] = self._qc[id].le(other, **kwargs)
        return new_id

    def gt(self, id, other, is_qc, **kwargs):
        if is_qc:
            other = self._qc[other]
        new_id = self._generate_id()
        self._qc[new_id] = self._qc[id].gt(other, **kwargs)
        return new_id

    def ge(self, id, other, is_qc, **kwargs):
        if is_qc:
            other = self._qc[other]
        new_id = self._generate_id()
        self._qc[new_id] = self._qc[id].ge(other, **kwargs)
        return new_id

    def ne(self, id, other, is_qc, **kwargs):
        if is_qc:
            other = self._qc[other]
        new_id = self._generate_id()
        self._qc[new_id] = self._qc[id].ne(other, **kwargs)
        return new_id

    def __and__(self, id, other, is_qc, **kwargs):
        if is_qc:
            other = self._qc[other]
        new_id = self._generate_id()
        self._qc[new_id] = self._qc[id].__and__(other, **kwargs)
        return new_id

    def __or__(self, id, other, is_qc, **kwargs):
        if is_qc:
            other = self._qc[other]
        new_id = self._generate_id()
        self._qc[new_id] = self._qc[id].__or__(other, **kwargs)
        return new_id

    def add(self, id, other, is_qc, **kwargs):
        if is_qc:
            other = self._qc[other]
        new_id = self._generate_id()
        self._qc[new_id] = self._qc[id].add(other, **kwargs)
        return new_id

    def radd(self, id, other, is_qc, **kwargs):
        if is_qc:
            other = self._qc[other]
        new_id = self._generate_id()
        self._qc[new_id] = self._qc[id].radd(other, **kwargs)
        return new_id

    def truediv(self, id, other, is_qc, **kwargs):
        if is_qc:
            other = self._qc[other]
        new_id = self._generate_id()
        self._qc[new_id] = self._qc[id].truediv(other, **kwargs)
        return new_id

    def rtruediv(self, id, other, is_qc, **kwargs):
        if is_qc:
            other = self._qc[other]
        new_id = self._generate_id()
        self._qc[new_id] = self._qc[id].rtruediv(other, **kwargs)
        return new_id

    def mod(self, id, other, is_qc, **kwargs):
        if is_qc:
            other = self._qc[other]
        new_id = self._generate_id()
        self._qc[new_id] = self._qc[id].mod(other, **kwargs)
        return new_id

    def rmod(self, id, other, is_qc, **kwargs):
        if is_qc:
            other = self._qc[other]
        new_id = self._generate_id()
        self._qc[new_id] = self._qc[id].rmod(other, **kwargs)
        return new_id

    def sub(self, id, other, is_qc, **kwargs):
        if is_qc:
            other = self._qc[other]
        new_id = self._generate_id()
        self._qc[new_id] = self._qc[id].sub(other, **kwargs)
        return new_id

    def rsub(self, id, other, is_qc, **kwargs):
        if is_qc:
            other = self._qc[other]
        new_id = self._generate_id()
        self._qc[new_id] = self._qc[id].rsub(other, **kwargs)
        return new_id

    def mul(self, id, other, is_qc, **kwargs):
        if is_qc:
            other = self._qc[other]
        new_id = self._generate_id()
        self._qc[new_id] = self._qc[id].mul(other, **kwargs)
        return new_id

    def rmul(self, id, other, is_qc, **kwargs):
        if is_qc:
            other = self._qc[other]
        new_id = self._generate_id()
        self._qc[new_id] = self._qc[id].rmul(other, **kwargs)
        return new_id

    def floordiv(self, id, other, is_qc, **kwargs):
        if is_qc:
            other = self._qc[other]
        new_id = self._generate_id()
        self._qc[new_id] = self._qc[id].floordiv(other, **kwargs)
        return new_id

    def rfloordiv(self, id, other, is_qc, **kwargs):
        if is_qc:
            other = self._qc[other]
        new_id = self._generate_id()
        self._qc[new_id] = self._qc[id].rfloordiv(other, **kwargs)
        return new_id

    def merge(self, id, right, **kwargs):
        new_id = self._generate_id()
        self._qc[new_id] = self._qc[id].merge(self._qc[right], **kwargs)
        return new_id

    def groupby_mean(
        self,
        id,
        by,
        axis,
        groupby_kwargs,
        agg_args,
        agg_kwargs,
        drop=False,
        is_qc: bool = False,
    ):
        if is_qc:
            by = self._qc[by]
        new_id = self._generate_id()
        self._qc[new_id] = self._qc[id].groupby_mean(
            by, axis, groupby_kwargs, agg_args, agg_kwargs, drop
        )
        return new_id

    def groupby_count(
        self,
        id,
        by,
        axis,
        groupby_kwargs,
        agg_args,
        agg_kwargs,
        drop=False,
        is_qc: bool = False,
    ):
        if is_qc:
            by = self._qc[by]
        new_id = self._generate_id()
        self._qc[new_id] = self._qc[id].groupby_count(
            by, axis, groupby_kwargs, agg_args, agg_kwargs, drop
        )
        return new_id

    def groupby_max(
        self,
        id,
        by,
        axis,
        groupby_kwargs,
        agg_args,
        agg_kwargs,
        drop=False,
        is_qc: bool = False,
    ):
        if is_qc:
            by = self._qc[by]
        new_id = self._generate_id()
        self._qc[new_id] = self._qc[id].groupby_max(
            by, axis, groupby_kwargs, agg_args, agg_kwargs, drop
        )
        return new_id

    def groupby_min(
        self,
        id,
        by,
        axis,
        groupby_kwargs,
        agg_args,
        agg_kwargs,
        drop=False,
        is_qc: bool = False,
    ):
        if is_qc:
            by = self._qc[by]
        new_id = self._generate_id()
        self._qc[new_id] = self._qc[id].groupby_min(
            by, axis, groupby_kwargs, agg_args, agg_kwargs, drop
        )
        return new_id

    def groupby_sum(
        self,
        id,
        by,
        axis,
        groupby_kwargs,
        agg_args,
        agg_kwargs,
        drop=False,
        is_qc: bool = False,
    ):
        if is_qc:
            by = self._qc[by]
        new_id = self._generate_id()
        self._qc[new_id] = self._qc[id].groupby_sum(
            by, axis, groupby_kwargs, agg_args, agg_kwargs, drop
        )
        return new_id

    def groupby_agg(
        self,
        id,
        by,
        agg_func,
        axis,
        groupby_kwargs,
        agg_args,
        agg_kwargs,
        how="axis_wise",
        drop=False,
        is_qc: bool = False,
    ):
        if is_qc:
            by = self._qc[by]
        new_id = self._generate_id()
        self._qc[new_id] = self._qc[id].groupby_agg(
            by, agg_func, axis, groupby_kwargs, agg_args, agg_kwargs, how, drop
        )
        return new_id

    def read_csv(self, connection, filepath, **kwargs) -> UUID:
        io_result = self._io_type._read_csv(filepath, **kwargs)
        if isinstance(io_result, self._qc_type):
            new_id = self._generate_id()
            self._qc[new_id] = io_result
            return new_id
        return io_result

    def read_sql(self, sql, connection, **kwargs) -> UUID:
        new_id = self._generate_id()
        self._qc[new_id] = self._io_type._read_sql(sql, connection, **kwargs)
        return new_id

    def to_sql(
        self,
        id,
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
        self._io_type.to_sql(
            self._qc[id],
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


def _set_forwarding_method_for_single_id(method_name: str):
    def forwarding_method(
        self: "ForwardingQueryCompilerService", id: UUID, *args, **kwargs
    ):
        new_id = self._generate_id()
        self._qc[new_id] = getattr(self._qc[id], method_name)(*args, **kwargs)
        return new_id

    setattr(ForwardingQueryCompilerService, method_name, forwarding_method)


_SINGLE_ID_FORWARDING_METHODS = frozenset(
    {
        "columnarize",
        "transpose",
        "take_2d",
        "getitem_column_array",
        "getitem_row_array",
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
        "cummax",
        "cummin",
        "cumsum",
        "cumprod",
        "is_monotonic_increasing",
        "is_monotonic_decreasing",
        "idxmax",
        "idxmin",
        "query",
    }
)

for method in _SINGLE_ID_FORWARDING_METHODS:
    _set_forwarding_method_for_single_id(method)
