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
import pickle
import inspect
from pandas.api.types import is_list_like
from pandas.core.computation.parsing import tokenize_string


class ClientQueryCompiler(BaseQueryCompiler):
    @classmethod
    def set_server_connection(cls, conn):
        cls._service = conn

    def __init__(self, id):
        assert (
            id is not None
        ), "Make sure the client is properly connected and returns and ID"
        self._id = id

    def _set_columns(self, new_columns):
        self._id = self._service.rename(self._id, new_col_labels=new_columns)

    def _get_columns(self):
        if self._columns_cache is None:
            self._columns_cache = pickle.loads(
                pickle.dumps(self._service.columns(self._id))
            )
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
            ref = self._service.dtypes(self._id)
            self._dtypes_cache = pickle.loads(pickle.dumps(ref))
        return self._dtypes_cache

    @classmethod
    def from_pandas(cls, df, data_cls):
        raise NotImplementedError

    def to_pandas(self):
        remote_obj = self._service.to_pandas(self._id)
        return pickle.loads(pickle.dumps(remote_obj))

    def default_to_pandas(self, pandas_op, *args, **kwargs):
        raise NotImplementedError

    def columnarize(self):
        return self.__constructor__(self._service.columnarize(self._id))

    def transpose(self):
        return self.__constructor__(self._service.transpose(self._id))

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

    def getitem_column_array(self, key, numeric=False):
        return self.__constructor__(
            self._service.getitem_column_array(self._id, key, numeric)
        )

    def getitem_row_labels_array(self, labels):
        return self.__constructor__(
            self._service.getitem_row_labels_array(self._id, labels)
        )

    def getitem_row_array(self, key):
        return self.__constructor__(self._service.getitem_row_array(self._id, key))

    def pivot(self, index, columns, values):
        return self.__constructor__(
            self._service.pivot(self._id, index, columns, values)
        )

    def get_dummies(self, columns, **kwargs):
        return self.__constructor__(
            self._service.get_dummies(self._id, columns, **kwargs)
        )

    def view(self, index=None, columns=None):
        return self.__constructor__(self._service.view(self._id, index, columns))

    take_2d = view

    def drop(self, index=None, columns=None):
        return self.__constructor__(self._service.drop(self._id, index, columns))

    def isna(self):
        return self.__constructor__(self._service.isna(self._id))

    def notna(self):
        return self.__constructor__(self._service.notna(self._id))

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

    def dropna(self, **kwargs):
        return self.__constructor__(self._service.dropna(self._id, **kwargs))

    def sum(self, **kwargs):
        return self.__constructor__(self._service.sum(self._id, **kwargs))

    def prod(self, **kwargs):
        return self.__constructor__(self._service.prod(self._id, **kwargs))

    def count(self, **kwargs):
        return self.__constructor__(self._service.count(self._id, **kwargs))

    def mean(self, **kwargs):
        return self.__constructor__(self._service.mean(self._id, **kwargs))

    def median(self, **kwargs):
        return self.__constructor__(self._service.median(self._id, **kwargs))

    def std(self, **kwargs):
        return self.__constructor__(self._service.std(self._id, **kwargs))

    def min(self, **kwargs):
        return self.__constructor__(self._service.min(self._id, **kwargs))

    def max(self, **kwargs):
        return self.__constructor__(self._service.max(self._id, **kwargs))

    def any(self, **kwargs):
        return self.__constructor__(self._service.any(self._id, **kwargs))

    def all(self, **kwargs):
        return self.__constructor__(self._service.all(self._id, **kwargs))

    def quantile_for_single_value(self, **kwargs):
        return self.__constructor__(
            self._service.quantile_for_single_value(self._id, **kwargs)
        )

    def quantile_for_list_of_values(self, **kwargs):
        return self.__constructor__(
            self._service.quantile_for_list_of_values(self._id, **kwargs)
        )

    def describe(self, **kwargs):
        return self.__constructor__(self._service.describe(self._id, **kwargs))

    def set_index_from_columns(self, keys, drop: bool = True, append: bool = False):
        return self.__constructor__(
            self._service.set_index_from_columns(self._id, keys, drop, append)
        )

    def reset_index(self, **kwargs):
        return self.__constructor__(self._service.reset_index(self._id, **kwargs))

    def concat(self, axis, other, **kwargs):
        if is_list_like(other):
            other = [o._id for o in other]
        else:
            other = [other._id]
        return self.__constructor__(
            self._service.concat(self._id, axis, other, **kwargs)
        )

    def eq(self, other, **kwargs):
        if isinstance(other, ClientQueryCompiler):
            other = other._id
            is_qc = True
        else:
            is_qc = False
        return self.__constructor__(self._service.eq(self._id, other, is_qc, **kwargs))

    def lt(self, other, **kwargs):
        if isinstance(other, ClientQueryCompiler):
            other = other._id
            is_qc = True
        else:
            is_qc = False
        return self.__constructor__(self._service.lt(self._id, other, is_qc, **kwargs))

    def le(self, other, **kwargs):
        if isinstance(other, ClientQueryCompiler):
            other = other._id
            is_qc = True
        else:
            is_qc = False
        return self.__constructor__(self._service.le(self._id, other, is_qc, **kwargs))

    def gt(self, other, **kwargs):
        if isinstance(other, ClientQueryCompiler):
            other = other._id
            is_qc = True
        else:
            is_qc = False
        return self.__constructor__(self._service.gt(self._id, other, is_qc, **kwargs))

    def ge(self, other, **kwargs):
        if isinstance(other, ClientQueryCompiler):
            other = other._id
            is_qc = True
        else:
            is_qc = False
        return self.__constructor__(self._service.ge(self._id, other, is_qc, **kwargs))

    def ne(self, other, **kwargs):
        if isinstance(other, ClientQueryCompiler):
            other = other._id
            is_qc = True
        else:
            is_qc = False
        return self.__constructor__(self._service.ne(self._id, other, is_qc, **kwargs))

    def __and__(self, other, **kwargs):
        if isinstance(other, ClientQueryCompiler):
            other = other._id
            is_qc = True
        else:
            is_qc = False
        return self.__constructor__(
            self._service.__and__(self._id, other, is_qc, **kwargs)
        )

    def __or__(self, other, **kwargs):
        if isinstance(other, ClientQueryCompiler):
            other = other._id
            is_qc = True
        else:
            is_qc = False
        return self.__constructor__(
            self._service.__or__(self._id, other, is_qc, **kwargs)
        )

    def add(self, other, **kwargs):
        if isinstance(other, ClientQueryCompiler):
            other = other._id
            is_qc = True
        else:
            is_qc = False
        return self.__constructor__(self._service.add(self._id, other, is_qc, **kwargs))

    def radd(self, other, **kwargs):
        if isinstance(other, ClientQueryCompiler):
            other = other._id
            is_qc = True
        else:
            is_qc = False
        return self.__constructor__(
            self._service.radd(self._id, other, is_qc, **kwargs)
        )

    def truediv(self, other, **kwargs):
        if isinstance(other, ClientQueryCompiler):
            other = other._id
            is_qc = True
        else:
            is_qc = False
        return self.__constructor__(
            self._service.truediv(self._id, other, is_qc, **kwargs)
        )

    def mod(self, other, **kwargs):
        if isinstance(other, ClientQueryCompiler):
            other = other._id
            is_qc = True
        else:
            is_qc = False
        return self.__constructor__(self._service.mod(self._id, other, is_qc, **kwargs))

    def rmod(self, other, **kwargs):
        if isinstance(other, ClientQueryCompiler):
            other = other._id
            is_qc = True
        else:
            is_qc = False
        return self.__constructor__(
            self._service.rmod(self._id, other, is_qc, **kwargs)
        )

    def sub(self, other, **kwargs):
        if isinstance(other, ClientQueryCompiler):
            other = other._id
            is_qc = True
        else:
            is_qc = False
        return self.__constructor__(
            self._service.rsub(self._id, other, is_qc, **kwargs)
        )

    def rsub(self, other, **kwargs):
        if isinstance(other, ClientQueryCompiler):
            other = other._id
            is_qc = True
        else:
            is_qc = False
        return self.__constructor__(
            self._service.rsub(self._id, other, is_qc, **kwargs)
        )

    def mul(self, other, **kwargs):
        if isinstance(other, ClientQueryCompiler):
            other = other._id
            is_qc = True
        else:
            is_qc = False
        return self.__constructor__(self._service.mul(self._id, other, is_qc, **kwargs))

    def rmul(self, other, **kwargs):
        if isinstance(other, ClientQueryCompiler):
            other = other._id
            is_qc = True
        else:
            is_qc = False
        return self.__constructor__(
            self._service.rmul(self._id, other, is_qc, **kwargs)
        )

    def floordiv(self, other, **kwargs):
        if isinstance(other, ClientQueryCompiler):
            other = other._id
            is_qc = True
        else:
            is_qc = False
        return self.__constructor__(
            self._service.floordiv(self._id, other, is_qc, **kwargs)
        )

    def rfloordiv(self, other, **kwargs):
        if isinstance(other, ClientQueryCompiler):
            other = other._id
            is_qc = True
        else:
            is_qc = False
        return self.__constructor__(
            self._service.rfloordiv(self._id, other, is_qc, **kwargs)
        )

    def sort_rows_by_column_values(self, columns, ascending=True, **kwargs):
        return self.__constructor__(
            self._service.sort_rows_by_column_values(
                self._id, columns, ascending=ascending, **kwargs
            )
        )

    def sort_index(self, **kwargs):
        return self.__constructor__(self._service.sort_index(self._id, **kwargs))

    def str_capitalize(self):
        return self.__constructor__(self._service.str_capitalize(self._id))

    def str_isalnum(self):
        return self.__constructor__(self._service.str_isalnum(self._id))

    def str_isalpha(self):
        return self.__constructor__(self._service.str_isalpha(self._id))

    def str_isdecimal(self):
        return self.__constructor__(self._service.str_isdecimal(self._id))

    def str_isdigit(self):
        return self.__constructor__(self._service.str_isdigit(self._id))

    def str_islower(self):
        return self.__constructor__(self._service.str_islower(self._id))

    def str_isnumeric(self):
        return self.__constructor__(self._service.str_isnumeric(self._id))

    def str_isspace(self):
        return self.__constructor__(self._service.str_isspace(self._id))

    def str_istitle(self):
        return self.__constructor__(self._service.str_istitle(self._id))

    def str_isupper(self):
        return self.__constructor__(self._service.str_isupper(self._id))

    def str_len(self):
        return self.__constructor__(self._service.str_len(self._id))

    def str_lower(self):
        return self.__constructor__(self._service.str_lower(self._id))

    def str_title(self):
        return self.__constructor__(self._service.str_title(self._id))

    def str_upper(self):
        return self.__constructor__(self._service.str_upper(self._id))

    def str_center(self, width, fillchar=" "):
        return self.__constructor__(self._service.str_center(self._id, width, fillchar))

    def str_contains(self, pat, case=True, flags=0, na=np.nan, regex=True):
        return self.__constructor__(
            self._service.str_contains(self._id, pat, case, flags, na, regex)
        )

    def str_count(self, pat, flags=0, **kwargs):
        return self.__constructor__(
            self._service.str_count(self._id, pat, flags, **kwargs)
        )

    def str_endswith(self, pat, na=np.nan):
        return self.__constructor__(self._service.str_endswith(self._id, pat, na))

    def str_find(self, sub, start=0, end=None):
        return self.__constructor__(self._service.str_find(self._id, sub, start, end))

    def str_findall(self, pat, flags=0, **kwargs):
        return self.__constructor__(
            self._service.str_findall(self._id, pat, flags, **kwargs)
        )

    def str_get(self, i):
        return self.__constructor__(self._service.str_get(self._id, i))

    str_index = str_find

    def str_join(self, sep):
        return self.__constructor__(self._service.str_join(self._id, sep))

    def str_lstrip(self, to_strip=None):
        return self.__constructor__(self._service.str_lstrip(self._id, to_strip))

    def str_ljust(self, width, fillchar=" "):
        return self.__constructor__(self._service.str_ljust(self._id, width, fillchar))

    def str_match(self, pat, case=True, flags=0, na=np.nan):
        return self.__constructor__(
            self._service.str_match(self._id, pat, case, flags, na)
        )

    def str_pad(self, width, side="left", fillchar=" "):
        return self.__constructor__(
            self._service.str_pad(self._id, width, side, fillchar)
        )

    def str_repeat(self, repeats):
        return self.__constructor__(self._service.str_repeat(self._id, repeats))

    def str_rsplit(self, pat=None, n=-1, expand=False):
        return self.__constructor__(self._service.str_rsplit(self._id, pat, n, expand))

    def str_rstrip(self, to_strip=None):
        return self.__constructor__(self._service.str_rstrip(self._id, to_strip))

    def str_slice(self, start=None, stop=None, step=None):
        return self.__constructor__(
            self._service.str_slice(self._id, start, stop, step)
        )

    def str_slice_replace(self, start=None, stop=None, repl=None):
        return self.__constructor__(
            self._service.str_slice_replace(self._id, start, stop, repl)
        )

    def str_startswith(self, pat, na=np.nan):
        return self.__constructor__(self._service.str_startswith(self._id, pat, na))

    def str_strip(self, to_strip=None):
        return self.__constructor__(self._service.str_strip(self._id, to_strip))

    def str_zfill(self, width):
        return self.__constructor__(self._service.str_zfill(self._id, width))

    def merge(self, right, **kwargs):
        return self.__constructor__(self._service.merge(self._id, right._id, **kwargs))

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

    def cummax(self, fold_axis, skipna, *args, **kwargs):
        return self.__constructor__(
            self._service.cummax(self._id, fold_axis, skipna, *args, **kwargs)
        )

    def cummin(self, fold_axis, skipna, *args, **kwargs):
        return self.__constructor__(
            self._service.cummin(self._id, fold_axis, skipna, *args, **kwargs)
        )

    def cumsum(self, fold_axis, skipna, *args, **kwargs):
        return self.__constructor__(
            self._service.cumsum(self._id, fold_axis, skipna, *args, **kwargs)
        )

    def cumprod(self, fold_axis, skipna, *args, **kwargs):
        return self.__constructor__(
            self._service.cumprod(self._id, fold_axis, skipna, *args, **kwargs)
        )

    def get_index_names(self, axis=0):
        if axis == 0:
            return self.index.names
        else:
            return self.columns.names

    def is_monotonic_increasing(self):
        return self.__constructor__(self._service.is_monotonic_increasing(self._id))

    def is_monotonic_decreasing(self):
        return self.__constructor__(self._service.is_monotonic_decreasing(self._id))

    def idxmin(self, **kwargs):
        return self.__constructor__(self._service.idxmin(self._id, **kwargs))

    def idxmax(self, **kwargs):
        return self.__constructor__(self._service.idxmax(self._id, **kwargs))

    def query(self, expr, **kwargs):
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
                variable_list.append(v)
        expr = " ".join(variable_list)
        return self.__constructor__(self._service.query(self._id, expr, **kwargs))

    def finalize(self):
        raise NotImplementedError

    def free(self):
        raise NotImplementedError

    @classmethod
    def from_arrow(cls, at, data_cls):
        raise NotImplementedError

    @classmethod
    def from_dataframe(cls, df, data_cls):
        raise NotImplementedError

    def to_dataframe(self, nan_as_null: bool = False, allow_copy: bool = True):
        raise NotImplementedError
