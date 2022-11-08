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

"""Module contains ``ClientQueryCompiler`` class."""

import pandas
from pandas._libs.lib import no_default, NoDefault
from pandas.api.types import is_list_like
from typing import Any
from uuid import UUID

from modin.core.storage_formats.base.query_compiler import BaseQueryCompiler
from modin.utils import _inherit_docstrings


@_inherit_docstrings(BaseQueryCompiler)
class ClientQueryCompiler(BaseQueryCompiler):
    """
    Query compiler for sending queries to a remote server.

    This class translates the query compiler API to function calls on a service
    object, which may be a remote service.

    Parameters
    ----------
    id : UUID
        ID of this query compiler.
    """

    lazy_execution: bool = True

    def __init__(self, id: UUID):
        self._id = id

    @classmethod
    def set_server_connection(cls, conn: Any):
        """
        Set the connection to the service.

        Parameters
        ----------
        conn : Any
            Connection to the service.
        """
        cls._service = conn

    def _set_columns(self, new_columns: pandas.Index) -> None:
        """
        Set this query compiler's columns.

        Parameters
        ----------
        new_columns : pandas.Index
            New columns to set.
        """
        self._id = self._service.rename(self._id, new_col_labels=new_columns)
        self._columns_cache = self._service.columns(self._id)

    def _get_columns(self) -> pandas.Index:
        """
        Get the columns of this query compiler.

        Returns
        -------
        pandas.Index : The columns of this query compiler.
        """
        if self._columns_cache is None:
            self._columns_cache = self._service.columns(self._id)
        return self._columns_cache

    def _set_index(self, new_index: pandas.Index):
        """
        Set this query compiler's index.

        Parameters
        ----------
        new_index : pandas.Index
            New index to set.
        """
        self._id = self._service.rename(self._id, new_row_labels=new_index)

    def _get_index(self) -> pandas.Index:
        """
        Get the index of this query compiler.

        Returns
        -------
        pandas.Index : The index of this query compiler.
        """
        return self._service.index(self._id)

    columns = property(_get_columns, _set_columns)
    _columns_cache: pandas.Index = None
    index = property(_get_index, _set_index)
    _dtypes_cache: pandas.Index = None

    @property
    def dtypes(self):
        if self._dtypes_cache is None:
            self._dtypes_cache = self._service.dtypes(self._id)
        return self._dtypes_cache

    @classmethod
    def from_pandas(cls, df, data_cls):
        raise NotImplementedError

    def to_pandas(self):
        value = self._service.to_pandas(self._id)
        if isinstance(value, Exception):
            raise value
        return value

    def default_to_pandas(self, pandas_op, *args, **kwargs):
        raise NotImplementedError

    def copy(self):
        return self.__constructor__(self._id)

    def insert(self, loc, column, value):
        value_is_qc = isinstance(value, type(self))
        if value_is_qc:
            value = value._id
        return self.__constructor__(
            self._service.insert(self._id, value_is_qc, loc, column, value)
        )

    def setitem(self, axis, key, value):
        value_is_qc = isinstance(value, type(self))
        if value_is_qc:
            value = value._id
        return self.__constructor__(
            self._service.setitem(self._id, value_is_qc, axis, key, value)
        )

    def getitem_array(self, key):
        key_is_qc = isinstance(key, type(self))
        if key_is_qc:
            key = key._id
        return self.__constructor__(
            self._service.getitem_array(self._id, key_is_qc, key)
        )

    def replace(
        self,
        to_replace=None,
        value=no_default,
        inplace=False,
        limit=None,
        regex=False,
        method: "str | NoDefault" = no_default,
    ):
        to_replace_is_qc = isinstance(to_replace, type(self))
        if to_replace_is_qc:
            to_replace = to_replace._id
        regex_is_qc = isinstance(regex, type(self))
        if regex_is_qc:
            regex = regex._id
        return self.__constructor__(
            self._service.replace(
                self._id,
                to_replace_is_qc,
                regex_is_qc,
                to_replace,
                value,
                inplace,
                limit,
                regex,
                method,
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
        value_is_qc = isinstance(value, type(self))
        if value_is_qc:
            value = value._id
        return self.__constructor__(
            self._service.fillna(
                self._id,
                value_is_qc,
                squeeze_self,
                squeeze_value,
                value,
                method,
                axis,
                inplace,
                limit,
                downcast,
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

    def merge(self, right, **kwargs):
        return self.__constructor__(self._service.merge(self._id, right._id, **kwargs))

    def get_index_names(self, axis=0):
        if axis == 0:
            return self.index.names
        else:
            return self.columns.names

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


def _set_forwarding_groupby_method(method_name: str):
    """
    Define a groupby method that forwards arguments to the service.

    Parameters
    ----------
    method_name : str
    """

    def forwading_method(self, by, *args, **kwargs):
        by_is_qc: bool = isinstance(by, type(self))
        if by_is_qc:
            by = by._id
        return self.__constructor__(
            getattr(self._service, method_name)(self._id, by_is_qc, by, *args, **kwargs)
        )

    setattr(ClientQueryCompiler, method_name, forwading_method)


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
            getattr(self._service, method_name)(self._id, other_is_qc, other, **kwargs)
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


_GROUPBY_FORWARDING_METHODS = frozenset({"mean", "count", "max", "min", "sum", "agg"})

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
        "sub",
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
    }
)

_SINGLE_ID_FORWARDING_METHODS = frozenset(
    {
        "columnarize",
        "transpose",
        "take_2d",
        "getitem_column_array",
        "getitem_row_array",
        "take_2d_labels",
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

for method in _BINARY_FORWARDING_METHODS:
    _set_forwarding_method_for_binary_function(method)

for method in _SINGLE_ID_FORWARDING_METHODS:
    _set_forwarding_method_for_single_id(method)

for method in _GROUPBY_FORWARDING_METHODS:
    _set_forwarding_groupby_method("groupby_" + method)
