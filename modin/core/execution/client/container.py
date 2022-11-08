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

"""Module contains ``ForwardingQueryCompilerContainer`` class."""

import numpy as np
import pandas
from typing import Any, NamedTuple, Optional, Union
from uuid import UUID, uuid4

from modin.core.storage_formats.base.query_compiler import BaseQueryCompiler


class ForwardingQueryCompilerContainer:
    """
    Container that forwards queries to query compilers within.

    Parameters
    ----------
    query_compiler_class : type
        Query compiler class to contain. Should be a subclass of BaseQueryCompiler.
    io_class : type
        The IO class to use for reading and writing data. Should be a subclass
        of modin.core.io.io.BaseIO.
    """

    def __init__(self, query_compiler_class: type, io_class: type):
        self._qc = {}
        self._query_compiler_class = query_compiler_class
        self._io_class = io_class

    def _generate_id(self) -> UUID:
        """
        Generate an ID for a new query compiler.

        Returns
        -------
        UUID
            The generated ID.
        """
        id = uuid4()
        while id in self._qc:
            id = uuid4()
        return id

    def add_query_compiler(self, qc: BaseQueryCompiler) -> UUID:
        """
        Add a query compiler to the container.

        Parameters
        ----------
        qc : BaseQueryCompiler

        Returns
        -------
        UUID
            The ID of the query compiler.
        """
        id = self._generate_id()
        self._qc[id] = qc
        return id

    def to_pandas(self, id: UUID) -> pandas.DataFrame:
        """
        Convert the query compiler to a pandas DataFrame.

        Parameters
        ----------
        id : UUID
            The ID of the query compiler to convert.

        Returns
        -------
        pandas.DataFrame
            The converted DataFrame.
        """
        return self._qc[id].to_pandas()

    class DefaultToPandasResult(NamedTuple):
        """
        The result of ``default_to_pandas``.

        Parameters
        ----------
        result : Any
            The result of the operation.
        result_is_qc_id : bool
            Whether the result is a query compiler ID.
        """

        result: Any
        result_is_qc_id: bool

    def default_to_pandas(
        self, id: UUID, pandas_op: Union[str, callable], *args: Any, **kwargs: dict
    ) -> DefaultToPandasResult:  # noqa: D401
        """
        Default to pandas for an operation on a query compiler.

        Use the inner query compiler's default_to_pandas to execute the
        operation on a pandas dataframe.

        Parameters
        ----------
        id : UUID
            The ID of the query compiler.
        pandas_op : Union[str, callable]
            The operation to perform.
        *args : iterable
            The arguments to pass to the operation.
        **kwargs : dict
            The keyword arguments to pass to the operation.

        Returns
        -------
        DefaultToPandasResult
            The result of the operation. The result is a query compiler ID if
            and only if the result of the pandas operation is a new
            query compiler.
        """
        result = self._qc[id].default_to_pandas(pandas_op, *args, **kwargs)
        result_is_qc_id = isinstance(result, self._query_compiler_class)
        if result_is_qc_id:
            new_id = self._generate_id()
            self._qc[new_id] = result
            result = new_id
        return self.DefaultToPandasResult(
            result=result, result_is_qc_id=result_is_qc_id
        )

    def rename(
        self,
        id: UUID,
        new_col_labels: Optional[pandas.Index] = None,
        new_row_labels: Optional[pandas.Index] = None,
    ) -> UUID:
        """
        Rename the columns and/or rows of a query compiler.

        Parameters
        ----------
        id : UUID
            The ID of the query compiler.
        new_col_labels : pandas.Index, default: None
            The new column labels.
        new_row_labels : pandas.Index, default: None
            The new row labels.

        Returns
        -------
        UUID
            The ID of the renamed query compiler.
        """
        new_id = self._generate_id()
        new_qc = self._qc[new_id] = self._qc[id].copy()
        if new_col_labels is not None:
            new_qc.columns = new_col_labels
        if new_row_labels is not None:
            new_qc.index = new_row_labels
        return new_id

    def columns(self, id) -> pandas.Index:
        """
        Get the columns of the query compiler.

        Parameters
        ----------
        id : UUID
            The ID of a query compiler.

        Returns
        -------
        pandas.Index
            The columns.
        """
        return self._qc[id].columns

    def index(self, id: UUID) -> pandas.Index:
        """
        Get the index of a query compiler.

        Parameters
        ----------
        id : UUID
            The ID of the query compiler.

        Returns
        -------
        pandas.Index
            The index.
        """
        return self._qc[id].index

    def dtypes(self, id: UUID) -> pandas.Series:
        """
        Get the dtypes of a query compiler.

        Parameters
        ----------
        id : UUID
            The ID of the query compiler.

        Returns
        -------
        pandas.Series
            The dtypes.
        """
        return self._qc[id].dtypes

    def insert(self, id: UUID, value_is_qc: bool, loc, column, value) -> UUID:
        """
        Insert a value into a query compiler.

        Parameters
        ----------
        id : UUID
            The ID of the query compiler.
        value_is_qc : bool
            Whether ``value`` is the ID of a query compiler.
        loc : int
            The location to insert the value.
        column : str
            The column to insert the value.
        value : Any
            The value to insert.

        Returns
        -------
        UUID
            The ID of the query compiler with the inserted value.
        """
        if value_is_qc:
            value = self._qc[value]
        new_id = self._generate_id()
        self._qc[new_id] = self._qc[id].insert(loc, column, value)
        return new_id

    def setitem(self, id, value_is_qc: bool, axis, key, value) -> UUID:
        """
        Set a value in a query compiler.

        Parameters
        ----------
        id : UUID
            The ID of the query compiler.
        value_is_qc : bool
            Whether ``value`` is the ID of a query compiler.
        axis : int
            The axis to set the value.
        key : Any
            The key to set the value.
        value : Any
            The value to set.

        Returns
        -------
        UUID
            The ID of the query compiler with the value set.
        """
        if value_is_qc:
            value = self._qc[value]
        new_id = self._generate_id()
        self._qc[new_id] = self._qc[id].setitem(axis, key, value)
        return new_id

    def getitem_array(
        self, id, key_is_qc: bool, key: Union[UUID, np.ndarray, list]
    ) -> UUID:
        """
        Get the values at ``key`` from a query compiler.

        Parameters
        ----------
        id : UUID
            The ID of the query compiler.
        key_is_qc : bool
            Whether ``key`` is the ID of a query compiler.
        key : UUID, np.ndarray or list of column labels
            Boolean mask represented by QueryCompiler UUID or ``np.ndarray`` of the same
            shape as query compiler with ID ``id``, or enumerable of columns to pick.

        Returns
        -------
        UUID
            The ID of the new query compiler.
        """
        if key_is_qc:
            key = self._qc[key]
        new_id = self._generate_id()
        self._qc[new_id] = self._qc[id].getitem_array(key)
        return new_id

    def replace(
        self,
        id,
        to_replace_is_qc: bool,
        regex_is_qc: bool,
        to_replace,
        value,
        inplace,
        limit,
        regex,
        method,
    ):
        """
        Replace values given in `to_replace` by `value`.

        Parameters
        ----------
        id : UUID
            The ID of the query compiler.
        to_replace_is_qc : bool
            Whether ``to_replace`` is the ID of a query compiler.
        regex_is_qc : bool
            Whether ``regex`` is the ID of a query compiler.
        to_replace : scalar, list-like, regex, modin.pandas.Series, or None
            Value to replace.
        value : scalar, list-like, regex or dict
            Value to replace matching values with.
        inplace : bool
            This parameter is for compatibility. Always has to be False.
        limit : Optional[int]
            Maximum size gap to forward or backward fill.
        regex : bool or same types as ``to_replace``
            Whether to interpret ``to_replace`` and/or ``value`` as regular
            expressions.
        method : {"pad", "ffill", "bfill", None}
            The method to use when for replacement, when to_replace is a
            scalar, list or tuple and value is None.

        Returns
        -------
        UUID
            UUID of query compiler with all `to_replace` values replaced by `value`.
        """
        if to_replace_is_qc:
            to_replace = self._qc[to_replace]
        if regex_is_qc:
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
        value_is_qc: bool,
        squeeze_self: bool,
        squeeze_value: bool,
        value,
        method,
        axis,
        inplace,
        limit,
        downcast,
    ):
        """
        Replace NaN values using provided method.

        Parameters
        ----------
        id : UUID
            The ID of the query compiler.
        value_is_qc : bool
            Whether ``value`` is the ID of a query compiler.
        squeeze_self : bool
            Whether to squeeze ``self``.
        squeeze_value : bool
            Whether to squeeze ``value``.
        value : scalar or dict
        method : {"backfill", "bfill", "pad", "ffill", None}
        axis : {0, 1}
        inplace : {False}
            This parameter is for compatibility. Always has to be False.
        limit : int, optional
        downcast : dict, optional

        Returns
        -------
        BaseQueryCompiler
            New QueryCompiler with all null values filled.
        """
        if value_is_qc:
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
        """
        Concatenate query compilers along the specified axis.

        Parameters
        ----------
        id : UUID
            The ID of the main query compiler to concatenate.
        axis : {0, 1}
            The axis to concatenate along.
        other : list of UUIDs
            The IDs of the query compilers to concatenate to the one
            represented by ``id``.
        **kwargs : dict
            Additional parameters to pass to the concatenation function.

        Returns
        -------
        UUID
            The ID of the query compiler containing the concatenation result.
        """
        other = [self._qc[o] for o in other]
        new_id = self._generate_id()
        self._qc[new_id] = self._qc[id].concat(axis, other, **kwargs)
        return new_id

    def merge(self, id, right, **kwargs):
        """
        Merge two query compilers using a database-style join.

        Parameters
        ----------
        id : UUID
            The ID of the left query compiler.
        right : UUID
            The ID of the right query compiler.
        **kwargs : dict
            Additional parameters to pass to the merge function.

        Returns
        -------
        UUID
            The ID of the query compiler containing the merge result.
        """
        new_id = self._generate_id()
        self._qc[new_id] = self._qc[id].merge(self._qc[right], **kwargs)
        return new_id

    ### I/O methods go below. ###

    def read_csv(self, connection, filepath, **kwargs) -> UUID:
        """
        Read a CSV file from the specified filepath.

        Parameters
        ----------
        connection : object
            The data connection, e.g. a connnection to the database where the
            service will store the result.
        filepath : str
            The filepath to read the CSV file from.
        **kwargs : dict
            Additional parameters to pass to the pandas read_csv function.

        Returns
        -------
        UUID
            The ID of the query compiler containing the read result.
        """
        io_result = self._io_class._read_csv(filepath, **kwargs)
        if isinstance(io_result, self._query_compiler_class):
            new_id = self._generate_id()
            self._qc[new_id] = io_result
            return new_id
        return io_result

    def read_sql(self, sql, connection, **kwargs) -> UUID:
        """
        Read data from a SQL connection.

        Parameters
        ----------
        sql : str
            SQL query to be executed or a table name.
        connection : SQLAlchemy connectable, str, or sqlite3 connection
            Using SQLAlchemy makes it possible to use any DB supported by that
            library. If a DBAPI2 object, only sqlite3 is supported. The user is responsible
            for engine disposal and connection closure for the SQLAlchemy
            connectable; str connections are closed automatically. See
            `here <https://docs.sqlalchemy.org/en/13/core/connections.html>`_.
        **kwargs : dict
            Parameters of ``read_sql`` function.

        Returns
        -------
        UUID
            ID of query compiler with data read in from SQL connection.
        """
        new_id = self._generate_id()
        self._qc[new_id] = self._io_class._read_sql(sql, connection, **kwargs)
        return new_id

    def to_sql(self, id, **kwargs) -> None:
        """
        Write records stored in a DataFrame to a SQL database.

        Databases supported by SQLAlchemy [1]_ are supported. Tables can be
        newly created, appended to, or overwritten.

        Parameters
        ----------
        id : UUID
            ID of query compiler to write to database.
        **kwargs : dict
            Parameters of ``read_sql`` function.
        """
        self._io_class.to_sql(self._qc[id], **kwargs)


def _set_forwarding_groupby_method(method_name: str):
    """
    Define a groupby method that forwards arguments to an inner query compiler.

    Parameters
    ----------
    method_name : str
    """

    def forwarding_method(self, id, by_is_qc, by, *args, **kwargs):
        if by_is_qc:
            by = self._qc[by]
        new_id = self._generate_id()
        self._qc[new_id] = getattr(self._qc[id], method_name)(by, *args, **kwargs)
        return new_id

    setattr(ForwardingQueryCompilerContainer, method_name, forwarding_method)


def _set_forwarding_method_for_single_id(method_name: str):
    """
    Define a method that forwards arguments to an inner query compiler.

    Parameters
    ----------
    method_name : str
    """

    def forwarding_method(
        self: ForwardingQueryCompilerContainer, id: UUID, *args, **kwargs
    ):
        new_id = self._generate_id()
        self._qc[new_id] = getattr(self._qc[id], method_name)(*args, **kwargs)
        return new_id

    setattr(ForwardingQueryCompilerContainer, method_name, forwarding_method)


def _set_forwarding_method_for_binary_function(method_name: str):
    """
    Define a binary method that forwards arguments to an inner query compiler.

    Parameters
    ----------
    method_name : str
    """

    def forwarding_method(
        self: ForwardingQueryCompilerContainer,
        id: UUID,
        other_is_qc: bool,
        other: Union[UUID, Any],
        **kwargs,
    ):
        if other_is_qc:
            other = self._qc[other]
        new_id = self._generate_id()
        self._qc[new_id] = getattr(self._qc[id], method_name)(other, **kwargs)
        return new_id

    setattr(ForwardingQueryCompilerContainer, method_name, forwarding_method)


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

for method in _SINGLE_ID_FORWARDING_METHODS:
    _set_forwarding_method_for_single_id(method)

for method in _BINARY_FORWARDING_METHODS:
    _set_forwarding_method_for_binary_function(method)

for method in _GROUPBY_FORWARDING_METHODS:
    _set_forwarding_groupby_method("groupby_" + method)
