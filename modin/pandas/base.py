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

import os
import numpy as np
from numpy import nan
import pandas
from pandas.compat import numpy as numpy_compat
from pandas.core.common import count_not_none, pipe
from pandas.core.dtypes.common import (
    is_list_like,
    is_dict_like,
    is_numeric_dtype,
    is_datetime_or_timedelta_dtype,
    is_dtype_equal,
    is_object_dtype,
)
from pandas.core.indexing import convert_to_index_sliceable
from pandas.util._validators import validate_bool_kwarg, validate_percentile
from pandas._libs.lib import no_default
from pandas._typing import (
    TimedeltaConvertibleTypes,
    TimestampConvertibleTypes,
    IndexKeyFunc,
)
import re
from typing import Optional, Union
import warnings
import pickle as pkl

from modin.utils import try_cast_to_pandas
from modin.error_message import ErrorMessage
from modin.pandas.utils import is_scalar

# Similar to pandas, sentinel value to use as kwarg in place of None when None has
# special meaning and needs to be distinguished from a user explicitly passing None.
sentinel = object()

# Do not lookup certain attributes in columns or index, as they're used for some
# special purposes, like serving remote context
_ATTRS_NO_LOOKUP = {"____id_pack__", "__name__"}


class BasePandasDataset(object):
    """This object is the base for most of the common code that exists in
    DataFrame/Series. Since both objects share the same underlying representation,
    and the algorithms are the same, we use this object to define the general
    behavior of those objects and then use those objects to define the output type.
    """

    # Siblings are other objects that share the same query compiler. We use this list
    # to update inplace when there is a shallow copy.
    _siblings = []

    def _add_sibling(self, sibling):
        sibling._siblings = self._siblings + [self]
        self._siblings += [sibling]
        for sib in self._siblings:
            sib._siblings += [sibling]

    def _build_repr_df(self, num_rows, num_cols):
        # Fast track for empty dataframe.
        if len(self.index) == 0 or (
            hasattr(self, "columns") and len(self.columns) == 0
        ):
            return pandas.DataFrame(
                index=self.index,
                columns=self.columns if hasattr(self, "columns") else None,
            )
        if len(self.index) <= num_rows:
            row_indexer = slice(None)
        else:
            # Add one here so that pandas automatically adds the dots
            # It turns out to be faster to extract 2 extra rows and columns than to
            # build the dots ourselves.
            num_rows_for_head = num_rows // 2 + 1
            num_rows_for_tail = (
                num_rows_for_head
                if len(self.index) > num_rows
                else len(self.index) - num_rows_for_head
                if len(self.index) - num_rows_for_head >= 0
                else None
            )
            row_indexer = list(range(len(self.index))[:num_rows_for_head]) + (
                list(range(len(self.index))[-num_rows_for_tail:])
                if num_rows_for_tail is not None
                else []
            )
        if hasattr(self, "columns"):
            if len(self.columns) <= num_cols:
                col_indexer = slice(None)
            else:
                num_cols_for_front = num_cols // 2 + 1
                num_cols_for_back = (
                    num_cols_for_front
                    if len(self.columns) > num_cols
                    else len(self.columns) - num_cols_for_front
                    if len(self.columns) - num_cols_for_front >= 0
                    else None
                )
                col_indexer = list(range(len(self.columns))[:num_cols_for_front]) + (
                    list(range(len(self.columns))[-num_cols_for_back:])
                    if num_cols_for_back is not None
                    else []
                )
            indexer = row_indexer, col_indexer
        else:
            indexer = row_indexer
        return self.iloc[indexer]._query_compiler.to_pandas()

    def _update_inplace(self, new_query_compiler):
        """Updates the current DataFrame inplace.

        Args:
            new_query_compiler: The new QueryCompiler to use to manage the data
        """
        old_query_compiler = self._query_compiler
        self._query_compiler = new_query_compiler
        for sib in self._siblings:
            sib._query_compiler = new_query_compiler
        old_query_compiler.free()

    def _handle_level_agg(self, axis, level, op, sort=False, **kwargs):
        """Helper method to perform error checking for aggregation functions with a level parameter.
        Args:
            axis: The axis to apply the operation on
            level: The level of the axis to apply the operation on
            op: String representation of the operation to be performed on the level
        """
        return getattr(self.groupby(level=level, axis=axis, sort=sort), op)(**kwargs)

    def _validate_other(
        self,
        other,
        axis,
        numeric_only=False,
        numeric_or_time_only=False,
        numeric_or_object_only=False,
        comparison_dtypes_only=False,
    ):
        """Helper method to check validity of other in inter-df operations"""
        # We skip dtype checking if the other is a scalar.
        if is_scalar(other):
            return other
        axis = self._get_axis_number(axis) if axis is not None else 1
        result = other
        if isinstance(other, BasePandasDataset):
            return other._query_compiler
        elif is_list_like(other):
            if axis == 0:
                if len(other) != len(self._query_compiler.index):
                    raise ValueError(
                        "Unable to coerce to Series, length must be {0}: "
                        "given {1}".format(len(self._query_compiler.index), len(other))
                    )
            else:
                if len(other) != len(self._query_compiler.columns):
                    raise ValueError(
                        "Unable to coerce to Series, length must be {0}: "
                        "given {1}".format(
                            len(self._query_compiler.columns), len(other)
                        )
                    )
            if hasattr(other, "dtype"):
                other_dtypes = [other.dtype] * len(other)
            else:
                other_dtypes = [type(x) for x in other]
        else:
            other_dtypes = [
                type(other)
                for _ in range(
                    len(self._query_compiler.index)
                    if axis
                    else len(self._query_compiler.columns)
                )
            ]
        # Do dtype checking.
        if numeric_only:
            if not all(
                is_numeric_dtype(self_dtype) and is_numeric_dtype(other_dtype)
                for self_dtype, other_dtype in zip(self._get_dtypes(), other_dtypes)
            ):
                raise TypeError("Cannot do operation on non-numeric dtypes")
        elif numeric_or_object_only:
            if not all(
                (is_numeric_dtype(self_dtype) and is_numeric_dtype(other_dtype))
                or (is_object_dtype(self_dtype) and is_object_dtype(other_dtype))
                for self_dtype, other_dtype in zip(self._get_dtypes(), other_dtypes)
            ):
                raise TypeError("Cannot do operation non-numeric dtypes")
        elif comparison_dtypes_only:
            if not all(
                (is_numeric_dtype(self_dtype) and is_numeric_dtype(other_dtype))
                or (
                    is_datetime_or_timedelta_dtype(self_dtype)
                    and is_datetime_or_timedelta_dtype(other_dtype)
                )
                or is_dtype_equal(self_dtype, other_dtype)
                for self_dtype, other_dtype in zip(self._get_dtypes(), other_dtypes)
            ):
                raise TypeError(
                    "Cannot do operation non-numeric objects with numeric objects"
                )
        elif numeric_or_time_only:
            if not all(
                (is_numeric_dtype(self_dtype) and is_numeric_dtype(other_dtype))
                or (
                    is_datetime_or_timedelta_dtype(self_dtype)
                    and is_datetime_or_timedelta_dtype(other_dtype)
                )
                for self_dtype, other_dtype in zip(self._get_dtypes(), other_dtypes)
            ):
                raise TypeError(
                    "Cannot do operation non-numeric objects with numeric objects"
                )
        return result

    def _binary_op(self, op, other, **kwargs):
        # _axis indicates the operator will use the default axis
        if kwargs.pop("_axis", None) is None:
            if kwargs.get("axis", None) is not None:
                kwargs["axis"] = axis = self._get_axis_number(kwargs.get("axis", None))
            else:
                kwargs["axis"] = axis = 1
        else:
            axis = 0
        if kwargs.get("level", None) is not None:
            # Broadcast is an internally used argument
            kwargs.pop("broadcast", None)
            return self._default_to_pandas(
                getattr(getattr(pandas, type(self).__name__), op), other, **kwargs
            )
        other = self._validate_other(other, axis, numeric_or_object_only=True)
        new_query_compiler = getattr(self._query_compiler, op)(other, **kwargs)
        return self._create_or_update_from_compiler(new_query_compiler)

    def _default_to_pandas(self, op, *args, **kwargs):
        """Helper method to use default pandas function"""
        empty_self_str = "" if not self.empty else " for empty DataFrame"
        ErrorMessage.default_to_pandas(
            "`{}.{}`{}".format(
                type(self).__name__,
                op if isinstance(op, str) else op.__name__,
                empty_self_str,
            )
        )

        args = try_cast_to_pandas(args)
        kwargs = try_cast_to_pandas(kwargs)
        pandas_obj = self._to_pandas()
        if callable(op):
            result = op(pandas_obj, *args, **kwargs)
        elif isinstance(op, str):
            # The inner `getattr` is ensuring that we are treating this object (whether
            # it is a DataFrame, Series, etc.) as a pandas object. The outer `getattr`
            # will get the operation (`op`) from the pandas version of the class and run
            # it on the object after we have converted it to pandas.
            result = getattr(getattr(pandas, type(self).__name__), op)(
                pandas_obj, *args, **kwargs
            )
        else:
            ErrorMessage.catch_bugs_and_request_email(
                failure_condition=True,
                extra_log="{} is an unsupported operation".format(op),
            )
        # SparseDataFrames cannot be serialized by arrow and cause problems for Modin.
        # For now we will use pandas.
        if isinstance(result, type(self)) and not isinstance(
            result, (pandas.SparseDataFrame, pandas.SparseSeries)
        ):
            return self._create_or_update_from_compiler(
                result, inplace=kwargs.get("inplace", False)
            )
        elif isinstance(result, pandas.DataFrame):
            from .dataframe import DataFrame

            return DataFrame(result)
        elif isinstance(result, pandas.Series):
            from .series import Series

            return Series(result)
        # inplace
        elif result is None:
            import modin.pandas as pd

            return self._create_or_update_from_compiler(
                getattr(pd, type(pandas_obj).__name__)(pandas_obj)._query_compiler,
                inplace=True,
            )
        else:
            try:
                if (
                    isinstance(result, (list, tuple))
                    and len(result) == 2
                    and isinstance(result[0], pandas.DataFrame)
                ):
                    # Some operations split the DataFrame into two (e.g. align). We need to wrap
                    # both of the returned results
                    if isinstance(result[1], pandas.DataFrame):
                        second = self.__constructor__(result[1])
                    else:
                        second = result[1]
                    return self.__constructor__(result[0]), second
                else:
                    return result
            except TypeError:
                return result

    def _get_axis_number(self, axis):
        return (
            getattr(pandas, type(self).__name__)()._get_axis_number(axis)
            if axis is not None
            else 0
        )

    def __constructor__(self, *args, **kwargs):
        return type(self)(*args, **kwargs)

    def abs(self):
        """Apply an absolute value function to all numeric columns.

        Returns:
            A new DataFrame with the applied absolute value.
        """
        self._validate_dtypes(numeric_only=True)
        return self.__constructor__(query_compiler=self._query_compiler.abs())

    def _set_index(self, new_index):
        """Set the index for this DataFrame.

        Args:
            new_index: The new index to set this
        """
        self._query_compiler.index = new_index

    def _get_index(self):
        """Get the index for this DataFrame.

        Returns:
            The union of all indexes across the partitions.
        """
        return self._query_compiler.index

    index = property(_get_index, _set_index)

    def add(self, other, axis="columns", level=None, fill_value=None):
        """Add this DataFrame to another or a scalar/list.

        Args:
            other: What to add this this DataFrame.
            axis: The axis to apply addition over. Only applicable to Series
                or list 'other'.
            level: A level in the multilevel axis to add over.
            fill_value: The value to fill NaN.

        Returns:
            A new DataFrame with the applied addition.
        """
        return self._binary_op(
            "add", other, axis=axis, level=level, fill_value=fill_value
        )

    def aggregate(self, func=None, axis=0, *args, **kwargs):
        warnings.warn(
            "Modin index may not match pandas index due to pandas issue pandas-dev/pandas#36189."
        )
        axis = self._get_axis_number(axis)
        result = None

        if axis == 0:
            try:
                result = self._aggregate(func, _axis=axis, *args, **kwargs)
            except TypeError:
                pass
        if result is None:
            kwargs.pop("is_transform", None)
            return self.apply(func, axis=axis, args=args, **kwargs)
        return result

    agg = aggregate

    def _aggregate(self, arg, *args, **kwargs):
        _axis = kwargs.pop("_axis", 0)
        kwargs.pop("_level", None)

        if isinstance(arg, str):
            kwargs.pop("is_transform", None)
            return self._string_function(arg, *args, **kwargs)

        # Dictionaries have complex behavior because they can be renamed here.
        elif isinstance(arg, dict):
            return self._default_to_pandas("agg", arg, *args, **kwargs)
        elif is_list_like(arg) or callable(arg):
            kwargs.pop("is_transform", None)
            return self.apply(arg, axis=_axis, args=args, **kwargs)
        else:
            raise TypeError("type {} is not callable".format(type(arg)))

    def _string_function(self, func, *args, **kwargs):
        assert isinstance(func, str)
        f = getattr(self, func, None)
        if f is not None:
            if callable(f):
                return f(*args, **kwargs)
            assert len(args) == 0
            assert (
                len([kwarg for kwarg in kwargs if kwarg not in ["axis", "_level"]]) == 0
            )
            return f
        f = getattr(np, func, None)
        if f is not None:
            return self._default_to_pandas("agg", func, *args, **kwargs)
        raise ValueError("{} is an unknown string function".format(func))

    def _get_dtypes(self):
        if hasattr(self, "dtype"):
            return [self.dtype]
        else:
            return list(self.dtypes)

    def align(
        self,
        other,
        join="outer",
        axis=None,
        level=None,
        copy=True,
        fill_value=None,
        method=None,
        limit=None,
        fill_axis=0,
        broadcast_axis=None,
    ):
        return self._default_to_pandas(
            "align",
            other,
            join=join,
            axis=axis,
            level=level,
            copy=copy,
            fill_value=fill_value,
            method=method,
            limit=limit,
            fill_axis=fill_axis,
            broadcast_axis=broadcast_axis,
        )

    def all(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        """Return whether all elements are True over requested axis

        Note:
            If axis=None or axis=0, this call applies df.all(axis=1)
                to the transpose of df.
        """
        if axis is not None:
            axis = self._get_axis_number(axis)
            if bool_only and axis == 0:
                if hasattr(self, "dtype"):
                    raise NotImplementedError(
                        "{}.{} does not implement numeric_only.".format(
                            type(self).__name__, "all"
                        )
                    )
                data_for_compute = self[self.columns[self.dtypes == np.bool]]
                return data_for_compute.all(
                    axis=axis, bool_only=False, skipna=skipna, level=level, **kwargs
                )
            if level is not None:
                if bool_only is not None:
                    raise NotImplementedError(
                        "Option bool_only is not implemented with option level."
                    )
                return self._handle_level_agg(
                    axis, level, "all", skipna=skipna, **kwargs
                )
            return self._reduce_dimension(
                self._query_compiler.all(
                    axis=axis, bool_only=bool_only, skipna=skipna, level=level, **kwargs
                )
            )
        else:
            if bool_only:
                raise ValueError("Axis must be 0 or 1 (got {})".format(axis))
            # Reduce to a scalar if axis is None.
            if level is not None:
                return self._handle_level_agg(
                    axis, level, "all", skipna=skipna, **kwargs
                )
            else:
                result = self._reduce_dimension(
                    self._query_compiler.all(
                        axis=0,
                        bool_only=bool_only,
                        skipna=skipna,
                        level=level,
                        **kwargs,
                    )
                )
            if isinstance(result, BasePandasDataset):
                return result.all(
                    axis=axis, bool_only=bool_only, skipna=skipna, level=level, **kwargs
                )
            return result

    def any(self, axis=0, bool_only=None, skipna=True, level=None, **kwargs):
        """Return whether any elements are True over requested axis

        Note:
            If axis=None or axis=0, this call applies on the column partitions,
                otherwise operates on row partitions
        """
        if axis is not None:
            axis = self._get_axis_number(axis)
            if bool_only and axis == 0:
                if hasattr(self, "dtype"):
                    raise NotImplementedError(
                        "{}.{} does not implement numeric_only.".format(
                            type(self).__name__, "all"
                        )
                    )
                data_for_compute = self[self.columns[self.dtypes == np.bool]]
                return data_for_compute.any(
                    axis=axis, bool_only=False, skipna=skipna, level=level, **kwargs
                )
            if level is not None:
                if bool_only is not None:
                    raise NotImplementedError(
                        "Option bool_only is not implemented with option level."
                    )
                return self._handle_level_agg(
                    axis, level, "any", skipna=skipna, **kwargs
                )
            return self._reduce_dimension(
                self._query_compiler.any(
                    axis=axis, bool_only=bool_only, skipna=skipna, level=level, **kwargs
                )
            )
        else:
            if bool_only:
                raise ValueError("Axis must be 0 or 1 (got {})".format(axis))
            # Reduce to a scalar if axis is None.
            if level is not None:
                return self._handle_level_agg(
                    axis, level, "any", skipna=skipna, **kwargs
                )
            else:
                result = self._reduce_dimension(
                    self._query_compiler.any(
                        axis=0,
                        bool_only=bool_only,
                        skipna=skipna,
                        level=level,
                        **kwargs,
                    )
                )
            if isinstance(result, BasePandasDataset):
                return result.any(
                    axis=axis, bool_only=bool_only, skipna=skipna, level=level, **kwargs
                )
            return result

    def apply(
        self,
        func,
        axis=0,
        broadcast=None,
        raw=False,
        reduce=None,
        result_type=None,
        convert_dtype=True,
        args=(),
        **kwds,
    ):
        """Apply a function along input axis of DataFrame.

        Args:
            func: The function to apply
            axis: The axis over which to apply the func.
            broadcast: Whether or not to broadcast.
            raw: Whether or not to convert to a Series.
            reduce: Whether or not to try to apply reduction procedures.

        Returns:
            Series or DataFrame, depending on func.
        """
        warnings.warn(
            "Modin index may not match pandas index due to pandas issue pandas-dev/pandas#36189."
        )
        axis = self._get_axis_number(axis)
        ErrorMessage.non_verified_udf()
        if isinstance(func, str):
            # if axis != 1 function can be bounded to the Series, which doesn't
            # support axis parameter
            if axis == 1:
                kwds["axis"] = axis
            result = self._string_function(func, *args, **kwds)
            if isinstance(result, BasePandasDataset):
                return result._query_compiler
            return result
        elif isinstance(func, dict):
            if axis == 1:
                raise TypeError(
                    "(\"'dict' object is not callable\", "
                    "'occurred at index {0}'".format(self.index[0])
                )
            if len(self.columns) != len(set(self.columns)):
                warnings.warn(
                    "duplicate column names not supported with apply().",
                    FutureWarning,
                    stacklevel=2,
                )
        elif not callable(func) and not is_list_like(func):
            raise TypeError("{} object is not callable".format(type(func)))
        query_compiler = self._query_compiler.apply(
            func,
            axis,
            args=args,
            raw=raw,
            result_type=result_type,
            **kwds,
        )
        return query_compiler

    def asfreq(self, freq, method=None, how=None, normalize=False, fill_value=None):
        return self._default_to_pandas(
            "asfreq",
            freq,
            method=method,
            how=how,
            normalize=normalize,
            fill_value=fill_value,
        )

    def asof(self, where, subset=None):
        scalar = not is_list_like(where)
        if isinstance(where, pandas.Index):
            # Prevent accidental mutation of original:
            where = where.copy()
        else:
            if scalar:
                where = [where]
            where = pandas.Index(where)

        if subset is None:
            data = self
        else:
            # Only relevant for DataFrames:
            data = self[subset]
        no_na_index = data.dropna().index
        new_index = pandas.Index([no_na_index.asof(i) for i in where])
        result = self.reindex(new_index)
        result.index = where

        if scalar:
            # Need to return a Series:
            result = result.squeeze()
        return result

    def astype(self, dtype, copy=True, errors="raise"):
        col_dtypes = {}
        if isinstance(dtype, dict):
            if (
                not set(dtype.keys()).issubset(set(self._query_compiler.columns))
                and errors == "raise"
            ):
                raise KeyError(
                    "Only a column name can be used for the key in"
                    "a dtype mappings argument."
                )
            col_dtypes = dtype
        else:
            for column in self._query_compiler.columns:
                col_dtypes[column] = dtype

        new_query_compiler = self._query_compiler.astype(col_dtypes)
        return self._create_or_update_from_compiler(new_query_compiler, not copy)

    @property
    def at(self, axis=None):
        from .indexing import _LocIndexer

        return _LocIndexer(self)

    def at_time(self, time, asof=False, axis=None):
        axis = self._get_axis_number(axis)
        if axis == 0:
            return self.iloc[self.index.indexer_at_time(time, asof=asof)]
        return self.iloc[:, self.columns.indexer_at_time(time, asof=asof)]

    def between_time(
        self, start_time, end_time, include_start=True, include_end=True, axis=None
    ):
        axis = self._get_axis_number(axis)
        idx = self.index if axis == 0 else self.columns
        indexer = idx.indexer_between_time(
            start_time, end_time, include_start=include_start, include_end=include_end
        )
        return self.iloc[indexer] if axis == 0 else self.iloc[:, indexer]

    def bfill(self, axis=None, inplace=False, limit=None, downcast=None):
        """Synonym for DataFrame.fillna(method='bfill')"""
        return self.fillna(
            method="bfill", axis=axis, limit=limit, downcast=downcast, inplace=inplace
        )

    backfill = bfill

    def bool(self):
        """Return the bool of a single element PandasObject.

        This must be a boolean scalar value, either True or False.  Raise a
        ValueError if the PandasObject does not have exactly 1 element, or that
        element is not boolean
        """
        shape = self.shape
        if shape != (1,) and shape != (1, 1):
            raise ValueError(
                """The PandasObject does not have exactly
                                1 element. Return the bool of a single
                                element PandasObject. The truth value is
                                ambiguous. Use a.empty, a.item(), a.any()
                                or a.all()."""
            )
        else:
            return self._to_pandas().bool()

    def clip(self, lower=None, upper=None, axis=None, inplace=False, *args, **kwargs):
        # validate inputs
        if axis is not None:
            axis = self._get_axis_number(axis)
        self._validate_dtypes(numeric_only=True)
        if is_list_like(lower) or is_list_like(upper):
            if axis is None:
                raise ValueError("Must specify axis = 0 or 1")
            self._validate_other(lower, axis)
            self._validate_other(upper, axis)
        inplace = validate_bool_kwarg(inplace, "inplace")
        axis = numpy_compat.function.validate_clip_with_axis(axis, args, kwargs)
        # any np.nan bounds are treated as None
        if lower is not None and np.any(np.isnan(lower)):
            lower = None
        if upper is not None and np.any(np.isnan(upper)):
            upper = None
        new_query_compiler = self._query_compiler.clip(
            lower=lower, upper=upper, axis=axis, inplace=inplace, *args, **kwargs
        )
        return self._create_or_update_from_compiler(new_query_compiler, inplace)

    def combine(self, other, func, fill_value=None, **kwargs):
        return self._binary_op(
            "combine", other, _axis=0, func=func, fill_value=fill_value, **kwargs
        )

    def combine_first(self, other):
        return self._binary_op("combine_first", other, _axis=0)

    def copy(self, deep=True):
        """Creates a shallow copy of the DataFrame.

        Returns:
            A new DataFrame pointing to the same partitions as this one.
        """
        if deep:
            return self.__constructor__(query_compiler=self._query_compiler.copy())
        new_obj = self.__constructor__(query_compiler=self._query_compiler)
        self._add_sibling(new_obj)
        return new_obj

    def count(self, axis=0, level=None, numeric_only=False):
        """Get the count of non-null objects in the DataFrame.

        Arguments:
            axis: 0 or 'index' for row-wise, 1 or 'columns' for column-wise.
            level: If the axis is a MultiIndex (hierarchical), count along a
                particular level, collapsing into a DataFrame.
            numeric_only: Include only float, int, boolean data

        Returns:
            The count, in a Series (or DataFrame if level is specified).
        """
        axis = self._get_axis_number(axis)
        if numeric_only is not None and numeric_only:
            self._validate_dtypes(numeric_only=numeric_only)

        if level is not None:
            if not self._query_compiler.has_multiindex(axis=axis):
                # error thrown by pandas
                raise TypeError("Can only count levels on hierarchical columns.")

            return self._handle_level_agg(axis=axis, level=level, op="count", sort=True)

        return self._reduce_dimension(
            self._query_compiler.count(
                axis=axis, level=level, numeric_only=numeric_only
            )
        )

    def cummax(self, axis=None, skipna=True, *args, **kwargs):
        """Perform a cumulative maximum across the DataFrame.

        Args:
            axis (int): The axis to take maximum on.
            skipna (bool): True to skip NA values, false otherwise.

        Returns:
            The cumulative maximum of the DataFrame.
        """
        axis = self._get_axis_number(axis)
        if axis == 1:
            self._validate_dtypes(numeric_only=True)
        return self.__constructor__(
            query_compiler=self._query_compiler.cummax(
                axis=axis, skipna=skipna, **kwargs
            )
        )

    def cummin(self, axis=None, skipna=True, *args, **kwargs):
        """Perform a cumulative minimum across the DataFrame.

        Args:
            axis (int): The axis to cummin on.
            skipna (bool): True to skip NA values, false otherwise.

        Returns:
            The cumulative minimum of the DataFrame.
        """
        axis = self._get_axis_number(axis)
        if axis == 1:
            self._validate_dtypes(numeric_only=True)
        return self.__constructor__(
            query_compiler=self._query_compiler.cummin(
                axis=axis, skipna=skipna, **kwargs
            )
        )

    def cumprod(self, axis=None, skipna=True, *args, **kwargs):
        """Perform a cumulative product across the DataFrame.

        Args:
            axis (int): The axis to take product on.
            skipna (bool): True to skip NA values, false otherwise.

        Returns:
            The cumulative product of the DataFrame.
        """
        axis = self._get_axis_number(axis)
        self._validate_dtypes(numeric_only=True)
        return self.__constructor__(
            query_compiler=self._query_compiler.cumprod(
                axis=axis, skipna=skipna, **kwargs
            )
        )

    def cumsum(self, axis=None, skipna=True, *args, **kwargs):
        """Perform a cumulative sum across the DataFrame.

        Args:
            axis (int): The axis to take sum on.
            skipna (bool): True to skip NA values, false otherwise.

        Returns:
            The cumulative sum of the DataFrame.
        """
        axis = self._get_axis_number(axis)
        self._validate_dtypes(numeric_only=True)
        return self.__constructor__(
            query_compiler=self._query_compiler.cumsum(
                axis=axis, skipna=skipna, **kwargs
            )
        )

    def describe(
        self, percentiles=None, include=None, exclude=None, datetime_is_numeric=False
    ):
        """
        Generates descriptive statistics that summarize the central tendency,
        dispersion and shape of a dataset's distribution, excluding NaN values.

        Args:
            percentiles (list-like of numbers, optional):
                The percentiles to include in the output.
            include: White-list of data types to include in results
            exclude: Black-list of data types to exclude in results
            datetime_is_numeric : bool, default False
                Whether to treat datetime dtypes as numeric.
                This affects statistics calculated for the column. For DataFrame input,
                this also controls whether datetime columns are included by default.

        Returns: Series/DataFrame of summary statistics
        """
        if include is not None and (isinstance(include, np.dtype) or include != "all"):
            if not is_list_like(include):
                include = [include]
            include = [
                np.dtype(i)
                if not (isinstance(i, type) and i.__module__ == "numpy")
                else i
                for i in include
            ]
            if not any(
                (isinstance(inc, np.dtype) and inc == d)
                or (
                    not isinstance(inc, np.dtype)
                    and inc.__subclasscheck__(getattr(np, d.__str__()))
                )
                for d in self._get_dtypes()
                for inc in include
            ):
                # This is the error that pandas throws.
                raise ValueError("No objects to concatenate")
        if exclude is not None:
            if not is_list_like(exclude):
                exclude = [exclude]
            exclude = [np.dtype(e) for e in exclude]
            if all(
                (isinstance(exc, np.dtype) and exc == d)
                or (
                    not isinstance(exc, np.dtype)
                    and exc.__subclasscheck__(getattr(np, d.__str__()))
                )
                for d in self._get_dtypes()
                for exc in exclude
            ):
                # This is the error that pandas throws.
                raise ValueError("No objects to concatenate")
        if percentiles is not None:
            # explicit conversion of `percentiles` to list
            percentiles = list(percentiles)

            # get them all to be in [0, 1]
            validate_percentile(percentiles)

            # median should always be included
            if 0.5 not in percentiles:
                percentiles.append(0.5)
            percentiles = np.asarray(percentiles)
        else:
            percentiles = np.array([0.25, 0.5, 0.75])
        return self.__constructor__(
            query_compiler=self._query_compiler.describe(
                percentiles=percentiles,
                include=include,
                exclude=exclude,
                datetime_is_numeric=datetime_is_numeric,
            )
        )

    def diff(self, periods=1, axis=0):
        """Finds the difference between elements on the axis requested

        Args:
            periods: Periods to shift for forming difference
            axis: Take difference over rows or columns

        Returns:
            DataFrame with the diff applied
        """
        axis = self._get_axis_number(axis)
        return self.__constructor__(
            query_compiler=self._query_compiler.diff(periods=periods, axis=axis)
        )

    def drop(
        self,
        labels=None,
        axis=0,
        index=None,
        columns=None,
        level=None,
        inplace=False,
        errors="raise",
    ):
        """Return new object with labels in requested axis removed.
        Args:
            labels: Index or column labels to drop.
            axis: Whether to drop labels from the index (0 / 'index') or
                columns (1 / 'columns').
            index, columns: Alternative to specifying axis (labels, axis=1 is
                equivalent to columns=labels).
            level: For MultiIndex
            inplace: If True, do operation inplace and return None.
            errors: If 'ignore', suppress error and existing labels are
                dropped.
        Returns:
            dropped : type of caller
        """
        # TODO implement level
        if level is not None:
            return self._default_to_pandas(
                "drop",
                labels=labels,
                axis=axis,
                index=index,
                columns=columns,
                level=level,
                inplace=inplace,
                errors=errors,
            )

        inplace = validate_bool_kwarg(inplace, "inplace")
        if labels is not None:
            if index is not None or columns is not None:
                raise ValueError("Cannot specify both 'labels' and 'index'/'columns'")
            axis = pandas.DataFrame()._get_axis_name(axis)
            axes = {axis: labels}
        elif index is not None or columns is not None:
            axes, _ = pandas.DataFrame()._construct_axes_from_arguments(
                (index, columns), {}
            )
        else:
            raise ValueError(
                "Need to specify at least one of 'labels', 'index' or 'columns'"
            )

        # TODO Clean up this error checking
        if "index" not in axes:
            axes["index"] = None
        elif axes["index"] is not None:
            if not is_list_like(axes["index"]):
                axes["index"] = [axes["index"]]
            if errors == "raise":
                non_existant = [obj for obj in axes["index"] if obj not in self.index]
                if len(non_existant):
                    raise ValueError(
                        "labels {} not contained in axis".format(non_existant)
                    )
            else:
                axes["index"] = [obj for obj in axes["index"] if obj in self.index]
                # If the length is zero, we will just do nothing
                if not len(axes["index"]):
                    axes["index"] = None

        if "columns" not in axes:
            axes["columns"] = None
        elif axes["columns"] is not None:
            if not is_list_like(axes["columns"]):
                axes["columns"] = [axes["columns"]]
            if errors == "raise":
                non_existant = [
                    obj for obj in axes["columns"] if obj not in self.columns
                ]
                if len(non_existant):
                    raise ValueError(
                        "labels {} not contained in axis".format(non_existant)
                    )
            else:
                axes["columns"] = [
                    obj for obj in axes["columns"] if obj in self.columns
                ]
                # If the length is zero, we will just do nothing
                if not len(axes["columns"]):
                    axes["columns"] = None

        new_query_compiler = self._query_compiler.drop(
            index=axes["index"], columns=axes["columns"]
        )
        return self._create_or_update_from_compiler(new_query_compiler, inplace)

    def dropna(self, axis=0, how="any", thresh=None, subset=None, inplace=False):
        """Create a new DataFrame from the removed NA values from this one.

        Args:
            axis (int, tuple, or list): The axis to apply the drop.
            how (str): How to drop the NA values.
                'all': drop the label if all values are NA.
                'any': drop the label if any values are NA.
            thresh (int): The minimum number of NAs to require.
            subset ([label]): Labels to consider from other axis.
            inplace (bool): Change this DataFrame or return a new DataFrame.
                True: Modify the data for this DataFrame, return None.
                False: Create a new DataFrame and return it.

        Returns:
            If inplace is set to True, returns None, otherwise returns a new
            DataFrame with the dropna applied.
        """
        inplace = validate_bool_kwarg(inplace, "inplace")

        if is_list_like(axis):
            raise TypeError("supplying multiple axes to axis is no longer supported.")

        axis = self._get_axis_number(axis)
        if how is not None and how not in ["any", "all"]:
            raise ValueError("invalid how option: %s" % how)
        if how is None and thresh is None:
            raise TypeError("must specify how or thresh")
        if subset is not None:
            if axis == 1:
                indices = self.index.get_indexer_for(subset)
                check = indices == -1
                if check.any():
                    raise KeyError(list(np.compress(check, subset)))
            else:
                indices = self.columns.get_indexer_for(subset)
                check = indices == -1
                if check.any():
                    raise KeyError(list(np.compress(check, subset)))
        new_query_compiler = self._query_compiler.dropna(
            axis=axis, how=how, thresh=thresh, subset=subset
        )
        return self._create_or_update_from_compiler(new_query_compiler, inplace)

    def droplevel(self, level, axis=0):
        """Return index with requested level(s) removed.

        Args:
            level: The level to drop

        Returns:
            Index or MultiIndex
        """
        axis = self._get_axis_number(axis)
        new_axis = self.axes[axis].droplevel(level)
        result = self.copy()
        if axis == 0:
            result.index = new_axis
        else:
            result.columns = new_axis
        return result

    def drop_duplicates(self, keep="first", inplace=False, **kwargs):
        """Return DataFrame with duplicate rows removed, optionally only considering certain columns

        Args:
            subset : column label or sequence of labels, optional
                Only consider certain columns for identifying duplicates, by
                default use all of the columns
            keep : {'first', 'last', False}, default 'first'
                - ``first`` : Drop duplicates except for the first occurrence.
                - ``last`` : Drop duplicates except for the last occurrence.
                - False : Drop all duplicates.
            inplace : boolean, default False
                Whether to drop duplicates in place or to return a copy

        Returns:
            deduplicated : DataFrame
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        subset = kwargs.get("subset", None)
        if subset is not None:
            if is_list_like(subset):
                if not isinstance(subset, list):
                    subset = list(subset)
            else:
                subset = [subset]
            duplicates = self.duplicated(keep=keep, subset=subset)
        else:
            duplicates = self.duplicated(keep=keep)
        indices = duplicates.values.nonzero()[0]
        return self.drop(index=self.index[indices], inplace=inplace)

    def eq(self, other, axis="columns", level=None):
        """Checks element-wise that this is equal to other.

        Args:
            other: A DataFrame or Series or scalar to compare to.
            axis: The axis to perform the eq over.
            level: The Multilevel index level to apply eq over.

        Returns:
            A new DataFrame filled with Booleans.
        """
        return self._binary_op("eq", other, axis=axis, level=level)

    def ewm(
        self,
        com=None,
        span=None,
        halflife=None,
        alpha=None,
        min_periods=0,
        adjust=True,
        ignore_na=False,
        axis=0,
        times=None,
    ):
        return self._default_to_pandas(
            "ewm",
            com=com,
            span=span,
            halflife=halflife,
            alpha=alpha,
            min_periods=min_periods,
            adjust=adjust,
            ignore_na=ignore_na,
            axis=axis,
            times=times,
        )

    def expanding(self, min_periods=1, center=None, axis=0):
        return self._default_to_pandas(
            "expanding", min_periods=min_periods, center=center, axis=axis
        )

    def ffill(self, axis=None, inplace=False, limit=None, downcast=None):
        """Synonym for fillna(method='ffill')"""
        return self.fillna(
            method="ffill", axis=axis, limit=limit, downcast=downcast, inplace=inplace
        )

    pad = ffill

    def fillna(
        self,
        value=None,
        method=None,
        axis=None,
        inplace=False,
        limit=None,
        downcast=None,
    ):
        """Fill NA/NaN values using the specified method.

        Args:
            value: Value to use to fill holes. This value cannot be a list.

            method: Method to use for filling holes in reindexed Series pad.
                ffill: propagate last valid observation forward to next valid
                backfill.
                bfill: use NEXT valid observation to fill gap.

            axis: 0 or 'index', 1 or 'columns'.

            inplace: If True, fill in place. Note: this will modify any other
                views on this object.

            limit: If method is specified, this is the maximum number of
                consecutive NaN values to forward/backward fill. In other
                words, if there is a gap with more than this number of
                consecutive NaNs, it will only be partially filled. If method
                is not specified, this is the maximum number of entries along
                the entire axis where NaNs will be filled. Must be greater
                than 0 if not None.

            downcast: A dict of item->dtype of what to downcast if possible,
                or the string 'infer' which will try to downcast to an
                appropriate equal type.

        Returns:
            filled: DataFrame
        """
        # TODO implement value passed as DataFrame/Series
        if isinstance(value, BasePandasDataset):
            new_query_compiler = self._default_to_pandas(
                "fillna",
                value=value,
                method=method,
                axis=axis,
                inplace=False,
                limit=limit,
                downcast=downcast,
            )._query_compiler
            return self._create_or_update_from_compiler(new_query_compiler, inplace)
        inplace = validate_bool_kwarg(inplace, "inplace")
        axis = self._get_axis_number(axis)
        if isinstance(value, (list, tuple)):
            raise TypeError(
                '"value" parameter must be a scalar or dict, but '
                'you passed a "{0}"'.format(type(value).__name__)
            )
        if value is None and method is None:
            raise ValueError("must specify a fill method or value")
        if value is not None and method is not None:
            raise ValueError("cannot specify both a fill method and value")
        if method is not None and method not in ["backfill", "bfill", "pad", "ffill"]:
            expecting = "pad (ffill) or backfill (bfill)"
            msg = "Invalid fill method. Expecting {expecting}. Got {method}".format(
                expecting=expecting, method=method
            )
            raise ValueError(msg)
        if limit is not None:
            if not isinstance(limit, int):
                raise ValueError("Limit must be an integer")
            elif limit <= 0:
                raise ValueError("Limit must be greater than 0")

        new_query_compiler = self._query_compiler.fillna(
            value=value,
            method=method,
            axis=axis,
            inplace=False,
            limit=limit,
            downcast=downcast,
        )
        return self._create_or_update_from_compiler(new_query_compiler, inplace)

    def filter(self, items=None, like=None, regex=None, axis=None):
        """Subset rows or columns based on their labels

        Args:
            items (list): list of labels to subset
            like (string): retain labels where `arg in label == True`
            regex (string): retain labels matching regex input
            axis: axis to filter on

        Returns:
            A new DataFrame with the filter applied.
        """
        nkw = count_not_none(items, like, regex)
        if nkw > 1:
            raise TypeError(
                "Keyword arguments `items`, `like`, or `regex` are mutually exclusive"
            )
        if nkw == 0:
            raise TypeError("Must pass either `items`, `like`, or `regex`")
        if axis is None:
            axis = "columns"  # This is the default info axis for dataframes

        axis = self._get_axis_number(axis)
        labels = self.columns if axis else self.index

        if items is not None:
            bool_arr = labels.isin(items)
        elif like is not None:

            def f(x):
                return like in str(x)

            bool_arr = labels.map(f).tolist()
        else:

            def f(x):
                return matcher.search(str(x)) is not None

            matcher = re.compile(regex)
            bool_arr = labels.map(f).tolist()
        if not axis:
            return self[bool_arr]
        return self[self.columns[bool_arr]]

    def first(self, offset):
        return self.loc[pandas.Series(index=self.index).first(offset).index]

    def first_valid_index(self):
        """Return index for first non-NA/null value.

        Returns:
            scalar: type of index
        """
        return self._query_compiler.first_valid_index()

    def floordiv(self, other, axis="columns", level=None, fill_value=None):
        """Divides this DataFrame against another DataFrame/Series/scalar.

        Args:
            other: The object to use to apply the divide against this.
            axis: The axis to divide over.
            level: The Multilevel index level to apply divide over.
            fill_value: The value to fill NaNs with.

        Returns:
            A new DataFrame with the Divide applied.
        """
        return self._binary_op(
            "floordiv", other, axis=axis, level=level, fill_value=fill_value
        )

    def ge(self, other, axis="columns", level=None):
        """Checks element-wise that this is greater than or equal to other.

        Args:
            other: A DataFrame or Series or scalar to compare to.
            axis: The axis to perform the gt over.
            level: The Multilevel index level to apply gt over.

        Returns:
            A new DataFrame filled with Booleans.
        """
        return self._binary_op("ge", other, axis=axis, level=level)

    def get(self, key, default=None):
        """Get item from object for given key (DataFrame column, Panel
        slice, etc.). Returns default value if not found.

        Args:
            key (DataFrame column, Panel slice) : the key for which value
            to get

        Returns:
            value (type of items contained in object) : A value that is
            stored at the key
        """
        if key in self.keys():
            return self.__getitem__(key)
        else:
            return default

    def gt(self, other, axis="columns", level=None):
        """Checks element-wise that this is greater than other.

        Args:
            other: A DataFrame or Series or scalar to compare to.
            axis: The axis to perform the gt over.
            level: The Multilevel index level to apply gt over.

        Returns:
            A new DataFrame filled with Booleans.
        """
        return self._binary_op("gt", other, axis=axis, level=level)

    def head(self, n=5):
        """Get the first n rows of the DataFrame or Series.

        Parameters
        ----------
        n : int
            The number of rows to return.

        Returns
        -------
        DataFrame or Series
            A new DataFrame or Series with the first n rows of the DataFrame or Series.
        """
        return self.iloc[:n]

    @property
    def iat(self, axis=None):
        from .indexing import _iLocIndexer

        return _iLocIndexer(self)

    def idxmax(self, axis=0, skipna=True):
        """Get the index of the first occurrence of the max value of the axis.

        Args:
            axis (int): Identify the max over the rows (1) or columns (0).
            skipna (bool): Whether or not to skip NA values.

        Returns:
            A Series with the index for each maximum value for the axis
                specified.
        """
        if not all(d != np.dtype("O") for d in self._get_dtypes()):
            raise TypeError("reduction operation 'argmax' not allowed for this dtype")
        axis = self._get_axis_number(axis)
        return self._reduce_dimension(
            self._query_compiler.idxmax(axis=axis, skipna=skipna)
        )

    def idxmin(self, axis=0, skipna=True):
        """Get the index of the first occurrence of the min value of the axis.

        Args:
            axis (int): Identify the min over the rows (1) or columns (0).
            skipna (bool): Whether or not to skip NA values.

        Returns:
            A Series with the index for each minimum value for the axis
                specified.
        """
        if not all(d != np.dtype("O") for d in self._get_dtypes()):
            raise TypeError("reduction operation 'argmin' not allowed for this dtype")
        axis = self._get_axis_number(axis)
        return self._reduce_dimension(
            self._query_compiler.idxmin(axis=axis, skipna=skipna)
        )

    def infer_objects(self):
        return self._default_to_pandas("infer_objects")

    def convert_dtypes(
        self,
        infer_objects: bool = True,
        convert_string: bool = True,
        convert_integer: bool = True,
        convert_boolean: bool = True,
    ):
        return self._default_to_pandas(
            "convert_dtypes",
            infer_objects=infer_objects,
            convert_string=convert_string,
            convert_integer=convert_integer,
            convert_boolean=convert_boolean,
        )

    def isin(self, values):
        """Fill a DataFrame with booleans for cells contained in values.

        Args:
            values (iterable, DataFrame, Series, or dict): The values to find.

        Returns:
            A new DataFrame with booleans representing whether or not a cell
            is in values.
            True: cell is contained in values.
            False: otherwise
        """
        return self.__constructor__(
            query_compiler=self._query_compiler.isin(values=values)
        )

    def isna(self):
        """Fill a DataFrame with booleans for cells containing NA.

        Returns:
            A new DataFrame with booleans representing whether or not a cell
            is NA.
            True: cell contains NA.
            False: otherwise.
        """
        return self.__constructor__(query_compiler=self._query_compiler.isna())

    isnull = isna

    @property
    def iloc(self):
        """Purely integer-location based indexing for selection by position.

        We currently support: single label, list array, slice object
        We do not support: boolean array, callable
        """
        from .indexing import _iLocIndexer

        return _iLocIndexer(self)

    def kurt(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        """Return unbiased kurtosis over requested axis. Normalized by N-1

        Kurtosis obtained using Fisher’s definition of kurtosis (kurtosis of normal == 0.0).

        Args:
            axis : {index (0), columns (1)}
            skipna : boolean, default True
                Exclude NA/null values when computing the result.
            level : int or level name, default None
            numeric_only : boolean, default None

        Returns:
            kurtosis : Series or DataFrame (if level specified)
        """
        axis = self._get_axis_number(axis)
        if level is not None:
            func_kwargs = {
                "skipna": skipna,
                "level": level,
                "numeric_only": numeric_only,
            }

            return self.__constructor__(
                query_compiler=self._query_compiler.apply("kurt", axis, **func_kwargs)
            )

        if numeric_only:
            self._validate_dtypes(numeric_only=True)
        return self._reduce_dimension(
            self._query_compiler.kurt(
                axis=axis,
                skipna=skipna,
                level=level,
                numeric_only=numeric_only,
                **kwargs,
            )
        )

    kurtosis = kurt

    def last(self, offset):
        return self.loc[pandas.Series(index=self.index).last(offset).index]

    def last_valid_index(self):
        """Return index for last non-NA/null value.

        Returns:
            scalar: type of index
        """
        return self._query_compiler.last_valid_index()

    def le(self, other, axis="columns", level=None):
        """Checks element-wise that this is less than or equal to other.

        Args:
            other: A DataFrame or Series or scalar to compare to.
            axis: The axis to perform the le over.
            level: The Multilevel index level to apply le over.

        Returns:
            A new DataFrame filled with Booleans.
        """
        return self._binary_op("le", other, axis=axis, level=level)

    def lt(self, other, axis="columns", level=None):
        """Checks element-wise that this is less than other.

        Args:
            other: A DataFrame or Series or scalar to compare to.
            axis: The axis to perform the lt over.
            level: The Multilevel index level to apply lt over.

        Returns:
            A new DataFrame filled with Booleans.
        """
        return self._binary_op("lt", other, axis=axis, level=level)

    @property
    def loc(self):
        """Purely label-location based indexer for selection by label.

        We currently support: single label, list array, slice object
        We do not support: boolean array, callable
        """
        from .indexing import _LocIndexer

        return _LocIndexer(self)

    def mad(self, axis=None, skipna=None, level=None):
        """
        Return the mean absolute deviation of the values for the requested axis.

        Parameters
        ----------
            axis : {index (0), columns (1)}
            skipna : bool, default True
                Exclude NA/null values when computing the result.
            level : int or level name, default None

        Returns
        -------
            Series or DataFrame (if level specified)

        """
        axis = self._get_axis_number(axis)

        if level is not None:
            return self._handle_level_agg(
                axis=axis, level=level, skipna=skipna, op="mad"
            )

        return self._reduce_dimension(
            self._query_compiler.mad(axis=axis, skipna=skipna, level=level)
        )

    def mask(
        self,
        cond,
        other=nan,
        inplace=False,
        axis=None,
        level=None,
        errors="raise",
        try_cast=False,
    ):
        return self._default_to_pandas(
            "mask",
            cond,
            other=other,
            inplace=inplace,
            axis=axis,
            level=level,
            errors=errors,
            try_cast=try_cast,
        )

    def max(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        """Perform max across the DataFrame.

        Args:
            axis (int): The axis to take the max on.
            skipna (bool): True to skip NA values, false otherwise.

        Returns:
            The max of the DataFrame.
        """
        axis = self._get_axis_number(axis)
        data = self._validate_dtypes_min_max(axis, numeric_only)
        return data._reduce_dimension(
            data._query_compiler.max(
                axis=axis,
                skipna=skipna,
                level=level,
                numeric_only=numeric_only,
                **kwargs,
            )
        )

    def mean(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        """Computes mean across the DataFrame.

        Args:
            axis (int): The axis to take the mean on.
            skipna (bool): True to skip NA values, false otherwise.

        Returns:
            The mean of the DataFrame. (Pandas series)
        """
        axis = self._get_axis_number(axis)
        data = self._validate_dtypes_sum_prod_mean(
            axis, numeric_only, ignore_axis=False
        )
        return data._reduce_dimension(
            data._query_compiler.mean(
                axis=axis,
                skipna=skipna,
                level=level,
                numeric_only=numeric_only,
                **kwargs,
            )
        )

    def memory_usage(self, index=True, deep=False):
        """Returns the memory usage of each column in bytes

        Args:
            index (bool): Whether to include the memory usage of the DataFrame's
                index in returned Series. Defaults to True
            deep (bool): If True, introspect the data deeply by interrogating
            objects dtypes for system-level memory consumption. Defaults to False

        Returns:
            A Series where the index are the column names and the values are
            the memory usage of each of the columns in bytes. If `index=true`,
            then the first value of the Series will be 'Index' with its memory usage.
        """
        return self._reduce_dimension(
            self._query_compiler.memory_usage(index=index, deep=deep)
        )

    def min(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        """Perform min across the DataFrame.

        Args:
            axis (int): The axis to take the min on.
            skipna (bool): True to skip NA values, false otherwise.

        Returns:
            The min of the DataFrame.
        """
        axis = self._get_axis_number(axis)
        data = self._validate_dtypes_min_max(axis, numeric_only)
        return data._reduce_dimension(
            data._query_compiler.min(
                axis=axis,
                skipna=skipna,
                level=level,
                numeric_only=numeric_only,
                **kwargs,
            )
        )

    def mod(self, other, axis="columns", level=None, fill_value=None):
        """Mods this DataFrame against another DataFrame/Series/scalar.

        Args:
            other: The object to use to apply the mod against this.
            axis: The axis to mod over.
            level: The Multilevel index level to apply mod over.
            fill_value: The value to fill NaNs with.

        Returns:
            A new DataFrame with the Mod applied.
        """
        return self._binary_op(
            "mod", other, axis=axis, level=level, fill_value=fill_value
        )

    def mode(self, axis=0, numeric_only=False, dropna=True):
        """Perform mode across the DataFrame.

        Args:
            axis (int): The axis to take the mode on.
            numeric_only (bool): if True, only apply to numeric columns.

        Returns:
            DataFrame: The mode of the DataFrame.
        """
        axis = self._get_axis_number(axis)
        return self.__constructor__(
            query_compiler=self._query_compiler.mode(
                axis=axis, numeric_only=numeric_only, dropna=dropna
            )
        )

    def mul(self, other, axis="columns", level=None, fill_value=None):
        """Multiplies this DataFrame against another DataFrame/Series/scalar.

        Args:
            other: The object to use to apply the multiply against this.
            axis: The axis to multiply over.
            level: The Multilevel index level to apply multiply over.
            fill_value: The value to fill NaNs with.

        Returns:
            A new DataFrame with the Multiply applied.
        """
        return self._binary_op(
            "mul", other, axis=axis, level=level, fill_value=fill_value
        )

    multiply = mul

    def ne(self, other, axis="columns", level=None):
        """Checks element-wise that this is not equal to other.

        Args:
            other: A DataFrame or Series or scalar to compare to.
            axis: The axis to perform the ne over.
            level: The Multilevel index level to apply ne over.

        Returns:
            A new DataFrame filled with Booleans.
        """
        return self._binary_op("ne", other, axis=axis, level=level)

    def notna(self):
        """Perform notna across the DataFrame.

        Returns:
            Boolean DataFrame where value is False if corresponding
            value is NaN, True otherwise
        """
        return self.__constructor__(query_compiler=self._query_compiler.notna())

    notnull = notna

    def nunique(self, axis=0, dropna=True):
        """Return Series with number of distinct
           observations over requested axis.

        Args:
            axis : {0 or 'index', 1 or 'columns'}, default 0
            dropna : boolean, default True

        Returns:
            nunique : Series
        """
        axis = self._get_axis_number(axis)
        return self._reduce_dimension(
            self._query_compiler.nunique(axis=axis, dropna=dropna)
        )

    def pct_change(self, periods=1, fill_method="pad", limit=None, freq=None, **kwargs):
        return self._default_to_pandas(
            "pct_change",
            periods=periods,
            fill_method=fill_method,
            limit=limit,
            freq=freq,
            **kwargs,
        )

    def pipe(self, func, *args, **kwargs):
        """Apply func(self, *args, **kwargs)

        Args:
            func: function to apply to the df.
            args: positional arguments passed into ``func``.
            kwargs: a dictionary of keyword arguments passed into ``func``.

        Returns:
            object: the return type of ``func``.
        """
        return pipe(self, func, *args, **kwargs)

    def pop(self, item):
        """Pops an item from this DataFrame and returns it.

        Args:
            item (str): Column label to be popped

        Returns:
            A Series containing the popped values. Also modifies this
            DataFrame.
        """
        result = self[item]
        del self[item]
        return result

    def pow(self, other, axis="columns", level=None, fill_value=None):
        """Pow this DataFrame against another DataFrame/Series/scalar.

        Args:
            other: The object to use to apply the pow against this.
            axis: The axis to pow over.
            level: The Multilevel index level to apply pow over.
            fill_value: The value to fill NaNs with.

        Returns:
            A new DataFrame with the Pow applied.
        """
        return self._binary_op(
            "pow", other, axis=axis, level=level, fill_value=fill_value
        )

    radd = add

    def quantile(self, q=0.5, axis=0, numeric_only=True, interpolation="linear"):
        """Return values at the given quantile over requested axis,
            a la numpy.percentile.

        Args:
            q (float): 0 <= q <= 1, the quantile(s) to compute
            axis (int): 0 or 'index' for row-wise,
                        1 or 'columns' for column-wise
            interpolation: {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
                Specifies which interpolation method to use

        Returns:
            quantiles : Series or DataFrame
                    If q is an array, a DataFrame will be returned where the
                    index is q, the columns are the columns of self, and the
                    values are the quantiles.

                    If q is a float, a Series will be returned where the
                    index is the columns of self and the values
                    are the quantiles.
        """
        axis = self._get_axis_number(axis)

        def check_dtype(t):
            return is_numeric_dtype(t) or is_datetime_or_timedelta_dtype(t)

        if not numeric_only:
            # If not numeric_only and columns, then check all columns are either
            # numeric, timestamp, or timedelta
            if not axis and not all(check_dtype(t) for t in self._get_dtypes()):
                raise TypeError("can't multiply sequence by non-int of type 'float'")
            # If over rows, then make sure that all dtypes are equal for not
            # numeric_only
            elif axis:
                for i in range(1, len(self._get_dtypes())):
                    pre_dtype = self._get_dtypes()[i - 1]
                    curr_dtype = self._get_dtypes()[i]
                    if not is_dtype_equal(pre_dtype, curr_dtype):
                        raise TypeError(
                            "Cannot compare type '{0}' with type '{1}'".format(
                                pre_dtype, curr_dtype
                            )
                        )
        else:
            # Normally pandas returns this near the end of the quantile, but we
            # can't afford the overhead of running the entire operation before
            # we error.
            if not any(is_numeric_dtype(t) for t in self._get_dtypes()):
                raise ValueError("need at least one array to concatenate")

        # check that all qs are between 0 and 1
        validate_percentile(q)
        axis = self._get_axis_number(axis)

        if isinstance(q, (pandas.Series, np.ndarray, pandas.Index, list)):
            return self.__constructor__(
                query_compiler=self._query_compiler.quantile_for_list_of_values(
                    q=q,
                    axis=axis,
                    numeric_only=numeric_only,
                    interpolation=interpolation,
                )
            )
        else:
            result = self._reduce_dimension(
                self._query_compiler.quantile_for_single_value(
                    q=q,
                    axis=axis,
                    numeric_only=numeric_only,
                    interpolation=interpolation,
                )
            )
            if isinstance(result, BasePandasDataset):
                result.name = q
            return result

    def rank(
        self,
        axis=0,
        method="average",
        numeric_only=None,
        na_option="keep",
        ascending=True,
        pct=False,
    ):
        """
        Compute numerical data ranks (1 through n) along axis.
        Equal values are assigned a rank that is the [method] of
        the ranks of those values.

        Args:
            axis (int): 0 or 'index' for row-wise,
                        1 or 'columns' for column-wise
            method: {'average', 'min', 'max', 'first', 'dense'}
                Specifies which method to use for equal vals
            numeric_only (boolean)
                Include only float, int, boolean data.
            na_option: {'keep', 'top', 'bottom'}
                Specifies how to handle NA options
            ascending (boolean):
                Decedes ranking order
            pct (boolean):
                Computes percentage ranking of data
        Returns:
            A new DataFrame
        """
        axis = self._get_axis_number(axis)
        return self.__constructor__(
            query_compiler=self._query_compiler.rank(
                axis=axis,
                method=method,
                numeric_only=numeric_only,
                na_option=na_option,
                ascending=ascending,
                pct=pct,
            )
        )

    def reindex(
        self,
        labels=None,
        index=None,
        columns=None,
        axis=None,
        method=None,
        copy=True,
        level=None,
        fill_value=np.nan,
        limit=None,
        tolerance=None,
    ):
        axis = self._get_axis_number(axis)
        if (
            level is not None
            or (
                (columns is not None or axis == 1)
                and self._query_compiler.has_multiindex(axis=1)
            )
            or (
                (index is not None or axis == 0)
                and self._query_compiler.has_multiindex()
            )
        ):
            return self._default_to_pandas(
                "reindex",
                labels=labels,
                index=index,
                columns=columns,
                axis=axis,
                method=method,
                copy=copy,
                level=level,
                fill_value=fill_value,
                limit=limit,
                tolerance=tolerance,
            )
        if axis == 0 and labels is not None:
            index = labels
        elif labels is not None:
            columns = labels
        new_query_compiler = None
        if index is not None:
            if not isinstance(index, pandas.Index):
                index = pandas.Index(index)
            if not index.equals(self.index):
                new_query_compiler = self._query_compiler.reindex(
                    axis=0,
                    labels=index,
                    method=method,
                    fill_value=fill_value,
                    limit=limit,
                    tolerance=tolerance,
                )
        if new_query_compiler is None:
            new_query_compiler = self._query_compiler
        final_query_compiler = None
        if columns is not None:
            if not isinstance(columns, pandas.Index):
                columns = pandas.Index(columns)
            if not columns.equals(self.columns):
                final_query_compiler = new_query_compiler.reindex(
                    axis=1,
                    labels=columns,
                    method=method,
                    fill_value=fill_value,
                    limit=limit,
                    tolerance=tolerance,
                )
        if final_query_compiler is None:
            final_query_compiler = new_query_compiler
        return self._create_or_update_from_compiler(final_query_compiler, not copy)

    def reindex_like(self, other, method=None, copy=True, limit=None, tolerance=None):
        return self._default_to_pandas(
            "reindex_like",
            other,
            method=method,
            copy=copy,
            limit=limit,
            tolerance=tolerance,
        )

    def rename_axis(
        self, mapper=None, index=None, columns=None, axis=None, copy=True, inplace=False
    ):
        kwargs = {
            "index": index,
            "columns": columns,
            "axis": axis,
            "copy": copy,
            "inplace": inplace,
        }
        axes, kwargs = getattr(
            pandas, type(self).__name__
        )()._construct_axes_from_arguments((), kwargs, sentinel=sentinel)
        if axis is not None:
            axis = self._get_axis_number(axis)
        else:
            axis = 0
        inplace = validate_bool_kwarg(inplace, "inplace")

        if mapper is not None:
            # Use v0.23 behavior if a scalar or list
            non_mapper = is_scalar(mapper) or (
                is_list_like(mapper) and not is_dict_like(mapper)
            )
            if non_mapper:
                return self._set_axis_name(mapper, axis=axis, inplace=inplace)
            else:
                raise ValueError("Use `.rename` to alter labels " "with a mapper.")
        else:
            # Use new behavior.  Means that index and/or columns is specified
            result = self if inplace else self.copy(deep=copy)

            for axis in axes:
                if axes[axis] is None:
                    continue
                v = axes[axis]
                axis = self._get_axis_number(axis)
                non_mapper = is_scalar(v) or (is_list_like(v) and not is_dict_like(v))
                if non_mapper:
                    newnames = v
                else:

                    def _get_rename_function(mapper):
                        if isinstance(mapper, (dict, BasePandasDataset)):

                            def f(x):
                                if x in mapper:
                                    return mapper[x]
                                else:
                                    return x

                        else:
                            f = mapper

                        return f

                    f = _get_rename_function(v)
                    curnames = self.index.names if axis == 0 else self.columns.names
                    newnames = [f(name) for name in curnames]
                result._set_axis_name(newnames, axis=axis, inplace=True)
            if not inplace:
                return result

    def reorder_levels(self, order, axis=0):
        axis = self._get_axis_number(axis)
        new_labels = self.axes[axis].reorder_levels(order)
        return self.set_axis(new_labels, axis=axis, inplace=False)

    def resample(
        self,
        rule,
        axis=0,
        closed=None,
        label=None,
        convention="start",
        kind=None,
        loffset=None,
        base: Optional[int] = None,
        on=None,
        level=None,
        origin: Union[str, TimestampConvertibleTypes] = "start_day",
        offset: Optional[TimedeltaConvertibleTypes] = None,
    ):
        """
        Resample time-series data.
        Convenience method for frequency conversion and resampling of time
        series. Object must have a datetime-like index (`DatetimeIndex`,
        `PeriodIndex`, or `TimedeltaIndex`), or pass datetime-like values
        to the `on` or `level` keyword.

        Parameters
        ----------
        rule : DateOffset, Timedelta or str
            The offset string or object representing target conversion.
        axis : {0 or 'index', 1 or 'columns'}, default 0
            Which axis to use for up- or down-sampling. For `Series` this
            will default to 0, i.e. along the rows. Must be
            `DatetimeIndex`, `TimedeltaIndex` or `PeriodIndex`.
        closed : {'right', 'left'}, default None
            Which side of bin interval is closed. The default is 'left'
            for all frequency offsets except for 'M', 'A', 'Q', 'BM',
            'BA', 'BQ', and 'W' which all have a default of 'right'.
        label : {'right', 'left'}, default None
            Which bin edge label to label bucket with. The default is 'left'
            for all frequency offsets except for 'M', 'A', 'Q', 'BM',
            'BA', 'BQ', and 'W' which all have a default of 'right'.
        convention : {'start', 'end', 's', 'e'}, default 'start'
            For `PeriodIndex` only, controls whether to use the start or
            end of `rule`.
        kind : {'timestamp', 'period'}, optional, default None
            Pass 'timestamp' to convert the resulting index to a
            `DateTimeIndex` or 'period' to convert it to a `PeriodIndex`.
            By default the input representation is retained.
        loffset : timedelta, default None
            Adjust the resampled time labels.
        base : int, default 0
            For frequencies that evenly subdivide 1 day, the "origin" of the
            aggregated intervals. For example, for '5min' frequency, base could
            range from 0 through 4. Defaults to 0.
        on : str, optional
            For a DataFrame, column to use instead of index for resampling.
            Column must be datetime-like.
        level : str or int, optional
            For a MultiIndex, level (name or number) to use for
            resampling. `level` must be datetime-like.
        origin : {'epoch', 'start', 'start_day'}, Timestamp or str, default 'start_day'
            The timestamp on which to adjust the grouping. The timezone of origin must match
            the timezone of the index. If a timestamp is not used, these values are also supported:

            - 'epoch': origin is 1970-01-01
            - 'start': origin is the first value of the timeseries
            - 'start_day': origin is the first day at midnight of the timeseries

        offset : Timedelta or str, default is None
            An offset timedelta added to the origin.

        Returns
        -------
        Resampler object
        """
        return Resampler(
            self,
            rule=rule,
            axis=axis,
            closed=closed,
            label=label,
            convention=convention,
            kind=kind,
            loffset=loffset,
            base=base,
            on=on,
            level=level,
            origin=origin,
            offset=offset,
        )

    def reset_index(
        self, level=None, drop=False, inplace=False, col_level=0, col_fill=""
    ):
        """Reset this index to default and create column from current index.

        Args:
            level: Only remove the given levels from the index. Removes all
                levels by default
            drop: Do not try to insert index into DataFrame columns. This
                resets the index to the default integer index.
            inplace: Modify the DataFrame in place (do not create a new object)
            col_level : If the columns have multiple levels, determines which
                level the labels are inserted into. By default it is inserted
                into the first level.
            col_fill: If the columns have multiple levels, determines how the
                other levels are named. If None then the index name is
                repeated.

        Returns:
            A new DataFrame if inplace is False, None otherwise.
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        # Error checking for matching Pandas. Pandas does not allow you to
        # insert a dropped index into a DataFrame if these columns already
        # exist.
        if (
            not drop
            and not self._query_compiler.has_multiindex()
            and all(n in self.columns for n in ["level_0", "index"])
        ):
            raise ValueError("cannot insert level_0, already exists")
        else:
            new_query_compiler = self._query_compiler.reset_index(
                drop=drop, level=level
            )
        return self._create_or_update_from_compiler(new_query_compiler, inplace)

    def rfloordiv(self, other, axis="columns", level=None, fill_value=None):
        return self._binary_op(
            "rfloordiv", other, axis=axis, level=level, fill_value=fill_value
        )

    def rmod(self, other, axis="columns", level=None, fill_value=None):
        """Mod this DataFrame against another DataFrame/Series/scalar.

        Args:
            other: The object to use to apply the div against this.
            axis: The axis to div over.
            level: The Multilevel index level to apply div over.
            fill_value: The value to fill NaNs with.

        Returns:
            A new DataFrame with the rdiv applied.
        """
        return self._binary_op(
            "rmod", other, axis=axis, level=level, fill_value=fill_value
        )

    rmul = mul

    def rolling(
        self,
        window,
        min_periods=None,
        center=False,
        win_type=None,
        on=None,
        axis=0,
        closed=None,
    ):
        """
        Provide rolling window calculations.

        Parameters
        ----------
        window : int, offset, or BaseIndexer subclass
            Size of the moving window. This is the number of observations used for
            calculating the statistic. Each window will be a fixed size.
            If its an offset then this will be the time period of each window. Each
            window will be a variable sized based on the observations included in
            the time-period. This is only valid for datetimelike indexes.
            If a BaseIndexer subclass is passed, calculates the window boundaries
            based on the defined ``get_window_bounds`` method. Additional rolling
            keyword arguments, namely `min_periods`, `center`, and
            `closed` will be passed to `get_window_bounds`.
        min_periods : int, default None
            Minimum number of observations in window required to have a value
            (otherwise result is NA). For a window that is specified by an offset,
            `min_periods` will default to 1. Otherwise, `min_periods` will default
            to the size of the window.
        center : bool, default False
            Set the labels at the center of the window.
        win_type : str, default None
            Provide a window type. If ``None``, all points are evenly weighted.
            See the notes below for further information.
        on : str, optional
            For a DataFrame, a datetime-like column or MultiIndex level on which
            to calculate the rolling window, rather than the DataFrame's index.
            Provided integer column is ignored and excluded from result since
            an integer index is not used to calculate the rolling window.
        axis : int or str, default 0
        closed : str, default None
            Make the interval closed on the 'right', 'left', 'both' or
            'neither' endpoints.
            For offset-based windows, it defaults to 'right'.
            For fixed windows, defaults to 'both'. Remaining cases not implemented
            for fixed windows.
        Returns
        -------
        a Window or Rolling sub-classed for the particular operation
        """
        if win_type is not None:
            return Window(
                self,
                window=window,
                min_periods=min_periods,
                center=center,
                win_type=win_type,
                on=on,
                axis=axis,
                closed=closed,
            )

        return Rolling(
            self,
            window=window,
            min_periods=min_periods,
            center=center,
            win_type=win_type,
            on=on,
            axis=axis,
            closed=closed,
        )

    def round(self, decimals=0, *args, **kwargs):
        """Round each element in the DataFrame.

        Args:
            decimals: The number of decimals to round to.

        Returns:
             A new DataFrame.
        """
        return self.__constructor__(
            query_compiler=self._query_compiler.round(decimals=decimals, **kwargs)
        )

    def rpow(self, other, axis="columns", level=None, fill_value=None):
        """Pow this DataFrame against another DataFrame/Series/scalar.

        Args:
            other: The object to use to apply the pow against this.
            axis: The axis to pow over.
            level: The Multilevel index level to apply pow over.
            fill_value: The value to fill NaNs with.

        Returns:
            A new DataFrame with the Pow applied.
        """
        return self._binary_op(
            "rpow", other, axis=axis, level=level, fill_value=fill_value
        )

    def rsub(self, other, axis="columns", level=None, fill_value=None):
        """Subtract a DataFrame/Series/scalar from this DataFrame.

        Args:
            other: The object to use to apply the subtraction to this.
            axis: The axis to apply the subtraction over.
            level: Mutlilevel index level to subtract over.
            fill_value: The value to fill NaNs with.

        Returns:
             A new DataFrame with the subtraciont applied.
        """
        return self._binary_op(
            "rsub", other, axis=axis, level=level, fill_value=fill_value
        )

    def rtruediv(self, other, axis="columns", level=None, fill_value=None):
        """Div this DataFrame against another DataFrame/Series/scalar.

        Args:
            other: The object to use to apply the div against this.
            axis: The axis to div over.
            level: The Multilevel index level to apply div over.
            fill_value: The value to fill NaNs with.

        Returns:
            A new DataFrame with the rdiv applied.
        """
        return self._binary_op(
            "rtruediv", other, axis=axis, level=level, fill_value=fill_value
        )

    rdiv = rtruediv

    def sample(
        self,
        n=None,
        frac=None,
        replace=False,
        weights=None,
        random_state=None,
        axis=None,
    ):
        """Returns a random sample of items from an axis of object.

        Args:
            n: Number of items from axis to return. Cannot be used with frac.
                Default = 1 if frac = None.
            frac: Fraction of axis items to return. Cannot be used with n.
            replace: Sample with or without replacement. Default = False.
            weights: Default 'None' results in equal probability weighting.
                If passed a Series, will align with target object on index.
                Index values in weights not found in sampled object will be
                ignored and index values in sampled object not in weights will
                be assigned weights of zero. If called on a DataFrame, will
                accept the name of a column when axis = 0. Unless weights are
                a Series, weights must be same length as axis being sampled.
                If weights do not sum to 1, they will be normalized to sum
                to 1. Missing values in the weights column will be treated as
                zero. inf and -inf values not allowed.
            random_state: Seed for the random number generator (if int), or
                numpy RandomState object.
            axis: Axis to sample. Accepts axis number or name.

        Returns:
            A new Dataframe
        """
        axis = self._get_axis_number(axis)
        if axis:
            axis_labels = self.columns
            axis_length = len(axis_labels)
        else:
            # Getting rows requires indices instead of labels. RangeIndex provides this.
            axis_labels = pandas.RangeIndex(len(self.index))
            axis_length = len(axis_labels)
        if weights is not None:
            # Index of the weights Series should correspond to the index of the
            # Dataframe in order to sample
            if isinstance(weights, BasePandasDataset):
                weights = weights.reindex(self.axes[axis])
            # If weights arg is a string, the weights used for sampling will
            # the be values in the column corresponding to that string
            if isinstance(weights, str):
                if axis == 0:
                    try:
                        weights = self[weights]
                    except KeyError:
                        raise KeyError("String passed to weights not a valid column")
                else:
                    raise ValueError(
                        "Strings can only be passed to "
                        "weights when sampling from rows on "
                        "a DataFrame"
                    )
            weights = pandas.Series(weights, dtype="float64")

            if len(weights) != axis_length:
                raise ValueError(
                    "Weights and axis to be sampled must be of same length"
                )
            if (weights == np.inf).any() or (weights == -np.inf).any():
                raise ValueError("weight vector may not include `inf` values")
            if (weights < 0).any():
                raise ValueError("weight vector many not include negative values")
            # weights cannot be NaN when sampling, so we must set all nan
            # values to 0
            weights = weights.fillna(0)
            # If passed in weights are not equal to 1, renormalize them
            # otherwise numpy sampling function will error
            weights_sum = weights.sum()
            if weights_sum != 1:
                if weights_sum != 0:
                    weights = weights / weights_sum
                else:
                    raise ValueError("Invalid weights: weights sum to zero")
            weights = weights.values

        if n is None and frac is None:
            # default to n = 1 if n and frac are both None (in accordance with
            # Pandas specification)
            n = 1
        elif n is not None and frac is None and n % 1 != 0:
            # n must be an integer
            raise ValueError("Only integers accepted as `n` values")
        elif n is None and frac is not None:
            # compute the number of samples based on frac
            n = int(round(frac * axis_length))
        elif n is not None and frac is not None:
            # Pandas specification does not allow both n and frac to be passed
            # in
            raise ValueError("Please enter a value for `frac` OR `n`, not both")
        if n < 0:
            raise ValueError(
                "A negative number of rows requested. Please provide positive value."
            )
        if n == 0:
            # This returns an empty object, and since it is a weird edge case that
            # doesn't need to be distributed, we default to pandas for n=0.
            return self._default_to_pandas(
                "sample",
                n=n,
                frac=frac,
                replace=replace,
                weights=weights,
                random_state=random_state,
                axis=axis,
            )
        if random_state is not None:
            # Get a random number generator depending on the type of
            # random_state that is passed in
            if isinstance(random_state, int):
                random_num_gen = np.random.RandomState(random_state)
            elif isinstance(random_state, np.random.RandomState):
                random_num_gen = random_state
            else:
                # random_state must be an int or a numpy RandomState object
                raise ValueError(
                    "Please enter an `int` OR a "
                    "np.random.RandomState for random_state"
                )
            # choose random numbers and then get corresponding labels from
            # chosen axis
            sample_indices = random_num_gen.choice(
                np.arange(0, axis_length), size=n, replace=replace, p=weights
            )
            samples = axis_labels[sample_indices]
        else:
            # randomly select labels from chosen axis
            samples = np.random.choice(
                a=axis_labels, size=n, replace=replace, p=weights
            )
        if axis:
            query_compiler = self._query_compiler.getitem_column_array(samples)
            return self.__constructor__(query_compiler=query_compiler)
        else:
            query_compiler = self._query_compiler.getitem_row_array(samples)
            return self.__constructor__(query_compiler=query_compiler)

    def set_axis(self, labels, axis=0, inplace=False):
        """Assign desired index to given axis.

        Args:
            labels (pandas.Index or list-like): The Index to assign.
            axis (string or int): The axis to reassign.
            inplace (bool): Whether to make these modifications inplace.

        Returns:
            If inplace is False, returns a new DataFrame, otherwise None.
        """
        if is_scalar(labels):
            warnings.warn(
                'set_axis now takes "labels" as first argument, and '
                '"axis" as named parameter. The old form, with "axis" as '
                'first parameter and "labels" as second, is still supported '
                "but will be deprecated in a future version of pandas.",
                FutureWarning,
                stacklevel=2,
            )
            labels, axis = axis, labels
        if inplace:
            setattr(self, pandas.DataFrame()._get_axis_name(axis), labels)
        else:
            obj = self.copy()
            obj.set_axis(labels, axis=axis, inplace=True)
            return obj

    def shift(self, periods=1, freq=None, axis=0, fill_value=None):
        """
        Shift index by desired number of periods with an optional time `freq`.
        When `freq` is not passed, shift the index without realigning the data.
        If `freq` is passed (in this case, the index must be date or datetime,
        or it will raise a `NotImplementedError`), the index will be
        increased using the periods and the `freq`.
        Parameters
        ----------
        periods : int
            Number of periods to shift. Can be positive or negative.
        freq : DateOffset, tseries.offsets, timedelta, or str, optional
            Offset to use from the tseries module or time rule (e.g. 'EOM').
            If `freq` is specified then the index values are shifted but the
            data is not realigned. That is, use `freq` if you would like to
            extend the index when shifting and preserve the original data.
        axis : {{0 or 'index', 1 or 'columns', None}}, default None
            Shift direction.
        fill_value : object, optional
            The scalar value to use for newly introduced missing values.
            the default depends on the dtype of `self`.
            For numeric data, ``np.nan`` is used.
            For datetime, timedelta, or period data, etc. :attr:`NaT` is used.
            For extension dtypes, ``self.dtype.na_value`` is used.
        Returns
        -------
            Copy of input object, shifted.
        """

        if periods == 0:
            # Check obvious case first
            return self.copy()

        empty_frame = False
        if axis == "index" or axis == 0:
            if abs(periods) >= len(self.index):
                fill_index = self.index
                empty_frame = True
            else:
                fill_index = pandas.RangeIndex(start=0, stop=abs(periods), step=1)
        else:
            fill_index = self.index
        from .dataframe import DataFrame

        fill_columns = None
        if isinstance(self, DataFrame):
            if axis == "columns" or axis == 1:
                if abs(periods) >= len(self.columns):
                    fill_columns = self.columns
                    empty_frame = True
                else:
                    fill_columns = pandas.RangeIndex(start=0, stop=abs(periods), step=1)
            else:
                fill_columns = self.columns

        filled_df = (
            self.__constructor__(index=fill_index, columns=fill_columns)
            if isinstance(self, DataFrame)
            else self.__constructor__(index=fill_index)
        )
        if fill_value is not None:
            filled_df.fillna(fill_value, inplace=True)

        if empty_frame:
            return filled_df

        if freq is None:
            if axis == "index" or axis == 0:
                if periods > 0:
                    dropped_df = self.drop(self.index[-periods:])
                    new_frame = filled_df.append(dropped_df, ignore_index=True)
                    new_frame.index = self.index.copy()
                    if isinstance(self, DataFrame):
                        new_frame.columns = self.columns.copy()
                    return new_frame
                else:
                    dropped_df = self.drop(self.index[:-periods])
                    new_frame = dropped_df.append(filled_df, ignore_index=True)
                    new_frame.index = self.index.copy()
                    if isinstance(self, DataFrame):
                        new_frame.columns = self.columns.copy()
                    return new_frame
            else:
                res_columns = self.columns
                from .concat import concat

                if periods > 0:
                    dropped_df = self.drop(self.columns[-periods:], axis="columns")
                    new_frame = concat([filled_df, dropped_df], axis="columns")
                    new_frame.columns = res_columns
                    return new_frame
                else:
                    dropped_df = self.drop(self.columns[:-periods], axis="columns")
                    new_frame = concat([dropped_df, filled_df], axis="columns")
                    new_frame.columns = res_columns
                    return new_frame
        else:
            return self.tshift(periods, freq)

    def sort_index(
        self,
        axis=0,
        level=None,
        ascending=True,
        inplace=False,
        kind="quicksort",
        na_position="last",
        sort_remaining=True,
        ignore_index: bool = False,
        key: Optional[IndexKeyFunc] = None,
    ):
        """
        Sort object by labels (along an axis).

        Parameters
        ----------
            axis : {0 or 'index', 1 or 'columns'}, default 0
                The axis along which to sort.
                The value 0 identifies the rows, and 1 identifies the columns.
            level : int or level name or list of ints or list of level names
                If not None, sort on values in specified index level(s).
            ascending : bool or list of bools, default True
                Sort ascending vs. descending. When the index is a MultiIndex
                the sort direction can be controlled for each level individually.
            inplace : bool, default False
                If True, perform operation in-place.
            kind : {'quicksort', 'mergesort', 'heapsort'}, default 'quicksort'
                Choice of sorting algorithm. See also ndarray.np.sort for more information.
                mergesort is the only stable algorithm. For DataFrames,
                this option is only applied when sorting on a single column or label.
            na_position : {'first', 'last'}, default 'last'
                Puts NaNs at the beginning if first; last puts NaNs at the end.
                Not implemented for MultiIndex.
            sort_remaining : bool, default True
                If True and sorting by level and index is multilevel,
                sort by other levels too (in order) after sorting by specified level.
            ignore_index : bool, default False
                If True, the resulting axis will be labeled 0, 1, ..., n - 1.
            key : callable, optional
                If not None, apply the key function to the index values before sorting.
                This is similar to the key argument in the builtin sorted() function,
                with the notable difference that this key function should be vectorized.
                It should expect an Index and return an Index of the same shape.
                For MultiIndex inputs, the key is applied per level.

        Returns
        -------
            A sorted DataFrame
        """
        axis = self._get_axis_number(axis)
        inplace = validate_bool_kwarg(inplace, "inplace")
        new_query_compiler = self._query_compiler.sort_index(
            axis=axis,
            level=level,
            ascending=ascending,
            inplace=inplace,
            kind=kind,
            na_position=na_position,
            sort_remaining=sort_remaining,
            ignore_index=ignore_index,
            key=key,
        )
        return self._create_or_update_from_compiler(new_query_compiler, inplace)

    def sort_values(
        self,
        by,
        axis=0,
        ascending=True,
        inplace: bool = False,
        kind="quicksort",
        na_position="last",
        ignore_index: bool = False,
        key: Optional[IndexKeyFunc] = None,
    ):
        """
        Sort by the values along either axis.

        Parameters
        ----------
            by : str or list of str
                Name or list of names to sort by.

                - if axis is 0 or 'index' then by may contain index levels and/or column labels.
                - if axis is 1 or 'columns' then by may contain column levels and/or index labels.

            axis : {0 or 'index', 1 or 'columns'}, default 0
                Axis to be sorted.
            ascending : bool or list of bool, default True
                Sort ascending vs. descending. Specify list for multiple sort orders.
                If this is a list of bools, must match the length of the by.
            inplace : bool, default False
                If True, perform operation in-place.
            kind : {'quicksort', 'mergesort', 'heapsort'}, default 'quicksort'
                Choice of sorting algorithm. See also ndarray.np.sort for more information.
                mergesort is the only stable algorithm. For DataFrames,
                this option is only applied when sorting on a single column or label.
            na_position : {'first', 'last'}, default 'last'
                Puts NaNs at the beginning if first; last puts NaNs at the end.
            ignore_index : bool, default False
                If True, the resulting axis will be labeled 0, 1, ..., n - 1.
            key : callable, optional
                Apply the key function to the values before sorting.
                This is similar to the key argument in the builtin sorted() function,
                with the notable difference that this key function should be vectorized.
                It should expect a Series and return a Series with the same shape as the input.
                It will be applied to each column in by independently.

        Returns
        -------
             A sorted DataFrame.
        """
        axis = self._get_axis_number(axis)
        inplace = validate_bool_kwarg(inplace, "inplace")
        if axis == 0:
            result = self._query_compiler.sort_rows_by_column_values(
                by,
                ascending=ascending,
                kind=kind,
                na_position=na_position,
                ignore_index=ignore_index,
                key=key,
            )
        else:
            result = self._query_compiler.sort_columns_by_row_values(
                by,
                ascending=ascending,
                kind=kind,
                na_position=na_position,
                ignore_index=ignore_index,
                key=key,
            )
        return self._create_or_update_from_compiler(result, inplace)

    def sub(self, other, axis="columns", level=None, fill_value=None):
        """Subtract a DataFrame/Series/scalar from this DataFrame.

        Args:
            other: The object to use to apply the subtraction to this.
            axis: The axis to apply the subtraction over.
            level: Mutlilevel index level to subtract over.
            fill_value: The value to fill NaNs with.

        Returns:
             A new DataFrame with the subtraciont applied.
        """
        return self._binary_op(
            "sub", other, axis=axis, level=level, fill_value=fill_value
        )

    subtract = sub

    def swapaxes(self, axis1, axis2, copy=True):
        axis1 = self._get_axis_number(axis1)
        axis2 = self._get_axis_number(axis2)
        if axis1 != axis2:
            return self.transpose()
        if copy:
            return self.copy()
        return self

    def swaplevel(self, i=-2, j=-1, axis=0):
        axis = self._get_axis_number(axis)
        idx = self.index if axis == 0 else self.columns
        return self.set_axis(idx.swaplevel(i, j), axis=axis, inplace=False)

    def tail(self, n=5):
        """Get the last n rows of the DataFrame or Series.

        Parameters
        ----------
        n : int
            The number of rows to return.

        Returns
        -------
        DataFrame or Series
            A new DataFrame or Series with the last n rows of this DataFrame or Series.
        """
        if n != 0:
            return self.iloc[-n:]
        return self.iloc[len(self.index) :]

    def take(self, indices, axis=0, is_copy=None, **kwargs):
        axis = self._get_axis_number(axis)
        slice_obj = indices if axis == 0 else (slice(None), indices)
        result = self.iloc[slice_obj]
        return result if not is_copy else result.copy()

    def to_clipboard(self, excel=True, sep=None, **kwargs):  # pragma: no cover
        return self._default_to_pandas("to_clipboard", excel=excel, sep=sep, **kwargs)

    def to_csv(
        self,
        path_or_buf=None,
        sep=",",
        na_rep="",
        float_format=None,
        columns=None,
        header=True,
        index=True,
        index_label=None,
        mode="w",
        encoding=None,
        compression="infer",
        quoting=None,
        quotechar='"',
        line_terminator=None,
        chunksize=None,
        date_format=None,
        doublequote=True,
        escapechar=None,
        decimal=".",
        errors: str = "strict",
    ):  # pragma: no cover

        kwargs = {
            "path_or_buf": path_or_buf,
            "sep": sep,
            "na_rep": na_rep,
            "float_format": float_format,
            "columns": columns,
            "header": header,
            "index": index,
            "index_label": index_label,
            "mode": mode,
            "encoding": encoding,
            "compression": compression,
            "quoting": quoting,
            "quotechar": quotechar,
            "line_terminator": line_terminator,
            "chunksize": chunksize,
            "date_format": date_format,
            "doublequote": doublequote,
            "escapechar": escapechar,
            "decimal": decimal,
            "errors": errors,
        }
        return self._default_to_pandas("to_csv", **kwargs)

    def to_dict(self, orient="dict", into=dict):  # pragma: no cover
        return self._default_to_pandas("to_dict", orient=orient, into=into)

    def to_excel(
        self,
        excel_writer,
        sheet_name="Sheet1",
        na_rep="",
        float_format=None,
        columns=None,
        header=True,
        index=True,
        index_label=None,
        startrow=0,
        startcol=0,
        engine=None,
        merge_cells=True,
        encoding=None,
        inf_rep="inf",
        verbose=True,
        freeze_panes=None,
    ):  # pragma: no cover
        return self._default_to_pandas(
            "to_excel",
            excel_writer,
            sheet_name,
            na_rep,
            float_format,
            columns,
            header,
            index,
            index_label,
            startrow,
            startcol,
            engine,
            merge_cells,
            encoding,
            inf_rep,
            verbose,
            freeze_panes,
        )

    def to_hdf(self, path_or_buf, key, format="table", **kwargs):  # pragma: no cover
        return self._default_to_pandas(
            "to_hdf", path_or_buf, key, format=format, **kwargs
        )

    def to_json(
        self,
        path_or_buf=None,
        orient=None,
        date_format=None,
        double_precision=10,
        force_ascii=True,
        date_unit="ms",
        default_handler=None,
        lines=False,
        compression="infer",
        index=True,
        indent=None,
    ):  # pragma: no cover
        return self._default_to_pandas(
            "to_json",
            path_or_buf,
            orient=orient,
            date_format=date_format,
            double_precision=double_precision,
            force_ascii=force_ascii,
            date_unit=date_unit,
            default_handler=default_handler,
            lines=lines,
            compression=compression,
            index=index,
            indent=indent,
        )

    def to_latex(
        self,
        buf=None,
        columns=None,
        col_space=None,
        header=True,
        index=True,
        na_rep="NaN",
        formatters=None,
        float_format=None,
        sparsify=None,
        index_names=True,
        bold_rows=False,
        column_format=None,
        longtable=None,
        escape=None,
        encoding=None,
        decimal=".",
        multicolumn=None,
        multicolumn_format=None,
        multirow=None,
        caption=None,
        label=None,
    ):  # pragma: no cover
        return self._default_to_pandas(
            "to_latex",
            buf=buf,
            columns=columns,
            col_space=col_space,
            header=header,
            index=index,
            na_rep=na_rep,
            formatters=formatters,
            float_format=float_format,
            sparsify=sparsify,
            index_names=index_names,
            bold_rows=bold_rows,
            column_format=column_format,
            longtable=longtable,
            escape=escape,
            encoding=encoding,
            decimal=decimal,
            multicolumn=multicolumn,
            multicolumn_format=multicolumn_format,
            multirow=multirow,
            caption=None,
            label=None,
        )

    def to_markdown(self, buf=None, mode=None, index: bool = True, **kwargs):
        return self._default_to_pandas(
            "to_markdown", buf=buf, mode=mode, index=index, **kwargs
        )

    def to_numpy(self, dtype=None, copy=False, na_value=no_default):
        """
        Convert the DataFrame to a NumPy array.

        Parameters
        ----------
            dtype : str or numpy.dtype, optional
                The dtype to pass to numpy.asarray().
            copy : bool, default False
                Whether to ensure that the returned value is not a view on another array.
                Note that copy=False does not ensure that to_numpy() is no-copy.
                Rather, copy=True ensure that a copy is made, even if not strictly necessary.
            na_value : Any, optional
                The value to use for missing values.
                The default value depends on dtype and the type of the array.

        Returns
        -------
            A NumPy array.
        """
        return self._query_compiler.to_numpy(
            dtype=dtype,
            copy=copy,
            na_value=na_value,
        )

    # TODO(williamma12): When this gets implemented, have the series one call this.
    def to_period(self, freq=None, axis=0, copy=True):  # pragma: no cover
        return self._default_to_pandas("to_period", freq=freq, axis=axis, copy=copy)

    def to_pickle(
        self, path, compression="infer", protocol=pkl.HIGHEST_PROTOCOL
    ):  # pragma: no cover
        return self._default_to_pandas(
            "to_pickle", path, compression=compression, protocol=protocol
        )

    def to_string(
        self,
        buf=None,
        columns=None,
        col_space=None,
        header=True,
        index=True,
        na_rep="NaN",
        formatters=None,
        float_format=None,
        sparsify=None,
        index_names=True,
        justify=None,
        max_rows=None,
        min_rows=None,
        max_cols=None,
        show_dimensions=False,
        decimal=".",
        line_width=None,
        max_colwidth=None,
        encoding=None,
    ):
        return self._default_to_pandas(
            "to_string",
            buf=buf,
            columns=columns,
            col_space=col_space,
            header=header,
            index=index,
            na_rep=na_rep,
            formatters=formatters,
            float_format=float_format,
            sparsify=sparsify,
            index_names=index_names,
            justify=justify,
            max_rows=max_rows,
            max_cols=max_cols,
            show_dimensions=show_dimensions,
            decimal=decimal,
            line_width=line_width,
            max_colwidth=max_colwidth,
            encoding=encoding,
        )

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
        new_query_compiler = self._query_compiler
        # writing the index to the database by inserting it to the DF
        if index:
            if not index_label:
                index_label = "index"
            new_query_compiler = new_query_compiler.insert(0, index_label, self.index)
            # so pandas._to_sql will not write the index to the database as well
            index = False

        from modin.data_management.factories.dispatcher import EngineDispatcher

        EngineDispatcher.to_sql(
            new_query_compiler,
            name=name,
            con=con,
            schema=schema,
            if_exists=if_exists,
            index=index,
            index_label=index_label,
            chunksize=chunksize,
            dtype=dtype,
            method=method,
        )

    # TODO(williamma12): When this gets implemented, have the series one call this.
    def to_timestamp(self, freq=None, how="start", axis=0, copy=True):
        return self._default_to_pandas(
            "to_timestamp", freq=freq, how=how, axis=axis, copy=copy
        )

    def to_xarray(self):
        return self._default_to_pandas("to_xarray")

    def truediv(self, other, axis="columns", level=None, fill_value=None):
        """Divides this DataFrame against another DataFrame/Series/scalar.

        Args:
            other: The object to use to apply the divide against this.
            axis: The axis to divide over.
            level: The Multilevel index level to apply divide over.
            fill_value: The value to fill NaNs with.

        Returns:
            A new DataFrame with the Divide applied.
        """
        return self._binary_op(
            "truediv", other, axis=axis, level=level, fill_value=fill_value
        )

    div = divide = truediv

    def truncate(self, before=None, after=None, axis=None, copy=True):
        axis = self._get_axis_number(axis)
        if (
            not self.axes[axis].is_monotonic_increasing
            and not self.axes[axis].is_monotonic_decreasing
        ):
            raise ValueError("truncate requires a sorted index")
        s = slice(*self.axes[axis].slice_locs(before, after))
        slice_obj = s if axis == 0 else (slice(None), s)
        return self.iloc[slice_obj]

    def tshift(self, periods=1, freq=None, axis=0):
        axis = self._get_axis_number(axis)
        new_labels = self.axes[axis].shift(periods, freq=freq)
        return self.set_axis(new_labels, axis=axis, inplace=False)

    def transform(self, func, axis=0, *args, **kwargs):
        kwargs["is_transform"] = True
        result = self.agg(func, axis=axis, *args, **kwargs)
        try:
            assert len(result) == len(self)
        except Exception:
            raise ValueError("transforms cannot produce aggregated results")
        return result

    def tz_convert(self, tz, axis=0, level=None, copy=True):
        axis = self._get_axis_number(axis)
        if level is not None:
            new_labels = (
                pandas.Series(index=self.axes[axis]).tz_convert(tz, level=level).index
            )
        else:
            new_labels = self.axes[axis].tz_convert(tz)
        obj = self.copy() if copy else self
        return obj.set_axis(new_labels, axis, inplace=not copy)

    def tz_localize(
        self, tz, axis=0, level=None, copy=True, ambiguous="raise", nonexistent="raise"
    ):
        axis = self._get_axis_number(axis)
        new_labels = (
            pandas.Series(index=self.axes[axis])
            .tz_localize(
                tz,
                axis=axis,
                level=level,
                copy=False,
                ambiguous=ambiguous,
                nonexistent=nonexistent,
            )
            .index
        )
        return self.set_axis(labels=new_labels, axis=axis, inplace=not copy)

    def __abs__(self):
        """Creates a modified DataFrame by taking the absolute value.

        Returns:
            A modified DataFrame
        """
        return self.abs()

    def __and__(self, other):
        return self._binary_op("__and__", other, axis=0)

    def __rand__(self, other):
        return self._binary_op("__rand__", other, axis=0)

    def __array__(self, dtype=None):
        arr = self.to_numpy(dtype)
        return arr

    def __array_wrap__(self, result, context=None):
        """TODO: This is very inefficient. __array__ and as_matrix have been
        changed to call the more efficient to_numpy, but this has been left
        unchanged since we are not sure of its purpose.
        """
        return self._default_to_pandas("__array_wrap__", result, context=context)

    def __copy__(self, deep=True):
        """Make a copy of this object.

        Args:
            deep: Boolean, deep copy or not.
                  Currently we do not support deep copy.

        Returns:
            A Modin Series/DataFrame object.
        """
        return self.copy(deep=deep)

    def __deepcopy__(self, memo=None):
        """Make a -deep- copy of this object.

        Note: This is equivalent to copy(deep=True).

        Args:
            memo: No effect. Just to comply with Pandas API.

        Returns:
            A Modin Series/DataFrame object.
        """
        return self.copy(deep=True)

    def __eq__(self, other):
        return self.eq(other)

    def __finalize__(self, other, method=None, **kwargs):
        return self._default_to_pandas("__finalize__", other, method=method, **kwargs)

    def __ge__(self, right):
        return self.ge(right)

    def __getitem__(self, key):
        if not self._query_compiler.lazy_execution and len(self) == 0:
            return self._default_to_pandas("__getitem__", key)
        # see if we can slice the rows
        # This lets us reuse code in Pandas to error check
        indexer = None
        if isinstance(key, slice) or (
            isinstance(key, str)
            and (not hasattr(self, "columns") or key not in self.columns)
        ):
            indexer = convert_to_index_sliceable(
                getattr(pandas, "DataFrame")(index=self.index), key
            )
        if indexer is not None:
            return self._getitem_slice(indexer)
        else:
            return self._getitem(key)

    def _getitem_slice(self, key):
        if key.start is None and key.stop is None:
            return self.copy()
        return self.iloc[key]

    def __getstate__(self):
        return self._default_to_pandas("__getstate__")

    def __gt__(self, right):
        return self.gt(right)

    def __invert__(self):
        if not all(is_numeric_dtype(d) for d in self._get_dtypes()):
            raise TypeError(
                "bad operand type for unary ~: '{}'".format(
                    next(d for d in self._get_dtypes() if not is_numeric_dtype(d))
                )
            )
        return self.__constructor__(query_compiler=self._query_compiler.invert())

    def __le__(self, right):
        return self.le(right)

    def __len__(self):
        """Gets the length of the DataFrame.

        Returns:
            Returns an integer length of the DataFrame object.
        """
        return len(self.index)

    def __lt__(self, right):
        return self.lt(right)

    def __matmul__(self, other):
        return self.dot(other)

    def __ne__(self, other):
        return self.ne(other)

    def __neg__(self):
        """Computes an element wise negative DataFrame

        Returns:
            A modified DataFrame where every element is the negation of before
        """
        self._validate_dtypes(numeric_only=True)
        return self.__constructor__(query_compiler=self._query_compiler.negative())

    def __nonzero__(self):
        raise ValueError(
            "The truth value of a {0} is ambiguous. "
            "Use a.empty, a.bool(), a.item(), a.any() or a.all().".format(
                self.__class__.__name__
            )
        )

    __bool__ = __nonzero__

    def __or__(self, other):
        return self._binary_op("__or__", other, axis=0)

    def __ror__(self, other):
        return self._binary_op("__ror__", other, axis=0)

    def __sizeof__(self):
        return self._default_to_pandas("__sizeof__")

    def __str__(self):  # pragma: no cover
        return repr(self)

    def __xor__(self, other):
        return self._binary_op("__xor__", other, axis=0)

    def __rxor__(self, other):
        return self._binary_op("__rxor__", other, axis=0)

    @property
    def size(self):
        """Get the number of elements in the DataFrame.

        Returns:
            The number of elements in the DataFrame.
        """
        return len(self._query_compiler.index) * len(self._query_compiler.columns)

    @property
    def values(self):
        """Create a NumPy array with the values from this object.

        Returns:
            The numpy representation of this object.
        """
        return self.to_numpy()

    def __getattribute__(self, item):
        default_behaviors = {
            "__init__",
            "__class__",
            "_get_index",
            "_set_index",
            "empty",
            "index",
            "columns",
            "name",
            "dtypes",
            "dtype",
            "_get_name",
            "_set_name",
            "_default_to_pandas",
            "_query_compiler",
            "_to_pandas",
            "_build_repr_df",
            "_reduce_dimension",
            "__repr__",
            "__len__",
            "_create_or_update_from_compiler",
            "_update_inplace",
        } | _ATTRS_NO_LOOKUP
        if item not in default_behaviors and not self._query_compiler.lazy_execution:
            method = object.__getattribute__(self, item)
            is_callable = callable(method)
            # We default to pandas on empty DataFrames. This avoids a large amount of
            # pain in underlying implementation and returns a result immediately rather
            # than dealing with the edge cases that empty DataFrames have.
            if is_callable and self.empty:

                def default_handler(*args, **kwargs):
                    return self._default_to_pandas(item, *args, **kwargs)

                return default_handler
        return object.__getattribute__(self, item)


if os.environ.get("MODIN_EXPERIMENTAL", "").title() == "True":
    from modin.experimental.cloud.meta_magic import make_wrapped_class

    make_wrapped_class(BasePandasDataset, "make_base_dataset_wrapper")


class Resampler(object):
    def __init__(
        self,
        dataframe,
        rule,
        axis=0,
        closed=None,
        label=None,
        convention="start",
        kind=None,
        loffset=None,
        base=0,
        on=None,
        level=None,
        origin: Union[str, TimestampConvertibleTypes] = "start_day",
        offset: Optional[TimedeltaConvertibleTypes] = None,
    ):
        self._dataframe = dataframe
        self._query_compiler = dataframe._query_compiler
        axis = self._dataframe._get_axis_number(axis)
        self.resample_args = [
            rule,
            axis,
            closed,
            label,
            convention,
            kind,
            loffset,
            base,
            on,
            level,
            origin,
            offset,
        ]
        self.__groups = self.__get_groups(*self.resample_args)

    def __get_groups(
        self,
        rule,
        axis,
        closed,
        label,
        convention,
        kind,
        loffset,
        base,
        on,
        level,
        origin,
        offset,
    ):
        if axis == 0:
            df = self._dataframe
        else:
            df = self._dataframe.T
        groups = df.groupby(
            pandas.Grouper(
                key=on,
                freq=rule,
                closed=closed,
                label=label,
                convention=convention,
                loffset=loffset,
                base=base,
                level=level,
                origin=origin,
                offset=offset,
            )
        )
        return groups

    @property
    def groups(self):
        return self._query_compiler.default_to_pandas(
            lambda df: pandas.DataFrame.resample(df, *self.resample_args).groups
        )

    @property
    def indices(self):
        return self._query_compiler.default_to_pandas(
            lambda df: pandas.DataFrame.resample(df, *self.resample_args).indices
        )

    def get_group(self, name, obj=None):
        if self.resample_args[1] == 0:
            result = self.__groups.get_group(name)
        else:
            result = self.__groups.get_group(name).T
        return result

    def apply(self, func, *args, **kwargs):
        from .dataframe import DataFrame

        if isinstance(self._dataframe, DataFrame):
            query_comp_op = self._query_compiler.resample_app_df
        else:
            query_comp_op = self._query_compiler.resample_app_ser

        dataframe = DataFrame(
            query_compiler=query_comp_op(
                self.resample_args,
                func,
                *args,
                **kwargs,
            )
        )
        if is_list_like(func) or isinstance(self._dataframe, DataFrame):
            return dataframe
        else:
            if len(dataframe.index) == 1:
                return dataframe.iloc[0]
            else:
                return dataframe.squeeze()

    def aggregate(self, func, *args, **kwargs):
        from .dataframe import DataFrame

        if isinstance(self._dataframe, DataFrame):
            query_comp_op = self._query_compiler.resample_agg_df
        else:
            query_comp_op = self._query_compiler.resample_agg_ser

        dataframe = DataFrame(
            query_compiler=query_comp_op(
                self.resample_args,
                func,
                *args,
                **kwargs,
            )
        )
        if is_list_like(func) or isinstance(self._dataframe, DataFrame):
            return dataframe
        else:
            if len(dataframe.index) == 1:
                return dataframe.iloc[0]
            else:
                return dataframe.squeeze()

    def transform(self, arg, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_transform(
                self.resample_args, arg, *args, **kwargs
            )
        )

    def pipe(self, func, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_pipe(
                self.resample_args, func, *args, **kwargs
            )
        )

    def ffill(self, limit=None):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_ffill(
                self.resample_args, limit
            )
        )

    def backfill(self, limit=None):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_backfill(
                self.resample_args, limit
            )
        )

    def bfill(self, limit=None):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_bfill(
                self.resample_args, limit
            )
        )

    def pad(self, limit=None):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_pad(self.resample_args, limit)
        )

    def nearest(self, limit=None):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_nearest(
                self.resample_args, limit
            )
        )

    def fillna(self, method, limit=None):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_fillna(
                self.resample_args, method, limit
            )
        )

    def asfreq(self, fill_value=None):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_asfreq(
                self.resample_args, fill_value
            )
        )

    def interpolate(
        self,
        method="linear",
        axis=0,
        limit=None,
        inplace=False,
        limit_direction: Optional[str] = None,
        limit_area=None,
        downcast=None,
        **kwargs,
    ):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_interpolate(
                self.resample_args,
                method,
                axis,
                limit,
                inplace,
                limit_direction,
                limit_area,
                downcast,
                **kwargs,
            )
        )

    def count(self):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_count(self.resample_args)
        )

    def nunique(self, _method="nunique", *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_nunique(
                self.resample_args, _method, *args, **kwargs
            )
        )

    def first(self, _method="first", *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_first(
                self.resample_args,
                _method,
                *args,
                **kwargs,
            )
        )

    def last(self, _method="last", *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_last(
                self.resample_args,
                _method,
                *args,
                **kwargs,
            )
        )

    def max(self, _method="max", *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_max(
                self.resample_args,
                _method,
                *args,
                **kwargs,
            )
        )

    def mean(self, _method="mean", *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_mean(
                self.resample_args,
                _method,
                *args,
                **kwargs,
            )
        )

    def median(self, _method="median", *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_median(
                self.resample_args,
                _method,
                *args,
                **kwargs,
            )
        )

    def min(self, _method="min", *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_min(
                self.resample_args,
                _method,
                *args,
                **kwargs,
            )
        )

    def ohlc(self, _method="ohlc", *args, **kwargs):
        from .dataframe import DataFrame

        if isinstance(self._dataframe, DataFrame):
            return DataFrame(
                query_compiler=self._query_compiler.resample_ohlc_df(
                    self.resample_args,
                    _method,
                    *args,
                    **kwargs,
                )
            )
        else:
            return DataFrame(
                query_compiler=self._query_compiler.resample_ohlc_ser(
                    self.resample_args,
                    _method,
                    *args,
                    **kwargs,
                )
            )

    def prod(self, _method="prod", min_count=0, *args, **kwargs):
        if self.resample_args[1] == 0:
            result = self.__groups.prod(min_count=min_count, *args, **kwargs)
        else:
            result = self.__groups.prod(min_count=min_count, *args, **kwargs).T
        return result

    def size(self):
        from .series import Series

        return Series(
            query_compiler=self._query_compiler.resample_size(self.resample_args)
        )

    def sem(self, _method="sem", *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_sem(
                self.resample_args,
                _method,
                *args,
                **kwargs,
            )
        )

    def std(self, ddof=1, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_std(
                self.resample_args, *args, ddof=ddof, **kwargs
            )
        )

    def sum(self, _method="sum", min_count=0, *args, **kwargs):
        if self.resample_args[1] == 0:
            result = self.__groups.sum(min_count=min_count, *args, **kwargs)
        else:
            result = self.__groups.sum(min_count=min_count, *args, **kwargs).T
        return result

    def var(self, ddof=1, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_var(
                self.resample_args, *args, ddof=ddof, **kwargs
            )
        )

    def quantile(self, q=0.5, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.resample_quantile(
                self.resample_args, q, **kwargs
            )
        )


class Window(object):
    def __init__(
        self,
        dataframe,
        window,
        min_periods=None,
        center=False,
        win_type=None,
        on=None,
        axis=0,
        closed=None,
    ):
        self._dataframe = dataframe
        self._query_compiler = dataframe._query_compiler
        self.window_args = [
            window,
            min_periods,
            center,
            win_type,
            on,
            axis,
            closed,
        ]

    def mean(self, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.window_mean(
                self.window_args, *args, **kwargs
            )
        )

    def sum(self, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.window_sum(
                self.window_args, *args, **kwargs
            )
        )

    def var(self, ddof=1, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.window_var(
                self.window_args, ddof, *args, **kwargs
            )
        )

    def std(self, ddof=1, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.window_std(
                self.window_args, ddof, *args, **kwargs
            )
        )


class Rolling(object):
    def __init__(
        self,
        dataframe,
        window,
        min_periods=None,
        center=False,
        win_type=None,
        on=None,
        axis=0,
        closed=None,
    ):
        self._dataframe = dataframe
        self._query_compiler = dataframe._query_compiler
        self.rolling_args = [
            window,
            min_periods,
            center,
            win_type,
            on,
            axis,
            closed,
        ]

    def count(self):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.rolling_count(self.rolling_args)
        )

    def sum(self, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.rolling_sum(
                self.rolling_args, *args, **kwargs
            )
        )

    def mean(self, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.rolling_mean(
                self.rolling_args, *args, **kwargs
            )
        )

    def median(self, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.rolling_median(
                self.rolling_args, **kwargs
            )
        )

    def var(self, ddof=1, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.rolling_var(
                self.rolling_args, ddof, *args, **kwargs
            )
        )

    def std(self, ddof=1, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.rolling_std(
                self.rolling_args, ddof, *args, **kwargs
            )
        )

    def min(self, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.rolling_min(
                self.rolling_args, *args, **kwargs
            )
        )

    def max(self, *args, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.rolling_max(
                self.rolling_args, *args, **kwargs
            )
        )

    def corr(self, other=None, pairwise=None, *args, **kwargs):
        from .dataframe import DataFrame
        from .series import Series

        if isinstance(other, DataFrame):
            other = other._query_compiler.to_pandas()
        elif isinstance(other, Series):
            other = other._query_compiler.to_pandas().squeeze()

        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.rolling_corr(
                self.rolling_args, other, pairwise, *args, **kwargs
            )
        )

    def cov(self, other=None, pairwise=None, ddof: Optional[int] = 1, **kwargs):
        from .dataframe import DataFrame
        from .series import Series

        if isinstance(other, DataFrame):
            other = other._query_compiler.to_pandas()
        elif isinstance(other, Series):
            other = other._query_compiler.to_pandas().squeeze()

        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.rolling_cov(
                self.rolling_args, other, pairwise, ddof, **kwargs
            )
        )

    def skew(self, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.rolling_skew(
                self.rolling_args, **kwargs
            )
        )

    def kurt(self, **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.rolling_kurt(
                self.rolling_args, **kwargs
            )
        )

    def apply(
        self,
        func,
        raw=False,
        engine="cython",
        engine_kwargs=None,
        args=None,
        kwargs=None,
    ):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.rolling_apply(
                self.rolling_args,
                func,
                raw,
                engine,
                engine_kwargs,
                args,
                kwargs,
            )
        )

    def aggregate(
        self,
        func,
        *args,
        **kwargs,
    ):
        from .dataframe import DataFrame

        dataframe = DataFrame(
            query_compiler=self._query_compiler.rolling_aggregate(
                self.rolling_args,
                func,
                *args,
                **kwargs,
            )
        )
        if isinstance(self._dataframe, DataFrame):
            return dataframe
        elif is_list_like(func):
            dataframe.columns = dataframe.columns.droplevel()
            return dataframe
        else:
            return dataframe.squeeze()

    agg = aggregate

    def quantile(self, quantile, interpolation="linear", **kwargs):
        return self._dataframe.__constructor__(
            query_compiler=self._query_compiler.rolling_quantile(
                self.rolling_args, quantile, interpolation, **kwargs
            )
        )
