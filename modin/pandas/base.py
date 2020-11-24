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
Implement DataFrame/Series public API as Pandas does.

Almost all docstrings for public and magic methods should be inherited from Pandas
for better maintability. So some codes are ignored in pydocstyle check:
    - D101: missing docstring in class
    - D102: missing docstring in public method
    - D105: missing docstring in magic method
Manually add documentation for methods which are not presented in pandas.
"""

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
import pandas.core.window.rolling
import pandas.core.resample
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

from modin.utils import try_cast_to_pandas, _inherit_docstrings
from modin.error_message import ErrorMessage
from modin.pandas.utils import is_scalar
from modin.config import IsExperimental

# Similar to pandas, sentinel value to use as kwarg in place of None when None has
# special meaning and needs to be distinguished from a user explicitly passing None.
sentinel = object()

# Do not lookup certain attributes in columns or index, as they're used for some
# special purposes, like serving remote context
_ATTRS_NO_LOOKUP = {"____id_pack__", "__name__"}


class BasePandasDataset(object):
    """
    Implement most of the common code that exists in DataFrame/Series.

    Since both objects share the same underlying representation, and the algorithms
    are the same, we use this object to define the general behavior of those objects
    and then use those objects to define the output type.
    """

    # Siblings are other objects that share the same query compiler. We use this list
    # to update inplace when there is a shallow copy.
    _siblings = []

    def _add_sibling(self, sibling):
        """
        Implement [METHOD_NAME].

        TODO: Add more details for this docstring template.

        Parameters
        ----------
        What arguments does this function have.
        [
        PARAMETER_NAME: PARAMETERS TYPES
            Description.
        ]

        Returns
        -------
        What this returns (if anything)
        """
        sibling._siblings = self._siblings + [self]
        self._siblings += [sibling]
        for sib in self._siblings:
            sib._siblings += [sibling]

    def _build_repr_df(self, num_rows, num_cols):
        """
        Implement [METHOD_NAME].

        TODO: Add more details for this docstring template.

        Parameters
        ----------
        What arguments does this function have.
        [
        PARAMETER_NAME: PARAMETERS TYPES
            Description.
        ]

        Returns
        -------
        What this returns (if anything)
        """
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
        """
        Update the current DataFrame inplace.

        Parameters
        ----------
        new_query_compiler: query_compiler
            The new QueryCompiler to use to manage the data
        """
        old_query_compiler = self._query_compiler
        self._query_compiler = new_query_compiler
        for sib in self._siblings:
            sib._query_compiler = new_query_compiler
        old_query_compiler.free()

    def _handle_level_agg(self, axis, level, op, sort=False, **kwargs):
        """
        Help to perform error checking for aggregation functions with a level parameter.

        TODO: add types.

        Parameters
        ----------
        axis:
            The axis to apply the operation on
        level:
            The level of the axis to apply the operation on
        op:
            String representation of the operation to be performed on the level
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
        """
        Help to check validity of other in inter-df operations.

        TODO: Add more details for this docstring template.

        Parameters
        ----------
        What arguments does this function have.
        [
        PARAMETER_NAME: PARAMETERS TYPES
            Description.
        ]

        Returns
        -------
        What this returns (if anything)
        """
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
        """
        Implement [METHOD_NAME].

        TODO: Add more details for this docstring template.

        Parameters
        ----------
        What arguments does this function have.
        [
        PARAMETER_NAME: PARAMETERS TYPES
            Description.
        ]

        Returns
        -------
        What this returns (if anything)
        """
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
        """
        Help to use default pandas function.

        TODO: Add more details for this docstring template.

        Parameters
        ----------
        What arguments does this function have.
        [
        PARAMETER_NAME: PARAMETERS TYPES
            Description.
        ]

        Returns
        -------
        What this returns (if anything)
        """
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
        """
        Implement [METHOD_NAME].

        TODO: Add more details for this docstring template.

        Parameters
        ----------
        What arguments does this function have.
        [
        PARAMETER_NAME: PARAMETERS TYPES
            Description.
        ]

        Returns
        -------
        What this returns (if anything)
        """
        return (
            getattr(pandas, type(self).__name__)()._get_axis_number(axis)
            if axis is not None
            else 0
        )

    def __constructor__(self, *args, **kwargs):
        """Construct DataFrame or Series object depending on self type."""
        return type(self)(*args, **kwargs)

    def abs(self):
        self._validate_dtypes(numeric_only=True)
        return self.__constructor__(query_compiler=self._query_compiler.abs())

    def _set_index(self, new_index):
        """
        Set the index for this DataFrame.

        TODO: add types.

        Parameters
        ----------
        new_index:
            The new index to set this.
        """
        self._query_compiler.index = new_index

    def _get_index(self):
        """
        Get the index for this DataFrame.

        Returns
        -------
        The union of all indexes across the partitions.
        """
        return self._query_compiler.index

    index = property(_get_index, _set_index)

    def add(self, other, axis="columns", level=None, fill_value=None):
        return self._binary_op(
            "add", other, axis=axis, level=level, fill_value=fill_value
        )

    def aggregate(self, func=None, axis=0, *args, **kwargs):
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
        idx = self.index if axis == 0 else self.columns
        indexer = pandas.Series(index=idx).at_time(time, asof=asof).index
        return self.loc[indexer] if axis == 0 else self.loc[:, indexer]

    def between_time(
        self, start_time, end_time, include_start=True, include_end=True, axis=None
    ):
        axis = self._get_axis_number(axis)
        idx = self.index if axis == 0 else self.columns
        indexer = (
            pandas.Series(index=idx)
            .between_time(
                start_time,
                end_time,
                include_start=include_start,
                include_end=include_end,
            )
            .index
        )
        return self.loc[indexer] if axis == 0 else self.loc[:, indexer]

    def bfill(self, axis=None, inplace=False, limit=None, downcast=None):
        return self.fillna(
            method="bfill", axis=axis, limit=limit, downcast=downcast, inplace=inplace
        )

    backfill = bfill

    def bool(self):
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
        if deep:
            return self.__constructor__(query_compiler=self._query_compiler.copy())
        new_obj = self.__constructor__(query_compiler=self._query_compiler)
        self._add_sibling(new_obj)
        return new_obj

    def count(self, axis=0, level=None, numeric_only=False):
        axis = self._get_axis_number(axis)
        frame = self.select_dtypes([np.number, np.bool]) if numeric_only else self

        if level is not None:
            if not frame._query_compiler.has_multiindex(axis=axis):
                # error thrown by pandas
                raise TypeError("Can only count levels on hierarchical columns.")

            return self._handle_level_agg(axis=axis, level=level, op="count", sort=True)

        return frame._reduce_dimension(
            frame._query_compiler.count(
                axis=axis, level=level, numeric_only=numeric_only
            )
        )

    def cummax(self, axis=None, skipna=True, *args, **kwargs):
        axis = self._get_axis_number(axis)
        if axis == 1:
            self._validate_dtypes(numeric_only=True)
        return self.__constructor__(
            query_compiler=self._query_compiler.cummax(
                axis=axis, skipna=skipna, **kwargs
            )
        )

    def cummin(self, axis=None, skipna=True, *args, **kwargs):
        axis = self._get_axis_number(axis)
        if axis == 1:
            self._validate_dtypes(numeric_only=True)
        return self.__constructor__(
            query_compiler=self._query_compiler.cummin(
                axis=axis, skipna=skipna, **kwargs
            )
        )

    def cumprod(self, axis=None, skipna=True, *args, **kwargs):
        axis = self._get_axis_number(axis)
        self._validate_dtypes(numeric_only=True)
        return self.__constructor__(
            query_compiler=self._query_compiler.cumprod(
                axis=axis, skipna=skipna, **kwargs
            )
        )

    def cumsum(self, axis=None, skipna=True, *args, **kwargs):
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
        axis = self._get_axis_number(axis)
        new_axis = self.axes[axis].droplevel(level)
        result = self.copy()
        if axis == 0:
            result.index = new_axis
        else:
            result.columns = new_axis
        return result

    def drop_duplicates(self, keep="first", inplace=False, **kwargs):
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
        return self._query_compiler.first_valid_index()

    def floordiv(self, other, axis="columns", level=None, fill_value=None):
        return self._binary_op(
            "floordiv", other, axis=axis, level=level, fill_value=fill_value
        )

    def ge(self, other, axis="columns", level=None):
        return self._binary_op("ge", other, axis=axis, level=level)

    def get(self, key, default=None):
        if key in self.keys():
            return self.__getitem__(key)
        else:
            return default

    def gt(self, other, axis="columns", level=None):
        return self._binary_op("gt", other, axis=axis, level=level)

    def head(self, n=5):
        return self.iloc[:n]

    @property
    def iat(self, axis=None):
        from .indexing import _iLocIndexer

        return _iLocIndexer(self)

    def idxmax(self, axis=0, skipna=True):
        if not all(d != np.dtype("O") for d in self._get_dtypes()):
            raise TypeError("reduction operation 'argmax' not allowed for this dtype")
        axis = self._get_axis_number(axis)
        return self._reduce_dimension(
            self._query_compiler.idxmax(axis=axis, skipna=skipna)
        )

    def idxmin(self, axis=0, skipna=True):
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
        return self.__constructor__(
            query_compiler=self._query_compiler.isin(values=values)
        )

    def isna(self):
        return self.__constructor__(query_compiler=self._query_compiler.isna())

    isnull = isna

    @property
    def iloc(self):
        from .indexing import _iLocIndexer

        return _iLocIndexer(self)

    def kurt(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
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

        if numeric_only is not None and not numeric_only:
            self._validate_dtypes(numeric_only=True)

        data = self._get_numeric_data(axis) if numeric_only else self

        return self._reduce_dimension(
            data._query_compiler.kurt(
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
        return self._query_compiler.last_valid_index()

    def le(self, other, axis="columns", level=None):
        return self._binary_op("le", other, axis=axis, level=level)

    def lt(self, other, axis="columns", level=None):
        return self._binary_op("lt", other, axis=axis, level=level)

    @property
    def loc(self):
        from .indexing import _LocIndexer

        return _LocIndexer(self)

    def mad(self, axis=None, skipna=None, level=None):
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

    def _stat_operation(
        self,
        op_name: str,
        axis: Union[int, str],
        skipna: bool,
        level: Optional[Union[int, str]],
        numeric_only: Optional[bool] = None,
        **kwargs,
    ):
        """
        Do common statistic reduce operations under frame.

        Parameters
        ----------
        op_name: str,
            Name of method to apply.
        axis: int or axis name,
            Axis to apply method on.
        skipna: bool,
            Exclude NA/null values when computing the result.
        level: int or level name,
            If specified `axis` is a MultiIndex, applying method along a particular
            level, collapsing into a Series
        numeric_only: bool
            Include only float, int, boolean columns. If None, will attempt
            to use everything, then use only numeric data.

        Returns
        -------
        In case of Series: scalar or Series (if level specified)
        In case of DataFrame: Series of DataFrame (if level specified)

        """
        axis = self._get_axis_number(axis)
        # If `numeric_only` is None, then we can do this precheck to whether or not
        # frame contains non-numeric columns, if it doesn't, then we can pass to a backend
        # `numeric_only=False` parameter and make its work easier in that case, rather than
        # performing under complicate `numeric_only=None` parameter
        if not numeric_only:
            try:
                self._validate_dtypes(numeric_only=True)
            except TypeError:
                if numeric_only is not None:
                    raise
            else:
                numeric_only = False

        data = self._get_numeric_data(axis) if numeric_only else self

        result_qc = getattr(data._query_compiler, op_name)(
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            **kwargs,
        )
        if level is not None:
            return self.__constructor__(query_compiler=result_qc)
        return self._reduce_dimension(result_qc)

    def mean(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        return self._stat_operation("mean", axis, skipna, level, numeric_only, **kwargs)

    def median(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        return self._stat_operation(
            "median", axis, skipna, level, numeric_only, **kwargs
        )

    def memory_usage(self, index=True, deep=False):
        return self._reduce_dimension(
            self._query_compiler.memory_usage(index=index, deep=deep)
        )

    def min(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
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
        return self._binary_op(
            "mod", other, axis=axis, level=level, fill_value=fill_value
        )

    def mode(self, axis=0, numeric_only=False, dropna=True):
        axis = self._get_axis_number(axis)
        return self.__constructor__(
            query_compiler=self._query_compiler.mode(
                axis=axis, numeric_only=numeric_only, dropna=dropna
            )
        )

    def mul(self, other, axis="columns", level=None, fill_value=None):
        return self._binary_op(
            "mul", other, axis=axis, level=level, fill_value=fill_value
        )

    multiply = mul

    def ne(self, other, axis="columns", level=None):
        return self._binary_op("ne", other, axis=axis, level=level)

    def notna(self):
        return self.__constructor__(query_compiler=self._query_compiler.notna())

    notnull = notna

    def nunique(self, axis=0, dropna=True):
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
        return pipe(self, func, *args, **kwargs)

    def pop(self, item):
        result = self[item]
        del self[item]
        return result

    def pow(self, other, axis="columns", level=None, fill_value=None):
        return self._binary_op(
            "pow", other, axis=axis, level=level, fill_value=fill_value
        )

    radd = add

    def quantile(self, q=0.5, axis=0, numeric_only=True, interpolation="linear"):
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
        return self.__constructor__(
            query_compiler=self._query_compiler.round(decimals=decimals, **kwargs)
        )

    def rpow(self, other, axis="columns", level=None, fill_value=None):
        return self._binary_op(
            "rpow", other, axis=axis, level=level, fill_value=fill_value
        )

    def rsub(self, other, axis="columns", level=None, fill_value=None):
        return self._binary_op(
            "rsub", other, axis=axis, level=level, fill_value=fill_value
        )

    def rtruediv(self, other, axis="columns", level=None, fill_value=None):
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

    def sem(
        self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs
    ):
        return self._stat_operation(
            "sem", axis, skipna, level, numeric_only, ddof=ddof, **kwargs
        )

    def set_axis(self, labels, axis=0, inplace=False):
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
                if not isinstance(self, DataFrame):
                    raise ValueError(
                        f"No axis named {axis} for object type {type(self)}"
                    )
                res_columns = self.columns
                from .general import concat

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

    def skew(self, axis=None, skipna=None, level=None, numeric_only=None, **kwargs):
        return self._stat_operation("skew", axis, skipna, level, numeric_only, **kwargs)

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

    def std(
        self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs
    ):
        return self._stat_operation(
            "std", axis, skipna, level, numeric_only, ddof=ddof, **kwargs
        )

    def sub(self, other, axis="columns", level=None, fill_value=None):
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

    def var(
        self, axis=None, skipna=None, level=None, ddof=1, numeric_only=None, **kwargs
    ):
        return self._stat_operation(
            "var", axis, skipna, level, numeric_only, ddof=ddof, **kwargs
        )

    def __abs__(self):
        return self.abs()

    def __and__(self, other):
        return self._binary_op("__and__", other, axis=0)

    def __rand__(self, other):
        return self._binary_op("__rand__", other, axis=0)

    def __array__(self, dtype=None):
        arr = self.to_numpy(dtype)
        return arr

    def __array_wrap__(self, result, context=None):
        # TODO: This is very inefficient. __array__ and as_matrix have been
        # changed to call the more efficient to_numpy, but this has been left
        # unchanged since we are not sure of its purpose.
        return self._default_to_pandas("__array_wrap__", result, context=context)

    def __copy__(self, deep=True):
        return self.copy(deep=deep)

    def __deepcopy__(self, memo=None):
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
        return len(self.index)

    def __lt__(self, right):
        return self.lt(right)

    def __matmul__(self, other):
        return self.dot(other)

    def __ne__(self, other):
        return self.ne(other)

    def __neg__(self):
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
        return len(self._query_compiler.index) * len(self._query_compiler.columns)

    @property
    def values(self):
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


if IsExperimental.get():
    from modin.experimental.cloud.meta_magic import make_wrapped_class

    make_wrapped_class(BasePandasDataset, "make_base_dataset_wrapper")


@_inherit_docstrings(
    pandas.core.resample.Resampler,
    excluded=[
        pandas.core.resample.Resampler.__init__,
    ],
)
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


@_inherit_docstrings(
    pandas.core.window.rolling.Window,
    excluded=[
        pandas.core.window.rolling.Window.__init__,
    ],
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


@_inherit_docstrings(
    pandas.core.window.rolling.Rolling,
    excluded=[
        pandas.core.window.rolling.Rolling.__init__,
    ],
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
