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

"""Implement DataFrame/Series public API as pandas does."""

from __future__ import annotations

import abc
import pickle as pkl
import re
import warnings
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Hashable,
    Literal,
    Optional,
    Sequence,
    Union,
)

import numpy as np
import pandas
import pandas.core.generic
import pandas.core.resample
import pandas.core.window.rolling
from pandas._libs import lib
from pandas._libs.tslibs import to_offset
from pandas._typing import (
    Axis,
    CompressionOptions,
    DtypeBackend,
    IndexKeyFunc,
    IndexLabel,
    Level,
    RandomState,
    Scalar,
    StorageOptions,
    T,
    TimedeltaConvertibleTypes,
    TimestampConvertibleTypes,
    npt,
)
from pandas.compat import numpy as numpy_compat
from pandas.core.common import count_not_none, pipe
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_dict_like,
    is_dtype_equal,
    is_integer,
    is_integer_dtype,
    is_list_like,
    is_numeric_dtype,
    is_object_dtype,
)
from pandas.core.indexes.api import ensure_index
from pandas.core.methods.describe import _refine_percentiles
from pandas.util._validators import (
    validate_ascending,
    validate_bool_kwarg,
    validate_percentile,
)

from modin import pandas as pd
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger, disable_logging
from modin.pandas.accessor import CachedAccessor, ModinAPI
from modin.pandas.utils import is_scalar
from modin.utils import _inherit_docstrings, expanduser_path_arg, try_cast_to_pandas

from .utils import _doc_binary_op, is_full_grab_slice

if TYPE_CHECKING:
    from typing_extensions import Self

    from modin.core.storage_formats import BaseQueryCompiler

    from .dataframe import DataFrame
    from .indexing import _iLocIndexer, _LocIndexer
    from .resample import Resampler
    from .series import Series
    from .window import Expanding, Rolling, Window

# Similar to pandas, sentinel value to use as kwarg in place of None when None has
# special meaning and needs to be distinguished from a user explicitly passing None.
sentinel = object()

# Do not lookup certain attributes in columns or index, as they're used for some
# special purposes, like serving remote context
_ATTRS_NO_LOOKUP = {
    "__name__",
    "_cache",
    "_ipython_canary_method_should_not_exist_",
    "_ipython_display_",
    "_repr_mimebundle_",
}

_DEFAULT_BEHAVIOUR = {
    "__init__",
    "__class__",
    "_get_index",
    "_set_index",
    "_pandas_class",
    "_get_axis_number",
    "empty",
    "index",
    "columns",
    "name",
    "dtypes",
    "dtype",
    "groupby",
    "_get_name",
    "_set_name",
    "_default_to_pandas",
    "_query_compiler",
    "_to_pandas",
    "_repartition",
    "_build_repr_df",
    "_reduce_dimension",
    "__repr__",
    "__len__",
    "__constructor__",
    "_create_or_update_from_compiler",
    "_update_inplace",
    # for persistance support;
    # see DataFrame methods docstrings for more
    "_inflate_light",
    "_inflate_full",
    "__reduce__",
    "__reduce_ex__",
    "_init",
} | _ATTRS_NO_LOOKUP

_doc_binary_op_kwargs = {"returns": "BasePandasDataset", "left": "BasePandasDataset"}


def _get_repr_axis_label_indexer(labels, num_for_repr):
    """
    Get the indexer for the given axis labels to be used for the repr.

    Parameters
    ----------
    labels : pandas.Index
        The axis labels.
    num_for_repr : int
        The number of elements to display.

    Returns
    -------
    slice or list
        The indexer to use for the repr.
    """
    if len(labels) <= num_for_repr:
        return slice(None)
    # At this point, the entire axis has len(labels) elements, and num_for_repr <
    # len(labels). We want to select a pandas subframe containing elements such that:
    #   - the repr of the pandas subframe will be the same as the repr of the entire
    #     frame.
    #   - the pandas repr will not be able to show all the elements and will put an
    #      ellipsis in the middle
    #
    # We accomplish this by selecting some elements from the front and some from the
    # back, with the front having at most 1 element more than the back. The total
    # number of elements will be num_for_repr + 1.

    if num_for_repr % 2 == 0:
        # If num_for_repr is even, take an extra element from the front.
        # The total number of elements we are selecting is (num_for_repr // 2) * 2 + 1
        # = num_for_repr + 1
        front_repr_num = num_for_repr // 2 + 1
        back_repr_num = num_for_repr // 2
    else:
        # If num_for_repr is odd, take an extra element from both the front and the
        # back. The total number of elements we are selecting is
        # (num_for_repr // 2) * 2 + 1 + 1 = num_for_repr + 1
        front_repr_num = num_for_repr // 2 + 1
        back_repr_num = num_for_repr // 2 + 1
    all_positions = range(len(labels))
    return list(all_positions[:front_repr_num]) + (
        [] if back_repr_num == 0 else list(all_positions[-back_repr_num:])
    )


@_inherit_docstrings(pandas.DataFrame, apilink=["pandas.DataFrame", "pandas.Series"])
class BasePandasDataset(ClassLogger):
    """
    Implement most of the common code that exists in DataFrame/Series.

    Since both objects share the same underlying representation, and the algorithms
    are the same, we use this object to define the general behavior of those objects
    and then use those objects to define the output type.
    """

    # Pandas class that we pretend to be; usually it has the same name as our class
    # but lives in "pandas" namespace.
    _pandas_class = pandas.core.generic.NDFrame
    _query_compiler: BaseQueryCompiler
    _siblings: list[BasePandasDataset]

    @cached_property
    def _is_dataframe(self) -> bool:
        """
        Tell whether this is a dataframe.

        Ideally, other methods of BasePandasDataset shouldn't care whether this
        is a dataframe or a series, but sometimes we need to know. This method
        is better than hasattr(self, "columns"), which for series will call
        self.__getattr__("columns"), which requires materializing the index.

        Returns
        -------
        bool : Whether this is a dataframe.
        """
        return issubclass(self._pandas_class, pandas.DataFrame)

    @abc.abstractmethod
    def _create_or_update_from_compiler(
        self, new_query_compiler: BaseQueryCompiler, inplace: bool = False
    ) -> Self | None:
        """
        Return or update a ``DataFrame`` or ``Series`` with given `new_query_compiler`.

        Parameters
        ----------
        new_query_compiler : BaseQueryCompiler
            QueryCompiler to use to manage the data.
        inplace : bool, default: False
            Whether or not to perform update or creation inplace.

        Returns
        -------
        DataFrame, Series or None
            None if update was done, ``DataFrame`` or ``Series`` otherwise.
        """
        pass

    def _add_sibling(self, sibling: BasePandasDataset) -> None:
        """
        Add a DataFrame or Series object to the list of siblings.

        Siblings are objects that share the same query compiler. This function is called
        when a shallow copy is made.

        Parameters
        ----------
        sibling : BasePandasDataset
            Dataset to add to siblings list.
        """
        sibling._siblings = self._siblings + [self]
        self._siblings += [sibling]
        for sib in self._siblings:
            sib._siblings += [sibling]

    def _build_repr_df(
        self, num_rows: int, num_cols: int
    ) -> pandas.DataFrame | pandas.Series:
        """
        Build pandas DataFrame for string representation.

        Parameters
        ----------
        num_rows : int
            Number of rows to show in string representation. If number of
            rows in this dataset is greater than `num_rows` then half of
            `num_rows` rows from the beginning and half of `num_rows` rows
            from the end are shown.
        num_cols : int
            Number of columns to show in string representation. If number of
            columns in this dataset is greater than `num_cols` then half of
            `num_cols` columns from the beginning and half of `num_cols`
            columns from the end are shown.

        Returns
        -------
        pandas.DataFrame or pandas.Series
            A pandas dataset with `num_rows` or fewer rows and `num_cols` or fewer columns.
        """
        # Fast track for empty dataframe.
        if len(self) == 0 or (
            self._is_dataframe and self._query_compiler.get_axis_len(1) == 0
        ):
            return pandas.DataFrame(
                index=self.index,
                columns=self.columns if self._is_dataframe else None,
            )
        row_indexer = _get_repr_axis_label_indexer(self.index, num_rows)
        if self._is_dataframe:
            indexer = row_indexer, _get_repr_axis_label_indexer(self.columns, num_cols)
        else:
            indexer = row_indexer
        return self.iloc[indexer]._query_compiler.to_pandas()

    def _update_inplace(self, new_query_compiler: BaseQueryCompiler) -> None:
        """
        Update the current DataFrame inplace.

        Parameters
        ----------
        new_query_compiler : BaseQueryCompiler
            The new QueryCompiler to use to manage the data.
        """
        old_query_compiler = self._query_compiler
        self._query_compiler = new_query_compiler
        for sib in self._siblings:
            sib._query_compiler = new_query_compiler
        old_query_compiler.free()

    def _validate_other(
        self,
        other,
        axis,
        dtype_check=False,
        compare_index=False,
    ):
        """
        Help to check validity of other in inter-df operations.

        Parameters
        ----------
        other : modin.pandas.BasePandasDataset
            Another dataset to validate against `self`.
        axis : {None, 0, 1}
            Specifies axis along which to do validation. When `1` or `None`
            is specified, validation is done along `index`, if `0` is specified
            validation is done along `columns` of `other` frame.
        dtype_check : bool, default: False
            Validates that both frames have compatible dtypes.
        compare_index : bool, default: False
            Compare Index if True.

        Returns
        -------
        BaseQueryCompiler or Any
            Other frame if it is determined to be valid.

        Raises
        ------
        ValueError
            If `other` is `Series` and its length is different from
            length of `self` `axis`.
        TypeError
            If any validation checks fail.
        """
        if isinstance(other, BasePandasDataset):
            return other._query_compiler
        if not is_list_like(other):
            # We skip dtype checking if the other is a scalar. Note that pandas
            # is_scalar can be misleading as it is False for almost all objects,
            # even when those objects should be treated as scalars. See e.g.
            # https://github.com/modin-project/modin/issues/5236. Therefore, we
            # detect scalars by checking that `other` is neither a list-like nor
            # another BasePandasDataset.
            return other
        axis = self._get_axis_number(axis) if axis is not None else 1
        result = other
        if axis == 0:
            if len(other) != len(self._query_compiler.index):
                raise ValueError(
                    f"Unable to coerce to Series, length must be {len(self._query_compiler.index)}: "
                    + f"given {len(other)}"
                )
        else:
            if len(other) != len(self._query_compiler.columns):
                raise ValueError(
                    f"Unable to coerce to Series, length must be {len(self._query_compiler.columns)}: "
                    + f"given {len(other)}"
                )
        if hasattr(other, "dtype"):
            other_dtypes = [other.dtype] * len(other)
        elif is_dict_like(other):
            other_dtypes = [
                other[label] if pandas.isna(other[label]) else type(other[label])
                for label in self._get_axis(axis)
                # The binary operation is applied for intersection of axis labels
                # and dictionary keys. So filtering out extra keys.
                if label in other
            ]
        else:
            other_dtypes = [x if pandas.isna(x) else type(x) for x in other]
        if compare_index:
            if not self.index.equals(other.index):
                raise TypeError("Cannot perform operation with non-equal index")
        # Do dtype checking.
        if dtype_check:
            self_dtypes = self._get_dtypes()
            if is_dict_like(other):
                # The binary operation is applied for the intersection of axis labels
                # and dictionary keys. So filtering `self_dtypes` to match the `other`
                # dictionary.
                self_dtypes = [
                    dtype
                    for label, dtype in zip(self._get_axis(axis), self._get_dtypes())
                    if label in other
                ]

            # TODO(https://github.com/modin-project/modin/issues/5239):
            # this spuriously rejects other that is a list including some
            # custom type that can be added to self's elements.
            for self_dtype, other_dtype in zip(self_dtypes, other_dtypes):
                if not (
                    (is_numeric_dtype(self_dtype) and is_numeric_dtype(other_dtype))
                    or (is_numeric_dtype(self_dtype) and pandas.isna(other_dtype))
                    or (is_object_dtype(self_dtype) and is_object_dtype(other_dtype))
                    or (
                        lib.is_np_dtype(self_dtype, "mM")
                        and lib.is_np_dtype(self_dtype, "mM")
                    )
                    or is_dtype_equal(self_dtype, other_dtype)
                ):
                    raise TypeError("Cannot do operation with improper dtypes")
        return result

    def _validate_function(self, func, on_invalid=None) -> None:
        """
        Check the validity of the function which is intended to be applied to the frame.

        Parameters
        ----------
        func : object
        on_invalid : callable(str, cls), optional
            Function to call in case invalid `func` is met, `on_invalid` takes an error
            message and an exception type as arguments. If not specified raise an
            appropriate exception.
            **Note:** This parameter is a hack to concord with pandas error types.
        """

        def error_raiser(msg, exception=Exception):
            raise exception(msg)

        if on_invalid is None:
            on_invalid = error_raiser

        if isinstance(func, dict):
            [self._validate_function(fn, on_invalid) for fn in func.values()]
            return
            # We also could validate this, but it may be quite expensive for lazy-frames
            # if not all(idx in self._get_axis(axis) for idx in func.keys()):
            #     error_raiser("Invalid dict keys", KeyError)

        if not is_list_like(func):
            func = [func]

        for fn in func:
            if isinstance(fn, str):
                if not (hasattr(self, fn) or hasattr(np, fn)):
                    on_invalid(
                        f"'{fn}' is not a valid function for '{type(self).__name__}' object",
                        AttributeError,
                    )
            elif not callable(fn):
                on_invalid(
                    f"One of the passed functions has an invalid type: {type(fn)}: {fn}, "
                    + "only callable or string is acceptable.",
                    TypeError,
                )

    def _binary_op(self, op, other, **kwargs) -> Self:
        """
        Do binary operation between two datasets.

        Parameters
        ----------
        op : str
            Name of binary operation.
        other : modin.pandas.BasePandasDataset
            Second operand of binary operation.
        **kwargs : dict
            Additional parameters to binary operation.

        Returns
        -------
        modin.pandas.BasePandasDataset
            Result of binary operation.
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
                getattr(self._pandas_class, op), other, **kwargs
            )
        other = self._validate_other(other, axis, dtype_check=True)
        exclude_list = [
            "__add__",
            "__radd__",
            "__and__",
            "__rand__",
            "__or__",
            "__ror__",
            "__xor__",
            "__rxor__",
        ]
        if op in exclude_list:
            kwargs.pop("axis")
        # Series logical operations take an additional fill_value argument that DF does not
        series_specialize_list = [
            "eq",
            "ge",
            "gt",
            "le",
            "lt",
            "ne",
        ]
        if not self._is_dataframe and op in series_specialize_list:
            op = "series_" + op
        new_query_compiler = getattr(self._query_compiler, op)(other, **kwargs)
        return self._create_or_update_from_compiler(new_query_compiler)

    def _default_to_pandas(self, op, *args, reason: str = None, **kwargs):
        """
        Convert dataset to pandas type and call a pandas function on it.

        Parameters
        ----------
        op : str
            Name of pandas function.
        *args : list
            Additional positional arguments to be passed to `op`.
        reason : str, optional
        **kwargs : dict
            Additional keywords arguments to be passed to `op`.

        Returns
        -------
        object
            Result of operation.
        """
        empty_self_str = "" if not self.empty else " for empty DataFrame"
        ErrorMessage.default_to_pandas(
            "`{}.{}`{}".format(
                type(self).__name__,
                op if isinstance(op, str) else op.__name__,
                empty_self_str,
            ),
            reason=reason,
        )

        args = try_cast_to_pandas(args)
        kwargs = try_cast_to_pandas(kwargs)
        pandas_obj = self._to_pandas()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            if callable(op):
                result = op(pandas_obj, *args, **kwargs)
            elif isinstance(op, str):
                # The inner `getattr` is ensuring that we are treating this object (whether
                # it is a DataFrame, Series, etc.) as a pandas object. The outer `getattr`
                # will get the operation (`op`) from the pandas version of the class and run
                # it on the object after we have converted it to pandas.
                attr = getattr(self._pandas_class, op)
                if isinstance(attr, property):
                    result = getattr(pandas_obj, op)
                else:
                    result = attr(pandas_obj, *args, **kwargs)
            else:
                ErrorMessage.catch_bugs_and_request_email(
                    failure_condition=True,
                    extra_log="{} is an unsupported operation".format(op),
                )
        if isinstance(result, pandas.DataFrame):
            from .dataframe import DataFrame

            return DataFrame(result)
        elif isinstance(result, pandas.Series):
            from .series import Series

            return Series(result)
        # inplace
        elif result is None:
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

    @classmethod
    def _get_axis_number(cls, axis) -> int:
        """
        Convert axis name or number to axis index.

        Parameters
        ----------
        axis : int, str or pandas._libs.lib.NoDefault
            Axis name ('index' or 'columns') or number to be converted to axis index.

        Returns
        -------
        int
            0 or 1 - axis index in the array of axes stored in the dataframe.
        """
        if axis is lib.no_default:
            axis = None

        return cls._pandas_class._get_axis_number(axis) if axis is not None else 0

    @cached_property
    def __constructor__(self) -> type[Self]:
        """
        Construct DataFrame or Series object depending on self type.

        Returns
        -------
        modin.pandas.BasePandasDataset
            Constructed object.
        """
        return type(self)

    def abs(self) -> Self:  # noqa: RT01, D200
        """
        Return a `BasePandasDataset` with absolute numeric value of each element.
        """
        self._validate_dtypes(numeric_only=True)
        return self.__constructor__(query_compiler=self._query_compiler.abs())

    def _set_index(self, new_index) -> None:
        """
        Set the index for this DataFrame.

        Parameters
        ----------
        new_index : pandas.Index
            The new index to set this.
        """
        self._query_compiler.index = new_index

    def _get_index(self) -> pandas.Index:
        """
        Get the index for this DataFrame.

        Returns
        -------
        pandas.Index
            The union of all indexes across the partitions.
        """
        return self._query_compiler.index

    index: pandas.Index = property(_get_index, _set_index)

    def _get_axis(self, axis) -> pandas.Index:
        """
        Return index labels of the specified axis.

        Parameters
        ----------
        axis : {0, 1}
            Axis to return labels on.
            0 is for index, when 1 is for columns.

        Returns
        -------
        pandas.Index
        """
        return self.index if axis == 0 else self.columns

    def add(
        self, other, axis="columns", level=None, fill_value=None
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Return addition of `BasePandasDataset` and `other`, element-wise (binary operator `add`).
        """
        return self._binary_op(
            "add", other, axis=axis, level=level, fill_value=fill_value
        )

    def aggregate(
        self, func=None, axis=0, *args, **kwargs
    ) -> DataFrame | Series | Scalar:  # noqa: PR01, RT01, D200
        """
        Aggregate using one or more operations over the specified axis.
        """
        axis = self._get_axis_number(axis)
        result = None

        if axis == 0:
            result = self._aggregate(func, _axis=axis, *args, **kwargs)
        # TODO: handle case when axis == 1
        if result is None:
            kwargs.pop("is_transform", None)
            return self.apply(func, axis=axis, args=args, **kwargs)
        return result

    agg: DataFrame | Series | Scalar = aggregate

    def _aggregate(self, func, *args, **kwargs):
        """
        Aggregate using one or more operations over index axis.

        Parameters
        ----------
        func : function, str, list or dict
            Function to use for aggregating the data.
        *args : list
            Positional arguments to pass to func.
        **kwargs : dict
            Keyword arguments to pass to func.

        Returns
        -------
        scalar or BasePandasDataset

        See Also
        --------
        aggregate : Aggregate along any axis.
        """
        _axis = kwargs.pop("_axis", 0)
        kwargs.pop("_level", None)

        if isinstance(func, str):
            kwargs.pop("is_transform", None)
            return self._string_function(func, *args, **kwargs)

        # Dictionaries have complex behavior because they can be renamed here.
        elif func is None or isinstance(func, dict):
            return self._default_to_pandas("agg", func, *args, **kwargs)
        kwargs.pop("is_transform", None)
        return self.apply(func, axis=_axis, args=args, **kwargs)

    def _string_function(self, func, *args, **kwargs):
        """
        Execute a function identified by its string name.

        Parameters
        ----------
        func : str
            Function name to call on `self`.
        *args : list
            Positional arguments to pass to func.
        **kwargs : dict
            Keyword arguments to pass to func.

        Returns
        -------
        object
            Function result.
        """
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

    def _get_dtypes(self) -> list:
        """
        Get dtypes as list.

        Returns
        -------
        list
            Either a one-element list that contains `dtype` if object denotes a Series
            or a list that contains `dtypes` if object denotes a DataFrame.
        """
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
        copy=None,
        fill_value=None,
        method=lib.no_default,
        limit=lib.no_default,
        fill_axis=lib.no_default,
        broadcast_axis=lib.no_default,
    ) -> tuple[Self, Self]:  # noqa: PR01, RT01, D200
        """
        Align two objects on their axes with the specified join method.
        """
        if (
            method is not lib.no_default
            or limit is not lib.no_default
            or fill_axis is not lib.no_default
        ):
            warnings.warn(
                "The 'method', 'limit', and 'fill_axis' keywords in "
                + f"{type(self).__name__}.align are deprecated and will be removed "
                + "in a future version. Call fillna directly on the returned objects "
                + "instead.",
                FutureWarning,
            )
        if fill_axis is lib.no_default:
            fill_axis = 0
        if method is lib.no_default:
            method = None
        if limit is lib.no_default:
            limit = None

        if broadcast_axis is not lib.no_default:
            msg = (
                f"The 'broadcast_axis' keyword in {type(self).__name__}.align is "
                + "deprecated and will be removed in a future version."
            )
            if broadcast_axis is not None:
                if self.ndim == 1 and other.ndim == 2:
                    msg += (
                        " Use left = DataFrame({col: left for col in right.columns}, "
                        + "index=right.index) before calling `left.align(right)` instead."
                    )
                elif self.ndim == 2 and other.ndim == 1:
                    msg += (
                        " Use right = DataFrame({col: right for col in left.columns}, "
                        + "index=left.index) before calling `left.align(right)` instead"
                    )
            warnings.warn(msg, FutureWarning)
        else:
            broadcast_axis = None

        left, right = self._query_compiler.align(
            other._query_compiler,
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
        return self.__constructor__(query_compiler=left), self.__constructor__(
            query_compiler=right
        )

    @abc.abstractmethod
    def _reduce_dimension(self, query_compiler: BaseQueryCompiler) -> Series | Scalar:
        """
        Reduce the dimension of data from the `query_compiler`.

        Parameters
        ----------
        query_compiler : BaseQueryCompiler
            Query compiler to retrieve the data.

        Returns
        -------
        Series | Scalar
        """
        pass

    def all(
        self, axis=0, bool_only=False, skipna=True, **kwargs
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Return whether all elements are True, potentially over an axis.
        """
        validate_bool_kwarg(skipna, "skipna", none_allowed=False)
        if axis is not None:
            axis = self._get_axis_number(axis)
            if bool_only and axis == 0:
                if hasattr(self, "dtype"):
                    raise NotImplementedError(
                        "{}.{} does not implement numeric_only.".format(
                            type(self).__name__, "all"
                        )
                    )
                data_for_compute = self[self.columns[self.dtypes == np.bool_]]
                return data_for_compute.all(
                    axis=axis, bool_only=False, skipna=skipna, **kwargs
                )
            return self._reduce_dimension(
                self._query_compiler.all(
                    axis=axis, bool_only=bool_only, skipna=skipna, **kwargs
                )
            )
        else:
            if bool_only:
                raise ValueError("Axis must be 0 or 1 (got {})".format(axis))
            # Reduce to a scalar if axis is None.
            result = self._reduce_dimension(
                # FIXME: Judging by pandas docs `**kwargs` serves only compatibility
                # purpose and does not affect the result, we shouldn't pass them to the query compiler.
                self._query_compiler.all(
                    axis=0,
                    bool_only=bool_only,
                    skipna=skipna,
                    **kwargs,
                )
            )
            if isinstance(result, BasePandasDataset):
                return result.all(
                    axis=axis, bool_only=bool_only, skipna=skipna, **kwargs
                )
            return result

    def any(
        self, *, axis=0, bool_only=False, skipna=True, **kwargs
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Return whether any element is True, potentially over an axis.
        """
        validate_bool_kwarg(skipna, "skipna", none_allowed=False)
        if axis is not None:
            axis = self._get_axis_number(axis)
            if bool_only and axis == 0:
                if hasattr(self, "dtype"):
                    raise NotImplementedError(
                        "{}.{} does not implement numeric_only.".format(
                            type(self).__name__, "all"
                        )
                    )
                data_for_compute = self[self.columns[self.dtypes == np.bool_]]
                return data_for_compute.any(
                    axis=axis, bool_only=False, skipna=skipna, **kwargs
                )
            return self._reduce_dimension(
                self._query_compiler.any(
                    axis=axis, bool_only=bool_only, skipna=skipna, **kwargs
                )
            )
        else:
            if bool_only:
                raise ValueError("Axis must be 0 or 1 (got {})".format(axis))
            # Reduce to a scalar if axis is None.
            result = self._reduce_dimension(
                self._query_compiler.any(
                    axis=0,
                    bool_only=bool_only,
                    skipna=skipna,
                    **kwargs,
                )
            )
            if isinstance(result, BasePandasDataset):
                return result.any(
                    axis=axis, bool_only=bool_only, skipna=skipna, **kwargs
                )
            return result

    def apply(
        self,
        func,
        axis,
        raw,
        result_type,
        args,
        **kwds,
    ) -> BaseQueryCompiler:  # noqa: PR01, RT01, D200
        """
        Apply a function along an axis of the `BasePandasDataset`.
        """

        def error_raiser(msg, exception):
            """Convert passed exception to the same type as pandas do and raise it."""
            # HACK: to concord with pandas error types by replacing all of the
            # TypeErrors to the AssertionErrors
            exception = exception if exception is not TypeError else AssertionError
            raise exception(msg)

        self._validate_function(func, on_invalid=error_raiser)
        axis = self._get_axis_number(axis)
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
            if self._query_compiler.get_axis_len(1) != len(set(self.columns)):
                warnings.warn(
                    "duplicate column names not supported with apply().",
                    FutureWarning,
                    stacklevel=2,
                )
        query_compiler = self._query_compiler.apply(
            func,
            axis,
            args=args,
            raw=raw,
            result_type=result_type,
            **kwds,
        )
        return query_compiler

    def asfreq(
        self, freq, method=None, how=None, normalize=False, fill_value=None
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Convert time series to specified frequency.
        """
        return self.__constructor__(
            query_compiler=self._query_compiler.asfreq(
                freq=freq,
                method=method,
                how=how,
                normalize=normalize,
                fill_value=fill_value,
            )
        )

    def asof(self, where, subset=None) -> Self:  # noqa: PR01, RT01, D200
        """
        Return the last row(s) without any NaNs before `where`.
        """
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

    def astype(
        self, dtype, copy=None, errors="raise"
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Cast a Modin object to a specified dtype `dtype`.
        """
        if copy is None:
            copy = True
        # dtype can be a series, a dict, or a scalar. If it's series,
        # convert it to a dict before passing it to the query compiler.
        if isinstance(dtype, (pd.Series, pandas.Series)):
            if not dtype.index.is_unique:
                raise ValueError("cannot reindex on an axis with duplicate labels")
            dtype = {column: dtype for column, dtype in dtype.items()}
        # If we got a series or dict originally, dtype is a dict now. Its keys
        # must be column names.
        if isinstance(dtype, dict):
            # avoid materializing columns in lazy mode. the query compiler
            # will handle errors where dtype dict includes keys that are not
            # in columns.
            if (
                not self._query_compiler.lazy_column_labels
                and not set(dtype.keys()).issubset(set(self._query_compiler.columns))
                and errors == "raise"
            ):
                raise KeyError(
                    "Only a column name can be used for the key in "
                    + "a dtype mappings argument."
                )

        if not copy:
            # If the new types match the old ones, then copying can be avoided
            if self._query_compiler.frame_has_materialized_dtypes:
                frame_dtypes = self._query_compiler.dtypes
                if isinstance(dtype, dict):
                    for col in dtype:
                        if dtype[col] != frame_dtypes[col]:
                            copy = True
                            break
                else:
                    if not (frame_dtypes == dtype).all():
                        copy = True
            else:
                copy = True

        if copy:
            new_query_compiler = self._query_compiler.astype(dtype, errors=errors)
            return self._create_or_update_from_compiler(new_query_compiler)
        return self

    @property
    def at(self, axis=None) -> _LocIndexer:  # noqa: PR01, RT01, D200
        """
        Get a single value for a row/column label pair.
        """
        from .indexing import _LocIndexer

        return _LocIndexer(self)

    def at_time(self, time, asof=False, axis=None) -> Self:  # noqa: PR01, RT01, D200
        """
        Select values at particular time of day (e.g., 9:30AM).
        """
        if asof:
            # pandas raises NotImplementedError for asof=True, so we do, too.
            raise NotImplementedError("'asof' argument is not supported")
        return self.between_time(
            start_time=time, end_time=time, inclusive="both", axis=axis
        )

    @_inherit_docstrings(
        pandas.DataFrame.between_time, apilink="pandas.DataFrame.between_time"
    )
    def between_time(
        self,
        start_time,
        end_time,
        inclusive="both",
        axis=None,
    ) -> Self:  # noqa: PR01, RT01, D200
        return self._create_or_update_from_compiler(
            self._query_compiler.between_time(
                start_time=pandas.core.tools.times.to_time(start_time),
                end_time=pandas.core.tools.times.to_time(end_time),
                inclusive=inclusive,
                axis=self._get_axis_number(axis),
            )
        )

    def _deprecate_downcast(self, downcast, method_name: str):  # noqa: GL08
        if downcast is not lib.no_default:
            warnings.warn(
                f"The 'downcast' keyword in {method_name} is deprecated and "
                + "will be removed in a future version. Use "
                + "res.infer_objects(copy=False) to infer non-object dtype, or "
                + "pd.to_numeric with the 'downcast' keyword to downcast numeric "
                + "results.",
                FutureWarning,
            )
        else:
            downcast = None
        return downcast

    def bfill(
        self,
        *,
        axis=None,
        inplace=False,
        limit=None,
        limit_area=None,
        downcast=lib.no_default,
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Synonym for `DataFrame.fillna` with ``method='bfill'``.
        """
        if limit_area is not None:
            return self._default_to_pandas(
                "bfill",
                reason="'limit_area' parameter isn't supported",
                axis=axis,
                inplace=inplace,
                limit=limit,
                limit_area=limit_area,
                downcast=downcast,
            )
        downcast = self._deprecate_downcast(downcast, "bfill")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", ".*fillna with 'method' is deprecated", category=FutureWarning
            )
            return self.fillna(
                method="bfill",
                axis=axis,
                limit=limit,
                downcast=downcast,
                inplace=inplace,
            )

    def backfill(
        self, *, axis=None, inplace=False, limit=None, downcast=lib.no_default
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Synonym for `DataFrame.bfill`.
        """
        warnings.warn(
            "DataFrame.backfill/Series.backfill is deprecated. Use DataFrame.bfill/Series.bfill instead",
            FutureWarning,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            return self.bfill(
                axis=axis, inplace=inplace, limit=limit, downcast=downcast
            )

    def bool(self) -> bool:  # noqa: RT01, D200
        """
        Return the bool of a single element `BasePandasDataset`.
        """
        warnings.warn(
            f"{type(self).__name__}.bool is now deprecated and will be removed "
            + "in future version of pandas",
            FutureWarning,
        )
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

    def clip(
        self, lower=None, upper=None, *, axis=None, inplace=False, **kwargs
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Trim values at input threshold(s).
        """
        # validate inputs
        if axis is not None:
            axis = self._get_axis_number(axis)
        self._validate_dtypes(numeric_only=True)
        inplace = validate_bool_kwarg(inplace, "inplace")
        axis = numpy_compat.function.validate_clip_with_axis(axis, (), kwargs)
        # any np.nan bounds are treated as None
        if lower is not None and np.any(np.isnan(lower)):
            lower = None
        if upper is not None and np.any(np.isnan(upper)):
            upper = None
        if is_list_like(lower) or is_list_like(upper):
            lower = self._validate_other(lower, axis)
            upper = self._validate_other(upper, axis)
        # FIXME: Judging by pandas docs `*args` and `**kwargs` serves only compatibility
        # purpose and does not affect the result, we shouldn't pass them to the query compiler.
        new_query_compiler = self._query_compiler.clip(
            lower=lower, upper=upper, axis=axis, **kwargs
        )
        return self._create_or_update_from_compiler(new_query_compiler, inplace)

    def combine(
        self, other, func, fill_value=None, **kwargs
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Perform combination of `BasePandasDataset`-s according to `func`.
        """
        return self._binary_op(
            "combine", other, _axis=0, func=func, fill_value=fill_value, **kwargs
        )

    def combine_first(self, other) -> Self:  # noqa: PR01, RT01, D200
        """
        Update null elements with value in the same location in `other`.
        """
        return self._binary_op("combine_first", other, _axis=0)

    def copy(self, deep=True) -> Self:  # noqa: PR01, RT01, D200
        """
        Make a copy of the object's metadata.
        """
        if deep:
            return self.__constructor__(query_compiler=self._query_compiler.copy())
        new_obj = self.__constructor__(query_compiler=self._query_compiler)
        self._add_sibling(new_obj)
        return new_obj

    def count(
        self, axis=0, numeric_only=False
    ) -> Series | Scalar:  # noqa: PR01, RT01, D200
        """
        Count non-NA cells for `BasePandasDataset`.
        """
        axis = self._get_axis_number(axis)
        # select_dtypes is only implemented on DataFrames, but the numeric_only
        # flag will always be set to false by the Series frontend
        frame = self.select_dtypes([np.number, np.bool_]) if numeric_only else self

        return frame._reduce_dimension(
            frame._query_compiler.count(axis=axis, numeric_only=numeric_only)
        )

    def cummax(
        self, axis=None, skipna=True, *args, **kwargs
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Return cumulative maximum over a `BasePandasDataset` axis.
        """
        axis = self._get_axis_number(axis)
        if axis == 1:
            self._validate_dtypes(numeric_only=True)
        return self.__constructor__(
            # FIXME: Judging by pandas docs `*args` and `**kwargs` serves only compatibility
            # purpose and does not affect the result, we shouldn't pass them to the query compiler.
            query_compiler=self._query_compiler.cummax(
                fold_axis=axis, axis=axis, skipna=skipna, **kwargs
            )
        )

    def cummin(
        self, axis=None, skipna=True, *args, **kwargs
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Return cumulative minimum over a `BasePandasDataset` axis.
        """
        axis = self._get_axis_number(axis)
        if axis == 1:
            self._validate_dtypes(numeric_only=True)
        return self.__constructor__(
            # FIXME: Judging by pandas docs `*args` and `**kwargs` serves only compatibility
            # purpose and does not affect the result, we shouldn't pass them to the query compiler.
            query_compiler=self._query_compiler.cummin(
                fold_axis=axis, axis=axis, skipna=skipna, **kwargs
            )
        )

    def cumprod(
        self, axis=None, skipna=True, *args, **kwargs
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Return cumulative product over a `BasePandasDataset` axis.
        """
        axis = self._get_axis_number(axis)
        self._validate_dtypes(numeric_only=True)
        return self.__constructor__(
            # FIXME: Judging by pandas docs `**kwargs` serves only compatibility
            # purpose and does not affect the result, we shouldn't pass them to the query compiler.
            query_compiler=self._query_compiler.cumprod(
                fold_axis=axis, axis=axis, skipna=skipna, **kwargs
            )
        )

    def cumsum(
        self, axis=None, skipna=True, *args, **kwargs
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Return cumulative sum over a `BasePandasDataset` axis.
        """
        axis = self._get_axis_number(axis)
        self._validate_dtypes(numeric_only=True)
        return self.__constructor__(
            # FIXME: Judging by pandas docs `*args` and `**kwargs` serves only compatibility
            # purpose and does not affect the result, we shouldn't pass them to the query compiler.
            query_compiler=self._query_compiler.cumsum(
                fold_axis=axis, axis=axis, skipna=skipna, **kwargs
            )
        )

    def describe(
        self,
        percentiles=None,
        include=None,
        exclude=None,
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Generate descriptive statistics.
        """
        # copied from pandas.core.describe.describe_ndframe
        percentiles = _refine_percentiles(percentiles)
        data = self
        if self._is_dataframe:
            # include/exclude arguments are ignored for Series
            if (include is None) and (exclude is None):
                # when some numerics are found, keep only numerics
                default_include: list[npt.DTypeLike] = [np.number]
                default_include.append("datetime")
                data = self.select_dtypes(include=default_include)
                if len(data.columns) == 0:
                    data = self
            elif include == "all":
                if exclude is not None:
                    msg = "exclude must be None when include is 'all'"
                    raise ValueError(msg)
                data = self
            else:
                data = self.select_dtypes(
                    include=include,
                    exclude=exclude,
                )
        if data.empty:
            # Match pandas error from concatenting empty list of series descriptions.
            raise ValueError("No objects to concatenate")
        return self.__constructor__(
            query_compiler=data._query_compiler.describe(percentiles=percentiles)
        )

    def diff(self, periods=1, axis=0) -> Self:  # noqa: PR01, RT01, D200
        """
        First discrete difference of element.
        """
        # Attempting to match pandas error behavior here
        if not isinstance(periods, int):
            raise ValueError(f"periods must be an int. got {type(periods)} instead")

        # Attempting to match pandas error behavior here
        for dtype in self._get_dtypes():
            if not (is_numeric_dtype(dtype) or lib.is_np_dtype(dtype, "mM")):
                raise TypeError(f"unsupported operand type for -: got {dtype}")

        axis = self._get_axis_number(axis)
        return self.__constructor__(
            query_compiler=self._query_compiler.diff(axis=axis, periods=periods)
        )

    def drop(
        self,
        labels=None,
        *,
        axis=0,
        index=None,
        columns=None,
        level=None,
        inplace=False,
        errors="raise",
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Drop specified labels from `BasePandasDataset`.
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
            axis_name = pandas.DataFrame._get_axis_name(axis)
            axes = {axis_name: labels}
        elif index is not None or columns is not None:
            axes = {"index": index}
            if self.ndim == 2:
                axes["columns"] = columns
        else:
            raise ValueError(
                "Need to specify at least one of 'labels', 'index' or 'columns'"
            )

        for axis in ["index", "columns"]:
            if axis not in axes:
                axes[axis] = None
            elif axes[axis] is not None:
                if not is_list_like(axes[axis]):
                    axes[axis] = [axes[axis]]
                # In case of lazy execution we should bypass these error checking components
                # because they can force the materialization of the row or column labels.
                if (axis == "index" and self._query_compiler.lazy_row_labels) or (
                    axis == "columns" and self._query_compiler.lazy_column_labels
                ):
                    continue
                if errors == "raise":
                    non_existent = pandas.Index(axes[axis]).difference(
                        getattr(self, axis)
                    )
                    if len(non_existent):
                        raise KeyError(f"labels {non_existent} not contained in axis")
                else:
                    axes[axis] = [
                        obj for obj in axes[axis] if obj in getattr(self, axis)
                    ]
                    # If the length is zero, we will just do nothing
                    if not len(axes[axis]):
                        axes[axis] = None

        new_query_compiler = self._query_compiler.drop(
            index=axes["index"], columns=axes["columns"], errors=errors
        )
        return self._create_or_update_from_compiler(new_query_compiler, inplace)

    def dropna(
        self,
        *,
        axis: Axis = 0,
        how: str | lib.NoDefault = lib.no_default,
        thresh: int | lib.NoDefault = lib.no_default,
        subset: IndexLabel = None,
        inplace: bool = False,
        ignore_index: bool = False,
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Remove missing values.
        """
        inplace = validate_bool_kwarg(inplace, "inplace")

        if is_list_like(axis):
            raise TypeError("supplying multiple axes to axis is no longer supported.")

        axis = self._get_axis_number(axis)
        if how is not None and how not in ["any", "all", lib.no_default]:
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
        if ignore_index:
            new_query_compiler.index = pandas.RangeIndex(
                stop=len(new_query_compiler.index)
            )
        return self._create_or_update_from_compiler(new_query_compiler, inplace)

    def droplevel(self, level, axis=0) -> Self:  # noqa: PR01, RT01, D200
        """
        Return `BasePandasDataset` with requested index / column level(s) removed.
        """
        axis = self._get_axis_number(axis)
        result = self.copy()
        if axis == 0:
            index_columns = result.index.names.copy()
            if is_integer(level):
                level = index_columns[level]
            elif is_list_like(level):
                level = [
                    index_columns[lev] if is_integer(lev) else lev for lev in level
                ]
            if is_list_like(level):
                for lev in level:
                    index_columns.remove(lev)
            else:
                index_columns.remove(level)
            if len(result.columns.names) > 1:
                # In this case, we are dealing with a MultiIndex column, so we need to
                # be careful when dropping the additional index column.
                if is_list_like(level):
                    drop_labels = [(lev, "") for lev in level]
                else:
                    drop_labels = [(level, "")]
                result = result.reset_index().drop(columns=drop_labels)
            else:
                result = result.reset_index().drop(columns=level)
            result = result.set_index(index_columns)
        else:
            result.columns = self.columns.droplevel(level)
        return result

    def drop_duplicates(
        self, keep="first", inplace=False, **kwargs
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Return `BasePandasDataset` with duplicate rows removed.
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        ignore_index = kwargs.get("ignore_index", False)
        subset = kwargs.get("subset", None)
        if subset is not None:
            if is_list_like(subset):
                if not isinstance(subset, list):
                    subset = list(subset)
            else:
                subset = [subset]
            if len(diff := pandas.Index(subset).difference(self.columns)) > 0:
                raise KeyError(diff)
        result_qc = self._query_compiler.unique(
            keep=keep, ignore_index=ignore_index, subset=subset
        )
        result = self.__constructor__(query_compiler=result_qc)
        if inplace:
            self._update_inplace(result._query_compiler)
        else:
            return result

    def eq(self, other, axis="columns", level=None) -> Self:  # noqa: PR01, RT01, D200
        """
        Get equality of `BasePandasDataset` and `other`, element-wise (binary operator `eq`).
        """
        return self._binary_op("eq", other, axis=axis, level=level, dtypes=np.bool_)

    def explode(
        self, column, ignore_index: bool = False
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Transform each element of a list-like to a row.
        """
        exploded = self.__constructor__(
            query_compiler=self._query_compiler.explode(column)
        )
        if ignore_index:
            exploded = exploded.reset_index(drop=True)
        return exploded

    def ewm(
        self,
        com: float | None = None,
        span: float | None = None,
        halflife: float | TimedeltaConvertibleTypes | None = None,
        alpha: float | None = None,
        min_periods: int | None = 0,
        adjust: bool = True,
        ignore_na: bool = False,
        axis: Axis = lib.no_default,
        times: str | np.ndarray | BasePandasDataset | None = None,
        method: str = "single",
    ) -> pandas.core.window.ewm.ExponentialMovingWindow:  # noqa: PR01, RT01, D200
        """
        Provide exponentially weighted (EW) calculations.
        """
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
            method=method,
        )

    def expanding(
        self, min_periods=1, axis=lib.no_default, method="single"
    ) -> Expanding:  # noqa: PR01, RT01, D200
        """
        Provide expanding window calculations.
        """
        from .window import Expanding

        if axis is not lib.no_default:
            axis = self._get_axis_number(axis)
            name = "expanding"
            if axis == 1:
                warnings.warn(
                    f"Support for axis=1 in {type(self).__name__}.{name} is "
                    + "deprecated and will be removed in a future version. "
                    + f"Use obj.T.{name}(...) instead",
                    FutureWarning,
                )
            else:
                warnings.warn(
                    f"The 'axis' keyword in {type(self).__name__}.{name} is "
                    + "deprecated and will be removed in a future version. "
                    + "Call the method without the axis keyword instead.",
                    FutureWarning,
                )
        else:
            axis = 0

        return Expanding(
            self,
            min_periods=min_periods,
            axis=axis,
            method=method,
        )

    def ffill(
        self,
        *,
        axis=None,
        inplace=False,
        limit=None,
        limit_area=None,
        downcast=lib.no_default,
    ) -> Self | None:  # noqa: PR01, RT01, D200
        """
        Synonym for `DataFrame.fillna` with ``method='ffill'``.
        """
        if limit_area is not None:
            return self._default_to_pandas(
                "ffill",
                reason="'limit_area' parameter isn't supported",
                axis=axis,
                inplace=inplace,
                limit=limit,
                limit_area=limit_area,
                downcast=downcast,
            )
        downcast = self._deprecate_downcast(downcast, "ffill")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", ".*fillna with 'method' is deprecated", category=FutureWarning
            )
            return self.fillna(
                method="ffill",
                axis=axis,
                limit=limit,
                downcast=downcast,
                inplace=inplace,
            )

    def pad(
        self, *, axis=None, inplace=False, limit=None, downcast=lib.no_default
    ) -> Self | None:  # noqa: PR01, RT01, D200
        """
        Synonym for `DataFrame.ffill`.
        """
        warnings.warn(
            "DataFrame.pad/Series.pad is deprecated. Use DataFrame.ffill/Series.ffill instead",
            FutureWarning,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            return self.ffill(
                axis=axis, inplace=inplace, limit=limit, downcast=downcast
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
        downcast=lib.no_default,
    ) -> Self | None:
        """
        Fill NA/NaN values using the specified method.

        Parameters
        ----------
        squeeze_self : bool
            If True then self contains a Series object, if False then self contains
            a DataFrame object.
        squeeze_value : bool
            If True then value contains a Series object, if False then value contains
            a DataFrame object.
        value : scalar, dict, Series, or DataFrame, default: None
            Value to use to fill holes (e.g. 0), alternately a
            dict/Series/DataFrame of values specifying which value to use for
            each index (for a Series) or column (for a DataFrame).  Values not
            in the dict/Series/DataFrame will not be filled. This value cannot
            be a list.
        method : {'backfill', 'bfill', 'pad', 'ffill', None}, default: None
            Method to use for filling holes in reindexed Series
            pad / ffill: propagate last valid observation forward to next valid
            backfill / bfill: use next valid observation to fill gap.
        axis : {None, 0, 1}, default: None
            Axis along which to fill missing values.
        inplace : bool, default: False
            If True, fill in-place. Note: this will modify any
            other views on this object (e.g., a no-copy slice for a column in a
            DataFrame).
        limit : int, default: None
            If method is specified, this is the maximum number of consecutive
            NaN values to forward/backward fill. In other words, if there is
            a gap with more than this number of consecutive NaNs, it will only
            be partially filled. If method is not specified, this is the
            maximum number of entries along the entire axis where NaNs will be
            filled. Must be greater than 0 if not None.
        downcast : dict, default: None
            A dict of item->dtype of what to downcast if possible,
            or the string 'infer' which will try to downcast to an appropriate
            equal type (e.g. float64 to int64 if possible).

        Returns
        -------
        Series, DataFrame or None
            Object with missing values filled or None if ``inplace=True``.
        """
        if method is not None:
            warnings.warn(
                f"{type(self).__name__}.fillna with 'method' is deprecated and "
                + "will raise in a future version. Use obj.ffill() or obj.bfill() "
                + "instead.",
                FutureWarning,
            )
        downcast = self._deprecate_downcast(downcast, "fillna")
        inplace = validate_bool_kwarg(inplace, "inplace")
        axis = self._get_axis_number(axis)
        if isinstance(value, (list, tuple)):
            raise TypeError(
                '"value" parameter must be a scalar or dict, but '
                + f'you passed a "{type(value).__name__}"'
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

        if isinstance(value, BasePandasDataset):
            value = value._query_compiler

        new_query_compiler = self._query_compiler.fillna(
            squeeze_self=squeeze_self,
            squeeze_value=squeeze_value,
            value=value,
            method=method,
            axis=axis,
            inplace=False,
            limit=limit,
            downcast=downcast,
        )
        return self._create_or_update_from_compiler(new_query_compiler, inplace)

    def filter(
        self, items=None, like=None, regex=None, axis=None
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Subset the `BasePandasDataset` rows or columns according to the specified index labels.
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

    def first(self, offset) -> Self | None:  # noqa: PR01, RT01, D200
        """
        Select initial periods of time series data based on a date offset.
        """
        warnings.warn(
            "first is deprecated and will be removed in a future version. "
            + "Please create a mask and filter using `.loc` instead",
            FutureWarning,
        )
        return self._create_or_update_from_compiler(
            self._query_compiler.first(offset=to_offset(offset))
        )

    def first_valid_index(self) -> int:  # noqa: RT01, D200
        """
        Return index for first non-NA value or None, if no non-NA value is found.
        """
        return self._query_compiler.first_valid_index()

    def floordiv(
        self, other, axis="columns", level=None, fill_value=None
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Get integer division of `BasePandasDataset` and `other`, element-wise (binary operator `floordiv`).
        """
        return self._binary_op(
            "floordiv", other, axis=axis, level=level, fill_value=fill_value
        )

    def ge(self, other, axis="columns", level=None) -> Self:  # noqa: PR01, RT01, D200
        """
        Get greater than or equal comparison of `BasePandasDataset` and `other`, element-wise (binary operator `ge`).
        """
        return self._binary_op("ge", other, axis=axis, level=level, dtypes=np.bool_)

    def get(
        self, key, default=None
    ) -> DataFrame | Series | Scalar:  # noqa: PR01, RT01, D200
        """
        Get item from object for given key.
        """
        # Match pandas behavior here
        try:
            return self.__getitem__(key)
        except (KeyError, ValueError, IndexError):
            return default

    def gt(self, other, axis="columns", level=None) -> Self:  # noqa: PR01, RT01, D200
        """
        Get greater than comparison of `BasePandasDataset` and `other`, element-wise (binary operator `gt`).
        """
        return self._binary_op("gt", other, axis=axis, level=level, dtypes=np.bool_)

    def head(self, n=5) -> Self:  # noqa: PR01, RT01, D200
        """
        Return the first `n` rows.
        """
        return self.iloc[:n]

    @property
    def iat(self, axis=None) -> _iLocIndexer:  # noqa: PR01, RT01, D200
        """
        Get a single value for a row/column pair by integer position.
        """
        from .indexing import _iLocIndexer

        return _iLocIndexer(self)

    def idxmax(
        self, axis=0, skipna=True, numeric_only=False
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Return index of first occurrence of maximum over requested axis.
        """
        axis = self._get_axis_number(axis)
        return self._reduce_dimension(
            self._query_compiler.idxmax(
                axis=axis, skipna=skipna, numeric_only=numeric_only
            )
        )

    def idxmin(
        self, axis=0, skipna=True, numeric_only=False
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Return index of first occurrence of minimum over requested axis.
        """
        axis = self._get_axis_number(axis)
        return self._reduce_dimension(
            self._query_compiler.idxmin(
                axis=axis, skipna=skipna, numeric_only=numeric_only
            )
        )

    def infer_objects(self, copy=None) -> Self:  # noqa: PR01, RT01, D200
        """
        Attempt to infer better dtypes for object columns.
        """
        new_query_compiler = self._query_compiler.infer_objects()
        return self._create_or_update_from_compiler(
            new_query_compiler, inplace=False if copy is None else not copy
        )

    def convert_dtypes(
        self,
        infer_objects: bool = True,
        convert_string: bool = True,
        convert_integer: bool = True,
        convert_boolean: bool = True,
        convert_floating: bool = True,
        dtype_backend: DtypeBackend = "numpy_nullable",
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Convert columns to best possible dtypes using dtypes supporting ``pd.NA``.
        """
        return self.__constructor__(
            query_compiler=self._query_compiler.convert_dtypes(
                infer_objects=infer_objects,
                convert_string=convert_string,
                convert_integer=convert_integer,
                convert_boolean=convert_boolean,
                convert_floating=convert_floating,
                dtype_backend=dtype_backend,
            )
        )

    def isin(self, values) -> Self:  # noqa: PR01, RT01, D200
        """
        Whether elements in `BasePandasDataset` are contained in `values`.
        """
        from .series import Series

        ignore_indices = isinstance(values, Series)
        values = getattr(values, "_query_compiler", values)
        return self.__constructor__(
            query_compiler=self._query_compiler.isin(
                values=values, ignore_indices=ignore_indices
            )
        )

    def isna(self) -> Self:  # noqa: RT01, D200
        """
        Detect missing values.
        """
        return self.__constructor__(query_compiler=self._query_compiler.isna())

    isnull: Self = isna

    @property
    def iloc(self) -> _iLocIndexer:  # noqa: RT01, D200
        """
        Purely integer-location based indexing for selection by position.
        """
        from .indexing import _iLocIndexer

        return _iLocIndexer(self)

    @_inherit_docstrings(pandas.DataFrame.kurt, apilink="pandas.DataFrame.kurt")
    def kurt(self, axis=0, skipna=True, numeric_only=False, **kwargs) -> Series | float:
        return self._stat_operation("kurt", axis, skipna, numeric_only, **kwargs)

    kurtosis: Series | float = kurt

    def last(self, offset) -> Self:  # noqa: PR01, RT01, D200
        """
        Select final periods of time series data based on a date offset.
        """
        warnings.warn(
            "last is deprecated and will be removed in a future version. "
            + "Please create a mask and filter using `.loc` instead",
            FutureWarning,
        )

        return self._create_or_update_from_compiler(
            self._query_compiler.last(offset=to_offset(offset))
        )

    def last_valid_index(self) -> int:  # noqa: RT01, D200
        """
        Return index for last non-NA value or None, if no non-NA value is found.
        """
        return self._query_compiler.last_valid_index()

    def le(self, other, axis="columns", level=None) -> Self:  # noqa: PR01, RT01, D200
        """
        Get less than or equal comparison of `BasePandasDataset` and `other`, element-wise (binary operator `le`).
        """
        return self._binary_op("le", other, axis=axis, level=level, dtypes=np.bool_)

    def lt(self, other, axis="columns", level=None) -> Self:  # noqa: PR01, RT01, D200
        """
        Get less than comparison of `BasePandasDataset` and `other`, element-wise (binary operator `lt`).
        """
        return self._binary_op("lt", other, axis=axis, level=level, dtypes=np.bool_)

    @property
    def loc(self) -> _LocIndexer:  # noqa: RT01, D200
        """
        Get a group of rows and columns by label(s) or a boolean array.
        """
        from .indexing import _LocIndexer

        return _LocIndexer(self)

    def mask(
        self,
        cond,
        other=lib.no_default,
        *,
        inplace: bool = False,
        axis: Optional[Axis] = None,
        level: Optional[Level] = None,
    ) -> Self | None:  # noqa: PR01, RT01, D200
        """
        Replace values where the condition is True.
        """
        return self._create_or_update_from_compiler(
            self._query_compiler.mask(
                cond,
                other=other,
                inplace=False,
                axis=axis,
                level=level,
            ),
            inplace=inplace,
        )

    def max(
        self,
        axis: Axis = 0,
        skipna=True,
        numeric_only=False,
        **kwargs,
    ) -> Series | None:  # noqa: PR01, RT01, D200
        """
        Return the maximum of the values over the requested axis.
        """
        validate_bool_kwarg(skipna, "skipna", none_allowed=False)
        orig_axis = axis
        axis = self._get_axis_number(axis)
        data = self._validate_dtypes_min_max(axis, numeric_only)
        res = data._reduce_dimension(
            data._query_compiler.max(
                axis=axis,
                skipna=skipna,
                numeric_only=numeric_only,
                **kwargs,
            )
        )
        if orig_axis is None:
            res = res._reduce_dimension(
                res._query_compiler.max(
                    axis=0,
                    skipna=skipna,
                    numeric_only=False,
                    **kwargs,
                )
            )
        return res

    def min(
        self,
        axis: Axis = 0,
        skipna: bool = True,
        numeric_only=False,
        **kwargs,
    ) -> Series | None:  # noqa: PR01, RT01, D200
        """
        Return the minimum of the values over the requested axis.
        """
        validate_bool_kwarg(skipna, "skipna", none_allowed=False)
        orig_axis = axis
        axis = self._get_axis_number(axis)
        data = self._validate_dtypes_min_max(axis, numeric_only)
        res = data._reduce_dimension(
            data._query_compiler.min(
                axis=axis,
                skipna=skipna,
                numeric_only=numeric_only,
                **kwargs,
            )
        )
        if orig_axis is None:
            res = res._reduce_dimension(
                res._query_compiler.min(
                    axis=0,
                    skipna=skipna,
                    numeric_only=False,
                    **kwargs,
                )
            )
        return res

    def _stat_operation(
        self,
        op_name: str,
        axis: Optional[Union[int, str]],
        skipna: bool,
        numeric_only: Optional[bool] = False,
        **kwargs,
    ):
        """
        Do common statistic reduce operations under frame.

        Parameters
        ----------
        op_name : str
            Name of method to apply.
        axis : int or str
            Axis to apply method on.
        skipna : bool
            Exclude NA/null values when computing the result.
        numeric_only : bool, default: False
            Include only float, int, boolean columns. If None, will attempt
            to use everything, then use only numeric data.
        **kwargs : dict
            Additional keyword arguments to pass to `op_name`.

        Returns
        -------
        scalar, Series or DataFrame
            `scalar` - self is Series and level is not specified.
            `Series` - self is Series and level is specified, or
                self is DataFrame and level is not specified.
            `DataFrame` - self is DataFrame and level is specified.
        """
        axis = self._get_axis_number(axis) if axis is not None else None
        validate_bool_kwarg(skipna, "skipna", none_allowed=False)
        if op_name == "median":
            numpy_compat.function.validate_median((), kwargs)
        elif op_name in ("sem", "var", "std"):
            val_kwargs = {k: v for k, v in kwargs.items() if k != "ddof"}
            numpy_compat.function.validate_stat_ddof_func((), val_kwargs, fname=op_name)
        else:
            numpy_compat.function.validate_stat_func((), kwargs, fname=op_name)

        if not numeric_only:
            self._validate_dtypes(numeric_only=True)

        data = (
            self._get_numeric_data(axis if axis is not None else 0)
            if numeric_only
            else self
        )
        result_qc = getattr(data._query_compiler, op_name)(
            axis=axis,
            skipna=skipna,
            numeric_only=numeric_only,
            **kwargs,
        )
        return (
            self._reduce_dimension(result_qc)
            if isinstance(result_qc, type(self._query_compiler))
            # scalar case
            else result_qc
        )

    def memory_usage(
        self, index=True, deep=False
    ) -> Series | None:  # noqa: PR01, RT01, D200
        """
        Return the memory usage of the `BasePandasDataset`.
        """
        return self._reduce_dimension(
            self._query_compiler.memory_usage(index=index, deep=deep)
        )

    def mod(
        self, other, axis="columns", level=None, fill_value=None
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Get modulo of `BasePandasDataset` and `other`, element-wise (binary operator `mod`).
        """
        return self._binary_op(
            "mod", other, axis=axis, level=level, fill_value=fill_value
        )

    def mode(
        self, axis=0, numeric_only=False, dropna=True
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Get the mode(s) of each element along the selected axis.
        """
        axis = self._get_axis_number(axis)
        return self.__constructor__(
            query_compiler=self._query_compiler.mode(
                axis=axis, numeric_only=numeric_only, dropna=dropna
            )
        )

    def mul(
        self, other, axis="columns", level=None, fill_value=None
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Get multiplication of `BasePandasDataset` and `other`, element-wise (binary operator `mul`).
        """
        return self._binary_op(
            "mul", other, axis=axis, level=level, fill_value=fill_value
        )

    multiply: Self = mul

    def ne(self, other, axis="columns", level=None) -> Self:  # noqa: PR01, RT01, D200
        """
        Get Not equal comparison of `BasePandasDataset` and `other`, element-wise (binary operator `ne`).
        """
        return self._binary_op("ne", other, axis=axis, level=level, dtypes=np.bool_)

    def notna(self) -> Self:  # noqa: RT01, D200
        """
        Detect existing (non-missing) values.
        """
        return self.__constructor__(query_compiler=self._query_compiler.notna())

    notnull: Self = notna

    def nunique(self, axis=0, dropna=True) -> Series | int:  # noqa: PR01, RT01, D200
        """
        Return number of unique elements in the `BasePandasDataset`.
        """
        axis = self._get_axis_number(axis)
        return self._reduce_dimension(
            self._query_compiler.nunique(axis=axis, dropna=dropna)
        )

    def pct_change(
        self,
        periods=1,
        fill_method=lib.no_default,
        limit=lib.no_default,
        freq=None,
        **kwargs,
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Percentage change between the current and a prior element.
        """
        if fill_method not in (lib.no_default, None) or limit is not lib.no_default:
            warnings.warn(
                "The 'fill_method' keyword being not None and the 'limit' keyword in "
                + f"{type(self).__name__}.pct_change are deprecated and will be removed "
                + "in a future version. Either fill in any non-leading NA values prior "
                + "to calling pct_change or specify 'fill_method=None' to not fill NA "
                + "values.",
                FutureWarning,
            )
        if fill_method is lib.no_default:
            if self.isna().values.any():
                warnings.warn(
                    "The default fill_method='pad' in "
                    + f"{type(self).__name__}.pct_change is deprecated and will be "
                    + "removed in a future version. Call ffill before calling "
                    + "pct_change to retain current behavior and silence this warning.",
                    FutureWarning,
                )
            fill_method = "pad"
        if limit is lib.no_default:
            limit = None

        # Attempting to match pandas error behavior here
        if not isinstance(periods, int):
            raise ValueError(f"periods must be an int. got {type(periods)} instead")

        # Attempting to match pandas error behavior here
        for dtype in self._get_dtypes():
            if not is_numeric_dtype(dtype):
                raise TypeError(f"unsupported operand type for /: got {dtype}")

        return self.__constructor__(
            query_compiler=self._query_compiler.pct_change(
                periods=periods,
                fill_method=fill_method,
                limit=limit,
                freq=freq,
                **kwargs,
            )
        )

    def pipe(
        self, func: Callable[..., T] | tuple[Callable[..., T], str], *args, **kwargs
    ) -> T:  # noqa: PR01, RT01, D200
        """
        Apply chainable functions that expect `BasePandasDataset`.
        """
        return pipe(self, func, *args, **kwargs)

    def pop(self, item) -> Series | Scalar:  # noqa: PR01, RT01, D200
        """
        Return item and drop from frame. Raise KeyError if not found.
        """
        result = self[item]
        del self[item]
        return result

    def pow(
        self, other, axis="columns", level=None, fill_value=None
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Get exponential power of `BasePandasDataset` and `other`, element-wise (binary operator `pow`).
        """
        return self._binary_op(
            "pow", other, axis=axis, level=level, fill_value=fill_value
        )

    def quantile(
        self, q, axis, numeric_only, interpolation, method
    ) -> DataFrame | Series | Scalar:  # noqa: PR01, RT01, D200
        """
        Return values at the given quantile over requested axis.
        """
        axis = self._get_axis_number(axis)

        def check_dtype(t):
            return is_numeric_dtype(t) or lib.is_np_dtype(t, "mM")

        numeric_only_df = self
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
            numeric_only_df = self.drop(
                columns=[
                    i for i in self.dtypes.index if not is_numeric_dtype(self.dtypes[i])
                ]
            )

        # check that all qs are between 0 and 1
        validate_percentile(q)
        axis = numeric_only_df._get_axis_number(axis)
        if isinstance(q, (pandas.Series, np.ndarray, pandas.Index, list, tuple)):
            return numeric_only_df.__constructor__(
                query_compiler=numeric_only_df._query_compiler.quantile_for_list_of_values(
                    q=q,
                    axis=axis,
                    # `numeric_only=True` has already been processed by using `self.drop` function
                    numeric_only=False,
                    interpolation=interpolation,
                    method=method,
                )
            )
        else:
            result = numeric_only_df._reduce_dimension(
                numeric_only_df._query_compiler.quantile_for_single_value(
                    q=q,
                    axis=axis,
                    # `numeric_only=True` has already been processed by using `self.drop` function
                    numeric_only=False,
                    interpolation=interpolation,
                    method=method,
                )
            )
            if isinstance(result, BasePandasDataset):
                result.name = q
            return result

    @_inherit_docstrings(pandas.DataFrame.rank, apilink="pandas.DataFrame.rank")
    def rank(
        self,
        axis=0,
        method: str = "average",
        numeric_only=False,
        na_option: str = "keep",
        ascending: bool = True,
        pct: bool = False,
    ) -> Self:
        if axis is None:
            raise ValueError(
                f"No axis named None for object type {type(self).__name__}"
            )
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

    def _copy_index_metadata(self, source, destination):  # noqa: PR01, RT01, D200
        """
        Copy Index metadata from `source` to `destination` inplace.
        """
        if hasattr(source, "name") and hasattr(destination, "name"):
            destination.name = source.name
        if hasattr(source, "names") and hasattr(destination, "names"):
            destination.names = source.names
        return destination

    def _ensure_index(self, index_like, axis=0):  # noqa: PR01, RT01, D200
        """
        Ensure that we have an index from some index-like object.
        """
        if (
            self._query_compiler.has_multiindex(axis=axis)
            and not isinstance(index_like, pandas.Index)
            and is_list_like(index_like)
            and len(index_like) > 0
            and isinstance(index_like[0], tuple)
        ):
            try:
                return pandas.MultiIndex.from_tuples(index_like)
            except TypeError:
                # not all tuples
                pass
        return ensure_index(index_like)

    def reindex(
        self,
        index=None,
        columns=None,
        copy=True,
        **kwargs,
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Conform `BasePandasDataset` to new index with optional filling logic.
        """
        new_query_compiler = None
        if index is not None:
            if not isinstance(index, pandas.Index) or not index.equals(self.index):
                new_query_compiler = self._query_compiler.reindex(
                    axis=0, labels=index, **kwargs
                )
        if new_query_compiler is None:
            new_query_compiler = self._query_compiler
        final_query_compiler = None
        if columns is not None:
            if not isinstance(index, pandas.Index) or not columns.equals(self.columns):
                final_query_compiler = new_query_compiler.reindex(
                    axis=1, labels=columns, **kwargs
                )
        if final_query_compiler is None:
            final_query_compiler = new_query_compiler
        return self._create_or_update_from_compiler(
            final_query_compiler, inplace=False if copy is None else not copy
        )

    def rename_axis(
        self,
        mapper=lib.no_default,
        *,
        index=lib.no_default,
        columns=lib.no_default,
        axis=0,
        copy=None,
        inplace=False,
    ) -> DataFrame | Series | None:  # noqa: PR01, RT01, D200
        """
        Set the name of the axis for the index or columns.
        """
        axes = {"index": index, "columns": columns}

        if copy is None:
            copy = True

        if axis is not None:
            axis = self._get_axis_number(axis)

        inplace = validate_bool_kwarg(inplace, "inplace")

        if mapper is not lib.no_default:
            # Use v0.23 behavior if a scalar or list
            non_mapper = is_scalar(mapper) or (
                is_list_like(mapper) and not is_dict_like(mapper)
            )
            if non_mapper:
                return self._set_axis_name(mapper, axis=axis, inplace=inplace)
            else:
                raise ValueError("Use `.rename` to alter labels with a mapper.")
        else:
            # Use new behavior.  Means that index and/or columns is specified
            result = self if inplace else self.copy(deep=copy)

            for axis in range(self.ndim):
                v = axes.get(pandas.DataFrame._get_axis_name(axis))
                if v is lib.no_default:
                    continue
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

    def reorder_levels(self, order, axis=0) -> Self:  # noqa: PR01, RT01, D200
        """
        Rearrange index levels using input order.
        """
        axis = self._get_axis_number(axis)
        new_labels = self._get_axis(axis).reorder_levels(order)
        return self.set_axis(new_labels, axis=axis)

    def resample(
        self,
        rule,
        axis: Axis = lib.no_default,
        closed: Optional[str] = None,
        label: Optional[str] = None,
        convention: str = lib.no_default,
        kind: Optional[str] = lib.no_default,
        on: Level = None,
        level: Level = None,
        origin: str | TimestampConvertibleTypes = "start_day",
        offset: Optional[TimedeltaConvertibleTypes] = None,
        group_keys=False,
    ) -> Resampler:  # noqa: PR01, RT01, D200
        """
        Resample time-series data.
        """
        from .resample import Resampler

        if axis is not lib.no_default:
            axis = self._get_axis_number(axis)
            if axis == 1:
                warnings.warn(
                    "DataFrame.resample with axis=1 is deprecated. Do "
                    + "`frame.T.resample(...)` without axis instead.",
                    FutureWarning,
                )
            else:
                warnings.warn(
                    f"The 'axis' keyword in {type(self).__name__}.resample is "
                    + "deprecated and will be removed in a future version.",
                    FutureWarning,
                )
        else:
            axis = 0

        return Resampler(
            dataframe=self,
            rule=rule,
            axis=axis,
            closed=closed,
            label=label,
            convention=convention,
            kind=kind,
            on=on,
            level=level,
            origin=origin,
            offset=offset,
            group_keys=group_keys,
        )

    def reset_index(
        self,
        level: IndexLabel = None,
        *,
        drop: bool = False,
        inplace: bool = False,
        col_level: Hashable = 0,
        col_fill: Hashable = "",
        allow_duplicates=lib.no_default,
        names: Hashable | Sequence[Hashable] = None,
    ) -> DataFrame | Series | None:  # noqa: PR01, RT01, D200
        """
        Reset the index, or a level of it.
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        # Error checking for matching pandas. Pandas does not allow you to
        # insert a dropped index into a DataFrame if these columns already
        # exist.
        if (
            not drop
            and not (
                self._query_compiler.lazy_column_labels
                or self._query_compiler.lazy_row_labels
            )
            and not self._query_compiler.has_multiindex()
            and all(n in self.columns for n in ["level_0", "index"])
        ):
            raise ValueError("cannot insert level_0, already exists")
        new_query_compiler = self._query_compiler.reset_index(
            drop=drop,
            level=level,
            col_level=col_level,
            col_fill=col_fill,
            allow_duplicates=allow_duplicates,
            names=names,
        )
        return self._create_or_update_from_compiler(new_query_compiler, inplace)

    def radd(
        self, other, axis="columns", level=None, fill_value=None
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Return addition of `BasePandasDataset` and `other`, element-wise (binary operator `radd`).
        """
        return self._binary_op(
            "radd", other, axis=axis, level=level, fill_value=fill_value
        )

    def rfloordiv(
        self, other, axis="columns", level=None, fill_value=None
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Get integer division of `BasePandasDataset` and `other`, element-wise (binary operator `rfloordiv`).
        """
        return self._binary_op(
            "rfloordiv", other, axis=axis, level=level, fill_value=fill_value
        )

    def rmod(
        self, other, axis="columns", level=None, fill_value=None
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Get modulo of `BasePandasDataset` and `other`, element-wise (binary operator `rmod`).
        """
        return self._binary_op(
            "rmod", other, axis=axis, level=level, fill_value=fill_value
        )

    def rmul(
        self, other, axis="columns", level=None, fill_value=None
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Get Multiplication of dataframe and other, element-wise (binary operator `rmul`).
        """
        return self._binary_op(
            "rmul", other, axis=axis, level=level, fill_value=fill_value
        )

    def rolling(
        self,
        window,
        min_periods: int | None = None,
        center: bool = False,
        win_type: str | None = None,
        on: str | None = None,
        axis: Axis = lib.no_default,
        closed: str | None = None,
        step: int | None = None,
        method: str = "single",
    ) -> Rolling | Window:  # noqa: PR01, RT01, D200
        """
        Provide rolling window calculations.
        """
        if axis is not lib.no_default:
            axis = self._get_axis_number(axis)
            name = "rolling"
            if axis == 1:
                warnings.warn(
                    f"Support for axis=1 in {type(self).__name__}.{name} is "
                    + "deprecated and will be removed in a future version. "
                    + f"Use obj.T.{name}(...) instead",
                    FutureWarning,
                )
            else:
                warnings.warn(
                    f"The 'axis' keyword in {type(self).__name__}.{name} is "
                    + "deprecated and will be removed in a future version. "
                    + "Call the method without the axis keyword instead.",
                    FutureWarning,
                )
        else:
            axis = 0

        if win_type is not None:
            from .window import Window

            return Window(
                self,
                window=window,
                min_periods=min_periods,
                center=center,
                win_type=win_type,
                on=on,
                axis=axis,
                closed=closed,
                step=step,
                method=method,
            )
        from .window import Rolling

        return Rolling(
            self,
            window=window,
            min_periods=min_periods,
            center=center,
            win_type=win_type,
            on=on,
            axis=axis,
            closed=closed,
            step=step,
            method=method,
        )

    def round(self, decimals=0, *args, **kwargs) -> Self:  # noqa: PR01, RT01, D200
        """
        Round a `BasePandasDataset` to a variable number of decimal places.
        """
        # FIXME: Judging by pandas docs `*args` and `**kwargs` serves only compatibility
        # purpose and does not affect the result, we shouldn't pass them to the query compiler.
        return self.__constructor__(
            query_compiler=self._query_compiler.round(decimals=decimals, **kwargs)
        )

    def rpow(
        self, other, axis="columns", level=None, fill_value=None
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Get exponential power of `BasePandasDataset` and `other`, element-wise (binary operator `rpow`).
        """
        return self._binary_op(
            "rpow", other, axis=axis, level=level, fill_value=fill_value
        )

    def rsub(
        self, other, axis="columns", level=None, fill_value=None
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Get subtraction of `BasePandasDataset` and `other`, element-wise (binary operator `rsub`).
        """
        return self._binary_op(
            "rsub", other, axis=axis, level=level, fill_value=fill_value
        )

    def rtruediv(
        self, other, axis="columns", level=None, fill_value=None
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Get floating division of `BasePandasDataset` and `other`, element-wise (binary operator `rtruediv`).
        """
        return self._binary_op(
            "rtruediv", other, axis=axis, level=level, fill_value=fill_value
        )

    rdiv: Self = rtruediv

    def sample(
        self,
        n: int | None = None,
        frac: float | None = None,
        replace: bool = False,
        weights=None,
        random_state: RandomState | None = None,
        axis: Axis | None = None,
        ignore_index: bool = False,
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Return a random sample of items from an axis of object.
        """
        axis = self._get_axis_number(axis)
        if axis:
            axis_labels = self.columns
            axis_length = len(axis_labels)
        else:
            # Getting rows requires indices instead of labels. RangeIndex provides this.
            axis_labels = pandas.RangeIndex(len(self))
            axis_length = len(axis_labels)
        if weights is not None:
            # Index of the weights Series should correspond to the index of the
            # Dataframe in order to sample
            if isinstance(weights, BasePandasDataset):
                weights = weights.reindex(self._get_axis(axis))
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
                        + "weights when sampling from rows on "
                        + "a DataFrame"
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
            # pandas specification)
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
            # We don't need frac to be set to anything since n is already 0.
            return self._default_to_pandas(
                "sample",
                n=n,
                frac=None,
                replace=replace,
                weights=weights,
                random_state=random_state,
                axis=axis,
                ignore_index=ignore_index,
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
                    + "np.random.RandomState for random_state"
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
        self,
        axis: Axis = 0,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only=False,
        **kwargs,
    ) -> Series | float:  # noqa: PR01, RT01, D200
        """
        Return unbiased standard error of the mean over requested axis.
        """
        return self._stat_operation(
            "sem", axis, skipna, numeric_only, ddof=ddof, **kwargs
        )

    def mean(
        self,
        axis: Axis = 0,
        skipna=True,
        numeric_only=False,
        **kwargs,
    ) -> Series | float:  # noqa: PR01, RT01, D200
        """
        Return the mean of the values over the requested axis.
        """
        return self._stat_operation("mean", axis, skipna, numeric_only, **kwargs)

    def median(
        self,
        axis: Axis = 0,
        skipna=True,
        numeric_only=False,
        **kwargs,
    ) -> Series | float:  # noqa: PR01, RT01, D200
        """
        Return the mean of the values over the requested axis.
        """
        return self._stat_operation("median", axis, skipna, numeric_only, **kwargs)

    def set_axis(
        self,
        labels,
        *,
        axis: Axis = 0,
        copy=None,
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Assign desired index to given axis.
        """
        if copy is None:
            copy = True
        obj = self.copy() if copy else self
        setattr(obj, pandas.DataFrame._get_axis_name(axis), labels)
        return obj

    def set_flags(
        self, *, copy: bool = False, allows_duplicate_labels: Optional[bool] = None
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Return a new `BasePandasDataset` with updated flags.
        """
        return self._default_to_pandas(
            pandas.DataFrame.set_flags,
            copy=copy,
            allows_duplicate_labels=allows_duplicate_labels,
        )

    @property
    def flags(self):
        return self._default_to_pandas(lambda df: df.flags)

    def shift(
        self,
        periods: int = 1,
        freq=None,
        axis: Axis = 0,
        fill_value: Hashable = lib.no_default,
        suffix=None,
    ) -> Self | DataFrame:  # noqa: PR01, RT01, D200
        """
        Shift index by desired number of periods with an optional time `freq`.
        """
        if suffix:
            return self._default_to_pandas(
                lambda df: df.shift(
                    periods=periods,
                    freq=freq,
                    axis=axis,
                    fill_value=fill_value,
                    suffix=suffix,
                )
            )

        if freq is not None and fill_value is not lib.no_default:
            raise ValueError(
                "Cannot pass both 'freq' and 'fill_value' to "
                + f"{type(self).__name__}.shift"
            )

        if periods == 0:
            # Check obvious case first
            return self.copy()
        return self._create_or_update_from_compiler(
            new_query_compiler=self._query_compiler.shift(
                periods, freq, axis, fill_value
            ),
            inplace=False,
        )

    def skew(
        self,
        axis: Axis = 0,
        skipna: bool = True,
        numeric_only=False,
        **kwargs,
    ) -> Series | float:  # noqa: PR01, RT01, D200
        """
        Return unbiased skew over requested axis.
        """
        return self._stat_operation("skew", axis, skipna, numeric_only, **kwargs)

    def sort_index(
        self,
        *,
        axis=0,
        level=None,
        ascending=True,
        inplace=False,
        kind="quicksort",
        na_position="last",
        sort_remaining=True,
        ignore_index: bool = False,
        key: Optional[IndexKeyFunc] = None,
    ) -> Self | None:  # noqa: PR01, RT01, D200
        """
        Sort object by labels (along an axis).
        """
        # pandas throws this exception. See pandas issie #39434
        if ascending is None:
            raise ValueError(
                "the `axis` parameter is not supported in the pandas implementation of argsort()"
            )
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
        *,
        axis=0,
        ascending=True,
        inplace: bool = False,
        kind="quicksort",
        na_position="last",
        ignore_index: bool = False,
        key: Optional[IndexKeyFunc] = None,
    ) -> Self | None:  # noqa: PR01, RT01, D200
        """
        Sort by the values along either axis.
        """
        axis = self._get_axis_number(axis)
        inplace = validate_bool_kwarg(inplace, "inplace")
        ascending = validate_ascending(ascending)
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
        self,
        axis: Axis = 0,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only=False,
        **kwargs,
    ) -> Series | float:  # noqa: PR01, RT01, D200
        """
        Return sample standard deviation over requested axis.
        """
        return self._stat_operation(
            "std", axis, skipna, numeric_only, ddof=ddof, **kwargs
        )

    def sub(
        self, other, axis="columns", level=None, fill_value=None
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Get subtraction of `BasePandasDataset` and `other`, element-wise (binary operator `sub`).
        """
        return self._binary_op(
            "sub", other, axis=axis, level=level, fill_value=fill_value
        )

    subtract: Self = sub

    def swapaxes(self, axis1, axis2, copy=None) -> Self:  # noqa: PR01, RT01, D200
        """
        Interchange axes and swap values axes appropriately.
        """
        if copy is None:
            copy = True
        axis1 = self._get_axis_number(axis1)
        axis2 = self._get_axis_number(axis2)
        if axis1 != axis2:
            return self.transpose()
        if copy:
            return self.copy()
        return self

    def swaplevel(self, i=-2, j=-1, axis=0) -> Self:  # noqa: PR01, RT01, D200
        """
        Swap levels `i` and `j` in a `MultiIndex`.
        """
        axis = self._get_axis_number(axis)
        idx = self.index if axis == 0 else self.columns
        return self.set_axis(idx.swaplevel(i, j), axis=axis)

    def tail(self, n=5) -> Self:  # noqa: PR01, RT01, D200
        """
        Return the last `n` rows.
        """
        if n != 0:
            return self.iloc[-n:]
        return self.iloc[len(self) :]

    def take(self, indices, axis=0, **kwargs) -> Self:  # noqa: PR01, RT01, D200
        """
        Return the elements in the given *positional* indices along an axis.
        """
        axis = self._get_axis_number(axis)
        slice_obj = indices if axis == 0 else (slice(None), indices)
        return self.iloc[slice_obj]

    def to_clipboard(
        self, excel=True, sep=None, **kwargs
    ):  # pragma: no cover  # noqa: PR01, RT01, D200
        """
        Copy object to the system clipboard.
        """
        return self._default_to_pandas("to_clipboard", excel=excel, sep=sep, **kwargs)

    @expanduser_path_arg("path_or_buf")
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
        lineterminator=None,
        chunksize=None,
        date_format=None,
        doublequote=True,
        escapechar=None,
        decimal=".",
        errors: str = "strict",
        storage_options: StorageOptions = None,
    ) -> str | None:  # pragma: no cover
        from modin.core.execution.dispatching.factories.dispatcher import (
            FactoryDispatcher,
        )

        return FactoryDispatcher.to_csv(
            self._query_compiler,
            path_or_buf=path_or_buf,
            sep=sep,
            na_rep=na_rep,
            float_format=float_format,
            columns=columns,
            header=header,
            index=index,
            index_label=index_label,
            mode=mode,
            encoding=encoding,
            compression=compression,
            quoting=quoting,
            quotechar=quotechar,
            lineterminator=lineterminator,
            chunksize=chunksize,
            date_format=date_format,
            doublequote=doublequote,
            escapechar=escapechar,
            decimal=decimal,
            errors=errors,
            storage_options=storage_options,
        )

    @expanduser_path_arg("excel_writer")
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
        inf_rep="inf",
        freeze_panes=None,
        storage_options: StorageOptions = None,
        engine_kwargs=None,
    ) -> None:  # pragma: no cover  # noqa: PR01, RT01, D200
        """
        Write object to an Excel sheet.
        """
        return self._default_to_pandas(
            "to_excel",
            excel_writer,
            sheet_name=sheet_name,
            na_rep=na_rep,
            float_format=float_format,
            columns=columns,
            header=header,
            index=index,
            index_label=index_label,
            startrow=startrow,
            startcol=startcol,
            engine=engine,
            merge_cells=merge_cells,
            inf_rep=inf_rep,
            freeze_panes=freeze_panes,
            storage_options=storage_options,
            engine_kwargs=engine_kwargs,
        )

    def to_dict(self, orient="dict", into=dict, index=True) -> dict:
        return self._query_compiler.dataframe_to_dict(orient, into, index)

    @expanduser_path_arg("path_or_buf")
    def to_hdf(
        self,
        path_or_buf,
        key: str,
        mode: Literal["a", "w", "r+"] = "a",
        complevel: int | None = None,
        complib: Literal["zlib", "lzo", "bzip2", "blosc"] | None = None,
        append: bool = False,
        format: Literal["fixed", "table"] | None = None,
        index: bool = True,
        min_itemsize: int | dict[str, int] | None = None,
        nan_rep=None,
        dropna: bool | None = None,
        data_columns: Literal[True] | list[str] | None = None,
        errors: str = "strict",
        encoding: str = "UTF-8",
    ) -> None:  # pragma: no cover  # noqa: PR01, RT01, D200
        """
        Write the contained data to an HDF5 file using HDFStore.
        """
        return self._default_to_pandas(
            "to_hdf",
            path_or_buf,
            key=key,
            mode=mode,
            complevel=complevel,
            complib=complib,
            append=append,
            format=format,
            index=index,
            min_itemsize=min_itemsize,
            nan_rep=nan_rep,
            dropna=dropna,
            data_columns=data_columns,
            errors=errors,
            encoding=encoding,
        )

    @expanduser_path_arg("path_or_buf")
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
        index=None,
        indent=None,
        storage_options: StorageOptions = None,
        mode="w",
    ) -> str | None:  # pragma: no cover  # noqa: PR01, RT01, D200
        """
        Convert the object to a JSON string.
        """
        from modin.core.execution.dispatching.factories.dispatcher import (
            FactoryDispatcher,
        )

        return FactoryDispatcher.to_json(
            self._query_compiler,
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
            storage_options=storage_options,
            mode=mode,
        )

    @expanduser_path_arg("buf")
    def to_latex(
        self,
        buf=None,
        columns=None,
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
        position=None,
    ) -> str | None:  # pragma: no cover  # noqa: PR01, RT01, D200
        """
        Render object to a LaTeX tabular, longtable, or nested table.
        """
        return self._default_to_pandas(
            "to_latex",
            buf=buf,
            columns=columns,
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
            caption=caption,
            label=label,
            position=position,
        )

    @expanduser_path_arg("buf")
    def to_markdown(
        self,
        buf=None,
        mode: str = "wt",
        index: bool = True,
        storage_options: StorageOptions = None,
        **kwargs,
    ) -> str:  # noqa: PR01, RT01, D200
        """
        Print `BasePandasDataset` in Markdown-friendly format.
        """
        return self._default_to_pandas(
            "to_markdown",
            buf=buf,
            mode=mode,
            index=index,
            storage_options=storage_options,
            **kwargs,
        )

    @expanduser_path_arg("path")
    def to_pickle(
        self,
        path,
        compression: CompressionOptions = "infer",
        protocol: int = pkl.HIGHEST_PROTOCOL,
        storage_options: StorageOptions = None,
    ) -> None:  # pragma: no cover  # noqa: PR01, D200
        """
        Pickle (serialize) object to file.
        """
        from modin.pandas import to_pickle

        to_pickle(
            self,
            path,
            compression=compression,
            protocol=protocol,
            storage_options=storage_options,
        )

    def _to_bare_numpy(
        self, dtype=None, copy=False, na_value=lib.no_default
    ):  # noqa: PR01, RT01, D200
        """
        Convert the `BasePandasDataset` to a NumPy array.
        """
        return self._query_compiler.to_numpy(
            dtype=dtype,
            copy=copy,
            na_value=na_value,
        )

    def to_numpy(
        self, dtype=None, copy=False, na_value=lib.no_default
    ) -> np.ndarray:  # noqa: PR01, RT01, D200
        """
        Convert the `BasePandasDataset` to a NumPy array or a Modin wrapper for NumPy array.
        """
        from modin.config import ModinNumpy

        if ModinNumpy.get():
            from ..numpy.arr import array

            return array(self, copy=copy)

        return self._to_bare_numpy(
            dtype=dtype,
            copy=copy,
            na_value=na_value,
        )

    # TODO(williamma12): When this gets implemented, have the series one call this.
    def to_period(
        self, freq=None, axis=0, copy=None
    ) -> Self:  # pragma: no cover  # noqa: PR01, RT01, D200
        """
        Convert `BasePandasDataset` from DatetimeIndex to PeriodIndex.
        """
        return self._default_to_pandas("to_period", freq=freq, axis=axis, copy=copy)

    @expanduser_path_arg("buf")
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
    ) -> str | None:  # noqa: PR01, RT01, D200
        """
        Render a `BasePandasDataset` to a console-friendly tabular output.
        """
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
    ) -> int | None:  # noqa: PR01, D200
        """
        Write records stored in a `BasePandasDataset` to a SQL database.
        """
        new_query_compiler = self._query_compiler
        # writing the index to the database by inserting it to the DF
        if index:
            new_query_compiler = new_query_compiler.reset_index()
            if index_label is not None:
                if not is_list_like(index_label):
                    index_label = [index_label]
                new_query_compiler.columns = list(index_label) + list(
                    new_query_compiler.columns[len(index_label) :]
                )
            # so pandas._to_sql will not write the index to the database as well
            index = False

        from modin.core.execution.dispatching.factories.dispatcher import (
            FactoryDispatcher,
        )

        FactoryDispatcher.to_sql(
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
    def to_timestamp(
        self, freq=None, how="start", axis=0, copy=None
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Cast to DatetimeIndex of timestamps, at *beginning* of period.
        """
        return self._default_to_pandas(
            "to_timestamp", freq=freq, how=how, axis=axis, copy=copy
        )

    def to_xarray(self):  # noqa: PR01, RT01, D200
        """
        Return an xarray object from the `BasePandasDataset`.
        """
        return self._default_to_pandas("to_xarray")

    def truediv(
        self, other, axis="columns", level=None, fill_value=None
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Get floating division of `BasePandasDataset` and `other`, element-wise (binary operator `truediv`).
        """
        return self._binary_op(
            "truediv", other, axis=axis, level=level, fill_value=fill_value
        )

    div: Self = truediv
    divide: Self = truediv

    def truncate(
        self, before=None, after=None, axis=None, copy=None
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Truncate a `BasePandasDataset` before and after some index value.
        """
        axis = self._get_axis_number(axis)
        if (
            not self._get_axis(axis).is_monotonic_increasing
            and not self._get_axis(axis).is_monotonic_decreasing
        ):
            raise ValueError("truncate requires a sorted index")

        if before is not None and after is not None and before > after:
            raise ValueError(f"Truncate: {after} must be after {before}")

        s = slice(*self._get_axis(axis).slice_locs(before, after))
        slice_obj = s if axis == 0 else (slice(None), s)
        return self.iloc[slice_obj]

    def transform(
        self, func, axis=0, *args, **kwargs
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Call ``func`` on self producing a `BasePandasDataset` with the same axis shape as self.
        """
        kwargs["is_transform"] = True
        self._validate_function(func)
        try:
            result = self.agg(func, axis=axis, *args, **kwargs)
        except (TypeError, pandas.errors.SpecificationError):
            raise
        except Exception as err:
            raise ValueError("Transform function failed") from err
        if getattr(result, "_pandas_class", None) not in (
            pandas.Series,
            pandas.DataFrame,
        ) or not result.index.equals(self.index):
            raise ValueError("Function did not transform")
        return result

    def tz_convert(
        self, tz, axis=0, level=None, copy=None
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Convert tz-aware axis to target time zone.
        """
        if copy is None:
            copy = True
        return self._create_or_update_from_compiler(
            self._query_compiler.tz_convert(
                tz, axis=self._get_axis_number(axis), level=level, copy=copy
            ),
            inplace=(not copy),
        )

    def tz_localize(
        self, tz, axis=0, level=None, copy=None, ambiguous="raise", nonexistent="raise"
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Localize tz-naive index of a `BasePandasDataset` to target time zone.
        """
        if copy is None:
            copy = True
        return self._create_or_update_from_compiler(
            self._query_compiler.tz_localize(
                tz,
                axis=self._get_axis_number(axis),
                level=level,
                copy=copy,
                ambiguous=ambiguous,
                nonexistent=nonexistent,
            ),
            inplace=(not copy),
        )

    def interpolate(
        self,
        method="linear",
        *,
        axis=0,
        limit=None,
        inplace=False,
        limit_direction: Optional[str] = None,
        limit_area=None,
        downcast=lib.no_default,
        **kwargs,
    ) -> Self:  # noqa: PR01, RT01, D200
        if downcast is not lib.no_default:
            warnings.warn(
                f"The 'downcast' keyword in {type(self).__name__}.interpolate "
                + "is deprecated and will be removed in a future version. "
                + "Call result.infer_objects(copy=False) on the result instead.",
                FutureWarning,
            )
        else:
            downcast = None

        return self._create_or_update_from_compiler(
            self._query_compiler.interpolate(
                method=method,
                axis=axis,
                limit=limit,
                inplace=False,
                limit_direction=limit_direction,
                limit_area=limit_area,
                downcast=downcast,
                **kwargs,
            ),
            inplace=inplace,
        )

    # TODO: uncomment the following lines when #3331 issue will be closed
    # @prepend_to_notes(
    #     """
    #     In comparison with pandas, Modin's ``value_counts`` returns Series with ``MultiIndex``
    #     only if multiple columns were passed via the `subset` parameter, otherwise, the resulted
    #     Series's index will be a regular single dimensional ``Index``.
    #     """
    # )
    @_inherit_docstrings(
        pandas.DataFrame.value_counts, apilink="pandas.DataFrame.value_counts"
    )
    def value_counts(
        self,
        subset: Sequence[Hashable] | None = None,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        dropna: bool = True,
    ) -> Series:
        if subset is None:
            subset = self._query_compiler.columns
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*groupby keys will be sorted anyway.*",
                category=UserWarning,
            )
            counted_values = self.groupby(
                by=subset, dropna=dropna, observed=True, sort=False
            ).size()
        if sort:
            counted_values.sort_values(ascending=ascending, inplace=True)
        if normalize:
            counted_values = counted_values / counted_values.sum()
        # TODO: uncomment when strict compability mode will be implemented:
        # https://github.com/modin-project/modin/issues/3411
        # if STRICT_COMPABILITY and not isinstance(counted_values.index, MultiIndex):
        #     counted_values.index = pandas.MultiIndex.from_arrays(
        #         [counted_values.index], names=counted_values.index.names
        #     )
        # https://pandas.pydata.org/pandas-docs/version/2.0/whatsnew/v2.0.0.html#value-counts-sets-the-resulting-name-to-count
        counted_values.name = "proportion" if normalize else "count"
        return counted_values

    def var(
        self,
        axis: Axis = 0,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only=False,
        **kwargs,
    ) -> Series | float:  # noqa: PR01, RT01, D200
        """
        Return unbiased variance over requested axis.
        """
        return self._stat_operation(
            "var", axis, skipna, numeric_only, ddof=ddof, **kwargs
        )

    def __abs__(self) -> Self:
        """
        Return a `BasePandasDataset` with absolute numeric value of each element.

        Returns
        -------
        BasePandasDataset
            Object containing the absolute value of each element.
        """
        return self.abs()

    @_doc_binary_op(
        operation="union", bin_op="and", right="other", **_doc_binary_op_kwargs
    )
    def __and__(self, other) -> Self:
        return self._binary_op("__and__", other, axis=0)

    @_doc_binary_op(
        operation="union", bin_op="rand", right="other", **_doc_binary_op_kwargs
    )
    def __rand__(self, other) -> Self:
        return self._binary_op("__rand__", other, axis=0)

    def __array__(self, dtype=None) -> np.ndarray:
        """
        Return the values as a NumPy array.

        Parameters
        ----------
        dtype : str or np.dtype, optional
            The dtype of returned array.

        Returns
        -------
        arr : np.ndarray
            NumPy representation of Modin object.
        """
        return self._to_bare_numpy(dtype)

    def __copy__(self, deep=True) -> Self:
        """
        Return the copy of the `BasePandasDataset`.

        Parameters
        ----------
        deep : bool, default: True
            Whether the copy should be deep or not.

        Returns
        -------
        BasePandasDataset
        """
        return self.copy(deep=deep)

    def __deepcopy__(self, memo=None) -> Self:
        """
        Return the deep copy of the `BasePandasDataset`.

        Parameters
        ----------
        memo : Any, optional
           Deprecated parameter.

        Returns
        -------
        BasePandasDataset
        """
        return self.copy(deep=True)

    @_doc_binary_op(
        operation="equality comparison",
        bin_op="eq",
        right="other",
        **_doc_binary_op_kwargs,
    )
    def __eq__(self, other) -> Self:
        return self.eq(other)

    def __finalize__(self, other, method=None, **kwargs) -> Self:
        """
        Propagate metadata from `other` to `self`.

        Parameters
        ----------
        other : BasePandasDataset
            The object from which to get the attributes that we are going
            to propagate.
        method : str, optional
            A passed method name providing context on where `__finalize__`
            was called.
        **kwargs : dict
            Additional keywords arguments to be passed to `__finalize__`.

        Returns
        -------
        BasePandasDataset
        """
        return self._default_to_pandas("__finalize__", other, method=method, **kwargs)

    @_doc_binary_op(
        operation="greater than or equal comparison",
        bin_op="ge",
        right="right",
        **_doc_binary_op_kwargs,
    )
    def __ge__(self, right) -> Self:
        return self.ge(right)

    def __getitem__(self, key) -> Self:
        """
        Retrieve dataset according to `key`.

        Parameters
        ----------
        key : callable, scalar, slice, str or tuple
            The global row index to retrieve data from.

        Returns
        -------
        BasePandasDataset
            Located dataset.
        """
        if not self._query_compiler.lazy_row_count and len(self) == 0:
            return self._default_to_pandas("__getitem__", key)
        # see if we can slice the rows
        # This lets us reuse code in pandas to error check
        indexer = None
        if isinstance(key, slice):
            indexer = self.index._convert_slice_indexer(key, kind="getitem")
        if indexer is not None:
            return self._getitem_slice(indexer)
        else:
            return self._getitem(key)

    def xs(
        self,
        key,
        axis=0,
        level=None,
        drop_level: bool = True,
    ) -> Self:  # noqa: PR01, RT01, D200
        """
        Return cross-section from the Series/DataFrame.
        """
        axis = self._get_axis_number(axis)
        labels = self.columns if axis else self.index

        if isinstance(key, list):
            # deprecated in pandas, to be removed in 2.0
            warnings.warn(
                "Passing lists as key for xs is deprecated and will be removed in a "
                + "future version. Pass key as a tuple instead.",
                FutureWarning,
            )

        if level is not None:
            if not isinstance(labels, pandas.MultiIndex):
                raise TypeError("Index must be a MultiIndex")
            loc, new_ax = labels.get_loc_level(key, level=level, drop_level=drop_level)

            # create the tuple of the indexer
            _indexer = [slice(None)] * self.ndim
            _indexer[axis] = loc
            indexer = tuple(_indexer)

            result = self.iloc[indexer]
            setattr(result, self._pandas_class._get_axis_name(axis), new_ax)
            return result

        if axis == 1:
            if drop_level:
                return self[key]
            index = self.columns
        else:
            index = self.index

        new_index = None
        if isinstance(index, pandas.MultiIndex):
            loc, new_index = index._get_loc_level(key, level=0)
            if not drop_level:
                if is_integer(loc):
                    new_index = index[loc : loc + 1]
                else:
                    new_index = index[loc]
        else:
            loc = index.get_loc(key)

            if isinstance(loc, np.ndarray):
                if loc.dtype == np.bool_:
                    (loc,) = loc.nonzero()
                # Note: pandas uses self._take_with_is_copy here
                return self.take(loc, axis=axis)

            if not is_scalar(loc):
                new_index = index[loc]

        if is_scalar(loc) and axis == 0:
            # In this case loc should be an integer
            if self.ndim == 1:
                # if we encounter an array-like and we only have 1 dim
                # that means that their are list/ndarrays inside the Series!
                # so just return them (pandas GH 6394)
                return self.iloc[loc]

            result = self.iloc[loc]
        elif is_scalar(loc):
            result = self.iloc[:, slice(loc, loc + 1)]
        elif axis == 1:
            result = self.iloc[:, loc]
        else:
            result = self.iloc[loc]
            if new_index is None:
                raise RuntimeError(
                    "`new_index` variable shouldn't be equal to None here, something went wrong."
                )
            result.index = new_index

        # Note: pandas does result._set_is_copy here
        return result

    __hash__ = None

    def _setitem_slice(self, key: slice, value) -> None:
        """
        Set rows specified by `key` slice with `value`.

        Parameters
        ----------
        key : location or index-based slice
            Key that points rows to modify.
        value : object
            Value to assing to the rows.
        """
        indexer = self.index._convert_slice_indexer(key, kind="getitem")
        self.iloc[indexer] = value

    def _getitem_slice(self, key: slice) -> Self:
        """
        Get rows specified by `key` slice.

        Parameters
        ----------
        key : location or index-based slice
            Key that points to rows to retrieve.

        Returns
        -------
        modin.pandas.BasePandasDataset
            Selected rows.
        """
        if is_full_grab_slice(
            key,
            # Avoid triggering shape computation for lazy executions
            sequence_len=(None if self._query_compiler.lazy_row_count else len(self)),
        ):
            return self.copy()
        return self.iloc[key]

    @_doc_binary_op(
        operation="greater than comparison",
        bin_op="gt",
        right="right",
        **_doc_binary_op_kwargs,
    )
    def __gt__(self, right) -> Self:
        return self.gt(right)

    def __invert__(self) -> Self:
        """
        Apply bitwise inverse to each element of the `BasePandasDataset`.

        Returns
        -------
        BasePandasDataset
            New BasePandasDataset containing bitwise inverse to each value.
        """
        if not all(is_bool_dtype(d) or is_integer_dtype(d) for d in self._get_dtypes()):
            raise TypeError(
                "bad operand type for unary ~: '{}'".format(
                    next(
                        d
                        for d in self._get_dtypes()
                        if not (is_bool_dtype(d) or is_integer_dtype(d))
                    )
                )
            )
        return self.__constructor__(query_compiler=self._query_compiler.invert())

    @_doc_binary_op(
        operation="less than or equal comparison",
        bin_op="le",
        right="right",
        **_doc_binary_op_kwargs,
    )
    def __le__(self, right) -> Self:
        return self.le(right)

    def __len__(self) -> int:
        """
        Return length of info axis.

        Returns
        -------
        int
        """
        return self._query_compiler.get_axis_len(0)

    @_doc_binary_op(
        operation="less than comparison",
        bin_op="lt",
        right="right",
        **_doc_binary_op_kwargs,
    )
    def __lt__(self, right) -> Self:
        return self.lt(right)

    def __matmul__(self, other) -> Self | np.ndarray | Scalar:
        """
        Compute the matrix multiplication between the `BasePandasDataset` and `other`.

        Parameters
        ----------
        other : BasePandasDataset or array-like
            The other object to compute the matrix product with.

        Returns
        -------
        BasePandasDataset, np.ndarray or scalar
        """
        return self.dot(other)

    @_doc_binary_op(
        operation="not equal comparison",
        bin_op="ne",
        right="other",
        **_doc_binary_op_kwargs,
    )
    def __ne__(self, other) -> Self:
        return self.ne(other)

    def __neg__(self) -> Self:
        """
        Change the sign for every value of self.

        Returns
        -------
        BasePandasDataset
        """
        self._validate_dtypes(numeric_only=True)
        return self.__constructor__(query_compiler=self._query_compiler.negative())

    def __nonzero__(self):
        """
        Evaluate `BasePandasDataset` as boolean object.

        Raises
        ------
        ValueError
            Always since truth value for self is ambiguous.
        """
        raise ValueError(
            f"The truth value of a {self.__class__.__name__} is ambiguous. "
            + "Use a.empty, a.bool(), a.item(), a.any() or a.all()."
        )

    __bool__ = __nonzero__

    @_doc_binary_op(
        operation="disjunction",
        bin_op="or",
        right="other",
        **_doc_binary_op_kwargs,
    )
    def __or__(self, other) -> Self:
        return self._binary_op("__or__", other, axis=0)

    @_doc_binary_op(
        operation="disjunction",
        bin_op="ror",
        right="other",
        **_doc_binary_op_kwargs,
    )
    def __ror__(self, other) -> Self:
        return self._binary_op("__ror__", other, axis=0)

    def __sizeof__(self) -> int:
        """
        Generate the total memory usage for an `BasePandasDataset`.

        Returns
        -------
        int
        """
        return self._query_compiler.sizeof()

    def __str__(self) -> str:  # pragma: no cover
        """
        Return str(self).

        Returns
        -------
        str
        """
        return repr(self)

    @_doc_binary_op(
        operation="exclusive disjunction",
        bin_op="xor",
        right="other",
        **_doc_binary_op_kwargs,
    )
    def __xor__(self, other) -> Self:
        return self._binary_op("__xor__", other, axis=0)

    @_doc_binary_op(
        operation="exclusive disjunction",
        bin_op="rxor",
        right="other",
        **_doc_binary_op_kwargs,
    )
    def __rxor__(self, other) -> Self:
        return self._binary_op("__rxor__", other, axis=0)

    @property
    def size(self) -> int:  # noqa: RT01, D200
        """
        Return an int representing the number of elements in this `BasePandasDataset` object.
        """
        return len(self._query_compiler.index) * len(self._query_compiler.columns)

    @property
    def values(self) -> np.ndarray:  # noqa: RT01, D200
        """
        Return a NumPy representation of the `BasePandasDataset`.
        """
        return self.to_numpy()

    def _repartition(self, axis: Optional[int] = None) -> Self:
        """
        Repartitioning Modin objects to get ideal partitions inside.

        Allows to improve performance where the query compiler can't improve
        yet by doing implicit repartitioning.

        Parameters
        ----------
        axis : {0, 1, None}, optional
            The axis along which the repartitioning occurs.
            `None` is used for repartitioning along both axes.

        Returns
        -------
        DataFrame or Series
            The repartitioned dataframe or series, depending on the original type.
        """
        allowed_axis_values = (0, 1, None)
        if axis not in allowed_axis_values:
            raise ValueError(
                f"Passed `axis` parameter: {axis}, but should be one of {allowed_axis_values}"
            )
        return self.__constructor__(
            query_compiler=self._query_compiler.repartition(axis=axis)
        )

    @disable_logging
    def __getattribute__(self, item) -> Any:
        """
        Return item from the `BasePandasDataset`.

        Parameters
        ----------
        item : hashable
            Item to get.

        Returns
        -------
        Any
        """
        attr = super().__getattribute__(item)
        if item not in _DEFAULT_BEHAVIOUR and not self._query_compiler.lazy_shape:
            # We default to pandas on empty DataFrames. This avoids a large amount of
            # pain in underlying implementation and returns a result immediately rather
            # than dealing with the edge cases that empty DataFrames have.
            if callable(attr) and self.empty and hasattr(self._pandas_class, item):

                def default_handler(*args, **kwargs):
                    return self._default_to_pandas(item, *args, **kwargs)

                return default_handler
        return attr

    def __array_ufunc__(
        self, ufunc: np.ufunc, method: str, *inputs: Any, **kwargs: Any
    ) -> DataFrame | Series | Any:
        """
        Apply the `ufunc` to the `BasePandasDataset`.

        Parameters
        ----------
        ufunc : np.ufunc
            The NumPy ufunc to apply.
        method : str
            The method to apply.
        *inputs : tuple
            The inputs to the ufunc.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        BasePandasDataset
            The result of the ufunc applied to the `BasePandasDataset`.
        """
        # we can't use the regular default_to_pandas() method because self is one of the
        # `inputs` to __array_ufunc__, and pandas has some checks on the identity of the
        # inputs [1]. The usual default to pandas will call _to_pandas() on the inputs
        # as well as on self, but that gives inputs[0] a different identity from self.
        #
        # [1] https://github.com/pandas-dev/pandas/blob/2c4c072ade78b96a9eb05097a5fcf4347a3768f3/pandas/_libs/ops_dispatch.pyx#L99-L109
        ErrorMessage.default_to_pandas(message="__array_ufunc__")
        pandas_self = self._to_pandas()
        pandas_result = pandas_self.__array_ufunc__(
            ufunc,
            method,
            *(
                pandas_self if each_input is self else try_cast_to_pandas(each_input)
                for each_input in inputs
            ),
            **try_cast_to_pandas(kwargs),
        )
        if isinstance(pandas_result, pandas.DataFrame):
            from .dataframe import DataFrame

            return DataFrame(pandas_result)
        elif isinstance(pandas_result, pandas.Series):
            from .series import Series

            return Series(pandas_result)
        return pandas_result

    # namespace for additional Modin functions that are not available in Pandas
    modin: ModinAPI = CachedAccessor("modin", ModinAPI)
