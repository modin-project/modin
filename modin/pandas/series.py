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

"""Module houses `Series` class, that is distributed version of `pandas.Series`."""

from __future__ import annotations

import os
import warnings
from typing import IO, TYPE_CHECKING, Any, Hashable, Iterable, Optional, Union

import numpy as np
import pandas
from pandas._libs import lib
from pandas._typing import ArrayLike, Axis, DtypeObj, IndexKeyFunc, Scalar, Sequence
from pandas.api.types import is_integer
from pandas.core.arrays import ExtensionArray
from pandas.core.common import apply_if_callable, is_bool_indexer
from pandas.core.dtypes.common import is_dict_like, is_list_like
from pandas.core.series import _coerce_method
from pandas.io.formats.info import SeriesInfo
from pandas.util._validators import validate_bool_kwarg

from modin.config import PersistentPickle
from modin.logging import disable_logging
from modin.pandas.io import from_pandas, to_pandas
from modin.utils import (
    MODIN_UNNAMED_SERIES_LABEL,
    _inherit_docstrings,
    import_optional_dependency,
)

from .accessor import CachedAccessor, SparseAccessor
from .base import _ATTRS_NO_LOOKUP, BasePandasDataset
from .iterator import PartitionIterator
from .series_utils import (
    CategoryMethods,
    DatetimeProperties,
    ListAccessor,
    StringMethods,
    StructAccessor,
)
from .utils import _doc_binary_op, cast_function_modin2pandas, is_scalar

if TYPE_CHECKING:
    import numpy.typing as npt

    from modin.core.storage_formats import BaseQueryCompiler

    from .dataframe import DataFrame

# Dictionary of extensions assigned to this class
_SERIES_EXTENSIONS_ = {}


@_inherit_docstrings(
    pandas.Series, excluded=[pandas.Series.__init__], apilink="pandas.Series"
)
class Series(BasePandasDataset):
    """
    Modin distributed representation of `pandas.Series`.

    Internally, the data can be divided into partitions in order to parallelize
    computations and utilize the user's hardware as much as possible.

    Inherit common for DataFrames and Series functionality from the
    `BasePandasDataset` class.

    Parameters
    ----------
    data : modin.pandas.Series, array-like, Iterable, dict, or scalar value, optional
        Contains data stored in Series. If data is a dict, argument order is
        maintained.
    index : array-like or Index (1d), optional
        Values must be hashable and have the same length as `data`.
    dtype : str, np.dtype, or pandas.ExtensionDtype, optional
        Data type for the output Series. If not specified, this will be
        inferred from `data`.
    name : str, optional
        The name to give to the Series.
    copy : bool, default: False
        Copy input data.
    fastpath : bool, default: False
        `pandas` internal parameter.
    query_compiler : BaseQueryCompiler, optional
        A query compiler object to create the Series from.
    """

    _pandas_class = pandas.Series
    __array_priority__ = pandas.Series.__array_priority__

    def __init__(
        self,
        data=None,
        index=None,
        dtype=None,
        name=None,
        copy=None,
        fastpath=lib.no_default,
        query_compiler: BaseQueryCompiler = None,
    ) -> None:
        from modin.numpy import array

        # Siblings are other dataframes that share the same query compiler. We
        # use this list to update inplace when there is a shallow copy.
        self._siblings = []
        if isinstance(data, type(self)):
            query_compiler = data._query_compiler.copy()
            if index is not None:
                if any(i not in data.index for i in index):
                    raise NotImplementedError(
                        "Passing non-existent columns or index values to constructor "
                        + "not yet implemented."
                    )
                query_compiler = data.loc[index]._query_compiler
        if isinstance(data, array):
            if data._ndim == 2:
                raise ValueError("Data must be 1-dimensional")
            query_compiler = data._query_compiler.copy()
            if index is not None:
                query_compiler.index = index
            if dtype is not None:
                query_compiler = query_compiler.astype(
                    {col_name: dtype for col_name in query_compiler.columns}
                )
            if name is None:
                query_compiler.columns = pandas.Index([MODIN_UNNAMED_SERIES_LABEL])
        if query_compiler is None:
            # Defaulting to pandas
            if name is None:
                name = MODIN_UNNAMED_SERIES_LABEL
                if isinstance(data, pandas.Series) and data.name is not None:
                    name = data.name

            pandas_df = pandas.DataFrame(
                pandas.Series(
                    data=data,
                    index=index,
                    dtype=dtype,
                    name=name,
                    copy=copy,
                    fastpath=fastpath,
                )
            )
            if pandas_df.size >= 2_500_000:
                warnings.warn(
                    "Distributing {} object. This may take some time.".format(
                        type(data)
                    )
                )
            query_compiler = from_pandas(pandas_df)._query_compiler
        self._query_compiler = query_compiler.columnarize()
        if name is not None:
            self.name = name

    def _get_name(self) -> Hashable:
        """
        Get the value of the `name` property.

        Returns
        -------
        hashable
        """
        name = self._query_compiler.columns[0]
        if name == MODIN_UNNAMED_SERIES_LABEL:
            return None
        return name

    def _set_name(self, name: Hashable) -> None:
        """
        Set the value of the `name` property.

        Parameters
        ----------
        name : hashable
            Name value to set.
        """
        if name is None:
            name = MODIN_UNNAMED_SERIES_LABEL
        if isinstance(name, tuple):
            columns = pandas.MultiIndex.from_tuples(tuples=[name])
        else:
            columns = [name]
        self._query_compiler.columns = columns

    name: Hashable = property(_get_name, _set_name)
    _parent = None
    # Parent axis denotes axis that was used to select series in a parent dataframe.
    # If _parent_axis == 0, then it means that index axis was used via df.loc[row]
    # indexing operations and assignments should be done to rows of parent.
    # If _parent_axis == 1 it means that column axis was used via df[column] and assignments
    # should be done to columns of parent.
    _parent_axis = 0

    @_doc_binary_op(operation="addition", bin_op="add")
    def __add__(self, right) -> Series:
        return self.add(right)

    @_doc_binary_op(operation="addition", bin_op="radd", right="left")
    def __radd__(self, left) -> Series:
        return self.radd(left)

    @_doc_binary_op(operation="union", bin_op="and", right="other")
    def __and__(self, other) -> Series:
        if isinstance(other, (list, np.ndarray, pandas.Series)):
            return self._default_to_pandas(pandas.Series.__and__, other)
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).__and__(new_other)

    @_doc_binary_op(operation="union", bin_op="and", right="other")
    def __rand__(self, other) -> Series:
        if isinstance(other, (list, np.ndarray, pandas.Series)):
            return self._default_to_pandas(pandas.Series.__rand__, other)
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).__rand__(new_other)

    # add `_inherit_docstrings` decorator to force method link addition.
    @_inherit_docstrings(pandas.Series.__array__, apilink="pandas.Series.__array__")
    def __array__(self, dtype=None) -> np.ndarray:  # noqa: PR01, RT01, D200
        """
        Return the values as a NumPy array.
        """
        return super(Series, self).__array__(dtype).flatten()

    def __column_consortium_standard__(
        self, *, api_version: str | None = None
    ):  # noqa: PR01, RT01
        """
        Provide entry point to the Consortium DataFrame Standard API.

        This is developed and maintained outside of Modin.
        Please report any issues to https://github.com/data-apis/dataframe-api-compat.
        """
        dataframe_api_compat = import_optional_dependency(
            "dataframe_api_compat", "implementation"
        )
        return dataframe_api_compat.modin_standard.convert_to_standard_compliant_column(
            self, api_version=api_version
        )

    def __contains__(self, key: Hashable) -> bool:
        """
        Check if `key` in the `Series.index`.

        Parameters
        ----------
        key : hashable
            Key to check the presence in the index.

        Returns
        -------
        bool
        """
        return key in self.index

    def __copy__(self, deep: bool = True) -> Series:
        """
        Return the copy of the Series.

        Parameters
        ----------
        deep : bool, default: True
            Whether the copy should be deep or not.

        Returns
        -------
        Series
        """
        return self.copy(deep=deep)

    def __deepcopy__(self, memo=None) -> Series:
        """
        Return the deep copy of the Series.

        Parameters
        ----------
        memo : Any, optional
           Deprecated parameter.

        Returns
        -------
        Series
        """
        return self.copy(deep=True)

    def __delitem__(self, key: Hashable) -> None:
        """
        Delete item identified by `key` label.

        Parameters
        ----------
        key : hashable
            Key to delete.
        """
        if key not in self.keys():
            raise KeyError(key)
        self.drop(labels=key, inplace=True)

    @_doc_binary_op(
        operation="integer division and modulo",
        bin_op="divmod",
        returns="tuple of two Series",
    )
    def __divmod__(self, right) -> tuple[Series, Series]:
        return self.divmod(right)

    @_doc_binary_op(
        operation="integer division and modulo",
        bin_op="divmod",
        right="left",
        returns="tuple of two Series",
    )
    def __rdivmod__(self, left) -> tuple[Series, Series]:
        return self.rdivmod(left)

    @_doc_binary_op(operation="integer division", bin_op="floordiv")
    def __floordiv__(self, right) -> Series:
        return self.floordiv(right)

    @_doc_binary_op(operation="integer division", bin_op="floordiv")
    def __rfloordiv__(self, right) -> Series:
        return self.rfloordiv(right)

    @disable_logging
    def __getattr__(self, key: Hashable) -> Any:
        """
        Return item identified by `key`.

        Parameters
        ----------
        key : hashable
            Key to get.

        Returns
        -------
        Any

        Notes
        -----
        First try to use `__getattribute__` method. If it fails
        try to get `key` from `Series` fields.
        """
        try:
            return _SERIES_EXTENSIONS_.get(key, object.__getattribute__(self, key))
        except AttributeError as err:
            if key not in _ATTRS_NO_LOOKUP and key in self.index:
                return self[key]
            raise err

    __float__ = _coerce_method(float)
    __int__ = _coerce_method(int)

    def __iter__(self):
        """
        Return an iterator of the values.

        Returns
        -------
        iterable
        """
        return self._to_pandas().__iter__()

    @_doc_binary_op(operation="modulo", bin_op="mod")
    def __mod__(self, right) -> Series:
        return self.mod(right)

    @_doc_binary_op(operation="modulo", bin_op="mod", right="left")
    def __rmod__(self, left) -> Series:
        return self.rmod(left)

    @_doc_binary_op(operation="multiplication", bin_op="mul")
    def __mul__(self, right) -> Series:
        return self.mul(right)

    @_doc_binary_op(operation="multiplication", bin_op="mul", right="left")
    def __rmul__(self, left) -> Series:
        return self.rmul(left)

    @_doc_binary_op(operation="disjunction", bin_op="or", right="other")
    def __or__(self, other) -> Series:
        if isinstance(other, (list, np.ndarray, pandas.Series)):
            return self._default_to_pandas(pandas.Series.__or__, other)
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).__or__(new_other)

    @_doc_binary_op(operation="disjunction", bin_op="or", right="other")
    def __ror__(self, other) -> Series:
        if isinstance(other, (list, np.ndarray, pandas.Series)):
            return self._default_to_pandas(pandas.Series.__ror__, other)
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).__ror__(new_other)

    @_doc_binary_op(operation="exclusive or", bin_op="xor", right="other")
    def __xor__(self, other) -> Series:
        if isinstance(other, (list, np.ndarray, pandas.Series)):
            return self._default_to_pandas(pandas.Series.__xor__, other)
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).__xor__(new_other)

    @_doc_binary_op(operation="exclusive or", bin_op="xor", right="other")
    def __rxor__(self, other) -> Series:
        if isinstance(other, (list, np.ndarray, pandas.Series)):
            return self._default_to_pandas(pandas.Series.__rxor__, other)
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).__rxor__(new_other)

    @_doc_binary_op(operation="exponential power", bin_op="pow")
    def __pow__(self, right) -> Series:
        return self.pow(right)

    @_doc_binary_op(operation="exponential power", bin_op="pow", right="left")
    def __rpow__(self, left) -> Series:
        return self.rpow(left)

    def __repr__(self) -> str:
        """
        Return a string representation for a particular Series.

        Returns
        -------
        str
        """
        num_rows = pandas.get_option("display.max_rows") or 60
        num_cols = pandas.get_option("display.max_columns") or 20
        temp_df = self._build_repr_df(num_rows, num_cols)
        if isinstance(temp_df, pandas.DataFrame) and not temp_df.empty:
            temp_df = temp_df.iloc[:, 0]
        temp_str = repr(temp_df)
        freq_str = (
            "Freq: {}, ".format(self.index.freqstr)
            if isinstance(self.index, pandas.DatetimeIndex)
            else ""
        )
        if self.name is not None:
            name_str = "Name: {}, ".format(str(self.name))
        else:
            name_str = ""
        if len(self.index) > num_rows:
            len_str = "Length: {}, ".format(len(self.index))
        else:
            len_str = ""
        dtype_str = "dtype: {}".format(
            str(self.dtype) + ")"
            if temp_df.empty
            else temp_str.rsplit("dtype: ", 1)[-1]
        )
        if len(self) == 0:
            return "Series([], {}{}{}".format(freq_str, name_str, dtype_str)
        maxsplit = 1
        if (
            isinstance(temp_df, pandas.Series)
            and temp_df.name is not None
            and isinstance(temp_df.dtype, pandas.CategoricalDtype)
        ):
            maxsplit = 2
        return temp_str.rsplit("\n", maxsplit)[0] + "\n{}{}{}{}".format(
            freq_str, name_str, len_str, dtype_str
        )

    def __round__(self, decimals=0) -> Series:
        """
        Round each value in a Series to the given number of decimals.

        Parameters
        ----------
        decimals : int, default: 0
            Number of decimal places to round to.

        Returns
        -------
        Series
        """
        return self._create_or_update_from_compiler(
            self._query_compiler.round(decimals=decimals)
        )

    def __setitem__(self, key: Hashable, value: Any) -> None:
        """
        Set `value` identified by `key` in the Series.

        Parameters
        ----------
        key : hashable
            Key to set.
        value : Any
            Value to set.
        """
        if isinstance(key, slice):
            self._setitem_slice(key, value)
        else:
            self.loc[key] = value

    @_doc_binary_op(operation="subtraction", bin_op="sub")
    def __sub__(self, right) -> Series:
        return self.sub(right)

    @_doc_binary_op(operation="subtraction", bin_op="sub", right="left")
    def __rsub__(self, left) -> Series:
        return self.rsub(left)

    @_doc_binary_op(operation="floating division", bin_op="truediv")
    def __truediv__(self, right) -> Series:
        return self.truediv(right)

    @_doc_binary_op(operation="floating division", bin_op="truediv", right="left")
    def __rtruediv__(self, left) -> Series:
        return self.rtruediv(left)

    __iadd__ = __add__
    __imul__ = __mul__
    __ipow__ = __pow__
    __isub__ = __sub__
    __itruediv__ = __truediv__

    @property
    def values(self):  # noqa: RT01, D200
        """
        Return Series as ndarray or ndarray-like depending on the dtype.
        """
        import modin.pandas as pd

        if isinstance(
            self.dtype, pandas.core.dtypes.dtypes.ExtensionDtype
        ) and not isinstance(self.dtype, pd.CategoricalDtype):
            return self._default_to_pandas("values")

        data = self.to_numpy()
        if isinstance(self.dtype, pd.CategoricalDtype):
            from modin.config import ModinNumpy

            if ModinNumpy.get():
                data = data._to_numpy()
            data = pd.Categorical(data, dtype=self.dtype)
        return data

    def __arrow_array__(self, type=None):  # noqa: GL08
        # Although pandas.Series does not implement this method (true for version 2.2.*),
        # however, pyarrow has support for it. This method emulates this behavior and
        # allows pyarrow to work with modin.pandas.Series.
        import pyarrow

        return pyarrow.array(self._to_pandas(), type=type)

    def add(
        self, other, level=None, fill_value=None, axis=0
    ) -> Series:  # noqa: PR01, RT01, D200
        """
        Return Addition of series and other, element-wise (binary operator add).
        """
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).add(
            new_other, level=level, fill_value=fill_value, axis=axis
        )

    def radd(
        self, other, level=None, fill_value=None, axis=0
    ) -> Series:  # noqa: PR01, RT01, D200
        """
        Return Addition of series and other, element-wise (binary operator radd).
        """
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).radd(
            new_other, level=level, fill_value=fill_value, axis=axis
        )

    def add_prefix(
        self, prefix, axis=None
    ) -> Union[DataFrame, Series]:  # noqa: PR01, RT01, D200
        """
        Prefix labels with string `prefix`.
        """
        axis = 0 if axis is None else self._get_axis_number(axis)
        return self.__constructor__(
            query_compiler=self._query_compiler.add_prefix(prefix, axis=axis)
        )

    def add_suffix(
        self, suffix, axis=None
    ) -> Union[DataFrame, Series]:  # noqa: PR01, RT01, D200
        """
        Suffix labels with string `suffix`.
        """
        axis = 0 if axis is None else self._get_axis_number(axis)
        return self.__constructor__(
            query_compiler=self._query_compiler.add_suffix(suffix, axis=axis)
        )

    def aggregate(
        self, func=None, axis=0, *args, **kwargs
    ) -> Union[Series, Scalar]:  # noqa: PR01, RT01, D200
        """
        Aggregate using one or more operations over the specified axis.
        """

        def error_raiser(msg, exception):
            """Convert passed exception to the same type as pandas do and raise it."""
            # HACK: to concord with pandas error types by replacing all of the
            # TypeErrors to the AssertionErrors
            exception = exception if exception is not TypeError else AssertionError
            raise exception(msg)

        self._validate_function(func, on_invalid=error_raiser)
        return super(Series, self).aggregate(func, axis, *args, **kwargs)

    agg = aggregate

    def apply(
        self, func, convert_dtype=lib.no_default, args=(), by_row="compat", **kwargs
    ) -> Union[DataFrame, Series]:  # noqa: PR01, RT01, D200
        """
        Invoke function on values of Series.
        """
        if by_row != "compat":
            # TODO: add test
            return self._default_to_pandas(
                pandas.Series.apply,
                func=func,
                convert_dtype=convert_dtype,
                args=args,
                by_row=by_row,
                **kwargs,
            )

        if convert_dtype is lib.no_default:
            convert_dtype = True
        else:
            warnings.warn(
                "the convert_dtype parameter is deprecated and will be removed in a "
                + "future version.  Do ``ser.astype(object).apply()`` "
                + "instead if you want ``convert_dtype=False``.",
                FutureWarning,
            )

        func = cast_function_modin2pandas(func)
        self._validate_function(func)
        # apply and aggregate have slightly different behaviors, so we have to use
        # each one separately to determine the correct return type. In the case of
        # `agg`, the axis is set, but it is not required for the computation, so we use
        # it to determine which function to run.
        if kwargs.pop("axis", None) is not None:
            apply_func = "agg"
        else:
            apply_func = "apply"

        # This is the simplest way to determine the return type, but there are checks
        # in pandas that verify that some results are created. This is a challenge for
        # empty DataFrames, but fortunately they only happen when the `func` type is
        # a list or a dictionary, which means that the return type won't change from
        # type(self), so we catch that error and use `type(self).__name__` for the return
        # type.
        # We create a "dummy" `Series` to do the error checking and determining
        # the return type.
        try:
            return_type = type(
                getattr(
                    pandas.Series(self[:1].values, index=self.index[:1]), apply_func
                )(func, *args, **kwargs)
            ).__name__
        except Exception:
            return_type = type(self).__name__
        if (
            isinstance(func, str)
            or is_list_like(func)
            or return_type not in ["DataFrame", "Series"]
        ):
            # use the explicit non-Compat parent to avoid infinite recursion
            result = super(Series, self).apply(
                func,
                axis=0,
                raw=False,
                result_type=None,
                args=args,
                **kwargs,
            )
        else:
            # handle ufuncs and lambdas
            if kwargs or args and not isinstance(func, np.ufunc):

                def f(x):
                    return func(x, *args, **kwargs)

            else:
                f = func
            with np.errstate(all="ignore"):
                if isinstance(f, np.ufunc):
                    return f(self)

                # The return_type is only a DataFrame when we have a function
                # return a Series object. This is a very particular case that
                # has to be handled by the underlying pandas.Series apply
                # function and not our default map call.
                if return_type == "DataFrame":
                    result = self._query_compiler.apply_on_series(f)
                else:
                    result = self.map(f)._query_compiler

        if return_type == "DataFrame":
            from .dataframe import DataFrame

            result = DataFrame(query_compiler=result)
        elif return_type == "Series":
            result = self.__constructor__(query_compiler=result)
            if result.name == self.index[0]:
                result.name = None
        elif isinstance(result, type(self._query_compiler)):
            # sometimes result can be not a query_compiler, but scalar (for example
            # for sum or count functions)
            return result.to_pandas().squeeze()
        return result

    def transform(
        self, func, axis=0, *args, **kwargs
    ) -> Series:  # noqa: PR01, RT01, D200
        """
        Call ``func`` on self producing a `BasePandasDataset` with the same axis shape as self.
        """
        if isinstance(func, list):
            # drop nonunique functions to align with pandas behavior instead of getting
            # "pandas.errors.SpecificationError: Function names must be unique..."
            # Example:
            # >>> pandas.Series([0., 1., 4.]).transform(["sqrt", "sqrt"])
            # sqrt
            # 0   0.0
            # 1   1.0
            # 2   2.0
            unique_func = [func[0]]
            for one_func in func[1:]:
                if one_func not in unique_func:
                    unique_func.append(one_func)
            func = unique_func
        return super(Series, self).transform(func, axis, *args, **kwargs)

    def argmax(
        self, axis=None, skipna=True, *args, **kwargs
    ) -> int:  # noqa: PR01, RT01, D200
        """
        Return int position of the largest value in the Series.
        """
        result = self.idxmax(axis=axis, skipna=skipna, *args, **kwargs)
        if np.isnan(result) or result is pandas.NA:
            result = -1
        return result

    def argmin(
        self, axis=None, skipna=True, *args, **kwargs
    ) -> int:  # noqa: PR01, RT01, D200
        """
        Return int position of the smallest value in the Series.
        """
        result = self.idxmin(axis=axis, skipna=skipna, *args, **kwargs)
        if np.isnan(result) or result is pandas.NA:
            result = -1
        return result

    def argsort(
        self, axis=0, kind="quicksort", order=None, stable=None
    ) -> Series:  # noqa: PR01, RT01, D200
        """
        Return the integer indices that would sort the Series values.
        """
        return self.__constructor__(
            query_compiler=self._query_compiler.argsort(
                # 'stable' parameter has no effect in Pandas and is only accepted
                # for compatibility with NumPy, so we're not passing it forward on purpose
                axis=axis,
                kind=kind,
                order=order,
            )
        )

    def autocorr(self, lag=1) -> float:  # noqa: PR01, RT01, D200
        """
        Compute the lag-N autocorrelation.
        """
        return self.corr(self.shift(lag))

    def between(
        self, left, right, inclusive: str = "both"
    ) -> Series:  # noqa: PR01, RT01, D200
        """
        Return boolean Series equivalent to left <= series <= right.
        """
        # 'pandas.Series.between()' only uses public Series' API,
        # so passing a Modin Series there is safe
        return pandas.Series.between(self, left, right, inclusive)

    def combine(self, other, func, fill_value=None) -> Series:  # noqa: PR01, RT01, D200
        """
        Combine the Series with a Series or scalar according to `func`.
        """
        return super(Series, self).combine(
            other, lambda s1, s2: s1.combine(s2, func, fill_value=fill_value)
        )

    def compare(
        self,
        other: Series,
        align_axis: Union[str, int] = 1,
        keep_shape: bool = False,
        keep_equal: bool = False,
        result_names: tuple = ("self", "other"),
    ) -> Union[DataFrame, Series]:  # noqa: PR01, RT01, D200
        """
        Compare to another Series and show the differences.
        """
        if not isinstance(other, Series):
            raise TypeError(f"Cannot compare Series to {type(other)}")
        result = self.to_frame().compare(
            other.to_frame(),
            align_axis=align_axis,
            keep_shape=keep_shape,
            keep_equal=keep_equal,
            result_names=result_names,
        )
        if align_axis == "columns" or align_axis == 1:
            # Pandas.DataFrame.Compare returns a dataframe with a multidimensional index object as the
            # columns so we have to change column object back.
            result.columns = pandas.Index(["self", "other"])
        else:
            result = result.squeeze().rename(None)
        return result

    def corr(
        self, other, method="pearson", min_periods=None
    ) -> float:  # noqa: PR01, RT01, D200
        """
        Compute correlation with `other` Series, excluding missing values.
        """
        if method == "pearson":
            this, other = self.align(other, join="inner", copy=False)
            this = self.__constructor__(this)
            other = self.__constructor__(other)

            if len(this) == 0:
                return np.nan
            if len(this) != len(other):
                raise ValueError("Operands must have same size")

            if min_periods is None:
                min_periods = 1

            valid = this.notna() & other.notna()
            if not valid.all():
                this = this[valid]
                other = other[valid]
            if len(this) < min_periods:
                return np.nan

            this = this.astype(dtype="float64")
            other = other.astype(dtype="float64")
            this -= this.mean()
            other -= other.mean()

            other = other.__constructor__(query_compiler=other._query_compiler.conj())
            result = this * other / (len(this) - 1)
            result = np.array([result.sum()])

            stddev_this = ((this * this) / (len(this) - 1)).sum()
            stddev_other = ((other * other) / (len(other) - 1)).sum()

            stddev_this = np.array([np.sqrt(stddev_this)])
            stddev_other = np.array([np.sqrt(stddev_other)])

            result /= stddev_this * stddev_other

            np.clip(result.real, -1, 1, out=result.real)
            if np.iscomplexobj(result):
                np.clip(result.imag, -1, 1, out=result.imag)
            return result[0]

        return self._query_compiler.series_corr(
            other=other, method=method, min_periods=min_periods
        )

    def count(self) -> int:  # noqa: PR01, RT01, D200
        """
        Return number of non-NA/null observations in the Series.
        """
        return super(Series, self).count()

    def cov(
        self, other, min_periods=None, ddof: Optional[int] = 1
    ) -> float:  # noqa: PR01, RT01, D200
        """
        Compute covariance with Series, excluding missing values.
        """
        this, other = self.align(other, join="inner", copy=False)
        this = self.__constructor__(this)
        other = self.__constructor__(other)
        if len(this) == 0:
            return np.nan

        if len(this) != len(other):
            raise ValueError("Operands must have same size")

        if min_periods is None:
            min_periods = 1

        valid = this.notna() & other.notna()
        if not valid.all():
            this = this[valid]
            other = other[valid]

        if len(this) < min_periods:
            return np.nan

        this = this.astype(dtype="float64")
        other = other.astype(dtype="float64")

        this -= this.mean()
        other -= other.mean()

        other = other.__constructor__(query_compiler=other._query_compiler.conj())
        result = this * other / (len(this) - ddof)
        result = result.sum()
        return result

    def describe(
        self,
        percentiles=None,
        include=None,
        exclude=None,
    ) -> Union[DataFrame, Series]:  # noqa: PR01, RT01, D200
        """
        Generate descriptive statistics.
        """
        # Pandas ignores the `include` and `exclude` for Series for some reason.
        return super(Series, self).describe(
            percentiles=percentiles,
            include=None,
            exclude=None,
        )

    def diff(self, periods=1) -> Series:  # noqa: PR01, RT01, D200
        """
        First discrete difference of element.
        """
        return super(Series, self).diff(periods=periods, axis=0)

    def divmod(
        self, other, level=None, fill_value=None, axis=0
    ) -> tuple[Series, Series]:  # noqa: PR01, RT01, D200
        """
        Return Integer division and modulo of series and `other`, element-wise (binary operator `divmod`).
        """
        division, modulo = self._query_compiler.divmod(
            other=other, level=level, fill_value=fill_value, axis=axis
        )
        return self.__constructor__(query_compiler=division), self.__constructor__(
            query_compiler=modulo
        )

    def dot(self, other) -> Union[Series, np.ndarray]:  # noqa: PR01, RT01, D200
        """
        Compute the dot product between the Series and the columns of `other`.
        """
        if isinstance(other, BasePandasDataset):
            common = self.index.union(other.index)
            if len(common) > len(self.index) or len(common) > len(other.index):
                raise ValueError("Matrices are not aligned")

            qc = other.reindex(index=common)._query_compiler
            if isinstance(other, Series):
                return self._reduce_dimension(
                    query_compiler=self._query_compiler.dot(
                        qc, squeeze_self=True, squeeze_other=True
                    )
                )
            else:
                return self.__constructor__(
                    query_compiler=self._query_compiler.dot(
                        qc, squeeze_self=True, squeeze_other=False
                    )
                )

        other = np.asarray(other)
        if self.shape[0] != other.shape[0]:
            raise ValueError(
                "Dot product shape mismatch, {} vs {}".format(self.shape, other.shape)
            )

        if len(other.shape) > 1:
            return (
                self._query_compiler.dot(other, squeeze_self=True).to_numpy().squeeze()
            )

        return self._reduce_dimension(
            query_compiler=self._query_compiler.dot(other, squeeze_self=True)
        )

    def drop_duplicates(
        self, *, keep="first", inplace=False, ignore_index=False
    ) -> Union[Series, None]:  # noqa: PR01, RT01, D200
        """
        Return Series with duplicate values removed.
        """
        return super(Series, self).drop_duplicates(
            keep=keep, inplace=inplace, ignore_index=ignore_index
        )

    def dropna(
        self, *, axis=0, inplace=False, how=None, ignore_index=False
    ) -> Union[Series, None]:  # noqa: PR01, RT01, D200
        """
        Return a new Series with missing values removed.
        """
        return super(Series, self).dropna(
            axis=axis, inplace=inplace, ignore_index=ignore_index
        )

    def duplicated(self, keep="first") -> Series:  # noqa: PR01, RT01, D200
        """
        Indicate duplicate Series values.
        """
        return self.to_frame().duplicated(keep=keep)

    def eq(
        self, other, level=None, fill_value=None, axis=0
    ) -> Series:  # noqa: PR01, RT01, D200
        """
        Return Equal to of series and `other`, element-wise (binary operator `eq`).
        """
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).eq(new_other, level=level, axis=axis)

    def equals(self, other) -> bool:  # noqa: PR01, RT01, D200
        """
        Test whether two objects contain the same elements.
        """
        if isinstance(other, pandas.Series):
            # Copy into a Modin Series to simplify logic below
            other = self.__constructor__(other)

        if type(self) is not type(other) or not self.index.equals(other.index):
            return False

        old_name_self = self.name
        old_name_other = other.name
        try:
            self.name = "temp_name_for_equals_op"
            other.name = "temp_name_for_equals_op"
            # this function should return only scalar
            res = self.__constructor__(
                query_compiler=self._query_compiler.equals(other._query_compiler)
            )
        finally:
            self.name = old_name_self
            other.name = old_name_other
        return res.all()

    def explode(self, ignore_index: bool = False) -> Series:  # noqa: PR01, RT01, D200
        """
        Transform each element of a list-like to a row.
        """
        return super(Series, self).explode(
            MODIN_UNNAMED_SERIES_LABEL if self.name is None else self.name,
            ignore_index=ignore_index,
        )

    def factorize(self, sort=False, use_na_sentinel=True):  # noqa: PR01, RT01, D200
        """
        Encode the object as an enumerated type or categorical variable.
        """
        return self._default_to_pandas(
            pandas.Series.factorize,
            sort=sort,
            use_na_sentinel=use_na_sentinel,
        )

    def case_when(self, caselist) -> Series:  # noqa: PR01, RT01, D200
        """
        Replace values where the conditions are True.
        """
        modin_type = type(self)
        caselist = [
            tuple(
                data._query_compiler if isinstance(data, modin_type) else data
                for data in case_tuple
            )
            for case_tuple in caselist
        ]
        return self.__constructor__(
            query_compiler=self._query_compiler.case_when(caselist=caselist)
        )

    def fillna(
        self,
        value=None,
        *,
        method=None,
        axis=None,
        inplace=False,
        limit=None,
        downcast=lib.no_default,
    ) -> Union[Series, None]:  # noqa: PR01, RT01, D200
        """
        Fill NaNs inside of a Series object.
        """
        if isinstance(value, BasePandasDataset) and not isinstance(value, Series):
            raise TypeError(
                '"value" parameter must be a scalar, dict or Series, but '
                + f'you passed a "{type(value).__name__}"'
            )
        return super(Series, self).fillna(
            squeeze_self=True,
            squeeze_value=isinstance(value, Series),
            value=value,
            method=method,
            axis=axis,
            inplace=inplace,
            limit=limit,
            downcast=downcast,
        )

    def floordiv(
        self, other, level=None, fill_value=None, axis=0
    ) -> Series:  # noqa: PR01, RT01, D200
        """
        Get Integer division of series and `other`, element-wise (binary operator `floordiv`).
        """
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).floordiv(
            new_other, level=level, fill_value=None, axis=axis
        )

    def ge(
        self, other, level=None, fill_value=None, axis=0
    ) -> Series:  # noqa: PR01, RT01, D200
        """
        Return greater than or equal to of series and `other`, element-wise (binary operator `ge`).
        """
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).ge(new_other, level=level, axis=axis)

    def groupby(
        self,
        by=None,
        axis=0,
        level=None,
        as_index=True,
        sort=True,
        group_keys=True,
        observed=lib.no_default,
        dropna: bool = True,
    ):  # noqa: PR01, RT01, D200
        """
        Group Series using a mapper or by a Series of columns.
        """
        from .groupby import SeriesGroupBy

        if not as_index:
            raise TypeError("as_index=False only valid with DataFrame")
        # SeriesGroupBy expects a query compiler object if it is available
        if isinstance(by, Series):
            by = by._query_compiler
        elif callable(by):
            by = by(self.index)
        elif by is None and level is None:
            raise TypeError("You have to supply one of 'by' and 'level'")
        return SeriesGroupBy(
            self,
            by,
            axis,
            level,
            as_index,
            sort,
            group_keys,
            idx_name=None,
            observed=observed,
            drop=False,
            dropna=dropna,
        )

    def gt(
        self, other, level=None, fill_value=None, axis=0
    ) -> Series:  # noqa: PR01, RT01, D200
        """
        Return greater than of series and `other`, element-wise (binary operator `gt`).
        """
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).gt(new_other, level=level, axis=axis)

    def hist(
        self,
        by=None,
        ax=None,
        grid: bool = True,
        xlabelsize: int | None = None,
        xrot: float | None = None,
        ylabelsize: int | None = None,
        yrot: float | None = None,
        figsize: tuple[int, int] | None = None,
        bins: int | Sequence[int] = 10,
        backend: str | None = None,
        legend: bool = False,
        **kwargs,
    ):  # noqa: PR01, RT01, D200
        """
        Draw histogram of the input series using matplotlib.
        """
        return self._default_to_pandas(
            pandas.Series.hist,
            by=by,
            ax=ax,
            grid=grid,
            xlabelsize=xlabelsize,
            xrot=xrot,
            ylabelsize=ylabelsize,
            yrot=yrot,
            figsize=figsize,
            bins=bins,
            backend=backend,
            legend=legend,
            **kwargs,
        )

    def idxmax(
        self, axis=0, skipna=True, *args, **kwargs
    ) -> Hashable:  # noqa: PR01, RT01, D200
        """
        Return the row label of the maximum value.
        """
        return super(Series, self).idxmax(axis=axis, skipna=skipna, *args, **kwargs)

    def idxmin(
        self, axis=0, skipna=True, *args, **kwargs
    ) -> Hashable:  # noqa: PR01, RT01, D200
        """
        Return the row label of the minimum value.
        """
        return super(Series, self).idxmin(axis=axis, skipna=skipna, *args, **kwargs)

    def info(
        self,
        verbose: bool | None = None,
        buf: IO[str] | None = None,
        max_cols: int | None = None,
        memory_usage: bool | str | None = None,
        show_counts: bool = True,
    ) -> None:
        return SeriesInfo(self, memory_usage).render(
            buf=buf,
            max_cols=max_cols,
            verbose=verbose,
            show_counts=show_counts,
        )

    def isna(self) -> Series:
        """
        Detect missing values.

        Returns
        -------
        The result of detecting missing values.
        """
        return super(Series, self).isna()

    def isnull(self) -> Series:
        """
        Detect missing values.

        Returns
        -------
        The result of detecting missing values.
        """
        return super(Series, self).isnull()

    def item(self) -> Scalar:  # noqa: RT01, D200
        """
        Return the first element of the underlying data as a Python scalar.
        """
        return self[0]

    def items(self) -> Iterable[tuple[Hashable, Any]]:  # noqa: D200
        """
        Lazily iterate over (index, value) tuples.
        """

        def item_builder(s):
            return s.name, s.squeeze()

        partition_iterator = PartitionIterator(self.to_frame(), 0, item_builder)
        for v in partition_iterator:
            yield v

    def keys(self) -> pandas.Index:  # noqa: RT01, D200
        """
        Return alias for index.
        """
        return self.index

    def le(
        self, other, level=None, fill_value=None, axis=0
    ) -> Series:  # noqa: PR01, RT01, D200
        """
        Return less than or equal to of series and `other`, element-wise (binary operator `le`).
        """
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).le(new_other, level=level, axis=axis)

    def lt(
        self, other, level=None, fill_value=None, axis=0
    ) -> Series:  # noqa: PR01, RT01, D200
        """
        Return less than of series and `other`, element-wise (binary operator `lt`).
        """
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).lt(new_other, level=level, axis=axis)

    def map(self, arg, na_action=None) -> Series:  # noqa: PR01, RT01, D200
        """
        Map values of Series according to input correspondence.
        """
        if isinstance(arg, type(self)):
            # HACK: if we don't cast to pandas, then the execution engine will try to
            # propagate the distributed Series to workers and most likely would have
            # some performance problems.
            # TODO: A better way of doing so could be passing this `arg` as a query compiler
            # and broadcast accordingly.
            arg = arg._to_pandas()

        if not callable(arg) and hasattr(arg, "get"):
            mapper = arg

            def arg(s):
                return mapper.get(s, np.nan)

        return self.__constructor__(
            query_compiler=self._query_compiler.map(
                lambda s: (
                    arg(s) if pandas.isnull(s) is not True or na_action is None else s
                )
            )
        )

    def sem(
        self,
        axis: Optional[Axis] = None,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only=False,
        **kwargs,
    ) -> Union[float, Series]:  # noqa: PR01, RT01, D200
        """
        Return unbiased standard error of the mean over requested axis.
        """
        return super(Series, self)._stat_operation(
            "sem", axis, skipna, numeric_only, ddof=ddof, **kwargs
        )

    def std(
        self,
        axis: Optional[Axis] = None,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only=False,
        **kwargs,
    ) -> Union[float, Series]:  # noqa: PR01, RT01, D200
        """
        Return sample standard deviation over requested axis.
        """
        return super(Series, self)._stat_operation(
            "std", axis, skipna, numeric_only, ddof=ddof, **kwargs
        )

    def var(
        self,
        axis: Optional[Axis] = None,
        skipna: bool = True,
        ddof: int = 1,
        numeric_only=False,
        **kwargs,
    ) -> Union[float, Series]:  # noqa: PR01, RT01, D200
        """
        Return unbiased variance over requested axis.
        """
        return super(Series, self)._stat_operation(
            "var", axis, skipna, numeric_only, ddof=ddof, **kwargs
        )

    def memory_usage(self, index=True, deep=False) -> int:  # noqa: PR01, RT01, D200
        """
        Return the memory usage of the Series.
        """
        return super(Series, self).memory_usage(index=index, deep=deep).sum()

    def mod(
        self, other, level=None, fill_value=None, axis=0
    ) -> Series:  # noqa: PR01, RT01, D200
        """
        Return Modulo of series and `other`, element-wise (binary operator `mod`).
        """
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).mod(
            new_other, level=level, fill_value=None, axis=axis
        )

    def mode(self, dropna=True) -> Series:  # noqa: PR01, RT01, D200
        """
        Return the mode(s) of the Series.
        """
        return super(Series, self).mode(numeric_only=False, dropna=dropna)

    def mul(
        self, other, level=None, fill_value=None, axis=0
    ) -> Series:  # noqa: PR01, RT01, D200
        """
        Return multiplication of series and `other`, element-wise (binary operator `mul`).
        """
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).mul(
            new_other, level=level, fill_value=None, axis=axis
        )

    multiply = mul

    def rmul(
        self, other, level=None, fill_value=None, axis=0
    ) -> Series:  # noqa: PR01, RT01, D200
        """
        Return multiplication of series and `other`, element-wise (binary operator `mul`).
        """
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).rmul(
            new_other, level=level, fill_value=None, axis=axis
        )

    def ne(
        self, other, level=None, fill_value=None, axis=0
    ) -> Series:  # noqa: PR01, RT01, D200
        """
        Return not equal to of series and `other`, element-wise (binary operator `ne`).
        """
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).ne(new_other, level=level, axis=axis)

    def nlargest(self, n=5, keep="first") -> Series:  # noqa: PR01, RT01, D200
        """
        Return the largest `n` elements.
        """
        if len(self._query_compiler.columns) == 0:
            # pandas returns empty series when requested largest/smallest from empty series
            return self.__constructor__(data=[], dtype=float)
        return Series(
            query_compiler=self._query_compiler.nlargest(
                n=n, columns=self.name, keep=keep
            )
        )

    def nsmallest(self, n=5, keep="first") -> Series:  # noqa: PR01, RT01, D200
        """
        Return the smallest `n` elements.
        """
        if len(self._query_compiler.columns) == 0:
            # pandas returns empty series when requested largest/smallest from empty series
            return self.__constructor__(data=[], dtype=float)
        return self.__constructor__(
            query_compiler=self._query_compiler.nsmallest(
                n=n, columns=self.name, keep=keep
            )
        )

    def shift(
        self,
        periods=1,
        freq=None,
        axis=0,
        fill_value=lib.no_default,
        suffix=None,
    ) -> Series:  # noqa: PR01, RT01, D200
        """
        Shift index by desired number of periods with an optional time `freq`.
        """
        # pandas 2.1.0 ignores suffix parameter (https://github.com/pandas-dev/pandas/issues/54806)
        if freq is not None and fill_value is not lib.no_default:
            raise ValueError(
                "Cannot pass both 'freq' and 'fill_value' to "
                + f"{type(self).__name__}.shift"
            )
        if axis == 1:
            raise ValueError(
                f"No axis named {axis} for object type {type(self).__name__}"
            )
        return super(type(self), self).shift(
            periods=periods, freq=freq, axis=axis, fill_value=fill_value
        )

    def unstack(
        self, level=-1, fill_value=None, sort=True
    ) -> DataFrame:  # noqa: PR01, RT01, D200
        """
        Unstack, also known as pivot, Series with MultiIndex to produce DataFrame.
        """
        from .dataframe import DataFrame

        if not sort:
            # TODO: it should be easy to add support for sort == False
            return self._default_to_pandas(
                pandas.Series.unstack, level=level, fill_value=fill_value, sort=sort
            )

        # We can't unstack a Series object, if we don't have a MultiIndex.
        if len(self.index.names) > 1:
            result = DataFrame(
                query_compiler=self._query_compiler.unstack(level, fill_value)
            )
        else:
            raise ValueError(
                f"index must be a MultiIndex to unstack, {type(self.index)} was passed"
            )

        return result.droplevel(0, axis=1) if result.columns.nlevels > 1 else result

    @property
    def plot(
        self,
        kind="line",
        ax=None,
        figsize=None,
        use_index=True,
        title=None,
        grid=None,
        legend=False,
        style=None,
        logx=False,
        logy=False,
        loglog=False,
        xticks=None,
        yticks=None,
        xlim=None,
        ylim=None,
        rot=None,
        fontsize=None,
        colormap=None,
        table=False,
        yerr=None,
        xerr=None,
        label=None,
        secondary_y=False,
        **kwds,
    ):  # noqa: PR01, RT01, D200
        """
        Make plot of Series.
        """
        return self._to_pandas().plot

    def pow(
        self, other, level=None, fill_value=None, axis=0
    ) -> Series:  # noqa: PR01, RT01, D200
        """
        Return exponential power of series and `other`, element-wise (binary operator `pow`).
        """
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).pow(
            new_other, level=level, fill_value=None, axis=axis
        )

    @_inherit_docstrings(pandas.Series.prod, apilink="pandas.Series.prod")
    def prod(
        self,
        axis=None,
        skipna=True,
        numeric_only=False,
        min_count=0,
        **kwargs,
    ) -> Scalar:
        validate_bool_kwarg(skipna, "skipna", none_allowed=False)
        axis = self._get_axis_number(axis)
        new_index = self.columns if axis else self.index
        if min_count > len(new_index):
            return np.nan

        data = self._validate_dtypes_prod_mean(axis, numeric_only, ignore_axis=True)
        if min_count > 1:
            return data._reduce_dimension(
                data._query_compiler.prod_min_count(
                    axis=axis,
                    skipna=skipna,
                    numeric_only=numeric_only,
                    min_count=min_count,
                    **kwargs,
                )
            )
        return data._reduce_dimension(
            data._query_compiler.prod(
                axis=axis,
                skipna=skipna,
                numeric_only=numeric_only,
                min_count=min_count,
                **kwargs,
            )
        )

    product = prod

    def ravel(self, order="C") -> ArrayLike:  # noqa: PR01, RT01, D200
        """
        Return the flattened underlying data as an ndarray.
        """
        data = self._query_compiler.to_numpy().flatten(order=order)
        if isinstance(self.dtype, pandas.CategoricalDtype):
            data = pandas.Categorical(data, dtype=self.dtype)

        return data

    @_inherit_docstrings(pandas.Series.reindex, apilink="pandas.Series.reindex")
    def reindex(
        self,
        index=None,
        *,
        axis: Axis = None,
        method: str = None,
        copy: Optional[bool] = None,
        level=None,
        fill_value=None,
        limit: int = None,
        tolerance=None,
    ) -> Series:  # noqa: PR01, RT01, D200
        if fill_value is None:
            fill_value = np.nan
        return super(Series, self).reindex(
            index=index,
            columns=None,
            method=method,
            level=level,
            copy=copy,
            limit=limit,
            tolerance=tolerance,
            fill_value=fill_value,
        )

    def rename_axis(
        self,
        mapper=lib.no_default,
        *,
        index=lib.no_default,
        axis=0,
        copy=True,
        inplace=False,
    ) -> Union[Series, None]:  # noqa: PR01, RT01, D200
        """
        Set the name of the axis for the index or columns.
        """
        return super().rename_axis(
            mapper=mapper, index=index, axis=axis, copy=copy, inplace=inplace
        )

    def rename(
        self,
        index=None,
        *,
        axis=None,
        copy=None,
        inplace=False,
        level=None,
        errors="ignore",
    ) -> Union[Series, None]:  # noqa: PR01, RT01, D200
        """
        Alter Series index labels or name.
        """
        non_mapping = is_scalar(index) or (
            is_list_like(index) and not is_dict_like(index)
        )
        if non_mapping:
            if inplace:
                self.name = index
            else:
                self_cp = self.copy()
                self_cp.name = index
                return self_cp
        else:
            from .dataframe import DataFrame

            result = DataFrame(self.copy()).rename(index=index).squeeze(axis=1)
            result.name = self.name
            return result

    def repeat(self, repeats, axis=None) -> Series:  # noqa: PR01, RT01, D200
        """
        Repeat elements of a Series.
        """
        if (isinstance(repeats, int) and repeats == 0) or (
            is_list_like(repeats) and len(repeats) == 1 and repeats[0] == 0
        ):
            return self.__constructor__()

        return self.__constructor__(query_compiler=self._query_compiler.repeat(repeats))

    def reset_index(
        self,
        level=None,
        *,
        drop=False,
        name=lib.no_default,
        inplace=False,
        allow_duplicates=False,
    ) -> Union[DataFrame, Series, None]:  # noqa: PR01, RT01, D200
        """
        Generate a new Series with the index reset.
        """
        if name is lib.no_default:
            # For backwards compatibility, keep columns as [0] instead of
            #  [None] when self.name is None
            name = 0 if self.name is None else self.name

        if drop and level is None:
            new_idx = pandas.RangeIndex(len(self.index))
            if inplace:
                self.index = new_idx
            else:
                result = self.copy()
                result.index = new_idx
                return result
        elif not drop and inplace:
            raise TypeError(
                "Cannot reset_index inplace on a Series to create a DataFrame"
            )
        else:
            obj = self.copy()
            obj.name = name
            from .dataframe import DataFrame

            # Here `query_compiler` is passed instead of `obj` to avoid unnecessary `copy()`
            # inside `DataFrame` constructor
            return DataFrame(query_compiler=obj._query_compiler).reset_index(
                level=level,
                drop=drop,
                inplace=inplace,
                col_level=0,
                col_fill="",
                allow_duplicates=allow_duplicates,
                names=None,
            )

    def rdivmod(
        self, other, level=None, fill_value=None, axis=0
    ) -> Series:  # noqa: PR01, RT01, D200
        """
        Return integer division and modulo of series and `other`, element-wise (binary operator `rdivmod`).
        """
        division, modulo = self._query_compiler.rdivmod(
            other=other, level=level, fill_value=fill_value, axis=axis
        )
        return self.__constructor__(query_compiler=division), self.__constructor__(
            query_compiler=modulo
        )

    def rfloordiv(
        self, other, level=None, fill_value=None, axis=0
    ) -> Series:  # noqa: PR01, RT01, D200
        """
        Return integer division of series and `other`, element-wise (binary operator `rfloordiv`).
        """
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).rfloordiv(
            new_other, level=level, fill_value=None, axis=axis
        )

    def rmod(
        self, other, level=None, fill_value=None, axis=0
    ) -> Series:  # noqa: PR01, RT01, D200
        """
        Return modulo of series and `other`, element-wise (binary operator `rmod`).
        """
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).rmod(
            new_other, level=level, fill_value=None, axis=axis
        )

    def rpow(
        self, other, level=None, fill_value=None, axis=0
    ) -> Series:  # noqa: PR01, RT01, D200
        """
        Return exponential power of series and `other`, element-wise (binary operator `rpow`).
        """
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).rpow(
            new_other, level=level, fill_value=None, axis=axis
        )

    def rsub(
        self, other, level=None, fill_value=None, axis=0
    ) -> Series:  # noqa: PR01, RT01, D200
        """
        Return subtraction of series and `other`, element-wise (binary operator `rsub`).
        """
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).rsub(
            new_other, level=level, fill_value=None, axis=axis
        )

    def rtruediv(
        self, other, level=None, fill_value=None, axis=0
    ) -> Series:  # noqa: PR01, RT01, D200
        """
        Return floating division of series and `other`, element-wise (binary operator `rtruediv`).
        """
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).rtruediv(
            new_other, level=level, fill_value=None, axis=axis
        )

    rdiv = rtruediv

    def quantile(
        self, q=0.5, interpolation="linear"
    ) -> Union[float, Series]:  # noqa: PR01, RT01, D200
        """
        Return value at the given quantile.
        """
        return super(Series, self).quantile(
            q=q,
            axis=0,
            numeric_only=False,
            interpolation=interpolation,
            method="single",
        )

    def reorder_levels(self, order) -> Series:  # noqa: PR01, RT01, D200
        """
        Rearrange index levels using input order.
        """
        return super(Series, self).reorder_levels(order)

    def replace(
        self,
        to_replace=None,
        value=lib.no_default,
        *,
        inplace=False,
        limit=None,
        regex=False,
        method: str | lib.NoDefault = lib.no_default,
    ) -> Series:  # noqa: PR01, RT01, D200
        """
        Replace values given in `to_replace` with `value`.
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        new_query_compiler = self._query_compiler.replace(
            to_replace=to_replace,
            value=value,
            inplace=False,
            limit=limit,
            regex=regex,
            method=method,
        )
        return self._create_or_update_from_compiler(new_query_compiler, inplace)

    def searchsorted(
        self, value, side="left", sorter=None
    ) -> Union[npt.NDArray[np.intp], np.intp]:  # noqa: PR01, RT01, D200
        """
        Find indices where elements should be inserted to maintain order.
        """
        searchsorted_qc = self._query_compiler
        if sorter is not None:
            # `iloc` method works slowly (https://github.com/modin-project/modin/issues/1903),
            # so _default_to_pandas is used for now
            # searchsorted_qc = self.iloc[sorter].reset_index(drop=True)._query_compiler
            # sorter = None
            return self._default_to_pandas(
                pandas.Series.searchsorted, value, side=side, sorter=sorter
            )
        # searchsorted should return item number irrespective of Series index, so
        # Series.index is always set to pandas.RangeIndex, which can be easily processed
        # on the query_compiler level
        if not isinstance(searchsorted_qc.index, pandas.RangeIndex):
            searchsorted_qc = searchsorted_qc.reset_index(drop=True)

        result = self.__constructor__(
            query_compiler=searchsorted_qc.searchsorted(
                value=value, side=side, sorter=sorter
            )
        ).squeeze()

        # matching Pandas output
        if not is_scalar(value) and not is_list_like(result):
            result = np.array([result])
        elif isinstance(result, type(self)):
            result = result.to_numpy()

        return result

    def sort_values(
        self,
        *,
        axis=0,
        ascending=True,
        inplace=False,
        kind="quicksort",
        na_position="last",
        ignore_index: bool = False,
        key: Optional[IndexKeyFunc] = None,
    ) -> Union[Series, None]:  # noqa: PR01, RT01, D200
        """
        Sort by the values.
        """
        from .dataframe import DataFrame

        # When we convert to a DataFrame, the name is automatically converted to 0 if it
        # is None, so we do this to avoid a KeyError.
        by = self.name if self.name is not None else 0
        result = (
            DataFrame(self.copy())
            .sort_values(
                by=by,
                ascending=ascending,
                inplace=False,
                kind=kind,
                na_position=na_position,
                ignore_index=ignore_index,
                key=key,
            )
            .squeeze(axis=1)
        )
        result.name = self.name
        return self._create_or_update_from_compiler(
            result._query_compiler, inplace=inplace
        )

    cat = CachedAccessor("cat", CategoryMethods)
    sparse = CachedAccessor("sparse", SparseAccessor)
    str = CachedAccessor("str", StringMethods)
    dt = CachedAccessor("dt", DatetimeProperties)
    list = CachedAccessor("list", ListAccessor)
    struct = CachedAccessor("struct", StructAccessor)

    def squeeze(self, axis=None) -> Union[Series, Scalar]:  # noqa: PR01, RT01, D200
        """
        Squeeze 1 dimensional axis objects into scalars.
        """
        if axis is not None:
            # Validate `axis`
            pandas.Series._get_axis_number(axis)
        if len(self.index) == 1:
            return self._reduce_dimension(self._query_compiler)
        else:
            return self.copy()

    def sub(
        self, other, level=None, fill_value=None, axis=0
    ) -> Series:  # noqa: PR01, RT01, D200
        """
        Return subtraction of Series and `other`, element-wise (binary operator `sub`).
        """
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).sub(
            new_other, level=level, fill_value=None, axis=axis
        )

    subtract = sub

    def sum(
        self,
        axis=None,
        skipna=True,
        numeric_only=False,
        min_count=0,
        **kwargs,
    ) -> Scalar:  # noqa: PR01, RT01, D200
        """
        Return the sum of the values.
        """
        validate_bool_kwarg(skipna, "skipna", none_allowed=False)
        axis = self._get_axis_number(axis)

        new_index = self.columns if axis else self.index
        if min_count > len(new_index):
            return np.nan

        data = self._validate_dtypes_prod_mean(axis, numeric_only, ignore_axis=False)
        if min_count > 1:
            return data._reduce_dimension(
                data._query_compiler.sum_min_count(
                    axis=axis,
                    skipna=skipna,
                    numeric_only=numeric_only,
                    min_count=min_count,
                    **kwargs,
                )
            )
        return data._reduce_dimension(
            data._query_compiler.sum(
                axis=axis,
                skipna=skipna,
                numeric_only=numeric_only,
                min_count=min_count,
                **kwargs,
            )
        )

    def swaplevel(self, i=-2, j=-1, copy=None) -> Series:  # noqa: PR01, RT01, D200
        """
        Swap levels `i` and `j` in a `MultiIndex`.
        """
        copy = True if copy is None else copy
        obj = self.copy() if copy else self
        return super(Series, obj).swaplevel(i, j, axis=0)

    def take(self, indices, axis=0, **kwargs) -> Series:  # noqa: PR01, RT01, D200
        """
        Return the elements in the given positional indices along an axis.
        """
        return super(Series, self).take(indices, axis=axis, **kwargs)

    def to_dict(self, into=dict) -> dict:  # pragma: no cover # noqa: PR01, RT01, D200
        """
        Convert Series to {label -> value} dict or dict-like object.
        """
        return self._query_compiler.series_to_dict(into)

    def to_frame(
        self, name: Hashable = lib.no_default
    ) -> DataFrame:  # noqa: PR01, RT01, D200
        """
        Convert Series to {label -> value} dict or dict-like object.
        """
        from .dataframe import DataFrame

        if name is None:
            name = lib.no_default

        self_cp = self.copy()
        if name is not lib.no_default:
            self_cp.name = name

        return DataFrame(self_cp)

    def to_list(self) -> list:  # noqa: RT01, D200
        """
        Return a list of the values.
        """
        return self._query_compiler.to_list()

    def to_numpy(
        self, dtype=None, copy=False, na_value=lib.no_default, **kwargs
    ) -> np.ndarray:  # noqa: PR01, RT01, D200
        """
        Return the NumPy ndarray representing the values in this Series or Index.
        """
        from modin.config import ModinNumpy

        if not ModinNumpy.get():
            return (
                super(Series, self)
                .to_numpy(
                    dtype=dtype,
                    copy=copy,
                    na_value=na_value,
                )
                .flatten()
            )
        else:
            from ..numpy.arr import array

            return array(self, copy=copy)

    tolist = to_list

    # TODO(williamma12): When we implement to_timestamp, have this call the version
    # in base.py
    def to_period(self, freq=None, copy=None) -> Series:  # noqa: PR01, RT01, D200
        """
        Cast to PeriodArray/Index at a particular frequency.
        """
        return self._default_to_pandas("to_period", freq=freq, copy=copy)

    def to_string(
        self,
        buf=None,
        na_rep="NaN",
        float_format=None,
        header=True,
        index=True,
        length=False,
        dtype=False,
        name=False,
        max_rows=None,
        min_rows=None,
    ) -> Union[str, None]:  # noqa: PR01, RT01, D200
        """
        Render a string representation of the Series.
        """
        return self._default_to_pandas(
            pandas.Series.to_string,
            buf=buf,
            na_rep=na_rep,
            float_format=float_format,
            header=header,
            index=index,
            length=length,
            dtype=dtype,
            name=name,
            max_rows=max_rows,
        )

    # TODO(williamma12): When we implement to_timestamp, have this call the version
    # in base.py
    def to_timestamp(
        self, freq=None, how="start", copy=None
    ) -> Series:  # noqa: PR01, RT01, D200
        """
        Cast to DatetimeIndex of Timestamps, at beginning of period.
        """
        return self._default_to_pandas("to_timestamp", freq=freq, how=how, copy=copy)

    def transpose(self, *args, **kwargs) -> Series:  # noqa: PR01, RT01, D200
        """
        Return the transpose, which is by definition `self`.
        """
        return self

    T: Series = property(transpose)

    def truediv(
        self, other, level=None, fill_value=None, axis=0
    ) -> Series:  # noqa: PR01, RT01, D200
        """
        Return floating division of series and `other`, element-wise (binary operator `truediv`).
        """
        new_self, new_other = self._prepare_inter_op(other)
        return super(Series, new_self).truediv(
            new_other, level=level, fill_value=None, axis=axis
        )

    div = divide = truediv

    def unique(self) -> ArrayLike:  # noqa: RT01, D200
        """
        Return unique values of Series object.
        """
        # `values` can't be used here because it performs unnecessary conversion,
        # after which the result type does not match the pandas
        return (
            self.__constructor__(query_compiler=self._query_compiler.unique())
            .modin.to_pandas()
            ._values
        )

    def update(self, other) -> None:  # noqa: PR01, D200
        """
        Modify Series in place using values from passed Series.
        """
        if not isinstance(other, Series):
            other = self.__constructor__(other)
        query_compiler = self._query_compiler.series_update(other._query_compiler)
        self._update_inplace(new_query_compiler=query_compiler)

    def value_counts(
        self, normalize=False, sort=True, ascending=False, bins=None, dropna=True
    ) -> Series:  # noqa: PR01, RT01, D200
        """
        Return a Series containing counts of unique values.
        """
        if bins is not None:
            # Potentially we could implement `cut` function from pandas API, which
            # bins values into intervals, and then we can just count them as regular values.
            # TODO #1333: new_self = self.__constructor__(pd.cut(self, bins, include_lowest=True), dtype="interval")
            return self._default_to_pandas(
                pandas.Series.value_counts,
                normalize=normalize,
                sort=sort,
                ascending=ascending,
                bins=bins,
                dropna=dropna,
            )
        counted_values = super(Series, self).value_counts(
            subset=self,
            normalize=normalize,
            sort=sort,
            ascending=ascending,
            dropna=dropna,
        )
        return counted_values

    def view(self, dtype=None) -> Series:  # noqa: PR01, RT01, D200
        """
        Create a new view of the Series.
        """
        return self.__constructor__(
            query_compiler=self._query_compiler.series_view(dtype=dtype)
        )

    def where(
        self,
        cond,
        other=np.nan,
        *,
        inplace=False,
        axis=None,
        level=None,
    ) -> Union[Series, None]:  # noqa: PR01, RT01, D200
        """
        Replace values where the condition is False.
        """
        # TODO: probably need to remove this conversion to pandas
        if isinstance(other, Series):
            other = to_pandas(other)
        # TODO: add error checking like for dataframe where, then forward to
        # same query compiler method
        return self._default_to_pandas(
            pandas.Series.where,
            cond,
            other=other,
            inplace=inplace,
            axis=axis,
            level=level,
        )

    @property
    def attrs(self) -> dict:  # noqa: RT01, D200
        """
        Return dictionary of global attributes of this dataset.
        """

        def attrs(df):
            return df.attrs

        return self._default_to_pandas(attrs)

    @property
    def array(self) -> ExtensionArray:  # noqa: RT01, D200
        """
        Return the ExtensionArray of the data backing this Series or Index.
        """

        def array(df):
            return df.array

        return self._default_to_pandas(array)

    @property
    def axes(self) -> list[pandas.Index]:  # noqa: RT01, D200
        """
        Return a list of the row axis labels.
        """
        return [self.index]

    @property
    def dtype(self) -> DtypeObj:  # noqa: RT01, D200
        """
        Return the dtype object of the underlying data.
        """
        return self._query_compiler.dtypes.squeeze()

    dtypes = dtype

    @property
    def empty(self) -> bool:  # noqa: RT01, D200
        """
        Indicate whether Series is empty.
        """
        return len(self.index) == 0

    @property
    def hasnans(self) -> bool:  # noqa: RT01, D200
        """
        Return True if Series has any nans.
        """
        return self.isna().sum() > 0

    @property
    def is_monotonic_increasing(self) -> bool:  # noqa: RT01, D200
        """
        Return True if values in the Series are monotonic_increasing.
        """
        return self._reduce_dimension(self._query_compiler.is_monotonic_increasing())

    @property
    def is_monotonic_decreasing(self) -> bool:  # noqa: RT01, D200
        """
        Return True if values in the Series are monotonic_decreasing.
        """
        return self._reduce_dimension(self._query_compiler.is_monotonic_decreasing())

    @property
    def is_unique(self) -> bool:  # noqa: RT01, D200
        """
        Return True if values in the Series are unique.
        """
        return self.nunique(dropna=False) == len(self)

    @property
    def nbytes(self) -> int:  # noqa: RT01, D200
        """
        Return the number of bytes in the underlying data.
        """
        return self.memory_usage(index=False)

    @property
    def ndim(self) -> int:  # noqa: RT01, D200
        """
        Return the number of dimensions of the underlying data, by definition 1.
        """
        return 1

    def nunique(self, dropna=True) -> int:  # noqa: PR01, RT01, D200
        """
        Return number of unique elements in the object.
        """
        return super(Series, self).nunique(dropna=dropna)

    @property
    def shape(self) -> tuple[int]:  # noqa: RT01, D200
        """
        Return a tuple of the shape of the underlying data.
        """
        return (len(self),)

    def reindex_like(
        self,
        other,
        method=None,
        copy: Optional[bool] = None,
        limit=None,
        tolerance=None,
    ) -> Series:
        # docs say "Same as calling .reindex(index=other.index, columns=other.columns,...).":
        # https://pandas.pydata.org/pandas-docs/version/1.4/reference/api/pandas.Series.reindex_like.html
        return self.reindex(
            index=other.index,
            method=method,
            copy=copy,
            limit=limit,
            tolerance=tolerance,
        )

    def _to_pandas(self) -> pandas.Series:
        """
        Convert Modin Series to pandas Series.

        Recommended conversion method: `series.modin.to_pandas()`.

        Returns
        -------
        pandas.Series
        """
        df = self._query_compiler.to_pandas()
        series = df[df.columns[0]]
        if self._query_compiler.columns[0] == MODIN_UNNAMED_SERIES_LABEL:
            series.name = None
        return series

    def _to_datetime(self, **kwargs) -> Series:
        """
        Convert `self` to datetime.

        Parameters
        ----------
        **kwargs : dict
            Optional arguments to use during query compiler's
            `to_datetime` invocation.

        Returns
        -------
        datetime
            Series of datetime64 dtype.
        """
        return self.__constructor__(
            query_compiler=self._query_compiler.to_datetime(**kwargs)
        )

    def _to_numeric(self, **kwargs) -> Series:
        """
        Convert `self` to numeric.

        Parameters
        ----------
        **kwargs : dict
            Optional arguments to use during query compiler's
            `to_numeric` invocation.

        Returns
        -------
        numeric
            Series of numeric dtype.
        """
        return self.__constructor__(
            query_compiler=self._query_compiler.to_numeric(**kwargs)
        )

    def _qcut(self, q, **kwargs):  # noqa: PR01, RT01, D200
        """
        Quantile-based discretization function.
        """
        return self._default_to_pandas(pandas.qcut, q, **kwargs)

    def _reduce_dimension(self, query_compiler) -> Series | Scalar:
        """
        Try to reduce the dimension of data from the `query_compiler`.

        Parameters
        ----------
        query_compiler : BaseQueryCompiler
            Query compiler to retrieve the data.

        Returns
        -------
        pandas.Series or scalar.
        """
        return query_compiler.to_pandas().squeeze()

    def _validate_dtypes_prod_mean(
        self, axis, numeric_only, ignore_axis=False
    ) -> Series:
        """
        Validate data dtype for `prod` and `mean` methods.

        Parameters
        ----------
        axis : {0, 1}
            Axis to validate over.
        numeric_only : bool
            Whether or not to allow only numeric data.
            If True and non-numeric data is found, exception
            will be raised.
        ignore_axis : bool, default: False
            Whether or not to ignore `axis` parameter.

        Returns
        -------
        Series

        Notes
        -----
        Actually returns unmodified `self` object,
        added for compatibility with Modin DataFrame.
        """
        return self

    def _validate_dtypes_min_max(self, axis, numeric_only) -> Series:
        """
        Validate data dtype for `min` and `max` methods.

        Parameters
        ----------
        axis : {0, 1}
            Axis to validate over.
        numeric_only : bool
            Whether or not to allow only numeric data.
            If True and non-numeric data is found, exception.

        Returns
        -------
        Series

        Notes
        -----
        Actually returns unmodified `self` object,
        added for compatibility with Modin DataFrame.
        """
        return self

    def _validate_dtypes(self, numeric_only=False) -> None:
        """
        Check that all the dtypes are the same.

        Parameters
        ----------
        numeric_only : bool, default: False
            Whether or not to allow only numeric data.
            If True and non-numeric data is found, exception
            will be raised.

        Notes
        -----
        Actually does nothing, added for compatibility with Modin DataFrame.
        """
        pass

    def _get_numeric_data(self, axis: int) -> Series:
        """
        Grab only numeric data from Series.

        Parameters
        ----------
        axis : {0, 1}
            Axis to inspect on having numeric types only.

        Returns
        -------
        Series

        Notes
        -----
        `numeric_only` parameter is not supported by Series, so this method
        does not do anything. The method is added for compatibility with Modin DataFrame.
        """
        return self

    def _update_inplace(self, new_query_compiler) -> None:
        """
        Update the current Series in-place using `new_query_compiler`.

        Parameters
        ----------
        new_query_compiler : BaseQueryCompiler
            QueryCompiler to use to manage the data.
        """
        super(Series, self)._update_inplace(new_query_compiler=new_query_compiler)
        # Propagate changes back to parent so that column in dataframe had the same contents
        if self._parent is not None:
            if self._parent_axis == 0:
                self._parent.loc[self.name] = self
            else:
                self._parent[self.name] = self

    def _create_or_update_from_compiler(
        self, new_query_compiler, inplace=False
    ) -> Union[Series, None]:
        """
        Return or update a Series with given `new_query_compiler`.

        Parameters
        ----------
        new_query_compiler : PandasQueryCompiler
            QueryCompiler to use to manage the data.
        inplace : bool, default: False
            Whether or not to perform update or creation inplace.

        Returns
        -------
        Series or None
            None if update was done, Series otherwise.
        """
        assert (
            isinstance(new_query_compiler, type(self._query_compiler))
            or type(new_query_compiler) in self._query_compiler.__class__.__bases__
        ), "Invalid Query Compiler object: {}".format(type(new_query_compiler))
        if not inplace and new_query_compiler.is_series_like():
            return self.__constructor__(query_compiler=new_query_compiler)
        elif not inplace:
            # This can happen with things like `reset_index` where we can add columns.
            from .dataframe import DataFrame

            return DataFrame(query_compiler=new_query_compiler)
        else:
            self._update_inplace(new_query_compiler=new_query_compiler)

    def _prepare_inter_op(self, other) -> tuple[Series, Series]:
        """
        Prepare `self` and `other` for further interaction.

        Parameters
        ----------
        other : Series or scalar value
            Another object `self` should interact with.

        Returns
        -------
        Series
            Prepared `self`.
        Series
            Prepared `other`.
        """
        if isinstance(other, Series):
            names_different = self.name != other.name
            # NB: if we don't need a rename, do the interaction with shallow
            # copies so that we preserve obj.index._id. It's fine to work
            # with shallow copies because we'll discard the copies but keep
            # the result after the interaction opreation. We can't do a rename
            # on shallow copies because we'll mutate the original objects.
            new_self = self.copy(deep=names_different)
            new_other = other.copy(deep=names_different)
            if names_different:
                new_self.name = new_other.name = MODIN_UNNAMED_SERIES_LABEL
        else:
            new_self = self
            new_other = other
        return new_self, new_other

    def _getitem(self, key) -> Union[Series, Scalar]:
        """
        Get the data specified by `key` for this Series.

        Parameters
        ----------
        key : Any
            Column id to retrieve from Series.

        Returns
        -------
        Series or scalar
            Retrieved data.
        """
        key = apply_if_callable(key, self)
        if isinstance(key, Series) and key.dtype == np.bool_:
            # This ends up being significantly faster than looping through and getting
            # each item individually.
            key = key._to_pandas()
        if is_bool_indexer(key):
            return self.__constructor__(
                query_compiler=self._query_compiler.getitem_row_array(
                    pandas.RangeIndex(len(self.index))[key]
                )
            )
        # TODO: More efficiently handle `tuple` case for `Series.__getitem__`
        if isinstance(key, tuple):
            return self._default_to_pandas(pandas.Series.__getitem__, key)

        if not is_list_like(key):
            reduce_dimension = True
            key = [key]
        else:
            reduce_dimension = False
        # The check for whether or not `key` is in `keys()` will throw a TypeError
        # if the object is not hashable. When that happens, we just assume the
        # key is a list-like of row positions.
        try:
            is_indexer = all(k in self.keys() for k in key)
        except TypeError:
            is_indexer = False
        row_positions = self.index.get_indexer_for(key) if is_indexer else key
        if not all(is_integer(x) for x in row_positions):
            raise KeyError(key[0] if reduce_dimension else key)
        result = self._query_compiler.getitem_row_array(row_positions)

        if reduce_dimension:
            return self._reduce_dimension(result)
        return self.__constructor__(query_compiler=result)

    def _repartition(self) -> Series:
        """
        Repartitioning Series to get ideal partitions inside.

        Allows to improve performance where the query compiler can't improve
        yet by doing implicit repartitioning.

        Returns
        -------
        Series
            The repartitioned Series.
        """
        return super()._repartition(axis=0)

    # Persistance support methods - BEGIN
    @classmethod
    def _inflate_light(cls, query_compiler, name, source_pid) -> Series:
        """
        Re-creates the object from previously-serialized lightweight representation.

        The method is used for faster but not disk-storable persistence.

        Parameters
        ----------
        query_compiler : BaseQueryCompiler
            Query compiler to use for object re-creation.
        name : str
            The name to give to the new object.
        source_pid : int
            Determines whether a Modin or pandas object needs to be created.
            Modin objects are created only on the main process.

        Returns
        -------
        Series
            New Series based on the `query_compiler`.
        """
        if os.getpid() != source_pid:
            res = query_compiler.to_pandas()
            # at the query compiler layer, `to_pandas` always returns a DataFrame,
            # even if it stores a Series, as a single-column DataFrame
            if res.columns == [MODIN_UNNAMED_SERIES_LABEL]:
                res = res.squeeze(axis=1)
                res.name = None
            return res
        # The current logic does not involve creating Modin objects
        # and manipulation with them in worker processes
        return cls(query_compiler=query_compiler, name=name)

    @classmethod
    def _inflate_full(cls, pandas_series, source_pid) -> Series:
        """
        Re-creates the object from previously-serialized disk-storable representation.

        Parameters
        ----------
        pandas_series : pandas.Series
            Data to use for object re-creation.
        source_pid : int
            Determines whether a Modin or pandas object needs to be created.
            Modin objects are created only on the main process.

        Returns
        -------
        Series
            New Series based on the `pandas_series`.
        """
        if os.getpid() != source_pid:
            return pandas_series
        # The current logic does not involve creating Modin objects
        # and manipulation with them in worker processes
        return cls(data=pandas_series)

    def __reduce__(self):
        self._query_compiler.finalize()
        pid = os.getpid()
        if (
            PersistentPickle.get()
            or not self._query_compiler.support_materialization_in_worker_process()
        ):
            return self._inflate_full, (self._to_pandas(), pid)
        return self._inflate_light, (self._query_compiler, self.name, pid)

    # Persistance support methods - END
