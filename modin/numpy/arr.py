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
"""Module houses ``array`` class, that is distributed version of ``numpy.array``."""

from math import prod
import numpy
from pandas.core.dtypes.common import is_list_like
from pandas.api.types import is_scalar
from inspect import signature

import modin.pandas as pd
from modin.error_message import ErrorMessage
from modin.core.dataframe.algebra import (
    Map,
    Reduce,
    Binary,
)


_INTEROPERABLE_TYPES = (pd.DataFrame, pd.Series)


class array(object):
    """
    Modin distributed representation of ``numpy.array``.

    Internally, the data can be divided into partitions along both columns and rows
    in order to parallelize computations and utilize the user's hardware as much as possible.

    Notes
    -----
    The ``array`` class is a lightweight shim that relies on the pandas Query Compiler in order to
    provide functionality.
    """

    def __init__(
        self,
        object=None,
        dtype=None,
        *,
        copy=True,
        order="K",
        subok=False,
        ndmin=0,
        like=numpy._NoValue,
        _query_compiler=None,
        _ndim=None,
    ):
        if _query_compiler is not None:
            self._query_compiler = _query_compiler
            self._ndim = _ndim
        elif is_list_like(object) and not is_list_like(object[0]):
            self._query_compiler = pd.Series(object)._query_compiler
            self._ndim = 1
        else:
            target_kwargs = {
                "dtype": None,
                "copy": True,
                "order": "K",
                "subok": False,
                "ndmin": 0,
                "like": numpy._NoValue,
            }
            for key, value in target_kwargs.copy().items():
                if value == locals()[key]:
                    target_kwargs.pop(key)
                else:
                    target_kwargs[key] = locals()[key]
            arr = numpy.array(object, **target_kwargs)
            self._ndim = len(arr.shape)
            if self._ndim > 2:
                ErrorMessage.not_implemented(
                    "NumPy arrays with dimensions higher than 2 are not yet supported."
                )

            self._query_compiler = pd.DataFrame(arr)._query_compiler
        # These two lines are necessary so that our query compiler does not keep track of indices
        # and try to map like indices to like indices. (e.g. if we multiply two arrays that used
        # to be dataframes, and the dataframes had the same column names but ordered differently
        # we want to do a simple broadcast where we only consider position, as numpy would, rather
        # than pair columns with the same name and multiply them.)
        self._query_compiler = self._query_compiler.reset_index(drop=True)
        self._query_compiler.columns = range(len(self._query_compiler.columns))

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        ufunc_name = ufunc.__name__
        supported_array_layer = hasattr(self, ufunc_name) or hasattr(
            self, f"__{ufunc_name}__"
        )
        if supported_array_layer:
            args = []
            for input in inputs:
                if not (isinstance(input, array) or is_scalar(input)):
                    if isinstance(input, _INTEROPERABLE_TYPES):
                        ndim = 2 if isinstance(input, pd.DataFrame) else 1
                        input = array(_query_compiler=input._query_compiler, _ndim=ndim)
                    else:
                        input = array(input)
                args += [input]
            function = (
                getattr(args[0], ufunc_name)
                if hasattr(args[0], ufunc_name)
                else getattr(args[0], f"__{ufunc_name}__")
            )
            len_expected_arguments = len(
                [
                    param
                    for param in signature(function).parameters.values()
                    if param.kind == param.POSITIONAL_ONLY
                ]
            )
            if len_expected_arguments == len(args):
                return function(*tuple(args[1:]), **kwargs)
            else:
                ErrorMessage.single_warning(
                    f"{ufunc} method {method} is not yet supported in Modin. Defaulting to NumPy."
                )
                args = []
                for input in inputs:
                    if isinstance(input, array):
                        input = input._to_numpy()
                    if isinstance(input, pd.DataFrame):
                        input = input._query_compiler.to_numpy()
                    if isinstance(input, pd.Series):
                        input = input._query_compiler.to_numpy().flatten()
                    args += [input]
                output = args[0].__array_ufunc__(ufunc, method, *args, **kwargs)
                if is_scalar(output):
                    return output
                return array(output)
        new_ufunc = None
        out_ndim = -1
        if method == "__call__":
            if len(inputs) == 1:
                new_ufunc = Map.register(ufunc)
                out_ndim = len(inputs[0].shape)
            else:
                new_ufunc = Binary.register(ufunc)
                out_ndim = max([len(inp.shape) for inp in inputs])
        elif method == "reduce":
            new_ufunc = Reduce.register(ufunc, axis=kwargs.get("axis", None))
            if kwargs.get("axis", None) is None:
                out_ndim = 0
            else:
                out_ndim = len(inputs[0].shape) - 1
        elif method == "accumulate":
            new_ufunc = Reduce.register(ufunc, axis=None)
            out_ndim = 0
        if new_ufunc is None:
            ErrorMessage.single_warning(
                f"{ufunc} is not yet supported in Modin. Defaulting to NumPy."
            )
            args = []
            for input in inputs:
                if isinstance(input, array):
                    input = input._to_numpy()
                if isinstance(input, pd.DataFrame):
                    input = input._query_compiler.to_numpy()
                if isinstance(input, pd.Series):
                    input = input._query_compiler.to_numpy().flatten()
                args += [input]
            output = ufunc(*args, **kwargs)
            if is_scalar(output):
                return output
            return array(output)
        args = []
        for input in inputs:
            if not (isinstance(input, array) or is_scalar(input)):
                if isinstance(input, _INTEROPERABLE_TYPES):
                    ndim = 2 if isinstance(input, pd.DataFrame) else 1
                    input = array(_query_compiler=input._query_compiler, _ndim=ndim)
                else:
                    input = array(input)
            args += [
                input._query_compiler if hasattr(input, "_query_compiler") else input
            ]
        return array(_query_compiler=new_ufunc(*args, **kwargs), _ndim=out_ndim)

    def __array_function__(self, func, types, args, kwargs):
        if func.__name__ == "ravel":
            return self.flatten()
        return NotImplemented

    def __abs__(
        self,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        result = self._query_compiler.abs()
        return array(_query_compiler=result, _ndim=self._ndim)

    absolute = __abs__

    def _binary_op(self, other):
        if not isinstance(other, array):
            if isinstance(other, _INTEROPERABLE_TYPES):
                ndim = 2 if isinstance(other, pd.DataFrame) else 1
                other = array(_query_compiler=other._query_compiler, _ndim=ndim)
            else:
                raise TypeError(
                    f"Unsupported operand type(s) for divide: '{type(self)}' and '{type(other)}'"
                )
        broadcast = self._ndim != other._ndim
        if broadcast:
            # In this case, we have a 1D object doing a binary op with a 2D object
            caller, callee = (self, other) if self._ndim == 2 else (other, self)
            if callee.shape[0] != caller.shape[1]:
                raise ValueError(
                    f"operands could not be broadcast together with shapes {self.shape} {other.shape}"
                )
            return (caller, callee, caller._ndim, {"broadcast": broadcast, "axis": 1})
        else:
            if self.shape != other.shape:
                # In this case, we either have two mismatched objects trying to do an operation
                # or a nested 1D object that must be broadcasted trying to do an operation.
                if self.shape[0] == other.shape[0]:
                    matched_dimension = 0
                elif self.shape[1] == other.shape[1]:
                    matched_dimension = 1
                else:
                    raise ValueError(
                        f"operands could not be broadcast together with shapes {self.shape} {other.shape}"
                    )
                if (
                    self.shape[matched_dimension ^ 1] == 1
                    or other.shape[matched_dimension ^ 1] == 1
                ):
                    return (self, other, self._ndim, {"broadcast": True, "axis": 1})
                else:
                    raise ValueError(
                        f"operands could not be broadcast together with shapes {self.shape} {other.shape}"
                    )
            else:
                return (self, other, self._ndim, {"broadcast": False})

    def __add__(
        self,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        if is_scalar(x2):
            return array(_query_compiler=self._query_compiler.add(x2), _ndim=self._ndim)
        caller, callee, new_ndim, kwargs = self._binary_op(x2)
        result = caller._query_compiler.add(callee._query_compiler, **kwargs)
        return array(_query_compiler=result, _ndim=new_ndim)

    def __radd__(
        self,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        return self._add(x2, out, where, casting, order, dtype, subok)

    def divide(
        self,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        if is_scalar(x2):
            return array(
                _query_compiler=self._query_compiler.truediv(x2), _ndim=self._ndim
            )
        caller, callee, new_ndim, kwargs = self._binary_op(x2)
        if caller != self:
            # In this case, we are doing an operation that looks like this 1D_object/2D_object.
            # For Modin to broadcast directly, we have to swap it so that the operation is actually
            # 2D_object.rtruediv(1D_object).
            result = caller._query_compiler.rtruediv(callee._query_compiler, **kwargs)
        else:
            result = caller._query_compiler.truediv(callee._query_compiler, **kwargs)
        return array(_query_compiler=result, _ndim=new_ndim)

    __truediv__ = divide

    def __rtruediv__(
        self,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        if is_scalar(x2):
            return array(
                _query_compiler=self._query_compiler.rtruediv(x2), _ndim=self._ndim
            )
        caller, callee, new_ndim, kwargs = self._binary_op(x2)
        if caller != self:
            result = caller._query_compiler.truediv(callee._query_compiler, **kwargs)
        else:
            result = caller._query_compiler.rtruediv(callee._query_compiler, **kwargs)
        return array(_query_compiler=result, _ndim=new_ndim)

    def floor_divide(
        self,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        if is_scalar(x2):
            result = self._query_compiler.floordiv(x2)
            if x2 == 0:
                # NumPy's floor_divide by 0 works differently from pandas', so we need to fix
                # the output.
                result = result.replace(numpy.inf, 0).replace(numpy.NINF, 0)
            return array(_query_compiler=result, _ndim=self._ndim)
        caller, callee, new_ndim, kwargs = self._binary_op(x2)
        if caller != self:
            # Modin does not correctly support broadcasting when the caller of the function is
            # a Series (1D), and the operand is a Dataframe (2D). We cannot workaround this using
            # commutativity, and `rfloordiv` also works incorrectly. GH#5529
            raise NotImplementedError(
                "Using floor_divide with broadcast is not currently available in Modin."
            )
        result = caller._query_compiler.floordiv(callee._query_compiler, **kwargs)
        if any(callee._query_compiler.eq(0).any()):
            # NumPy's floor_divide by 0 works differently from pandas', so we need to fix
            # the output.
            result = result.replace(numpy.inf, 0).replace(numpy.NINF, 0)
        return array(_query_compiler=result, _ndim=new_ndim)

    __floordiv__ = floor_divide

    def power(
        self,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        if is_scalar(x2):
            return array(_query_compiler=self._query_compiler.pow(x2), _ndim=self._ndim)
        caller, callee, new_ndim, kwargs = self._binary_op(x2)
        if caller != self:
            # Modin does not correctly support broadcasting when the caller of the function is
            # a Series (1D), and the operand is a Dataframe (2D). We cannot workaround this using
            # commutativity, and `rpow` also works incorrectly. GH#5529
            raise NotImplementedError(
                "Using power with broadcast is not currently available in Modin."
            )
        result = caller._query_compiler.pow(callee._query_compiler, **kwargs)
        return array(_query_compiler=result, _ndim=new_ndim)

    __pow__ = power

    def prod(self, axis=None, out=None, keepdims=None, where=None):
        if axis is None:
            result = self._query_compiler.prod(axis=0).prod(axis=1)
            return result.to_numpy()[0, 0]
        else:
            result = self._query_compiler.prod(axis=axis)
            if self._ndim == 1:
                return result.to_numpy()[0, 0]
            return array(_query_compiler=result, _ndim=1)

    def multiply(
        self,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        if is_scalar(x2):
            return array(_query_compiler=self._query_compiler.mul(x2), _ndim=self._ndim)
        caller, callee, new_ndim, kwargs = self._binary_op(x2)
        result = caller._query_compiler.mul(callee._query_compiler, **kwargs)
        return array(_query_compiler=result, _ndim=new_ndim)

    __mul__ = multiply

    def __rmul__(
        self,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        return self._multiply(x2, out, where, casting, order, dtype, subok)

    def remainder(
        self,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        if is_scalar(x2):
            result = array(
                _query_compiler=self._query_compiler.mod(x2), _ndim=self._ndim
            )
            if x2 == 0:
                # NumPy's remainder by 0 works differently from pandas', so we need to fix
                # the output.
                result._query_compiler = result._query_compiler.replace(numpy.NaN, 0)
            return result
        caller, callee, new_ndim, kwargs = self._binary_op(x2)
        if caller != self:
            # Modin does not correctly support broadcasting when the caller of the function is
            # a Series (1D), and the operand is a Dataframe (2D). We cannot workaround this using
            # commutativity, and `rmod` also works incorrectly. GH#5529
            raise NotImplementedError(
                "Using remainder with broadcast is not currently available in Modin."
            )
        result = caller._query_compiler.mod(callee._query_compiler, **kwargs)
        if any(callee._query_compiler.eq(0).any()):
            # NumPy's floor_divide by 0 works differently from pandas', so we need to fix
            # the output.
            result = result.replace(numpy.NaN, 0)
        return array(_query_compiler=result, _ndim=new_ndim)

    __mod__ = remainder

    def subtract(
        self,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        if is_scalar(x2):
            return array(_query_compiler=self._query_compiler.sub(x2), _ndim=self._ndim)
        caller, callee, new_ndim, kwargs = self._binary_op(x2)
        if caller != self:
            # In this case, we are doing an operation that looks like this 1D_object - 2D_object.
            # For Modin to broadcast directly, we have to swap it so that the operation is actually
            # 2D_object.rsub(1D_object).
            result = caller._query_compiler.rsub(callee._query_compiler, **kwargs)
        else:
            result = caller._query_compiler.sub(callee._query_compiler, **kwargs)
        return array(_query_compiler=result, _ndim=new_ndim)

    __sub__ = subtract

    def __rsub__(
        self,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        if is_scalar(x2):
            return array(
                _query_compiler=self._query_compiler.rsub(x2), _ndim=self._ndim
            )
        caller, callee, new_ndim, kwargs = self._binary_op(x2)
        if caller != self:
            # In this case, we are doing an operation that looks like this 1D_object - 2D_object.
            # For Modin to broadcast directly, we have to swap it so that the operation is actually
            # 2D_object.sub(1D_object).
            result = caller._query_compiler.sub(callee._query_compiler, **kwargs)
        else:
            result = caller._query_compiler.rsub(callee._query_compiler, **kwargs)
        return array(_query_compiler=result, _ndim=new_ndim)

    def sum(
        self, axis=None, dtype=None, out=None, keepdims=None, initial=None, where=None
    ):
        result = self._query_compiler.sum(axis=axis)
        new_ndim = self._ndim - 1
        if axis is None or new_ndim == 0:
            return result.to_numpy()[0, 0]
        if dtype is not None:
            result = result.astype(dtype)
        return array(_query_compiler=result, _ndim=new_ndim)

    def flatten(self, order="C"):
        qcs = [
            self._query_compiler.getitem_row_array([index_val]).reset_index(drop=True)
            for index_val in self._query_compiler.index[1:]
        ]
        new_query_compiler = (
            self._query_compiler.getitem_row_array([self._query_compiler.index[0]])
            .reset_index(drop=True)
            .concat(1, qcs, ignore_index=True)
        )
        new_query_compiler.columns = range(len(new_query_compiler.columns))
        new_ndim = 1
        return array(_query_compiler=new_query_compiler, _ndim=new_ndim)

    def _get_shape(self):
        if self._ndim == 1:
            return (len(self._query_compiler.index),)
        return (len(self._query_compiler.index), len(self._query_compiler.columns))

    def _set_shape(self, new_shape):
        if not (isinstance(new_shape, int)) and not isinstance(new_shape, tuple):
            raise TypeError(
                f"expected a sequence of integers or a single integer, got '{new_shape}'"
            )
        elif isinstance(new_shape, tuple):
            for dim in new_shape:
                if not isinstance(dim, int):
                    raise TypeError(
                        f"'{type(dim)}' object cannot be interpreted as an integer"
                    )

        new_dimensions = new_shape if isinstance(new_shape, int) else prod(new_shape)
        if new_dimensions != prod(self._get_shape()):
            raise ValueError(
                f"cannot reshape array of size {prod(self._get_shape())} into {new_shape if isinstance(new_shape, tuple) else (new_shape,)}"
            )
        if isinstance(new_shape, int):
            self._query_compiler = self.flatten()._query_compiler
            self._ndim = 1
        else:
            raise NotImplementedError(
                "Reshaping from a 2D object to a 2D object is not currently supported!"
            )

    shape = property(_get_shape, _set_shape)

    def __repr__(self):
        return repr(self._to_numpy())

    def _to_numpy(self):
        arr = self._query_compiler.to_numpy()
        if self._ndim == 1:
            arr = arr.flatten()
        return arr
