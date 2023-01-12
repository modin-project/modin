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

import numpy

from pandas.core.dtypes.common import is_list_like
from modin.error_message import ErrorMessage


class array(object):
    def __init__(
        self,
        object=None,
        dtype=None,
        *,
        copy=True,
        order="K",
        subok=False,
        ndmin=0,
        like=None,
        _query_compiler=None,
        _ndim=None,
    ):
        if _query_compiler is not None:
            self._query_compiler = _query_compiler
            self._ndim = _ndim
        elif is_list_like(object) and not is_list_like(object[0]):
            import modin.pandas as pd

            qc = pd.Series(object)._query_compiler
            self._query_compiler = qc
            self._ndim = 1
        else:
            expected_kwargs = {
                "dtype": None,
                "copy": True,
                "order": "K",
                "subok": False,
                "ndmin": 0,
                "like": None,
            }
            rcvd_kwargs = {
                "dtype": dtype,
                "copy": copy,
                "order": order,
                "subok": subok,
                "ndmin": ndmin,
                "like": like,
            }
            for key, value in rcvd_kwargs.copy().items():
                if value == expected_kwargs[key]:
                    rcvd_kwargs.pop(key)
            arr = numpy.array(object, **rcvd_kwargs)
            self._ndim = len(arr.shape)
            if self._ndim > 2:
                ErrorMessage.not_implemented(
                    "NumPy arrays with dimensions higher than 2 are not yet supported."
                )
            import modin.pandas as pd

            self._query_compiler = pd.DataFrame(arr)._query_compiler

    def _absolute(
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

    __abs__ = _absolute

    def _binary_op(self, other):
        broadcast = self._ndim != other._ndim
        if broadcast:
            # In this case, we have a 1D object doing a binary op with a 2D object
            caller = self if self._ndim == 2 else other
            callee = other if self._ndim == 2 else self
            return (caller, callee, caller._ndim, {"broadcast": broadcast, "axis": 1})
        else:
            if self.shape != other.shape:
                # In this case, we either have two mismatched objects trying to do an operation
                # or a nested 1D object that must be broadcasted trying to do an operation.
                matched_dimension = None
                if self.shape[0] == other.shape[0]:
                    matched_dimension = 0
                elif self.shape[1] == other.shape[1]:
                    matched_dimension = 1
                if matched_dimension is not None:
                    if (
                        self.shape[matched_dimension ^ 1] == 1
                        or other.shape[matched_dimension ^ 1] == 1
                    ):
                        # caller = self if other.shape[matched_dimension ^ 1] == 1 else other
                        # callee = other if other.shape[matched_dimension ^ 1] == 1 else self
                        return (self, other, self._ndim, {"broadcast": True, "axis": 1})
                else:
                    raise ValueError(
                        f"operands could not be broadcast together with shapes {self.shape} {other.shape}"
                    )
            else:
                return (self, other, self._ndim, {"broadcast": False})

    def _add(
        self,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        caller, callee, new_ndim, kwargs = self._binary_op(x2)
        result = caller._query_compiler.add(callee._query_compiler, **kwargs)
        return array(_query_compiler=result, _ndim=new_ndim)

    __add__ = _add

    def _divide(
        self,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        caller, callee, new_ndim, kwargs = self._binary_op(x2)
        result = caller._query_compiler.truediv(callee._query_compiler, **kwargs)
        if caller != self:
            result = result.rtruediv(1)
        return array(_query_compiler=result, _ndim=new_ndim)

    __truediv__ = _divide

    def _float_power(
        self,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        pass

    def _floor_divide(
        self,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        caller, callee, new_ndim, kwargs = self._binary_op(x2)
        if caller != self:
            # No workaround possible until broadcasting fixed. GH#5529.
            pass
        result = caller._query_compiler.floordiv(callee._query_compiler, **kwargs)
        if any(callee._query_compiler.eq(0).to_pandas()):
            result = result.replace(numpy.inf, 0)
        return array(_query_compiler=result, _ndim=new_ndim)

    __floordiv__ = _floor_divide

    def _power(
        self,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        caller, callee, new_ndim, kwargs = self._binary_op(x2)
        if caller != self:
            # No workaround possible until broadcasting fixed. GH#5529.
            pass
        result = caller._query_compiler.pow(callee._query_compiler, **kwargs)
        return array(_query_compiler=result, _ndim=new_ndim)

    __pow__ = _power

    def _prod(self, axis=None, out=None, keepdims=None, where=None):
        if axis is None:
            result = self._query_compiler.prod(axis=0).prod(axis=1)
            return array(_query_compiler=result)
        else:
            result = self._query_compiler.prod(axis=axis)
            return array(_query_compiler=result)

    def _multiply(
        self,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        caller, callee, new_ndim, kwargs = self._binary_op(x2)
        result = caller._query_compiler.mul(callee._query_compiler, **kwargs)
        return array(_query_compiler=result, _ndim=new_ndim)

    __mul__ = _multiply

    def _remainder(
        self,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        caller, callee, new_ndim, kwargs = self._binary_op(x2)
        if caller != self:
            # No workaround possible until broadcasting fixed. GH#5529.
            pass
        result = caller._query_compiler.mod(callee._query_compiler, **kwargs)
        if any(callee._query_compiler.eq(0).to_pandas()):
            result = result.replace(numpy.NaN, 0)
        return array(_query_compiler=result, _ndim=new_ndim)

    __mod__ = _remainder

    def _subtract(
        self,
        x2,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        caller, callee, new_ndim, kwargs = self._binary_op(x2)
        result = caller._query_compiler.sub(callee._query_compiler, **kwargs)
        if caller != self:
            result = result.rsub(0)
        return array(_query_compiler=result, _ndim=new_ndim)

    __sub__ = _subtract

    def _sum(
        self, axis=None, dtype=None, out=None, keepdims=None, initial=None, where=None
    ):
        result = self._query_compiler.sum(axis=axis)
        if dtype is not None:
            result = result.astype(dtype)
        if out is not None:
            out._query_compiler = result
            return
        if axis is None:
            return
        else:
            new_ndim = self._ndim - 1
        return array(_query_compiler=result, _ndim=new_ndim)

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
        from math import prod

        new_dimensions = new_shape if isinstance(new_shape, int) else prod(new_shape)
        if new_dimensions != prod(self._get_shape()):
            raise ValueError(
                f"cannot reshape array of size {prod(self._get_shape)} into {new_shape if isinstance(new_shape, tuple) else (new_shape,)}"
            )
        if isinstance(new_shape, int):
            qcs = []
            for index_val in self._query_compiler.index[1:]:
                qcs.append(
                    self._query_compiler.getitem_row_array([index_val]).reset_index(
                        drop=True
                    )
                )
            self._query_compiler = (
                self._query_compiler.getitem_row_array([self._query_compiler.index[0]])
                .reset_index(drop=True)
                .concat(1, qcs, ignore_index=True)
            )
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
