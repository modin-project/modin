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
from pandas.core.dtypes.common import is_list_like, is_numeric_dtype, is_bool_dtype
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


def try_convert_from_interoperable_type(obj):
    if isinstance(obj, _INTEROPERABLE_TYPES):
        new_qc = obj._query_compiler.reset_index(drop=True)
        new_qc.columns = range(len(new_qc.columns))
        obj = array(
            _query_compiler=new_qc,
            _ndim=2 if isinstance(obj, pd.DataFrame) else 1,
        )
    return obj


def check_kwargs(order="C", subok=True, keepdims=None, casting="same_kind", where=True):
    if order not in ["K", "C"]:
        ErrorMessage.single_warning(
            "Array order besides 'C' is not currently supported in Modin. Defaulting to 'C' order."
        )
    if not subok:
        ErrorMessage.single_warning(
            "Subclassing types is not currently supported in Modin. Defaulting to the same base dtype."
        )
    if keepdims:
        ErrorMessage.single_warning(
            "Modin does not yet support broadcasting between nested 1D arrays and 2D arrays."
        )
    if casting != "same_kind":
        ErrorMessage.single_warning(
            "Modin does not yet support the `casting` argument."
        )
    if not (
        is_scalar(where) or (isinstance(where, array) and is_bool_dtype(where.dtype))
    ):
        if not isinstance(where, array):
            raise NotImplementedError(
                f"Modin only supports scalar or modin.numpy.array `where` parameter, not `where` parameter of type {type(where)}"
            )
        raise TypeError(
            f"Cannot cast array data from {where.dtype} to dtype('bool') according to the rule 'safe'"
        )


def check_how_broadcast_to_output(arr_in: "array", arr_out: "array"):
    if not isinstance(arr_out, array):
        raise TypeError("return arrays must be of modin.numpy.array type.")
    if arr_out._ndim == arr_in._ndim and arr_out.shape != arr_in.shape:
        raise ValueError(
            f"non-broadcastable output operand with shape {arr_out.shape} doesn't match the broadcast shape {arr_in.shape}"
        )
    elif arr_out._ndim == arr_in._ndim:
        return "broadcastable"
    elif arr_out._ndim == 1:
        if prod(arr_in.shape) == arr_out.shape[0]:
            return "flatten"
        else:
            raise ValueError(
                f"non-broadcastable output operand with shape {arr_out.shape} doesn't match the broadcast shape {arr_in.shape}"
            )
    elif arr_in._ndim == 1:
        if prod(arr_out.shape) == arr_in.shape[0]:
            return "reshape"
        else:
            raise ValueError(
                f"non-broadcastable output operand with shape {arr_out.shape} doesn't match the broadcast shape {arr_in.shape}"
            )


def fix_dtypes_and_determine_return(
    query_compiler_in, _ndim, dtype=None, out=None, where=True
):
    if dtype is not None:
        query_compiler_in = query_compiler_in.astype(
            {col_name: dtype for col_name in query_compiler_in.columns}
        )
    result = array(_query_compiler=query_compiler_in, _ndim=_ndim)
    if out is not None:
        out = try_convert_from_interoperable_type(out)
        broadcast_method = check_how_broadcast_to_output(result, out)
        result._query_compiler = result._query_compiler.astype(
            {col_name: out.dtype for col_name in result._query_compiler.columns}
        )
        if broadcast_method == "flatten":
            result = result.flatten()
        elif broadcast_method != "broadcastable":
            # TODO(RehanSD): Replace this when reshape is implemented.
            raise NotImplementedError("Reshape is currently not supported in Modin.")
        if isinstance(where, array):
            out._query_compiler = where.where(result, out)._query_compiler
        elif where:
            out._query_compiler = result._query_compiler
        return out
    if isinstance(where, array) and out is None:
        from array_creation import zeros_like

        out = zeros_like(result).astype(dtype if dtype is not None else result.dtype)
        out._query_compiler = where.where(result, out)._query_compiler
        return out
    elif not where:
        from .array_creation import zeros_like

        return zeros_like(result)
    return result


def find_common_dtype(dtypes):
    if len(dtypes) == 1:
        return dtypes[0]
    elif len(dtypes) == 2:
        return numpy.promote_types(*dtypes)
    midpoint = len(dtypes) // 2
    return numpy.promote_types(
        find_common_dtype(dtypes[:midpoint]), find_common_dtype(dtypes[midpoint:])
    )


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
            new_dtype = find_common_dtype(
                numpy.unique(self._query_compiler.dtypes.values)
            )
        elif is_list_like(object) and not is_list_like(object[0]):
            series = pd.Series(object)
            self._query_compiler = series._query_compiler
            self._ndim = 1
            new_dtype = self._query_compiler.dtypes.values[0]
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
            assert arr.ndim in (
                1,
                2,
            ), "Modin.NumPy currently only supports 1D and 2D objects."
            self._ndim = len(arr.shape)
            if self._ndim > 2:
                ErrorMessage.not_implemented(
                    "NumPy arrays with dimensions higher than 2 are not yet supported."
                )

            self._query_compiler = pd.DataFrame(arr)._query_compiler
            new_dtype = arr.dtype
        # These two lines are necessary so that our query compiler does not keep track of indices
        # and try to map like indices to like indices. (e.g. if we multiply two arrays that used
        # to be dataframes, and the dataframes had the same column names but ordered differently
        # we want to do a simple broadcast where we only consider position, as numpy would, rather
        # than pair columns with the same name and multiply them.)
        self._query_compiler = self._query_compiler.reset_index(drop=True)
        self._query_compiler.columns = range(len(self._query_compiler.columns))
        new_dtype = new_dtype if dtype is None else dtype
        self._query_compiler = self._query_compiler.astype(
            {col_name: new_dtype for col_name in self._query_compiler.columns}
        )

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        ufunc_name = ufunc.__name__
        supported_array_layer = hasattr(self, ufunc_name) or hasattr(
            self, f"__{ufunc_name}__"
        )
        if supported_array_layer:
            args = []
            for input in inputs:
                input = try_convert_from_interoperable_type(input)
                if not (isinstance(input, array) or is_scalar(input)):
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
                    if param.default == param.empty
                ]
            )
            if len_expected_arguments == (len(args) - 1) and method == "__call__":
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
                output = self._to_numpy().__array_ufunc__(
                    ufunc, method, *args, **kwargs
                )
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
                out_ndim = max(
                    [len(inp.shape) for inp in inputs if hasattr(inp, "shape")]
                )
        elif method == "reduce":
            if len(inputs) == 1:
                new_ufunc = Reduce.register(ufunc, axis=kwargs.get("axis", None))
            if kwargs.get("axis", None) is None:
                out_ndim = 0
            else:
                out_ndim = len(inputs[0].shape) - 1
        elif method == "accumulate":
            if len(inputs) == 1:
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
            output = self._to_numpy().__array_ufunc__(ufunc, method, *args, **kwargs)
            if is_scalar(output):
                return output
            return array(output)
        args = []
        for input in inputs:
            input = try_convert_from_interoperable_type(input)
            if not (isinstance(input, array) or is_scalar(input)):
                input = array(input)
            args += [
                input._query_compiler if hasattr(input, "_query_compiler") else input
            ]
        return array(_query_compiler=new_ufunc(*args, **kwargs), _ndim=out_ndim)

    def __array_function__(self, func, types, args, kwargs):
        from . import array_creation as creation, array_shaping as shaping, math

        func_name = func.__name__
        modin_func = None
        if hasattr(math, func_name):
            modin_func = getattr(math, func_name)
        elif hasattr(shaping, func_name):
            modin_func = getattr(shaping, func_name)
        elif hasattr(creation, func_name):
            modin_func = getattr(creation, func_name)
        if modin_func is None:
            return NotImplemented
        return modin_func(*args, **kwargs)

    def where(self, x=None, y=None):
        if x is None and y is None:
            ErrorMessage.single_warning(
                "np.where method with only condition specified is not yet supported in Modin. Defaulting to NumPy."
            )
            condition = self._to_numpy()
            return array(numpy.where(condition))
        x, y = try_convert_from_interoperable_type(
            x
        ), try_convert_from_interoperable_type(y)
        if not (
            (isinstance(x, array) or is_scalar(x))
            and (isinstance(y, array) or is_scalar(y))
        ):
            raise ValueError(
                "np.where requires x and y to either be np.arrays or scalars."
            )
        if is_scalar(x) and is_scalar(y):
            ErrorMessage.single_warning(
                "np.where not supported when both x and y are scalars. Defaulting to NumPy."
            )
            return array(numpy.where(self._to_numpy(), x, y))
        if is_scalar(x) and not is_scalar(y):
            if self._ndim < y._ndim:
                if not self.shape[0] == y.shape[1]:
                    raise ValueError(
                        f"operands could not be broadcast together with shapes {self.shape} {y.shape}"
                    )
                ErrorMessage.single_warning(
                    "np.where method where condition must be broadcast is not yet available in Modin. Defaulting to NumPy."
                )
                return array(numpy.where(self._to_numpy(), x, y._to_numpy()))
            elif self._ndim == y._ndim:
                if not self.shape == y.shape:
                    raise ValueError(
                        f"operands could not be broadcast together with shapes {self.shape} {y.shape}"
                    )
                return array(
                    _query_compiler=y._query_compiler.where((~self)._query_compiler, x),
                    _ndim=y._ndim,
                )
            else:
                ErrorMessage.single_warning(
                    "np.where method with broadcast is not yet available in Modin. Defaulting to NumPy."
                )
                return numpy.where(self._to_numpy(), x, y._to_numpy())
        if not is_scalar(x) and is_scalar(y):
            if self._ndim < x._ndim:
                if not self.shape[0] == x.shape[1]:
                    raise ValueError(
                        f"operands could not be broadcast together with shapes {self.shape} {x.shape}"
                    )
                ErrorMessage.single_warning(
                    "np.where method where condition must be broadcast is not yet available in Modin. Defaulting to NumPy."
                )
                return array(numpy.where(self._to_numpy(), x._to_numpy(), y))
            elif self._ndim == x._ndim:
                if not self.shape == x.shape:
                    raise ValueError(
                        f"operands could not be broadcast together with shapes {self.shape} {x.shape}"
                    )
                return array(
                    _query_compiler=x._query_compiler.where(self._query_compiler, y),
                    _ndim=x._ndim,
                )
            else:
                ErrorMessage.single_warning(
                    "np.where method with broadcast is not yet available in Modin. Defaulting to NumPy."
                )
                return array(numpy.where(self._to_numpy(), x._to_numpy(), y))
        if not (x.shape == y.shape and y.shape == self.shape):
            ErrorMessage.single_warning(
                "np.where method with broadcast is not yet available in Modin. Defaulting to NumPy."
            )
            return array(numpy.where(self._to_numpy(), x._to_numpy(), y._to_numpy()))
        return array(
            _query_compiler=x._query_compiler.where(
                self._query_compiler, y._query_compiler
            ),
            _ndim=self._ndim,
        )

    def max(
        self, axis=None, dtype=None, out=None, keepdims=None, initial=None, where=True
    ):
        check_kwargs(keepdims=keepdims, where=where)
        if initial is None and where is not True:
            raise ValueError(
                "reduction operation 'maximum' does not have an identity, so to use a where mask one has to specify 'initial'"
            )
        if self._ndim == 1:
            if axis == 1:
                raise numpy.AxisError(1, 1)
            result = self._query_compiler.max(axis=0)
            if keepdims:
                if initial is not None and result.lt(initial):
                    result = pd.Series([initial])._query_compiler
                if initial is not None:
                    if out is not None:
                        out._query_compiler = (
                            numpy.ones_like(out) * initial
                        )._query_compiler
                    else:
                        out = array([initial]).astype(self.dtype)
                if out is not None and out.shape != (1,):
                    raise ValueError(
                        f"operand was set up as a reduction along axis 0, but the length of the axis is {out.shape[0]} (it has to be 1)"
                    )
                return fix_dtypes_and_determine_return(result, 1, dtype, out, where)
            if initial is not None:
                result = max(result.to_numpy()[0, 0], initial)
            else:
                result = result.to_numpy()[0, 0]
            return result if where else initial
        if axis is None:
            result = self.flatten().max(
                axis=axis,
                dtype=dtype,
                out=out,
                keepdims=None,
                initial=initial,
                where=where,
            )
            if keepdims:
                if out is not None and out.shape != (1, 1):
                    raise ValueError(
                        f"operand was set up as a reduction along axis 0, but the length of the axis is {out.shape[0]} (it has to be 1)"
                    )
                return fix_dtypes_and_determine_return(
                    array(numpy.array([[result]]))._query_compiler, 2, dtype, out, where
                )
            return result
        result = self._query_compiler.max(axis=axis)
        new_ndim = self._ndim - 1 if not keepdims else self._ndim
        if new_ndim == 0:
            if initial is not None:
                result = max(result.to_numpy()[0, 0], initial)
            else:
                result = result.to_numpy()[0, 0]
            return result if where else initial
        if not keepdims and axis != 1:
            result = result.transpose()
        if initial is not None:
            if out is not None:
                out._query_compiler = (numpy.ones_like(out) * initial)._query_compiler
            else:
                out = (
                    numpy.ones_like(array(_query_compiler=result, _ndim=new_ndim))
                    * initial
                ).astype(self.dtype)
        intermediate = fix_dtypes_and_determine_return(
            result, new_ndim, dtype, out, where
        )
        if initial is not None:
            intermediate._query_compiler = (
                (intermediate > initial).where(intermediate, initial)._query_compiler
            )
        return intermediate

    def min(
        self, axis=None, dtype=None, out=None, keepdims=None, initial=None, where=True
    ):
        check_kwargs(keepdims=keepdims, where=where)
        if initial is None and where is not True:
            raise ValueError(
                "reduction operation 'minimum' does not have an identity, so to use a where mask one has to specify 'initial'"
            )
        if self._ndim == 1:
            if axis == 1:
                raise numpy.AxisError(1, 1)
            result = self._query_compiler.min(axis=0)
            if keepdims:
                if initial is not None and result.lt(initial):
                    result = pd.Series([initial])._query_compiler
                if initial is not None:
                    if out is not None:
                        out._query_compiler = (
                            numpy.ones_like(out) * initial
                        )._query_compiler
                    else:
                        out = array([initial]).astype(self.dtype)
                if out is not None and out.shape != (1,):
                    raise ValueError(
                        f"operand was set up as a reduction along axis 0, but the length of the axis is {out.shape[0]} (it has to be 1)"
                    )
                return fix_dtypes_and_determine_return(result, 1, dtype, out, where)
            if initial is not None:
                result = min(result.to_numpy()[0, 0], initial)
            else:
                result = result.to_numpy()[0, 0]
            return result if where else initial
        if axis is None:
            result = self.flatten().min(
                axis=axis,
                dtype=dtype,
                out=out,
                keepdims=None,
                initial=initial,
                where=where,
            )
            if keepdims:
                if out is not None and out.shape != (1, 1):
                    raise ValueError(
                        f"operand was set up as a reduction along axis 0, but the length of the axis is {out.shape[0]} (it has to be 1)"
                    )
                return fix_dtypes_and_determine_return(
                    array(numpy.array([[result]]))._query_compiler, 2, dtype, out, where
                )
            return result
        result = self._query_compiler.min(axis=axis)
        new_ndim = self._ndim - 1 if not keepdims else self._ndim
        if new_ndim == 0:
            if initial is not None:
                result = min(result.to_numpy()[0, 0], initial)
            else:
                result = result.to_numpy()[0, 0]
            return result if where else initial
        if not keepdims and axis != 1:
            result = result.transpose()
        if initial is not None:
            if out is not None:
                out._query_compiler = (numpy.ones_like(out) * initial)._query_compiler
            else:
                out = (
                    numpy.ones_like(array(_query_compiler=result, _ndim=new_ndim))
                    * initial
                ).astype(self.dtype)
        intermediate = fix_dtypes_and_determine_return(
            result, new_ndim, dtype, out, where
        )
        if initial is not None:
            intermediate._query_compiler = (
                (intermediate < initial).where(intermediate, initial)._query_compiler
            )
        return intermediate

    def __abs__(
        self,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        out_dtype = (
            dtype
            if dtype is not None
            else (out.dtype if out is not None else self.dtype)
        )
        check_kwargs(order=order, casting=casting, subok=subok, where=where)
        result = self._query_compiler.astype(
            {col_name: out_dtype for col_name in self._query_compiler.columns}
        ).abs()
        if dtype is not None:
            result = result.astype({col_name: dtype for col_name in result.columns})
        if out is not None:
            out = try_convert_from_interoperable_type(out)
            broadcast_method = check_how_broadcast_to_output(self, out)
            if broadcast_method == "broadcastable":
                out._query_compiler = result
                return out
            elif broadcast_method == "flatten":
                out._query_compiler = (
                    array(_query_compiler=result, _ndim=self._ndim)
                    .flatten()
                    ._query_compiler
                )
            else:
                # TODO(RehanSD): Replace this when reshape is implemented.
                raise NotImplementedError(
                    "Reshape is currently not supported in Modin."
                )
        return array(_query_compiler=result, _ndim=self._ndim)

    absolute = __abs__

    def __invert__(self):
        """
        Apply bitwise inverse to each element of the `BasePandasDataset`.

        Returns
        -------
        BasePandasDataset
            New BasePandasDataset containing bitwise inverse to each value.
        """
        if not is_numeric_dtype(self.dtype):
            raise TypeError(f"bad operand type for unary ~: '{self.dtype}'")
        return array(_query_compiler=self._query_compiler.invert(), _ndim=self._ndim)

    def _binary_op(self, other):
        other = try_convert_from_interoperable_type(other)
        if not isinstance(other, array):
            raise TypeError(
                f"Unsupported operand type(s): '{type(self)}' and '{type(other)}'"
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

    def __ge__(self, x2):
        if is_scalar(x2):
            return array(_query_compiler=self._query_compiler.ge(x2), _ndim=self._ndim)
        caller, callee, new_ndim, kwargs = self._binary_op(x2)
        if caller._query_compiler != self._query_compiler:
            # In this case, we are doing an operation that looks like this 1D_object >= 2D_object.
            # For Modin to broadcast directly, we have to swap it so that the operation is actually
            # 2D_object <= 1D_object.
            result = caller._query_compiler.le(callee._query_compiler, **kwargs)
        else:
            result = caller._query_compiler.ge(callee._query_compiler, **kwargs)
        return array(_query_compiler=result, _ndim=new_ndim)

    def __gt__(self, x2):
        if is_scalar(x2):
            return array(_query_compiler=self._query_compiler.gt(x2), _ndim=self._ndim)
        caller, callee, new_ndim, kwargs = self._binary_op(x2)
        if caller._query_compiler != self._query_compiler:
            # In this case, we are doing an operation that looks like this 1D_object > 2D_object.
            # For Modin to broadcast directly, we hiave to swap it so that the operation is actually
            # 2D_object < 1D_object.
            result = caller._query_compiler.lt(callee._query_compiler, **kwargs)
        else:
            result = caller._query_compiler.gt(callee._query_compiler, **kwargs)
        return array(_query_compiler=result, _ndim=new_ndim)

    def __le__(self, x2):
        if is_scalar(x2):
            return array(_query_compiler=self._query_compiler.le(x2), _ndim=self._ndim)
        caller, callee, new_ndim, kwargs = self._binary_op(x2)
        if caller._query_compiler != self._query_compiler:
            # In this case, we are doing an operation that looks like this 1D_object <= 2D_object.
            # For Modin to broadcast directly, we have to swap it so that the operation is actually
            # 2D_object >= 1D_object.
            result = caller._query_compiler.ge(callee._query_compiler, **kwargs)
        else:
            result = caller._query_compiler.le(callee._query_compiler, **kwargs)
        return array(_query_compiler=result, _ndim=new_ndim)

    def __lt__(self, x2):
        if is_scalar(x2):
            return array(_query_compiler=self._query_compiler.lt(x2), _ndim=self._ndim)
        caller, callee, new_ndim, kwargs = self._binary_op(x2)
        if caller._query_compiler != self._query_compiler:
            # In this case, we are doing an operation that looks like this 1D_object < 2D_object.
            # For Modin to broadcast directly, we have to swap it so that the operation is actually
            # 2D_object > 1D_object.
            result = caller._query_compiler.gt(callee._query_compiler, **kwargs)
        else:
            result = caller._query_compiler.lt(callee._query_compiler, **kwargs)
        return array(_query_compiler=result, _ndim=new_ndim)

    def __eq__(self, x2):
        if is_scalar(x2):
            return array(_query_compiler=self._query_compiler.eq(x2), _ndim=self._ndim)
        caller, callee, new_ndim, kwargs = self._binary_op(x2)
        result = caller._query_compiler.eq(callee._query_compiler, **kwargs)
        return array(_query_compiler=result, _ndim=new_ndim)

    def __ne__(self, x2):
        if is_scalar(x2):
            return array(_query_compiler=self._query_compiler.ne(x2), _ndim=self._ndim)
        caller, callee, new_ndim, kwargs = self._binary_op(x2)
        result = caller._query_compiler.ne(callee._query_compiler, **kwargs)
        return array(_query_compiler=result, _ndim=new_ndim)

    def mean(self, axis=None, dtype=None, out=None, keepdims=None, *, where=True):
        out_dtype = (
            dtype
            if dtype is not None
            else (out.dtype if out is not None else self.dtype)
        )
        check_kwargs(keepdims=keepdims, where=where)
        if self._ndim == 1:
            if axis == 1:
                raise numpy.AxisError(1, 1)
            result = self._query_compiler.astype(
                {col_name: out_dtype for col_name in self._query_compiler.columns}
            ).mean(axis=0)
            if keepdims:
                if out is not None and out.shape != (1,):
                    raise ValueError(
                        f"operand was set up as a reduction along axis 0, but the length of the axis is {out.shape[0]} (it has to be 1)"
                    )
                return fix_dtypes_and_determine_return(result, 1, dtype, out, where)
            return result.to_numpy()[0, 0] if where else numpy.nan
        if axis is None:
            result = (
                self.flatten()
                .astype(out_dtype)
                .mean(axis=axis, dtype=dtype, out=out, keepdims=None, where=where)
            )
            if keepdims:
                if out is not None and out.shape != (1, 1):
                    raise ValueError(
                        f"operand was set up as a reduction along axis 0, but the length of the axis is {out.shape[0]} (it has to be 1)"
                    )
                return fix_dtypes_and_determine_return(
                    array(numpy.array([[result]])).astype(out_dtype)._query_compiler,
                    2,
                    dtype,
                    out,
                    where,
                )
            return result
        result = self._query_compiler.astype(
            {col_name: out_dtype for col_name in self._query_compiler.columns}
        ).mean(axis=axis)
        new_ndim = self._ndim - 1 if not keepdims else self._ndim
        if new_ndim == 0:
            return result.to_numpy()[0, 0] if where else numpy.nan
        if not keepdims and axis != 1:
            result = result.transpose()
        if out is not None:
            out._query_compiler = (out * numpy.nan).astype(out_dtype)._query_compiler
        return fix_dtypes_and_determine_return(result, new_ndim, dtype, out, where)

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
        out_dtype = (
            dtype
            if dtype is not None
            else (out.dtype if out is not None else self.dtype)
        )
        check_kwargs(order=order, subok=subok, casting=casting, where=where)
        if is_scalar(x2):
            result = self._query_compiler.astype(
                {col_name: out_dtype for col_name in self._query_compiler.columns}
            ).add(x2)
            return fix_dtypes_and_determine_return(
                result, self._ndim, dtype, out, where
            )
        caller, callee, new_ndim, kwargs = self._binary_op(x2)
        caller_qc = caller._query_compiler.astype(
            {col_name: out_dtype for col_name in caller._query_compiler.columns}
        )
        callee_qc = callee._query_compiler.astype(
            {col_name: out_dtype for col_name in callee._query_compiler.columns}
        )
        result = caller_qc.add(callee_qc, **kwargs)
        return fix_dtypes_and_determine_return(result, new_ndim, dtype, out, where)

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
        return self.__add__(x2, out, where, casting, order, dtype, subok)

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
        out_dtype = (
            dtype
            if dtype is not None
            else (out.dtype if out is not None else self.dtype)
        )
        check_kwargs(order=order, subok=subok, casting=casting, where=where)
        if is_scalar(x2):
            return fix_dtypes_and_determine_return(
                self._query_compiler.astype(
                    {col_name: out_dtype for col_name in self._query_compiler.columns}
                ).truediv(x2),
                self._ndim,
                dtype,
                out,
                where,
            )
        caller, callee, new_ndim, kwargs = self._binary_op(x2)
        caller_qc = caller._query_compiler.astype(
            {col_name: out_dtype for col_name in caller._query_compiler.columns}
        )
        callee_qc = callee._query_compiler.astype(
            {col_name: out_dtype for col_name in callee._query_compiler.columns}
        )
        if caller._query_compiler != self._query_compiler:
            # In this case, we are doing an operation that looks like this 1D_object/2D_object.
            # For Modin to broadcast directly, we have to swap it so that the operation is actually
            # 2D_object.rtruediv(1D_object).
            result = caller_qc.rtruediv(callee_qc, **kwargs)
        else:
            result = caller_qc.truediv(callee_qc, **kwargs)
        return fix_dtypes_and_determine_return(result, new_ndim, dtype, out, where)

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
        out_dtype = (
            dtype
            if dtype is not None
            else (out.dtype if out is not None else self.dtype)
        )
        check_kwargs(order=order, subok=subok, casting=casting, where=where)
        if is_scalar(x2):
            return fix_dtypes_and_determine_return(
                self._query_compiler.astype(
                    {col_name: out_dtype for col_name in self._query_compiler.columns}
                ).rtruediv(x2),
                self._ndim,
                dtype,
                out,
                where,
            )
        caller, callee, new_ndim, kwargs = self._binary_op(x2)
        caller_qc = caller._query_compiler.astype(
            {col_name: out_dtype for col_name in caller._query_compiler.columns}
        )
        callee_qc = callee._query_compiler.astype(
            {col_name: out_dtype for col_name in callee._query_compiler.columns}
        )
        if caller._query_compiler != self._query_compiler:
            result = caller_qc.truediv(callee_qc, **kwargs)
        else:
            result = caller_qc.rtruediv(callee_qc, **kwargs)
        return fix_dtypes_and_determine_return(result, new_ndim, dtype, out, where)

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
        out_dtype = (
            dtype
            if dtype is not None
            else (out.dtype if out is not None else self.dtype)
        )
        check_kwargs(order=order, subok=subok, casting=casting, where=where)
        if is_scalar(x2):
            result = self._query_compiler.astype(
                {col_name: out_dtype for col_name in self._query_compiler.columns}
            ).floordiv(x2)
            if x2 == 0:
                # NumPy's floor_divide by 0 works differently from pandas', so we need to fix
                # the output.
                result = (
                    result.replace(numpy.inf, 0)
                    .replace(numpy.NINF, 0)
                    .replace(numpy.nan, 0)
                )
            return fix_dtypes_and_determine_return(
                result, self._ndim, dtype, out, where
            )
        caller, callee, new_ndim, kwargs = self._binary_op(x2)
        caller_qc = caller._query_compiler.astype(
            {col_name: out_dtype for col_name in caller._query_compiler.columns}
        )
        callee_qc = callee._query_compiler.astype(
            {col_name: out_dtype for col_name in callee._query_compiler.columns}
        )
        if caller._query_compiler != self._query_compiler:
            # Modin does not correctly support broadcasting when the caller of the function is
            # a Series (1D), and the operand is a Dataframe (2D). We cannot workaround this using
            # commutativity, and `rfloordiv` also works incorrectly. GH#5529
            raise NotImplementedError(
                "Using floor_divide with broadcast is not currently available in Modin."
            )
        result = caller_qc.floordiv(callee_qc, **kwargs)
        if callee._query_compiler.eq(0).any():
            # NumPy's floor_divide by 0 works differently from pandas', so we need to fix
            # the output.
            result = result.replace(numpy.inf, 0).replace(numpy.NINF, 0)
        result = result.replace(numpy.nan, 0)
        return fix_dtypes_and_determine_return(result, new_ndim, dtype, out, where)

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
        out_dtype = (
            dtype
            if dtype is not None
            else (out.dtype if out is not None else self.dtype)
        )
        check_kwargs(order=order, subok=subok, casting=casting, where=where)
        if is_scalar(x2):
            return fix_dtypes_and_determine_return(
                self._query_compiler.astype(
                    {col_name: out_dtype for col_name in self._query_compiler.columns}
                ).pow(x2),
                self._ndim,
                dtype,
                out,
                where,
            )
        caller, callee, new_ndim, kwargs = self._binary_op(x2)
        caller_qc = caller._query_compiler.astype(
            {col_name: out_dtype for col_name in caller._query_compiler.columns}
        )
        callee_qc = callee._query_compiler.astype(
            {col_name: out_dtype for col_name in callee._query_compiler.columns}
        )
        if caller._query_compiler != self._query_compiler:
            # Modin does not correctly support broadcasting when the caller of the function is
            # a Series (1D), and the operand is a Dataframe (2D). We cannot workaround this using
            # commutativity, and `rpow` also works incorrectly. GH#5529
            raise NotImplementedError(
                "Using power with broadcast is not currently available in Modin."
            )
        result = caller_qc.pow(callee_qc, **kwargs)
        return fix_dtypes_and_determine_return(result, new_ndim, dtype, out, where)

    __pow__ = power

    def prod(
        self, axis=None, dtype=None, out=None, keepdims=None, initial=None, where=True
    ):
        out_dtype = (
            dtype
            if dtype is not None
            else (out.dtype if out is not None else self.dtype)
        )
        initial = 1 if initial is None else initial
        check_kwargs(keepdims=keepdims, where=where)
        if self._ndim == 1:
            if axis == 1:
                raise numpy.AxisError(1, 1)
            result = self._query_compiler.astype(
                {col_name: out_dtype for col_name in self._query_compiler.columns}
            ).prod(axis=0)
            if initial is not None:
                result = result.mul(initial)
            if keepdims:
                if initial is not None:
                    if out is not None:
                        out._query_compiler = (
                            (numpy.ones_like(out) * initial)
                            .astype(out_dtype)
                            ._query_compiler
                        )
                    else:
                        out = array([initial]).astype(out_dtype)
                if out is not None and out.shape != (1,):
                    raise ValueError(
                        f"operand was set up as a reduction along axis 0, but the length of the axis is {out.shape[0]} (it has to be 1)"
                    )
                return fix_dtypes_and_determine_return(result, 1, dtype, out, where)
            return result.to_numpy()[0, 0] if where else initial
        if axis is None:
            result = (
                self.flatten()
                .astype(out_dtype)
                .prod(
                    axis=axis,
                    dtype=dtype,
                    out=out,
                    keepdims=None,
                    initial=initial,
                    where=where,
                )
            )
            if keepdims:
                if out is not None and out.shape != (1, 1):
                    raise ValueError(
                        f"operand was set up as a reduction along axis 0, but the length of the axis is {out.shape[0]} (it has to be 1)"
                    )
                return fix_dtypes_and_determine_return(
                    array(numpy.array([[result]])).astype(out_dtype)._query_compiler,
                    2,
                    dtype,
                    out,
                    where,
                )
            return result
        result = self._query_compiler.astype(
            {col_name: out_dtype for col_name in self._query_compiler.columns}
        ).prod(axis=axis)
        if initial is not None:
            result = result.mul(initial)
        new_ndim = self._ndim - 1 if not keepdims else self._ndim
        if new_ndim == 0:
            return result.to_numpy()[0, 0] if where else initial
        if not keepdims and axis != 1:
            result = result.transpose()
        if initial is not None:
            if out is not None:
                out._query_compiler = (
                    (numpy.ones_like(out) * initial).astype(out_dtype)._query_compiler
                )
            else:
                out = (
                    numpy.ones_like(array(_query_compiler=result, _ndim=new_ndim))
                    * initial
                ).astype(out_dtype)
        return fix_dtypes_and_determine_return(result, new_ndim, dtype, out, where)

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
        out_dtype = (
            dtype
            if dtype is not None
            else (out.dtype if out is not None else self.dtype)
        )
        check_kwargs(order=order, subok=subok, casting=casting, where=where)
        if is_scalar(x2):
            return fix_dtypes_and_determine_return(
                self._query_compiler.astype(
                    {col_name: out_dtype for col_name in self._query_compiler.columns}
                ).mul(x2),
                self._ndim,
                dtype,
                out,
                where,
            )
        caller, callee, new_ndim, kwargs = self._binary_op(x2)
        caller_qc = caller._query_compiler.astype(
            {col_name: out_dtype for col_name in caller._query_compiler.columns}
        )
        callee_qc = callee._query_compiler.astype(
            {col_name: out_dtype for col_name in callee._query_compiler.columns}
        )
        result = caller_qc.mul(callee_qc, **kwargs)
        return fix_dtypes_and_determine_return(result, new_ndim, dtype, out, where)

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
        return self.multiply(x2, out, where, casting, order, dtype, subok)

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
        out_dtype = (
            dtype
            if dtype is not None
            else (out.dtype if out is not None else self.dtype)
        )
        check_kwargs(order=order, subok=subok, casting=casting, where=where)
        if is_scalar(x2):
            result = self._query_compiler.astype(
                {col_name: out_dtype for col_name in self._query_compiler.columns}
            ).mod(x2)
            if x2 == 0:
                # NumPy's remainder by 0 works differently from pandas', so we need to fix
                # the output.
                result = result.replace(numpy.NaN, 0)
            return fix_dtypes_and_determine_return(
                result, self._ndim, dtype, out, where
            )
        caller, callee, new_ndim, kwargs = self._binary_op(x2)
        if caller._query_compiler != self._query_compiler:
            # Modin does not correctly support broadcasting when the caller of the function is
            # a Series (1D), and the operand is a Dataframe (2D). We cannot workaround this using
            # commutativity, and `rmod` also works incorrectly. GH#5529
            raise NotImplementedError(
                "Using remainder with broadcast is not currently available in Modin."
            )
        caller_qc = caller._query_compiler.astype(
            {col_name: out_dtype for col_name in caller._query_compiler.columns}
        )
        callee_qc = callee._query_compiler.astype(
            {col_name: out_dtype for col_name in callee._query_compiler.columns}
        )
        result = caller_qc.mod(callee_qc, **kwargs)
        if callee._query_compiler.eq(0).any():
            # NumPy's floor_divide by 0 works differently from pandas', so we need to fix
            # the output.
            result = result.replace(numpy.NaN, 0)
        return fix_dtypes_and_determine_return(result, new_ndim, dtype, out, where)

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
        out_dtype = (
            dtype
            if dtype is not None
            else (out.dtype if out is not None else self.dtype)
        )
        check_kwargs(order=order, subok=subok, casting=casting, where=where)
        if is_scalar(x2):
            return fix_dtypes_and_determine_return(
                self._query_compiler.astype(
                    {col_name: out_dtype for col_name in self._query_compiler.columns}
                ).sub(x2),
                self._ndim,
                dtype,
                out,
                where,
            )
        caller, callee, new_ndim, kwargs = self._binary_op(x2)
        caller_qc = caller._query_compiler.astype(
            {col_name: out_dtype for col_name in caller._query_compiler.columns}
        )
        callee_qc = callee._query_compiler.astype(
            {col_name: out_dtype for col_name in callee._query_compiler.columns}
        )
        if caller._query_compiler != self._query_compiler:
            # In this case, we are doing an operation that looks like this 1D_object - 2D_object.
            # For Modin to broadcast directly, we have to swap it so that the operation is actually
            # 2D_object.rsub(1D_object).
            result = caller_qc.rsub(callee_qc, **kwargs)
        else:
            result = caller_qc.sub(callee_qc, **kwargs)
        return fix_dtypes_and_determine_return(result, new_ndim, dtype, out, where)

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
        out_dtype = (
            dtype
            if dtype is not None
            else (out.dtype if out is not None else self.dtype)
        )
        check_kwargs(order=order, subok=subok, casting=casting, where=where)
        if is_scalar(x2):
            return fix_dtypes_and_determine_return(
                self._query_compiler.astype(
                    {col_name: out_dtype for col_name in self._query_compiler.columns}
                ).rsub(x2),
                self._ndim,
                dtype,
                out,
                where,
            )
        caller, callee, new_ndim, kwargs = self._binary_op(x2)
        caller_qc = caller._query_compiler.astype(
            {col_name: out_dtype for col_name in caller._query_compiler.columns}
        )
        callee_qc = callee._query_compiler.astype(
            {col_name: out_dtype for col_name in callee._query_compiler.columns}
        )
        if caller._query_compiler != self._query_compiler:
            # In this case, we are doing an operation that looks like this 1D_object - 2D_object.
            # For Modin to broadcast directly, we have to swap it so that the operation is actually
            # 2D_object.sub(1D_object).
            result = caller_qc.sub(callee_qc, **kwargs)
        else:
            result = caller_qc.rsub(callee_qc, **kwargs)
        return fix_dtypes_and_determine_return(result, new_ndim, dtype, out, where)

    def sum(
        self, axis=None, dtype=None, out=None, keepdims=None, initial=None, where=True
    ):
        out_dtype = (
            dtype
            if dtype is not None
            else (out.dtype if out is not None else self.dtype)
        )
        initial = 0 if initial is None else initial
        check_kwargs(keepdims=keepdims, where=where)
        if self._ndim == 1:
            if axis == 1:
                raise numpy.AxisError(1, 1)
            result = self._query_compiler.astype(
                {col_name: out_dtype for col_name in self._query_compiler.columns}
            ).sum(axis=0)
            if initial is not None:
                result = result.add(initial)
            if keepdims:
                if initial is not None:
                    if out is not None:
                        out._query_compiler = (
                            numpy.ones_like(out, dtype=out_dtype) * initial
                        )._query_compiler
                    else:
                        out = array([initial], dtype=out_dtype)
                if out is not None and out.shape != (1,):
                    raise ValueError(
                        f"operand was set up as a reduction along axis 0, but the length of the axis is {out.shape[0]} (it has to be 1)"
                    )
                return fix_dtypes_and_determine_return(result, 1, dtype, out, where)
            return result.to_numpy()[0, 0] if where else initial
        if axis is None:
            result = (
                self.flatten()
                .astype(out_dtype)
                .sum(
                    axis=axis,
                    dtype=dtype,
                    out=out,
                    keepdims=None,
                    initial=initial,
                    where=where,
                )
            )
            if keepdims:
                if out is not None and out.shape != (1, 1):
                    raise ValueError(
                        f"operand was set up as a reduction along axis 0, but the length of the axis is {out.shape[0]} (it has to be 1)"
                    )
                return fix_dtypes_and_determine_return(
                    array(numpy.array([[result]], dtype=out_dtype))._query_compiler,
                    2,
                    dtype,
                    out,
                    where,
                )
            return result
        result = self._query_compiler.astype(
            {col_name: out_dtype for col_name in self._query_compiler.columns}
        ).sum(axis=axis)
        if initial is not None:
            result = result.add(initial)
        new_ndim = self._ndim - 1 if not keepdims else self._ndim
        if new_ndim == 0:
            return result.to_numpy()[0, 0] if where else initial
        if not keepdims and axis != 1:
            result = result.transpose()
        if initial is not None:
            if out is not None:
                out._query_compiler = (
                    (numpy.ones_like(out) * initial).astype(out_dtype)._query_compiler
                )
            else:
                out = (
                    numpy.ones_like(array(_query_compiler=result, _ndim=new_ndim))
                    * initial
                ).astype(out_dtype)
        return fix_dtypes_and_determine_return(result, new_ndim, dtype, out, where)

    def flatten(self, order="C"):
        check_kwargs(order=order)
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
        new_query_compiler = new_query_compiler.transpose()
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

    def transpose(self):
        if self._ndim == 1:
            return self
        return array(_query_compiler=self._query_compiler.transpose(), _ndim=self._ndim)

    T = property(transpose)

    @property
    def dtype(self):
        dtype = self._query_compiler.dtypes
        if self._ndim == 1:
            return dtype[0]
        else:
            return find_common_dtype(dtype.values)

    def astype(self, dtype, order="K", casting="unsafe", subok=True, copy=True):
        if casting != "unsafe":
            raise ValueError(
                "Modin does not support `astype` with `casting != unsafe`."
            )
        check_kwargs(order=order, subok=subok)
        result = self._query_compiler.astype(
            {col_name: dtype for col_name in self._query_compiler.columns}
        )
        if not copy and subok and numpy.issubdtype(self.dtype, dtype):
            return self
        return array(_query_compiler=result, _ndim=self._ndim)

    def __repr__(self):
        return repr(self._to_numpy())

    def _to_numpy(self):
        arr = self._query_compiler.to_numpy()
        if self._ndim == 1:
            arr = arr.flatten()
        return arr
