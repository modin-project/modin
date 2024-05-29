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

from inspect import signature
from math import prod

import numpy
import pandas
from pandas.api.types import is_scalar
from pandas.core.dtypes.common import is_bool_dtype, is_list_like, is_numeric_dtype

import modin.pandas as pd
from modin.core.dataframe.algebra import Binary, Map, Reduce
from modin.error_message import ErrorMessage

from .utils import try_convert_from_interoperable_type


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


def check_can_broadcast_to_output(arr_in: "array", arr_out: "array"):
    if not isinstance(arr_out, array):
        raise TypeError("return arrays must be of modin.numpy.array type.")
    # Broadcasting is ok if both arrays have matching ndim + shape, OR
    # arr_in is 1xN or a 1D N-element array and arr_out is MxN.
    # Note that 1xN arr_in cannot be broadcasted into a 1D N-element arr_out.
    #
    # This is slightly different from the rules for checking if two inputs
    # of a binary operation can be broadcasted together.
    broadcast_ok = (
        (
            # Case 1: arrays have matching ndim + shape
            # Case 2a: arr_in is 1D N-element, arr_out is 1D N-element (covered here)
            arr_in._ndim == arr_out._ndim
            and arr_in.shape == arr_out.shape
        )
        or (
            # Case 2b: both arrays are 2D, arr_in is 1xN and arr_out is MxN
            arr_in._ndim == 2
            and arr_out._ndim == 2
            and arr_in.shape[0] == 1
            and arr_in.shape[1] == arr_out.shape[1]
        )
        or (
            # Case 2c: arr_in is 1D N-element, arr_out is MxN
            arr_in._ndim == 1
            and arr_out._ndim == 2
            and arr_in.shape[0] == arr_out.shape[1]
            and arr_out.shape[0] == 1
        )
    )
    # Case 2b would require duplicating the 1xN result M times to match the shape of out,
    # which we currently do not support. See GH#5831.
    if (
        arr_in._ndim == 2
        and arr_out._ndim == 2
        and arr_in.shape[0] == 1
        and arr_in.shape[1] == arr_out.shape[1]
        and arr_in.shape[0] != 1
    ):
        raise NotImplementedError(
            f"Modin does not currently support broadcasting shape {arr_in.shape} to output operand with shape {arr_out.shape}"
        )
    if not broadcast_ok:
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
        out = try_convert_from_interoperable_type(out, copy=False)
        check_can_broadcast_to_output(result, out)
        result._query_compiler = result._query_compiler.astype(
            {col_name: out.dtype for col_name in result._query_compiler.columns}
        )
        if isinstance(where, array):
            out._update_inplace(where.where(result, out)._query_compiler)
        elif where:
            out._update_inplace(result._query_compiler)
        return out
    if isinstance(where, array) and out is None:
        from .array_creation import zeros_like

        out = zeros_like(result).astype(dtype if dtype is not None else result.dtype)
        out._query_compiler = where.where(result, out)._query_compiler
        return out
    elif not where:
        from .array_creation import zeros_like

        return zeros_like(result)
    return result


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
        self._siblings = []
        ErrorMessage.single_warning(
            "Using Modin's new NumPy API. To convert from a Modin object to a NumPy array, either turn off the ModinNumpy flag, or use `modin.pandas.io.to_numpy`."
        )
        if isinstance(object, array):
            _query_compiler = object._query_compiler.copy()
            if not copy:
                object._add_sibling(self)
            _ndim = object._ndim
        elif isinstance(object, (pd.DataFrame, pd.Series)):
            _query_compiler = object._query_compiler.copy()
            if not copy:
                object._add_sibling(self)
            _ndim = 1 if isinstance(object, pd.Series) else 2
        if _query_compiler is not None:
            self._query_compiler = _query_compiler
            self._ndim = _ndim
            new_dtype = pandas.core.dtypes.cast.find_common_type(
                list(self._query_compiler.dtypes.values)
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
            arr = numpy.asarray(object)
            assert arr.ndim in (
                1,
                2,
            ), "modin.numpy currently only supports 1D and 2D objects."
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
        if isinstance(new_dtype, pandas.Float64Dtype):
            new_dtype = numpy.float64
        cols_with_wrong_dtype = self._query_compiler.dtypes != new_dtype
        if cols_with_wrong_dtype.any():
            self._query_compiler = self._query_compiler.astype(
                {
                    col_name: new_dtype
                    for col_name in self._query_compiler.columns[cols_with_wrong_dtype]
                }
            )
        self.indexer = None

    def __getitem__(self, key):
        if isinstance(key, array) and is_bool_dtype(key.dtype) and key._ndim == 2:
            raise NotImplementedError(
                "Advanced indexing with 2D boolean indexes is not currently supported."
            )
        if self.indexer is None:
            from .indexing import ArrayIndexer

            self.indexer = ArrayIndexer(self)
        return self.indexer.__getitem__(key)

    def __setitem__(self, key, item):
        if self.indexer is None:
            from .indexing import ArrayIndexer

            self.indexer = ArrayIndexer(self)
        return self.indexer.__setitem__(key, item)

    def _add_sibling(self, sibling):
        """
        Add an array object to the list of siblings.

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

    def _update_inplace(self, new_query_compiler):
        """
        Update the current array inplace.

        Parameters
        ----------
        new_query_compiler : query_compiler
            The new QueryCompiler to use to manage the data.
        """
        old_query_compiler = self._query_compiler
        self._query_compiler = new_query_compiler
        for sib in self._siblings:
            sib._query_compiler = new_query_compiler
        old_query_compiler.free()

    def _validate_axis(self, axis):
        """
        Check that the provided axis argument is valid on this array.

        Parameters
        ----------
        axis : int, optional
            The axis argument passed to the function.

        Returns
        -------
        int, optional
            Axis to apply the function over (None, 0, or 1).

        Raises
        -------
        numpy.AxisError
            if the axis is invalid.
        """
        if axis is not None and axis < 0:
            new_axis = axis + self._ndim
            if self._ndim == 1 and new_axis != 0:
                raise numpy.AxisError(axis, 1)
            elif self._ndim == 2 and new_axis not in [0, 1]:
                raise numpy.AxisError(axis, 2)
            return new_axis
        return axis

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
        out_kwarg = kwargs.get("out", None)
        if out_kwarg is not None:
            # If `out` is a modin.numpy.array, `kwargs.get("out")` returns a 1-tuple
            # whose only element is that array, so we need to unwrap it from the tuple.
            out_kwarg = out_kwarg[0]
        where_kwarg = kwargs.get("where", True)
        kwargs["out"] = None
        kwargs["where"] = True
        result = new_ufunc(*args, **kwargs)
        return fix_dtypes_and_determine_return(
            result,
            out_ndim,
            dtype=kwargs.get("dtype", None),
            out=out_kwarg,
            where=where_kwarg,
        )

    def __array_function__(self, func, types, args, kwargs):
        from . import array_creation as creation
        from . import array_shaping as shaping
        from . import math

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
        if not is_bool_dtype(self.dtype):
            raise NotImplementedError(
                "Modin currently only supports where on condition arrays with boolean dtype."
            )
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
        apply_axis = self._validate_axis(axis)
        truthy_where = bool(where)
        if initial is None and where is not True:
            raise ValueError(
                "reduction operation 'maximum' does not have an identity, so to use a where mask one has to specify 'initial'"
            )
        if self._ndim == 1:
            if apply_axis == 1:
                raise numpy.AxisError(1, 1)
            target = where.where(self, initial) if isinstance(where, array) else self
            result = target._query_compiler.max(axis=0)
            if keepdims:
                if initial is not None and result.lt(initial).any():
                    result = pd.Series([initial])._query_compiler
                if initial is not None and out is not None:
                    out._update_inplace(
                        (numpy.ones_like(out) * initial)._query_compiler
                    )
                if out is not None and out.shape != (1,):
                    raise ValueError(
                        f"operand was set up as a reduction along axis 0, but the length of the axis is {out.shape[0]} (it has to be 1)"
                    )
                if truthy_where or out is not None:
                    return fix_dtypes_and_determine_return(
                        result, 1, dtype, out, truthy_where
                    )
                else:
                    return array([initial])
            if initial is not None:
                result = max(result.to_numpy()[0, 0], initial)
            else:
                result = result.to_numpy()[0, 0]
            return result if truthy_where else initial
        if axis is None:
            target = where.where(self, initial) if isinstance(where, array) else self
            result = target._query_compiler.max(axis=0).max(axis=1).to_numpy()[0, 0]
            if initial is not None:
                result = max(result, initial)
            if keepdims:
                if out is not None and out.shape != (1, 1):
                    raise ValueError(
                        f"operand was set up as a reduction along axis 1, but the length of the axis is {out.shape[0]} (it has to be 1)"
                    )
                if initial is not None and out is not None:
                    out._update_inplace(
                        (numpy.ones_like(out) * initial)._query_compiler
                    )
                if truthy_where or out is not None:
                    return fix_dtypes_and_determine_return(
                        array(numpy.array([[result]]))._query_compiler,
                        2,
                        dtype,
                        out,
                        truthy_where,
                    )
                else:
                    return array([[initial]])
            return result if truthy_where else initial
        if apply_axis > 1:
            raise numpy.AxisError(axis, 2)
        target = where.where(self, initial) if isinstance(where, array) else self
        result = target._query_compiler.max(axis=apply_axis)
        new_ndim = self._ndim - 1 if not keepdims else self._ndim
        if new_ndim == 0:
            if initial is not None:
                result = max(result.to_numpy()[0, 0], initial)
            else:
                result = result.to_numpy()[0, 0]
            return result if truthy_where else initial
        if not keepdims and apply_axis != 1:
            result = result.transpose()
        if initial is not None and out is not None:
            out._update_inplace((numpy.ones_like(out) * initial)._query_compiler)
        intermediate = fix_dtypes_and_determine_return(
            result, new_ndim, dtype, out, truthy_where
        )
        if initial is not None:
            intermediate._update_inplace(
                (intermediate > initial).where(intermediate, initial)._query_compiler
            )
        if truthy_where or out is not None:
            return intermediate
        else:
            return numpy.ones_like(intermediate) * initial

    def min(
        self, axis=None, dtype=None, out=None, keepdims=None, initial=None, where=True
    ):
        check_kwargs(keepdims=keepdims, where=where)
        truthy_where = bool(where)
        apply_axis = self._validate_axis(axis)
        if initial is None and where is not True:
            raise ValueError(
                "reduction operation 'minimum' does not have an identity, so to use a where mask one has to specify 'initial'"
            )
        if self._ndim == 1:
            if apply_axis == 1:
                raise numpy.AxisError(1, 1)
            target = where.where(self, initial) if isinstance(where, array) else self
            result = target._query_compiler.min(axis=0)
            if keepdims:
                if initial is not None and result.gt(initial).any():
                    result = pd.Series([initial])._query_compiler
                if initial is not None and out is not None:
                    out._update_inplace(
                        (numpy.ones_like(out) * initial)._query_compiler
                    )
                if out is not None and out.shape != (1,):
                    raise ValueError(
                        f"operand was set up as a reduction along axis 0, but the length of the axis is {out.shape[0]} (it has to be 1)"
                    )
                if truthy_where or out is not None:
                    return fix_dtypes_and_determine_return(
                        result, 1, dtype, out, truthy_where
                    )
                else:
                    return array([initial])
            if initial is not None:
                result = min(result.to_numpy()[0, 0], initial)
            else:
                result = result.to_numpy()[0, 0]
            return result if truthy_where else initial
        if apply_axis is None:
            target = where.where(self, initial) if isinstance(where, array) else self
            result = target._query_compiler.min(axis=0).min(axis=1).to_numpy()[0, 0]
            if initial is not None:
                result = min(result, initial)
            if keepdims:
                if out is not None and out.shape != (1, 1):
                    raise ValueError(
                        f"operand was set up as a reduction along axis 1, but the length of the axis is {out.shape[0]} (it has to be 1)"
                    )
                if initial is not None and out is not None:
                    out._update_inplace(
                        (numpy.ones_like(out) * initial)._query_compiler
                    )
                if truthy_where or out is not None:
                    return fix_dtypes_and_determine_return(
                        array(numpy.array([[result]]))._query_compiler,
                        2,
                        dtype,
                        out,
                        truthy_where,
                    )
                else:
                    return array([[initial]])
            return result if truthy_where else initial
        if apply_axis > 1:
            raise numpy.AxisError(axis, 2)
        target = where.where(self, initial) if isinstance(where, array) else self
        result = target._query_compiler.min(axis=apply_axis)
        new_ndim = self._ndim - 1 if not keepdims else self._ndim
        if new_ndim == 0:
            if initial is not None:
                result = min(result.to_numpy()[0, 0], initial)
            else:
                result = result.to_numpy()[0, 0]
            return result if truthy_where else initial
        if not keepdims and apply_axis != 1:
            result = result.transpose()
        if initial is not None and out is not None:
            out._update_inplace((numpy.ones_like(out) * initial)._query_compiler)
        intermediate = fix_dtypes_and_determine_return(
            result, new_ndim, dtype, out, truthy_where
        )
        if initial is not None:
            intermediate._update_inplace(
                (intermediate < initial).where(intermediate, initial)._query_compiler
            )
        if truthy_where or out is not None:
            return intermediate
        else:
            return numpy.ones_like(intermediate) * initial

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
            out = try_convert_from_interoperable_type(out, copy=False)
            check_can_broadcast_to_output(self, out)
            out._update_inplace(result)
            return out
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

    def _preprocess_binary_op(self, other, cast_input_types=True, dtype=None, out=None):
        """
        Processes arguments and performs dtype conversions necessary to perform binary
        operations. If the arguments to the binary operation are a 1D object and a 2D object,
        then it will swap the order of the caller and callee return values in order to
        facilitate native broadcasting by modin.

        This function may modify `self._query_compiler` and `other._query_compiler` by replacing
        it with the result of `astype`.

        Parameters
        ----------
        other : array or scalar
            The RHS of the binary operation.
        cast_input_types : bool, default: True
            If specified, the columns of the caller/callee query compilers will be assigned
            dtypes in the following priority, depending on what values were specified:
            (1) the `dtype` argument,
            (2) the dtype of the `out` array,
            (3) the common parent dtype of `self` and `other`.
            If this flag is not specified, then the resulting dtype is left to be determined
            by the result of the modin operation.
        dtype : numpy type, optional
            The desired dtype of the output array.
        out : array, optional
            Existing array object to which to assign the computation's result.

        Returns
        -------
        tuple
            Returns a 4-tuple with the following elements:
            - 0: QueryCompiler object that is the LHS of the binary operation, with types converted
                 as needed.
            - 1: QueryCompiler object OR scalar that is the RHS of the binary operation, with types
                 converted as needed.
            - 2: The ndim of the result.
            - 3: kwargs to pass to the query compiler.
        """
        other = try_convert_from_interoperable_type(other)

        if cast_input_types:
            operand_dtype = (
                self.dtype
                if not isinstance(other, array)
                else pandas.core.dtypes.cast.find_common_type([self.dtype, other.dtype])
            )
            out_dtype = (
                dtype
                if dtype is not None
                else (out.dtype if out is not None else operand_dtype)
            )
            self._query_compiler = self._query_compiler.astype(
                {col_name: out_dtype for col_name in self._query_compiler.columns}
            )
        if is_scalar(other):
            # Return early, since no need to check broadcasting behavior if RHS is a scalar
            return (self._query_compiler, other, self._ndim, {})
        elif cast_input_types:
            other._query_compiler = other._query_compiler.astype(
                {col_name: out_dtype for col_name in other._query_compiler.columns}
            )

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
            return (
                caller._query_compiler,
                callee._query_compiler,
                caller._ndim,
                {"broadcast": broadcast, "axis": 1},
            )
        else:
            if self.shape != other.shape:
                # In this case, we either have two mismatched objects trying to do an operation
                # or a nested 1D object that must be broadcasted trying to do an operation.
                broadcast = True
                if self.shape[0] == other.shape[0]:
                    matched_dimension = 0
                elif self.shape[1] == other.shape[1]:
                    matched_dimension = 1
                    broadcast = False
                else:
                    raise ValueError(
                        f"operands could not be broadcast together with shapes {self.shape} {other.shape}"
                    )
                if (
                    self.shape[matched_dimension ^ 1] == 1
                    or other.shape[matched_dimension ^ 1] == 1
                ):
                    return (
                        self._query_compiler,
                        other._query_compiler,
                        self._ndim,
                        {"broadcast": broadcast, "axis": matched_dimension},
                    )
                else:
                    raise ValueError(
                        f"operands could not be broadcast together with shapes {self.shape} {other.shape}"
                    )
            else:
                return (
                    self._query_compiler,
                    other._query_compiler,
                    self._ndim,
                    {"broadcast": False},
                )

    def _greater(
        self,
        x2,
        /,
        out=None,
        *,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        check_kwargs(where=where, casting=casting, order=order, subok=subok)
        if is_scalar(x2):
            return array(_query_compiler=self._query_compiler.gt(x2), _ndim=self._ndim)
        caller, callee, new_ndim, kwargs = self._preprocess_binary_op(
            x2, cast_input_types=False, dtype=dtype, out=out
        )
        if caller != self._query_compiler:
            # In this case, we are doing an operation that looks like this 1D_object > 2D_object.
            # For Modin to broadcast directly, we have to swap it so that the operation is actually
            # 2D_object < 1D_object.
            result = caller.lt(callee, **kwargs)
        else:
            result = caller.gt(callee, **kwargs)
        return fix_dtypes_and_determine_return(result, new_ndim, dtype, out, where)

    def __gt__(self, x2):
        return self._greater(x2)

    def _greater_equal(
        self,
        x2,
        /,
        out=None,
        *,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        check_kwargs(where=where, casting=casting, order=order, subok=subok)
        if is_scalar(x2):
            return array(_query_compiler=self._query_compiler.ge(x2), _ndim=self._ndim)
        caller, callee, new_ndim, kwargs = self._preprocess_binary_op(
            x2, cast_input_types=False, dtype=dtype, out=out
        )
        if caller != self._query_compiler:
            # In this case, we are doing an operation that looks like this 1D_object >= 2D_object.
            # For Modin to broadcast directly, we have to swap it so that the operation is actually
            # 2D_object <= 1D_object.
            result = caller.le(callee, **kwargs)
        else:
            result = caller.ge(callee, **kwargs)
        return fix_dtypes_and_determine_return(result, new_ndim, dtype, out, where)

    def __ge__(self, x2):
        return self._greater_equal(x2)

    def _less(
        self,
        x2,
        /,
        out=None,
        *,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        check_kwargs(where=where, casting=casting, order=order, subok=subok)
        if is_scalar(x2):
            return array(_query_compiler=self._query_compiler.lt(x2), _ndim=self._ndim)
        caller, callee, new_ndim, kwargs = self._preprocess_binary_op(
            x2, cast_input_types=False, dtype=dtype, out=out
        )
        if caller != self._query_compiler:
            # In this case, we are doing an operation that looks like this 1D_object < 2D_object.
            # For Modin to broadcast directly, we have to swap it so that the operation is actually
            # 2D_object < 1D_object.
            result = caller.gt(callee, **kwargs)
        else:
            result = caller.lt(callee, **kwargs)
        return fix_dtypes_and_determine_return(result, new_ndim, dtype, out, where)

    def __lt__(self, x2):
        return self._less(x2)

    def _less_equal(
        self,
        x2,
        /,
        out=None,
        *,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        check_kwargs(where=where, casting=casting, order=order, subok=subok)
        if is_scalar(x2):
            return array(_query_compiler=self._query_compiler.le(x2), _ndim=self._ndim)
        caller, callee, new_ndim, kwargs = self._preprocess_binary_op(
            x2, cast_input_types=False, dtype=dtype, out=out
        )
        if caller != self._query_compiler:
            # In this case, we are doing an operation that looks like this 1D_object <= 2D_object.
            # For Modin to broadcast directly, we have to swap it so that the operation is actually
            # 2D_object <= 1D_object.
            result = caller.ge(callee, **kwargs)
        else:
            result = caller.le(callee, **kwargs)
        return fix_dtypes_and_determine_return(result, new_ndim, dtype, out, where)

    def __le__(self, x2):
        return self._less_equal(x2)

    def _equal(
        self,
        x2,
        /,
        out=None,
        *,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        check_kwargs(where=where, casting=casting, order=order, subok=subok)
        if is_scalar(x2):
            return array(_query_compiler=self._query_compiler.eq(x2), _ndim=self._ndim)
        caller, callee, new_ndim, kwargs = self._preprocess_binary_op(
            x2, cast_input_types=False, dtype=dtype, out=out
        )
        result = caller.eq(callee, **kwargs)
        return fix_dtypes_and_determine_return(result, new_ndim, dtype, out, where)

    def __eq__(self, x2):
        return self._equal(x2)

    def _not_equal(
        self,
        x2,
        /,
        out=None,
        *,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        check_kwargs(where=where, casting=casting, order=order, subok=subok)
        if is_scalar(x2):
            return array(_query_compiler=self._query_compiler.ne(x2), _ndim=self._ndim)
        caller, callee, new_ndim, kwargs = self._preprocess_binary_op(
            x2, cast_input_types=False, dtype=dtype, out=out
        )
        result = caller.ne(callee, **kwargs)
        return fix_dtypes_and_determine_return(result, new_ndim, dtype, out, where)

    def __ne__(self, x2):
        return self._not_equal(x2)

    def _unary_math_operator(
        self,
        opName,
        *args,
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
        )
        result = getattr(result, opName)(*args)
        if dtype is not None:
            result = result.astype({col_name: dtype for col_name in result.columns})
        if out is not None:
            out = try_convert_from_interoperable_type(out)
            check_can_broadcast_to_output(self, out)
            out._query_compiler = result
            return out
        return array(_query_compiler=result, _ndim=self._ndim)

    def tanh(
        self,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        return self._unary_math_operator(
            "_tanh",
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )

    def exp(
        self,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        return self._unary_math_operator(
            "_exp",
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )

    def sqrt(
        self,
        out=None,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        return self._unary_math_operator(
            "_sqrt",
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )

    def append(self, values, axis=None):
        if not isinstance(values, array):
            if is_list_like(values):
                lengths = [len(a) if is_list_like(a) else None for a in values]
                if any(numpy.array(lengths[1:]) != lengths[0]):
                    raise ValueError(
                        "setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (2,) + inhomogeneous part."
                    )
            values = array(values)
        if axis is None:
            return self.flatten().hstack([values.flatten()])
        elif self._ndim == 1:
            if values._ndim == 1:
                return self.hstack([values])
            raise ValueError(
                f"all the input arrays must have same number of dimensions, but the array at index 0 has 1 dimension(s) and the array at index 1 has {values._ndim} dimension(s)"
            )
        if (axis ^ 1 < values._ndim) and self.shape[axis ^ 1] != values.shape[axis ^ 1]:
            raise ValueError(
                f"all the input array dimensions except for the concatenation axis must match exactly, but along dimension {axis ^ 1}, the array at index 0 has size {self.shape[axis^1]} and the array at index 1 has size {values.shape[axis^1]}"
            )
        new_qc = self._query_compiler.concat(axis, values._query_compiler)
        return array(_query_compiler=new_qc, _ndim=self._ndim)

    def hstack(self, others, dtype=None, casting="same_kind"):
        check_kwargs(casting=casting)
        new_dtype = (
            dtype
            if dtype is not None
            else pandas.core.dtypes.cast.find_common_type(
                [self.dtype] + [a.dtype for a in others]
            )
        )
        for index, i in enumerate([a._ndim for a in others]):
            if i != self._ndim:
                raise ValueError(
                    f"all the input arrays must have same number of dimensions, but the array at index 0 has {self._ndim} dimension(s) and the array at index {index} has {i} dimension(s)"
                )
        if self._ndim == 1:
            new_qc = self._query_compiler.concat(0, [o._query_compiler for o in others])
        else:
            for index, i in enumerate([a.shape[0] for a in others]):
                if i != self.shape[0]:
                    raise ValueError(
                        f"all the input array dimensions except for the concatenation axis must match exactly, but along dimension 0, the array at index 0 has size {self.shape[0]} and the array at index {index} has size {i}"
                    )
            new_qc = self._query_compiler.concat(1, [o._query_compiler for o in others])
        return array(_query_compiler=new_qc, _ndim=self._ndim, dtype=new_dtype)

    def split(self, indices, axis=0):
        if axis is not None and axis < 0:
            new_axis = axis + self._ndim
            if self._ndim == 1 and new_axis != 0:
                raise IndexError
            elif self._ndim == 2 and new_axis not in [0, 1]:
                raise IndexError
            axis = new_axis
        if self._ndim == 1:
            if axis != 0:
                raise IndexError
        if self._ndim == 2:
            if axis > 1:
                raise IndexError
        arrays = []
        if is_list_like(indices) or isinstance(indices, array):
            if not isinstance(indices, array):
                indices = array(indices)
            if indices._ndim != 1:
                raise TypeError(
                    "only integer scalar arrays can be converted to a scalar index"
                )
            prev_index = 0
            for i in range(len(indices) + 1):
                if i < len(indices):
                    end_index = indices._query_compiler.take_2d_positional(
                        [i]
                    ).to_numpy()[0, 0]
                    if end_index == 0:
                        ErrorMessage.single_warning(
                            "Defaulting to NumPy for empty arrays."
                        )
                        new_shape = list(self.shape)
                        new_shape[axis] = 0
                        arrays.append(numpy.empty(new_shape, dtype=self.dtype))
                        continue
                    if end_index < 0:
                        end_index = self.shape[axis] + end_index
                else:
                    end_index = self.shape[axis]
                if prev_index > self.shape[axis] or prev_index == end_index:
                    ErrorMessage.single_warning("Defaulting to NumPy for empty arrays.")
                    new_shape = list(self.shape)
                    new_shape[axis] = 0
                    arrays.append(numpy.empty(new_shape, dtype=self.dtype))
                else:
                    idxs = list(range(prev_index, min(end_index, self.shape[axis])))
                    if axis == 0:
                        new_qc = self._query_compiler.take_2d_positional(index=idxs)
                    else:
                        new_qc = self._query_compiler.take_2d_positional(columns=idxs)
                    arrays.append(array(_query_compiler=new_qc, _ndim=self._ndim))
                prev_index = end_index
        else:
            if self.shape[axis] % indices != 0:
                raise ValueError("array split does not result in an equal division")
            for i in range(0, self.shape[axis], self.shape[axis] // indices):
                if axis == 0:
                    new_qc = self._query_compiler.take_2d_positional(
                        index=list(range(i, i + self.shape[axis] // indices))
                    )
                else:
                    new_qc = self._query_compiler.take_2d_positional(
                        columns=list(range(i, i + self.shape[axis] // indices))
                    )
                arrays.append(array(_query_compiler=new_qc, _ndim=self._ndim))
        return arrays

    def _compute_masked_variance(self, mask, output_dtype, axis, ddof):
        if axis == 0 and self._ndim != 1:
            # Our broadcasting is wrong, so we can't do the final subtraction at the end.
            raise NotImplementedError(
                "Masked variance on 2D arrays along axis = 0 is currently unsupported."
            )
        axis_mean = self.mean(axis, output_dtype, keepdims=True, where=mask)
        target = mask.where(self, numpy.nan)
        if self._ndim == 1:
            axis_mean = axis_mean._to_numpy()[0]
            target = target._query_compiler.sub(axis_mean).pow(2).sum(axis=axis)
        else:
            target = (target - axis_mean)._query_compiler.pow(2).sum(axis=axis)
        num_elems = (
            mask.where(self, 0)._query_compiler.notna().sum(axis=axis, skipna=False)
        )
        num_elems = num_elems.sub(ddof)
        target = target.truediv(num_elems)
        na_propagation_mask = mask.where(self, 0)._query_compiler.sum(
            axis=axis, skipna=False
        )
        target = target.where(na_propagation_mask.notna(), numpy.nan)
        return target

    def var(
        self, axis=None, dtype=None, out=None, ddof=0, keepdims=None, *, where=True
    ):
        out_dtype = (
            dtype
            if dtype is not None
            else (out.dtype if out is not None else self.dtype)
        )
        out_type = getattr(out_dtype, "type", out_dtype)
        if isinstance(where, array) and issubclass(out_type, numpy.integer):
            out_dtype = numpy.float64
        apply_axis = self._validate_axis(axis)
        check_kwargs(keepdims=keepdims, where=where)
        truthy_where = bool(where)
        if self._ndim == 1:
            if apply_axis == 1:
                raise numpy.AxisError(1, 1)
            if isinstance(where, array):
                result = self._compute_masked_variance(where, out_dtype, 0, ddof)
            else:
                result = self._query_compiler.var(axis=0, skipna=False, ddof=ddof)
            if keepdims:
                if out is not None and out.shape != (1,):
                    raise ValueError(
                        f"operand was set up as a reduction along axis 0, but the length of the axis is {out.shape[0]} (it has to be 1)"
                    )
                if out is not None:
                    out._query_compiler = (
                        numpy.ones_like(out) * numpy.nan
                    )._query_compiler
                if truthy_where or out is not None:
                    return fix_dtypes_and_determine_return(
                        result, 1, dtype, out, truthy_where
                    )
                else:
                    return array([numpy.nan], dtype=out_dtype)
        if apply_axis is None:
            # If any of the (non-masked) elements of our array are `NaN`, we know that the
            # result of `mean` must be `NaN`. This is a fastpath to see if any unmasked elements
            # are `NaN`.
            contains_na_check = (
                where.where(self, 0) if isinstance(where, array) else self
            )
            if (
                contains_na_check._query_compiler.isna()
                .any(axis=1)
                .any(axis=0)
                .to_numpy()[0, 0]
            ):
                return numpy.nan
            result = where.where(self, numpy.nan) if isinstance(where, array) else self
            # Since our current QueryCompiler does not have a variance that reduces 2D objects to
            # a single value, we need to calculate the variance ourselves. First though, we need
            # to figure out how many objects that we are taking the variance over (since any
            # entries in our array that are `numpy.nan` must be ignored when taking the variance,
            # and so cannot be included in the final division (of the sum over num total elements))
            num_na_elements = (
                result._query_compiler.isna().sum(axis=1).sum(axis=0).to_numpy()[0, 0]
            )
            num_total_elements = prod(self.shape) - num_na_elements
            mean = (
                numpy.array(
                    [result._query_compiler.sum(axis=1).sum(axis=0).to_numpy()[0, 0]],
                    dtype=out_dtype,
                )
                / num_total_elements
            )[0]
            result = (
                numpy.array(
                    [
                        result._query_compiler.sub(mean)
                        .pow(2)
                        .sum(axis=1)
                        .sum(axis=0)
                        .to_numpy()[0, 0]
                    ],
                    dtype=out_dtype,
                )
                / (num_total_elements - ddof)
            )[0]
            if keepdims:
                if out is not None and out.shape != (1, 1):
                    raise ValueError(
                        f"operand was set up as a reduction along axis 1, but the length of the axis is {out.shape[0]} (it has to be 1)"
                    )
                if out is not None:
                    out._query_compiler = (
                        numpy.ones_like(out) * numpy.nan
                    )._query_compiler
                if truthy_where or out is not None:
                    return fix_dtypes_and_determine_return(
                        array(numpy.array([[result]]))
                        .astype(out_dtype)
                        ._query_compiler,
                        2,
                        dtype,
                        out,
                        truthy_where,
                    )
                else:
                    return array([[numpy.nan]], dtype=out_dtype)
            return result if truthy_where else numpy.nan
        if apply_axis > 1:
            raise numpy.AxisError(axis, 2)
        if isinstance(where, array):
            result = self._compute_masked_variance(where, out_dtype, apply_axis, ddof)
        else:
            result = self._query_compiler.astype(
                {col_name: out_dtype for col_name in self._query_compiler.columns}
            ).var(axis=apply_axis, skipna=False, ddof=ddof)
        new_ndim = self._ndim - 1 if not keepdims else self._ndim
        if new_ndim == 0:
            return result.to_numpy()[0, 0] if truthy_where else numpy.nan
        if not keepdims and apply_axis != 1:
            result = result.transpose()
        if out is not None:
            out._query_compiler = (numpy.ones_like(out) * numpy.nan)._query_compiler
        if truthy_where or out is not None:
            return fix_dtypes_and_determine_return(
                result, new_ndim, dtype, out, truthy_where
            )
        else:
            return (
                numpy.ones(array(_query_compiler=result, _ndim=new_ndim).shape)
            ) * numpy.nan

    def _compute_masked_mean(self, mask, output_dtype, axis):
        # By default, pandas ignores NaN values when doing computations.
        # NumPy; however, propagates the value by default. We use pandas
        # default behaviour in order to mask values (by replacing them)
        # with NaN when initially computing the mean, but we need to propagate
        # NaN values that were not masked to the final output, so we do a
        # sum along the same axis (where masked values are 0) to see where
        # NumPy would propagate NaN, and swap out those values in our result
        # with NaN.
        target = mask.where(self, numpy.nan)._query_compiler
        target = target.astype(
            {col_name: output_dtype for col_name in target.columns}
        ).mean(axis=axis)
        na_propagation_mask = mask.where(self, 0)._query_compiler
        na_propagation_mask = na_propagation_mask.sum(axis=axis, skipna=False)
        target = target.where(na_propagation_mask.notna(), numpy.nan)
        return target

    def mean(self, axis=None, dtype=None, out=None, keepdims=None, *, where=True):
        out_dtype = (
            dtype
            if dtype is not None
            else (out.dtype if out is not None else self.dtype)
        )
        out_type = getattr(out_dtype, "type", out_dtype)
        if isinstance(where, array) and issubclass(out_type, numpy.integer):
            out_dtype = numpy.float64
        apply_axis = self._validate_axis(axis)
        check_kwargs(keepdims=keepdims, where=where)
        truthy_where = bool(where)
        if self._ndim == 1:
            if apply_axis == 1:
                raise numpy.AxisError(1, 1)
            if isinstance(where, array):
                result = self._compute_masked_mean(where, out_dtype, 0)
            else:
                result = self._query_compiler.astype(
                    {col_name: out_dtype for col_name in self._query_compiler.columns}
                ).mean(axis=0, skipna=False)
            if keepdims:
                if out is not None and out.shape != (1,):
                    raise ValueError(
                        f"operand was set up as a reduction along axis 0, but the length of the axis is {out.shape[0]} (it has to be 1)"
                    )
                if out is not None:
                    out._update_inplace(
                        (numpy.ones_like(out) * numpy.nan)._query_compiler
                    )
                if truthy_where or out is not None:
                    return fix_dtypes_and_determine_return(
                        result, 1, dtype, out, truthy_where
                    )
                else:
                    return array([numpy.nan], dtype=out_dtype)
            # This is just to see if `where` is a truthy value. If `where` is an array,
            # we would have already masked the input before computing `result`, so here
            # we just want to ensure that `where=False` was not passed in, and if it was
            # we return `numpy.nan`, since that is what NumPy would do.
            return result.to_numpy()[0, 0] if where else numpy.nan
        if apply_axis is None:
            # If any of the (non-masked) elements of our array are `NaN`, we know that the
            # result of `mean` must be `NaN`. This is a fastpath to see if any unmasked elements
            # are `NaN`.
            contains_na_check = (
                where.where(self, 0) if isinstance(where, array) else self
            )
            if (
                contains_na_check._query_compiler.isna()
                .any(axis=1)
                .any(axis=0)
                .to_numpy()[0, 0]
            ):
                return numpy.nan
            result = where.where(self, numpy.nan) if isinstance(where, array) else self
            # Since our current QueryCompiler does not have a mean that reduces 2D objects to
            # a single value, we need to calculate the mean ourselves. First though, we need
            # to figure out how many objects that we are taking the mean over (since any
            # entries in our array that are `numpy.nan` must be ignored when taking the mean,
            # and so cannot be included in the final division (of the sum over num total elements))
            num_na_elements = (
                result._query_compiler.isna().sum(axis=1).sum(axis=0).to_numpy()[0, 0]
            )
            num_total_elements = prod(self.shape) - num_na_elements
            result = (
                numpy.array(
                    [result._query_compiler.sum(axis=1).sum(axis=0).to_numpy()[0, 0]],
                    dtype=out_dtype,
                )
                / num_total_elements
            )[0]
            if keepdims:
                if out is not None and out.shape != (1, 1):
                    raise ValueError(
                        f"operand was set up as a reduction along axis 1, but the length of the axis is {out.shape[0]} (it has to be 1)"
                    )
                if out is not None:
                    out._update_inplace(
                        (numpy.ones_like(out) * numpy.nan)._query_compiler
                    )
                if truthy_where or out is not None:
                    return fix_dtypes_and_determine_return(
                        array(numpy.array([[result]]))
                        .astype(out_dtype)
                        ._query_compiler,
                        2,
                        dtype,
                        out,
                        truthy_where,
                    )
                else:
                    return array([[numpy.nan]], dtype=out_dtype)
            return result if truthy_where else numpy.nan
        if apply_axis > 1:
            raise numpy.AxisError(axis, 2)
        if isinstance(where, array):
            result = self._compute_masked_mean(where, out_dtype, apply_axis)
        else:
            result = self._query_compiler.astype(
                {col_name: out_dtype for col_name in self._query_compiler.columns}
            ).mean(axis=apply_axis, skipna=False)
        new_ndim = self._ndim - 1 if not keepdims else self._ndim
        if new_ndim == 0:
            return result.to_numpy()[0, 0] if truthy_where else numpy.nan
        if not keepdims and apply_axis != 1:
            result = result.transpose()
        if out is not None:
            out._update_inplace((numpy.ones_like(out) * numpy.nan)._query_compiler)
        if truthy_where or out is not None:
            return fix_dtypes_and_determine_return(
                result, new_ndim, dtype, out, truthy_where
            )
        else:
            return (
                numpy.ones(array(_query_compiler=result, _ndim=new_ndim).shape)
            ) * numpy.nan

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
        check_kwargs(order=order, subok=subok, casting=casting, where=where)
        caller, callee, new_ndim, kwargs = self._preprocess_binary_op(
            x2, dtype=dtype, out=out
        )
        result = caller.add(callee, **kwargs)
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
        check_kwargs(order=order, subok=subok, casting=casting, where=where)
        caller, callee, new_ndim, kwargs = self._preprocess_binary_op(
            x2, dtype=dtype, out=out
        )
        if caller != self._query_compiler:
            # In this case, we are doing an operation that looks like this 1D_object/2D_object.
            # For Modin to broadcast directly, we have to swap it so that the operation is actually
            # 2D_object.rtruediv(1D_object).
            result = caller.rtruediv(callee, **kwargs)
        else:
            result = caller.truediv(callee, **kwargs)
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
        check_kwargs(order=order, subok=subok, casting=casting, where=where)
        caller, callee, new_ndim, kwargs = self._preprocess_binary_op(
            x2, dtype=dtype, out=out
        )
        if caller != self._query_compiler:
            result = caller.truediv(callee, **kwargs)
        else:
            result = caller.rtruediv(callee, **kwargs)
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
        operand_dtype = (
            self.dtype
            if not isinstance(x2, array)
            else pandas.core.dtypes.cast.find_common_type([self.dtype, x2.dtype])
        )
        out_dtype = (
            dtype
            if dtype is not None
            else (out.dtype if out is not None else operand_dtype)
        )
        check_kwargs(order=order, subok=subok, casting=casting, where=where)
        if is_scalar(x2):
            result = self._query_compiler.floordiv(x2)
            if x2 == 0 and numpy.issubdtype(out_dtype, numpy.integer):
                # NumPy's floor_divide by 0 works differently from pandas', so we need to fix
                # the output.
                result = (
                    result.replace(numpy.inf, 0)
                    .replace(-numpy.inf, 0)
                    .where(self._query_compiler.ne(0), 0)
                )
            return fix_dtypes_and_determine_return(
                result, self._ndim, dtype, out, where
            )
        caller, callee, new_ndim, kwargs = self._preprocess_binary_op(
            x2, dtype=dtype, out=out
        )
        if caller != self._query_compiler:
            # Modin does not correctly support broadcasting when the caller of the function is
            # a Series (1D), and the operand is a Dataframe (2D). We cannot workaround this using
            # commutativity, and `rfloordiv` also works incorrectly. GH#5529
            raise NotImplementedError(
                "Using floor_divide with broadcast is not currently available in Modin."
            )
        result = caller.floordiv(callee, **kwargs)
        if callee.eq(0).any() and numpy.issubdtype(out_dtype, numpy.integer):
            # NumPy's floor_divide by 0 works differently from pandas', so we need to fix
            # the output.
            result = (
                result.replace(numpy.inf, 0)
                .replace(-numpy.inf, 0)
                .where(callee.ne(0), 0)
            )
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
        check_kwargs(order=order, subok=subok, casting=casting, where=where)
        caller, callee, new_ndim, kwargs = self._preprocess_binary_op(
            x2, dtype=dtype, out=out
        )
        if caller != self._query_compiler:
            # Modin does not correctly support broadcasting when the caller of the function is
            # a Series (1D), and the operand is a Dataframe (2D). We cannot workaround this using
            # commutativity, and `rpow` also works incorrectly. GH#5529
            raise NotImplementedError(
                "Using power with broadcast is not currently available in Modin."
            )
        result = caller.pow(callee, **kwargs)
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
        apply_axis = self._validate_axis(axis)
        truthy_where = bool(where)
        if self._ndim == 1:
            if apply_axis == 1:
                raise numpy.AxisError(1, 1)
            target = where.where(self, 1) if isinstance(where, array) else self
            result = target._query_compiler.astype(
                {col_name: out_dtype for col_name in target._query_compiler.columns}
            ).prod(axis=0, skipna=False)
            result = result.mul(initial)
            if keepdims:
                if out is not None:
                    out._update_inplace(
                        (numpy.ones_like(out) * initial)
                        .astype(out_dtype)
                        ._query_compiler
                    )
                if out is not None and out.shape != (1,):
                    raise ValueError(
                        f"operand was set up as a reduction along axis 0, but the length of the axis is {out.shape[0]} (it has to be 1)"
                    )
                if truthy_where or out is not None:
                    return fix_dtypes_and_determine_return(
                        result, 1, dtype, out, truthy_where
                    )
                else:
                    return array([initial], dtype=out_dtype)
            return result.to_numpy()[0, 0] if truthy_where else initial
        if apply_axis is None:
            result = self
            if isinstance(where, array):
                result = where.where(self, 1)
            result = (
                result.astype(out_dtype)
                ._query_compiler.prod(axis=1, skipna=False)
                .prod(axis=0, skipna=False)
                .to_numpy()[0, 0]
            )
            result *= initial
            if keepdims:
                if out is not None and out.shape != (1, 1):
                    raise ValueError(
                        f"operand was set up as a reduction along axis 1, but the length of the axis is {out.shape[0]} (it has to be 1)"
                    )
                if out is not None:
                    out._update_inplace(
                        (numpy.ones_like(out) * initial)
                        .astype(out_dtype)
                        ._query_compiler
                    )
                if truthy_where or out is not None:
                    return fix_dtypes_and_determine_return(
                        array(numpy.array([[result]]))
                        .astype(out_dtype)
                        ._query_compiler,
                        2,
                        dtype,
                        out,
                        truthy_where,
                    )
                else:
                    return array([[initial]], dtype=out_dtype)
            return result if truthy_where else initial
        if apply_axis > 1:
            raise numpy.AxisError(axis, 2)
        target = where.where(self, 1) if isinstance(where, array) else self
        result = target._query_compiler.astype(
            {col_name: out_dtype for col_name in target._query_compiler.columns}
        ).prod(axis=apply_axis, skipna=False)
        result = result.mul(initial)
        new_ndim = self._ndim - 1 if not keepdims else self._ndim
        if new_ndim == 0:
            return result.to_numpy()[0, 0] if truthy_where else initial
        if not keepdims and apply_axis != 1:
            result = result.transpose()
        if initial is not None and out is not None:
            out._update_inplace(
                (numpy.ones_like(out) * initial).astype(out_dtype)._query_compiler
            )
        if truthy_where or out is not None:
            return fix_dtypes_and_determine_return(
                result, new_ndim, dtype, out, truthy_where
            )
        else:
            return (
                numpy.ones_like(array(_query_compiler=result, _ndim=new_ndim)) * initial
            )

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
        check_kwargs(order=order, subok=subok, casting=casting, where=where)
        caller, callee, new_ndim, kwargs = self._preprocess_binary_op(
            x2, dtype=dtype, out=out
        )
        result = caller.mul(callee, **kwargs)
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

    def dot(self, other, out=None):
        other = try_convert_from_interoperable_type(other)
        if numpy.isscalar(other):
            # other is scalar -- result is an array
            result = self._query_compiler.mul(other)
            result_ndim = self._ndim
        elif not isinstance(other, array):
            raise TypeError(
                f"Unsupported operand type(s): '{type(self)}' and '{type(other)}'"
            )
        elif self._ndim == 1 and other._ndim == 1:
            # both 1D arrays -- result is a scalar
            result = self._query_compiler.dot(
                other._query_compiler, squeeze_self=True, squeeze_other=True
            )
            return result.to_numpy()[0, 0]
        elif self._ndim == 2 and other._ndim == 2:
            # both 2D arrays -- result is a 2D array
            result = self._query_compiler.dot(other._query_compiler)
            result_ndim = 2
        elif self._ndim == 1 and other._ndim == 2:
            result = self._query_compiler.dot(other._query_compiler, squeeze_self=True)
            result_ndim = 1
        elif self._ndim == 2 and other._ndim == 1:
            result = self._query_compiler.dot(other._query_compiler)
            result_ndim = 1
        return fix_dtypes_and_determine_return(
            result,
            result_ndim,
            out=out,
        )

    def __matmul__(self, other):
        if numpy.isscalar(other):
            # numpy's original error message is something cryptic about a gufunc signature
            raise ValueError(
                "cannot call matmul with a scalar argument (use np.dot instead)"
            )
        return self.dot(other)

    def _norm(self, ord=None, axis=None, keepdims=False):
        check_kwargs(keepdims=keepdims)
        if ord is not None and ord not in ("fro",):  # , numpy.inf, -numpy.inf, 0):
            raise NotImplementedError("unsupported ord argument for norm:", ord)
        if isinstance(axis, int) and axis < 0:
            apply_axis = self._ndim + axis
        else:
            apply_axis = axis or 0
        if apply_axis >= self._ndim or apply_axis < 0:
            raise numpy.AxisError(axis, self._ndim)
        result = self._query_compiler.pow(2)
        if self._ndim == 2:
            result = result.sum(axis=apply_axis)
            if axis is None:
                result = result.sum(axis=apply_axis ^ 1)
        else:
            result = result.sum(axis=0)
        if axis is None:
            # Return a scalar
            return result._sqrt().to_numpy()[0, 0]
        else:
            result = result._sqrt()
            # the DF may be transposed after processing through pandas
            # check query compiler shape to ensure this is a row vector (1xN) not column (Nx1)
            if len(result.index) != 1:
                result = result.transpose()
            return array(_query_compiler=result, _ndim=1)

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
        operand_dtype = (
            self.dtype
            if not isinstance(x2, array)
            else pandas.core.dtypes.cast.find_common_type([self.dtype, x2.dtype])
        )
        out_dtype = (
            dtype
            if dtype is not None
            else (out.dtype if out is not None else operand_dtype)
        )
        check_kwargs(order=order, subok=subok, casting=casting, where=where)
        if is_scalar(x2):
            result = self._query_compiler.astype(
                {col_name: out_dtype for col_name in self._query_compiler.columns}
            ).mod(x2)
            if x2 == 0 and numpy.issubdtype(out_dtype, numpy.integer):
                # NumPy's remainder by 0 works differently from pandas', so we need to fix
                # the output.
                result = result.replace(numpy.nan, 0)
            return fix_dtypes_and_determine_return(
                result, self._ndim, dtype, out, where
            )
        caller, callee, new_ndim, kwargs = self._preprocess_binary_op(
            x2, dtype=dtype, out=out
        )
        if caller != self._query_compiler:
            # Modin does not correctly support broadcasting when the caller of the function is
            # a Series (1D), and the operand is a Dataframe (2D). We cannot workaround this using
            # commutativity, and `rmod` also works incorrectly. GH#5529
            raise NotImplementedError(
                "Using remainder with broadcast is not currently available in Modin."
            )
        result = caller.mod(callee, **kwargs)
        if callee.eq(0).any() and numpy.issubdtype(out_dtype, numpy.integer):
            # NumPy's floor_divide by 0 works differently from pandas', so we need to fix
            # the output.
            result = result.replace(numpy.nan, 0)
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
        check_kwargs(order=order, subok=subok, casting=casting, where=where)
        caller, callee, new_ndim, kwargs = self._preprocess_binary_op(
            x2, dtype=dtype, out=out
        )
        if caller != self._query_compiler:
            # In this case, we are doing an operation that looks like this 1D_object - 2D_object.
            # For Modin to broadcast directly, we have to swap it so that the operation is actually
            # 2D_object.rsub(1D_object).
            result = caller.rsub(callee, **kwargs)
        else:
            result = caller.sub(callee, **kwargs)
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
        check_kwargs(order=order, subok=subok, casting=casting, where=where)
        caller, callee, new_ndim, kwargs = self._preprocess_binary_op(
            x2, dtype=dtype, out=out
        )
        if caller != self._query_compiler:
            # In this case, we are doing an operation that looks like this 1D_object - 2D_object.
            # For Modin to broadcast directly, we have to swap it so that the operation is actually
            # 2D_object.sub(1D_object).
            result = caller.sub(callee, **kwargs)
        else:
            result = caller.rsub(callee, **kwargs)
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
        apply_axis = self._validate_axis(axis)
        truthy_where = bool(where)
        if self._ndim == 1:
            if apply_axis == 1:
                raise numpy.AxisError(1, 1)
            target = where.where(self, 0) if isinstance(where, array) else self
            if target.dtype != out_dtype:
                target = target.astype(out_dtype)
            result = target._query_compiler.sum(axis=0, skipna=False)
            if initial != 0:
                result = result.add(initial)
            if keepdims:
                if out is not None:
                    out._update_inplace(
                        (
                            numpy.ones_like(out, dtype=out_dtype) * initial
                        )._query_compiler
                    )
                if out is not None and out.shape != (1,):
                    raise ValueError(
                        f"operand was set up as a reduction along axis 0, but the length of the axis is {out.shape[0]} (it has to be 1)"
                    )
                if truthy_where or out is not None:
                    return fix_dtypes_and_determine_return(
                        result, 1, dtype, out, truthy_where
                    )
                else:
                    return array([initial], dtype=out_dtype)
            return result.to_numpy()[0, 0] if truthy_where else initial
        if apply_axis is None:
            target = where.where(self, 0) if isinstance(where, array) else self
            if target.dtype != out_dtype:
                target = target.astype(out_dtype)
            result = (
                target._query_compiler.sum(axis=1, skipna=False)
                .sum(axis=0, skipna=False)
                .to_numpy()[0, 0]
            )
            if initial != 0:
                result += initial
            if keepdims:
                if out is not None and out.shape != (1, 1):
                    raise ValueError(
                        f"operand was set up as a reduction along axis 1, but the length of the axis is {out.shape[0]} (it has to be 1)"
                    )
                if out is not None:
                    out._update_inplace(
                        (
                            numpy.ones_like(out, dtype=out_dtype) * initial
                        )._query_compiler
                    )
                if truthy_where or out is not None:
                    return fix_dtypes_and_determine_return(
                        array(numpy.array([[result]], dtype=out_dtype))._query_compiler,
                        2,
                        dtype,
                        out,
                        truthy_where,
                    )
                else:
                    return array([[initial]], dtype=out_dtype)
            return result if truthy_where else initial
        if apply_axis > 1:
            raise numpy.AxisError(axis, 2)
        target = where.where(self, 0) if isinstance(where, array) else self
        if target.dtype != out_dtype:
            target = target.astype(out_dtype)
        result = target._query_compiler.sum(axis=apply_axis, skipna=False)
        if initial != 0:
            result = result.add(initial)
        new_ndim = self._ndim - 1 if not keepdims else self._ndim
        if new_ndim == 0:
            return result.to_numpy()[0, 0] if truthy_where else initial
        if not keepdims and apply_axis != 1:
            result = result.transpose()
        if out is not None:
            out._update_inplace(
                (numpy.ones_like(out, dtype=out_dtype) * initial)._query_compiler
            )
        if truthy_where or out is not None:
            return fix_dtypes_and_determine_return(
                result, new_ndim, dtype, out, truthy_where
            )
        else:
            return (
                numpy.zeros_like(array(_query_compiler=result, _ndim=new_ndim))
                + initial
            )

    def all(self, axis=None, out=None, keepdims=None, *, where=True):
        check_kwargs(keepdims=keepdims, where=where)
        truthy_where = bool(where)
        apply_axis = self._validate_axis(axis)
        target = where.where(self, True) if isinstance(where, array) else self
        if self._ndim == 1:
            if apply_axis == 1:
                raise numpy.AxisError(1, 1)
            result = target._query_compiler.all(axis=0)
            if keepdims:
                if out is not None and out.shape != (1,):
                    raise ValueError(
                        f"operand was set up as a reduction along axis 0, but the length of the axis is {out.shape[0]} (it has to be 1)"
                    )
                if truthy_where or out is not None:
                    return fix_dtypes_and_determine_return(
                        result, 1, bool, out, truthy_where
                    )
                else:
                    return array([True], dtype=bool)
            return result.to_numpy()[0, 0] if truthy_where else True
        if apply_axis is None:
            result = target._query_compiler.all(axis=1).all(axis=0)
            if keepdims:
                if out is not None and out.shape != (1, 1):
                    raise ValueError(
                        f"operand was set up as a reduction along axis 0, but the length of the axis is {out.shape[0]} (it has to be 1)"
                    )
                if truthy_where or out is not None:
                    return fix_dtypes_and_determine_return(
                        array(numpy.array([[result]], dtype=bool))._query_compiler,
                        2,
                        bool,
                        out,
                        truthy_where,
                    )
                else:
                    return array([[True]], dtype=bool)
            return result.to_numpy()[0, 0] if truthy_where else True
        if apply_axis > 1:
            raise numpy.AxisError(axis, 2)
        result = target._query_compiler.all(axis=apply_axis)
        new_ndim = self._ndim - 1 if not keepdims else self._ndim
        if new_ndim == 0:
            result = result.to_numpy()[0, 0]
            return result if truthy_where else True
        if not keepdims and apply_axis != 1:
            result = result.transpose()
        if truthy_where or out is not None:
            return fix_dtypes_and_determine_return(
                result, new_ndim, bool, out, truthy_where
            )
        else:
            return numpy.ones_like(array(_query_compiler=result, _ndim=new_ndim))

    _all = all

    def any(self, axis=None, out=None, keepdims=None, *, where=True):
        check_kwargs(keepdims=keepdims, where=where)
        truthy_where = bool(where)
        apply_axis = self._validate_axis(axis)
        target = where.where(self, False) if isinstance(where, array) else self
        if self._ndim == 1:
            if apply_axis == 1:
                raise numpy.AxisError(1, 1)
            result = target._query_compiler.any(axis=0)
            if keepdims:
                if out is not None and out.shape != (1,):
                    raise ValueError(
                        f"operand was set up as a reduction along axis 0, but the length of the axis is {out.shape[0]} (it has to be 1)"
                    )
                if truthy_where or out is not None:
                    return fix_dtypes_and_determine_return(
                        result, 1, bool, out, truthy_where
                    )
                else:
                    return array([False], dtype=bool)
            return result.to_numpy()[0, 0] if truthy_where else False
        if apply_axis is None:
            result = target._query_compiler.any(axis=1).any(axis=0)
            if keepdims:
                if out is not None and out.shape != (1, 1):
                    raise ValueError(
                        f"operand was set up as a reduction along axis 0, but the length of the axis is {out.shape[0]} (it has to be 1)"
                    )
                if truthy_where or out is not None:
                    return fix_dtypes_and_determine_return(
                        array(numpy.array([[result]], dtype=bool))._query_compiler,
                        2,
                        bool,
                        out,
                        truthy_where,
                    )
                else:
                    return array([[False]], dtype=bool)
            return result.to_numpy()[0, 0] if truthy_where else False
        if apply_axis > 1:
            raise numpy.AxisError(axis, 2)
        result = target._query_compiler.any(axis=apply_axis)
        new_ndim = self._ndim - 1 if not keepdims else self._ndim
        if new_ndim == 0:
            result = result.to_numpy()[0, 0]
            return result if truthy_where else False
        if not keepdims and apply_axis != 1:
            result = result.transpose()
        if truthy_where or out is not None:
            return fix_dtypes_and_determine_return(
                result, new_ndim, bool, out, truthy_where
            )
        else:
            return numpy.zeros_like(array(_query_compiler=result, _ndim=new_ndim))

    _any = any

    def argmax(self, axis=None, out=None, keepdims=None):
        check_kwargs(keepdims=keepdims)
        apply_axis = self._validate_axis(axis)
        if self._ndim == 1:
            if apply_axis == 1:
                raise numpy.AxisError(1, 1)
            if self._query_compiler.isna().any(axis=1).any(axis=0).to_numpy()[0, 0]:
                na_row_map = self._query_compiler.isna().any(axis=1)
                result = na_row_map.idxmax()
            else:
                result = self._query_compiler.idxmax(axis=0)
            if keepdims:
                if out is not None and out.shape != (1,):
                    raise ValueError(
                        f"operand was set up as a reduction along axis 0, but the length of the axis is {out.shape[0]} (it has to be 1)"
                    )
                return fix_dtypes_and_determine_return(
                    result, 1, numpy.int64, out, True
                )
            return result.to_numpy()[0, 0]
        if apply_axis is None:
            if self._query_compiler.isna().any(axis=1).any(axis=0).to_numpy()[0, 0]:
                na_row_map = self._query_compiler.isna().any(axis=1)
                na_row = self._query_compiler.getitem_array(na_row_map)
                col_idx = na_row.to_numpy().argmax()
                final_idxmax = na_row_map.idxmax().to_numpy().flatten()
            else:
                inner_idxs = self._query_compiler.idxmax(axis=1)
                final_idxmax = (
                    self._query_compiler.max(axis=1).idxmax(axis=0).to_numpy().flatten()
                )
                col_idx = inner_idxs.take_2d_positional(final_idxmax, [0]).to_numpy()[
                    0, 0
                ]
            result = (self.shape[1] * final_idxmax[0]) + col_idx
            if keepdims:
                if out is not None and out.shape != (1, 1):
                    raise ValueError(
                        f"operand was set up as a reduction along axis 0, but the length of the axis is {out.shape[0]} (it has to be 1)"
                    )
                return fix_dtypes_and_determine_return(
                    array(numpy.array([[result]], dtype=bool))._query_compiler,
                    2,
                    numpy.int64,
                    out,
                    True,
                )
            return result
        if apply_axis > 1:
            raise numpy.AxisError(axis, 2)
        result = self._query_compiler.idxmax(axis=apply_axis)
        na_mask = self._query_compiler.isna().any(axis=apply_axis)
        if na_mask.any(axis=apply_axis ^ 1).to_numpy()[0, 0]:
            na_idxs = self._query_compiler.isna().idxmax(axis=apply_axis)
            result = na_idxs.where(na_mask, result)
        new_ndim = self._ndim - 1 if not keepdims else self._ndim
        if new_ndim == 0:
            result = result.to_numpy()[0, 0]
            return result
        if not keepdims and apply_axis != 1:
            result = result.transpose()
        return fix_dtypes_and_determine_return(result, new_ndim, numpy.int64, out, True)

    def argmin(self, axis=None, out=None, keepdims=None):
        check_kwargs(keepdims=keepdims)
        apply_axis = self._validate_axis(axis)
        if self._ndim == 1:
            if apply_axis == 1:
                raise numpy.AxisError(1, 1)
            if self._query_compiler.isna().any(axis=1).any(axis=0).to_numpy()[0, 0]:
                na_row_map = self._query_compiler.isna().any(axis=1)
                # numpy apparently considers nan to be the minimum value in an array if present
                # therefore, we use idxmax on the mask array to identify where nans are
                result = na_row_map.idxmax()
            else:
                result = self._query_compiler.idxmin(axis=0)
            if keepdims:
                if out is not None and out.shape != (1,):
                    raise ValueError(
                        f"operand was set up as a reduction along axis 0, but the length of the axis is {out.shape[0]} (it has to be 1)"
                    )
                return fix_dtypes_and_determine_return(
                    result, 1, numpy.int64, out, True
                )
            return result.to_numpy()[0, 0]
        if apply_axis is None:
            if self._query_compiler.isna().any(axis=1).any(axis=0).to_numpy()[0, 0]:
                na_row_map = self._query_compiler.isna().any(axis=1)
                na_row = self._query_compiler.getitem_array(na_row_map)
                col_idx = na_row.to_numpy().argmax()
                final_idxmax = na_row_map.idxmax().to_numpy().flatten()
            else:
                inner_idxs = self._query_compiler.idxmin(axis=1)
                final_idxmax = (
                    self._query_compiler.min(axis=1).idxmin(axis=0).to_numpy().flatten()
                )
                col_idx = inner_idxs.take_2d_positional(final_idxmax, [0]).to_numpy()[
                    0, 0
                ]
            result = (self.shape[1] * final_idxmax[0]) + col_idx
            if keepdims:
                if out is not None and out.shape != (1, 1):
                    raise ValueError(
                        f"operand was set up as a reduction along axis 0, but the length of the axis is {out.shape[0]} (it has to be 1)"
                    )
                return fix_dtypes_and_determine_return(
                    array(numpy.array([[result]], dtype=bool))._query_compiler,
                    2,
                    numpy.int64,
                    out,
                    True,
                )
            return result
        if apply_axis > 1:
            raise numpy.AxisError(axis, 2)
        result = self._query_compiler.idxmin(axis=apply_axis)
        na_mask = self._query_compiler.isna().any(axis=apply_axis)
        if na_mask.any(axis=apply_axis ^ 1).to_numpy()[0, 0]:
            na_idxs = self._query_compiler.isna().idxmax(axis=apply_axis)
            result = na_idxs.where(na_mask, result)
        new_ndim = self._ndim - 1 if not keepdims else self._ndim
        if new_ndim == 0:
            result = result.to_numpy()[0, 0]
            return result
        if not keepdims and apply_axis != 1:
            result = result.transpose()
        return fix_dtypes_and_determine_return(result, new_ndim, numpy.int64, out, True)

    def _isfinite(
        self,
        /,
        out=None,
        *,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        check_kwargs(where=where, casting=casting, order=order, subok=subok)
        result = self._query_compiler._isfinite()
        return fix_dtypes_and_determine_return(result, self._ndim, dtype, out, where)

    def _isinf(
        self,
        /,
        out=None,
        *,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        check_kwargs(where=where, casting=casting, order=order, subok=subok)
        result = self._query_compiler._isinf()
        return fix_dtypes_and_determine_return(result, self._ndim, dtype, out, where)

    def _isnan(
        self,
        /,
        out=None,
        *,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        check_kwargs(where=where, casting=casting, order=order, subok=subok)
        result = self._query_compiler.isna()
        return fix_dtypes_and_determine_return(result, self._ndim, dtype, out, where)

    def _isnat(
        self,
        /,
        out=None,
        *,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        check_kwargs(where=where, casting=casting, order=order, subok=subok)
        result = self._query_compiler._isnat()
        return fix_dtypes_and_determine_return(result, self._ndim, dtype, out, where)

    def _isneginf(self, out=None):
        result = self._query_compiler._isneginf()
        return fix_dtypes_and_determine_return(result, self._ndim, out=out)

    def _isposinf(self, out=None):
        result = self._query_compiler._isposinf()
        return fix_dtypes_and_determine_return(result, self._ndim, out=out)

    def _iscomplex(self):
        result = self._query_compiler._iscomplex()
        return fix_dtypes_and_determine_return(result, self._ndim)

    def _isreal(self):
        result = self._query_compiler._isreal()
        return fix_dtypes_and_determine_return(result, self._ndim)

    def _logical_not(
        self,
        /,
        out=None,
        *,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        check_kwargs(where=where, casting=casting, order=order, subok=subok)
        result = self._query_compiler._logical_not()
        return fix_dtypes_and_determine_return(result, self._ndim, dtype, out, where)

    def _logical_binop(
        self, qc_method_name, x2, out, where, casting, order, dtype, subok
    ):
        check_kwargs(where=where, casting=casting, order=order, subok=subok)
        if self._ndim != x2._ndim:
            raise ValueError(
                "modin.numpy logic operators do not currently support broadcasting between arrays of different dimensions"
            )
        caller, callee, new_ndim, kwargs = self._preprocess_binary_op(
            x2, cast_input_types=False, dtype=dtype, out=out
        )
        # Deliberately do not pass **kwargs, since they're not used
        result = getattr(caller, qc_method_name)(callee)
        return fix_dtypes_and_determine_return(result, new_ndim, dtype, out, where)

    def _logical_and(
        self,
        x2,
        /,
        out=None,
        *,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        return self._logical_binop(
            "_logical_and", x2, out, where, casting, order, dtype, subok
        )

    def _logical_or(
        self,
        x2,
        /,
        out=None,
        *,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        return self._logical_binop(
            "_logical_or", x2, out, where, casting, order, dtype, subok
        )

    def _logical_xor(
        self,
        x2,
        /,
        out=None,
        *,
        where=True,
        casting="same_kind",
        order="K",
        dtype=None,
        subok=True,
    ):
        return self._logical_binop(
            "_logical_xor", x2, out, where, casting, order, dtype, subok
        )

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
        if isinstance(new_shape, int) or len(new_shape) == 1:
            self._update_inplace(self.flatten()._query_compiler)
            self._ndim = 1
        else:
            raise NotImplementedError(
                "Modin numpy does not currently support reshaping to a 2D object"
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
            return pandas.core.dtypes.cast.find_common_type(list(dtype.values))

    @property
    def size(self):
        return prod(self.shape)

    def __len__(self):
        return self.shape[0]

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

    def _build_repr_array(self):
        def _generate_indices_for_axis(
            axis_size, num_elements=numpy.get_printoptions()["edgeitems"]
        ):
            if axis_size > num_elements * 2:
                return list(range(num_elements + 1)) + list(
                    range(axis_size - num_elements, axis_size)
                )
            return list(range(axis_size))

        # We want to rely on NumPy for creating a string representation of this array; however
        # we also don't want to materialize all of the data to the head node. Instead, we will
        # materialize enough data that NumPy can build the summarized representation of the array
        # (while changing with the NumPy print options so it will format this smaller array as
        # abridged) and return this smaller array. In the worst case, this array will have
        # (2*numpy.get_printoptions()["edgeitems"] + 1)^2 items, so 49 items max for the default
        # value of 3.
        if self._ndim == 1 or self.shape[1] == 0:
            idxs = _generate_indices_for_axis(len(self))
            arr = self._query_compiler.getitem_row_array(idxs).to_numpy()
            if self._ndim == 1:
                arr = arr.flatten()
        elif self.shape[0] == 1:
            idxs = _generate_indices_for_axis(self.shape[1])
            arr = self._query_compiler.getitem_column_array(idxs).to_numpy()
        else:
            row_idxs = _generate_indices_for_axis(len(self))
            col_idxs = _generate_indices_for_axis(self.shape[1])
            arr = self._query_compiler.take_2d_positional(row_idxs, col_idxs).to_numpy()
        return arr

    def __repr__(self):
        # If we are dealing with a small array, we can just collate all the data on the
        # head node and let numpy handle the logic to get a string representation.
        if self.size <= numpy.get_printoptions()["threshold"]:
            return repr(self._to_numpy())
        arr = self._build_repr_array()
        prev_threshold = numpy.get_printoptions()["threshold"]
        numpy.set_printoptions(threshold=arr.size - 1)
        try:
            repr_str = repr(arr)
        finally:
            numpy.set_printoptions(threshold=prev_threshold)
        return repr_str

    def _to_numpy(self):
        arr = self._query_compiler.to_numpy()
        if self._ndim == 1:
            arr = arr.flatten()
        return arr
