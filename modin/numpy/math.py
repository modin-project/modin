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

from modin.error_message import ErrorMessage
from modin.utils import _inherit_docstrings

from .arr import array
from .utils import try_convert_from_interoperable_type


def _dispatch_math(operator_name, arr_method_name=None):
    # `operator_name` is the name of the method on the numpy API
    # `arr_method_name` is the name of the method on the modin.numpy.array object,
    # which is assumed to be `operator_name` by default
    @_inherit_docstrings(getattr(numpy, operator_name))
    def call(x, *args, **kwargs):
        x = try_convert_from_interoperable_type(x)
        if not isinstance(x, array):
            ErrorMessage.bad_type_for_numpy_op(operator_name, type(x))
            return getattr(numpy, operator_name)(x, *args, **kwargs)

        return getattr(x, arr_method_name or operator_name)(*args, **kwargs)

    return call


absolute = _dispatch_math("absolute")
abs = absolute
add = _dispatch_math("add", "__add__")
divide = _dispatch_math("divide")
dot = _dispatch_math("dot")
float_power = _dispatch_math("float_power")
floor_divide = _dispatch_math("floor_divide")
power = _dispatch_math("power")
prod = _dispatch_math("prod")
multiply = _dispatch_math("multiply")
remainder = _dispatch_math("remainder")
mod = remainder
subtract = _dispatch_math("subtract")
sum = _dispatch_math("sum")
true_divide = _dispatch_math("true_divide", "divide")
mean = _dispatch_math("mean")


def var(x1, axis=None, dtype=None, out=None, keepdims=None, *, where=True):
    x1 = try_convert_from_interoperable_type(x1)
    if not isinstance(x1, array):
        ErrorMessage.bad_type_for_numpy_op("var", type(x1))
        return numpy.var(
            x1, axis=axis, out=out, keepdims=keepdims, where=where, dtype=dtype
        )
    return x1.var(axis=axis, out=out, keepdims=keepdims, where=where, dtype=dtype)


# Maximum and minimum are ufunc's in NumPy, which means that our array's __array_ufunc__
# implementation will automatically handle this. We still need the function though, so that
# if the operands are modin.pandas objects, we can convert them to arrays, but after that
# we can just use NumPy's maximum/minimum since that will route to our array's ufunc.
def maximum(
    x1, x2, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True
):
    x1 = try_convert_from_interoperable_type(x1)
    if not isinstance(x1, array):
        ErrorMessage.bad_type_for_numpy_op("maximum", type(x1))
    return numpy.maximum(
        x1,
        x2,
        out=out,
        where=where,
        casting=casting,
        order=order,
        dtype=dtype,
        subok=subok,
    )


def minimum(
    x1, x2, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True
):
    x1 = try_convert_from_interoperable_type(x1)
    if not isinstance(x1, array):
        ErrorMessage.bad_type_for_numpy_op("minimum", type(x1))
    return numpy.minimum(
        x1,
        x2,
        out=out,
        where=where,
        casting=casting,
        order=order,
        dtype=dtype,
        subok=subok,
    )


amax = _dispatch_math("amax", "max")
amin = _dispatch_math("amin", "min")
max = amax
min = amin


def sqrt(
    x, out=None, *, where=True, casting="same_kind", order="K", dtype=None, subok=True
):
    x = try_convert_from_interoperable_type(x)
    if not isinstance(x, array):
        ErrorMessage.bad_type_for_numpy_op("sqrt", type(x))
        return numpy.sqrt(
            x,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )
    return x.sqrt(out, where, casting, order, dtype, subok)


def exp(
    x, out=None, *, where=True, casting="same_kind", order="K", dtype=None, subok=True
):
    x = try_convert_from_interoperable_type(x)
    if not isinstance(x, array):
        ErrorMessage.bad_type_for_numpy_op("exp", type(x))
        return numpy.exp(
            x,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )
    return x.exp(out, where, casting, order, dtype, subok)


def argmax(a, axis=None, out=None, *, keepdims=None):
    a = try_convert_from_interoperable_type(a)
    if not isinstance(a, array):
        ErrorMessage.bad_type_for_numpy_op("argmax", type(a))
        return numpy.argmax(a, axis=axis, out=out, keepdims=keepdims)
    return a.argmax(axis=axis, out=out, keepdims=keepdims)


def argmin(a, axis=None, out=None, *, keepdims=None):
    a = try_convert_from_interoperable_type(a)
    if not isinstance(a, array):
        ErrorMessage.bad_type_for_numpy_op("argmin", type(a))
        return numpy.argmin(a, axis=axis, out=out, keepdims=keepdims)
    return a.argmin(axis=axis, out=out, keepdims=keepdims)
