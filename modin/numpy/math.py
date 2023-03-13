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

from .arr import array
from .utils import try_convert_from_interoperable_type
from modin.error_message import ErrorMessage


def absolute(
    x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True
):
    x = try_convert_from_interoperable_type(x)
    if not isinstance(x, array):
        ErrorMessage.bad_type_for_numpy_op("absolute", type(x))
        return numpy.absolute(
            x,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )
    return x.absolute(
        out=out, where=where, casting=casting, order=order, dtype=dtype, subok=subok
    )


abs = absolute


def add(
    x1, x2, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True
):
    x1 = try_convert_from_interoperable_type(x1)
    if not isinstance(x1, array):
        ErrorMessage.bad_type_for_numpy_op("add", type(x1))
        return numpy.add(
            x1,
            x2,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )
    return x1.__add__(
        x2,
        out=out,
        where=where,
        casting=casting,
        order=order,
        dtype=dtype,
        subok=subok,
    )


def divide(
    x1, x2, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True
):
    x1 = try_convert_from_interoperable_type(x1)
    if not isinstance(x1, array):
        ErrorMessage.bad_type_for_numpy_op("divide", type(x1))
        return numpy.divide(
            x1,
            x2,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )
    return x1.divide(
        x2,
        out=out,
        where=where,
        casting=casting,
        order=order,
        dtype=dtype,
        subok=subok,
    )


def dot(a, b, out=None):
    a = try_convert_from_interoperable_type(a)
    if not isinstance(a, array):
        ErrorMessage.bad_type_for_numpy_op("dot", type(a))
        return numpy.dot(a, b, out=out)
    return a.dot(b, out=out)


def float_power(
    x1, x2, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True
):
    x1 = try_convert_from_interoperable_type(x1)
    if not isinstance(x1, array):
        ErrorMessage.bad_type_for_numpy_op("float_power", type(x1))
        return numpy.float_power(
            x1,
            x2,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )
    return x1.float_power(
        x2,
        out=out,
        where=where,
        casting=casting,
        order=order,
        dtype=dtype,
        subok=subok,
    )


def floor_divide(
    x1, x2, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True
):
    x1 = try_convert_from_interoperable_type(x1)
    if not isinstance(x1, array):
        ErrorMessage.bad_type_for_numpy_op("floor_divide", type(x1))
        return numpy.floor_divide(
            x1,
            x2,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )
    return x1.floor_divide(
        x2,
        out=out,
        where=where,
        casting=casting,
        order=order,
        dtype=dtype,
        subok=subok,
    )


def power(
    x1, x2, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True
):
    x1 = try_convert_from_interoperable_type(x1)
    if not isinstance(x1, array):
        ErrorMessage.bad_type_for_numpy_op("power", type(x1))
        return numpy.power(
            x1,
            x2,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )
    return x1.power(
        x2,
        out=out,
        where=where,
        casting=casting,
        order=order,
        dtype=dtype,
        subok=subok,
    )


def prod(a, axis=None, out=None, keepdims=None, where=True, dtype=None, initial=None):
    a = try_convert_from_interoperable_type(a)
    if not isinstance(a, array):
        ErrorMessage.bad_type_for_numpy_op("prod", type(a))
        return numpy.prod(
            a,
            axis=axis,
            out=out,
            keepdims=keepdims,
            where=where,
            dtype=dtype,
            initial=initial,
        )
    return a.prod(
        axis=axis, out=out, keepdims=keepdims, where=where, dtype=dtype, initial=initial
    )


def multiply(
    x1, x2, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True
):
    x1 = try_convert_from_interoperable_type(x1)
    if not isinstance(x1, array):
        ErrorMessage.bad_type_for_numpy_op("multiply", type(x1))
        return numpy.multiply(
            x1,
            x2,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )
    return x1.multiply(
        x2,
        out=out,
        where=where,
        casting=casting,
        order=order,
        dtype=dtype,
        subok=subok,
    )


def remainder(
    x1, x2, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True
):
    x1 = try_convert_from_interoperable_type(x1)
    if not isinstance(x1, array):
        ErrorMessage.bad_type_for_numpy_op("remainder", type(x1))
        return numpy.remainder(
            x1,
            x2,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )
    return x1.remainder(
        x2,
        out=out,
        where=where,
        casting=casting,
        order=order,
        dtype=dtype,
        subok=subok,
    )


mod = remainder


def subtract(
    x1, x2, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True
):
    x1 = try_convert_from_interoperable_type(x1)
    if not isinstance(x1, array):
        ErrorMessage.bad_type_for_numpy_op("subtract", type(x1))
        return numpy.subtract(
            x1,
            x2,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )
    return x1.subtract(
        x2,
        out=out,
        where=where,
        casting=casting,
        order=order,
        dtype=dtype,
        subok=subok,
    )


def sum(arr, axis=None, dtype=None, out=None, keepdims=None, initial=None, where=True):
    arr = try_convert_from_interoperable_type(arr)
    if not isinstance(arr, array):
        ErrorMessage.bad_type_for_numpy_op("sum", type(arr))
        return numpy.sum(
            arr,
            axis=axis,
            out=out,
            keepdims=keepdims,
            where=where,
            dtype=dtype,
            initial=initial,
        )
    return arr.sum(
        axis=axis, out=out, keepdims=keepdims, where=where, dtype=dtype, initial=initial
    )


def true_divide(
    x1, x2, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True
):
    x1 = try_convert_from_interoperable_type(x1)
    if not isinstance(x1, array):
        ErrorMessage.bad_type_for_numpy_op("true_divide", type(x1))
        return numpy.true_divide(
            x1,
            x2,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )
    return x1.divide(
        x2,
        out=out,
        where=where,
        casting=casting,
        order=order,
        dtype=dtype,
        subok=subok,
    )


def mean(x1, axis=None, dtype=None, out=None, keepdims=None, *, where=True):
    x1 = try_convert_from_interoperable_type(x1)
    if not isinstance(x1, array):
        ErrorMessage.bad_type_for_numpy_op("mean", type(x1))
        return numpy.mean(
            x1, axis=axis, out=out, keepdims=keepdims, where=where, dtype=dtype
        )
    return x1.mean(axis=axis, out=out, keepdims=keepdims, where=where, dtype=dtype)


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


def amax(x1, axis=None, out=None, keepdims=None, initial=None, where=True):
    x1 = try_convert_from_interoperable_type(x1)
    if not isinstance(x1, array):
        ErrorMessage.bad_type_for_numpy_op("amax", type(x1))
        return numpy.amax(
            x1, axis=axis, out=out, keepdims=keepdims, initial=initial, where=where
        )
    return x1.max(axis=axis, out=out, keepdims=keepdims, initial=initial, where=where)


max = amax


def amin(x1, axis=None, out=None, keepdims=None, initial=None, where=True):
    x1 = try_convert_from_interoperable_type(x1)
    if not isinstance(x1, array):
        ErrorMessage.bad_type_for_numpy_op("amin", type(x1))
        return numpy.amin(
            x1, axis=axis, out=out, keepdims=keepdims, initial=initial, where=where
        )
    return x1.min(axis=axis, out=out, keepdims=keepdims, initial=initial, where=where)


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
