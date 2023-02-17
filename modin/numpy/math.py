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
        ErrorMessage.single_warning(
            f"Modin NumPy only supports objects of modin.numpy.array types for absolute, not {type(x)}. Defaulting to NumPy."
        )
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
        ErrorMessage.single_warning(
            f"Modin NumPy only supports objects of modin.numpy.array types for add, not {type(x1)}. Defaulting to NumPy."
        )
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
        ErrorMessage.single_warning(
            f"Modin NumPy only supports objects of modin.numpy.array types for divide, not {type(x1)}. Defaulting to NumPy."
        )
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


def float_power(
    x1, x2, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True
):
    x1 = try_convert_from_interoperable_type(x1)
    if not isinstance(x1, array):
        ErrorMessage.single_warning(
            f"Modin NumPy only supports objects of modin.numpy.array types for float_power, not {type(x1)}. Defaulting to NumPy."
        )
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
        ErrorMessage.single_warning(
            f"Modin NumPy only supports objects of modin.numpy.array types for floor_divide, not {type(x1)}. Defaulting to NumPy."
        )
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
        ErrorMessage.single_warning(
            f"Modin NumPy only supports objects of modin.numpy.array types for power, not {type(x1)}. Defaulting to NumPy."
        )
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
        ErrorMessage.single_warning(
            f"Modin NumPy only supports objects of modin.numpy.array types for prod, not {type(a)}. Defaulting to NumPy."
        )
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
        ErrorMessage.single_warning(
            f"Modin NumPy only supports objects of modin.numpy.array types for multiply, not {type(x1)}. Defaulting to NumPy."
        )
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
        ErrorMessage.single_warning(
            f"Modin NumPy only supports objects of modin.numpy.array types for remainder, not {type(x1)}. Defaulting to NumPy."
        )
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
        ErrorMessage.single_warning(
            f"Modin NumPy only supports objects of modin.numpy.array types for power, not {type(x1)}. Defaulting to NumPy."
        )
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
        ErrorMessage.single_warning(
            f"Modin NumPy only supports objects of modin.numpy.array types for sum, not {type(arr)}. Defaulting to NumPy."
        )
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
        ErrorMessage.single_warning(
            f"Modin NumPy only supports objects of modin.numpy.array types for true_divide, not {type(x1)}. Defaulting to NumPy."
        )
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
        ErrorMessage.single_warning(
            f"Modin NumPy only supports objects of modin.numpy.array types for mean, not {type(x1)}. Defaulting to NumPy."
        )
        return numpy.mean(
            x1, axis=axis, out=out, keepdims=keepdims, where=where, dtype=dtype
        )
    return x1.mean(axis=axis, out=out, keepdims=keepdims, where=where, dtype=dtype)


# Maximum and minimum are ufunc's in NumPy, which means that our array's __array_ufunc__
# implementation will automatically handle this, so we can just use NumPy's maximum/minimum
# since that will route to our array's ufunc.
def maximum(
    x1, x2, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True
):
    x1 = try_convert_from_interoperable_type(x1)
    if not isinstance(x1, array):
        ErrorMessage.single_warning(
            f"Modin NumPy only supports objects of modin.numpy.array types for maximum, not {type(x1)}. Defaulting to NumPy."
        )
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
        ErrorMessage.single_warning(
            f"Modin NumPy only supports objects of modin.numpy.array types for minimum, not {type(x1)}. Defaulting to NumPy."
        )
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
        ErrorMessage.single_warning(
            f"Modin NumPy only supports objects of modin.numpy.array types for amax, not {type(x1)}. Defaulting to NumPy."
        )
        return numpy.amax(
            x1, axis=axis, out=out, keepdims=keepdims, initial=initial, where=where
        )
    return x1.max(axis=axis, out=out, keepdims=keepdims, initial=initial, where=where)


max = amax


def amin(x1, axis=None, out=None, keepdims=None, initial=None, where=True):
    x1 = try_convert_from_interoperable_type(x1)
    if not isinstance(x1, array):
        ErrorMessage.single_warning(
            f"Modin NumPy only supports objects of modin.numpy.array types for amin, not {type(x1)}. Defaulting to NumPy."
        )
        return numpy.amin(
            x1, axis=axis, out=out, keepdims=keepdims, initial=initial, where=where
        )
    return x1.min(axis=axis, out=out, keepdims=keepdims, initial=initial, where=where)


min = amin
