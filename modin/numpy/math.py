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


def absolute(
    x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True
):
    if hasattr(x, "_absolute"):
        return x._absolute(
            out=out, where=where, casting=casting, order=order, dtype=dtype, subok=subok
        )
    return numpy.absolute(
        x, out=out, where=where, casting=casting, order=order, dtype=dtype, subok=subok
    )


abs = absolute


def add(
    x1, x2, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True
):
    if hasattr(x1, "_add"):
        return x1._add(
            x2,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
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


def all(a, axis=None, out=None, keepdims=None, where=None):
    if hasattr(a, "_all"):
        return a._all(axis=axis, out=out, keepdims=keepdims, where=where)
    return numpy.all(a, axis=axis, out=out, keepdims=keepdims, where=where)


def divide(
    x1, x2, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True
):
    if hasattr(x1, "_divide"):
        return x1._divide(
            x2,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
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


def float_power(
    x1, x2, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True
):
    if hasattr(x1, "_float_power"):
        return x1._float_power(
            x2,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
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


def floor_divide(
    x1, x2, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True
):
    if hasattr(x1, "_floor_divide"):
        return x1._floor_divide(
            x2,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
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


def power(
    x1, x2, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True
):
    if hasattr(x1, "_power"):
        return x1._power(
            x2,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
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


def prod(a, axis=None, out=None, keepdims=None, where=None):
    if hasattr(a, "_prod"):
        return a._prod(axis=axis, out=out, keepdims=keepdims, where=where)
    return numpy.prod(a, axis=axis, out=out, keepdims=keepdims, where=where)


def multiply(
    x1, x2, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True
):
    if hasattr(x1, "_multiply"):
        return x1._multiply(
            x2,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
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


def remainder(
    x1, x2, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True
):
    if hasattr(x1, "_remainder"):
        return x1._remainder(
            x2,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
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


mod = remainder


def subtract(
    x1, x2, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True
):
    if hasattr(x1, "_subtract"):
        return x1._subtract(
            x2,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
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


def sum(arr, axis=None, dtype=None, out=None, keepdims=None, initial=None, where=None):
    if hasattr(arr, "_sum"):
        return arr._sum(axis)
    else:
        return numpy.sum(arr)


def true_divide(
    x1, x2, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True
):
    if hasattr(x1, "_divide"):
        return x1._divide(
            x2,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
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
