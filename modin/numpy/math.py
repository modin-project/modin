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


def absolute(
    x, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True
):
    if hasattr(x, "absolute"):
        return x.absolute(
            out=out, where=where, casting=casting, order=order, dtype=dtype, subok=subok
        )


abs = absolute


def add(
    x1, x2, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True
):
    if hasattr(x1, "add"):
        return x1.add(
            x2,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )


def all(a, axis=None, out=None, keepdims=None, where=None):
    if hasattr(a, "all"):
        return a.all(axis=axis, out=out, keepdims=keepdims, where=where)


def divide(
    x1, x2, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True
):
    if hasattr(x1, "divide"):
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
    if hasattr(x1, "float_power"):
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
    if hasattr(x1, "floor_divide"):
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
    if hasattr(x1, "power"):
        return x1.power(
            x2,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )


def prod(a, axis=None, out=None, keepdims=None, where=None):
    if hasattr(a, "prod"):
        return a.prod(axis=axis, out=out, keepdims=keepdims, where=where)


def multiply(
    x1, x2, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True
):
    if hasattr(x1, "multiply"):
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
    if hasattr(x1, "remainder"):
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
    if hasattr(x1, "subtract"):
        return x1.subtract(
            x2,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )


def sum(arr, axis=None, dtype=None, out=None, keepdims=None, initial=None, where=None):
    if hasattr(arr, "sum"):
        return arr.sum(axis)


def true_divide(
    x1, x2, out=None, where=True, casting="same_kind", order="K", dtype=None, subok=True
):
    if hasattr(x1, "divide"):
        return x1.divide(
            x2,
            out=out,
            where=where,
            casting=casting,
            order=order,
            dtype=dtype,
            subok=subok,
        )


def ravel(arr, order="C"):
    if hasattr(arr, "flatten"):
        return arr.flatten(order)


# def where(condition, *, x=None, y=None):
#     condition = condition(self) if callable else condition
#     if not isinstance(condition, array):
#         if isinstance(condition, _INTEROPERABLE_TYPES):
#             condition = array(_query_compiler=condition._query_compiler, _ndim = 2 if isinstance(condition, pd.DataFrame) else 1)
#         elif is_list_like(condition):
#             condition = array(condition)
#     if not (is_scalar(condition) or isinstance(condition, (bool, array))):
#         raise ValueError(f"np.where does not support conditionals of type {type(condition)}")
