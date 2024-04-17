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


# MIT License

# Copyright (c) 2023, Marco Gorelli

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import numpy as np
import pytest

from modin.tests.dataframe_api_standard.utils import (
    BaseHandler,
    compare_dataframe_with_reference,
    integer_dataframe_1,
    integer_dataframe_2,
)


@pytest.mark.parametrize(
    "attr",
    [
        "__lt__",
        "__le__",
        "__eq__",
        "__ne__",
        "__gt__",
        "__ge__",
        "__add__",
        "__radd__",
        "__sub__",
        "__rsub__",
        "__mul__",
        "__rmul__",
        "__mod__",
        "__rmod__",
        "__pow__",
        "__rpow__",
        "__floordiv__",
        "__rfloordiv__",
        "__truediv__",
        "__rtruediv__",
    ],
)
def test_float_binary(library: BaseHandler, attr: str) -> None:
    other = 0.5
    df = integer_dataframe_2(library).persist()
    scalar = df.col("a").mean()
    float_scalar = float(scalar)  # type: ignore[arg-type]
    assert getattr(scalar, attr)(other) == getattr(float_scalar, attr)(other)


def test_float_binary_invalid(library: BaseHandler) -> None:
    lhs = integer_dataframe_2(library).col("a").mean()
    rhs = integer_dataframe_1(library).col("b").mean()
    with pytest.raises(ValueError):
        _ = lhs > rhs


def test_float_binary_lazy_valid(library: BaseHandler) -> None:
    df = integer_dataframe_2(library).persist()
    lhs = df.col("a").mean()
    rhs = df.col("b").mean()
    result = lhs > rhs
    assert not bool(result)


@pytest.mark.parametrize(
    "attr",
    [
        "__abs__",
        "__neg__",
    ],
)
def test_float_unary(library: BaseHandler, attr: str) -> None:
    df = integer_dataframe_2(library).persist()
    with pytest.warns(UserWarning):
        scalar = df.col("a").persist().mean()
    float_scalar = float(scalar)  # type: ignore[arg-type]
    assert getattr(scalar, attr)() == getattr(float_scalar, attr)()


@pytest.mark.parametrize(
    "attr",
    [
        "__int__",
        "__float__",
        "__bool__",
    ],
)
def test_float_unary_invalid(library: BaseHandler, attr: str) -> None:
    df = integer_dataframe_2(library)
    scalar = df.col("a").mean()
    float_scalar = float(scalar.persist())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError):
        assert getattr(scalar, attr)() == getattr(float_scalar, attr)()


def test_free_standing(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    ser = namespace.column_from_1d_array(  # type: ignore[call-arg]
        np.array([1, 2, 3]),
        name="a",
    )
    result = float(ser.mean() + 1)  # type: ignore[arg-type]
    assert result == 3.0


def test_right_comparand(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    col = df.col("a")  # [1, 2, 3]
    scalar = df.col("b").get_value(0)  # 4
    result = df.assign((scalar - col).rename("c"))
    expected = {
        "a": [1, 2, 3],
        "b": [4, 5, 6],
        "c": [3, 2, 1],
    }
    compare_dataframe_with_reference(result, expected, ns.Int64)
