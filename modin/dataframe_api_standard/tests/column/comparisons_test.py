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


from __future__ import annotations

from typing import Any

import pytest

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_column_with_reference,
    integer_dataframe_1,
    integer_dataframe_7,
)


@pytest.mark.parametrize(
    ("comparison", "expected_data", "expected_dtype"),
    [
        ("__eq__", [True, True, False], "Bool"),
        ("__ne__", [False, False, True], "Bool"),
        ("__ge__", [True, True, False], "Bool"),
        ("__gt__", [False, False, False], "Bool"),
        ("__le__", [True, True, True], "Bool"),
        ("__lt__", [False, False, True], "Bool"),
        ("__add__", [2, 4, 7], "Int64"),
        ("__sub__", [0, 0, -1], "Int64"),
        ("__mul__", [1, 4, 12], "Int64"),
        ("__truediv__", [1, 1, 0.75], "Float64"),
        ("__floordiv__", [1, 1, 0], "Int64"),
        ("__pow__", [1, 4, 81], "Int64"),
        ("__mod__", [0, 0, 3], "Int64"),
    ],
)
def test_column_comparisons(
    library: BaseHandler,
    comparison: str,
    expected_data: list[object],
    expected_dtype: str,
) -> None:
    ser: Any
    df = integer_dataframe_7(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    other = df.col("b")
    result = df.assign(getattr(ser, comparison)(other).rename("result"))
    expected_ns_dtype = getattr(ns, expected_dtype)
    compare_column_with_reference(
        result.col("result"), expected_data, expected_ns_dtype
    )


@pytest.mark.parametrize(
    ("comparison", "expected_data", "expected_dtype"),
    [
        ("__eq__", [False, False, True], "Bool"),
        ("__ne__", [True, True, False], "Bool"),
        ("__ge__", [False, False, True], "Bool"),
        ("__gt__", [False, False, False], "Bool"),
        ("__le__", [True, True, True], "Bool"),
        ("__lt__", [True, True, False], "Bool"),
        ("__add__", [4, 5, 6], "Int64"),
        ("__sub__", [-2, -1, 0], "Int64"),
        ("__mul__", [3, 6, 9], "Int64"),
        ("__truediv__", [1 / 3, 2 / 3, 1], "Float64"),
        ("__floordiv__", [0, 0, 1], "Int64"),
        ("__pow__", [1, 8, 27], "Int64"),
        ("__mod__", [1, 2, 0], "Int64"),
    ],
)
def test_column_comparisons_scalar(
    library: BaseHandler,
    comparison: str,
    expected_data: list[object],
    expected_dtype: str,
) -> None:
    ser: Any
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    other = 3
    result = df.assign(getattr(ser, comparison)(other).rename("result"))
    expected_ns_dtype = getattr(ns, expected_dtype)
    compare_column_with_reference(
        result.col("result"), expected_data, expected_ns_dtype
    )


@pytest.mark.parametrize(
    ("comparison", "expected_data"),
    [
        ("__radd__", [3, 4, 5]),
        ("__rsub__", [1, 0, -1]),
        ("__rmul__", [2, 4, 6]),
    ],
)
def test_right_column_comparisons(
    library: BaseHandler,
    comparison: str,
    expected_data: list[object],
) -> None:
    # 1,2,3
    ser: Any
    df = integer_dataframe_7(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    other = 2
    result = df.assign(getattr(ser, comparison)(other).rename("result"))
    compare_column_with_reference(result.col("result"), expected_data, dtype=ns.Int64)
