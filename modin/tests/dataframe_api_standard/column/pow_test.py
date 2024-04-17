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

from modin.tests.dataframe_api_standard.utils import (
    BaseHandler,
    compare_dataframe_with_reference,
    integer_dataframe_1,
)


def test_float_powers_column(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    other = df.col("b") * 1.0
    result = df.assign(ser.__pow__(other).rename("result"))
    expected = {"a": [1, 2, 3], "b": [4, 5, 6], "result": [1.0, 32.0, 729.0]}
    expected_dtype = {"a": ns.Int64, "b": ns.Int64, "result": ns.Float64}
    compare_dataframe_with_reference(result, expected, expected_dtype)  # type: ignore[arg-type]


def test_float_powers_scalar_column(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    other = 1.0
    result = df.assign(ser.__pow__(other).rename("result"))
    expected = {"a": [1, 2, 3], "b": [4, 5, 6], "result": [1.0, 2.0, 3.0]}
    expected_dtype = {"a": ns.Int64, "b": ns.Int64, "result": ns.Float64}
    compare_dataframe_with_reference(result, expected, expected_dtype)  # type: ignore[arg-type]


def test_int_powers_column(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    other = df.col("b") * 1
    result = df.assign(ser.__pow__(other).rename("result"))
    expected = {"a": [1, 2, 3], "b": [4, 5, 6], "result": [1, 32, 729]}
    expected_dtype = {name: ns.Int64 for name in ("a", "b", "result")}
    compare_dataframe_with_reference(result, expected, expected_dtype)


def test_int_powers_scalar_column(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    other = 1
    result = df.assign(ser.__pow__(other).rename("result"))
    expected = {"a": [1, 2, 3], "b": [4, 5, 6], "result": [1, 2, 3]}
    expected_dtype = {name: ns.Int64 for name in ("a", "b", "result")}
    compare_dataframe_with_reference(result, expected, expected_dtype)
