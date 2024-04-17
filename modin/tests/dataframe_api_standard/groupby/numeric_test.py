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

import pytest

from modin.tests.dataframe_api_standard.utils import (
    BaseHandler,
    compare_dataframe_with_reference,
    integer_dataframe_4,
)


@pytest.mark.parametrize(
    ("aggregation", "expected_b", "expected_c", "expected_dtype"),
    [
        ("min", [1, 3], [4, 6], "Int64"),
        ("max", [2, 4], [5, 7], "Int64"),
        ("sum", [3, 7], [9, 13], "Int64"),
        ("prod", [2, 12], [20, 42], "Int64"),
        ("median", [1.5, 3.5], [4.5, 6.5], "Float64"),
        ("mean", [1.5, 3.5], [4.5, 6.5], "Float64"),
        (
            "std",
            [0.7071067811865476, 0.7071067811865476],
            [0.7071067811865476, 0.7071067811865476],
            "Float64",
        ),
        ("var", [0.5, 0.5], [0.5, 0.5], "Float64"),
    ],
)
def test_group_by_numeric(
    library: BaseHandler,
    aggregation: str,
    expected_b: list[float],
    expected_c: list[float],
    expected_dtype: str,
) -> None:
    df = integer_dataframe_4(library)
    ns = df.__dataframe_namespace__()
    result = getattr(df.group_by("key"), aggregation)()
    result = result.sort("key")
    expected = {"key": [1, 2], "b": expected_b, "c": expected_c}
    dtype = getattr(ns, expected_dtype)
    expected_ns_dtype = {"key": ns.Int64, "b": dtype, "c": dtype}
    compare_dataframe_with_reference(result, expected, dtype=expected_ns_dtype)  # type: ignore[arg-type]
