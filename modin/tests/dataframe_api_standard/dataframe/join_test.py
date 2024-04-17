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
    integer_dataframe_1,
    integer_dataframe_2,
)
from modin.tests.pandas.utils import default_to_pandas_ignore_string

pytestmark = pytest.mark.filterwarnings(default_to_pandas_ignore_string)


def test_join_left(library: BaseHandler) -> None:
    if library.name == "modin":
        pytest.skip("TODO: enable for modin")
    left = integer_dataframe_1(library)
    right = integer_dataframe_2(library).rename({"b": "c"})
    result = left.join(right, left_on="a", right_on="a", how="left")
    ns = result.__dataframe_namespace__()
    expected = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [4.0, 2.0, float("nan")]}
    expected_dtype = {
        "a": ns.Int64,
        "b": ns.Int64,
        "c": ns.Float64,
    }
    compare_dataframe_with_reference(result, expected, dtype=expected_dtype)  # type: ignore[arg-type]


def test_join_overlapping_names(library: BaseHandler) -> None:
    left = integer_dataframe_1(library)
    right = integer_dataframe_2(library)
    with pytest.raises(ValueError):
        _ = left.join(right, left_on="a", right_on="a", how="left")


def test_join_inner(library: BaseHandler) -> None:
    left = integer_dataframe_1(library)
    right = integer_dataframe_2(library).rename({"b": "c"})
    result = left.join(right, left_on="a", right_on="a", how="inner")
    ns = result.__dataframe_namespace__()
    expected = {"a": [1, 2], "b": [4, 5], "c": [4, 2]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)


def test_join_outer(library: BaseHandler) -> None:  # pragma: no cover
    left = integer_dataframe_1(library)
    right = integer_dataframe_2(library).rename({"b": "c"})
    result = left.join(right, left_on="a", right_on="a", how="outer").sort("a")
    ns = result.__dataframe_namespace__()
    expected = {
        "a": [1, 2, 3, 4],
        "b": [4, 5, 6, float("nan")],
        "c": [4.0, 2.0, float("nan"), 6.0],
    }
    expected_dtype = {
        "a": ns.Int64,
        "b": ns.Float64,
        "c": ns.Float64,
    }
    compare_dataframe_with_reference(result, expected, dtype=expected_dtype)  # type: ignore[arg-type]


def test_join_two_keys(library: BaseHandler) -> None:
    if library.name == "modin":
        pytest.skip("TODO: enable for modin")
    left = integer_dataframe_1(library)
    right = integer_dataframe_2(library).rename({"b": "c"})
    result = left.join(right, left_on=["a", "b"], right_on=["a", "c"], how="left")
    ns = result.__dataframe_namespace__()
    expected = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [4.0, float("nan"), float("nan")]}
    expected_dtype = {
        "a": ns.Int64,
        "b": ns.Int64,
        "c": ns.Float64,
    }
    compare_dataframe_with_reference(result, expected, dtype=expected_dtype)  # type: ignore[arg-type]


def test_join_invalid(library: BaseHandler) -> None:
    left = integer_dataframe_1(library)
    right = integer_dataframe_2(library).rename({"b": "c"})
    with pytest.raises(ValueError):
        left.join(right, left_on=["a", "b"], right_on=["a", "c"], how="right")  # type: ignore  # noqa: PGH003
