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

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_dataframe_with_reference,
    integer_dataframe_6,
)


def test_column_sorted_indices_ascending(library: BaseHandler) -> None:
    df = integer_dataframe_6(library)
    ns = df.__dataframe_namespace__()
    sorted_indices = df.col("b").sorted_indices()
    result = df.assign(sorted_indices.rename("result"))
    expected_1 = {
        "a": [1, 1, 1, 2, 2],
        "b": [4, 4, 3, 1, 2],
        "result": [3, 4, 2, 0, 1],
    }
    expected_2 = {
        "a": [1, 1, 1, 2, 2],
        "b": [4, 4, 3, 1, 2],
        "result": [3, 4, 2, 1, 0],
    }
    try:
        compare_dataframe_with_reference(result, expected_1, dtype=ns.Int64)
    except AssertionError:  # pragma: no cover
        # order isn't determinist, so try both
        compare_dataframe_with_reference(result, expected_2, dtype=ns.Int64)


def test_column_sorted_indices_descending(library: BaseHandler) -> None:
    df = integer_dataframe_6(library)
    ns = df.__dataframe_namespace__()
    sorted_indices = df.col("b").sorted_indices(ascending=False)
    result = df.assign(sorted_indices.rename("result"))
    expected_1 = {
        "a": [1, 1, 1, 2, 2],
        "b": [4, 4, 3, 1, 2],
        "result": [1, 0, 2, 4, 3],
    }
    expected_2 = {
        "a": [1, 1, 1, 2, 2],
        "b": [4, 4, 3, 1, 2],
        "result": [0, 1, 2, 4, 3],
    }
    try:
        compare_dataframe_with_reference(result, expected_1, dtype=ns.Int64)
    except AssertionError:
        # order isn't determinist, so try both
        compare_dataframe_with_reference(result, expected_2, dtype=ns.Int64)
