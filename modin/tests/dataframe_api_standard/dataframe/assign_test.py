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
)


def test_insert_columns(library: BaseHandler) -> None:
    df = integer_dataframe_1(library, api_version="2023.09-beta")
    ns = df.__dataframe_namespace__()
    new_col = (df.col("b") + 3).rename("result")
    result = df.assign(new_col.rename("c"))
    expected = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)
    # check original df didn't change
    expected = {"a": [1, 2, 3], "b": [4, 5, 6]}
    compare_dataframe_with_reference(df, expected, dtype=ns.Int64)


def test_insert_multiple_columns(library: BaseHandler) -> None:
    df = integer_dataframe_1(library, api_version="2023.09-beta")
    ns = df.__dataframe_namespace__()
    new_col = (df.col("b") + 3).rename("result")
    result = df.assign(new_col.rename("c"), new_col.rename("d"))
    expected = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [7, 8, 9]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)
    # check original df didn't change
    expected = {"a": [1, 2, 3], "b": [4, 5, 6]}
    compare_dataframe_with_reference(df, expected, dtype=ns.Int64)


def test_insert_multiple_columns_invalid(library: BaseHandler) -> None:
    df = integer_dataframe_1(library, api_version="2023.09-beta")
    df.__dataframe_namespace__()
    new_col = (df.col("b") + 3).rename("result")
    with pytest.raises(TypeError):
        _ = df.assign([new_col.rename("c"), new_col.rename("d")])  # type: ignore[arg-type]


def test_insert_eager_columns(library: BaseHandler) -> None:
    df = integer_dataframe_1(library, api_version="2023.09-beta")
    ns = df.__dataframe_namespace__()
    new_col = (df.col("b") + 3).rename("result")
    result = df.assign(new_col.rename("c"), new_col.rename("d"))
    expected = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [7, 8, 9]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)
    # check original df didn't change
    expected = {"a": [1, 2, 3], "b": [4, 5, 6]}
    compare_dataframe_with_reference(df, expected, dtype=ns.Int64)
