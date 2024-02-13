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

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    bool_dataframe_1,
    bool_dataframe_3,
    compare_dataframe_with_reference,
)


@pytest.mark.parametrize(
    ("reduction", "expected_data"),
    [
        ("any", {"a": [True], "b": [True]}),
        ("all", {"a": [False], "b": [True]}),
    ],
)
def test_reductions(
    library: BaseHandler,
    reduction: str,
    expected_data: dict[str, object],
) -> None:
    df = bool_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    result = getattr(df, reduction)()
    compare_dataframe_with_reference(result, expected_data, dtype=ns.Bool)  # type: ignore[arg-type]


def test_any(library: BaseHandler) -> None:
    df = bool_dataframe_3(library)
    ns = df.__dataframe_namespace__()
    result = df.any()
    expected = {"a": [False], "b": [True], "c": [True]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Bool)


def test_all(library: BaseHandler) -> None:
    df = bool_dataframe_3(library)
    ns = df.__dataframe_namespace__()
    result = df.all()
    expected = {"a": [False], "b": [False], "c": [True]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Bool)
