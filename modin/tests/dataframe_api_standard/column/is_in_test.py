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

from typing import TYPE_CHECKING, Any

import pytest

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_column_with_reference,
    float_dataframe_1,
    float_dataframe_2,
    float_dataframe_3,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize(
    ("df_factory", "expected_values"),
    [
        (float_dataframe_1, [False, True]),
        (float_dataframe_2, [True, False]),
        (float_dataframe_3, [True, False]),
    ],
)
@pytest.mark.filterwarnings("ignore:np.find_common_type is deprecated")
def test_is_in(
    library: BaseHandler,
    df_factory: Callable[[BaseHandler], Any],
    expected_values: list[bool],
) -> None:
    df = df_factory(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    other = ser + 1
    result = df.assign(ser.is_in(other).rename("result"))
    compare_column_with_reference(result.col("result"), expected_values, dtype=ns.Bool)


@pytest.mark.parametrize(
    ("df_factory", "expected_values"),
    [
        (float_dataframe_1, [False, True]),
        (float_dataframe_2, [True, False]),
        (float_dataframe_3, [True, False]),
    ],
)
@pytest.mark.filterwarnings("ignore:np.find_common_type is deprecated")
def test_expr_is_in(
    library: BaseHandler,
    df_factory: Callable[[BaseHandler], Any],
    expected_values: list[bool],
) -> None:
    df = df_factory(library)
    ns = df.__dataframe_namespace__()
    col = df.col
    ser = col("a")
    other = ser + 1
    result = df.assign(ser.is_in(other).rename("result"))
    compare_column_with_reference(result.col("result"), expected_values, dtype=ns.Bool)
