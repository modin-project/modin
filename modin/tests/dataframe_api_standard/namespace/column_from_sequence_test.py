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

from datetime import datetime, timedelta
from typing import Any

import pytest

from modin.tests.dataframe_api_standard.utils import (
    BaseHandler,
    compare_column_with_reference,
    integer_dataframe_1,
)


@pytest.mark.parametrize(
    ("values", "dtype", "kwargs"),
    [
        ([1, 2, 3], "Int64", {}),
        ([1, 2, 3], "Int32", {}),
        ([1, 2, 3], "Int16", {}),
        ([1, 2, 3], "Int8", {}),
        ([1, 2, 3], "UInt64", {}),
        ([1, 2, 3], "UInt32", {}),
        ([1, 2, 3], "UInt16", {}),
        ([1, 2, 3], "UInt8", {}),
        ([1.0, 2.0, 3.0], "Float64", {}),
        ([1.0, 2.0, 3.0], "Float32", {}),
        ([True, False, True], "Bool", {}),
        (["express", "yourself"], "String", {}),
        ([datetime(2020, 1, 1), datetime(2020, 1, 2)], "Datetime", {"time_unit": "us"}),
        ([timedelta(1), timedelta(2)], "Duration", {"time_unit": "us"}),
    ],
)
def test_column_from_sequence(
    library: BaseHandler,
    values: list[Any],
    dtype: str,
    kwargs: dict[str, Any],
) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    ns = ser.__column_namespace__()
    expected_dtype = getattr(ns, dtype)
    result = ns.dataframe_from_columns(
        ns.column_from_sequence(
            values,
            dtype=expected_dtype(**kwargs),
            name="result",
        ),
    )
    compare_column_with_reference(result.col("result"), values, dtype=expected_dtype)


def test_column_from_sequence_no_dtype(
    library: BaseHandler,
) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    result = ns.dataframe_from_columns(ns.column_from_sequence([1, 2, 3], name="result"))  # type: ignore[call-arg]
    expected = [1, 2, 3]
    compare_column_with_reference(result.col("result"), expected, dtype=ns.Int64)
