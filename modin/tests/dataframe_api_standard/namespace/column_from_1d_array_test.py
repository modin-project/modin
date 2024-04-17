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

from datetime import date, datetime, timedelta

import numpy as np
import pytest

from modin.tests.dataframe_api_standard.utils import (
    BaseHandler,
    compare_column_with_reference,
    integer_dataframe_1,
)


@pytest.mark.parametrize(
    ("pandas_dtype", "column_dtype"),
    [
        ("float64", "Float64"),
        ("float32", "Float32"),
        ("int64", "Int64"),
        ("int32", "Int32"),
        ("int16", "Int16"),
        ("int8", "Int8"),
        ("uint64", "UInt64"),
        ("uint32", "UInt32"),
        ("uint16", "UInt16"),
        ("uint8", "UInt8"),
    ],
)
def test_column_from_1d_array(
    library: BaseHandler,
    pandas_dtype: str,
    column_dtype: str,
) -> None:
    ser = integer_dataframe_1(library).col("a").persist()
    ns = ser.__column_namespace__()
    arr = np.array([1, 2, 3], dtype=pandas_dtype)
    result = ns.dataframe_from_columns(
        ns.column_from_1d_array(  # type: ignore[call-arg]
            arr,
            name="result",
        ),
    )
    expected = [1, 2, 3]
    compare_column_with_reference(
        result.col("result"),
        expected,
        dtype=getattr(ns, column_dtype),
    )


def test_column_from_1d_array_string(
    library: BaseHandler,
) -> None:
    ser = integer_dataframe_1(library).persist().col("a")
    ns = ser.__column_namespace__()
    arr = np.array(["a", "b", "c"])
    result = ns.dataframe_from_columns(
        ns.column_from_1d_array(  # type: ignore[call-arg]
            arr,
            name="result",
        ),
    )
    expected = ["a", "b", "c"]
    compare_column_with_reference(result.col("result"), expected, dtype=ns.String)


def test_column_from_1d_array_bool(
    library: BaseHandler,
) -> None:
    ser = integer_dataframe_1(library).persist().col("a")
    ns = ser.__column_namespace__()
    arr = np.array([True, False, True])
    result = ns.dataframe_from_columns(
        ns.column_from_1d_array(  # type: ignore[call-arg]
            arr,
            name="result",
        ),
    )
    expected = [True, False, True]
    compare_column_with_reference(result.col("result"), expected, dtype=ns.Bool)


def test_datetime_from_1d_array(library: BaseHandler) -> None:
    ser = integer_dataframe_1(library).persist().col("a")
    ns = ser.__column_namespace__()
    arr = np.array([date(2020, 1, 1), date(2020, 1, 2)], dtype="datetime64[ms]")
    result = ns.dataframe_from_columns(
        ns.column_from_1d_array(  # type: ignore[call-arg]
            arr,
            name="result",
        ),
    )
    expected = [datetime(2020, 1, 1), datetime(2020, 1, 2)]
    compare_column_with_reference(result.col("result"), expected, dtype=ns.Datetime)


def test_duration_from_1d_array(library: BaseHandler) -> None:
    ser = integer_dataframe_1(library).persist().col("a")
    ns = ser.__column_namespace__()
    arr = np.array([timedelta(1), timedelta(2)], dtype="timedelta64[ms]")
    result = ns.dataframe_from_columns(
        ns.column_from_1d_array(  # type: ignore[call-arg]
            arr,
            name="result",
        ),
    )
    expected = [timedelta(1), timedelta(2)]
    compare_column_with_reference(result.col("result"), expected, dtype=ns.Duration)
