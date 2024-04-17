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

from typing import Literal

import pytest

from modin.pandas.test.utils import default_to_pandas_ignore_string
from modin.tests.dataframe_api_standard.utils import (
    BaseHandler,
    compare_column_with_reference,
    temporal_dataframe_1,
)

pytestmark = pytest.mark.filterwarnings(default_to_pandas_ignore_string)


@pytest.mark.parametrize(
    ("attr", "expected"),
    [
        ("year", [2020, 2020, 2020]),
        ("month", [1, 1, 1]),
        ("day", [1, 2, 3]),
        ("hour", [1, 3, 5]),
        ("minute", [2, 1, 4]),
        ("second", [1, 2, 9]),
        ("iso_weekday", [3, 4, 5]),
        ("unix_timestamp", [1577840521, 1577934062, 1578027849]),
    ],
)
def test_col_components(library: BaseHandler, attr: str, expected: list[int]) -> None:
    df = temporal_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    for col_name in ("a", "c", "e"):
        result = (
            df.assign(getattr(df.col(col_name), attr)().rename("result"))
            .select(
                "result",
            )
            .cast({"result": ns.Int64()})
        )
        compare_column_with_reference(result.col("result"), expected, dtype=ns.Int64)


@pytest.mark.parametrize(
    ("col_name", "expected"),
    [
        ("a", [123000, 321000, 987000]),
        ("c", [123543, 321654, 987321]),
        ("e", [123543, 321654, 987321]),
    ],
)
def test_col_microsecond(
    library: BaseHandler,
    col_name: str,
    expected: list[int],
) -> None:
    df = temporal_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    result = (
        df.assign(df.col(col_name).microsecond().rename("result"))
        .select(
            "result",
        )
        .cast({"result": ns.Int64()})
    )
    compare_column_with_reference(result.col("result"), expected, dtype=ns.Int64)


@pytest.mark.parametrize(
    ("col_name", "expected"),
    [
        ("a", [123000000, 321000000, 987000000]),
        ("c", [123543000, 321654000, 987321000]),
        ("e", [123543000, 321654000, 987321000]),
    ],
)
def test_col_nanosecond(
    library: BaseHandler, col_name: str, expected: list[int]
) -> None:
    df = temporal_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    result = (
        df.assign(df.col(col_name).nanosecond().rename("result"))  # type: ignore[attr-defined]
        .select(
            "result",
        )
        .cast({"result": ns.Int64()})
    )
    compare_column_with_reference(result.col("result"), expected, dtype=ns.Int64)


@pytest.mark.parametrize(
    ("time_unit", "expected"),
    [
        ("s", [1577840521, 1577934062, 1578027849]),
        ("ms", [1577840521123, 1577934062321, 1578027849987]),
        ("us", [1577840521123543, 1577934062321654, 1578027849987321]),
        ("ns", [1577840521123543000, 1577934062321654000, 1578027849987321000]),
    ],
)
def test_col_unix_timestamp_time_units(
    library: BaseHandler,
    time_unit: Literal["s", "ms", "us", "ns"],
    expected: list[int],
) -> None:
    df = temporal_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    result = (
        df.assign(
            df.col("e").unix_timestamp(time_unit=time_unit).rename("result"),
        )
        .select(
            "result",
        )
        .cast({"result": ns.Int64()})
    )
    compare_column_with_reference(result.col("result"), expected, dtype=ns.Int64)
