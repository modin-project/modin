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


import pytest

from modin.tests.dataframe_api_standard.utils import (
    BaseHandler,
    compare_dataframe_with_reference,
    integer_dataframe_1,
    integer_dataframe_2,
)


def test_within_df_propagation(library: BaseHandler) -> None:
    df1 = integer_dataframe_1(library)
    df1 = df1 + 1
    with pytest.raises(RuntimeError):
        _ = int(df1.col("a").get_value(0))  # type: ignore[call-overload]

    df1 = integer_dataframe_1(library)
    df1 = df1.persist()
    df1 = df1 + 1
    # the call below would recompute `df1 + 1` multiple times
    assert int(df1.col("a").get_value(0)) == 2  # type: ignore[call-overload]

    # this is the correct way
    df1 = integer_dataframe_1(library)
    df1 = df1 + 1
    df1 = df1.persist()
    assert int(df1.col("a").get_value(0)) == 2  # type: ignore[call-overload]

    # persisting the column works too
    df1 = integer_dataframe_1(library)
    df1 = df1 + 1
    assert int(df1.col("a").persist().get_value(0)) == 2  # type: ignore[call-overload]

    # ...but not if the column was modified
    df1 = integer_dataframe_1(library)
    df1 = df1 + 1
    col = df1.col("a").persist()
    assert int((col + 1).get_value(0)) == 3  # type: ignore[call-overload]

    # persisting the scalar works too
    df1 = integer_dataframe_1(library)
    df1 = df1 + 1
    assert int(df1.col("a").get_value(0).persist()) == 2  # type: ignore[call-overload]

    # ...but not if you modify the scalar
    df1 = integer_dataframe_1(library)
    df1 = df1 + 1
    scalar = df1.col("a").get_value(0).persist()
    assert int(scalar + 1) == 3  # type: ignore[call-overload]


def test_within_df_within_col_propagation(library: BaseHandler) -> None:
    df1 = integer_dataframe_1(library)
    df1 = df1 + 1
    df1 = df1.persist()
    assert int((df1.col("a") + 1).mean()) == 4  # type: ignore[call-overload]


def test_cross_df_propagation(library: BaseHandler) -> None:
    if library.name == "modin":
        pytest.skip("TODO: enable for modin")
    df1 = integer_dataframe_1(library)
    df2 = integer_dataframe_2(library)
    ns = df1.__dataframe_namespace__()
    df1 = df1 + 1
    df2 = df2.rename({"b": "c"})
    result = df1.join(df2, how="left", left_on="a", right_on="a")
    ns = result.__dataframe_namespace__()
    expected = {
        "a": [2, 3, 4],
        "b": [5, 6, 7],
        "c": [2.0, float("nan"), 6.0],
    }
    expected_dtype = {
        "a": ns.Int64,
        "b": ns.Int64,
        "c": ns.Float64,
    }
    compare_dataframe_with_reference(result, expected, dtype=expected_dtype)  # type: ignore[arg-type]


def test_multiple_propagations(library: BaseHandler) -> None:
    # This is a bit "ugly", as the user is "required" to call `persist`
    # multiple times to do things optimally
    df = integer_dataframe_1(library)
    df = df.persist()
    with pytest.warns(UserWarning):
        df1 = df.filter(df.col("a") > 1).persist()
        df2 = df.filter(df.col("a") <= 1).persist()
    assert int(df1.col("a").mean()) == 2  # type: ignore[call-overload]
    assert int(df2.col("a").mean()) == 1  # type: ignore[call-overload]

    # But what if I want to do this
    df = integer_dataframe_1(library)
    df = df.persist()
    df1 = df.filter(df.col("a") > 1)
    df2 = df.filter(df.col("a") <= 1)

    df1 = df1 + 1
    # without this persist, `df1 + 1` will be computed twice
    int(df1.col("a").mean())  # type: ignore[call-overload]
    int(df1.col("a").mean())  # type: ignore[call-overload]


def test_parent_propagations(library: BaseHandler) -> None:
    # Set up something like this:
    #
    #         df
    #     df1    df2
    #
    # If I persist df1, then that triggers df.
    # If I then want call some scalar on df2, that will have to trigger
    # df again. If df2 wasn't persisted, then df would be recomputed.
    # So, we need to persist df2 as well.
    df = integer_dataframe_1(library)
    df1 = df.filter(df.col("a") > 1)
    df2 = df.filter(df.col("a") <= 1)

    df1 = df1.persist()
    with pytest.raises(RuntimeError):
        int(df2.col("a").mean())  # type: ignore[call-overload]
