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

import pytest
import pandas
import matplotlib
import modin.pandas as pd

from modin.pandas.test.utils import (
    df_equals,
    test_data_values,
    test_data_keys,
    eval_general,
    test_data,
    create_test_dfs,
)

pd.DEFAULT_NPARTITIONS = 4

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")


@pytest.mark.parametrize(
    "other",
    [
        lambda df: 4,
        lambda df, axis: df.iloc[0] if axis == "columns" else list(df[df.columns[0]]),
    ],
    ids=["scalar", "series_or_list"],
)
@pytest.mark.parametrize("axis", ["rows", "columns"])
@pytest.mark.parametrize(
    "op",
    [
        *("add", "radd", "sub", "rsub", "mod", "rmod", "pow", "rpow"),
        *("truediv", "rtruediv", "mul", "rmul", "floordiv", "rfloordiv"),
    ],
)
def test_math_functions(other, axis, op):
    data = test_data["float_nan_data"]
    if (op == "floordiv" or op == "rfloordiv") and axis == "rows":
        # lambda == "series_or_list"
        pytest.xfail(reason="different behaviour")

    if op == "rmod" and axis == "rows":
        # lambda == "series_or_list"
        pytest.xfail(reason="different behaviour")

    eval_general(
        *create_test_dfs(data), lambda df: getattr(df, op)(other(df, axis), axis=axis)
    )


@pytest.mark.parametrize(
    "other",
    [lambda df: df[: -(2 ** 4)], lambda df: df[df.columns[0]].reset_index(drop=True)],
    ids=["check_missing_value", "check_different_index"],
)
@pytest.mark.parametrize("fill_value", [None, 3.0])
@pytest.mark.parametrize(
    "op",
    [
        *("add", "radd", "sub", "rsub", "mod", "rmod", "pow", "rpow"),
        *("truediv", "rtruediv", "mul", "rmul", "floordiv", "rfloordiv"),
    ],
)
def test_math_functions_fill_value(other, fill_value, op):
    data = test_data["int_data"]
    modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)

    eval_general(
        modin_df,
        pandas_df,
        lambda df: getattr(df, op)(other(df), axis=0, fill_value=fill_value),
    )


@pytest.mark.parametrize(
    "op",
    [
        *("add", "radd", "sub", "rsub", "mod", "rmod", "pow", "rpow"),
        *("truediv", "rtruediv", "mul", "rmul", "floordiv", "rfloordiv"),
    ],
)
def test_math_functions_level(op):
    modin_df = pd.DataFrame(test_data["int_data"])
    modin_df.index = pandas.MultiIndex.from_tuples(
        [(i // 4, i // 2, i) for i in modin_df.index]
    )

    # Defaults to pandas
    with pytest.warns(UserWarning):
        # Operation against self for sanity check
        getattr(modin_df, op)(modin_df, axis=0, level=1)


@pytest.mark.parametrize(
    "math_op, alias",
    [
        ("truediv", "divide"),
        ("truediv", "div"),
        ("rtruediv", "rdiv"),
        ("mul", "multiply"),
        ("sub", "subtract"),
        ("add", "__add__"),
        ("radd", "__radd__"),
        ("div", "__div__"),
        ("rdiv", "__rdiv__"),
        ("truediv", "__truediv__"),
        ("rtruediv", "__rtruediv__"),
        ("floordiv", "__floordiv__"),
        ("rfloordiv", "__rfloordiv__"),
        ("mod", "__mod__"),
        ("rmod", "__rmod__"),
        ("mul", "__mul__"),
        ("rmul", "__rmul__"),
        ("pow", "__pow__"),
        ("rpow", "__rpow__"),
        ("sub", "__sub__"),
        ("rsub", "__rsub__"),
    ],
)
def test_math_alias(math_op, alias):
    assert getattr(pd.DataFrame, math_op) == getattr(pd.DataFrame, alias)


@pytest.mark.parametrize("other", ["as_left", 4, 4.0])
@pytest.mark.parametrize("op", ["eq", "ge", "gt", "le", "lt", "ne"])
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_comparison(data, op, other):
    eval_general(
        *create_test_dfs(data),
        lambda df: getattr(df, op)(df if other == "as_left" else other),
    )


@pytest.mark.xfail_backends(
    ["BaseOnPython"],
    reason="Test is failing because of mismathing of thrown exceptions. See pandas issue #36377",
)
@pytest.mark.parametrize("other", ["a"])
@pytest.mark.parametrize("op", ["ge", "gt", "le", "lt"])
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_comparison_except(data, op, other):
    # So far `eq` and `ne` are excluded from testing because
    # pandas throws an exception but Modin doesn't throw it.
    with pytest.raises(AssertionError):
        eval_general(
            *create_test_dfs(data),
            lambda df: getattr(df, op)(other),
        )


@pytest.mark.parametrize("op", ["eq", "ge", "gt", "le", "lt", "ne"])
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_multi_level_comparison(data, op):
    modin_df_multi_level = pd.DataFrame(data)

    new_idx = pandas.MultiIndex.from_tuples(
        [(i // 4, i // 2, i) for i in modin_df_multi_level.index]
    )
    modin_df_multi_level.index = new_idx

    # Defaults to pandas
    with pytest.warns(UserWarning):
        # Operation against self for sanity check
        getattr(modin_df_multi_level, op)(modin_df_multi_level, axis=0, level=1)


def test_equals():
    frame_data = {"col1": [2.9, 3, 3, 3], "col2": [2, 3, 4, 1]}
    modin_df1 = pd.DataFrame(frame_data)
    modin_df2 = pd.DataFrame(frame_data)

    assert modin_df1.equals(modin_df2)

    df_equals(modin_df1, modin_df2)
    df_equals(modin_df1, pd.DataFrame(modin_df1))

    frame_data = {"col1": [2.9, 3, 3, 3], "col2": [2, 3, 5, 1]}
    modin_df3 = pd.DataFrame(frame_data, index=list("abcd"))

    assert not modin_df1.equals(modin_df3)

    with pytest.raises(AssertionError):
        df_equals(modin_df3, modin_df1)

    with pytest.raises(AssertionError):
        df_equals(modin_df3, modin_df2)

    assert modin_df1.equals(modin_df2._query_compiler.to_pandas())
