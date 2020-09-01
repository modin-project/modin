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
import numpy as np
import pandas
import matplotlib
import modin.pandas as pd

from modin.pandas.test.utils import (
    random_state,
    df_equals,
    name_contains,
    test_data_values,
    test_data_keys,
    numeric_dfs,
    query_func_keys,
    query_func_values,
    agg_func_keys,
    agg_func_values,
    numeric_agg_funcs,
    axis_keys,
    axis_values,
    eval_general,
    create_test_dfs,
    udf_func_values,
    udf_func_keys,
)

pd.DEFAULT_NPARTITIONS = 4

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
def test_agg(data, axis, func):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.agg(func, axis)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.agg(func, axis)
    else:
        modin_result = modin_df.agg(func, axis)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
def test_agg_numeric(request, data, axis, func):
    if name_contains(request.node.name, numeric_agg_funcs) and name_contains(
        request.node.name, numeric_dfs
    ):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        try:
            pandas_result = pandas_df.agg(func, axis)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.agg(func, axis)
        else:
            modin_result = modin_df.agg(func, axis)
            df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
def test_aggregate(request, data, func, axis):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.aggregate(func, axis)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.aggregate(func, axis)
    else:
        modin_result = modin_df.aggregate(func, axis)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
def test_aggregate_numeric(request, data, axis, func):
    if name_contains(request.node.name, numeric_agg_funcs) and name_contains(
        request.node.name, numeric_dfs
    ):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        try:
            pandas_result = pandas_df.agg(func, axis)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.agg(func, axis)
        else:
            modin_result = modin_df.agg(func, axis)
            df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_aggregate_error_checking(data):
    modin_df = pd.DataFrame(data)

    assert modin_df.aggregate("ndim") == 2

    with pytest.warns(UserWarning):
        modin_df.aggregate({modin_df.columns[0]: "sum", modin_df.columns[1]: "mean"})

    with pytest.warns(UserWarning):
        modin_df.aggregate("cumproduct")

    with pytest.raises(ValueError):
        modin_df.aggregate("NOT_EXISTS")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
def test_apply(request, data, func, axis):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    with pytest.raises(TypeError):
        modin_df.apply({"row": func}, axis=1)

    try:
        pandas_result = pandas_df.apply(func, axis)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.apply(func, axis)
    else:
        modin_result = modin_df.apply(func, axis)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("level", [None, -1, 0, 1])
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize(
    "func",
    [
        "kurt",
        pytest.param(
            "count",
            marks=pytest.mark.xfail(
                reason="count method handle level parameter incorrectly"
            ),
        ),
        pytest.param(
            "sum",
            marks=pytest.mark.xfail(
                reason="sum method handle level parameter incorrectly"
            ),
        ),
        pytest.param(
            "mean",
            marks=pytest.mark.xfail(
                reason="mean method handle level parameter incorrectly"
            ),
        ),
        pytest.param(
            "all",
            marks=pytest.mark.xfail(
                reason="all method handle level parameter incorrectly"
            ),
        ),
    ],
)
def test_apply_text_func_with_level(level, data, func, axis):
    func_kwargs = {"level": level, "axis": axis}
    rows_number = len(next(iter(data.values())))  # length of the first data column
    level_0 = np.random.choice([0, 1, 2], rows_number)
    level_1 = np.random.choice([3, 4, 5], rows_number)
    index = pd.MultiIndex.from_arrays([level_0, level_1])

    eval_general(
        pd.DataFrame(data, index=index),
        pandas.DataFrame(data, index=index),
        lambda df, *args, **kwargs: df.apply(func, *args, **kwargs),
        **func_kwargs,
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
def test_apply_args(data, axis):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    def apply_func(series, y):
        try:
            return series + y
        except TypeError:
            return series.map(str) + str(y)

    modin_result = modin_df.apply(apply_func, axis=axis, args=(1,))
    pandas_result = pandas_df.apply(apply_func, axis=axis, args=(1,))
    df_equals(modin_result, pandas_result)

    modin_result = modin_df.apply(apply_func, axis=axis, args=("_A",))
    pandas_result = pandas_df.apply(apply_func, axis=axis, args=("_A",))
    df_equals(modin_result, pandas_result)


def test_apply_metadata():
    def add(a, b, c):
        return a + b + c

    data = {"A": [1, 2, 3], "B": [4, 5, 6], "C": [7, 8, 9]}

    modin_df = pd.DataFrame(data)
    modin_df["add"] = modin_df.apply(
        lambda row: add(row["A"], row["B"], row["C"]), axis=1
    )

    pandas_df = pandas.DataFrame(data)
    pandas_df["add"] = pandas_df.apply(
        lambda row: add(row["A"], row["B"], row["C"]), axis=1
    )
    df_equals(modin_df, pandas_df)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
def test_apply_numeric(request, data, func, axis):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    if name_contains(request.node.name, numeric_dfs):
        try:
            pandas_result = pandas_df.apply(func, axis)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.apply(func, axis)
        else:
            modin_result = modin_df.apply(func, axis)
            df_equals(modin_result, pandas_result)

    if "empty_data" not in request.node.name:
        key = modin_df.columns[0]
        modin_result = modin_df.apply(lambda df: df.drop(key), axis=1)
        pandas_result = pandas_df.apply(lambda df: df.drop(key), axis=1)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("func", udf_func_values, ids=udf_func_keys)
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_apply_udf(data, func):
    eval_general(
        *create_test_dfs(data),
        lambda df, *args, **kwargs: df.apply(*args, **kwargs),
        func=func,
        other=lambda df: df,
    )


def test_eval_df_use_case():
    frame_data = {"a": random_state.randn(10), "b": random_state.randn(10)}
    df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)

    # test eval for series results
    tmp_pandas = df.eval("arctan2(sin(a), b)", engine="python", parser="pandas")
    tmp_modin = modin_df.eval("arctan2(sin(a), b)", engine="python", parser="pandas")

    assert isinstance(tmp_modin, pd.Series)
    df_equals(tmp_modin, tmp_pandas)

    # Test not inplace assignments
    tmp_pandas = df.eval("e = arctan2(sin(a), b)", engine="python", parser="pandas")
    tmp_modin = modin_df.eval(
        "e = arctan2(sin(a), b)", engine="python", parser="pandas"
    )
    df_equals(tmp_modin, tmp_pandas)

    # Test inplace assignments
    df.eval("e = arctan2(sin(a), b)", engine="python", parser="pandas", inplace=True)
    modin_df.eval(
        "e = arctan2(sin(a), b)", engine="python", parser="pandas", inplace=True
    )
    # TODO: Use a series equality validator.
    df_equals(modin_df, df)


def test_eval_df_arithmetic_subexpression():
    frame_data = {"a": random_state.randn(10), "b": random_state.randn(10)}
    df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)
    df.eval("not_e = sin(a + b)", engine="python", parser="pandas", inplace=True)
    modin_df.eval("not_e = sin(a + b)", engine="python", parser="pandas", inplace=True)
    # TODO: Use a series equality validator.
    df_equals(modin_df, df)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_filter(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    by = {"items": ["col1", "col5"], "regex": "4$|3$", "like": "col"}
    df_equals(modin_df.filter(items=by["items"]), pandas_df.filter(items=by["items"]))

    df_equals(
        modin_df.filter(regex=by["regex"], axis=0),
        pandas_df.filter(regex=by["regex"], axis=0),
    )
    df_equals(
        modin_df.filter(regex=by["regex"], axis=1),
        pandas_df.filter(regex=by["regex"], axis=1),
    )

    df_equals(modin_df.filter(like=by["like"]), pandas_df.filter(like=by["like"]))

    with pytest.raises(TypeError):
        modin_df.filter(items=by["items"], regex=by["regex"])

    with pytest.raises(TypeError):
        modin_df.filter()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_pipe(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    n = len(modin_df.index)
    a, b, c = 2 % n, 0, 3 % n
    col = modin_df.columns[3 % len(modin_df.columns)]

    def h(x):
        return x.drop(columns=[col])

    def g(x, arg1=0):
        for _ in range(arg1):
            x = x.append(x)
        return x

    def f(x, arg2=0, arg3=0):
        return x.drop([arg2, arg3])

    df_equals(
        f(g(h(modin_df), arg1=a), arg2=b, arg3=c),
        (modin_df.pipe(h).pipe(g, arg1=a).pipe(f, arg2=b, arg3=c)),
    )
    df_equals(
        (modin_df.pipe(h).pipe(g, arg1=a).pipe(f, arg2=b, arg3=c)),
        (pandas_df.pipe(h).pipe(g, arg1=a).pipe(f, arg2=b, arg3=c)),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("funcs", query_func_values, ids=query_func_keys)
def test_query(data, funcs):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    with pytest.raises(ValueError):
        modin_df.query("")
    with pytest.raises(NotImplementedError):
        x = 2  # noqa F841
        modin_df.query("col1 < @x")

    try:
        pandas_result = pandas_df.query(funcs)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.query(funcs)
    else:
        modin_result = modin_df.query(funcs)
        df_equals(modin_result, pandas_result)


def test_query_after_insert():
    modin_df = pd.DataFrame({"x": [-1, 0, 1, None], "y": [1, 2, None, 3]})
    modin_df["z"] = modin_df.eval("x / y")
    modin_df = modin_df.query("z >= 0")
    modin_result = modin_df.reset_index(drop=True)
    modin_result.columns = ["a", "b", "c"]

    pandas_df = pd.DataFrame({"x": [-1, 0, 1, None], "y": [1, 2, None, 3]})
    pandas_df["z"] = pandas_df.eval("x / y")
    pandas_df = pandas_df.query("z >= 0")
    pandas_result = pandas_df.reset_index(drop=True)
    pandas_result.columns = ["a", "b", "c"]

    df_equals(modin_result, pandas_result)
    df_equals(modin_df, pandas_df)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
def test_transform(request, data, func):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.transform(func)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.transform(func)
    else:
        modin_result = modin_df.transform(func)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
def test_transform_numeric(request, data, func):
    if name_contains(request.node.name, numeric_agg_funcs) and name_contains(
        request.node.name, numeric_dfs
    ):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        try:
            pandas_result = pandas_df.transform(func)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.transform(func)
        else:
            modin_result = modin_df.transform(func)
            df_equals(modin_result, pandas_result)
