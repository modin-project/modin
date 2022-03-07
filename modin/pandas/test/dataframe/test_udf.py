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
from modin.config import MinPartitionSize
import modin.pandas as pd

from pandas.core.dtypes.common import is_list_like
from modin.pandas.test.utils import (
    random_state,
    df_equals,
    test_data_values,
    test_data_keys,
    query_func_keys,
    query_func_values,
    agg_func_keys,
    agg_func_values,
    agg_func_except_keys,
    agg_func_except_values,
    eval_general,
    create_test_dfs,
    udf_func_values,
    udf_func_keys,
    test_data,
    bool_arg_keys,
    bool_arg_values,
    arg_keys,
    default_to_pandas_ignore_string,
)
from modin.config import NPartitions, StorageFormat
from modin.test.test_utils import warns_that_defaulting_to_pandas

NPartitions.put(4)

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")

# Our configuration in pytest.ini requires that we explicitly catch all
# instances of defaulting to pandas, but some test modules, like this one,
# have too many such instances.
# TODO(https://github.com/modin-project/modin/issues/3655): catch all instances
# of defaulting to pandas.
pytestmark = pytest.mark.filterwarnings(default_to_pandas_ignore_string)


def test_agg_dict():
    md_df, pd_df = create_test_dfs(test_data_values[0])
    agg_dict = {pd_df.columns[0]: "sum", pd_df.columns[-1]: ("sum", "count")}
    eval_general(md_df, pd_df, lambda df: df.agg(agg_dict), raising_exceptions=True)

    agg_dict = {
        "new_col1": (pd_df.columns[0], "sum"),
        "new_col2": (pd_df.columns[-1], "count"),
    }
    eval_general(md_df, pd_df, lambda df: df.agg(**agg_dict), raising_exceptions=True)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize(
    "func",
    agg_func_values + agg_func_except_values,
    ids=agg_func_keys + agg_func_except_keys,
)
@pytest.mark.parametrize("op", ["agg", "apply"])
def test_agg_apply(axis, func, op):
    eval_general(
        *create_test_dfs(test_data["float_nan_data"]),
        lambda df: getattr(df, op)(func, axis),
    )


@pytest.mark.parametrize("axis", ["rows", "columns"])
@pytest.mark.parametrize(
    "func",
    agg_func_values + agg_func_except_values,
    ids=agg_func_keys + agg_func_except_keys,
)
@pytest.mark.parametrize("op", ["agg", "apply"])
def test_agg_apply_axis_names(axis, func, op):
    eval_general(
        *create_test_dfs(test_data["int_data"]),
        lambda df: getattr(df, op)(func, axis),
    )


def test_aggregate_alias():
    assert pd.DataFrame.agg == pd.DataFrame.aggregate


def test_aggregate_error_checking():
    modin_df = pd.DataFrame(test_data["float_nan_data"])

    with warns_that_defaulting_to_pandas():
        modin_df.aggregate({modin_df.columns[0]: "sum", modin_df.columns[1]: "mean"})

    with warns_that_defaulting_to_pandas():
        modin_df.aggregate("cumproduct")

    with pytest.raises(ValueError):
        modin_df.aggregate("NOT_EXISTS")


@pytest.mark.xfail(
    StorageFormat.get() == "Pandas",
    reason="DataFrame.apply(dict) raises an exception because of a bug in its"
    + "implementation for pandas storage format, this prevents us from catching the desired"
    + "exception. You can track this bug at:"
    + "https://github.com/modin-project/modin/issues/3221",
)
@pytest.mark.parametrize(
    "func",
    agg_func_values + agg_func_except_values,
    ids=agg_func_keys + agg_func_except_keys,
)
def test_apply_key_error(func):
    if not (is_list_like(func) or callable(func) or isinstance(func, str)):
        pytest.xfail(
            reason="Because index materialization is expensive Modin first"
            + "checks the validity of the function itself and only then the engine level"
            + "checks the validity of the indices. Pandas order of such checks is reversed,"
            + "so we get different errors when both (function and index) are invalid."
        )
    eval_general(
        *create_test_dfs(test_data["int_data"]),
        lambda df: df.apply({"row": func}, axis=1),
    )


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("level", [None, -1, 0, 1])
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("func", ["kurt", "count", "sum", "mean", "all", "any"])
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


@pytest.mark.parametrize(
    "column", ["A", ["A", "C"]], ids=arg_keys("column", ["A", ["A", "C"]])
)
@pytest.mark.parametrize(
    "ignore_index", bool_arg_values, ids=arg_keys("ignore_index", bool_arg_keys)
)
def test_explode_single_partition(column, ignore_index):
    # This test data has two columns where some items are lists that
    # explode() should expand. In some rows, the columns have list-like
    # elements that must be expanded, and in others, they have empty lists
    # or items that aren't list-like at all.
    data = {
        "A": [[0, 1, 2], "foo", [], [3, 4]],
        "B": 1,
        "C": [["a", "b", "c"], np.nan, [], ["d", "e"]],
    }
    eval_general(
        *create_test_dfs(data),
        lambda df: df.explode(column, ignore_index=ignore_index),
    )


@pytest.mark.parametrize(
    "column", ["A", ["A", "C"]], ids=arg_keys("column", ["A", ["A", "C"]])
)
@pytest.mark.parametrize(
    "ignore_index", bool_arg_values, ids=arg_keys("ignore_index", bool_arg_keys)
)
def test_explode_all_partitions(column, ignore_index):
    # Test explode with enough rows to fill all partitions. explode should
    # expand every row in the input data into two rows. It's especially
    # important that the input data has list-like elements that must be
    # expanded at the boundaries of the partitions, e.g. at row 31.
    num_rows = NPartitions.get() * MinPartitionSize.get()
    data = {"A": [[3, 4]] * num_rows, "C": [["a", "b"]] * num_rows}
    eval_general(
        *create_test_dfs(data),
        lambda df: df.explode(column, ignore_index=ignore_index),
    )


@pytest.mark.parametrize("axis", ["rows", "columns"])
@pytest.mark.parametrize("args", [(1,), ("_A",)])
def test_apply_args(axis, args):
    def apply_func(series, y):
        try:
            return series + y
        except TypeError:
            return series.map(str) + str(y)

    eval_general(
        *create_test_dfs(test_data["int_data"]),
        lambda df: df.apply(apply_func, axis=axis, args=args),
    )


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


@pytest.mark.parametrize("func", udf_func_values, ids=udf_func_keys)
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_apply_udf(data, func):
    eval_general(
        *create_test_dfs(data),
        lambda df, *args, **kwargs: df.apply(func, *args, **kwargs),
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


TEST_VAR = 2


@pytest.mark.parametrize("method", ["query", "eval"])
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("local_var", [2])
def test_eval_and_query_with_local_and_global_var(method, data, local_var):
    modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)
    op = "+" if method == "eval" else "<"
    for expr in (f"col1 {op} @local_var", f"col1 {op} @TEST_VAR"):
        df_equals(getattr(modin_df, method)(expr), getattr(pandas_df, method)(expr))


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

    try:
        pandas_result = pandas_df.query(funcs)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.query(funcs)
    else:
        modin_result = modin_df.query(funcs)
        df_equals(modin_result, pandas_result)


def test_empty_query():
    modin_df = pd.DataFrame([1, 2, 3, 4, 5])

    with pytest.raises(ValueError):
        modin_df.query("")


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
@pytest.mark.parametrize(
    "func",
    agg_func_values + agg_func_except_values,
    ids=agg_func_keys + agg_func_except_keys,
)
def test_transform(data, func):
    eval_general(
        *create_test_dfs(data),
        lambda df: df.transform(func),
    )
