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
from pandas.testing import assert_index_equal
import matplotlib
import modin.pandas as pd
import sys

from modin.pandas.test.utils import (
    NROWS,
    RAND_LOW,
    RAND_HIGH,
    df_equals,
    arg_keys,
    name_contains,
    test_data,
    test_data_values,
    test_data_keys,
    axis_keys,
    axis_values,
    int_arg_keys,
    int_arg_values,
    create_test_dfs,
    eval_general,
)

pd.DEFAULT_NPARTITIONS = 4

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")


def eval_setitem(md_df, pd_df, value, col=None, loc=None):
    if loc is not None:
        col = pd_df.columns[loc]

    value_getter = value if callable(value) else (lambda *args, **kwargs: value)

    eval_general(
        md_df, pd_df, lambda df: df.__setitem__(col, value_getter(df)), __inplace__=True
    )


@pytest.mark.parametrize(
    "dates",
    [
        ["2018-02-27 09:03:30", "2018-02-27 09:04:30"],
        ["2018-02-27 09:03:00", "2018-02-27 09:05:00"],
    ],
)
@pytest.mark.parametrize("subset", ["a", "b", ["a", "b"], None])
def test_asof_with_nan(dates, subset):
    data = {"a": [10, 20, 30, 40, 50], "b": [None, None, None, None, 500]}
    index = pd.DatetimeIndex(
        [
            "2018-02-27 09:01:00",
            "2018-02-27 09:02:00",
            "2018-02-27 09:03:00",
            "2018-02-27 09:04:00",
            "2018-02-27 09:05:00",
        ]
    )
    modin_where = pd.DatetimeIndex(dates)
    pandas_where = pandas.DatetimeIndex(dates)
    compare_asof(data, index, modin_where, pandas_where, subset)


@pytest.mark.parametrize(
    "dates",
    [
        ["2018-02-27 09:03:30", "2018-02-27 09:04:30"],
        ["2018-02-27 09:03:00", "2018-02-27 09:05:00"],
    ],
)
@pytest.mark.parametrize("subset", ["a", "b", ["a", "b"], None])
def test_asof_without_nan(dates, subset):
    data = {"a": [10, 20, 30, 40, 50], "b": [70, 600, 30, -200, 500]}
    index = pd.DatetimeIndex(
        [
            "2018-02-27 09:01:00",
            "2018-02-27 09:02:00",
            "2018-02-27 09:03:00",
            "2018-02-27 09:04:00",
            "2018-02-27 09:05:00",
        ]
    )
    modin_where = pd.DatetimeIndex(dates)
    pandas_where = pandas.DatetimeIndex(dates)
    compare_asof(data, index, modin_where, pandas_where, subset)


@pytest.mark.parametrize(
    "lookup",
    [
        [60, 70, 90],
        [60.5, 70.5, 100],
    ],
)
@pytest.mark.parametrize("subset", ["col2", "col1", ["col1", "col2"], None])
def test_asof_large(lookup, subset):
    data = test_data["float_nan_data"]
    index = list(range(NROWS))
    modin_where = pd.Index(lookup)
    pandas_where = pandas.Index(lookup)
    compare_asof(data, index, modin_where, pandas_where, subset)


def compare_asof(
    data, index, modin_where: pd.Index, pandas_where: pandas.Index, subset
):
    modin_df = pd.DataFrame(data, index=index)
    pandas_df = pandas.DataFrame(data, index=index)
    df_equals(
        modin_df.asof(modin_where, subset=subset),
        pandas_df.asof(pandas_where, subset=subset),
    )
    df_equals(
        modin_df.asof(modin_where.values, subset=subset),
        pandas_df.asof(pandas_where.values, subset=subset),
    )
    df_equals(
        modin_df.asof(list(modin_where.values), subset=subset),
        pandas_df.asof(list(pandas_where.values), subset=subset),
    )
    df_equals(
        modin_df.asof(modin_where.values[0], subset=subset),
        pandas_df.asof(pandas_where.values[0], subset=subset),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_first_valid_index(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    assert modin_df.first_valid_index() == (pandas_df.first_valid_index())


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("n", int_arg_values, ids=arg_keys("n", int_arg_keys))
def test_head(data, n):
    # Test normal dataframe head
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    df_equals(modin_df.head(n), pandas_df.head(n))
    df_equals(modin_df.head(len(modin_df) + 1), pandas_df.head(len(pandas_df) + 1))

    # Test head when we call it from a QueryCompilerView
    modin_result = modin_df.loc[:, ["col1", "col3", "col3"]].head(n)
    pandas_result = pandas_df.loc[:, ["col1", "col3", "col3"]].head(n)
    df_equals(modin_result, pandas_result)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_iat(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)  # noqa F841

    with pytest.raises(NotImplementedError):
        modin_df.iat()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_iloc(request, data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    if not name_contains(request.node.name, ["empty_data"]):
        # Scaler
        np.testing.assert_equal(modin_df.iloc[0, 1], pandas_df.iloc[0, 1])

        # Series
        df_equals(modin_df.iloc[0], pandas_df.iloc[0])
        df_equals(modin_df.iloc[1:, 0], pandas_df.iloc[1:, 0])
        df_equals(modin_df.iloc[1:2, 0], pandas_df.iloc[1:2, 0])

        # DataFrame
        df_equals(modin_df.iloc[[1, 2]], pandas_df.iloc[[1, 2]])
        # See issue #80
        # df_equals(modin_df.iloc[[1, 2], [1, 0]], pandas_df.iloc[[1, 2], [1, 0]])
        df_equals(modin_df.iloc[1:2, 0:2], pandas_df.iloc[1:2, 0:2])

        # Issue #43
        modin_df.iloc[0:3, :]

        # Write Item
        modin_df.iloc[[1, 2]] = 42
        pandas_df.iloc[[1, 2]] = 42
        df_equals(modin_df, pandas_df)

        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)
        modin_df.iloc[0] = modin_df.iloc[1]
        pandas_df.iloc[0] = pandas_df.iloc[1]
        df_equals(modin_df, pandas_df)

        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)
        modin_df.iloc[:, 0] = modin_df.iloc[:, 1]
        pandas_df.iloc[:, 0] = pandas_df.iloc[:, 1]
        df_equals(modin_df, pandas_df)

        # From issue #1775
        df_equals(
            modin_df.iloc[lambda df: df.index.get_indexer_for(df.index[:5])],
            pandas_df.iloc[lambda df: df.index.get_indexer_for(df.index[:5])],
        )
    else:
        with pytest.raises(IndexError):
            modin_df.iloc[0, 1]


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_index(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    df_equals(modin_df.index, pandas_df.index)
    modin_df_cp = modin_df.copy()
    pandas_df_cp = pandas_df.copy()

    modin_df_cp.index = [str(i) for i in modin_df_cp.index]
    pandas_df_cp.index = [str(i) for i in pandas_df_cp.index]
    df_equals(modin_df_cp.index, pandas_df_cp.index)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_indexing_duplicate_axis(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    modin_df.index = pandas_df.index = [i // 3 for i in range(len(modin_df))]
    assert any(modin_df.index.duplicated())
    assert any(pandas_df.index.duplicated())

    df_equals(modin_df.iloc[0], pandas_df.iloc[0])
    df_equals(modin_df.loc[0], pandas_df.loc[0])
    df_equals(modin_df.iloc[0, 0:4], pandas_df.iloc[0, 0:4])
    df_equals(
        modin_df.loc[0, modin_df.columns[0:4]],
        pandas_df.loc[0, pandas_df.columns[0:4]],
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_keys(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    df_equals(modin_df.keys(), pandas_df.keys())


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_loc(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    key1 = modin_df.columns[0]
    key2 = modin_df.columns[1]
    # Scaler
    df_equals(modin_df.loc[0, key1], pandas_df.loc[0, key1])

    # Series
    df_equals(modin_df.loc[0], pandas_df.loc[0])
    df_equals(modin_df.loc[1:, key1], pandas_df.loc[1:, key1])
    df_equals(modin_df.loc[1:2, key1], pandas_df.loc[1:2, key1])

    # DataFrame
    df_equals(modin_df.loc[[1, 2]], pandas_df.loc[[1, 2]])

    # List-like of booleans
    indices = [i % 3 == 0 for i in range(len(modin_df.index))]
    columns = [i % 5 == 0 for i in range(len(modin_df.columns))]
    modin_result = modin_df.loc[indices, columns]
    pandas_result = pandas_df.loc[indices, columns]
    df_equals(modin_result, pandas_result)

    modin_result = modin_df.loc[:, columns]
    pandas_result = pandas_df.loc[:, columns]
    df_equals(modin_result, pandas_result)

    modin_result = modin_df.loc[indices]
    pandas_result = pandas_df.loc[indices]
    df_equals(modin_result, pandas_result)

    # See issue #80
    # df_equals(modin_df.loc[[1, 2], ['col1']], pandas_df.loc[[1, 2], ['col1']])
    df_equals(modin_df.loc[1:2, key1:key2], pandas_df.loc[1:2, key1:key2])

    # From issue #421
    df_equals(modin_df.loc[:, [key2, key1]], pandas_df.loc[:, [key2, key1]])
    df_equals(modin_df.loc[[2, 1], :], pandas_df.loc[[2, 1], :])

    # From issue #1023
    key1 = modin_df.columns[0]
    key2 = modin_df.columns[-2]
    df_equals(modin_df.loc[:, key1:key2], pandas_df.loc[:, key1:key2])

    # Write Item
    modin_df_copy = modin_df.copy()
    pandas_df_copy = pandas_df.copy()
    modin_df_copy.loc[[1, 2]] = 42
    pandas_df_copy.loc[[1, 2]] = 42
    df_equals(modin_df_copy, pandas_df_copy)

    # From issue #1775
    df_equals(
        modin_df.loc[lambda df: df.iloc[:, 0].isin(list(range(1000)))],
        pandas_df.loc[lambda df: df.iloc[:, 0].isin(list(range(1000)))],
    )

    # From issue #1374
    with pytest.raises(KeyError):
        modin_df.loc["NO_EXIST"]


def test_loc_multi_index():
    modin_df = pd.read_csv(
        "modin/pandas/test/data/blah.csv", header=[0, 1, 2, 3], index_col=0
    )
    pandas_df = pandas.read_csv(
        "modin/pandas/test/data/blah.csv", header=[0, 1, 2, 3], index_col=0
    )

    df_equals(modin_df.loc[1], pandas_df.loc[1])
    df_equals(modin_df.loc[1, "Presidents"], pandas_df.loc[1, "Presidents"])
    df_equals(
        modin_df.loc[1, ("Presidents", "Pure mentions")],
        pandas_df.loc[1, ("Presidents", "Pure mentions")],
    )
    assert (
        modin_df.loc[1, ("Presidents", "Pure mentions", "IND", "all")]
        == pandas_df.loc[1, ("Presidents", "Pure mentions", "IND", "all")]
    )
    df_equals(modin_df.loc[(1, 2), "Presidents"], pandas_df.loc[(1, 2), "Presidents"])

    tuples = [
        ("bar", "one"),
        ("bar", "two"),
        ("bar", "three"),
        ("bar", "four"),
        ("baz", "one"),
        ("baz", "two"),
        ("baz", "three"),
        ("baz", "four"),
        ("foo", "one"),
        ("foo", "two"),
        ("foo", "three"),
        ("foo", "four"),
        ("qux", "one"),
        ("qux", "two"),
        ("qux", "three"),
        ("qux", "four"),
    ]

    modin_index = pd.MultiIndex.from_tuples(tuples, names=["first", "second"])
    pandas_index = pandas.MultiIndex.from_tuples(tuples, names=["first", "second"])
    frame_data = np.random.randint(0, 100, size=(16, 100))
    modin_df = pd.DataFrame(
        frame_data,
        index=modin_index,
        columns=["col{}".format(i) for i in range(100)],
    )
    pandas_df = pandas.DataFrame(
        frame_data,
        index=pandas_index,
        columns=["col{}".format(i) for i in range(100)],
    )
    df_equals(modin_df.loc["bar", "col1"], pandas_df.loc["bar", "col1"])
    assert modin_df.loc[("bar", "one"), "col1"] == pandas_df.loc[("bar", "one"), "col1"]
    df_equals(
        modin_df.loc["bar", ("col1", "col2")],
        pandas_df.loc["bar", ("col1", "col2")],
    )

    # From issue #1456
    transposed_modin = modin_df.T
    transposed_pandas = pandas_df.T
    df_equals(
        transposed_modin.loc[transposed_modin.index[:-2], :],
        transposed_pandas.loc[transposed_pandas.index[:-2], :],
    )

    # From issue #1610
    df_equals(modin_df.loc[modin_df.index], pandas_df.loc[pandas_df.index])
    df_equals(modin_df.loc[modin_df.index[:7]], pandas_df.loc[pandas_df.index[:7]])


@pytest.mark.parametrize("index", [["row1", "row2", "row3"]])
@pytest.mark.parametrize("columns", [["col1", "col2"]])
def test_loc_assignment(index, columns):
    md_df, pd_df = create_test_dfs(index=index, columns=columns)
    for i, ind in enumerate(index):
        for j, col in enumerate(columns):
            value_to_assign = int(str(i) + str(j))
            md_df.loc[ind][col] = value_to_assign
            pd_df.loc[ind][col] = value_to_assign
    df_equals(md_df, pd_df)


@pytest.fixture
def loc_iter_dfs():
    columns = ["col1", "col2", "col3"]
    index = ["row1", "row2", "row3"]
    return create_test_dfs(
        {col: ([idx] * len(index)) for idx, col in enumerate(columns)},
        columns=columns,
        index=index,
    )


@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize("axis", [0, 1])
def test_loc_iter_assignment(loc_iter_dfs, reverse_order, axis):
    if reverse_order and axis:
        pytest.xfail(
            "Due to internal sorting of lookup values assignment order is lost, see GH-#2552"
        )

    md_df, pd_df = loc_iter_dfs

    select = [slice(None), slice(None)]
    select[axis] = sorted(pd_df.axes[axis][:-1], reverse=reverse_order)
    select = tuple(select)

    pd_df.loc[select] = pd_df.loc[select] + pd_df.loc[select]
    md_df.loc[select] = md_df.loc[select] + md_df.loc[select]
    df_equals(md_df, pd_df)


@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize("axis", [0, 1])
def test_loc_order(loc_iter_dfs, reverse_order, axis):
    md_df, pd_df = loc_iter_dfs

    select = [slice(None), slice(None)]
    select[axis] = sorted(pd_df.axes[axis][:-1], reverse=reverse_order)
    select = tuple(select)

    df_equals(pd_df.loc[select], md_df.loc[select])


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_loc_nested_assignment(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    key1 = modin_df.columns[0]
    key2 = modin_df.columns[1]

    modin_df[key1].loc[0] = 500
    pandas_df[key1].loc[0] = 500
    df_equals(modin_df, pandas_df)

    modin_df[key2].loc[0] = None
    pandas_df[key2].loc[0] = None
    df_equals(modin_df, pandas_df)


def test_iloc_assignment():
    modin_df = pd.DataFrame(index=["row1", "row2", "row3"], columns=["col1", "col2"])
    pandas_df = pandas.DataFrame(
        index=["row1", "row2", "row3"], columns=["col1", "col2"]
    )
    modin_df.iloc[0]["col1"] = 11
    modin_df.iloc[1]["col1"] = 21
    modin_df.iloc[2]["col1"] = 31
    modin_df.iloc[0]["col2"] = 12
    modin_df.iloc[1]["col2"] = 22
    modin_df.iloc[2]["col2"] = 32
    pandas_df.iloc[0]["col1"] = 11
    pandas_df.iloc[1]["col1"] = 21
    pandas_df.iloc[2]["col1"] = 31
    pandas_df.iloc[0]["col2"] = 12
    pandas_df.iloc[1]["col2"] = 22
    pandas_df.iloc[2]["col2"] = 32
    df_equals(modin_df, pandas_df)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_iloc_nested_assignment(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    key1 = modin_df.columns[0]
    key2 = modin_df.columns[1]

    modin_df[key1].iloc[0] = 500
    pandas_df[key1].iloc[0] = 500
    df_equals(modin_df, pandas_df)

    modin_df[key2].iloc[0] = None
    pandas_df[key2].iloc[0] = None
    df_equals(modin_df, pandas_df)


def test_loc_series():
    md_df, pd_df = create_test_dfs({"a": [1, 2], "b": [3, 4]})

    pd_df.loc[pd_df["a"] > 1, "b"] = np.log(pd_df["b"])
    md_df.loc[md_df["a"] > 1, "b"] = np.log(md_df["b"])

    df_equals(pd_df, md_df)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_pop(request, data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    if "empty_data" not in request.node.name:
        key = modin_df.columns[0]
        temp_modin_df = modin_df.copy()
        temp_pandas_df = pandas_df.copy()
        modin_popped = temp_modin_df.pop(key)
        pandas_popped = temp_pandas_df.pop(key)
        df_equals(modin_popped, pandas_popped)
        df_equals(temp_modin_df, temp_pandas_df)


def test_reindex():
    frame_data = {
        "col1": [0, 1, 2, 3],
        "col2": [4, 5, 6, 7],
        "col3": [8, 9, 10, 11],
        "col4": [12, 13, 14, 15],
        "col5": [0, 0, 0, 0],
    }
    pandas_df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)

    df_equals(modin_df.reindex([0, 3, 2, 1]), pandas_df.reindex([0, 3, 2, 1]))
    df_equals(modin_df.reindex([0, 6, 2]), pandas_df.reindex([0, 6, 2]))
    df_equals(
        modin_df.reindex(["col1", "col3", "col4", "col2"], axis=1),
        pandas_df.reindex(["col1", "col3", "col4", "col2"], axis=1),
    )
    df_equals(
        modin_df.reindex(["col1", "col7", "col4", "col8"], axis=1),
        pandas_df.reindex(["col1", "col7", "col4", "col8"], axis=1),
    )
    df_equals(
        modin_df.reindex(index=[0, 1, 5], columns=["col1", "col7", "col4", "col8"]),
        pandas_df.reindex(index=[0, 1, 5], columns=["col1", "col7", "col4", "col8"]),
    )
    df_equals(
        modin_df.T.reindex(["col1", "col7", "col4", "col8"], axis=0),
        pandas_df.T.reindex(["col1", "col7", "col4", "col8"], axis=0),
    )


def test_reindex_like():
    df1 = pd.DataFrame(
        [
            [24.3, 75.7, "high"],
            [31, 87.8, "high"],
            [22, 71.6, "medium"],
            [35, 95, "medium"],
        ],
        columns=["temp_celsius", "temp_fahrenheit", "windspeed"],
        index=pd.date_range(start="2014-02-12", end="2014-02-15", freq="D"),
    )
    df2 = pd.DataFrame(
        [[28, "low"], [30, "low"], [35.1, "medium"]],
        columns=["temp_celsius", "windspeed"],
        index=pd.DatetimeIndex(["2014-02-12", "2014-02-13", "2014-02-15"]),
    )
    with pytest.warns(UserWarning):
        df2.reindex_like(df1)


def test_rename_sanity():
    source_df = pandas.DataFrame(test_data["int_data"])[
        ["col1", "index", "col3", "col4"]
    ]
    mapping = {"col1": "a", "index": "b", "col3": "c", "col4": "d"}

    modin_df = pd.DataFrame(source_df)
    df_equals(modin_df.rename(columns=mapping), source_df.rename(columns=mapping))

    renamed2 = source_df.rename(columns=str.lower)
    df_equals(modin_df.rename(columns=str.lower), renamed2)

    modin_df = pd.DataFrame(renamed2)
    df_equals(modin_df.rename(columns=str.upper), renamed2.rename(columns=str.upper))

    # index
    data = {"A": {"foo": 0, "bar": 1}}

    # gets sorted alphabetical
    df = pandas.DataFrame(data)
    modin_df = pd.DataFrame(data)
    assert_index_equal(
        modin_df.rename(index={"foo": "bar", "bar": "foo"}).index,
        df.rename(index={"foo": "bar", "bar": "foo"}).index,
    )

    assert_index_equal(
        modin_df.rename(index=str.upper).index, df.rename(index=str.upper).index
    )

    # Using the `mapper` functionality with `axis`
    assert_index_equal(
        modin_df.rename(str.upper, axis=0).index, df.rename(str.upper, axis=0).index
    )
    assert_index_equal(
        modin_df.rename(str.upper, axis=1).columns,
        df.rename(str.upper, axis=1).columns,
    )

    # have to pass something
    with pytest.raises(TypeError):
        modin_df.rename()

    # partial columns
    renamed = source_df.rename(columns={"col3": "foo", "col4": "bar"})
    modin_df = pd.DataFrame(source_df)
    assert_index_equal(
        modin_df.rename(columns={"col3": "foo", "col4": "bar"}).index,
        source_df.rename(columns={"col3": "foo", "col4": "bar"}).index,
    )

    # other axis
    renamed = source_df.T.rename(index={"col3": "foo", "col4": "bar"})
    assert_index_equal(
        source_df.T.rename(index={"col3": "foo", "col4": "bar"}).index,
        modin_df.T.rename(index={"col3": "foo", "col4": "bar"}).index,
    )

    # index with name
    index = pandas.Index(["foo", "bar"], name="name")
    renamer = pandas.DataFrame(data, index=index)
    modin_df = pd.DataFrame(data, index=index)

    renamed = renamer.rename(index={"foo": "bar", "bar": "foo"})
    modin_renamed = modin_df.rename(index={"foo": "bar", "bar": "foo"})
    assert_index_equal(renamed.index, modin_renamed.index)

    assert renamed.index.name == modin_renamed.index.name


def test_rename_multiindex():
    tuples_index = [("foo1", "bar1"), ("foo2", "bar2")]
    tuples_columns = [("fizz1", "buzz1"), ("fizz2", "buzz2")]
    index = pandas.MultiIndex.from_tuples(tuples_index, names=["foo", "bar"])
    columns = pandas.MultiIndex.from_tuples(tuples_columns, names=["fizz", "buzz"])

    frame_data = [(0, 0), (1, 1)]
    df = pandas.DataFrame(frame_data, index=index, columns=columns)
    modin_df = pd.DataFrame(frame_data, index=index, columns=columns)

    #
    # without specifying level -> accross all levels
    renamed = df.rename(
        index={"foo1": "foo3", "bar2": "bar3"},
        columns={"fizz1": "fizz3", "buzz2": "buzz3"},
    )
    modin_renamed = modin_df.rename(
        index={"foo1": "foo3", "bar2": "bar3"},
        columns={"fizz1": "fizz3", "buzz2": "buzz3"},
    )
    assert_index_equal(renamed.index, modin_renamed.index)

    renamed = df.rename(
        index={"foo1": "foo3", "bar2": "bar3"},
        columns={"fizz1": "fizz3", "buzz2": "buzz3"},
    )
    assert_index_equal(renamed.columns, modin_renamed.columns)
    assert renamed.index.names == modin_renamed.index.names
    assert renamed.columns.names == modin_renamed.columns.names

    #
    # with specifying a level

    # dict
    renamed = df.rename(columns={"fizz1": "fizz3", "buzz2": "buzz3"}, level=0)
    modin_renamed = modin_df.rename(
        columns={"fizz1": "fizz3", "buzz2": "buzz3"}, level=0
    )
    assert_index_equal(renamed.columns, modin_renamed.columns)
    renamed = df.rename(columns={"fizz1": "fizz3", "buzz2": "buzz3"}, level="fizz")
    modin_renamed = modin_df.rename(
        columns={"fizz1": "fizz3", "buzz2": "buzz3"}, level="fizz"
    )
    assert_index_equal(renamed.columns, modin_renamed.columns)

    renamed = df.rename(columns={"fizz1": "fizz3", "buzz2": "buzz3"}, level=1)
    modin_renamed = modin_df.rename(
        columns={"fizz1": "fizz3", "buzz2": "buzz3"}, level=1
    )
    assert_index_equal(renamed.columns, modin_renamed.columns)
    renamed = df.rename(columns={"fizz1": "fizz3", "buzz2": "buzz3"}, level="buzz")
    modin_renamed = modin_df.rename(
        columns={"fizz1": "fizz3", "buzz2": "buzz3"}, level="buzz"
    )
    assert_index_equal(renamed.columns, modin_renamed.columns)

    # function
    func = str.upper
    renamed = df.rename(columns=func, level=0)
    modin_renamed = modin_df.rename(columns=func, level=0)
    assert_index_equal(renamed.columns, modin_renamed.columns)
    renamed = df.rename(columns=func, level="fizz")
    modin_renamed = modin_df.rename(columns=func, level="fizz")
    assert_index_equal(renamed.columns, modin_renamed.columns)

    renamed = df.rename(columns=func, level=1)
    modin_renamed = modin_df.rename(columns=func, level=1)
    assert_index_equal(renamed.columns, modin_renamed.columns)
    renamed = df.rename(columns=func, level="buzz")
    modin_renamed = modin_df.rename(columns=func, level="buzz")
    assert_index_equal(renamed.columns, modin_renamed.columns)

    # index
    renamed = df.rename(index={"foo1": "foo3", "bar2": "bar3"}, level=0)
    modin_renamed = modin_df.rename(index={"foo1": "foo3", "bar2": "bar3"}, level=0)
    assert_index_equal(modin_renamed.index, renamed.index)


@pytest.mark.skip(reason="Pandas does not pass this test")
def test_rename_nocopy():
    source_df = pandas.DataFrame(test_data["int_data"])[
        ["col1", "index", "col3", "col4"]
    ]
    modin_df = pd.DataFrame(source_df)
    modin_renamed = modin_df.rename(columns={"col3": "foo"}, copy=False)
    modin_renamed["foo"] = 1
    assert (modin_df["col3"] == 1).all()


def test_rename_inplace():
    source_df = pandas.DataFrame(test_data["int_data"])[
        ["col1", "index", "col3", "col4"]
    ]
    modin_df = pd.DataFrame(source_df)

    df_equals(
        modin_df.rename(columns={"col3": "foo"}),
        source_df.rename(columns={"col3": "foo"}),
    )

    frame = source_df.copy()
    modin_frame = modin_df.copy()
    frame.rename(columns={"col3": "foo"}, inplace=True)
    modin_frame.rename(columns={"col3": "foo"}, inplace=True)

    df_equals(modin_frame, frame)


def test_rename_bug():
    # rename set ref_locs, and set_index was not resetting
    frame_data = {0: ["foo", "bar"], 1: ["bah", "bas"], 2: [1, 2]}
    df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)
    df = df.rename(columns={0: "a"})
    df = df.rename(columns={1: "b"})
    df = df.set_index(["a", "b"])
    df.columns = ["2001-01-01"]

    modin_df = modin_df.rename(columns={0: "a"})
    modin_df = modin_df.rename(columns={1: "b"})
    modin_df = modin_df.set_index(["a", "b"])
    modin_df.columns = ["2001-01-01"]

    df_equals(modin_df, df)


def test_rename_axis():
    data = {"num_legs": [4, 4, 2], "num_arms": [0, 0, 2]}
    index = ["dog", "cat", "monkey"]
    modin_df = pd.DataFrame(data, index)
    pandas_df = pandas.DataFrame(data, index)
    df_equals(modin_df.rename_axis("animal"), pandas_df.rename_axis("animal"))
    df_equals(
        modin_df.rename_axis("limbs", axis="columns"),
        pandas_df.rename_axis("limbs", axis="columns"),
    )

    modin_df.rename_axis("limbs", axis="columns", inplace=True)
    pandas_df.rename_axis("limbs", axis="columns", inplace=True)
    df_equals(modin_df, pandas_df)

    new_index = pd.MultiIndex.from_product(
        [["mammal"], ["dog", "cat", "monkey"]], names=["type", "name"]
    )
    modin_df.index = new_index
    pandas_df.index = new_index

    df_equals(
        modin_df.rename_axis(index={"type": "class"}),
        pandas_df.rename_axis(index={"type": "class"}),
    )
    df_equals(
        modin_df.rename_axis(columns=str.upper),
        pandas_df.rename_axis(columns=str.upper),
    )
    df_equals(
        modin_df.rename_axis(columns=[str.upper(o) for o in modin_df.columns.names]),
        pandas_df.rename_axis(columns=[str.upper(o) for o in pandas_df.columns.names]),
    )

    with pytest.raises(ValueError):
        df_equals(
            modin_df.rename_axis(str.upper, axis=1),
            pandas_df.rename_axis(str.upper, axis=1),
        )


def test_rename_axis_inplace():
    test_frame = pandas.DataFrame(test_data["int_data"])
    modin_df = pd.DataFrame(test_frame)

    result = test_frame.copy()
    modin_result = modin_df.copy()
    no_return = result.rename_axis("foo", inplace=True)
    modin_no_return = modin_result.rename_axis("foo", inplace=True)

    assert no_return is modin_no_return
    df_equals(modin_result, result)

    result = test_frame.copy()
    modin_result = modin_df.copy()
    no_return = result.rename_axis("bar", axis=1, inplace=True)
    modin_no_return = modin_result.rename_axis("bar", axis=1, inplace=True)

    assert no_return is modin_no_return
    df_equals(modin_result, result)


def test_reorder_levels():
    data = np.random.randint(1, 100, 12)
    modin_df = pd.DataFrame(
        data,
        index=pd.MultiIndex.from_tuples(
            [
                (num, letter, color)
                for num in range(1, 3)
                for letter in ["a", "b", "c"]
                for color in ["Red", "Green"]
            ],
            names=["Number", "Letter", "Color"],
        ),
    )
    pandas_df = pandas.DataFrame(
        data,
        index=pandas.MultiIndex.from_tuples(
            [
                (num, letter, color)
                for num in range(1, 3)
                for letter in ["a", "b", "c"]
                for color in ["Red", "Green"]
            ],
            names=["Number", "Letter", "Color"],
        ),
    )
    df_equals(
        modin_df.reorder_levels(["Letter", "Color", "Number"]),
        pandas_df.reorder_levels(["Letter", "Color", "Number"]),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_reset_index(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    modin_result = modin_df.reset_index(inplace=False)
    pandas_result = pandas_df.reset_index(inplace=False)
    df_equals(modin_result, pandas_result)

    modin_df_cp = modin_df.copy()
    pd_df_cp = pandas_df.copy()
    modin_df_cp.reset_index(inplace=True)
    pd_df_cp.reset_index(inplace=True)
    df_equals(modin_df_cp, pd_df_cp)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
def test_sample(data, axis):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    with pytest.raises(ValueError):
        modin_df.sample(n=3, frac=0.4, axis=axis)

    with pytest.raises(KeyError):
        modin_df.sample(frac=0.5, weights="CoLuMn_No_ExIsT", axis=0)

    with pytest.raises(ValueError):
        modin_df.sample(frac=0.5, weights=modin_df.columns[0], axis=1)

    with pytest.raises(ValueError):
        modin_df.sample(
            frac=0.5, weights=[0.5 for _ in range(len(modin_df.index[:-1]))], axis=0
        )

    with pytest.raises(ValueError):
        modin_df.sample(
            frac=0.5,
            weights=[0.5 for _ in range(len(modin_df.columns[:-1]))],
            axis=1,
        )

    with pytest.raises(ValueError):
        modin_df.sample(n=-3, axis=axis)

    with pytest.raises(ValueError):
        modin_df.sample(frac=0.2, weights=pandas.Series(), axis=axis)

    if isinstance(axis, str):
        num_axis = pandas.DataFrame()._get_axis_number(axis)
    else:
        num_axis = axis

    # weights that sum to 1
    sums = sum(i % 2 for i in range(len(modin_df.axes[num_axis])))
    weights = [i % 2 / sums for i in range(len(modin_df.axes[num_axis]))]

    modin_result = modin_df.sample(
        frac=0.5, random_state=42, weights=weights, axis=axis
    )
    pandas_result = pandas_df.sample(
        frac=0.5, random_state=42, weights=weights, axis=axis
    )
    df_equals(modin_result, pandas_result)

    # weights that don't sum to 1
    weights = [i % 2 for i in range(len(modin_df.axes[num_axis]))]
    modin_result = modin_df.sample(
        frac=0.5, random_state=42, weights=weights, axis=axis
    )
    pandas_result = pandas_df.sample(
        frac=0.5, random_state=42, weights=weights, axis=axis
    )
    df_equals(modin_result, pandas_result)

    modin_result = modin_df.sample(n=0, axis=axis)
    pandas_result = pandas_df.sample(n=0, axis=axis)
    df_equals(modin_result, pandas_result)

    modin_result = modin_df.sample(frac=0.5, random_state=42, axis=axis)
    pandas_result = pandas_df.sample(frac=0.5, random_state=42, axis=axis)
    df_equals(modin_result, pandas_result)

    modin_result = modin_df.sample(n=2, random_state=42, axis=axis)
    pandas_result = pandas_df.sample(n=2, random_state=42, axis=axis)
    df_equals(modin_result, pandas_result)

    # issue #1692, numpy RandomState object
    # We must create a new random state for each iteration because the values that
    # are selected will be impacted if the object has already been used.
    random_state = np.random.RandomState(42)
    modin_result = modin_df.sample(frac=0.5, random_state=random_state, axis=axis)

    random_state = np.random.RandomState(42)
    pandas_result = pandas_df.sample(frac=0.5, random_state=random_state, axis=axis)
    df_equals(modin_result, pandas_result)


def test_select_dtypes():
    frame_data = {
        "test1": list("abc"),
        "test2": np.arange(3, 6).astype("u1"),
        "test3": np.arange(8.0, 11.0, dtype="float64"),
        "test4": [True, False, True],
        "test5": pandas.date_range("now", periods=3).values,
        "test6": list(range(5, 8)),
    }
    df = pandas.DataFrame(frame_data)
    rd = pd.DataFrame(frame_data)

    include = np.float, "integer"
    exclude = (np.bool_,)
    r = rd.select_dtypes(include=include, exclude=exclude)

    e = df[["test2", "test3", "test6"]]
    df_equals(r, e)

    r = rd.select_dtypes(include=np.bool_)
    e = df[["test4"]]
    df_equals(r, e)

    r = rd.select_dtypes(exclude=np.bool_)
    e = df[["test1", "test2", "test3", "test5", "test6"]]
    df_equals(r, e)

    try:
        pd.DataFrame().select_dtypes()
        assert False
    except ValueError:
        assert True


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("n", int_arg_values, ids=arg_keys("n", int_arg_keys))
def test_tail(data, n):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    df_equals(modin_df.tail(n), pandas_df.tail(n))
    df_equals(modin_df.tail(len(modin_df)), pandas_df.tail(len(pandas_df)))


def test_xs():
    d = {
        "num_legs": [4, 4, 2, 2],
        "num_wings": [0, 0, 2, 2],
        "class": ["mammal", "mammal", "mammal", "bird"],
        "animal": ["cat", "dog", "bat", "penguin"],
        "locomotion": ["walks", "walks", "flies", "walks"],
    }
    df = pd.DataFrame(data=d)
    df = df.set_index(["class", "animal", "locomotion"])
    with pytest.warns(UserWarning):
        df.xs("mammal")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___getitem__(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    key = modin_df.columns[0]
    modin_col = modin_df.__getitem__(key)
    assert isinstance(modin_col, pd.Series)

    pd_col = pandas_df[key]
    df_equals(pd_col, modin_col)

    slices = [
        (None, -1),
        (-1, None),
        (1, 2),
        (1, None),
        (None, 1),
        (1, -1),
        (-3, -1),
        (1, -1, 2),
    ]

    # slice test
    for slice_param in slices:
        s = slice(*slice_param)
        df_equals(modin_df[s], pandas_df[s])

    # Test empty
    df_equals(pd.DataFrame([])[:10], pandas.DataFrame([])[:10])


def test_getitem_empty_mask():
    # modin-project/modin#517
    modin_frames = []
    pandas_frames = []
    data1 = np.random.randint(0, 100, size=(100, 4))
    mdf1 = pd.DataFrame(data1, columns=list("ABCD"))
    pdf1 = pandas.DataFrame(data1, columns=list("ABCD"))
    modin_frames.append(mdf1)
    pandas_frames.append(pdf1)

    data2 = np.random.randint(0, 100, size=(100, 4))
    mdf2 = pd.DataFrame(data2, columns=list("ABCD"))
    pdf2 = pandas.DataFrame(data2, columns=list("ABCD"))
    modin_frames.append(mdf2)
    pandas_frames.append(pdf2)

    data3 = np.random.randint(0, 100, size=(100, 4))
    mdf3 = pd.DataFrame(data3, columns=list("ABCD"))
    pdf3 = pandas.DataFrame(data3, columns=list("ABCD"))
    modin_frames.append(mdf3)
    pandas_frames.append(pdf3)

    modin_data = pd.concat(modin_frames)
    pandas_data = pandas.concat(pandas_frames)
    df_equals(
        modin_data[[False for _ in modin_data.index]],
        pandas_data[[False for _ in modin_data.index]],
    )


def test_getitem_datetime_slice():
    data = {"data": range(1000)}
    index = pd.date_range("2017/1/4", periods=1000)
    modin_df = pd.DataFrame(data=data, index=index)
    pandas_df = pandas.DataFrame(data=data, index=index)

    s = slice("2017-01-06", "2017-01-09")
    df_equals(modin_df[s], pandas_df[s])


def test_getitem_same_name():
    data = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
        [17, 18, 19, 20],
    ]
    columns = ["c1", "c2", "c1", "c3"]
    modin_df = pd.DataFrame(data, columns=columns)
    pandas_df = pandas.DataFrame(data, columns=columns)
    df_equals(modin_df["c1"], pandas_df["c1"])
    df_equals(modin_df["c2"], pandas_df["c2"])
    df_equals(modin_df[["c1", "c2"]], pandas_df[["c1", "c2"]])
    df_equals(modin_df["c3"], pandas_df["c3"])


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___getattr__(request, data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)  # noqa F841

    if "empty_data" not in request.node.name:
        key = modin_df.columns[0]
        col = modin_df.__getattr__(key)

        col = modin_df.__getattr__("col1")
        assert isinstance(col, pd.Series)

        col = getattr(modin_df, "col1")
        assert isinstance(col, pd.Series)

        # Check that lookup in column doesn't override other attributes
        df2 = modin_df.rename(index=str, columns={key: "columns"})
        assert isinstance(df2.columns, pandas.Index)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___setitem__(data):
    eval_setitem(*create_test_dfs(data), loc=-1, value=1)
    eval_setitem(
        *create_test_dfs(data), loc=-1, value=lambda df: type(df)(df[df.columns[0]])
    )

    nrows = len(data[list(data.keys())[0]])
    arr = np.arange(nrows * 2).reshape(-1, 2)

    eval_setitem(*create_test_dfs(data), loc=-1, value=arr)
    eval_setitem(*create_test_dfs(data), col="___NON EXISTENT COLUMN", value=arr)
    eval_setitem(*create_test_dfs(data), loc=0, value=np.arange(nrows))

    modin_df = pd.DataFrame(columns=data.keys())
    pandas_df = pandas.DataFrame(columns=data.keys())

    for col in modin_df.columns:
        modin_df[col] = np.arange(1000)

    for col in pandas_df.columns:
        pandas_df[col] = np.arange(1000)

    df_equals(modin_df, pandas_df)

    # Test series assignment to column
    modin_df = pd.DataFrame(columns=modin_df.columns)
    pandas_df = pandas.DataFrame(columns=pandas_df.columns)
    modin_df[modin_df.columns[-1]] = modin_df[modin_df.columns[0]]
    pandas_df[pandas_df.columns[-1]] = pandas_df[pandas_df.columns[0]]
    df_equals(modin_df, pandas_df)

    if not sys.version_info.major == 3 and sys.version_info.minor > 6:
        # This test doesn't work correctly on Python 3.6
        # Test 2d ndarray assignment to column
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)
        modin_df["new_col"] = modin_df[[modin_df.columns[0]]].values
        pandas_df["new_col"] = pandas_df[[pandas_df.columns[0]]].values
        df_equals(modin_df, pandas_df)
        assert isinstance(modin_df["new_col"][0], type(pandas_df["new_col"][0]))

    # Transpose test
    modin_df = pd.DataFrame(data).T
    pandas_df = pandas.DataFrame(data).T

    # We default to pandas on non-string column names
    if not all(isinstance(c, str) for c in modin_df.columns):
        with pytest.warns(UserWarning):
            modin_df[modin_df.columns[0]] = 0
    else:
        modin_df[modin_df.columns[0]] = 0

    pandas_df[pandas_df.columns[0]] = 0

    df_equals(modin_df, pandas_df)

    modin_df.columns = [str(i) for i in modin_df.columns]
    pandas_df.columns = [str(i) for i in pandas_df.columns]

    modin_df[modin_df.columns[0]] = 0
    pandas_df[pandas_df.columns[0]] = 0

    df_equals(modin_df, pandas_df)

    modin_df[modin_df.columns[0]][modin_df.index[0]] = 12345
    pandas_df[pandas_df.columns[0]][pandas_df.index[0]] = 12345

    df_equals(modin_df, pandas_df)

    # from issue #2390
    modin_df = pd.DataFrame({"a": [1, 2, 3]})
    pandas_df = pandas.DataFrame({"a": [1, 2, 3]})
    modin_df["b"] = pd.Series([4, 5, 6, 7, 8])
    pandas_df["b"] = pandas.Series([4, 5, 6, 7, 8])
    df_equals(modin_df, pandas_df)

    # from issue #2442
    data = {"a": [1, 2, 3, 4]}
    # Index with duplicated timestamp
    index = pandas.to_datetime(["2020-02-06", "2020-02-06", "2020-02-22", "2020-03-26"])

    md_df, pd_df = create_test_dfs(data, index=index)
    # Setting new column
    pd_df["b"] = pandas.Series(np.arange(4))
    md_df["b"] = pd.Series(np.arange(4))

    df_equals(md_df, pd_df)
    # Setting existing column
    pd_df["b"] = pandas.Series(np.arange(4))
    md_df["b"] = pd.Series(np.arange(4))

    df_equals(md_df, pd_df)


def test___setitem__with_mismatched_partitions():
    fname = "200kx99.csv"
    np.savetxt(fname, np.random.randint(0, 100, size=(200_000, 99)), delimiter=",")
    modin_df = pd.read_csv(fname)
    pandas_df = pandas.read_csv(fname)
    modin_df["new"] = pd.Series(list(range(len(modin_df))))
    pandas_df["new"] = pandas.Series(list(range(len(pandas_df))))
    df_equals(modin_df, pandas_df)


def test___setitem__mask():
    # DataFrame mask:
    data = test_data["int_data"]
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    mean = int((RAND_HIGH + RAND_LOW) / 2)
    pandas_df[pandas_df > mean] = -50
    modin_df[modin_df > mean] = -50

    df_equals(modin_df, pandas_df)

    # Array mask:
    pandas_df = pandas.DataFrame(data)
    modin_df = pd.DataFrame(data)
    array = (pandas_df > mean).to_numpy()

    modin_df[array] = -50
    pandas_df[array] = -50

    df_equals(modin_df, pandas_df)

    # Array mask of wrong size:
    with pytest.raises(ValueError):
        array = np.array([[1, 2], [3, 4]])
        modin_df[array] = 20


@pytest.mark.parametrize(
    "data",
    [
        {},
        pytest.param(
            {"id": [], "max_speed": [], "health": []},
            marks=pytest.mark.xfail(
                reason="Throws an exception because generally assigning Series or other objects of length different from DataFrame does not work right now"
            ),
        ),
    ],
    ids=["empty", "empty_columns"],
)
@pytest.mark.parametrize(
    "value",
    [np.array(["one", "two"]), [11, 22]],
    ids=["ndarray", "list"],
)
@pytest.mark.parametrize("convert_to_series", [False, True])
@pytest.mark.parametrize("new_col_id", [123, "new_col"], ids=["integer", "string"])
def test_setitem_on_empty_df(data, value, convert_to_series, new_col_id):
    pandas_df = pandas.DataFrame(data)
    modin_df = pd.DataFrame(data)

    pandas_df[new_col_id] = pandas.Series(value) if convert_to_series else value
    modin_df[new_col_id] = pd.Series(value) if convert_to_series else value
    df_equals(modin_df, pandas_df)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___len__(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    assert len(modin_df) == len(pandas_df)


def test_index_order():
    # see #1708 and #1869 for details
    df_modin, df_pandas = (
        pd.DataFrame(test_data["float_nan_data"]),
        pandas.DataFrame(test_data["float_nan_data"]),
    )
    rows_number = len(df_modin.index)
    level_0 = np.random.choice([x for x in range(10)], rows_number)
    level_1 = np.random.choice([x for x in range(10)], rows_number)
    index = pandas.MultiIndex.from_arrays([level_0, level_1])

    df_modin.index = index
    df_pandas.index = index

    for func in ["all", "any", "mad", "count"]:
        df_equals(
            getattr(df_modin, func)(level=0).index,
            getattr(df_pandas, func)(level=0).index,
        )
