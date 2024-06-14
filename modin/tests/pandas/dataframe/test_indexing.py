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

import os
import sys

import matplotlib
import numpy as np
import pandas
import pytest
from pandas._testing import ensure_clean

import modin.pandas as pd
from modin.config import MinRowPartitionSize, NPartitions
from modin.pandas.indexing import is_range_like
from modin.pandas.testing import assert_index_equal
from modin.tests.pandas.utils import (
    NROWS,
    RAND_HIGH,
    RAND_LOW,
    arg_keys,
    axis_keys,
    axis_values,
    create_test_dfs,
    default_to_pandas_ignore_string,
    df_equals,
    eval_general,
    generate_multiindex,
    int_arg_keys,
    int_arg_values,
    name_contains,
    test_data,
    test_data_keys,
    test_data_values,
)
from modin.utils import get_current_execution

NPartitions.put(4)

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")

# Our configuration in pytest.ini requires that we explicitly catch all
# instances of defaulting to pandas, but some test modules, like this one,
# have too many such instances.
# TODO(https://github.com/modin-project/modin/issues/3655): catch all instances
# of defaulting to pandas.
pytestmark = pytest.mark.filterwarnings(default_to_pandas_ignore_string)


def eval_setitem(md_df, pd_df, value, col=None, loc=None, expected_exception=None):
    if loc is not None:
        col = pd_df.columns[loc]

    value_getter = value if callable(value) else (lambda *args, **kwargs: value)

    eval_general(
        md_df,
        pd_df,
        lambda df: df.__setitem__(col, value_getter(df)),
        __inplace__=True,
        expected_exception=expected_exception,
    )


def eval_loc(md_df, pd_df, value, key):
    if isinstance(value, tuple):
        assert len(value) == 2
        # case when value for pandas different
        md_value, pd_value = value
    else:
        md_value, pd_value = value, value

    eval_general(
        md_df,
        pd_df,
        lambda df: df.loc.__setitem__(
            key, pd_value if isinstance(df, pandas.DataFrame) else md_value
        ),
        __inplace__=True,
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
    [[60, 70, 90], [60.5, 70.5, 100]],
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

    with pytest.raises(NotImplementedError):
        modin_df.iat()


@pytest.mark.gpu
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_iloc(request, data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    if not name_contains(request.node.name, ["empty_data"]):
        # Scalar
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

        # Read values, selecting rows with callable and a column with a scalar.
        df_equals(
            pandas_df.iloc[lambda df: df.index.get_indexer_for(df.index[:5]), 0],
            modin_df.iloc[lambda df: df.index.get_indexer_for(df.index[:5]), 0],
        )
    else:
        with pytest.raises(IndexError):
            modin_df.iloc[0, 1]


@pytest.mark.gpu
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


@pytest.mark.gpu
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
@pytest.mark.parametrize(
    "key_func",
    [
        # test for the case from https://github.com/modin-project/modin/issues/4308
        lambda df: "non_existing_column",
        lambda df: df.columns[0],
        lambda df: df.index,
        lambda df: [df.index, df.columns[0]],
        lambda df: (
            pandas.Series(list(range(len(df.index))))
            if isinstance(df, pandas.DataFrame)
            else pd.Series(list(range(len(df))))
        ),
    ],
    ids=[
        "non_existing_column",
        "first_column_name",
        "original_index",
        "list_of_index_and_first_column_name",
        "series_of_integers",
    ],
)
@pytest.mark.parametrize(
    "drop_kwargs",
    [{"drop": True}, {"drop": False}, {}],
    ids=["drop_True", "drop_False", "no_drop_param"],
)
def test_set_index(data, key_func, drop_kwargs, request):
    if (
        "list_of_index_and_first_column_name" in request.node.name
        and "drop_False" in request.node.name
    ):
        pytest.xfail(
            reason="KeyError: https://github.com/modin-project/modin/issues/5636"
        )
    expected_exception = None
    if "non_existing_column" in request.node.callspec.id:
        expected_exception = KeyError(
            "None of ['non_existing_column'] are in the columns"
        )
    eval_general(
        *create_test_dfs(data),
        lambda df: df.set_index(key_func(df), **drop_kwargs),
        expected_exception=expected_exception,
    )


@pytest.mark.parametrize("index", ["a", ["a", ("b", "")]])
def test_set_index_with_multiindex(index):
    # see #5186 for details
    kwargs = {"columns": [["a", "b", "c", "d"], ["", "", "x", "y"]]}
    modin_df, pandas_df = create_test_dfs(np.random.rand(2, 4), **kwargs)
    eval_general(modin_df, pandas_df, lambda df: df.set_index(index))


@pytest.mark.gpu
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
    # Scalar
    df_equals(modin_df.loc[0, key1], pandas_df.loc[0, key1])

    # Series
    df_equals(modin_df.loc[0], pandas_df.loc[0])
    df_equals(modin_df.loc[1:, key1], pandas_df.loc[1:, key1])
    df_equals(modin_df.loc[1:2, key1], pandas_df.loc[1:2, key1])
    df_equals(modin_df.loc[:, key1], pandas_df.loc[:, key1])

    # DataFrame
    df_equals(modin_df.loc[[1, 2]], pandas_df.loc[[1, 2]])

    indices = [i % 3 == 0 for i in range(len(modin_df.index))]
    columns = [i % 5 == 0 for i in range(len(modin_df.columns))]

    # Key is a list of booleans
    modin_result = modin_df.loc[indices, columns]
    pandas_result = pandas_df.loc[indices, columns]
    df_equals(modin_result, pandas_result)

    # Key is a Modin or pandas series of booleans
    df_equals(
        modin_df.loc[pd.Series(indices), pd.Series(columns, index=modin_df.columns)],
        pandas_df.loc[
            pandas.Series(indices), pandas.Series(columns, index=modin_df.columns)
        ],
    )

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

    # Write an item, selecting rows with a callable.
    modin_df_copy2 = modin_df.copy()
    pandas_df_copy2 = pandas_df.copy()
    modin_df_copy2.loc[lambda df: df[key1].isin(list(range(1000)))] = 42
    pandas_df_copy2.loc[lambda df: df[key1].isin(list(range(1000)))] = 42
    df_equals(modin_df_copy2, pandas_df_copy2)

    # Write an item, selecting rows with a callable and a column with a scalar.
    modin_df_copy3 = modin_df.copy()
    pandas_df_copy3 = pandas_df.copy()
    modin_df_copy3.loc[lambda df: df[key1].isin(list(range(1000))), key1] = 42
    pandas_df_copy3.loc[lambda df: df[key1].isin(list(range(1000))), key1] = 42
    df_equals(modin_df_copy3, pandas_df_copy3)

    # Disabled for `BaseOnPython` because of the issue with `getitem_array`:
    # https://github.com/modin-project/modin/issues/3701
    if get_current_execution() != "BaseOnPython":
        # From issue #1775
        df_equals(
            modin_df.loc[lambda df: df.iloc[:, 0].isin(list(range(1000)))],
            pandas_df.loc[lambda df: df.iloc[:, 0].isin(list(range(1000)))],
        )

        # Read values, selecting rows with a callable and a column with a scalar.
        df_equals(
            pandas_df.loc[lambda df: df[key1].isin(list(range(1000))), key1],
            modin_df.loc[lambda df: df[key1].isin(list(range(1000))), key1],
        )

    # From issue #1374
    with pytest.raises(KeyError):
        modin_df.loc["NO_EXIST"]


@pytest.mark.parametrize(
    "key_getter, value_getter",
    [
        pytest.param(
            lambda df, axis: (
                (slice(None), df.axes[axis][:2])
                if axis
                else (df.axes[axis][:2], slice(None))
            ),
            lambda df, axis: df.iloc[:, :1] if axis else df.iloc[:1, :],
            id="len(key)_>_len(value)",
        ),
        pytest.param(
            lambda df, axis: (
                (slice(None), df.axes[axis][:2])
                if axis
                else (df.axes[axis][:2], slice(None))
            ),
            lambda df, axis: df.iloc[:, :3] if axis else df.iloc[:3, :],
            id="len(key)_<_len(value)",
        ),
        pytest.param(
            lambda df, axis: (
                (slice(None), df.axes[axis][:2])
                if axis
                else (df.axes[axis][:2], slice(None))
            ),
            lambda df, axis: df.iloc[:, :2] if axis else df.iloc[:2, :],
            id="len(key)_==_len(value)",
        ),
    ],
)
@pytest.mark.parametrize("key_axis", [0, 1])
@pytest.mark.parametrize("reverse_value_index", [True, False])
@pytest.mark.parametrize("reverse_value_columns", [True, False])
def test_loc_4456(
    key_getter, value_getter, key_axis, reverse_value_index, reverse_value_columns
):
    data = test_data["float_nan_data"]
    modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)

    key = key_getter(pandas_df, key_axis)

    # `df.loc` doesn't work right for range-like indexers. Converting them to a list.
    # https://github.com/modin-project/modin/issues/4497
    if is_range_like(key[0]):
        key = (list(key[0]), key[1])
    if is_range_like(key[1]):
        key = (key[0], list(key[1]))

    value = pandas.DataFrame(
        np.random.randint(0, 100, size=pandas_df.shape),
        index=pandas_df.index,
        columns=pandas_df.columns,
    )
    pdf_value = value_getter(value, key_axis)
    mdf_value = value_getter(pd.DataFrame(value), key_axis)

    if reverse_value_index:
        pdf_value = pdf_value.reindex(index=pdf_value.index[::-1])
        mdf_value = mdf_value.reindex(index=mdf_value.index[::-1])
    if reverse_value_columns:
        pdf_value = pdf_value.reindex(columns=pdf_value.columns[::-1])
        mdf_value = mdf_value.reindex(columns=mdf_value.columns[::-1])

    eval_loc(modin_df, pandas_df, pdf_value, key)
    eval_loc(modin_df, pandas_df, (mdf_value, pdf_value), key)


def test_loc_6774():
    modin_df, pandas_df = create_test_dfs(
        {"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]}
    )
    pandas_df.loc[:, "c"] = [10, 20, 30, 40, 51]
    modin_df.loc[:, "c"] = [10, 20, 30, 40, 51]
    df_equals(modin_df, pandas_df)

    pandas_df.loc[2:, "y"] = [30, 40, 51]
    modin_df.loc[2:, "y"] = [30, 40, 51]
    df_equals(modin_df, pandas_df)

    pandas_df.loc[:, ["b", "c", "d"]] = (
        pd.DataFrame([[10, 20, 30, 40, 50], [10, 20, 30, 40], [10, 20, 30]])
        .transpose()
        .values
    )
    modin_df.loc[:, ["b", "c", "d"]] = (
        pd.DataFrame([[10, 20, 30, 40, 50], [10, 20, 30, 40], [10, 20, 30]])
        .transpose()
        .values
    )
    df_equals(modin_df, pandas_df)


def test_loc_5829():
    data = {"a": [1, 2, 3, 4, 5], "b": [11, 12, 13, 14, 15]}
    modin_df = pd.DataFrame(data, dtype=object)
    pandas_df = pandas.DataFrame(data, dtype=object)
    eval_loc(
        modin_df,
        pandas_df,
        value=np.array([[24, 34, 44], [25, 35, 45]]),
        key=([3, 4], ["c", "d", "e"]),
    )


def test_loc_7135():
    data = np.random.randint(0, 100, size=(2**16, 2**8))
    modin_df, pandas_df = create_test_dfs(data)
    key = len(pandas_df)
    eval_loc(
        modin_df,
        pandas_df,
        value=list(range(2**8)),
        key=key,
    )


# This tests the bug from https://github.com/modin-project/modin/issues/3736
def test_loc_setting_single_categorical_column():
    modin_df = pd.DataFrame({"status": ["a", "b", "c"]}, dtype="category")
    pandas_df = pandas.DataFrame({"status": ["a", "b", "c"]}, dtype="category")
    modin_df.loc[1:3, "status"] = "a"
    pandas_df.loc[1:3, "status"] = "a"
    df_equals(modin_df, pandas_df)


def test_loc_multi_index():
    modin_df = pd.read_csv(
        "modin/tests/pandas/data/blah.csv", header=[0, 1, 2, 3], index_col=0
    )
    pandas_df = pandas.read_csv(
        "modin/tests/pandas/data/blah.csv", header=[0, 1, 2, 3], index_col=0
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


def test_loc_multi_index_with_tuples():
    arrays = [
        ["bar", "bar", "baz", "baz"],
        ["one", "two", "one", "two"],
    ]
    nrows = 5
    columns = pd.MultiIndex.from_tuples(zip(*arrays), names=["a", "b"])
    data = np.arange(0, nrows * len(columns)).reshape(nrows, len(columns))
    modin_df, pandas_df = create_test_dfs(data, columns=columns)
    eval_general(modin_df, pandas_df, lambda df: df.loc[:, ("bar", "two")])


def test_loc_multi_index_rows_with_tuples_5721():
    arrays = [
        ["bar", "bar", "baz", "baz"],
        ["one", "two", "one", "two"],
    ]
    ncols = 5
    index = pd.MultiIndex.from_tuples(zip(*arrays), names=["a", "b"])
    data = np.arange(0, ncols * len(index)).reshape(len(index), ncols)
    modin_df, pandas_df = create_test_dfs(data, index=index)
    eval_general(modin_df, pandas_df, lambda df: df.loc[("bar",)])
    eval_general(modin_df, pandas_df, lambda df: df.loc[("bar", "two")])


def test_loc_multi_index_level_two_has_same_name_as_column():
    eval_general(
        *create_test_dfs(
            pandas.DataFrame(
                [[0]], index=[pd.Index(["foo"]), pd.Index(["bar"])], columns=["bar"]
            )
        ),
        lambda df: df.loc[("foo", "bar")],
    )


def test_loc_multi_index_duplicate_keys():
    modin_df, pandas_df = create_test_dfs([1, 2], index=[["a", "a"], ["b", "b"]])
    eval_general(modin_df, pandas_df, lambda df: df.loc[("a", "b"), 0])
    eval_general(modin_df, pandas_df, lambda df: df.loc[("a", "b"), :])


def test_loc_multi_index_both_axes():
    multi_index = pd.MultiIndex.from_tuples(
        [("r0", "rA"), ("r1", "rB")], names=["Courses", "Fee"]
    )
    cols = pd.MultiIndex.from_tuples(
        [
            ("Gasoline", "Toyota"),
            ("Gasoline", "Ford"),
            ("Electric", "Tesla"),
            ("Electric", "Nio"),
        ]
    )
    data = [[100, 300, 900, 400], [200, 500, 300, 600]]
    modin_df, pandas_df = create_test_dfs(data, columns=cols, index=multi_index)
    eval_general(modin_df, pandas_df, lambda df: df.loc[("r0", "rA"), :])
    eval_general(modin_df, pandas_df, lambda df: df.loc[:, ("Gasoline", "Toyota")])


def test_loc_empty():
    pandas_df = pandas.DataFrame(index=range(5))
    modin_df = pd.DataFrame(index=range(5))

    df_equals(pandas_df.loc[1], modin_df.loc[1])
    pandas_df.loc[1] = 3
    modin_df.loc[1] = 3
    df_equals(pandas_df, modin_df)


@pytest.mark.parametrize("locator_name", ["iloc", "loc"])
def test_loc_iloc_2064(locator_name):
    modin_df, pandas_df = create_test_dfs(columns=["col1", "col2"])
    if locator_name == "iloc":
        expected_exception = IndexError(
            "index 1 is out of bounds for axis 0 with size 0"
        )
    else:
        _type = "int32" if os.name == "nt" else "int64"
        expected_exception = KeyError(
            f"None of [Index([1], dtype='{_type}')] are in the [index]"
        )
    eval_general(
        modin_df,
        pandas_df,
        lambda df: getattr(df, locator_name).__setitem__([1], [11, 22]),
        __inplace__=True,
        expected_exception=expected_exception,
    )


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


@pytest.mark.parametrize("left, right", [(2, 1), (6, 1), (lambda df: 70, 1), (90, 70)])
def test_loc_insert_row(left, right):
    # This test case comes from
    # https://github.com/modin-project/modin/issues/3764
    pandas_df = pandas.DataFrame([[1, 2, 3], [4, 5, 6]])
    modin_df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])

    def _test_loc_rows(df):
        df.loc[left] = df.loc[right]
        return df

    expected_exception = None
    if right == 70:
        pytest.xfail(reason="https://github.com/modin-project/modin/issues/7024")
    eval_general(
        modin_df, pandas_df, _test_loc_rows, expected_exception=expected_exception
    )


@pytest.mark.parametrize(
    "columns", [10, (100, 102), (2, 6), [10, 11, 12], "a", ["b", "c", "d"]]
)
def test_loc_insert_col(columns):
    # This test case comes from
    # https://github.com/modin-project/modin/issues/3764
    pandas_df = pandas.DataFrame([[1, 2, 3], [4, 5, 6]])
    modin_df = pd.DataFrame([[1, 2, 3], [4, 5, 6]])

    if isinstance(columns, tuple) and len(columns) == 2:

        def _test_loc_cols(df):
            df.loc[:, columns[0] : columns[1]] = 1

    else:

        def _test_loc_cols(df):
            df.loc[:, columns] = 1

    eval_general(modin_df, pandas_df, _test_loc_cols)


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


@pytest.mark.gpu
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
    modin_df.iloc[lambda df: 0]["col2"] = 12
    modin_df.iloc[1][lambda df: ["col2"]] = 22
    modin_df.iloc[lambda df: 2][lambda df: ["col2"]] = 32
    pandas_df.iloc[0]["col1"] = 11
    pandas_df.iloc[1]["col1"] = 21
    pandas_df.iloc[2]["col1"] = 31
    pandas_df.iloc[lambda df: 0]["col2"] = 12
    pandas_df.iloc[1][lambda df: ["col2"]] = 22
    pandas_df.iloc[lambda df: 2][lambda df: ["col2"]] = 32

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


def test_iloc_empty():
    pandas_df = pandas.DataFrame(index=range(5))
    modin_df = pd.DataFrame(index=range(5))

    df_equals(pandas_df.iloc[1], modin_df.iloc[1])
    pandas_df.iloc[1] = 3
    modin_df.iloc[1] = 3
    df_equals(pandas_df, modin_df)


def test_iloc_loc_key_length_except():
    modin_ser, pandas_ser = pd.Series(0), pandas.Series(0)
    eval_general(
        modin_ser,
        pandas_ser,
        lambda ser: ser.iloc[0, 0],
        expected_exception=pandas.errors.IndexingError("Too many indexers"),
    )
    eval_general(
        modin_ser,
        pandas_ser,
        lambda ser: ser.loc[0, 0],
        expected_exception=pandas.errors.IndexingError("Too many indexers"),
    )


def test_loc_series():
    md_df, pd_df = create_test_dfs({"a": [1, 2], "b": [3, 4]})

    pd_df.loc[pd_df["a"] > 1, "b"] = np.log(pd_df["b"])
    md_df.loc[md_df["a"] > 1, "b"] = np.log(md_df["b"])

    df_equals(pd_df, md_df)


@pytest.mark.parametrize("locator_name", ["loc", "iloc"])
@pytest.mark.parametrize(
    "slice_indexer",
    [
        slice(None, None, -2),
        slice(1, 10, None),
        slice(None, 10, None),
        slice(10, None, None),
        slice(10, None, -2),
        slice(-10, None, -2),
        slice(None, 1_000_000_000, None),
    ],
)
def test_loc_iloc_slice_indexer(locator_name, slice_indexer):
    md_df, pd_df = create_test_dfs(test_data_values[0])
    # Shifting the index, so labels won't match its position
    shifted_index = pandas.RangeIndex(1, len(md_df) + 1)
    md_df.index = shifted_index
    pd_df.index = shifted_index

    eval_general(md_df, pd_df, lambda df: getattr(df, locator_name)[slice_indexer])


@pytest.mark.parametrize(
    "indexer_size",
    [
        1,
        2,
        NROWS,
        pytest.param(
            NROWS + 1,
            marks=pytest.mark.xfail(
                reason="https://github.com/modin-project/modin/issues/5739", strict=True
            ),
        ),
    ],
)
class TestLocRangeLikeIndexer:
    """Test cases related to https://github.com/modin-project/modin/issues/5702"""

    def test_range_index_getitem_single_value(self, indexer_size):
        eval_general(
            *create_test_dfs(test_data["int_data"]),
            lambda df: df.loc[pd.RangeIndex(indexer_size)],
        )

    def test_range_index_getitem_two_values(self, indexer_size):
        eval_general(
            *create_test_dfs(test_data["int_data"]),
            lambda df: df.loc[pd.RangeIndex(indexer_size), :],
        )

    def test_range_getitem_single_value(self, indexer_size):
        eval_general(
            *create_test_dfs(test_data["int_data"]),
            lambda df: df.loc[range(indexer_size)],
        )

    def test_range_getitem_two_values_5702(self, indexer_size):
        eval_general(
            *create_test_dfs(test_data["int_data"]),
            lambda df: df.loc[range(indexer_size), :],
        )


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


def test_reindex_4438():
    index = pd.date_range(end="1/1/2018", periods=3, freq="h", name="some meta")
    new_index = list(reversed(index))

    # index case
    modin_df = pd.DataFrame([1, 2, 3], index=index)
    pandas_df = pandas.DataFrame([1, 2, 3], index=index)
    new_modin_df = modin_df.reindex(new_index)
    new_pandas_df = pandas_df.reindex(new_index)
    df_equals(new_modin_df, new_pandas_df)

    # column case
    modin_df = pd.DataFrame(np.array([[1], [2], [3]]).T, columns=index)
    pandas_df = pandas.DataFrame(np.array([[1], [2], [3]]).T, columns=index)
    new_modin_df = modin_df.reindex(columns=new_index)
    new_pandas_df = pandas_df.reindex(columns=new_index)
    df_equals(new_modin_df, new_pandas_df)

    # multiindex case
    multi_index = pandas.MultiIndex.from_arrays(
        [("a", "b", "c"), ("a", "b", "c")], names=["first", "second"]
    )
    new_multi_index = list(reversed(multi_index))

    modin_df = pd.DataFrame([1, 2, 3], index=multi_index)
    pandas_df = pandas.DataFrame([1, 2, 3], index=multi_index)
    new_modin_df = modin_df.reindex(new_multi_index)
    new_pandas_df = pandas_df.reindex(new_multi_index)
    df_equals(new_modin_df, new_pandas_df)

    # multicolumn case
    modin_df = pd.DataFrame(np.array([[1], [2], [3]]).T, columns=multi_index)
    pandas_df = pandas.DataFrame(np.array([[1], [2], [3]]).T, columns=multi_index)
    new_modin_df = modin_df.reindex(columns=new_multi_index)
    new_pandas_df = pandas_df.reindex(columns=new_multi_index)
    df_equals(new_modin_df, new_pandas_df)

    # index + multiindex case
    modin_df = pd.DataFrame([1, 2, 3], index=index)
    pandas_df = pandas.DataFrame([1, 2, 3], index=index)
    new_modin_df = modin_df.reindex(new_multi_index)
    new_pandas_df = pandas_df.reindex(new_multi_index)
    df_equals(new_modin_df, new_pandas_df)


def test_reindex_like():
    o_data = [
        [24.3, 75.7, "high"],
        [31, 87.8, "high"],
        [22, 71.6, "medium"],
        [35, 95, "medium"],
    ]
    o_columns = ["temp_celsius", "temp_fahrenheit", "windspeed"]
    o_index = pd.date_range(start="2014-02-12", end="2014-02-15", freq="D")
    new_data = [[28, "low"], [30, "low"], [35.1, "medium"]]
    new_columns = ["temp_celsius", "windspeed"]
    new_index = pd.DatetimeIndex(["2014-02-12", "2014-02-13", "2014-02-15"])
    modin_df1 = pd.DataFrame(o_data, columns=o_columns, index=o_index)
    modin_df2 = pd.DataFrame(new_data, columns=new_columns, index=new_index)
    modin_result = modin_df2.reindex_like(modin_df1)

    pandas_df1 = pandas.DataFrame(o_data, columns=o_columns, index=o_index)
    pandas_df2 = pandas.DataFrame(new_data, columns=new_columns, index=new_index)
    pandas_result = pandas_df2.reindex_like(pandas_df1)
    df_equals(modin_result, pandas_result)


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
    source_df.rename(columns={"col3": "foo", "col4": "bar"})
    modin_df = pd.DataFrame(source_df)
    assert_index_equal(
        modin_df.rename(columns={"col3": "foo", "col4": "bar"}).index,
        source_df.rename(columns={"col3": "foo", "col4": "bar"}).index,
    )

    # other axis
    source_df.T.rename(index={"col3": "foo", "col4": "bar"})
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


@pytest.mark.xfail(reason="Pandas does not pass this test")
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


def test_index_to_datetime_using_set_index():
    data = {"YEAR": ["1992", "1993", "1994"], "ALIENS": [1, 99, 1]}
    modin_df_years = pd.DataFrame(data=data)
    df_years = pandas.DataFrame(data=data)
    modin_df_years = modin_df_years.set_index("YEAR")
    df_years = df_years.set_index("YEAR")
    modin_datetime_index = pd.to_datetime(modin_df_years.index, format="%Y")
    pandas_datetime_index = pandas.to_datetime(df_years.index, format="%Y")

    modin_df_years.index = modin_datetime_index
    df_years.index = pandas_datetime_index

    modin_df_years.set_index(modin_datetime_index)
    df_years.set_index(pandas_datetime_index)
    df_equals(modin_df_years, df_years)


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


def test_rename_issue5600():
    # Check the issue for more details
    # https://github.com/modin-project/modin/issues/5600
    df = pd.DataFrame({"a": [1, 2]})
    df_renamed = df.rename(columns={"a": "new_a"}, copy=True, inplace=False)

    # Check that the source frame was untouched
    assert df.dtypes.keys().tolist() == ["a"]
    assert df.columns.tolist() == ["a"]

    assert df_renamed.dtypes.keys().tolist() == ["new_a"]
    assert df_renamed.columns.tolist() == ["new_a"]


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


def test_reindex_multiindex():
    data1, data2 = np.random.randint(1, 20, (5, 5)), np.random.randint(10, 25, 6)
    index = np.array(["AUD", "BRL", "CAD", "EUR", "INR"])
    modin_midx = pd.MultiIndex.from_product(
        [["Bank_1", "Bank_2"], ["AUD", "CAD", "EUR"]], names=["Bank", "Curency"]
    )
    pandas_midx = pandas.MultiIndex.from_product(
        [["Bank_1", "Bank_2"], ["AUD", "CAD", "EUR"]], names=["Bank", "Curency"]
    )
    modin_df1, modin_df2 = (
        pd.DataFrame(data=data1, index=index, columns=index),
        pd.DataFrame(data2, modin_midx),
    )
    pandas_df1, pandas_df2 = (
        pandas.DataFrame(data=data1, index=index, columns=index),
        pandas.DataFrame(data2, pandas_midx),
    )
    modin_df2.columns, pandas_df2.columns = ["Notional"], ["Notional"]
    md_midx = pd.MultiIndex.from_product([modin_df2.index.levels[0], modin_df1.index])
    pd_midx = pandas.MultiIndex.from_product(
        [pandas_df2.index.levels[0], pandas_df1.index]
    )
    # reindex without axis, index, or columns
    modin_result = modin_df1.reindex(md_midx, fill_value=0)
    pandas_result = pandas_df1.reindex(pd_midx, fill_value=0)
    df_equals(modin_result, pandas_result)
    # reindex with only axis
    modin_result = modin_df1.reindex(md_midx, fill_value=0, axis=0)
    pandas_result = pandas_df1.reindex(pd_midx, fill_value=0, axis=0)
    df_equals(modin_result, pandas_result)
    # reindex with axis and level
    modin_result = modin_df1.reindex(md_midx, fill_value=0, axis=0, level=0)
    pandas_result = pandas_df1.reindex(pd_midx, fill_value=0, axis=0, level=0)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("test_async_reset_index", [False, True])
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_reset_index(data, test_async_reset_index):
    modin_df, pandas_df = create_test_dfs(data)
    if test_async_reset_index:
        modin_df._query_compiler.set_frame_index_cache(None)
    modin_result = modin_df.reset_index(inplace=False)
    pandas_result = pandas_df.reset_index(inplace=False)
    df_equals(modin_result, pandas_result)

    modin_df_cp = modin_df.copy()
    pd_df_cp = pandas_df.copy()
    if test_async_reset_index:
        modin_df._query_compiler.set_frame_index_cache(None)
    modin_df_cp.reset_index(inplace=True)
    pd_df_cp.reset_index(inplace=True)
    df_equals(modin_df_cp, pd_df_cp)


@pytest.mark.parametrize(
    "data",
    [
        test_data["int_data"],
        test_data["float_nan_data"],
    ],
)
def test_reset_index_multiindex_groupby(data):
    # GH#4394
    modin_df, pandas_df = create_test_dfs(data)
    modin_df.index = pd.MultiIndex.from_tuples(
        [(i // 10, i // 5, i) for i in range(len(modin_df))]
    )
    pandas_df.index = pandas.MultiIndex.from_tuples(
        [(i // 10, i // 5, i) for i in range(len(pandas_df))]
    )
    eval_general(
        modin_df,
        pandas_df,
        lambda df: df.reset_index().groupby(list(df.columns[:2])).count(),
    )


@pytest.mark.parametrize("test_async_reset_index", [False, True])
@pytest.mark.parametrize(
    "data",
    [
        pytest.param(
            test_data["int_data"],
            marks=pytest.mark.exclude_by_default,
        ),
        test_data["float_nan_data"],
    ],
    ids=["int_data", "float_nan_data"],
)
@pytest.mark.parametrize("nlevels", [3])
@pytest.mark.parametrize("columns_multiindex", [True, False])
@pytest.mark.parametrize(
    "level",
    [
        "no_level",
        None,
        0,
        1,
        2,
        [2, 0],
        [2, 1],
        [1, 0],
        pytest.param(
            [2, 1, 2],
            marks=pytest.mark.exclude_by_default,
        ),
        pytest.param(
            [0, 0, 0, 0],
            marks=pytest.mark.exclude_by_default,
        ),
        pytest.param(
            ["level_name_1"],
            marks=pytest.mark.exclude_by_default,
        ),
        pytest.param(
            ["level_name_2", "level_name_1"],
            marks=pytest.mark.exclude_by_default,
        ),
        pytest.param(
            [2, "level_name_0"],
            marks=pytest.mark.exclude_by_default,
        ),
    ],
)
@pytest.mark.parametrize("col_level", ["no_col_level", 0, 1, 2])
@pytest.mark.parametrize("col_fill", ["no_col_fill", None, 0, "new"])
@pytest.mark.parametrize("drop", [False])
@pytest.mark.parametrize(
    "multiindex_levels_names_max_levels",
    [
        0,
        1,
        2,
        pytest.param(3, marks=pytest.mark.exclude_by_default),
        pytest.param(4, marks=pytest.mark.exclude_by_default),
    ],
)
@pytest.mark.parametrize(
    "none_in_index_names",
    [
        pytest.param(
            False,
            marks=pytest.mark.exclude_by_default,
        ),
        True,
        "mixed_1st_None",
        pytest.param(
            "mixed_2nd_None",
            marks=pytest.mark.exclude_by_default,
        ),
    ],
)
def test_reset_index_with_multi_index_no_drop(
    data,
    nlevels,
    columns_multiindex,
    level,
    col_level,
    col_fill,
    drop,
    multiindex_levels_names_max_levels,
    none_in_index_names,
    test_async_reset_index,
):
    data_rows = len(data[list(data.keys())[0]])
    index = generate_multiindex(data_rows, nlevels=nlevels)
    data_columns = len(data.keys())
    columns = (
        generate_multiindex(data_columns, nlevels=nlevels)
        if columns_multiindex
        else pandas.RangeIndex(0, data_columns)
    )
    # Replace original data columns with generated
    data = {columns[ind]: data[key] for ind, key in enumerate(data)}
    index.names = (
        [f"level_{i}" for i in range(index.nlevels)]
        if multiindex_levels_names_max_levels == 0
        else [
            (
                tuple(
                    [
                        f"level_{i}_name_{j}"
                        for j in range(
                            0,
                            max(
                                multiindex_levels_names_max_levels + 1 - index.nlevels,
                                0,
                            )
                            + i,
                        )
                    ]
                )
                if max(multiindex_levels_names_max_levels + 1 - index.nlevels, 0) + i
                > 0
                else f"level_{i}"
            )
            for i in range(index.nlevels)
        ]
    )

    if none_in_index_names is True:
        index.names = [None] * len(index.names)
    elif none_in_index_names:
        names_list = list(index.names)
        start_index = 0 if none_in_index_names == "mixed_1st_None" else 1
        names_list[start_index::2] = [None] * len(names_list[start_index::2])
        index.names = names_list

    modin_df = pd.DataFrame(data, index=index, columns=columns)
    pandas_df = pandas.DataFrame(data, index=index, columns=columns)

    if isinstance(level, list):
        level = [
            (
                index.names[int(x[len("level_name_") :])]
                if isinstance(x, str) and x.startswith("level_name_")
                else x
            )
            for x in level
        ]

    kwargs = {"drop": drop}
    if level != "no_level":
        kwargs["level"] = level
    if col_level != "no_col_level":
        kwargs["col_level"] = col_level
    if col_fill != "no_col_fill":
        kwargs["col_fill"] = col_fill
    if test_async_reset_index:
        modin_df._query_compiler.set_frame_index_cache(None)
    eval_general(
        modin_df,
        pandas_df,
        lambda df: df.reset_index(**kwargs),
        # https://github.com/modin-project/modin/issues/5960
        comparator_kwargs={"check_dtypes": False},
    )


@pytest.mark.parametrize("test_async_reset_index", [False, True])
@pytest.mark.parametrize(
    "data",
    [
        pytest.param(
            test_data["int_data"],
            marks=pytest.mark.exclude_by_default,
        ),
        test_data["float_nan_data"],
    ],
    ids=["int_data", "float_nan_data"],
)
@pytest.mark.parametrize("nlevels", [3])
@pytest.mark.parametrize(
    "level",
    [
        "no_level",
        None,
        0,
        1,
        2,
        [2, 0],
        [2, 1],
        [1, 0],
        pytest.param(
            [2, 1, 2],
            marks=pytest.mark.exclude_by_default,
        ),
        pytest.param(
            [0, 0, 0, 0],
            marks=pytest.mark.exclude_by_default,
        ),
        pytest.param(
            ["level_name_1"],
            marks=pytest.mark.exclude_by_default,
        ),
        pytest.param(
            ["level_name_2", "level_name_1"],
            marks=pytest.mark.exclude_by_default,
        ),
        pytest.param(
            [2, "level_name_0"],
            marks=pytest.mark.exclude_by_default,
        ),
    ],
)
@pytest.mark.parametrize(
    "multiindex_levels_names_max_levels",
    [
        0,
        1,
        2,
        pytest.param(3, marks=pytest.mark.exclude_by_default),
        pytest.param(4, marks=pytest.mark.exclude_by_default),
    ],
)
@pytest.mark.parametrize(
    "none_in_index_names",
    [
        pytest.param(
            False,
            marks=pytest.mark.exclude_by_default,
        ),
        True,
        "mixed_1st_None",
        pytest.param(
            "mixed_2nd_None",
            marks=pytest.mark.exclude_by_default,
        ),
    ],
)
def test_reset_index_with_multi_index_drop(
    data,
    nlevels,
    level,
    multiindex_levels_names_max_levels,
    none_in_index_names,
    test_async_reset_index,
):
    test_reset_index_with_multi_index_no_drop(
        data,
        nlevels,
        True,
        level,
        "no_col_level",
        "no_col_fill",
        True,
        multiindex_levels_names_max_levels,
        none_in_index_names,
        test_async_reset_index,
    )


@pytest.mark.parametrize("test_async_reset_index", [False, True])
@pytest.mark.parametrize("index_levels_names_max_levels", [0, 1, 2])
def test_reset_index_with_named_index(
    index_levels_names_max_levels, test_async_reset_index
):
    modin_df = pd.DataFrame(test_data_values[0])
    pandas_df = pandas.DataFrame(test_data_values[0])

    index_name = (
        tuple([f"name_{j}" for j in range(0, index_levels_names_max_levels)])
        if index_levels_names_max_levels > 0
        else "NAME_OF_INDEX"
    )
    modin_df.index.name = pandas_df.index.name = index_name
    df_equals(modin_df, pandas_df)
    if test_async_reset_index:
        # The change in index is not automatically handled by Modin. See #3941.
        modin_df.index = modin_df.index
        modin_df.modin.to_pandas()

        modin_df._query_compiler.set_frame_index_cache(None)
    df_equals(modin_df.reset_index(drop=False), pandas_df.reset_index(drop=False))

    if test_async_reset_index:
        # The change in index is not automatically handled by Modin. See #3941.
        modin_df.index = modin_df.index
        modin_df.modin.to_pandas()

        modin_df._query_compiler.set_frame_index_cache(None)
    modin_df.reset_index(drop=True, inplace=True)
    pandas_df.reset_index(drop=True, inplace=True)
    df_equals(modin_df, pandas_df)

    modin_df = pd.DataFrame(test_data_values[0])
    pandas_df = pandas.DataFrame(test_data_values[0])
    modin_df.index.name = pandas_df.index.name = index_name
    if test_async_reset_index:
        # The change in index is not automatically handled by Modin. See #3941.
        modin_df.index = modin_df.index
        modin_df._to_pandas()

        modin_df._query_compiler.set_frame_index_cache(None)
    df_equals(modin_df.reset_index(drop=False), pandas_df.reset_index(drop=False))


@pytest.mark.parametrize("test_async_reset_index", [False, True])
@pytest.mark.parametrize(
    "index",
    [
        pandas.Index([11, 22, 33, 44], name="col0"),
        pandas.MultiIndex.from_product(
            [[100, 200], [300, 400]], names=["level1", "col0"]
        ),
    ],
    ids=["index", "multiindex"],
)
def test_reset_index_metadata_update(index, test_async_reset_index):
    modin_df, pandas_df = create_test_dfs({"col0": [0, 1, 2, 3]}, index=index)
    modin_df.columns = pandas_df.columns = ["col1"]
    if test_async_reset_index:
        # The change in index is not automatically handled by Modin. See #3941.
        modin_df.index = modin_df.index
        modin_df._to_pandas()

        modin_df._query_compiler.set_frame_index_cache(None)
    eval_general(modin_df, pandas_df, lambda df: df.reset_index())


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


def test_empty_sample():
    modin_df, pandas_df = create_test_dfs([1])
    # issue #4983
    # If we have a fraction of the dataset that results in n=0, we should
    # make sure that we don't pass in both n and frac to sample internally.
    eval_general(modin_df, pandas_df, lambda df: df.sample(frac=0.12))


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

    include = np.float64, "integer"
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
    # example is based on the doctest in the upstream pandas docstring
    data = {
        "num_legs": [4, 4, 2, 2],
        "num_wings": [0, 0, 2, 2],
        "class": ["mammal", "mammal", "mammal", "bird"],
        "animal": ["cat", "dog", "bat", "penguin"],
        "locomotion": ["walks", "walks", "flies", "walks"],
    }
    modin_df, pandas_df = create_test_dfs(data)

    def prepare_dataframes(df):
        # to make several partitions (only for Modin dataframe)
        df = (pd if isinstance(df, pd.DataFrame) else pandas).concat([df, df], axis=0)
        # looks like pandas is sorting the index whereas modin is not, performing a join operation.
        df = df.reset_index(drop=True)
        df = df.join(df, rsuffix="_y")
        return df.set_index(["class", "animal", "locomotion"])

    modin_df = prepare_dataframes(modin_df)
    pandas_df = prepare_dataframes(pandas_df)
    eval_general(modin_df, pandas_df, lambda df: df.xs("mammal"))
    eval_general(modin_df, pandas_df, lambda df: df.xs("cat", level=1))
    eval_general(modin_df, pandas_df, lambda df: df.xs("num_legs", axis=1))
    eval_general(
        modin_df, pandas_df, lambda df: df.xs("cat", level=1, drop_level=False)
    )
    eval_general(modin_df, pandas_df, lambda df: df.xs(("mammal", "cat")))
    eval_general(
        modin_df, pandas_df, lambda df: df.xs(("mammal", "cat"), drop_level=False)
    )


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
        (-1, 1, -1),
        (None, None, 2),
    ]

    # slice test
    for slice_param in slices:
        s = slice(*slice_param)
        df_equals(modin_df[s], pandas_df[s])

    # Test empty
    df_equals(pd.DataFrame([])[:10], pandas.DataFrame([])[:10])


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___getitem_bool_indexers(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    indices = [i % 3 == 0 for i in range(len(modin_df.index))]
    columns = [i % 5 == 0 for i in range(len(modin_df.columns))]

    # Key is a list of booleans
    modin_result = modin_df.loc[indices, columns]
    pandas_result = pandas_df.loc[indices, columns]
    df_equals(modin_result, pandas_result)

    # Key is a Modin or pandas series of booleans
    df_equals(
        modin_df.loc[pd.Series(indices), pd.Series(columns, index=modin_df.columns)],
        pandas_df.loc[
            pandas.Series(indices), pandas.Series(columns, index=modin_df.columns)
        ],
    )


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

    if "empty_data" not in request.node.name:
        key = modin_df.columns[0]
        modin_df.__getattr__(key)

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
    eval_setitem(*create_test_dfs(data), col="___NON EXISTENT COLUMN", value=arr.T[0])
    eval_setitem(*create_test_dfs(data), loc=0, value=np.arange(nrows))

    modin_df = pd.DataFrame(columns=data.keys())
    pandas_df = pandas.DataFrame(columns=data.keys())

    for col in modin_df.columns:
        modin_df[col] = np.arange(1000)

    for col in pandas_df.columns:
        pandas_df[col] = np.arange(1000)

    df_equals(modin_df, pandas_df)

    # Test df assignment to a columns selection
    modin_df[modin_df.columns[[0, -1]]] = modin_df[modin_df.columns[[0, -1]]]
    pandas_df[pandas_df.columns[[0, -1]]] = pandas_df[pandas_df.columns[[0, -1]]]
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

    modin_df[1:5] = 10
    pandas_df[1:5] = 10
    df_equals(modin_df, pandas_df)

    # Transpose test
    modin_df = pd.DataFrame(data).T
    pandas_df = pandas.DataFrame(data).T

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

    modin_df[1:5] = 10
    pandas_df[1:5] = 10
    df_equals(modin_df, pandas_df)


def test___setitem__partitions_aligning():
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

    pd_df["a"] = pandas.Series(np.arange(4))
    md_df["a"] = pd.Series(np.arange(4))
    df_equals(md_df, pd_df)


def test___setitem__with_mismatched_partitions():
    with ensure_clean(".csv") as fname:
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
        {"id": [], "max_speed": [], "health": []},
        {"id": [1], "max_speed": [2], "health": [3]},
        {"id": [4, 40, 400], "max_speed": [111, 222, 333], "health": [33, 22, 11]},
    ],
    ids=["empty_frame", "empty_cols", "1_length_cols", "2_length_cols"],
)
@pytest.mark.parametrize(
    "value",
    [[11, 22], [11, 22, 33]],
    ids=["2_length_val", "3_length_val"],
)
@pytest.mark.parametrize("convert_to_series", [False, True])
@pytest.mark.parametrize("new_col_id", [123, "new_col"], ids=["integer", "string"])
def test_setitem_on_empty_df(data, value, convert_to_series, new_col_id):
    pandas_df = pandas.DataFrame(data)
    modin_df = pd.DataFrame(data)

    def applyier(df):
        if convert_to_series:
            converted_value = (
                pandas.Series(value)
                if isinstance(df, pandas.DataFrame)
                else pd.Series(value)
            )
        else:
            converted_value = value
        df[new_col_id] = converted_value
        return df

    expected_exception = None
    if not convert_to_series:
        values_length = len(value)
        index_length = len(pandas_df.index)
        expected_exception = ValueError(
            f"Length of values ({values_length}) does not match length of index ({index_length})"
        )

    eval_general(
        modin_df,
        pandas_df,
        applyier,
        # https://github.com/modin-project/modin/issues/5961
        comparator_kwargs={
            "check_dtypes": not (len(pandas_df) == 0 and len(pandas_df.columns) != 0)
        },
        expected_exception=expected_exception,
    )


def test_setitem_on_empty_df_4407():
    data = {}
    index = pd.date_range(end="1/1/2018", periods=0, freq="D")
    column = pd.date_range(end="1/1/2018", periods=1, freq="h")[0]
    modin_df = pd.DataFrame(data, columns=index)
    pandas_df = pandas.DataFrame(data, columns=index)

    modin_df[column] = pd.Series([1])
    pandas_df[column] = pandas.Series([1])

    df_equals(modin_df, pandas_df)
    assert modin_df.columns.freq == pandas_df.columns.freq


def test___setitem__unhashable_list():
    # from #3258 and #3291
    cols = ["a", "b"]
    modin_df = pd.DataFrame([[0, 0]], columns=cols)
    modin_df[cols] = modin_df[cols]
    pandas_df = pandas.DataFrame([[0, 0]], columns=cols)
    pandas_df[cols] = pandas_df[cols]
    df_equals(modin_df, pandas_df)


def test_setitem_unhashable_key():
    source_modin_df, source_pandas_df = create_test_dfs(test_data["float_nan_data"])
    row_count = source_modin_df.shape[0]

    def _make_copy(df1, df2):
        return df1.copy(deep=True), df2.copy(deep=True)

    for key in (["col1", "col2"], ["new_col1", "new_col2"]):
        # 1d list case
        value = [1, 2]
        modin_df, pandas_df = _make_copy(source_modin_df, source_pandas_df)
        eval_setitem(modin_df, pandas_df, value, key)

        # 2d list case
        value = [[1, 2]] * row_count
        modin_df, pandas_df = _make_copy(source_modin_df, source_pandas_df)
        eval_setitem(modin_df, pandas_df, value, key)

        # pandas DataFrame case
        df_value = pandas.DataFrame(value, columns=["value_col1", "value_col2"])
        modin_df, pandas_df = _make_copy(source_modin_df, source_pandas_df)
        eval_setitem(modin_df, pandas_df, df_value, key)

        # numpy array case
        value = df_value.to_numpy()
        modin_df, pandas_df = _make_copy(source_modin_df, source_pandas_df)
        eval_setitem(modin_df, pandas_df, value, key)

        # pandas Series case
        value = df_value["value_col1"]
        modin_df, pandas_df = _make_copy(source_modin_df, source_pandas_df)
        eval_setitem(
            modin_df,
            pandas_df,
            value,
            key[:1],
            expected_exception=ValueError("Columns must be same length as key"),
        )

        # pandas Index case
        value = df_value.index
        modin_df, pandas_df = _make_copy(source_modin_df, source_pandas_df)
        eval_setitem(
            modin_df,
            pandas_df,
            value,
            key[:1],
            expected_exception=ValueError("Columns must be same length as key"),
        )

        # scalar case
        value = 3
        modin_df, pandas_df = _make_copy(source_modin_df, source_pandas_df)
        eval_setitem(modin_df, pandas_df, value, key)

        # test failed case: ValueError('Columns must be same length as key')
        eval_setitem(
            modin_df,
            pandas_df,
            df_value[["value_col1"]],
            key,
            expected_exception=ValueError("Columns must be same length as key"),
        )


def test_setitem_2d_insertion():
    def build_value_picker(modin_value, pandas_value):
        """Build a function that returns either Modin or pandas DataFrame depending on the passed frame."""
        return lambda source_df, *args, **kwargs: (
            modin_value
            if isinstance(source_df, (pd.DataFrame, pd.Series))
            else pandas_value
        )

    modin_df, pandas_df = create_test_dfs(test_data["int_data"])

    # Easy case - key and value.columns are equal
    modin_value, pandas_value = create_test_dfs(
        {"new_value1": np.arange(len(modin_df)), "new_value2": np.arange(len(modin_df))}
    )
    eval_setitem(
        modin_df,
        pandas_df,
        build_value_picker(modin_value, pandas_value),
        col=["new_value1", "new_value2"],
    )

    # Key and value.columns have equal values but in different order
    new_columns = ["new_value3", "new_value4"]
    modin_value.columns, pandas_value.columns = new_columns, new_columns
    eval_setitem(
        modin_df,
        pandas_df,
        build_value_picker(modin_value, pandas_value),
        col=["new_value4", "new_value3"],
    )

    # Key and value.columns have different values
    new_columns = ["new_value5", "new_value6"]
    modin_value.columns, pandas_value.columns = new_columns, new_columns
    eval_setitem(
        modin_df,
        pandas_df,
        build_value_picker(modin_value, pandas_value),
        col=["__new_value5", "__new_value6"],
    )

    # Key and value.columns have different lengths, testing that both raise the same exception
    eval_setitem(
        modin_df,
        pandas_df,
        build_value_picker(modin_value.iloc[:, [0]], pandas_value.iloc[:, [0]]),
        col=["new_value7", "new_value8"],
        expected_exception=ValueError("Columns must be same length as key"),
    )


@pytest.mark.parametrize("does_value_have_different_columns", [True, False])
def test_setitem_2d_update(does_value_have_different_columns):
    def test(dfs, iloc):
        """Update columns on the given numeric indices."""
        df1, df2 = dfs
        cols1 = df1.columns[iloc].tolist()
        cols2 = df2.columns[iloc].tolist()
        df1[cols1] = df2[cols2]
        return df1

    modin_df, pandas_df = create_test_dfs(test_data["int_data"])
    modin_df2, pandas_df2 = create_test_dfs(test_data["int_data"])
    modin_df2 *= 10
    pandas_df2 *= 10

    if does_value_have_different_columns:
        new_columns = [f"{col}_new" for col in modin_df.columns]
        modin_df2.columns = new_columns
        pandas_df2.columns = new_columns

    modin_dfs = (modin_df, modin_df2)
    pandas_dfs = (pandas_df, pandas_df2)

    eval_general(modin_dfs, pandas_dfs, test, iloc=[0, 1, 2])
    eval_general(modin_dfs, pandas_dfs, test, iloc=[0, -1])
    eval_general(
        modin_dfs, pandas_dfs, test, iloc=slice(1, None)
    )  # (start=1, stop=None)
    eval_general(
        modin_dfs, pandas_dfs, test, iloc=slice(None, -2)
    )  # (start=None, stop=-2)
    eval_general(
        modin_dfs,
        pandas_dfs,
        test,
        iloc=[0, 1, 5, 6, 9, 10, -2, -1],
    )
    eval_general(
        modin_dfs,
        pandas_dfs,
        test,
        iloc=[5, 4, 0, 10, 1, -1],
    )
    eval_general(
        modin_dfs, pandas_dfs, test, iloc=slice(None, None, 2)
    )  # (start=None, stop=None, step=2)


def test___setitem__single_item_in_series():
    # Test assigning a single item in a Series for issue
    # https://github.com/modin-project/modin/issues/3860
    modin_series = pd.Series(99)
    pandas_series = pandas.Series(99)
    modin_series[:1] = pd.Series(100)
    pandas_series[:1] = pandas.Series(100)
    df_equals(modin_series, pandas_series)


def test___setitem__assigning_single_categorical_sets_correct_dtypes():
    # This test case comes from
    # https://github.com/modin-project/modin/issues/3895
    modin_df = pd.DataFrame({"categories": ["A"]})
    modin_df["categories"] = pd.Categorical(["A"])
    pandas_df = pandas.DataFrame({"categories": ["A"]})
    pandas_df["categories"] = pandas.Categorical(["A"])
    df_equals(modin_df, pandas_df)


def test_iloc_assigning_scalar_none_to_string_frame():
    # This test case comes from
    # https://github.com/modin-project/modin/issues/3981
    data = [["A"]]
    modin_df = pd.DataFrame(data, dtype="string")
    modin_df.iloc[0, 0] = None
    pandas_df = pandas.DataFrame(data, dtype="string")
    pandas_df.iloc[0, 0] = None
    df_equals(modin_df, pandas_df)


@pytest.mark.parametrize(
    "value",
    [
        1,
        np.int32(1),
        1.0,
        "str val",
        pandas.Timestamp("1/4/2018"),
        np.datetime64(0, "ms"),
        True,
    ],
)
def test_loc_boolean_assignment_scalar_dtypes(value):
    modin_df, pandas_df = create_test_dfs(
        {
            "a": [1, 2, 3],
            "b": [3.0, 5.0, 6.0],
            "c": ["a", "b", "c"],
            "d": [1.0, "c", 2.0],
            "e": pandas.to_datetime(["1/1/2018", "1/2/2018", "1/3/2018"]),
            "f": [True, False, True],
        }
    )
    modin_idx, pandas_idx = pd.Series([False, True, True]), pandas.Series(
        [False, True, True]
    )

    modin_df.loc[modin_idx] = value
    pandas_df.loc[pandas_idx] = value
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

    for func in ["all", "any", "count"]:
        df_equals(
            getattr(df_modin, func)().index,
            getattr(df_pandas, func)().index,
        )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("sortorder", [0, 3, 5])
def test_multiindex_from_frame(data, sortorder):
    modin_df, pandas_df = create_test_dfs(data)

    def call_from_frame(df):
        if type(df).__module__.startswith("pandas"):
            return pandas.MultiIndex.from_frame(df, sortorder)
        else:
            return pd.MultiIndex.from_frame(df, sortorder)

    eval_general(modin_df, pandas_df, call_from_frame, comparator=assert_index_equal)


def test__getitem_bool_single_row_dataframe():
    # This test case comes from
    # https://github.com/modin-project/modin/issues/4845
    eval_general(pd, pandas, lambda lib: lib.DataFrame([1])[lib.Series([True])])


def test__getitem_bool_with_empty_partition():
    # This test case comes from
    # https://github.com/modin-project/modin/issues/5188

    size = MinRowPartitionSize.get()

    pandas_series = pandas.Series([True if i % 2 else False for i in range(size)])
    modin_series = pd.Series(pandas_series)

    pandas_df = pandas.DataFrame([i for i in range(size + 1)])
    pandas_df.iloc[size] = np.nan
    modin_df = pd.DataFrame(pandas_df)

    pandas_tmp_result = pandas_df.dropna()
    modin_tmp_result = modin_df.dropna()

    eval_general(
        modin_tmp_result,
        pandas_tmp_result,
        lambda df: (
            df[modin_series] if isinstance(df, pd.DataFrame) else df[pandas_series]
        ),
    )


# This is a very subtle bug that comes from:
# https://github.com/modin-project/modin/issues/4945
def test_lazy_eval_index():
    modin_df, pandas_df = create_test_dfs({"col0": [0, 1]})

    def func(df):
        df_copy = df[df["col0"] < 6].copy()
        # The problem here is that the index is not copied over so it needs
        # to get recomputed at some point. Our implementation of __setitem__
        # requires us to build a mask and insert the value from the right
        # handside into the new DataFrame. However, it's possible that we
        # won't have any new partitions, so we will end up computing an empty
        # index.
        df_copy["col0"] = df_copy["col0"].apply(lambda x: x + 1)
        return df_copy

    eval_general(modin_df, pandas_df, func)


def test_index_of_empty_frame():
    # Test on an empty frame created by user
    md_df, pd_df = create_test_dfs(
        {}, index=pandas.Index([], name="index name"), columns=["a", "b"]
    )
    assert md_df.empty and pd_df.empty
    df_equals(md_df.index, pd_df.index)

    # Test on an empty frame produced by Modin's logic
    data = test_data_values[0]
    md_df, pd_df = create_test_dfs(
        data, index=pandas.RangeIndex(len(next(iter(data.values()))), name="index name")
    )

    md_res = md_df.query(f"{md_df.columns[0]} > {RAND_HIGH}")
    pd_res = pd_df.query(f"{pd_df.columns[0]} > {RAND_HIGH}")

    assert md_res.empty and pd_res.empty
    df_equals(md_res.index, pd_res.index)
