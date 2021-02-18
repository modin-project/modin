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
from modin.utils import get_current_backend

from modin.pandas.test.utils import (
    random_state,
    RAND_LOW,
    RAND_HIGH,
    df_equals,
    df_is_empty,
    arg_keys,
    name_contains,
    test_data,
    test_data_values,
    test_data_keys,
    test_data_with_duplicates_values,
    test_data_with_duplicates_keys,
    numeric_dfs,
    test_func_keys,
    test_func_values,
    indices_keys,
    indices_values,
    axis_keys,
    axis_values,
    bool_arg_keys,
    bool_arg_values,
    int_arg_keys,
    int_arg_values,
    eval_general,
    create_test_dfs,
)
from modin.config import NPartitions

NPartitions.put(4)

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")


def eval_insert(modin_df, pandas_df, **kwargs):
    if "col" in kwargs and "column" not in kwargs:
        kwargs["column"] = kwargs.pop("col")
    _kwargs = {"loc": 0, "column": "New column"}
    _kwargs.update(kwargs)

    eval_general(
        modin_df,
        pandas_df,
        operation=lambda df, **kwargs: df.insert(**kwargs),
        **_kwargs,
    )


def test_indexing():
    modin_df = pd.DataFrame(
        dict(a=[1, 2, 3], b=[4, 5, 6], c=[7, 8, 9]), index=["a", "b", "c"]
    )
    pandas_df = pandas.DataFrame(
        dict(a=[1, 2, 3], b=[4, 5, 6], c=[7, 8, 9]), index=["a", "b", "c"]
    )

    modin_result = modin_df
    pandas_result = pandas_df
    df_equals(modin_result, pandas_result)

    modin_result = modin_df["b"]
    pandas_result = pandas_df["b"]
    df_equals(modin_result, pandas_result)

    modin_result = modin_df[["b"]]
    pandas_result = pandas_df[["b"]]
    df_equals(modin_result, pandas_result)

    modin_result = modin_df[["b", "a"]]
    pandas_result = pandas_df[["b", "a"]]
    df_equals(modin_result, pandas_result)

    modin_result = modin_df.loc["b"]
    pandas_result = pandas_df.loc["b"]
    df_equals(modin_result, pandas_result)

    modin_result = modin_df.loc[["b"]]
    pandas_result = pandas_df.loc[["b"]]
    df_equals(modin_result, pandas_result)

    modin_result = modin_df.loc[["b", "a"]]
    pandas_result = pandas_df.loc[["b", "a"]]
    df_equals(modin_result, pandas_result)

    modin_result = modin_df.loc[["b", "a"], ["a", "c"]]
    pandas_result = pandas_df.loc[["b", "a"], ["a", "c"]]
    df_equals(modin_result, pandas_result)

    modin_result = modin_df.loc[:, ["a", "c"]]
    pandas_result = pandas_df.loc[:, ["a", "c"]]
    df_equals(modin_result, pandas_result)

    modin_result = modin_df.loc[:, ["c"]]
    pandas_result = pandas_df.loc[:, ["c"]]
    df_equals(modin_result, pandas_result)

    modin_result = modin_df.loc[[]]
    pandas_result = pandas_df.loc[[]]
    df_equals(modin_result, pandas_result)


def test_empty_df():
    df = pd.DataFrame(index=["a", "b"])
    df_is_empty(df)
    assert_index_equal(df.index, pd.Index(["a", "b"]))
    assert len(df.columns) == 0

    df = pd.DataFrame(columns=["a", "b"])
    df_is_empty(df)
    assert len(df.index) == 0
    assert_index_equal(df.columns, pd.Index(["a", "b"]))

    df = pd.DataFrame()
    df_is_empty(df)
    assert len(df.index) == 0
    assert len(df.columns) == 0

    df = pd.DataFrame(index=["a", "b"])
    df_is_empty(df)
    assert_index_equal(df.index, pd.Index(["a", "b"]))
    assert len(df.columns) == 0

    df = pd.DataFrame(columns=["a", "b"])
    df_is_empty(df)
    assert len(df.index) == 0
    assert_index_equal(df.columns, pd.Index(["a", "b"]))

    df = pd.DataFrame()
    df_is_empty(df)
    assert len(df.index) == 0
    assert len(df.columns) == 0

    df = pd.DataFrame()
    pd_df = pandas.DataFrame()
    df["a"] = [1, 2, 3, 4, 5]
    pd_df["a"] = [1, 2, 3, 4, 5]
    df_equals(df, pd_df)

    df = pd.DataFrame()
    pd_df = pandas.DataFrame()
    df["a"] = list("ABCDEF")
    pd_df["a"] = list("ABCDEF")
    df_equals(df, pd_df)

    df = pd.DataFrame()
    pd_df = pandas.DataFrame()
    df["a"] = pd.Series([1, 2, 3, 4, 5])
    pd_df["a"] = pandas.Series([1, 2, 3, 4, 5])
    df_equals(df, pd_df)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_abs(request, data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.abs()
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.abs()
    else:
        modin_result = modin_df.abs()
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_add_prefix(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    test_prefix = "TEST"
    new_modin_df = modin_df.add_prefix(test_prefix)
    new_pandas_df = pandas_df.add_prefix(test_prefix)
    df_equals(new_modin_df.columns, new_pandas_df.columns)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("testfunc", test_func_values, ids=test_func_keys)
def test_applymap(request, data, testfunc):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    with pytest.raises(ValueError):
        x = 2
        modin_df.applymap(x)

    try:
        pandas_result = pandas_df.applymap(testfunc)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.applymap(testfunc)
    else:
        modin_result = modin_df.applymap(testfunc)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("testfunc", test_func_values, ids=test_func_keys)
def test_applymap_numeric(request, data, testfunc):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    if name_contains(request.node.name, numeric_dfs):
        try:
            pandas_result = pandas_df.applymap(testfunc)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.applymap(testfunc)
        else:
            modin_result = modin_df.applymap(testfunc)
            df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_add_suffix(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    test_suffix = "TEST"
    new_modin_df = modin_df.add_suffix(test_suffix)
    new_pandas_df = pandas_df.add_suffix(test_suffix)

    df_equals(new_modin_df.columns, new_pandas_df.columns)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_at(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    key1 = modin_df.columns[0]
    # Scaler
    df_equals(modin_df.at[0, key1], pandas_df.at[0, key1])

    # Series
    df_equals(modin_df.loc[0].at[key1], pandas_df.loc[0].at[key1])

    # Write Item
    modin_df_copy = modin_df.copy()
    pandas_df_copy = pandas_df.copy()
    modin_df_copy.at[1, key1] = modin_df.at[0, key1]
    pandas_df_copy.at[1, key1] = pandas_df.at[0, key1]
    df_equals(modin_df_copy, pandas_df_copy)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_axes(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    for modin_axis, pd_axis in zip(modin_df.axes, pandas_df.axes):
        assert np.array_equal(modin_axis, pd_axis)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_copy(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)  # noqa F841

    # pandas_df is unused but there so there won't be confusing list comprehension
    # stuff in the pytest.mark.parametrize
    new_modin_df = modin_df.copy()

    assert new_modin_df is not modin_df
    if get_current_backend() != "BaseOnPython":
        assert np.array_equal(
            new_modin_df._query_compiler._modin_frame._partitions,
            modin_df._query_compiler._modin_frame._partitions,
        )
    assert new_modin_df is not modin_df
    df_equals(new_modin_df, modin_df)

    # Shallow copy tests
    modin_df = pd.DataFrame(data)
    modin_df_cp = modin_df.copy(False)

    modin_df[modin_df.columns[0]] = 0
    df_equals(modin_df, modin_df_cp)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_dtypes(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    df_equals(modin_df.dtypes, pandas_df.dtypes)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("key", indices_values, ids=indices_keys)
def test_get(data, key):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    df_equals(modin_df.get(key), pandas_df.get(key))
    df_equals(
        modin_df.get(key, default="default"), pandas_df.get(key, default="default")
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize(
    "dummy_na", bool_arg_values, ids=arg_keys("dummy_na", bool_arg_keys)
)
@pytest.mark.parametrize(
    "drop_first", bool_arg_values, ids=arg_keys("drop_first", bool_arg_keys)
)
def test_get_dummies(request, data, dummy_na, drop_first):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas.get_dummies(
            pandas_df, dummy_na=dummy_na, drop_first=drop_first
        )
    except Exception as e:
        with pytest.raises(type(e)):
            pd.get_dummies(modin_df, dummy_na=dummy_na, drop_first=drop_first)
    else:
        modin_result = pd.get_dummies(
            modin_df, dummy_na=dummy_na, drop_first=drop_first
        )
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_isna(data):
    pandas_df = pandas.DataFrame(data)
    modin_df = pd.DataFrame(data)

    pandas_result = pandas_df.isna()
    modin_result = modin_df.isna()

    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_isnull(data):
    pandas_df = pandas.DataFrame(data)
    modin_df = pd.DataFrame(data)

    pandas_result = pandas_df.isnull()
    modin_result = modin_df.isnull()

    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_append(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    data_to_append = {"append_a": 2, "append_b": 1000}

    ignore_idx_values = [True, False]

    for ignore in ignore_idx_values:
        try:
            pandas_result = pandas_df.append(data_to_append, ignore_index=ignore)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.append(data_to_append, ignore_index=ignore)
        else:
            modin_result = modin_df.append(data_to_append, ignore_index=ignore)
            df_equals(modin_result, pandas_result)

    try:
        pandas_result = pandas_df.append(pandas_df.iloc[-1])
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.append(modin_df.iloc[-1])
    else:
        modin_result = modin_df.append(modin_df.iloc[-1])
        df_equals(modin_result, pandas_result)

    try:
        pandas_result = pandas_df.append(list(pandas_df.iloc[-1]))
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.append(list(modin_df.iloc[-1]))
    else:
        modin_result = modin_df.append(list(modin_df.iloc[-1]))
        # Pandas has bug where sort=False is ignored
        # (https://github.com/pandas-dev/pandas/issues/35092), but Modin
        # now does the right thing, so for now manually sort to workaround
        # this. Once the Pandas bug is fixed and Modin upgrades to that
        # Pandas release, this sort will cause the test to fail, and the
        # next three lines should be deleted.
        if get_current_backend() != "BaseOnPython":
            assert list(modin_result.columns) == list(modin_df.columns) + [0]
            modin_result = modin_result[[0] + sorted(modin_df.columns)]
        df_equals(modin_result, pandas_result)

    verify_integrity_values = [True, False]

    for verify_integrity in verify_integrity_values:
        try:
            pandas_result = pandas_df.append(
                [pandas_df, pandas_df], verify_integrity=verify_integrity
            )
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.append([modin_df, modin_df], verify_integrity=verify_integrity)
        else:
            modin_result = modin_df.append(
                [modin_df, modin_df], verify_integrity=verify_integrity
            )
            df_equals(modin_result, pandas_result)

        try:
            pandas_result = pandas_df.append(
                pandas_df, verify_integrity=verify_integrity
            )
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.append(modin_df, verify_integrity=verify_integrity)
        else:
            modin_result = modin_df.append(modin_df, verify_integrity=verify_integrity)
            df_equals(modin_result, pandas_result)


def test_astype():
    td = pandas.DataFrame(test_data["int_data"])[["col1", "index", "col3", "col4"]]
    modin_df = pd.DataFrame(td.values, index=td.index, columns=td.columns)
    expected_df = pandas.DataFrame(td.values, index=td.index, columns=td.columns)

    modin_df_casted = modin_df.astype(np.int32)
    expected_df_casted = expected_df.astype(np.int32)
    df_equals(modin_df_casted, expected_df_casted)

    modin_df_casted = modin_df.astype(np.float64)
    expected_df_casted = expected_df.astype(np.float64)
    df_equals(modin_df_casted, expected_df_casted)

    modin_df_casted = modin_df.astype(str)
    expected_df_casted = expected_df.astype(str)
    df_equals(modin_df_casted, expected_df_casted)

    modin_df_casted = modin_df.astype("category")
    expected_df_casted = expected_df.astype("category")
    df_equals(modin_df_casted, expected_df_casted)

    dtype_dict = {"col1": np.int32, "index": np.int64, "col3": str}
    modin_df_casted = modin_df.astype(dtype_dict)
    expected_df_casted = expected_df.astype(dtype_dict)
    df_equals(modin_df_casted, expected_df_casted)

    # Ignore lint because this is testing bad input
    bad_dtype_dict = {"index": np.int32, "index": np.int64, "index": str}  # noqa F601
    modin_df_casted = modin_df.astype(bad_dtype_dict)
    expected_df_casted = expected_df.astype(bad_dtype_dict)
    df_equals(modin_df_casted, expected_df_casted)

    modin_df = pd.DataFrame(index=["row1"], columns=["col1"])
    modin_df["col1"]["row1"] = 11
    modin_df_casted = modin_df.astype(int)
    expected_df = pandas.DataFrame(index=["row1"], columns=["col1"])
    expected_df["col1"]["row1"] = 11
    expected_df_casted = expected_df.astype(int)
    df_equals(modin_df_casted, expected_df_casted)

    with pytest.raises(KeyError):
        modin_df.astype({"not_exists": np.uint8})


def test_astype_category():
    modin_df = pd.DataFrame(
        {"col1": ["A", "A", "B", "B", "A"], "col2": [1, 2, 3, 4, 5]}
    )
    pandas_df = pandas.DataFrame(
        {"col1": ["A", "A", "B", "B", "A"], "col2": [1, 2, 3, 4, 5]}
    )

    modin_result = modin_df.astype({"col1": "category"})
    pandas_result = pandas_df.astype({"col1": "category"})
    df_equals(modin_result, pandas_result)
    assert modin_result.dtypes.equals(pandas_result.dtypes)

    modin_result = modin_df.astype("category")
    pandas_result = pandas_df.astype("category")
    df_equals(modin_result, pandas_result)
    assert modin_result.dtypes.equals(pandas_result.dtypes)


@pytest.mark.xfail(
    reason="Categorical dataframe created in memory don't work yet and categorical dtype is lost"
)
def test_astype_category_large():
    series_length = 10_000
    modin_df = pd.DataFrame(
        {
            "col1": ["str{0}".format(i) for i in range(0, series_length)],
            "col2": [i for i in range(0, series_length)],
        }
    )
    pandas_df = pandas.DataFrame(
        {
            "col1": ["str{0}".format(i) for i in range(0, series_length)],
            "col2": [i for i in range(0, series_length)],
        }
    )

    modin_result = modin_df.astype({"col1": "category"})
    pandas_result = pandas_df.astype({"col1": "category"})
    df_equals(modin_result, pandas_result)
    assert modin_result.dtypes.equals(pandas_result.dtypes)

    modin_result = modin_df.astype("category")
    pandas_result = pandas_df.astype("category")
    df_equals(modin_result, pandas_result)
    assert modin_result.dtypes.equals(pandas_result.dtypes)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
def test_clip(request, data, axis):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    if name_contains(request.node.name, numeric_dfs):
        ind_len = (
            len(modin_df.index)
            if not pandas.DataFrame()._get_axis_number(axis)
            else len(modin_df.columns)
        )
        # set bounds
        lower, upper = np.sort(random_state.random_integers(RAND_LOW, RAND_HIGH, 2))
        lower_list = random_state.random_integers(RAND_LOW, RAND_HIGH, ind_len)
        upper_list = random_state.random_integers(RAND_LOW, RAND_HIGH, ind_len)

        # test only upper scalar bound
        modin_result = modin_df.clip(None, upper, axis=axis)
        pandas_result = pandas_df.clip(None, upper, axis=axis)
        df_equals(modin_result, pandas_result)

        # test lower and upper scalar bound
        modin_result = modin_df.clip(lower, upper, axis=axis)
        pandas_result = pandas_df.clip(lower, upper, axis=axis)
        df_equals(modin_result, pandas_result)

        # test lower and upper list bound on each column
        modin_result = modin_df.clip(lower_list, upper_list, axis=axis)
        pandas_result = pandas_df.clip(lower_list, upper_list, axis=axis)
        df_equals(modin_result, pandas_result)

        # test only upper list bound on each column
        modin_result = modin_df.clip(np.nan, upper_list, axis=axis)
        pandas_result = pandas_df.clip(np.nan, upper_list, axis=axis)
        df_equals(modin_result, pandas_result)

        with pytest.raises(ValueError):
            modin_df.clip(lower=[1, 2, 3], axis=None)


def test_drop():
    frame_data = {"A": [1, 2, 3, 4], "B": [0, 1, 2, 3]}
    simple = pandas.DataFrame(frame_data)
    modin_simple = pd.DataFrame(frame_data)
    df_equals(modin_simple.drop("A", axis=1), simple[["B"]])
    df_equals(modin_simple.drop(["A", "B"], axis="columns"), simple[[]])
    df_equals(modin_simple.drop([0, 1, 3], axis=0), simple.loc[[2], :])
    df_equals(modin_simple.drop([0, 3], axis="index"), simple.loc[[1, 2], :])

    pytest.raises(ValueError, modin_simple.drop, 5)
    pytest.raises(ValueError, modin_simple.drop, "C", 1)
    pytest.raises(ValueError, modin_simple.drop, [1, 5])
    pytest.raises(ValueError, modin_simple.drop, ["A", "C"], 1)

    # errors = 'ignore'
    df_equals(modin_simple.drop(5, errors="ignore"), simple)
    df_equals(modin_simple.drop([0, 5], errors="ignore"), simple.loc[[1, 2, 3], :])
    df_equals(modin_simple.drop("C", axis=1, errors="ignore"), simple)
    df_equals(modin_simple.drop(["A", "C"], axis=1, errors="ignore"), simple[["B"]])

    # non-unique
    nu_df = pandas.DataFrame(
        zip(range(3), range(-3, 1), list("abc")), columns=["a", "a", "b"]
    )
    modin_nu_df = pd.DataFrame(nu_df)
    df_equals(modin_nu_df.drop("a", axis=1), nu_df[["b"]])
    df_equals(modin_nu_df.drop("b", axis="columns"), nu_df["a"])
    df_equals(modin_nu_df.drop([]), nu_df)

    nu_df = nu_df.set_index(pandas.Index(["X", "Y", "X"]))
    nu_df.columns = list("abc")
    modin_nu_df = pd.DataFrame(nu_df)
    df_equals(modin_nu_df.drop("X", axis="rows"), nu_df.loc[["Y"], :])
    df_equals(modin_nu_df.drop(["X", "Y"], axis=0), nu_df.loc[[], :])

    # inplace cache issue
    frame_data = random_state.randn(10, 3)
    df = pandas.DataFrame(frame_data, columns=list("abc"))
    modin_df = pd.DataFrame(frame_data, columns=list("abc"))
    expected = df[~(df.b > 0)]
    modin_df.drop(labels=df[df.b > 0].index, inplace=True)
    df_equals(modin_df, expected)

    midx = pd.MultiIndex(
        levels=[["lama", "cow", "falcon"], ["speed", "weight", "length"]],
        codes=[[0, 0, 0, 1, 1, 1, 2, 2, 2], [0, 1, 2, 0, 1, 2, 0, 1, 2]],
    )
    df = pd.DataFrame(
        index=midx,
        columns=["big", "small"],
        data=[
            [45, 30],
            [200, 100],
            [1.5, 1],
            [30, 20],
            [250, 150],
            [1.5, 0.8],
            [320, 250],
            [1, 0.8],
            [0.3, 0.2],
        ],
    )
    with pytest.warns(UserWarning):
        df.drop(index="length", level=1)


def test_drop_api_equivalence():
    # equivalence of the labels/axis and index/columns API's
    frame_data = [[1, 2, 3], [3, 4, 5], [5, 6, 7]]

    modin_df = pd.DataFrame(frame_data, index=["a", "b", "c"], columns=["d", "e", "f"])

    modin_df1 = modin_df.drop("a")
    modin_df2 = modin_df.drop(index="a")
    df_equals(modin_df1, modin_df2)

    modin_df1 = modin_df.drop("d", 1)
    modin_df2 = modin_df.drop(columns="d")
    df_equals(modin_df1, modin_df2)

    modin_df1 = modin_df.drop(labels="e", axis=1)
    modin_df2 = modin_df.drop(columns="e")
    df_equals(modin_df1, modin_df2)

    modin_df1 = modin_df.drop(["a"], axis=0)
    modin_df2 = modin_df.drop(index=["a"])
    df_equals(modin_df1, modin_df2)

    modin_df1 = modin_df.drop(["a"], axis=0).drop(["d"], axis=1)
    modin_df2 = modin_df.drop(index=["a"], columns=["d"])
    df_equals(modin_df1, modin_df2)

    with pytest.raises(ValueError):
        modin_df.drop(labels="a", index="b")

    with pytest.raises(ValueError):
        modin_df.drop(labels="a", columns="b")

    with pytest.raises(ValueError):
        modin_df.drop(axis=1)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_drop_transpose(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    modin_result = modin_df.T.drop(columns=[0, 1, 2])
    pandas_result = pandas_df.T.drop(columns=[0, 1, 2])
    df_equals(modin_result, pandas_result)

    modin_result = modin_df.T.drop(index=["col3", "col1"])
    pandas_result = pandas_df.T.drop(index=["col3", "col1"])
    df_equals(modin_result, pandas_result)

    modin_result = modin_df.T.drop(columns=[0, 1, 2], index=["col3", "col1"])
    pandas_result = pandas_df.T.drop(columns=[0, 1, 2], index=["col3", "col1"])
    df_equals(modin_result, pandas_result)


def test_droplevel():
    df = (
        pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        .set_index([0, 1])
        .rename_axis(["a", "b"])
    )
    df.columns = pd.MultiIndex.from_tuples(
        [("c", "e"), ("d", "f")], names=["level_1", "level_2"]
    )
    df.droplevel("a")
    df.droplevel("level_2", axis=1)


@pytest.mark.parametrize(
    "data", test_data_with_duplicates_values, ids=test_data_with_duplicates_keys
)
@pytest.mark.parametrize(
    "keep", ["last", "first", False], ids=["last", "first", "False"]
)
@pytest.mark.parametrize(
    "subset",
    [None, "col1", "name", ("col1", "col3"), ["col1", "col3", "col7"]],
    ids=["None", "string", "name", "tuple", "list"],
)
def test_drop_duplicates(data, keep, subset):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_df.drop_duplicates(keep=keep, inplace=False, subset=subset)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.drop_duplicates(keep=keep, inplace=False, subset=subset)
    else:
        df_equals(
            pandas_df.drop_duplicates(keep=keep, inplace=False, subset=subset),
            modin_df.drop_duplicates(keep=keep, inplace=False, subset=subset),
        )

    try:
        pandas_results = pandas_df.drop_duplicates(
            keep=keep, inplace=True, subset=subset
        )
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.drop_duplicates(keep=keep, inplace=True, subset=subset)
    else:
        modin_results = modin_df.drop_duplicates(keep=keep, inplace=True, subset=subset)
        df_equals(modin_results, pandas_results)


def test_drop_duplicates_with_missing_index_values():
    data = {
        "columns": ["value", "time", "id"],
        "index": [
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
        ],
        "data": [
            ["3", 1279213398000.0, 88.0],
            ["3", 1279204682000.0, 88.0],
            ["0", 1245772835000.0, 448.0],
            ["0", 1270564258000.0, 32.0],
            ["0", 1267106669000.0, 118.0],
            ["7", 1300621123000.0, 5.0],
            ["0", 1251130752000.0, 957.0],
            ["0", 1311683506000.0, 62.0],
            ["9", 1283692698000.0, 89.0],
            ["9", 1270234253000.0, 64.0],
            ["0", 1285088818000.0, 50.0],
            ["0", 1218212725000.0, 695.0],
            ["2", 1383933968000.0, 348.0],
            ["0", 1368227625000.0, 257.0],
            ["1", 1454514093000.0, 446.0],
            ["1", 1428497427000.0, 134.0],
            ["1", 1459184936000.0, 568.0],
            ["1", 1502293302000.0, 599.0],
            ["1", 1491833358000.0, 829.0],
            ["1", 1485431534000.0, 806.0],
            ["8", 1351800505000.0, 101.0],
            ["0", 1357247721000.0, 916.0],
            ["0", 1335804423000.0, 370.0],
            ["24", 1327547726000.0, 720.0],
            ["0", 1332334140000.0, 415.0],
            ["0", 1309543100000.0, 30.0],
            ["18", 1309541141000.0, 30.0],
            ["0", 1298979435000.0, 48.0],
            ["14", 1276098160000.0, 59.0],
            ["0", 1233936302000.0, 109.0],
        ],
    }

    pandas_df = pandas.DataFrame(
        data["data"], index=data["index"], columns=data["columns"]
    )
    modin_df = pd.DataFrame(data["data"], index=data["index"], columns=data["columns"])
    modin_result = modin_df.sort_values(["id", "time"]).drop_duplicates(["id"])
    pandas_result = pandas_df.sort_values(["id", "time"]).drop_duplicates(["id"])
    df_equals(modin_result, pandas_result)


def test_drop_duplicates_after_sort():
    data = [
        {"value": 1, "time": 2},
        {"value": 1, "time": 1},
        {"value": 2, "time": 1},
        {"value": 2, "time": 2},
    ]
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    modin_result = modin_df.sort_values(["value", "time"]).drop_duplicates(["value"])
    pandas_result = pandas_df.sort_values(["value", "time"]).drop_duplicates(["value"])
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("how", ["any", "all"], ids=["any", "all"])
def test_dropna(data, axis, how):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    with pytest.raises(ValueError):
        modin_df.dropna(axis=axis, how="invalid")

    with pytest.raises(TypeError):
        modin_df.dropna(axis=axis, how=None, thresh=None)

    with pytest.raises(KeyError):
        modin_df.dropna(axis=axis, subset=["NotExists"], how=how)

    modin_result = modin_df.dropna(axis=axis, how=how)
    pandas_result = pandas_df.dropna(axis=axis, how=how)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_dropna_inplace(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    pandas_result = pandas_df.dropna()
    modin_df.dropna(inplace=True)
    df_equals(modin_df, pandas_result)

    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    pandas_df.dropna(thresh=2, inplace=True)
    modin_df.dropna(thresh=2, inplace=True)
    df_equals(modin_df, pandas_df)

    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    pandas_df.dropna(axis=1, how="any", inplace=True)
    modin_df.dropna(axis=1, how="any", inplace=True)
    df_equals(modin_df, pandas_df)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_dropna_multiple_axes(data):
    modin_df = pd.DataFrame(data)

    with pytest.raises(TypeError):
        modin_df.dropna(how="all", axis=[0, 1])
    with pytest.raises(TypeError):
        modin_df.dropna(how="all", axis=(0, 1))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_dropna_subset(request, data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    if "empty_data" not in request.node.name:
        column_subset = modin_df.columns[0:2]
        df_equals(
            modin_df.dropna(how="all", subset=column_subset),
            pandas_df.dropna(how="all", subset=column_subset),
        )
        df_equals(
            modin_df.dropna(how="any", subset=column_subset),
            pandas_df.dropna(how="any", subset=column_subset),
        )

        row_subset = modin_df.index[0:2]
        df_equals(
            modin_df.dropna(how="all", axis=1, subset=row_subset),
            pandas_df.dropna(how="all", axis=1, subset=row_subset),
        )
        df_equals(
            modin_df.dropna(how="any", axis=1, subset=row_subset),
            pandas_df.dropna(how="any", axis=1, subset=row_subset),
        )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis,subset", [(0, list("EF")), (1, [4, 5])])
def test_dropna_subset_error(data, axis, subset):
    eval_general(*create_test_dfs(data), lambda df: df.dropna(axis=axis, subset=subset))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("astype", ["category", "int32", "float"])
def test_insert_dtypes(data, astype):
    modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)

    # categories with NaN works incorrect for now
    if astype == "category" and pandas_df.iloc[:, 0].isnull().any():
        return

    eval_insert(
        modin_df,
        pandas_df,
        col="TypeSaver",
        value=lambda df: df.iloc[:, 0].astype(astype),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("loc", int_arg_values, ids=arg_keys("loc", int_arg_keys))
def test_insert_loc(data, loc):
    modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)
    value = modin_df.iloc[:, 0]

    eval_insert(modin_df, pandas_df, loc=loc, value=value)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_insert(data):
    modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)

    eval_insert(
        modin_df, pandas_df, col="Duplicate", value=lambda df: df[df.columns[0]]
    )
    eval_insert(modin_df, pandas_df, col="Scalar", value=100)
    eval_insert(
        pd.DataFrame(columns=list("ab")),
        pandas.DataFrame(columns=list("ab")),
        col=lambda df: df.columns[0],
        value=lambda df: df[df.columns[0]],
    )
    eval_insert(
        pd.DataFrame(index=modin_df.index),
        pandas.DataFrame(index=pandas_df.index),
        col=lambda df: df.columns[0],
        value=lambda df: df[df.columns[0]],
    )
    eval_insert(
        modin_df,
        pandas_df,
        col="DataFrame insert",
        value=lambda df: df[[df.columns[0]]],
    )
    eval_insert(
        modin_df,
        pandas_df,
        col="Different indices",
        value=lambda df: df[[df.columns[0]]].set_index(df.index[::-1]),
    )

    # Bad inserts
    eval_insert(modin_df, pandas_df, col="Bad Column", value=lambda df: df)
    eval_insert(
        modin_df,
        pandas_df,
        col="Too Short",
        value=lambda df: list(df[df.columns[0]])[:-1],
    )
    eval_insert(
        modin_df,
        pandas_df,
        col=lambda df: df.columns[0],
        value=lambda df: df[df.columns[0]],
    )
    eval_insert(
        modin_df,
        pandas_df,
        loc=lambda df: len(df.columns) + 100,
        col="Bad Loc",
        value=100,
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_ndim(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    assert modin_df.ndim == pandas_df.ndim


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_notna(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    df_equals(modin_df.notna(), pandas_df.notna())


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_notnull(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    df_equals(modin_df.notnull(), pandas_df.notnull())


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_round(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    df_equals(modin_df.round(), pandas_df.round())
    df_equals(modin_df.round(1), pandas_df.round(1))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
def test_set_axis(data, axis):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    x = pandas.DataFrame()._get_axis_number(axis)
    index = modin_df.columns if x else modin_df.index
    labels = ["{0}_{1}".format(index[i], i) for i in range(modin_df.shape[x])]

    modin_result = modin_df.set_axis(labels, axis=axis, inplace=False)
    pandas_result = pandas_df.set_axis(labels, axis=axis, inplace=False)
    df_equals(modin_result, pandas_result)

    modin_df_copy = modin_df.copy()
    modin_df.set_axis(labels, axis=axis, inplace=True)

    # Check that the copy and original are different
    try:
        df_equals(modin_df, modin_df_copy)
    except AssertionError:
        assert True
    else:
        assert False

    pandas_df.set_axis(labels, axis=axis, inplace=True)
    df_equals(modin_df, pandas_df)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("drop", bool_arg_values, ids=arg_keys("drop", bool_arg_keys))
@pytest.mark.parametrize(
    "append", bool_arg_values, ids=arg_keys("append", bool_arg_keys)
)
def test_set_index(request, data, drop, append):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    if "empty_data" not in request.node.name:
        key = modin_df.columns[0]
        modin_result = modin_df.set_index(key, drop=drop, append=append, inplace=False)
        pandas_result = pandas_df.set_index(
            key, drop=drop, append=append, inplace=False
        )
        df_equals(modin_result, pandas_result)

        modin_df_copy = modin_df.copy()
        modin_df.set_index(key, drop=drop, append=append, inplace=True)

        # Check that the copy and original are different
        try:
            df_equals(modin_df, modin_df_copy)
        except AssertionError:
            assert True
        else:
            assert False

        pandas_df.set_index(key, drop=drop, append=append, inplace=True)
        df_equals(modin_df, pandas_df)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_shape(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    assert modin_df.shape == pandas_df.shape


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_size(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    assert modin_df.size == pandas_df.size


def test_squeeze():
    frame_data = {
        "col1": [0, 1, 2, 3],
        "col2": [4, 5, 6, 7],
        "col3": [8, 9, 10, 11],
        "col4": [12, 13, 14, 15],
        "col5": [0, 0, 0, 0],
    }
    frame_data_2 = {"col1": [0, 1, 2, 3]}
    frame_data_3 = {
        "col1": [0],
        "col2": [4],
        "col3": [8],
        "col4": [12],
        "col5": [0],
    }
    frame_data_4 = {"col1": [2]}
    frame_data_5 = {"col1": ["string"]}
    # Different data for different cases
    pandas_df = pandas.DataFrame(frame_data).squeeze()
    modin_df = pd.DataFrame(frame_data).squeeze()
    df_equals(modin_df, pandas_df)

    pandas_df_2 = pandas.DataFrame(frame_data_2).squeeze()
    modin_df_2 = pd.DataFrame(frame_data_2).squeeze()
    df_equals(modin_df_2, pandas_df_2)

    pandas_df_3 = pandas.DataFrame(frame_data_3).squeeze()
    modin_df_3 = pd.DataFrame(frame_data_3).squeeze()
    df_equals(modin_df_3, pandas_df_3)

    pandas_df_4 = pandas.DataFrame(frame_data_4).squeeze()
    modin_df_4 = pd.DataFrame(frame_data_4).squeeze()
    df_equals(modin_df_4, pandas_df_4)

    pandas_df_5 = pandas.DataFrame(frame_data_5).squeeze()
    modin_df_5 = pd.DataFrame(frame_data_5).squeeze()
    df_equals(modin_df_5, pandas_df_5)

    data = [
        [
            pd.Timestamp("2019-01-02"),
            pd.Timestamp("2019-01-03"),
            pd.Timestamp("2019-01-04"),
            pd.Timestamp("2019-01-05"),
        ],
        [1, 1, 1, 2],
    ]
    df = pd.DataFrame(data, index=["date", "value"]).T
    pf = pandas.DataFrame(data, index=["date", "value"]).T
    df.set_index("date", inplace=True)
    pf.set_index("date", inplace=True)
    df_equals(df.iloc[0], pf.iloc[0])


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_transpose(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    df_equals(modin_df.T, pandas_df.T)
    df_equals(modin_df.transpose(), pandas_df.transpose())

    # Test for map across full axis for select indices
    df_equals(modin_df.T.dropna(), pandas_df.T.dropna())
    # Test for map across full axis
    df_equals(modin_df.T.nunique(), pandas_df.T.nunique())
    # Test for map across blocks
    df_equals(modin_df.T.notna(), pandas_df.T.notna())


@pytest.mark.parametrize(
    "data, other_data",
    [
        ({"A": [1, 2, 3], "B": [400, 500, 600]}, {"B": [4, 5, 6], "C": [7, 8, 9]}),
        (
            {"A": ["a", "b", "c"], "B": ["x", "y", "z"]},
            {"B": ["d", "e", "f", "g", "h", "i"]},
        ),
        ({"A": [1, 2, 3], "B": [400, 500, 600]}, {"B": [4, np.nan, 6]}),
    ],
)
def test_update(data, other_data):
    modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)
    other_modin_df, other_pandas_df = (
        pd.DataFrame(other_data),
        pandas.DataFrame(other_data),
    )
    modin_df.update(other_modin_df)
    pandas_df.update(other_pandas_df)
    df_equals(modin_df, pandas_df)

    with pytest.raises(ValueError):
        modin_df.update(other_modin_df, errors="raise")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___neg__(request, data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.__neg__()
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.__neg__()
    else:
        modin_result = modin_df.__neg__()
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___invert__(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    try:
        pandas_result = ~pandas_df
    except Exception as e:
        with pytest.raises(type(e)):
            repr(~modin_df)
    else:
        modin_result = ~modin_df
        df_equals(modin_result, pandas_result)


def test___hash__():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        try:
            pd.DataFrame(data).__hash__()
        except TypeError:
            pass


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___delitem__(request, data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    if "empty_data" not in request.node.name:
        key = pandas_df.columns[0]

        modin_df = modin_df.copy()
        pandas_df = pandas_df.copy()
        modin_df.__delitem__(key)
        pandas_df.__delitem__(key)
        df_equals(modin_df, pandas_df)

        # Issue 2027
        last_label = pandas_df.iloc[:, -1].name
        modin_df.__delitem__(last_label)
        pandas_df.__delitem__(last_label)
        df_equals(modin_df, pandas_df)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___nonzero__(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)  # noqa F841

    with pytest.raises(ValueError):
        # Always raises ValueError
        modin_df.__nonzero__()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___abs__(request, data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = abs(pandas_df)
    except Exception as e:
        with pytest.raises(type(e)):
            abs(modin_df)
    else:
        modin_result = abs(modin_df)
        df_equals(modin_result, pandas_result)


def test___round__():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).__round__()
