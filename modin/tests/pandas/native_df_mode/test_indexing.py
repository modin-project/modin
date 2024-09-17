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
from itertools import product

import matplotlib
import numpy as np
import pandas
import pytest

import modin.pandas as pd
from modin.config import NativeDataframeMode, NPartitions
from modin.tests.pandas.native_df_mode.utils import (
    create_test_df_in_defined_mode,
    create_test_series_in_defined_mode,
    eval_general_interop,
)
from modin.tests.pandas.utils import (
    RAND_HIGH,
    RAND_LOW,
    default_to_pandas_ignore_string,
    df_equals,
    eval_general,
    test_data,
    test_data_keys,
    test_data_values,
)

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
    df_mode_pair_list = list(product(NativeDataframeMode.choices, repeat=2))
    for df_mode_pair in df_mode_pair_list:
        eval_general_interop(
            pd_df,
            None,
            lambda df1, df2: df1.__setitem__(col, value_getter(df2)),
            df_mode_pair,
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
@pytest.mark.parametrize(
    "df_mode_pair", list(product(NativeDataframeMode.choices, repeat=2))
)
def test_set_index(data, key_func, drop_kwargs, request, df_mode_pair):
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

    eval_general_interop(
        data,
        None,
        lambda df1, df2: df1.set_index(key_func(df2), **drop_kwargs),
        expected_exception=expected_exception,
        df_mode_pair=df_mode_pair,
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize(
    "df_mode_pair", list(product(NativeDataframeMode.choices, repeat=2))
)
def test_loc(data, df_mode_pair):
    modin_df, pandas_df = create_test_df_in_defined_mode(data, df_mode=df_mode_pair[0])

    indices = [i % 3 == 0 for i in range(len(modin_df.index))]
    columns = [i % 5 == 0 for i in range(len(modin_df.columns))]

    # Key is a Modin or pandas series of booleans
    series1, _ = create_test_series_in_defined_mode(indices, df_mode=df_mode_pair[0])
    series2, _ = create_test_series_in_defined_mode(
        columns, index=modin_df.columns, df_mode=df_mode_pair[0]
    )
    df_equals(
        modin_df.loc[series1, series2],
        pandas_df.loc[
            pandas.Series(indices), pandas.Series(columns, index=modin_df.columns)
        ],
    )


@pytest.mark.parametrize("left, right", [(2, 1), (6, 1), (lambda df: 70, 1), (90, 70)])
@pytest.mark.parametrize(
    "df_mode_pair", list(product(NativeDataframeMode.choices, repeat=2))
)
def test_loc_insert_row(left, right, df_mode_pair):
    # This test case comes from
    # https://github.com/modin-project/modin/issues/3764
    data = [[1, 2, 3], [4, 5, 6]]

    def _test_loc_rows(df1, df2):
        df1.loc[left] = df2.loc[right]
        return df1

    expected_exception = None
    if right == 70:
        pytest.xfail(reason="https://github.com/modin-project/modin/issues/7024")

    eval_general_interop(
        data,
        None,
        _test_loc_rows,
        expected_exception=expected_exception,
        df_mode_pair=df_mode_pair,
    )


@pytest.fixture(params=list(product(NativeDataframeMode.choices, repeat=2)))
def loc_iter_dfs_interop(request):
    df_mode_pair = request.param
    columns = ["col1", "col2", "col3"]
    index = ["row1", "row2", "row3"]
    md_df1, pd_df1 = create_test_df_in_defined_mode(
        {col: ([idx] * len(index)) for idx, col in enumerate(columns)},
        columns=columns,
        index=index,
        df_mode=df_mode_pair[0],
    )
    md_df2, pd_df2 = create_test_df_in_defined_mode(
        {col: ([idx] * len(index)) for idx, col in enumerate(columns)},
        columns=columns,
        index=index,
        df_mode=df_mode_pair[1],
    )
    return md_df1, pd_df1, md_df2, pd_df2


@pytest.mark.parametrize("reverse_order", [False, True])
@pytest.mark.parametrize("axis", [0, 1])
def test_loc_iter_assignment(loc_iter_dfs_interop, reverse_order, axis):
    if reverse_order and axis:
        pytest.xfail(
            "Due to internal sorting of lookup values assignment order is lost, see GH-#2552"
        )

    md_df1, pd_df1, md_df2, pd_df2 = loc_iter_dfs_interop

    select = [slice(None), slice(None)]
    select[axis] = sorted(pd_df1.axes[axis][:-1], reverse=reverse_order)
    select = tuple(select)

    pd_df1.loc[select] = pd_df1.loc[select] + pd_df2.loc[select]
    md_df1.loc[select] = md_df1.loc[select] + md_df2.loc[select]
    df_equals(md_df1, pd_df1)


@pytest.mark.parametrize(
    "df_mode_pair", list(product(NativeDataframeMode.choices, repeat=2))
)
def test_loc_series(df_mode_pair):
    md_df1, pd_df1 = create_test_df_in_defined_mode(
        {"a": [1, 2], "b": [3, 4]}, df_mode=df_mode_pair[0]
    )
    md_df2, pd_df2 = create_test_df_in_defined_mode(
        {"a": [1, 2], "b": [3, 4]}, df_mode=df_mode_pair[1]
    )

    pd_df1.loc[pd_df2["a"] > 1, "b"] = np.log(pd_df1["b"])
    md_df1.loc[md_df2["a"] > 1, "b"] = np.log(md_df1["b"])

    df_equals(pd_df1, md_df1)


@pytest.mark.parametrize(
    "df_mode_pair", list(product(NativeDataframeMode.choices, repeat=2))
)
def test_reindex_like(df_mode_pair):
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
    modin_df1, pandas_df1 = create_test_df_in_defined_mode(
        o_data,
        columns=o_columns,
        index=o_index,
        df_mode=df_mode_pair[0],
    )
    modin_df2, pandas_df2 = create_test_df_in_defined_mode(
        new_data,
        columns=new_columns,
        index=new_index,
        df_mode=df_mode_pair[1],
    )
    modin_result = modin_df2.reindex_like(modin_df1)
    pandas_result = pandas_df2.reindex_like(pandas_df1)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "df_mode_pair", list(product(NativeDataframeMode.choices, repeat=2))
)
def test_reindex_multiindex(df_mode_pair):
    data1, data2 = np.random.randint(1, 20, (5, 5)), np.random.randint(10, 25, 6)
    index = np.array(["AUD", "BRL", "CAD", "EUR", "INR"])
    pandas_midx = pandas.MultiIndex.from_product(
        [["Bank_1", "Bank_2"], ["AUD", "CAD", "EUR"]], names=["Bank", "Curency"]
    )
    modin_df1, pandas_df1 = create_test_df_in_defined_mode(
        data=data1, index=index, columns=index, df_mode=df_mode_pair[0]
    )
    modin_df2, pandas_df2 = create_test_df_in_defined_mode(
        data=data2, index=pandas_midx, df_mode=df_mode_pair[1]
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


@pytest.mark.parametrize(
    "df_mode_pair", list(product(NativeDataframeMode.choices, repeat=2))
)
def test_getitem_empty_mask(df_mode_pair):
    # modin-project/modin#517
    modin_frames = []
    pandas_frames = []
    data1 = np.random.randint(0, 100, size=(100, 4))
    mdf1, pdf1 = create_test_df_in_defined_mode(
        data1, columns=list("ABCD"), df_mode=df_mode_pair[0]
    )

    modin_frames.append(mdf1)
    pandas_frames.append(pdf1)

    data2 = np.random.randint(0, 100, size=(100, 4))
    mdf2, pdf2 = create_test_df_in_defined_mode(
        data2, columns=list("ABCD"), df_mode=df_mode_pair[1]
    )
    modin_frames.append(mdf2)
    pandas_frames.append(pdf2)

    data3 = np.random.randint(0, 100, size=(100, 4))
    mdf3, pdf3 = create_test_df_in_defined_mode(
        data3, columns=list("ABCD"), df_mode=df_mode_pair[0]
    )
    modin_frames.append(mdf3)
    pandas_frames.append(pdf3)

    modin_data = pd.concat(modin_frames)
    pandas_data = pandas.concat(pandas_frames)
    df_equals(
        modin_data[[False for _ in modin_data.index]],
        pandas_data[[False for _ in modin_data.index]],
    )


@pytest.mark.parametrize(
    "df_mode_pair", list(product(NativeDataframeMode.choices, repeat=2))
)
def test___setitem__mask(df_mode_pair):
    # DataFrame mask:
    data = test_data["int_data"]
    modin_df1, pandas_df1 = create_test_df_in_defined_mode(
        data, df_mode=df_mode_pair[0]
    )
    modin_df2, pandas_df2 = create_test_df_in_defined_mode(
        data, df_mode=df_mode_pair[0]
    )

    mean = int((RAND_HIGH + RAND_LOW) / 2)
    pandas_df1[pandas_df2 > mean] = -50
    modin_df1[modin_df2 > mean] = -50

    df_equals(modin_df1, pandas_df1)


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
@pytest.mark.parametrize(
    "df_mode_pair", list(product(NativeDataframeMode.choices, repeat=2))
)
def test_setitem_on_empty_df(data, value, convert_to_series, new_col_id, df_mode_pair):
    modin_df, pandas_df = create_test_df_in_defined_mode(data, df_mode=df_mode_pair[0])

    def applyier(df):
        if convert_to_series:
            converted_value = (
                pandas.Series(value)
                if isinstance(df, pandas.DataFrame)
                else create_test_series_in_defined_mode(value, df_mode=df_mode_pair[1])[
                    1
                ]
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


@pytest.mark.parametrize(
    "df_mode_pair", list(product(NativeDataframeMode.choices, repeat=2))
)
def test_setitem_on_empty_df_4407(df_mode_pair):
    data = {}
    index = pd.date_range(end="1/1/2018", periods=0, freq="D")
    column = pd.date_range(end="1/1/2018", periods=1, freq="h")[0]
    modin_df, pandas_df = create_test_df_in_defined_mode(
        data, columns=index, df_mode=df_mode_pair[0]
    )
    modin_ser, pandas_ser = create_test_series_in_defined_mode(
        [1], df_mode=df_mode_pair[1]
    )
    modin_df[column] = modin_ser
    pandas_df[column] = pandas_ser

    df_equals(modin_df, pandas_df)
    assert modin_df.columns.freq == pandas_df.columns.freq


@pytest.mark.parametrize(
    "df_mode_pair", list(product(NativeDataframeMode.choices, repeat=2))
)
def test_setitem_2d_insertion(df_mode_pair):
    def build_value_picker(modin_value, pandas_value):
        """Build a function that returns either Modin or pandas DataFrame depending on the passed frame."""
        return lambda source_df, *args, **kwargs: (
            modin_value
            if isinstance(source_df, (pd.DataFrame, pd.Series))
            else pandas_value
        )

    modin_df, pandas_df = create_test_df_in_defined_mode(
        test_data["int_data"], df_mode=df_mode_pair[0]
    )

    # Easy case - key and value.columns are equal
    modin_value, pandas_value = create_test_df_in_defined_mode(
        {
            "new_value1": np.arange(len(modin_df)),
            "new_value2": np.arange(len(modin_df)),
        },
        df_mode=df_mode_pair[1],
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
@pytest.mark.parametrize(
    "df_mode_pair", list(product(NativeDataframeMode.choices, repeat=2))
)
def test_setitem_2d_update(does_value_have_different_columns, df_mode_pair):
    def test(dfs, iloc):
        """Update columns on the given numeric indices."""
        df1, df2 = dfs
        cols1 = df1.columns[iloc].tolist()
        cols2 = df2.columns[iloc].tolist()
        df1[cols1] = df2[cols2]
        return df1

    modin_df, pandas_df = create_test_df_in_defined_mode(
        test_data["int_data"], df_mode=df_mode_pair[0]
    )
    modin_df2, pandas_df2 = create_test_df_in_defined_mode(
        test_data["int_data"], df_mode=df_mode_pair[1]
    )
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


@pytest.mark.parametrize(
    "df_mode_pair", list(product(NativeDataframeMode.choices, repeat=2))
)
def test___setitem__single_item_in_series(df_mode_pair):
    # Test assigning a single item in a Series for issue
    # https://github.com/modin-project/modin/issues/3860
    modin_series1, pandas_series1 = create_test_series_in_defined_mode(
        99, df_mode=df_mode_pair[0]
    )
    modin_series2, pandas_series2 = create_test_series_in_defined_mode(
        100, df_mode=df_mode_pair[1]
    )
    modin_series1[:1] = modin_series2
    pandas_series1[:1] = pandas_series2
    df_equals(modin_series1, pandas_series1)


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
@pytest.mark.parametrize(
    "df_mode_pair", list(product(NativeDataframeMode.choices, repeat=2))
)
def test_loc_boolean_assignment_scalar_dtypes(value, df_mode_pair):
    modin_df, pandas_df = create_test_df_in_defined_mode(
        {
            "a": [1, 2, 3],
            "b": [3.0, 5.0, 6.0],
            "c": ["a", "b", "c"],
            "d": [1.0, "c", 2.0],
            "e": pandas.to_datetime(["1/1/2018", "1/2/2018", "1/3/2018"]),
            "f": [True, False, True],
        },
        df_mode=df_mode_pair[1],
    )
    modin_idx, pandas_idx = create_test_series_in_defined_mode(
        [False, True, True], df_mode=df_mode_pair[1]
    )

    modin_df.loc[modin_idx] = value
    pandas_df.loc[pandas_idx] = value
    df_equals(modin_df, pandas_df)


# This is a very subtle bug that comes from:
# https://github.com/modin-project/modin/issues/4945
@pytest.mark.parametrize(
    "df_mode_pair", list(product(NativeDataframeMode.choices, repeat=2))
)
def test_lazy_eval_index(df_mode_pair):
    data = {"col0": [0, 1]}

    def func(df1, df2):
        df_copy = df1[df2["col0"] < 6].copy()
        # The problem here is that the index is not copied over so it needs
        # to get recomputed at some point. Our implementation of __setitem__
        # requires us to build a mask and insert the value from the right
        # handside into the new DataFrame. However, it's possible that we
        # won't have any new partitions, so we will end up computing an empty
        # index.
        df_copy["col0"] = df_copy["col0"].apply(lambda x: x + 1)
        return df_copy

    eval_general_interop(data, None, func, df_mode_pair=df_mode_pair)


@pytest.mark.parametrize(
    "df_mode_pair", list(product(NativeDataframeMode.choices, repeat=2))
)
def test_index_of_empty_frame(df_mode_pair):
    # Test on an empty frame created by user

    # Test on an empty frame produced by Modin's logic
    data = test_data_values[0]
    md_df1, pd_df1 = create_test_df_in_defined_mode(
        data,
        index=pandas.RangeIndex(len(next(iter(data.values()))), name="index name"),
        df_mode=df_mode_pair[0],
    )
    md_df2, pd_df2 = create_test_df_in_defined_mode(
        data,
        index=pandas.RangeIndex(len(next(iter(data.values()))), name="index name"),
        df_mode=df_mode_pair[1],
    )

    md_res = md_df1.query(f"{md_df2.columns[0]} > {RAND_HIGH}")
    pd_res = pd_df1.query(f"{pd_df2.columns[0]} > {RAND_HIGH}")

    assert md_res.empty and pd_res.empty
    df_equals(md_res.index, pd_res.index)
