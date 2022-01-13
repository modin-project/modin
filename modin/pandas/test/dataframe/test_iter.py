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
import io

from modin.pandas.test.utils import (
    random_state,
    RAND_LOW,
    RAND_HIGH,
    df_equals,
    test_data_values,
    test_data_keys,
    create_test_dfs,
    test_data,
)
from modin.config import NPartitions
from modin.test.test_utils import warns_that_defaulting_to_pandas

NPartitions.put(4)

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")


@pytest.mark.parametrize("method", ["items", "iteritems", "iterrows"])
def test_items_iteritems_iterrows(method):
    data = test_data["float_nan_data"]
    modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)

    for modin_item, pandas_item in zip(
        getattr(modin_df, method)(), getattr(pandas_df, method)()
    ):
        modin_index, modin_series = modin_item
        pandas_index, pandas_series = pandas_item
        df_equals(pandas_series, modin_series)
        assert pandas_index == modin_index


@pytest.mark.parametrize("name", [None, "NotPandas"])
def test_itertuples_name(name):
    data = test_data["float_nan_data"]
    modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)

    modin_it_custom = modin_df.itertuples(name=name)
    pandas_it_custom = pandas_df.itertuples(name=name)
    for modin_row, pandas_row in zip(modin_it_custom, pandas_it_custom):
        np.testing.assert_equal(modin_row, pandas_row)


def test_itertuples_multiindex():
    data = test_data["int_data"]
    modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)

    new_idx = pd.MultiIndex.from_tuples(
        [(i // 4, i // 2, i) for i in range(len(modin_df.columns))]
    )
    modin_df.columns = new_idx
    pandas_df.columns = new_idx
    modin_it_custom = modin_df.itertuples()
    pandas_it_custom = pandas_df.itertuples()
    for modin_row, pandas_row in zip(modin_it_custom, pandas_it_custom):
        np.testing.assert_equal(modin_row, pandas_row)


def test___iter__():
    modin_df = pd.DataFrame(test_data_values[0])
    pandas_df = pandas.DataFrame(test_data_values[0])

    modin_iterator = modin_df.__iter__()

    # Check that modin_iterator implements the iterator interface
    assert hasattr(modin_iterator, "__iter__")
    assert hasattr(modin_iterator, "next") or hasattr(modin_iterator, "__next__")

    pd_iterator = pandas_df.__iter__()
    assert list(modin_iterator) == list(pd_iterator)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___contains__(request, data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    result = False
    key = "Not Exist"
    assert result == modin_df.__contains__(key)
    assert result == (key in modin_df)

    if "empty_data" not in request.node.name:
        result = True
        key = pandas_df.columns[0]
        assert result == modin_df.__contains__(key)
        assert result == (key in modin_df)


def test__options_display():
    frame_data = random_state.randint(RAND_LOW, RAND_HIGH, size=(1000, 102))
    pandas_df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)

    pandas.options.display.max_rows = 10
    pandas.options.display.max_columns = 10
    x = repr(pandas_df)
    pd.options.display.max_rows = 5
    pd.options.display.max_columns = 5
    y = repr(modin_df)
    assert x != y
    pd.options.display.max_rows = 10
    pd.options.display.max_columns = 10
    y = repr(modin_df)
    assert x == y

    # test for old fixed max values
    pandas.options.display.max_rows = 75
    pandas.options.display.max_columns = 75
    x = repr(pandas_df)
    pd.options.display.max_rows = 75
    pd.options.display.max_columns = 75
    y = repr(modin_df)
    assert x == y


def test___finalize__():
    data = test_data_values[0]
    with warns_that_defaulting_to_pandas():
        pd.DataFrame(data).__finalize__(None)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___copy__(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    modin_df_copy, pandas_df_copy = modin_df.__copy__(), pandas_df.__copy__()
    df_equals(modin_df_copy, pandas_df_copy)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___deepcopy__(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    modin_df_copy, pandas_df_copy = (
        modin_df.__deepcopy__(),
        pandas_df.__deepcopy__(),
    )
    df_equals(modin_df_copy, pandas_df_copy)


def test___repr__():
    frame_data = random_state.randint(RAND_LOW, RAND_HIGH, size=(1000, 100))
    pandas_df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)
    assert repr(pandas_df) == repr(modin_df)

    frame_data = random_state.randint(RAND_LOW, RAND_HIGH, size=(1000, 99))
    pandas_df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)
    assert repr(pandas_df) == repr(modin_df)

    frame_data = random_state.randint(RAND_LOW, RAND_HIGH, size=(1000, 101))
    pandas_df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)
    assert repr(pandas_df) == repr(modin_df)

    frame_data = random_state.randint(RAND_LOW, RAND_HIGH, size=(1000, 102))
    pandas_df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)
    assert repr(pandas_df) == repr(modin_df)

    # ___repr___ method has a different code path depending on
    # whether the number of rows is >60; and a different code path
    # depending on the number of columns is >20.
    # Previous test cases already check the case when cols>20
    # and rows>60. The cases that follow exercise the other three
    # combinations.
    # rows <= 60, cols > 20
    frame_data = random_state.randint(RAND_LOW, RAND_HIGH, size=(10, 100))
    pandas_df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)

    assert repr(pandas_df) == repr(modin_df)

    # rows <= 60, cols <= 20
    frame_data = random_state.randint(RAND_LOW, RAND_HIGH, size=(10, 10))
    pandas_df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)

    assert repr(pandas_df) == repr(modin_df)

    # rows > 60, cols <= 20
    frame_data = random_state.randint(RAND_LOW, RAND_HIGH, size=(100, 10))
    pandas_df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)

    assert repr(pandas_df) == repr(modin_df)

    # Empty
    pandas_df = pandas.DataFrame(columns=["col{}".format(i) for i in range(100)])
    modin_df = pd.DataFrame(columns=["col{}".format(i) for i in range(100)])

    assert repr(pandas_df) == repr(modin_df)

    # From Issue #1705
    string_data = """"time","device_id","lat","lng","accuracy","activity_1","activity_1_conf","activity_2","activity_2_conf","activity_3","activity_3_conf"
"2016-08-26 09:00:00.206",2,60.186805,24.821049,33.6080017089844,"STILL",75,"IN_VEHICLE",5,"ON_BICYCLE",5
"2016-08-26 09:00:05.428",5,60.192928,24.767222,5,"WALKING",62,"ON_BICYCLE",29,"RUNNING",6
"2016-08-26 09:00:05.818",1,60.166382,24.700443,3,"WALKING",75,"IN_VEHICLE",5,"ON_BICYCLE",5
"2016-08-26 09:00:15.816",1,60.166254,24.700671,3,"WALKING",75,"IN_VEHICLE",5,"ON_BICYCLE",5
"2016-08-26 09:00:16.413",5,60.193055,24.767427,5,"WALKING",85,"ON_BICYCLE",15,"UNKNOWN",0
"2016-08-26 09:00:20.578",3,60.152996,24.745216,3.90000009536743,"STILL",69,"IN_VEHICLE",31,"UNKNOWN",0"""
    pandas_df = pandas.read_csv(io.StringIO(string_data))
    with warns_that_defaulting_to_pandas():
        modin_df = pd.read_csv(io.StringIO(string_data))
    assert repr(pandas_df) == repr(modin_df)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_inplace_series_ops(data):
    pandas_df = pandas.DataFrame(data)
    modin_df = pd.DataFrame(data)

    if len(modin_df.columns) > len(pandas_df.columns):
        col0 = modin_df.columns[0]
        col1 = modin_df.columns[1]
        pandas_df[col1].dropna(inplace=True)
        modin_df[col1].dropna(inplace=True)
        df_equals(modin_df, pandas_df)

        pandas_df[col0].fillna(0, inplace=True)
        modin_df[col0].fillna(0, inplace=True)
        df_equals(modin_df, pandas_df)


def test___setattr__():
    pandas_df = pandas.DataFrame([1, 2, 3])
    modin_df = pd.DataFrame([1, 2, 3])

    pandas_df.new_col = [4, 5, 6]
    modin_df.new_col = [4, 5, 6]

    df_equals(modin_df, pandas_df)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_isin(data):
    pandas_df = pandas.DataFrame(data)
    modin_df = pd.DataFrame(data)

    val = [1, 2, 3, 4]
    pandas_result = pandas_df.isin(val)
    modin_result = modin_df.isin(val)

    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_constructor(data):
    pandas_df = pandas.DataFrame(data)
    modin_df = pd.DataFrame(data)
    df_equals(pandas_df, modin_df)

    pandas_df = pandas.DataFrame({k: pandas.Series(v) for k, v in data.items()})
    modin_df = pd.DataFrame({k: pd.Series(v) for k, v in data.items()})
    df_equals(pandas_df, modin_df)


@pytest.mark.parametrize(
    "data",
    [
        np.arange(1, 10000, dtype=np.float32),
        [
            pd.Series([1, 2, 3], dtype="int32"),
            pandas.Series([4, 5, 6], dtype="int64"),
            np.array([7, 8, 9], dtype=np.float32),
        ],
        pandas.Categorical([1, 2, 3, 4, 5]),
    ],
)
def test_constructor_dtypes(data):
    md_df, pd_df = create_test_dfs(data)
    df_equals(md_df, pd_df)


def test_constructor_columns_and_index():
    modin_df = pd.DataFrame(
        [[1, 1, 10], [2, 4, 20], [3, 7, 30]],
        index=[1, 2, 3],
        columns=["id", "max_speed", "health"],
    )
    pandas_df = pandas.DataFrame(
        [[1, 1, 10], [2, 4, 20], [3, 7, 30]],
        index=[1, 2, 3],
        columns=["id", "max_speed", "health"],
    )
    df_equals(modin_df, pandas_df)
    df_equals(pd.DataFrame(modin_df), pandas.DataFrame(pandas_df))
    df_equals(
        pd.DataFrame(modin_df, columns=["max_speed", "health"]),
        pandas.DataFrame(pandas_df, columns=["max_speed", "health"]),
    )
    df_equals(
        pd.DataFrame(modin_df, index=[1, 2]),
        pandas.DataFrame(pandas_df, index=[1, 2]),
    )
    df_equals(
        pd.DataFrame(modin_df, index=[1, 2], columns=["health"]),
        pandas.DataFrame(pandas_df, index=[1, 2], columns=["health"]),
    )
    df_equals(
        pd.DataFrame(modin_df.iloc[:, 0], index=[1, 2, 3]),
        pandas.DataFrame(pandas_df.iloc[:, 0], index=[1, 2, 3]),
    )
    df_equals(
        pd.DataFrame(modin_df.iloc[:, 0], columns=["NO_EXIST"]),
        pandas.DataFrame(pandas_df.iloc[:, 0], columns=["NO_EXIST"]),
    )
    with pytest.raises(NotImplementedError):
        pd.DataFrame(modin_df, index=[1, 2, 99999])
    with pytest.raises(NotImplementedError):
        pd.DataFrame(modin_df, columns=["NO_EXIST"])
