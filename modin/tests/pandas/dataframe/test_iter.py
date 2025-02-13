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

import io
import warnings

import matplotlib
import numpy as np
import pandas
import pytest

import modin.pandas as pd
from modin.config import NPartitions
from modin.pandas.utils import SET_DATAFRAME_ATTRIBUTE_WARNING
from modin.tests.pandas.utils import (
    RAND_HIGH,
    RAND_LOW,
    create_test_dfs,
    df_equals,
    eval_general,
    random_state,
    test_data,
    test_data_keys,
    test_data_values,
)
from modin.tests.test_utils import (
    current_execution_is_native,
    warns_that_defaulting_to_pandas,
    warns_that_defaulting_to_pandas_if,
)

NPartitions.put(4)

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")


@pytest.mark.parametrize("method", ["items", "iterrows"])
def test_items_iterrows(method):
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


@pytest.mark.parametrize("expand_frame_repr", [False, True])
@pytest.mark.parametrize(
    "max_rows_columns",
    [(5, 5), (10, 10), (50, 50), (51, 51), (52, 52), (75, 75), (None, None)],
)
@pytest.mark.parametrize("frame_size", [101, 102])
def test_display_options_for___repr__(max_rows_columns, expand_frame_repr, frame_size):
    frame_data = random_state.randint(
        RAND_LOW, RAND_HIGH, size=(frame_size, frame_size)
    )
    pandas_df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)

    context_arg = [
        "display.max_rows",
        max_rows_columns[0],
        "display.max_columns",
        max_rows_columns[1],
        "display.expand_frame_repr",
        expand_frame_repr,
    ]
    with pd.option_context(*context_arg):
        modin_df_repr = repr(modin_df)
    with pandas.option_context(*context_arg):
        pandas_df_repr = repr(pandas_df)
    assert modin_df_repr == pandas_df_repr


def test___finalize__():
    data = test_data_values[0]
    # NOTE: __finalize__() defaults to pandas at the API layer.
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
    with warns_that_defaulting_to_pandas_if(not current_execution_is_native()):
        modin_df = pd.read_csv(io.StringIO(string_data))
    assert repr(pandas_df) == repr(modin_df)


def test___repr__does_not_raise_attribute_column_warning():
    # See https://github.com/modin-project/modin/issues/5380
    df = pd.DataFrame([1])
    with warnings.catch_warnings():
        warnings.filterwarnings(action="error", message=SET_DATAFRAME_ATTRIBUTE_WARNING)
        repr(df)


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


# Note: Tests setting an attribute that is not an existing column label
def test___setattr__not_column():
    pandas_df = pandas.DataFrame([1, 2, 3])
    modin_df = pd.DataFrame([1, 2, 3])

    pandas_df.new_col = [4, 5, 6]
    modin_df.new_col = [4, 5, 6]

    df_equals(modin_df, pandas_df)

    # While `new_col` is not a column of the dataframe,
    # it should be accessible with __getattr__.
    assert modin_df.new_col == pandas_df.new_col


def test___setattr__mutating_column():
    # Use case from issue #4577
    pandas_df = pandas.DataFrame([[1]], columns=["col0"])
    modin_df = pd.DataFrame([[1]], columns=["col0"])

    # Replacing a column with a list should mutate the column in place.
    pandas_df.col0 = [3]
    modin_df.col0 = [3]

    df_equals(modin_df, pandas_df)
    # Check that the col0 attribute reflects the value update.
    df_equals(modin_df.col0, pandas_df.col0)

    pandas_df.col0 = pandas.Series([5])
    modin_df.col0 = pd.Series([5])

    # Check that the col0 attribute reflects this update
    df_equals(modin_df, pandas_df)

    pandas_df.loc[0, "col0"] = 4
    modin_df.loc[0, "col0"] = 4

    # Check that the col0 attribute reflects update via loc
    df_equals(modin_df, pandas_df)
    assert modin_df.col0.equals(modin_df["col0"])

    # Check that attempting to add a new col via attributes raises warning
    # and adds the provided list as a new attribute and not a column.
    with pytest.warns(
        UserWarning,
        match=SET_DATAFRAME_ATTRIBUTE_WARNING,
    ):
        modin_df.col1 = [4]

    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="error",
            message=SET_DATAFRAME_ATTRIBUTE_WARNING,
        )
        modin_df.col1 = [5]
        modin_df.new_attr = 6
        modin_df.col0 = 7

    assert "new_attr" in dir(
        modin_df
    ), "Modin attribute was not correctly added to the df."
    assert (
        "new_attr" not in modin_df
    ), "New attribute was not correctly added to columns."
    assert modin_df.new_attr == 6, "Modin attribute value was set incorrectly."
    assert isinstance(
        modin_df.col0, pd.Series
    ), "Scalar was not broadcasted properly to an existing column."


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_isin(data):
    pandas_df = pandas.DataFrame(data)
    modin_df = pd.DataFrame(data)

    val = [1, 2, 3, 4]
    pandas_result = pandas_df.isin(val)
    modin_result = modin_df.isin(val)

    df_equals(modin_result, pandas_result)


def test_isin_with_modin_objects():
    modin_df1, pandas_df1 = create_test_dfs({"a": [1, 2], "b": [3, 4]})
    modin_series, pandas_series = pd.Series([1, 4, 5, 6]), pandas.Series([1, 4, 5, 6])

    eval_general(
        (modin_df1, modin_series),
        (pandas_df1, pandas_series),
        lambda srs: srs[0].isin(srs[1]),
    )

    modin_df2 = modin_series.to_frame("a")
    pandas_df2 = pandas_series.to_frame("a")

    eval_general(
        (modin_df1, modin_df2),
        (pandas_df1, pandas_df2),
        lambda srs: srs[0].isin(srs[1]),
    )

    # Check case when indices are not matching
    modin_df1, pandas_df1 = create_test_dfs({"a": [1, 2], "b": [3, 4]}, index=[10, 11])

    eval_general(
        (modin_df1, modin_series),
        (pandas_df1, pandas_series),
        lambda srs: srs[0].isin(srs[1]),
    )
    eval_general(
        (modin_df1, modin_df2),
        (pandas_df1, pandas_df2),
        lambda srs: srs[0].isin(srs[1]),
    )
