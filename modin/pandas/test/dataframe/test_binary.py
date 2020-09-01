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
    random_state,
    RAND_LOW,
    RAND_HIGH,
    df_equals,
    test_data_values,
    test_data_keys,
    eval_general,
)

pd.DEFAULT_NPARTITIONS = 4

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")


def inter_df_math_helper(modin_df, pandas_df, op):
    # Test dataframe to dataframe
    try:
        pandas_result = getattr(pandas_df, op)(pandas_df)
    except Exception as e:
        with pytest.raises(type(e)):
            getattr(modin_df, op)(modin_df)
    else:
        modin_result = getattr(modin_df, op)(modin_df)
        df_equals(modin_result, pandas_result)

    # Test dataframe to int
    try:
        pandas_result = getattr(pandas_df, op)(4)
    except Exception as e:
        with pytest.raises(type(e)):
            getattr(modin_df, op)(4)
    else:
        modin_result = getattr(modin_df, op)(4)
        df_equals(modin_result, pandas_result)

    # Test dataframe to float
    try:
        pandas_result = getattr(pandas_df, op)(4.0)
    except Exception as e:
        with pytest.raises(type(e)):
            getattr(modin_df, op)(4.0)
    else:
        modin_result = getattr(modin_df, op)(4.0)
        df_equals(modin_result, pandas_result)

    # Test transposed dataframes to float
    try:
        pandas_result = getattr(pandas_df.T, op)(4.0)
    except Exception as e:
        with pytest.raises(type(e)):
            getattr(modin_df.T, op)(4.0)
    else:
        modin_result = getattr(modin_df.T, op)(4.0)
        df_equals(modin_result, pandas_result)

    frame_data = {
        "{}_other".format(modin_df.columns[0]): [0, 2],
        modin_df.columns[0]: [0, 19],
        modin_df.columns[1]: [1, 1],
    }
    modin_df2 = pd.DataFrame(frame_data)
    pandas_df2 = pandas.DataFrame(frame_data)

    # Test dataframe to different dataframe shape
    try:
        pandas_result = getattr(pandas_df, op)(pandas_df2)
    except Exception as e:
        with pytest.raises(type(e)):
            getattr(modin_df, op)(modin_df2)
    else:
        modin_result = getattr(modin_df, op)(modin_df2)
        df_equals(modin_result, pandas_result)

    # Test dataframe fill value
    try:
        pandas_result = getattr(pandas_df, op)(pandas_df2, fill_value=0)
    except Exception as e:
        with pytest.raises(type(e)):
            getattr(modin_df, op)(modin_df2, fill_value=0)
    else:
        modin_result = getattr(modin_df, op)(modin_df2, fill_value=0)
        df_equals(modin_result, pandas_result)

    # Test dataframe to list
    list_test = random_state.randint(RAND_LOW, RAND_HIGH, size=(modin_df.shape[1]))
    try:
        pandas_result = getattr(pandas_df, op)(list_test, axis=1)
    except Exception as e:
        with pytest.raises(type(e)):
            getattr(modin_df, op)(list_test, axis=1)
    else:
        modin_result = getattr(modin_df, op)(list_test, axis=1)
        df_equals(modin_result, pandas_result)

    # Test dataframe to series axis=0
    series_test_modin = modin_df[modin_df.columns[0]]
    series_test_pandas = pandas_df[pandas_df.columns[0]]
    try:
        pandas_result = getattr(pandas_df, op)(series_test_pandas, axis=0)
    except Exception as e:
        with pytest.raises(type(e)):
            getattr(modin_df, op)(series_test_modin, axis=0)
    else:
        modin_result = getattr(modin_df, op)(series_test_modin, axis=0)
        df_equals(modin_result, pandas_result)

    # Test dataframe to series axis=1
    series_test_modin = modin_df.iloc[0]
    series_test_pandas = pandas_df.iloc[0]
    try:
        pandas_result = getattr(pandas_df, op)(series_test_pandas, axis=1)
    except Exception as e:
        with pytest.raises(type(e)):
            getattr(modin_df, op)(series_test_modin, axis=1)
    else:
        modin_result = getattr(modin_df, op)(series_test_modin, axis=1)
        df_equals(modin_result, pandas_result)

    # Test dataframe to list axis=1
    series_test_modin = series_test_pandas = list(pandas_df.iloc[0])
    try:
        pandas_result = getattr(pandas_df, op)(series_test_pandas, axis=1)
    except Exception as e:
        with pytest.raises(type(e)):
            getattr(modin_df, op)(series_test_modin, axis=1)
    else:
        modin_result = getattr(modin_df, op)(series_test_modin, axis=1)
        df_equals(modin_result, pandas_result)

    # Test dataframe to list axis=0
    series_test_modin = series_test_pandas = list(pandas_df[pandas_df.columns[0]])
    try:
        pandas_result = getattr(pandas_df, op)(series_test_pandas, axis=0)
    except Exception as e:
        with pytest.raises(type(e)):
            getattr(modin_df, op)(series_test_modin, axis=0)
    else:
        modin_result = getattr(modin_df, op)(series_test_modin, axis=0)
        df_equals(modin_result, pandas_result)

    # Test dataframe to series missing values
    series_test_modin = modin_df.iloc[0, :-2]
    series_test_pandas = pandas_df.iloc[0, :-2]
    try:
        pandas_result = getattr(pandas_df, op)(series_test_pandas, axis=1)
    except Exception as e:
        with pytest.raises(type(e)):
            getattr(modin_df, op)(series_test_modin, axis=1)
    else:
        modin_result = getattr(modin_df, op)(series_test_modin, axis=1)
        df_equals(modin_result, pandas_result)

    # Test dataframe to series with different index
    series_test_modin = modin_df[modin_df.columns[0]].reset_index(drop=True)
    series_test_pandas = pandas_df[pandas_df.columns[0]].reset_index(drop=True)
    try:
        pandas_result = getattr(pandas_df, op)(series_test_pandas, axis=0)
    except Exception as e:
        with pytest.raises(type(e)):
            getattr(modin_df, op)(series_test_modin, axis=0)
    else:
        modin_result = getattr(modin_df, op)(series_test_modin, axis=0)
        df_equals(modin_result, pandas_result)

    # Level test
    new_idx = pandas.MultiIndex.from_tuples(
        [(i // 4, i // 2, i) for i in modin_df.index]
    )
    modin_df_multi_level = modin_df.copy()
    modin_df_multi_level.index = new_idx
    # Defaults to pandas
    with pytest.warns(UserWarning):
        # Operation against self for sanity check
        getattr(modin_df_multi_level, op)(modin_df_multi_level, axis=0, level=1)


@pytest.mark.parametrize(
    "function",
    [
        "add",
        "div",
        "divide",
        "floordiv",
        "mod",
        "mul",
        "multiply",
        "pow",
        "sub",
        "subtract",
        "truediv",
        "__div__",
        "__add__",
        "__radd__",
        "__mul__",
        "__rmul__",
        "__pow__",
        "__rpow__",
        "__sub__",
        "__floordiv__",
        "__rfloordiv__",
        "__truediv__",
        "__rtruediv__",
        "__mod__",
        "__rmod__",
        "__rdiv__",
    ],
)
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_math_functions(data, function):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    inter_df_math_helper(modin_df, pandas_df, function)


@pytest.mark.parametrize("other", ["as_left", 4, 4.0, "a"])
@pytest.mark.parametrize("op", ["eq", "ge", "gt", "le", "lt", "ne"])
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_comparison(data, op, other):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    eval_general(
        modin_df,
        pandas_df,
        operation=lambda df, **kwargs: getattr(df, op)(
            df if other == "as_left" else other
        ),
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


# Test dataframe right operations
def inter_df_math_right_ops_helper(modin_df, pandas_df, op):
    try:
        pandas_result = getattr(pandas_df, op)(4)
    except Exception as e:
        with pytest.raises(type(e)):
            getattr(modin_df, op)(4)
    else:
        modin_result = getattr(modin_df, op)(4)
        df_equals(modin_result, pandas_result)

    try:
        pandas_result = getattr(pandas_df, op)(4.0)
    except Exception as e:
        with pytest.raises(type(e)):
            getattr(modin_df, op)(4.0)
    else:
        modin_result = getattr(modin_df, op)(4.0)
        df_equals(modin_result, pandas_result)

    new_idx = pandas.MultiIndex.from_tuples(
        [(i // 4, i // 2, i) for i in modin_df.index]
    )
    modin_df_multi_level = modin_df.copy()
    modin_df_multi_level.index = new_idx

    # Defaults to pandas
    with pytest.warns(UserWarning):
        # Operation against self for sanity check
        getattr(modin_df_multi_level, op)(modin_df_multi_level, axis=0, level=1)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_radd(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    inter_df_math_right_ops_helper(modin_df, pandas_df, "radd")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_rdiv(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    inter_df_math_right_ops_helper(modin_df, pandas_df, "rdiv")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_rfloordiv(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    inter_df_math_right_ops_helper(modin_df, pandas_df, "rfloordiv")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_rmod(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    inter_df_math_right_ops_helper(modin_df, pandas_df, "rmod")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_rmul(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    inter_df_math_right_ops_helper(modin_df, pandas_df, "rmul")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_rpow(request, data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    inter_df_math_right_ops_helper(modin_df, pandas_df, "rpow")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_rsub(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    inter_df_math_right_ops_helper(modin_df, pandas_df, "rsub")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_rtruediv(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    inter_df_math_right_ops_helper(modin_df, pandas_df, "rtruediv")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___rsub__(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    inter_df_math_right_ops_helper(modin_df, pandas_df, "__rsub__")


# END test dataframe right operations


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
