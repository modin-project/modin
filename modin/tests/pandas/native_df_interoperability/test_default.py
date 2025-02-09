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

import matplotlib
import numpy as np
import pandas
import pytest
from numpy.testing import assert_array_equal

import modin.pandas as pd
from modin.config import NPartitions
from modin.pandas.io import to_pandas
from modin.tests.pandas.native_df_interoperability.utils import (
    create_test_df_in_defined_mode,
    create_test_series_in_defined_mode,
    eval_general_interop,
)
from modin.tests.pandas.utils import (
    default_to_pandas_ignore_string,
    df_equals,
    test_data,
    test_data_diff_dtype,
    test_data_keys,
    test_data_large_categorical_dataframe,
    test_data_values,
)
from modin.tests.test_utils import (
    df_or_series_using_native_execution,
    warns_that_defaulting_to_pandas_if,
)

NPartitions.put(4)

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")

# Our configuration in pytest.ini requires that we explicitly catch all
# instances of defaulting to pandas, but some test modules, like this one,
# have too many such instances.
pytestmark = [
    pytest.mark.filterwarnings(default_to_pandas_ignore_string),
    # IGNORE FUTUREWARNINGS MARKS TO CLEANUP OUTPUT
    pytest.mark.filterwarnings(
        "ignore:.*bool is now deprecated and will be removed:FutureWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:first is deprecated and will be removed:FutureWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:last is deprecated and will be removed:FutureWarning"
    ),
]


@pytest.mark.parametrize(
    "op, make_args",
    [
        ("align", lambda df: {"other": df}),
        ("corrwith", lambda df: {"other": df}),
        ("ewm", lambda df: {"com": 0.5}),
        ("from_dict", lambda df: {"data": None}),
        ("from_records", lambda df: {"data": to_pandas(df)}),
        ("hist", lambda df: {"column": "int_col"}),
        ("interpolate", None),
        ("mask", lambda df: {"cond": df != 0}),
        ("pct_change", None),
        ("to_xarray", None),
        ("flags", None),
        ("set_flags", lambda df: {"allows_duplicate_labels": False}),
    ],
)
def test_ops_defaulting_to_pandas(op, make_args, df_mode_pair):
    modin_df1, _ = create_test_df_in_defined_mode(
        test_data_diff_dtype,
        post_fn=lambda df: df.drop(["str_col", "bool_col"], axis=1),
        native=df_mode_pair[0],
    )
    modin_df2, _ = create_test_df_in_defined_mode(
        test_data_diff_dtype,
        post_fn=lambda df: df.drop(["str_col", "bool_col"], axis=1),
        native=df_mode_pair[1],
    )
    with warns_that_defaulting_to_pandas_if(
        not df_or_series_using_native_execution(modin_df1)
    ):
        operation = getattr(modin_df1, op)
        if make_args is not None:
            operation(**make_args(modin_df2))
        else:
            try:
                operation()
            # `except` for non callable attributes
            except TypeError:
                pass


@pytest.mark.parametrize(
    "data",
    test_data_values + [test_data_large_categorical_dataframe],
    ids=test_data_keys + ["categorical_ints"],
)
def test_to_numpy(data):
    modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)
    assert_array_equal(modin_df.values, pandas_df.values)


def test_asfreq(df_mode_pair):
    index = pd.date_range("1/1/2000", periods=4, freq="min")
    series, _ = create_test_series_in_defined_mode(
        [0.0, None, 2.0, 3.0], index=index, native=df_mode_pair[0]
    )
    df, _ = create_test_df_in_defined_mode({"s": series}, native=df_mode_pair[1])
    with warns_that_defaulting_to_pandas_if(
        not df_or_series_using_native_execution(df)
    ):
        # We are only testing that this defaults to pandas, so we will just check for
        # the warning
        df.asfreq(freq="30S")


def test_assign(df_mode_pair):
    data = test_data_values[0]

    def assign_one_column(df1, df2):
        df1.assign(new_column=pd.Series(df2.iloc[:, 0]))

    eval_general_interop(data, None, assign_one_column, df_mode_pair)

    def assign_multiple_columns(df1, df2):
        df1.assign(
            new_column=pd.Series(df2.iloc[:, 0]), new_column2=pd.Series(df2.iloc[:, 1])
        )

    eval_general_interop(data, None, assign_multiple_columns, df_mode_pair)


def test_combine_first(df_mode_pair):
    data1 = {"A": [None, 0], "B": [None, 4]}
    modin_df1, pandas_df1 = create_test_df_in_defined_mode(
        data1, native=df_mode_pair[0]
    )
    data2 = {"A": [1, 1], "B": [3, 3]}
    modin_df2, pandas_df2 = create_test_df_in_defined_mode(
        data2, native=df_mode_pair[1]
    )

    df_equals(
        modin_df1.combine_first(modin_df2),
        pandas_df1.combine_first(pandas_df2),
        # https://github.com/modin-project/modin/issues/5959
        check_dtypes=False,
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_dot(data, df_mode_pair):

    modin_df, pandas_df = create_test_df_in_defined_mode(data, native=df_mode_pair[0])
    col_len = len(modin_df.columns)

    # Test series input
    modin_series, pandas_series = create_test_series_in_defined_mode(
        np.arange(col_len),
        index=pandas_df.columns,
        native=df_mode_pair[1],
    )
    modin_result = modin_df.dot(modin_series)
    pandas_result = pandas_df.dot(pandas_series)
    df_equals(modin_result, pandas_result)

    def dot_func(df1, df2):
        return df1.dot(df2.T)

    # modin_result = modin_df.dot(modin_df.T)
    # pandas_result = pandas_df.dot(pandas_df.T)
    # df_equals(modin_result, pandas_result)
    # Test dataframe input
    eval_general_interop(data, None, dot_func, df_mode_pair)

    # Test when input series index doesn't line up with columns
    with pytest.raises(ValueError):
        modin_series_without_index, _ = create_test_series_in_defined_mode(
            np.arange(col_len), native=df_mode_pair[1]
        )
        modin_df.dot(modin_series_without_index)

    # Test case when left dataframe has size (n x 1)
    # and right dataframe has size (1 x n)
    eval_general_interop(pandas_series, None, dot_func, df_mode_pair)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_matmul(data, df_mode_pair):
    modin_df, pandas_df = create_test_df_in_defined_mode(data, native=df_mode_pair[0])
    col_len = len(modin_df.columns)

    # Test list input
    arr = np.arange(col_len)
    modin_result = modin_df @ arr
    pandas_result = pandas_df @ arr
    df_equals(modin_result, pandas_result)

    # Test bad dimensions
    with pytest.raises(ValueError):
        modin_df @ np.arange(col_len + 10)

    # Test series input
    modin_series, pandas_series = create_test_series_in_defined_mode(
        np.arange(col_len),
        index=pandas_df.columns,
        native=df_mode_pair[1],
    )
    modin_result = modin_df @ modin_series
    pandas_result = pandas_df @ pandas_series
    df_equals(modin_result, pandas_result)

    # Test dataframe input
    def matmul_func(df1, df2):
        return df1 @ df2.T

    # Test dataframe input
    eval_general_interop(data, None, matmul_func, df_mode_pair)

    # Test when input series index doesn't line up with columns
    with pytest.raises(ValueError):
        modin_series_without_index, _ = create_test_series_in_defined_mode(
            np.arange(col_len), native=df_mode_pair[1]
        )
        modin_df @ modin_series_without_index


@pytest.mark.parametrize("data", [test_data["int_data"]], ids=["int_data"])
@pytest.mark.parametrize(
    "index",
    [
        pytest.param(lambda _, df: df.columns[0], id="single_index_col"),
        pytest.param(
            lambda _, df: [*df.columns[0:2], *df.columns[-7:-4]],
            id="multiple_index_cols",
        ),
        pytest.param(None, id="default_index"),
    ],
)
@pytest.mark.parametrize(
    "columns",
    [
        pytest.param(lambda _, df: df.columns[len(df.columns) // 2], id="single_col"),
        pytest.param(
            lambda _, df: [
                *df.columns[(len(df.columns) // 2) : (len(df.columns) // 2 + 4)],
                df.columns[-7],
            ],
            id="multiple_cols",
        ),
        pytest.param(None, id="default_columns"),
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        pytest.param(lambda _, df: df.columns[-1], id="single_value_col"),
        pytest.param(lambda _, df: df.columns[-4:-1], id="multiple_value_cols"),
    ],
)
@pytest.mark.parametrize(
    "aggfunc",
    [
        pytest.param(lambda df, _: np.mean(df), id="callable_tree_reduce_func"),
        pytest.param("mean", id="tree_reduce_func"),
        pytest.param("nunique", id="full_axis_func"),
    ],
)
def test_pivot_table_data(data, index, columns, values, aggfunc, request, df_mode_pair):
    if (
        "callable_tree_reduce_func-single_value_col-multiple_cols-multiple_index_cols"
        in request.node.callspec.id
        or "callable_tree_reduce_func-multiple_value_cols-multiple_cols-multiple_index_cols"
        in request.node.callspec.id
        or "tree_reduce_func-single_value_col-multiple_cols-multiple_index_cols"
        in request.node.callspec.id
        or "tree_reduce_func-multiple_value_cols-multiple_cols-multiple_index_cols"
        in request.node.callspec.id
        or "full_axis_func-single_value_col-multiple_cols-multiple_index_cols"
        in request.node.callspec.id
        or "full_axis_func-multiple_value_cols-multiple_cols-multiple_index_cols"
        in request.node.callspec.id
    ):
        pytest.xfail(reason="https://github.com/modin-project/modin/issues/7011")

    expected_exception = None
    if "default_columns-default_index" in request.node.callspec.id:
        expected_exception = ValueError("No group keys passed!")
    elif (
        "callable_tree_reduce_func" in request.node.callspec.id
        and "int_data" in request.node.callspec.id
    ):
        expected_exception = TypeError("'numpy.float64' object is not callable")

    eval_general_interop(
        data,
        None,
        operation=lambda df, _, *args, **kwargs: df.pivot_table(
            *args, **kwargs
        ).sort_index(axis=int(index is not None)),
        df_mode_pair=df_mode_pair,
        index=index,
        columns=columns,
        values=values,
        aggfunc=aggfunc,
        expected_exception=expected_exception,
    )
