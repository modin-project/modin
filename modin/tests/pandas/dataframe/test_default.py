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
import pandas._libs.lib as lib
import pytest
from numpy.testing import assert_array_equal

import modin.pandas as pd
from modin.config import Engine, NPartitions, StorageFormat
from modin.pandas.io import to_pandas
from modin.tests.pandas.utils import (
    axis_keys,
    axis_values,
    create_test_dfs,
    default_to_pandas_ignore_string,
    df_equals,
    eval_general,
    generate_multiindex,
    modin_df_almost_equals_pandas,
    name_contains,
    numeric_dfs,
    test_data,
    test_data_diff_dtype,
    test_data_keys,
    test_data_large_categorical_dataframe,
    test_data_resample,
    test_data_values,
)
from modin.tests.test_utils import (
    current_execution_is_native,
    df_or_series_using_native_execution,
    warns_that_defaulting_to_pandas_if,
)
from modin.utils import get_current_execution

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
def test_ops_defaulting_to_pandas(op, make_args):
    modin_df = pd.DataFrame(test_data_diff_dtype).drop(["str_col", "bool_col"], axis=1)
    with warns_that_defaulting_to_pandas_if(
        not df_or_series_using_native_execution(modin_df)
    ):
        operation = getattr(modin_df, op)
        if make_args is not None:
            operation(**make_args(modin_df))
        else:
            try:
                operation()
            # `except` for non callable attributes
            except TypeError:
                pass


def test_style():
    data = test_data_values[0]
    with warns_that_defaulting_to_pandas_if(not current_execution_is_native()):
        pd.DataFrame(data).style


def test_to_timestamp():
    idx = pd.date_range("1/1/2012", periods=5, freq="M")
    df = pd.DataFrame(np.random.randint(0, 100, size=(len(idx), 4)), index=idx)

    with warns_that_defaulting_to_pandas_if(
        not df_or_series_using_native_execution(df)
    ):
        df.to_period().to_timestamp()


@pytest.mark.parametrize(
    "data",
    test_data_values + [test_data_large_categorical_dataframe],
    ids=test_data_keys + ["categorical_ints"],
)
def test_to_numpy(data):
    modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)
    assert_array_equal(modin_df.values, pandas_df.values)


@pytest.mark.skipif(
    StorageFormat.get() != "Pandas",
    reason="NativeQueryCompiler does not contain partitions.",
)
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_partition_to_numpy(data):
    frame = pd.DataFrame(data)
    for partition in frame._query_compiler._modin_frame._partitions.flatten().tolist():
        assert_array_equal(partition.to_pandas().values, partition.to_numpy())


def test_asfreq():
    index = pd.date_range("1/1/2000", periods=4, freq="min")
    series = pd.Series([0.0, None, 2.0, 3.0], index=index)
    df = pd.DataFrame({"s": series})
    with warns_that_defaulting_to_pandas_if(
        not df_or_series_using_native_execution(df)
    ):
        # We are only testing that this defaults to pandas, so we will just check for
        # the warning
        df.asfreq(freq="30S")


def test_assign():
    data = test_data_values[0]
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    modin_result = modin_df.assign(new_column=pd.Series(modin_df.iloc[:, 0]))
    pandas_result = pandas_df.assign(new_column=pandas.Series(pandas_df.iloc[:, 0]))
    df_equals(modin_result, pandas_result)
    modin_result = modin_df.assign(
        new_column=pd.Series(modin_df.iloc[:, 0]),
        new_column2=pd.Series(modin_df.iloc[:, 1]),
    )
    pandas_result = pandas_df.assign(
        new_column=pandas.Series(pandas_df.iloc[:, 0]),
        new_column2=pandas.Series(pandas_df.iloc[:, 1]),
    )
    df_equals(modin_result, pandas_result)


def test_at_time():
    i = pd.date_range("2008-01-01", periods=1000, freq="12H")
    modin_df = pd.DataFrame({"A": list(range(1000)), "B": list(range(1000))}, index=i)
    pandas_df = pandas.DataFrame(
        {"A": list(range(1000)), "B": list(range(1000))}, index=i
    )
    df_equals(modin_df.at_time("12:00"), pandas_df.at_time("12:00"))
    df_equals(modin_df.at_time("3:00"), pandas_df.at_time("3:00"))
    df_equals(modin_df.T.at_time("12:00", axis=1), pandas_df.T.at_time("12:00", axis=1))


def test_between_time():
    i = pd.date_range("2008-01-01", periods=1000, freq="12H")
    modin_df = pd.DataFrame({"A": list(range(1000)), "B": list(range(1000))}, index=i)
    pandas_df = pandas.DataFrame(
        {"A": list(range(1000)), "B": list(range(1000))}, index=i
    )
    df_equals(
        modin_df.between_time("12:00", "17:00"),
        pandas_df.between_time("12:00", "17:00"),
    )
    df_equals(
        modin_df.between_time("3:00", "4:00"),
        pandas_df.between_time("3:00", "4:00"),
    )
    df_equals(
        modin_df.T.between_time("12:00", "17:00", axis=1),
        pandas_df.T.between_time("12:00", "17:00", axis=1),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_bfill(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    df_equals(modin_df.bfill(), pandas_df.bfill())


@pytest.mark.parametrize("limit_area", [None, "inside", "outside"])
@pytest.mark.parametrize("method", ["ffill", "bfill"])
def test_ffill_bfill_limit_area(method, limit_area):
    modin_df, pandas_df = create_test_dfs([1, None, 2, None])
    eval_general(
        modin_df, pandas_df, lambda df: getattr(df, method)(limit_area=limit_area)
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_bool(data):
    modin_df = pd.DataFrame(data)

    with pytest.warns(
        FutureWarning, match="bool is now deprecated and will be removed"
    ):
        with pytest.raises(ValueError):
            modin_df.bool()
            modin_df.__bool__()

    single_bool_pandas_df = pandas.DataFrame([True])
    single_bool_modin_df = pd.DataFrame([True])

    assert single_bool_pandas_df.bool() == single_bool_modin_df.bool()

    with pytest.raises(ValueError):
        # __bool__ always raises this error for DataFrames
        single_bool_modin_df.__bool__()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_boxplot(data):
    modin_df = pd.DataFrame(data)

    assert modin_df.boxplot() == to_pandas(modin_df).boxplot()


def test_combine_first():
    data1 = {"A": [None, 0], "B": [None, 4]}
    modin_df1 = pd.DataFrame(data1)
    pandas_df1 = pandas.DataFrame(data1)
    data2 = {"A": [1, 1], "B": [3, 3]}
    modin_df2 = pd.DataFrame(data2)
    pandas_df2 = pandas.DataFrame(data2)
    df_equals(
        modin_df1.combine_first(modin_df2),
        pandas_df1.combine_first(pandas_df2),
        # https://github.com/modin-project/modin/issues/5959
        check_dtypes=False,
    )


class TestCorr:
    @pytest.mark.parametrize("method", ["pearson", "kendall", "spearman"])
    @pytest.mark.parametrize("backend", [None, "pyarrow"])
    def test_corr(self, method, backend):
        eval_general(
            *create_test_dfs(test_data["int_data"], backend=backend),
            lambda df: df.corr(method=method),
        )
        # Modin result may slightly differ from pandas result
        # due to floating pointing arithmetic.
        eval_general(
            *create_test_dfs(test_data["float_nan_data"], backend=backend),
            lambda df: df.corr(method=method),
            comparator=modin_df_almost_equals_pandas,
        )

    @pytest.mark.parametrize("min_periods", [1, 3, 5, 6])
    def test_corr_min_periods(self, min_periods):
        # only 3 valid values (a valid value is considered a row with no NaNs)
        eval_general(
            *create_test_dfs({"a": [1, 2, 3], "b": [3, 1, 5]}),
            lambda df: df.corr(min_periods=min_periods),
        )

        # only 5 valid values (a valid value is considered a row with no NaNs)
        eval_general(
            *create_test_dfs(
                {"a": [1, 2, 3, 4, 5, np.nan], "b": [1, 2, 1, 4, 5, np.nan]}
            ),
            lambda df: df.corr(min_periods=min_periods),
        )

        # only 4 valid values (a valid value is considered a row with no NaNs)
        eval_general(
            *create_test_dfs(
                {"a": [1, np.nan, 3, 4, 5, 6], "b": [1, 2, 1, 4, 5, np.nan]}
            ),
            lambda df: df.corr(min_periods=min_periods),
        )

        if StorageFormat.get() == "Pandas":
            # only 4 valid values located in different partitions (a valid value is considered a row with no NaNs)
            modin_df, pandas_df = create_test_dfs(
                {"a": [1, np.nan, 3, 4, 5, 6], "b": [1, 2, 1, 4, 5, np.nan]}
            )
            modin_df = pd.concat([modin_df.iloc[:3], modin_df.iloc[3:]])
            assert modin_df._query_compiler._modin_frame._partitions.shape == (2, 1)
            eval_general(
                modin_df, pandas_df, lambda df: df.corr(min_periods=min_periods)
            )

    @pytest.mark.parametrize("numeric_only", [True, False])
    def test_corr_non_numeric(self, numeric_only):
        if not numeric_only:
            pytest.xfail(reason="https://github.com/modin-project/modin/issues/7023")
        eval_general(
            *create_test_dfs({"a": [1, 2, 3], "b": [3, 2, 5], "c": ["a", "b", "c"]}),
            lambda df: df.corr(numeric_only=numeric_only),
        )

    @pytest.mark.skipif(
        StorageFormat.get() != "Pandas",
        reason="doesn't make sense for non-partitioned executions",
    )
    def test_corr_nans_in_different_partitions(self):
        # NaN in the first partition
        modin_df, pandas_df = create_test_dfs(
            {"a": [np.nan, 2, 3, 4, 5, 6], "b": [3, 4, 2, 0, 7, 8]}
        )
        modin_df = pd.concat([modin_df.iloc[:2], modin_df.iloc[2:4], modin_df.iloc[4:]])

        assert modin_df._query_compiler._modin_frame._partitions.shape == (3, 1)
        eval_general(modin_df, pandas_df, lambda df: df.corr())

        # NaN in the last partition
        modin_df, pandas_df = create_test_dfs(
            {"a": [1, 2, 3, 4, 5, np.nan], "b": [3, 4, 2, 0, 7, 8]}
        )
        modin_df = pd.concat([modin_df.iloc[:2], modin_df.iloc[2:4], modin_df.iloc[4:]])

        assert modin_df._query_compiler._modin_frame._partitions.shape == (3, 1)
        eval_general(modin_df, pandas_df, lambda df: df.corr())

        # NaN in two partitions
        modin_df, pandas_df = create_test_dfs(
            {"a": [np.nan, 2, 3, 4, 5, 6], "b": [3, 4, 2, 0, 7, np.nan]}
        )
        modin_df = pd.concat([modin_df.iloc[:2], modin_df.iloc[2:4], modin_df.iloc[4:]])

        assert modin_df._query_compiler._modin_frame._partitions.shape == (3, 1)
        eval_general(modin_df, pandas_df, lambda df: df.corr())

        # NaN in all partitions
        modin_df, pandas_df = create_test_dfs(
            {"a": [np.nan, 2, 3, np.nan, 5, 6], "b": [3, 4, 2, 0, 7, np.nan]}
        )
        modin_df = pd.concat([modin_df.iloc[:2], modin_df.iloc[2:4], modin_df.iloc[4:]])

        assert modin_df._query_compiler._modin_frame._partitions.shape == (3, 1)
        eval_general(modin_df, pandas_df, lambda df: df.corr())


@pytest.mark.parametrize("min_periods", [1, 3, 5], ids=lambda x: f"min_periods={x}")
@pytest.mark.parametrize("ddof", [1, 2, 4], ids=lambda x: f"ddof={x}")
@pytest.mark.parametrize("backend", [None, "pyarrow"])
def test_cov(min_periods, ddof, backend):
    eval_general(
        *create_test_dfs(test_data["int_data"], backend=backend),
        lambda df: df.cov(min_periods=min_periods, ddof=ddof),
        comparator=df_equals,
    )
    # Modin result may slightly differ from pandas result
    # due to floating pointing arithmetic. That's why we use `modin_df_almost_equals_pandas`.
    eval_general(
        *create_test_dfs(test_data["float_nan_data"], backend=backend),
        lambda df: df.cov(min_periods=min_periods),
        comparator=modin_df_almost_equals_pandas,
    )


@pytest.mark.parametrize("numeric_only", [True, False])
def test_cov_numeric_only(numeric_only):
    if not numeric_only:
        pytest.xfail(reason="https://github.com/modin-project/modin/issues/7023")
    eval_general(
        *create_test_dfs({"a": [1, 2, 3], "b": [3, 2, 5], "c": ["a", "b", "c"]}),
        lambda df: df.cov(numeric_only=numeric_only),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_dot(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    col_len = len(modin_df.columns)

    # Test list input
    arr = np.arange(col_len)
    modin_result = modin_df.dot(arr)
    pandas_result = pandas_df.dot(arr)
    df_equals(modin_result, pandas_result)

    # Test bad dimensions
    with pytest.raises(ValueError):
        modin_df.dot(np.arange(col_len + 10))

    # Test series input
    modin_series = pd.Series(np.arange(col_len), index=modin_df.columns)
    pandas_series = pandas.Series(np.arange(col_len), index=pandas_df.columns)
    modin_result = modin_df.dot(modin_series)
    pandas_result = pandas_df.dot(pandas_series)
    df_equals(modin_result, pandas_result)

    # Test dataframe input
    modin_result = modin_df.dot(modin_df.T)
    pandas_result = pandas_df.dot(pandas_df.T)
    df_equals(modin_result, pandas_result)

    # Test when input series index doesn't line up with columns
    with pytest.raises(ValueError):
        modin_df.dot(pd.Series(np.arange(col_len)))

    # Test case when left dataframe has size (n x 1)
    # and right dataframe has size (1 x n)
    modin_df = pd.DataFrame(modin_series)
    pandas_df = pandas.DataFrame(pandas_series)
    modin_result = modin_df.dot(modin_df.T)
    pandas_result = pandas_df.dot(pandas_df.T)
    df_equals(modin_result, pandas_result)

    # Test case when left dataframe has size (1 x 1)
    # and right dataframe has size (1 x n)
    modin_result = pd.DataFrame([1]).dot(modin_df.T)
    pandas_result = pandas.DataFrame([1]).dot(pandas_df.T)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_matmul(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
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
    modin_series = pd.Series(np.arange(col_len), index=modin_df.columns)
    pandas_series = pandas.Series(np.arange(col_len), index=pandas_df.columns)
    modin_result = modin_df @ modin_series
    pandas_result = pandas_df @ pandas_series
    df_equals(modin_result, pandas_result)

    # Test dataframe input
    modin_result = modin_df @ modin_df.T
    pandas_result = pandas_df @ pandas_df.T
    df_equals(modin_result, pandas_result)

    # Test when input series index doesn't line up with columns
    with pytest.raises(ValueError):
        modin_df @ pd.Series(np.arange(col_len))


def test_first():
    i = pd.date_range("2010-04-09", periods=400, freq="2D")
    modin_df = pd.DataFrame({"A": list(range(400)), "B": list(range(400))}, index=i)
    pandas_df = pandas.DataFrame(
        {"A": list(range(400)), "B": list(range(400))}, index=i
    )
    with pytest.warns(FutureWarning, match="first is deprecated and will be removed"):
        modin_result = modin_df.first("3D")
    df_equals(modin_result, pandas_df.first("3D"))
    df_equals(modin_df.first("20D"), pandas_df.first("20D"))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_info_default_param(data):
    with io.StringIO() as first, io.StringIO() as second:
        eval_general(
            pd.DataFrame(data),
            pandas.DataFrame(data),
            verbose=None,
            max_cols=None,
            memory_usage=None,
            operation=lambda df, **kwargs: df.info(**kwargs),
            buf=lambda df: second if isinstance(df, pandas.DataFrame) else first,
        )
        modin_info = first.getvalue().splitlines()
        pandas_info = second.getvalue().splitlines()

        assert modin_info[0] == str(pd.DataFrame)
        assert pandas_info[0] == str(pandas.DataFrame)
        assert modin_info[1:] == pandas_info[1:]


# randint data covers https://github.com/modin-project/modin/issues/5137
@pytest.mark.parametrize(
    "data", [test_data_values[0], np.random.randint(0, 100, (10, 10))]
)
@pytest.mark.parametrize("verbose", [True, False])
@pytest.mark.parametrize("max_cols", [10, 99999999])
@pytest.mark.parametrize("memory_usage", [True, False, "deep"])
@pytest.mark.parametrize("show_counts", [True, False])
def test_info(data, verbose, max_cols, memory_usage, show_counts):
    with io.StringIO() as first, io.StringIO() as second:
        eval_general(
            pd.DataFrame(data),
            pandas.DataFrame(data),
            operation=lambda df, **kwargs: df.info(**kwargs),
            verbose=verbose,
            max_cols=max_cols,
            memory_usage=memory_usage,
            show_counts=show_counts,
            buf=lambda df: second if isinstance(df, pandas.DataFrame) else first,
        )
        modin_info = first.getvalue().splitlines()
        pandas_info = second.getvalue().splitlines()

        assert modin_info[0] == str(pd.DataFrame)
        assert pandas_info[0] == str(pandas.DataFrame)
        assert modin_info[1:] == pandas_info[1:]


@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("skipna", [False, True])
@pytest.mark.parametrize("numeric_only", [False, True])
@pytest.mark.parametrize("method", ["kurtosis", "kurt"])
def test_kurt_kurtosis(axis, skipna, numeric_only, method):
    data = test_data["float_nan_data"]

    eval_general(
        *create_test_dfs(data),
        lambda df: getattr(df, method)(
            axis=axis, skipna=skipna, numeric_only=numeric_only
        ),
    )


def test_last():
    modin_index = pd.date_range("2010-04-09", periods=400, freq="2D")
    pandas_index = pandas.date_range("2010-04-09", periods=400, freq="2D")
    modin_df = pd.DataFrame(
        {"A": list(range(400)), "B": list(range(400))}, index=modin_index
    )
    pandas_df = pandas.DataFrame(
        {"A": list(range(400)), "B": list(range(400))}, index=pandas_index
    )
    with pytest.warns(FutureWarning, match="last is deprecated and will be removed"):
        modin_result = modin_df.last("3D")
    df_equals(modin_result, pandas_df.last("3D"))
    df_equals(modin_df.last("20D"), pandas_df.last("20D"))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize(
    "id_vars", [lambda df: df.columns[0], lambda df: df.columns[:4], None]
)
@pytest.mark.parametrize(
    "value_vars", [lambda df: df.columns[-1], lambda df: df.columns[-4:], None]
)
def test_melt(data, id_vars, value_vars):
    def melt(df, *args, **kwargs):
        return df.melt(*args, **kwargs).sort_values(["variable", "value"])

    eval_general(
        *create_test_dfs(data),
        lambda df, *args, **kwargs: melt(df, *args, **kwargs).reset_index(drop=True),
        id_vars=id_vars,
        value_vars=value_vars,
    )


# Functional test for BUG:7206
def test_melt_duplicate_col_names():
    data = {"data": [[1, 2], [3, 4]], "columns": ["dupe", "dupe"]}

    def melt(df, *args, **kwargs):
        return df.melt(*args, **kwargs).sort_values(["variable", "value"])

    eval_general(
        *create_test_dfs(**data),
        lambda df, *args, **kwargs: melt(df, *args, **kwargs).reset_index(drop=True),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize(
    "index",
    [lambda df: df.columns[0], lambda df: df.columns[:2], lib.no_default],
    ids=["one_column_index", "several_columns_index", "default"],
)
@pytest.mark.parametrize(
    "columns", [lambda df: df.columns[len(df.columns) // 2]], ids=["one_column"]
)
@pytest.mark.parametrize(
    "values",
    [lambda df: df.columns[-1], lambda df: df.columns[-2:], lib.no_default],
    ids=["one_column_values", "several_columns_values", "default"],
)
def test_pivot(data, index, columns, values, request):
    current_execution = get_current_execution()
    if (
        "one_column_values-one_column-default-float_nan_data"
        in request.node.callspec.id
        or "default-one_column-several_columns_index" in request.node.callspec.id
        or "default-one_column-one_column_index" in request.node.callspec.id
        or (
            (current_execution == "BaseOnPython" or current_execution_is_native())
            and index is lib.no_default
        )
    ):
        pytest.xfail(reason="https://github.com/modin-project/modin/issues/7010")

    expected_exception = None
    if index is not lib.no_default:
        expected_exception = ValueError(
            "Index contains duplicate entries, cannot reshape"
        )
    eval_general(
        *create_test_dfs(data),
        lambda df, *args, **kwargs: df.pivot(*args, **kwargs),
        index=index,
        columns=columns,
        values=values,
        expected_exception=expected_exception,
    )


@pytest.mark.parametrize("data", [test_data["int_data"]], ids=["int_data"])
@pytest.mark.parametrize(
    "index",
    [
        pytest.param(lambda df: df.columns[0], id="single_index_col"),
        pytest.param(
            lambda df: [*df.columns[0:2], *df.columns[-7:-4]], id="multiple_index_cols"
        ),
        pytest.param(None, id="default_index"),
    ],
)
@pytest.mark.parametrize(
    "columns",
    [
        pytest.param(lambda df: df.columns[len(df.columns) // 2], id="single_col"),
        pytest.param(
            lambda df: [
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
        pytest.param(lambda df: df.columns[-1], id="single_value_col"),
        pytest.param(lambda df: df.columns[-4:-1], id="multiple_value_cols"),
        pytest.param(None, id="default_values"),
    ],
)
@pytest.mark.parametrize(
    "aggfunc",
    [
        pytest.param(np.mean, id="callable_tree_reduce_func"),
        pytest.param("mean", id="tree_reduce_func"),
        pytest.param("nunique", id="full_axis_func"),
    ],
)
def test_pivot_table_data(data, index, columns, values, aggfunc, request):
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
    md_df, pd_df = create_test_dfs(data)

    # when values is None the output will be huge-dimensional,
    # so reducing dimension of testing data at that case
    if values is None:
        md_df, pd_df = md_df.iloc[:42, :42], pd_df.iloc[:42, :42]

    expected_exception = None
    if "default_columns-default_index" in request.node.callspec.id:
        expected_exception = ValueError("No group keys passed!")
    elif (
        "callable_tree_reduce_func" in request.node.callspec.id
        and "int_data" in request.node.callspec.id
    ):
        expected_exception = TypeError("'numpy.float64' object is not callable")

    eval_general(
        md_df,
        pd_df,
        operation=lambda df, *args, **kwargs: df.pivot_table(
            *args, **kwargs
        ).sort_index(axis=int(index is not None)),
        index=index,
        columns=columns,
        values=values,
        aggfunc=aggfunc,
        expected_exception=expected_exception,
    )


@pytest.mark.parametrize("data", [test_data["int_data"]], ids=["int_data"])
@pytest.mark.parametrize(
    "index",
    [
        pytest.param([], id="no_index_cols"),
        pytest.param(lambda df: df.columns[0], id="single_index_column"),
        pytest.param(
            lambda df: [df.columns[0], df.columns[len(df.columns) // 2 - 1]],
            id="multiple_index_cols",
        ),
    ],
)
@pytest.mark.parametrize(
    "columns",
    [
        pytest.param(lambda df: df.columns[len(df.columns) // 2], id="single_column"),
        pytest.param(
            lambda df: [
                *df.columns[(len(df.columns) // 2) : (len(df.columns) // 2 + 4)],
                df.columns[-7],
            ],
            id="multiple_cols",
        ),
    ],
)
@pytest.mark.parametrize(
    "values",
    [
        pytest.param(lambda df: df.columns[-1], id="single_value"),
        pytest.param(lambda df: df.columns[-4:-1], id="multiple_values"),
    ],
)
@pytest.mark.parametrize(
    "aggfunc",
    [
        pytest.param(["mean", "sum"], id="list_func"),
        pytest.param(
            lambda df: {df.columns[5]: "mean", df.columns[-5]: "sum"}, id="dict_func"
        ),
    ],
)
@pytest.mark.parametrize(
    "margins_name",
    [pytest.param("Custom name", id="str_name")],
)
@pytest.mark.parametrize("fill_value", [None, 0])
@pytest.mark.parametrize("backend", [None, "pyarrow"])
def test_pivot_table_margins(
    data,
    index,
    columns,
    values,
    aggfunc,
    margins_name,
    fill_value,
    backend,
    request,
):
    expected_exception = None
    if "dict_func" in request.node.callspec.id:
        expected_exception = KeyError("Column(s) ['col28', 'col38'] do not exist")
    eval_general(
        *create_test_dfs(data, backend=backend),
        operation=lambda df, *args, **kwargs: df.pivot_table(*args, **kwargs),
        index=index,
        columns=columns,
        values=values,
        aggfunc=aggfunc,
        margins=True,
        margins_name=margins_name,
        fill_value=fill_value,
        expected_exception=expected_exception,
    )


@pytest.mark.parametrize(
    "aggfunc",
    [
        pytest.param("sum", id="MapReduce_func"),
        pytest.param("nunique", id="FullAxis_func"),
    ],
)
@pytest.mark.parametrize("margins", [True, False])
def test_pivot_table_fill_value(aggfunc, margins):
    md_df, pd_df = create_test_dfs(test_data["int_data"])
    eval_general(
        md_df,
        pd_df,
        operation=lambda df, *args, **kwargs: df.pivot_table(*args, **kwargs),
        index=md_df.columns[0],
        columns=md_df.columns[1],
        values=md_df.columns[2],
        aggfunc=aggfunc,
        margins=margins,
        fill_value=10,
    )


@pytest.mark.parametrize("data", [test_data["int_data"]], ids=["int_data"])
def test_pivot_table_dropna(data):
    eval_general(
        *create_test_dfs(data),
        operation=lambda df, *args, **kwargs: df.pivot_table(*args, **kwargs),
        index=lambda df: df.columns[0],
        columns=lambda df: df.columns[1],
        values=lambda df: df.columns[-1],
        dropna=False,
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_plot(request, data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    if name_contains(request.node.name, numeric_dfs):
        # We have to test this way because equality in plots means same object.
        zipped_plot_lines = zip(modin_df.plot().lines, pandas_df.plot().lines)
        for left, right in zipped_plot_lines:
            if isinstance(left.get_xdata(), np.ma.core.MaskedArray) and isinstance(
                right.get_xdata(), np.ma.core.MaskedArray
            ):
                assert all((left.get_xdata() == right.get_xdata()).data)
            else:
                assert np.array_equal(left.get_xdata(), right.get_xdata())
            if isinstance(left.get_ydata(), np.ma.core.MaskedArray) and isinstance(
                right.get_ydata(), np.ma.core.MaskedArray
            ):
                assert all((left.get_ydata() == right.get_ydata()).data)
            else:
                assert np.array_equal(left.get_xdata(), right.get_xdata())


def test_replace():
    modin_df = pd.DataFrame(
        {"A": [0, 1, 2, 3, 4], "B": [5, 6, 7, 8, 9], "C": ["a", "b", "c", "d", "e"]}
    )
    pandas_df = pandas.DataFrame(
        {"A": [0, 1, 2, 3, 4], "B": [5, 6, 7, 8, 9], "C": ["a", "b", "c", "d", "e"]}
    )
    modin_result = modin_df.replace({"A": 0, "B": 5}, 100)
    pandas_result = pandas_df.replace({"A": 0, "B": 5}, 100)
    df_equals(modin_result, pandas_result)

    modin_result = modin_df.replace({"A": {0: 100, 4: 400}})
    pandas_result = pandas_df.replace({"A": {0: 100, 4: 400}})
    df_equals(modin_result, pandas_result)

    modin_df = pd.DataFrame({"A": ["bat", "foo", "bait"], "B": ["abc", "bar", "xyz"]})
    pandas_df = pandas.DataFrame(
        {"A": ["bat", "foo", "bait"], "B": ["abc", "bar", "xyz"]}
    )
    modin_result = modin_df.replace(regex={r"^ba.$": "new", "foo": "xyz"})
    pandas_result = pandas_df.replace(regex={r"^ba.$": "new", "foo": "xyz"})
    df_equals(modin_result, pandas_result)

    modin_result = modin_df.replace(regex=[r"^ba.$", "foo"], value="new")
    pandas_result = pandas_df.replace(regex=[r"^ba.$", "foo"], value="new")
    df_equals(modin_result, pandas_result)

    modin_df.replace(regex=[r"^ba.$", "foo"], value="new", inplace=True)
    pandas_df.replace(regex=[r"^ba.$", "foo"], value="new", inplace=True)
    df_equals(modin_df, pandas_df)


@pytest.mark.parametrize("rule", ["5min", pandas.offsets.Hour()])
@pytest.mark.parametrize("axis", [0])
def test_resampler(rule, axis):
    data, index = (
        test_data_resample["data"],
        test_data_resample["index"],
    )
    modin_resampler = pd.DataFrame(data, index=index).resample(rule, axis=axis)
    pandas_resampler = pandas.DataFrame(data, index=index).resample(rule, axis=axis)

    assert pandas_resampler.indices == modin_resampler.indices
    assert pandas_resampler.groups == modin_resampler.groups

    df_equals(
        modin_resampler.get_group(name=list(modin_resampler.groups)[0]),
        pandas_resampler.get_group(name=list(pandas_resampler.groups)[0]),
    )


@pytest.mark.parametrize("rule", ["5min"])
@pytest.mark.parametrize("axis", ["index", "columns"])
@pytest.mark.parametrize(
    "method",
    [
        *("count", "sum", "std", "sem", "size", "prod", "ohlc", "quantile"),
        *("min", "median", "mean", "max", "last", "first", "nunique", "var"),
        *("interpolate", "asfreq", "nearest", "bfill", "ffill"),
    ],
)
def test_resampler_functions(rule, axis, method):
    data, index = (
        test_data_resample["data"],
        test_data_resample["index"],
    )
    modin_df = pd.DataFrame(data, index=index)
    pandas_df = pandas.DataFrame(data, index=index)
    if axis == "columns":
        columns = pandas.date_range(
            "31/12/2000", periods=len(pandas_df.columns), freq="min"
        )
        modin_df.columns = columns
        pandas_df.columns = columns

    expected_exception = None
    if method in ("interpolate", "asfreq", "nearest", "bfill", "ffill"):
        # It looks like pandas is preparing to completely
        # remove `axis` parameter for `resample` function.
        expected_exception = AssertionError("axis must be 0")

    eval_general(
        modin_df,
        pandas_df,
        lambda df: getattr(df.resample(rule, axis=axis), method)(),
        expected_exception=expected_exception,
    )


@pytest.mark.parametrize("rule", ["5min"])
@pytest.mark.parametrize("axis", ["index", "columns"])
@pytest.mark.parametrize(
    "method_arg",
    [
        ("pipe", lambda x: x.max() - x.min()),
        ("transform", lambda x: (x - x.mean()) / x.std()),
        ("apply", ["sum", "mean", "max"]),
        ("aggregate", ["sum", "mean", "max"]),
    ],
)
def test_resampler_functions_with_arg(rule, axis, method_arg):
    data, index = (
        test_data_resample["data"],
        test_data_resample["index"],
    )
    modin_df = pd.DataFrame(data, index=index)
    pandas_df = pandas.DataFrame(data, index=index)
    if axis == "columns":
        columns = pandas.date_range(
            "31/12/2000", periods=len(pandas_df.columns), freq="min"
        )
        modin_df.columns = columns
        pandas_df.columns = columns

    method, arg = method_arg[0], method_arg[1]

    expected_exception = None
    if method in ("apply", "aggregate"):
        expected_exception = NotImplementedError("axis other than 0 is not supported")

    eval_general(
        modin_df,
        pandas_df,
        lambda df: getattr(df.resample(rule, axis=axis), method)(arg),
        expected_exception=expected_exception,
    )


@pytest.mark.parametrize("rule", ["5min"])
@pytest.mark.parametrize("closed", ["left", "right"])
@pytest.mark.parametrize("label", ["right", "left"])
@pytest.mark.parametrize(
    "on",
    [
        None,
        pytest.param(
            "DateColumn",
            marks=pytest.mark.xfail(
                condition=Engine.get() in ("Ray", "Unidist", "Dask", "Python")
                and StorageFormat.get() != "Base",
                reason="https://github.com/modin-project/modin/issues/6399",
            ),
        ),
    ],
)
@pytest.mark.parametrize("level", [None, 1])
def test_resample_specific(rule, closed, label, on, level):
    data, index = (
        test_data_resample["data"],
        test_data_resample["index"],
    )
    modin_df = pd.DataFrame(data, index=index)
    pandas_df = pandas.DataFrame(data, index=index)

    if on is None and level is not None:
        index = pandas.MultiIndex.from_product(
            [
                ["a", "b", "c", "d"],
                pandas.date_range("31/12/2000", periods=len(pandas_df) // 4, freq="h"),
            ]
        )
        pandas_df.index = index
        modin_df.index = index
    else:
        level = None

    if on is not None:
        pandas_df[on] = pandas.date_range(
            "22/06/1941", periods=len(pandas_df), freq="min"
        )
        modin_df[on] = pandas.date_range(
            "22/06/1941", periods=len(modin_df), freq="min"
        )

    pandas_resampler = pandas_df.resample(
        rule,
        closed=closed,
        label=label,
        on=on,
        level=level,
    )
    modin_resampler = modin_df.resample(
        rule,
        closed=closed,
        label=label,
        on=on,
        level=level,
    )
    df_equals(modin_resampler.var(0), pandas_resampler.var(0))
    if on is None and level is None:
        df_equals(
            modin_resampler.fillna(method="nearest"),
            pandas_resampler.fillna(method="nearest"),
        )


@pytest.mark.parametrize(
    "columns",
    [
        "volume",
        "date",
        ["volume"],
        ("volume",),
        pandas.Series(["volume"]),
        pandas.Index(["volume"]),
        ["volume", "volume", "volume"],
        ["volume", "price", "date"],
    ],
    ids=[
        "column",
        "only_missed_column",
        "list",
        "tuple",
        "series",
        "index",
        "duplicate_column",
        "missed_column",
    ],
)
def test_resample_getitem(columns, request):
    index = pandas.date_range("1/1/2013", periods=9, freq="min")
    data = {
        "price": range(9),
        "volume": range(10, 19),
    }
    expected_exception = None
    if "only_missed_column" in request.node.callspec.id:
        expected_exception = KeyError("Column not found: date")
    elif "missed_column" in request.node.callspec.id:
        expected_exception = KeyError("Columns not found: 'date'")
    eval_general(
        *create_test_dfs(data, index=index),
        lambda df: df.resample("3min")[columns].mean(),
        expected_exception=expected_exception,
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("index", ["default", "ndarray", "has_duplicates"])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("periods", [0, 1, -1, 10, -10, 1000000000, -1000000000])
def test_shift(data, index, axis, periods):
    modin_df, pandas_df = create_test_dfs(data)
    if index == "ndarray":
        data_column_length = len(data[next(iter(data))])
        modin_df.index = pandas_df.index = np.arange(2, data_column_length + 2)
    elif index == "has_duplicates":
        modin_df.index = pandas_df.index = list(modin_df.index[:-3]) + [0, 1, 2]

    df_equals(
        modin_df.shift(periods=periods, axis=axis),
        pandas_df.shift(periods=periods, axis=axis),
    )
    df_equals(
        modin_df.shift(periods=periods, axis=axis, fill_value=777),
        pandas_df.shift(periods=periods, axis=axis, fill_value=777),
    )


@pytest.mark.parametrize("is_multi_idx", [True, False], ids=["idx_multi", "idx_index"])
@pytest.mark.parametrize("is_multi_col", [True, False], ids=["col_multi", "col_index"])
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_stack(data, is_multi_idx, is_multi_col):
    pandas_df = pandas.DataFrame(data)
    modin_df = pd.DataFrame(data)

    if is_multi_idx:
        if len(pandas_df.index) == 256:
            index = pd.MultiIndex.from_product(
                [
                    ["a", "b", "c", "d"],
                    ["x", "y", "z", "last"],
                    ["i", "j", "k", "index"],
                    [1, 2, 3, 4],
                ]
            )
        elif len(pandas_df.index) == 100:
            index = pd.MultiIndex.from_product(
                [
                    ["x", "y", "z", "last"],
                    ["a", "b", "c", "d", "f"],
                    ["i", "j", "k", "l", "index"],
                ]
            )
        else:
            index = pd.MultiIndex.from_tuples(
                [(i, i * 2, i * 3) for i in range(len(pandas_df.index))]
            )
    else:
        index = pandas_df.index

    if is_multi_col:
        if len(pandas_df.columns) == 64:
            columns = pd.MultiIndex.from_product(
                [["A", "B", "C", "D"], ["xx", "yy", "zz", "LAST"], [10, 20, 30, 40]]
            )
        elif len(pandas_df.columns) == 100:
            columns = pd.MultiIndex.from_product(
                [
                    ["xx", "yy", "zz", "LAST"],
                    ["A", "B", "C", "D", "F"],
                    ["I", "J", "K", "L", "INDEX"],
                ]
            )
        else:
            columns = pd.MultiIndex.from_tuples(
                [(i, i * 2, i * 3) for i in range(len(pandas_df.columns))]
            )
    else:
        columns = pandas_df.columns

    pandas_df.columns = columns
    pandas_df.index = index

    modin_df.columns = columns
    modin_df.index = index

    df_equals(modin_df.stack(), pandas_df.stack())

    if is_multi_col:
        df_equals(modin_df.stack(level=0), pandas_df.stack(level=0))
        df_equals(modin_df.stack(level=[0, 1]), pandas_df.stack(level=[0, 1]))
        df_equals(modin_df.stack(level=[0, 1, 2]), pandas_df.stack(level=[0, 1, 2]))


@pytest.mark.parametrize("sort", [True, False])
def test_stack_sort(sort):
    # Example frame slightly modified from pandas docs to be unsorted
    cols = pd.MultiIndex.from_tuples([("weight", "pounds"), ("weight", "kg")])
    modin_df, pandas_df = create_test_dfs(
        [[1, 2], [2, 4]], index=["cat", "dog"], columns=cols
    )
    df_equals(modin_df.stack(sort=sort), pandas_df.stack(sort=sort))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis1", [0, 1])
@pytest.mark.parametrize("axis2", [0, 1])
def test_swapaxes(data, axis1, axis2):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    pandas_result = pandas_df.swapaxes(axis1, axis2)
    modin_result = modin_df.swapaxes(axis1, axis2)
    df_equals(modin_result, pandas_result)


def test_swapaxes_axes_names():
    modin_df = pd.DataFrame(test_data_values[0])
    modin_result1 = modin_df.swapaxes(0, 1)
    modin_result2 = modin_df.swapaxes("columns", "index")
    df_equals(modin_result1, modin_result2)


def test_swaplevel():
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
        modin_df.swaplevel("Number", "Color"),
        pandas_df.swaplevel("Number", "Color"),
    )
    df_equals(modin_df.swaplevel(), pandas_df.swaplevel())
    df_equals(modin_df.swaplevel(0, 1), pandas_df.swaplevel(0, 1))


def test_take():
    modin_df = pd.DataFrame(
        [
            ("falcon", "bird", 389.0),
            ("parrot", "bird", 24.0),
            ("lion", "mammal", 80.5),
            ("monkey", "mammal", np.nan),
        ],
        columns=["name", "class", "max_speed"],
        index=[0, 2, 3, 1],
    )
    pandas_df = pandas.DataFrame(
        [
            ("falcon", "bird", 389.0),
            ("parrot", "bird", 24.0),
            ("lion", "mammal", 80.5),
            ("monkey", "mammal", np.nan),
        ],
        columns=["name", "class", "max_speed"],
        index=[0, 2, 3, 1],
    )
    df_equals(modin_df.take([0, 3]), pandas_df.take([0, 3]))
    df_equals(modin_df.take([2], axis=1), pandas_df.take([2], axis=1))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_to_records(data):
    # `to_records` doesn't work when `index` is among column names
    eval_general(
        *create_test_dfs(data),
        lambda df: (
            df.dropna().drop("index", axis=1) if "index" in df.columns else df.dropna()
        ).to_records(),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_to_string(data):
    eval_general(
        *create_test_dfs(data),
        lambda df: df.to_string(),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_truncate(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    before = 1
    after = len(modin_df - 3)
    df_equals(modin_df.truncate(before, after), pandas_df.truncate(before, after))

    before = 1
    after = 3
    df_equals(modin_df.truncate(before, after), pandas_df.truncate(before, after))

    before = modin_df.columns[1]
    after = modin_df.columns[-3]
    try:
        pandas_result = pandas_df.truncate(before, after, axis=1)
    except Exception as err:
        with pytest.raises(type(err)):
            modin_df.truncate(before, after, axis=1)
    else:
        modin_result = modin_df.truncate(before, after, axis=1)
        df_equals(modin_result, pandas_result)

    before = modin_df.columns[1]
    after = modin_df.columns[3]
    try:
        pandas_result = pandas_df.truncate(before, after, axis=1)
    except Exception as err:
        with pytest.raises(type(err)):
            modin_df.truncate(before, after, axis=1)
    else:
        modin_result = modin_df.truncate(before, after, axis=1)
        df_equals(modin_result, pandas_result)

    before = None
    after = None
    df_equals(modin_df.truncate(before, after), pandas_df.truncate(before, after))
    try:
        pandas_result = pandas_df.truncate(before, after, axis=1)
    except Exception as err:
        with pytest.raises(type(err)):
            modin_df.truncate(before, after, axis=1)
    else:
        modin_result = modin_df.truncate(before, after, axis=1)
        df_equals(modin_result, pandas_result)


def test_truncate_before_greater_than_after():
    df = pd.DataFrame([[1, 2, 3]])
    with pytest.raises(ValueError, match="Truncate: 1 must be after 2"):
        df.truncate(before=2, after=1)


def test_tz_convert():
    modin_idx = pd.date_range(
        "1/1/2012", periods=500, freq="2D", tz="America/Los_Angeles"
    )
    pandas_idx = pandas.date_range(
        "1/1/2012", periods=500, freq="2D", tz="America/Los_Angeles"
    )
    data = np.random.randint(0, 100, size=(len(modin_idx), 4))
    modin_df = pd.DataFrame(data, index=modin_idx)
    pandas_df = pandas.DataFrame(data, index=pandas_idx)
    modin_result = modin_df.tz_convert("UTC", axis=0)
    pandas_result = pandas_df.tz_convert("UTC", axis=0)
    df_equals(modin_result, pandas_result)

    modin_multi = pd.MultiIndex.from_arrays([modin_idx, range(len(modin_idx))])
    pandas_multi = pandas.MultiIndex.from_arrays([pandas_idx, range(len(modin_idx))])
    modin_series = pd.DataFrame(data, index=modin_multi)
    pandas_series = pandas.DataFrame(data, index=pandas_multi)
    df_equals(
        modin_series.tz_convert("UTC", axis=0, level=0),
        pandas_series.tz_convert("UTC", axis=0, level=0),
    )


def test_tz_localize():
    idx = pd.date_range("1/1/2012", periods=400, freq="2D")
    data = np.random.randint(0, 100, size=(len(idx), 4))
    modin_df = pd.DataFrame(data, index=idx)
    pandas_df = pandas.DataFrame(data, index=idx)
    df_equals(modin_df.tz_localize("UTC", axis=0), pandas_df.tz_localize("UTC", axis=0))
    df_equals(
        modin_df.tz_localize("America/Los_Angeles", axis=0),
        pandas_df.tz_localize("America/Los_Angeles", axis=0),
    )


@pytest.mark.parametrize("is_multi_idx", [True, False], ids=["idx_multi", "idx_index"])
@pytest.mark.parametrize("is_multi_col", [True, False], ids=["col_multi", "col_index"])
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_unstack(data, is_multi_idx, is_multi_col):
    modin_df, pandas_df = create_test_dfs(data)

    if is_multi_idx:
        index = generate_multiindex(len(pandas_df), nlevels=4, is_tree_like=True)
    else:
        index = pandas_df.index

    if is_multi_col:
        columns = generate_multiindex(
            len(pandas_df.columns), nlevels=3, is_tree_like=True
        )
    else:
        columns = pandas_df.columns

    pandas_df.columns = modin_df.columns = columns
    pandas_df.index = modin_df.index = index

    df_equals(modin_df.unstack(), pandas_df.unstack())
    df_equals(modin_df.unstack(level=1), pandas_df.unstack(level=1))
    if is_multi_idx:
        df_equals(modin_df.unstack(level=[0, 1]), pandas_df.unstack(level=[0, 1]))
        df_equals(modin_df.unstack(level=[0, 1, 2]), pandas_df.unstack(level=[0, 1, 2]))
        df_equals(
            modin_df.unstack(level=[0, 1, 2, 3]), pandas_df.unstack(level=[0, 1, 2, 3])
        )


@pytest.mark.parametrize(
    "multi_col", ["col_multi_tree", "col_multi_not_tree", "col_index"]
)
@pytest.mark.parametrize(
    "multi_idx", ["idx_multi_tree", "idx_multi_not_tree", "idx_index"]
)
def test_unstack_multiindex_types(multi_col, multi_idx):
    MAX_NROWS = MAX_NCOLS = 36

    pandas_df = pandas.DataFrame(test_data["int_data"]).iloc[:MAX_NROWS, :MAX_NCOLS]
    modin_df = pd.DataFrame(test_data["int_data"]).iloc[:MAX_NROWS, :MAX_NCOLS]

    def get_new_index(index, cond):
        if cond == "col_multi_tree" or cond == "idx_multi_tree":
            return generate_multiindex(len(index), nlevels=3, is_tree_like=True)
        elif cond == "col_multi_not_tree" or cond == "idx_multi_not_tree":
            return generate_multiindex(len(index), nlevels=3)
        else:
            return index

    pandas_df.columns = modin_df.columns = get_new_index(pandas_df.columns, multi_col)
    pandas_df.index = modin_df.index = get_new_index(pandas_df.index, multi_idx)

    df_equals(modin_df.unstack(), pandas_df.unstack())
    df_equals(modin_df.unstack(level=1), pandas_df.unstack(level=1))
    if multi_idx != "idx_index":
        df_equals(modin_df.unstack(level=[0, 1]), pandas_df.unstack(level=[0, 1]))
        df_equals(modin_df.unstack(level=[0, 1, 2]), pandas_df.unstack(level=[0, 1, 2]))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___array__(data):
    modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)
    assert_array_equal(modin_df.__array__(), pandas_df.__array__())


@pytest.mark.parametrize("data", [[False], [True], [1, 2]])
def test___bool__(data):
    eval_general(
        *create_test_dfs(data),
        lambda df: df.__bool__(),
        expected_exception=ValueError(
            "The truth value of a DataFrame is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all()."
        ),
    )


@pytest.mark.parametrize(
    "is_sparse_data", [True, False], ids=["is_sparse", "is_not_sparse"]
)
def test_hasattr_sparse(is_sparse_data):
    modin_df, pandas_df = (
        create_test_dfs(pandas.arrays.SparseArray(test_data["float_nan_data"].values()))
        if is_sparse_data
        else create_test_dfs(test_data["float_nan_data"])
    )
    eval_general(modin_df, pandas_df, lambda df: hasattr(df, "sparse"))


def test_setattr_axes():
    # Test that setting .index or .columns does not warn
    df = pd.DataFrame([[1, 2], [3, 4]])
    with warnings.catch_warnings():
        if get_current_execution() != "BaseOnPython":
            # In BaseOnPython, setting columns raises a warning because get_axis
            #  defaults to pandas.
            warnings.simplefilter("error")
        df.index = ["foo", "bar"]
        # Check that ensure_index was called
        pd.testing.assert_index_equal(df.index, pandas.Index(["foo", "bar"]))

        df.columns = [9, 10]
        pd.testing.assert_index_equal(df.columns, pandas.Index([9, 10]))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_attrs(data):
    modin_df, pandas_df = create_test_dfs(data)
    eval_general(modin_df, pandas_df, lambda df: df.attrs)


def test_df_from_series_with_tuple_name():
    # Tests that creating a DataFrame from a series with a tuple name results in
    # a DataFrame with MultiIndex columns.
    pandas_result = pandas.DataFrame(pandas.Series(name=("a", 1)))
    # 1. Creating a Modin DF from native pandas Series
    df_equals(pd.DataFrame(pandas.Series(name=("a", 1))), pandas_result)
    # 2. Creating a Modin DF from Modin Series
    df_equals(pd.DataFrame(pd.Series(name=("a", 1))), pandas_result)


def test_large_df_warns_distributing_takes_time():
    # https://github.com/modin-project/modin/issues/6574

    regex = r"Distributing (.*) object\. This may take some time\."
    with pytest.warns(UserWarning, match=regex):
        pd.DataFrame(np.random.randint(1_000_000, size=(100_000, 10)))


def test_large_series_warns_distributing_takes_time():
    # https://github.com/modin-project/modin/issues/6574

    regex = r"Distributing (.*) object\. This may take some time\."
    with pytest.warns(UserWarning, match=regex):
        pd.Series(np.random.randint(1_000_000, size=(2_500_000)))


def test_df_does_not_warn_distributing_takes_time():
    # https://github.com/modin-project/modin/issues/6574

    regex = r"Distributing (.*) object\. This may take some time\."
    with warnings.catch_warnings():
        warnings.filterwarnings("error", regex, UserWarning)
        pd.DataFrame(np.random.randint(1_000_000, size=(100_000, 9)))


def test_series_does_not_warn_distributing_takes_time():
    # https://github.com/modin-project/modin/issues/6574

    regex = r"Distributing (.*) object\. This may take some time\."
    with warnings.catch_warnings():
        warnings.filterwarnings("error", regex, UserWarning)
        pd.Series(np.random.randint(1_000_000, size=(2_400_000)))
