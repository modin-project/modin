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
from modin.utils import to_pandas
from numpy.testing import assert_array_equal
import io

from modin.pandas.test.utils import (
    df_equals,
    name_contains,
    test_data_values,
    test_data_keys,
    numeric_dfs,
    axis_keys,
    axis_values,
    bool_arg_keys,
    bool_arg_values,
    eval_general,
    create_test_dfs,
    generate_multiindex,
    test_data_resample,
    test_data,
    test_data_diff_dtype,
    modin_df_almost_equals_pandas,
    test_data_large_categorical_dataframe,
    default_to_pandas_ignore_string,
)
from modin.config import NPartitions
from modin.test.test_utils import warns_that_defaulting_to_pandas

NPartitions.put(4)

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")

# Our configuration in pytest.ini requires that we explicitly catch all
# instances of defaulting to pandas, but some test modules, like this one,
# have too many such instances.
pytestmark = pytest.mark.filterwarnings(default_to_pandas_ignore_string)


@pytest.mark.parametrize(
    "op, make_args",
    [
        ("align", lambda df: {"other": df}),
        ("expanding", None),
        ("corrwith", lambda df: {"other": df}),
        ("ewm", lambda df: {"com": 0.5}),
        ("from_dict", lambda df: {"data": None}),
        ("from_records", lambda df: {"data": to_pandas(df)}),
        ("hist", lambda df: {"column": "int_col"}),
        ("infer_objects", None),
        ("interpolate", None),
        ("lookup", lambda df: {"row_labels": [0], "col_labels": ["int_col"]}),
        ("mask", lambda df: {"cond": df != 0}),
        ("pct_change", None),
        ("to_xarray", None),
        ("flags", None),
        ("set_flags", lambda df: {"allows_duplicate_labels": False}),
    ],
)
def test_ops_defaulting_to_pandas(op, make_args):
    modin_df = pd.DataFrame(test_data_diff_dtype).drop(["str_col", "bool_col"], axis=1)
    with warns_that_defaulting_to_pandas():
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
    with warns_that_defaulting_to_pandas():
        pd.DataFrame(data).style


def test_to_timestamp():
    idx = pd.date_range("1/1/2012", periods=5, freq="M")
    df = pd.DataFrame(np.random.randint(0, 100, size=(len(idx), 4)), index=idx)

    with warns_that_defaulting_to_pandas():
        df.to_period().to_timestamp()


@pytest.mark.parametrize(
    "data",
    test_data_values + [test_data_large_categorical_dataframe],
    ids=test_data_keys + ["categorical_ints"],
)
def test_to_numpy(data):
    modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)
    assert_array_equal(modin_df.values, pandas_df.values)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_partition_to_numpy(data):
    frame = pd.DataFrame(data)
    for partition in frame._query_compiler._modin_frame._partitions.flatten().tolist():
        assert_array_equal(partition.to_pandas().values, partition.to_numpy())


def test_asfreq():
    index = pd.date_range("1/1/2000", periods=4, freq="T")
    series = pd.Series([0.0, None, 2.0, 3.0], index=index)
    df = pd.DataFrame({"s": series})
    with warns_that_defaulting_to_pandas():
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


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_bool(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)  # noqa F841

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
    pandas_df = pandas.DataFrame(data)  # noqa F841

    assert modin_df.boxplot() == to_pandas(modin_df).boxplot()


def test_combine_first():
    data1 = {"A": [None, 0], "B": [None, 4]}
    modin_df1 = pd.DataFrame(data1)
    pandas_df1 = pandas.DataFrame(data1)
    data2 = {"A": [1, 1], "B": [3, 3]}
    modin_df2 = pd.DataFrame(data2)
    pandas_df2 = pandas.DataFrame(data2)
    df_equals(modin_df1.combine_first(modin_df2), pandas_df1.combine_first(pandas_df2))


@pytest.mark.parametrize("min_periods", [1, 3, 5])
def test_corr(min_periods):
    eval_general(
        *create_test_dfs(test_data["int_data"]),
        lambda df: df.corr(min_periods=min_periods),
    )
    # Modin result may slightly differ from pandas result
    # due to floating pointing arithmetic.
    eval_general(
        *create_test_dfs(test_data["float_nan_data"]),
        lambda df: df.corr(min_periods=min_periods),
        comparator=modin_df_almost_equals_pandas,
    )


@pytest.mark.parametrize("min_periods", [1, 3, 5])
@pytest.mark.parametrize("ddof", [1, 2, 4])
def test_cov(min_periods, ddof):
    eval_general(
        *create_test_dfs(test_data["int_data"]),
        lambda df: df.cov(min_periods=min_periods, ddof=ddof),
    )
    # Modin result may slightly differ from pandas result
    # due to floating pointing arithmetic.
    eval_general(
        *create_test_dfs(test_data["float_nan_data"]),
        lambda df: df.cov(min_periods=min_periods),
        comparator=modin_df_almost_equals_pandas,
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
        modin_result = modin_df.dot(np.arange(col_len + 10))

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
        modin_result = modin_df.dot(pd.Series(np.arange(col_len)))

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
        modin_result = modin_df @ np.arange(col_len + 10)

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
        modin_result = modin_df @ pd.Series(np.arange(col_len))


def test_first():
    i = pd.date_range("2010-04-09", periods=400, freq="2D")
    modin_df = pd.DataFrame({"A": list(range(400)), "B": list(range(400))}, index=i)
    pandas_df = pandas.DataFrame(
        {"A": list(range(400)), "B": list(range(400))}, index=i
    )
    df_equals(modin_df.first("3D"), pandas_df.first("3D"))
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
            null_counts=None,
            operation=lambda df, **kwargs: df.info(**kwargs),
            buf=lambda df: second if isinstance(df, pandas.DataFrame) else first,
        )
        modin_info = first.getvalue().splitlines()
        pandas_info = second.getvalue().splitlines()

        assert modin_info[0] == str(pd.DataFrame)
        assert pandas_info[0] == str(pandas.DataFrame)
        assert modin_info[1:] == pandas_info[1:]


@pytest.mark.parametrize("verbose", [True, False])
@pytest.mark.parametrize("max_cols", [10, 99999999])
@pytest.mark.parametrize("memory_usage", [True, False, "deep"])
@pytest.mark.parametrize("null_counts", [True, False])
def test_info(verbose, max_cols, memory_usage, null_counts):
    data = test_data_values[0]
    with io.StringIO() as first, io.StringIO() as second:
        eval_general(
            pd.DataFrame(data),
            pandas.DataFrame(data),
            operation=lambda df, **kwargs: df.info(**kwargs),
            verbose=verbose,
            max_cols=max_cols,
            memory_usage=memory_usage,
            null_counts=null_counts,
            buf=lambda df: second if isinstance(df, pandas.DataFrame) else first,
        )
        modin_info = first.getvalue().splitlines()
        pandas_info = second.getvalue().splitlines()

        assert modin_info[0] == str(pd.DataFrame)
        assert pandas_info[0] == str(pandas.DataFrame)
        assert modin_info[1:] == pandas_info[1:]


@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("skipna", bool_arg_values, ids=bool_arg_keys)
@pytest.mark.parametrize("numeric_only", bool_arg_values, ids=bool_arg_keys)
@pytest.mark.parametrize("method", ["kurtosis", "kurt"])
def test_kurt_kurtosis(axis, skipna, numeric_only, method):
    data = test_data["float_nan_data"]

    eval_general(
        *create_test_dfs(data),
        lambda df: getattr(df, method)(
            axis=axis, skipna=skipna, numeric_only=numeric_only
        ),
    )


@pytest.mark.parametrize("level", [-1, 0, 1])
def test_kurt_kurtosis_level(level):
    data = test_data["int_data"]
    df_modin, df_pandas = pd.DataFrame(data), pandas.DataFrame(data)

    index = generate_multiindex(len(data.keys()))
    df_modin.columns = index
    df_pandas.columns = index

    eval_general(
        df_modin,
        df_pandas,
        lambda df: df.kurtosis(axis=1, level=level),
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
    df_equals(modin_df.last("3D"), pandas_df.last("3D"))
    df_equals(modin_df.last("20D"), pandas_df.last("20D"))


@pytest.mark.parametrize("data", test_data_values)
@pytest.mark.parametrize("axis", [None, 0, 1])
@pytest.mark.parametrize("skipna", [None, True, False])
def test_mad(data, axis, skipna):
    modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)
    df_equals(
        modin_df.mad(axis=axis, skipna=skipna, level=None),
        pandas_df.mad(axis=axis, skipna=skipna, level=None),
    )


@pytest.mark.parametrize("level", [-1, 0, 1])
def test_mad_level(level):
    data = test_data_values[0]
    modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)

    index = generate_multiindex(len(data.keys()))
    modin_df.columns = index
    pandas_df.columns = index
    eval_general(
        modin_df,
        pandas_df,
        lambda df: df.mad(axis=1, level=level),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize(
    "id_vars", [lambda df: df.columns[0], lambda df: df.columns[:4], None]
)
@pytest.mark.parametrize(
    "value_vars", [lambda df: df.columns[-1], lambda df: df.columns[-4:], None]
)
def test_melt(data, id_vars, value_vars):
    eval_general(
        *create_test_dfs(data),
        lambda df, *args, **kwargs: df.melt(*args, **kwargs)
        .sort_values(["variable", "value"])
        .reset_index(drop=True),
        id_vars=id_vars,
        value_vars=value_vars,
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize(
    "index", [lambda df: df.columns[0], lambda df: df[df.columns[0]].values, None]
)
@pytest.mark.parametrize("columns", [lambda df: df.columns[len(df.columns) // 2]])
@pytest.mark.parametrize(
    "values", [lambda df: df.columns[-1], lambda df: df.columns[-2:], None]
)
def test_pivot(data, index, columns, values):
    eval_general(
        *create_test_dfs(data),
        lambda df, *args, **kwargs: df.pivot(*args, **kwargs),
        index=index,
        columns=columns,
        values=values,
        check_exception_type=None,
    )


@pytest.mark.parametrize("data", [test_data["int_data"]], ids=["int_data"])
@pytest.mark.parametrize(
    "index",
    [
        lambda df: df.columns[0],
        lambda df: [*df.columns[0:2], *df.columns[-7:-4]],
        None,
    ],
)
@pytest.mark.parametrize(
    "columns",
    [
        lambda df: df.columns[len(df.columns) // 2],
        lambda df: [
            *df.columns[(len(df.columns) // 2) : (len(df.columns) // 2 + 4)],
            df.columns[-7],
        ],
        None,
    ],
)
@pytest.mark.parametrize(
    "values", [lambda df: df.columns[-1], lambda df: df.columns[-4:-1], None]
)
def test_pivot_table_data(data, index, columns, values):
    md_df, pd_df = create_test_dfs(data)

    # when values is None the output will be huge-dimensional,
    # so reducing dimension of testing data at that case
    if values is None:
        md_df, pd_df = md_df.iloc[:42, :42], pd_df.iloc[:42, :42]
    eval_general(
        md_df,
        pd_df,
        operation=lambda df, *args, **kwargs: df.pivot_table(
            *args, **kwargs
        ).sort_index(axis=int(index is not None)),
        index=index,
        columns=columns,
        values=values,
        check_exception_type=None,
    )


@pytest.mark.parametrize("data", [test_data["int_data"]], ids=["int_data"])
@pytest.mark.parametrize(
    "index",
    [
        lambda df: df.columns[0],
        lambda df: [df.columns[0], df.columns[len(df.columns) // 2 - 1]],
    ],
)
@pytest.mark.parametrize(
    "columns",
    [
        lambda df: df.columns[len(df.columns) // 2],
        lambda df: [
            *df.columns[(len(df.columns) // 2) : (len(df.columns) // 2 + 4)],
            df.columns[-7],
        ],
    ],
)
@pytest.mark.parametrize(
    "values", [lambda df: df.columns[-1], lambda df: df.columns[-4:-1]]
)
@pytest.mark.parametrize(
    "aggfunc",
    [["mean", "sum"], lambda df: {df.columns[5]: "mean", df.columns[-5]: "sum"}],
)
@pytest.mark.parametrize("margins_name", ["Custom name", None])
def test_pivot_table_margins(
    data,
    index,
    columns,
    values,
    aggfunc,
    margins_name,
):
    eval_general(
        *create_test_dfs(data),
        operation=lambda df, *args, **kwargs: df.pivot_table(*args, **kwargs),
        index=index,
        columns=columns,
        values=values,
        aggfunc=aggfunc,
        margins=True,
        margins_name=margins_name,
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


@pytest.mark.parametrize("rule", ["5T", pandas.offsets.Hour()])
@pytest.mark.parametrize("axis", [0])
def test_resampler(rule, axis):
    data, index, = (
        test_data_resample["data"],
        test_data_resample["index"],
    )
    modin_resampler = pd.DataFrame(data, index=index).resample(rule, axis=axis, base=2)
    pandas_resampler = pandas.DataFrame(data, index=index).resample(
        rule, axis=axis, base=2
    )

    assert pandas_resampler.indices == modin_resampler.indices
    assert pandas_resampler.groups == modin_resampler.groups

    df_equals(
        modin_resampler.get_group(name=list(modin_resampler.groups)[0]),
        pandas_resampler.get_group(name=list(pandas_resampler.groups)[0]),
    )


@pytest.mark.parametrize("rule", ["5T"])
@pytest.mark.parametrize("axis", ["index", "columns"])
@pytest.mark.parametrize(
    "method",
    [
        *("count", "sum", "std", "sem", "size", "prod", "ohlc", "quantile"),
        *("min", "median", "mean", "max", "last", "first", "nunique", "var"),
        *("interpolate", "asfreq", "pad", "nearest", "bfill", "backfill", "ffill"),
    ],
)
def test_resampler_functions(rule, axis, method):
    data, index, = (
        test_data_resample["data"],
        test_data_resample["index"],
    )
    modin_df = pd.DataFrame(data, index=index)
    pandas_df = pandas.DataFrame(data, index=index)

    eval_general(
        modin_df,
        pandas_df,
        lambda df: getattr(df.resample(rule, axis=axis, base=2), method)(),
    )


@pytest.mark.parametrize("rule", ["5T"])
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
    data, index, = (
        test_data_resample["data"],
        test_data_resample["index"],
    )
    modin_df = pd.DataFrame(data, index=index)
    pandas_df = pandas.DataFrame(data, index=index)

    method, arg = method_arg[0], method_arg[1]

    eval_general(
        modin_df,
        pandas_df,
        lambda df: getattr(df.resample(rule, axis=axis, base=2), method)(arg),
    )


@pytest.mark.parametrize("rule", ["5T"])
@pytest.mark.parametrize("closed", ["left", "right"])
@pytest.mark.parametrize("label", ["right", "left"])
@pytest.mark.parametrize("on", [None, "DateColumn"])
@pytest.mark.parametrize("level", [None, 1])
def test_resample_specific(rule, closed, label, on, level):
    data, index, = (
        test_data_resample["data"],
        test_data_resample["index"],
    )
    modin_df = pd.DataFrame(data, index=index)
    pandas_df = pandas.DataFrame(data, index=index)

    if on is None and level is not None:
        index = pandas.MultiIndex.from_product(
            [["a", "b", "c"], pandas.date_range("31/12/2000", periods=4, freq="H")]
        )
        pandas_df.index = index
        modin_df.index = index
    else:
        level = None

    if on is not None:
        pandas_df[on] = pandas.date_range("22/06/1941", periods=12, freq="T")
        modin_df[on] = pandas.date_range("22/06/1941", periods=12, freq="T")

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
        ["price", "date"],
        ("volume",),
        pandas.Series(["volume"]),
        pandas.Index(["volume"]),
        ["volume", "volume", "volume"],
        ["volume", "price", "date"],
    ],
    ids=[
        "column",
        "missed_column",
        "list",
        "missed_column",
        "tuple",
        "series",
        "index",
        "duplicate_column",
        "missed_columns",
    ],
)
def test_resample_getitem(columns):
    index = pandas.date_range("1/1/2013", periods=9, freq="T")
    data = {
        "price": range(9),
        "volume": range(10, 19),
    }
    eval_general(
        *create_test_dfs(data, index=index),
        lambda df: df.resample("3T")[columns].mean(),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("index", ["default", "ndarray", "has_duplicates"])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("periods", [0, 1, -1, 10, -10, 1000000000, -1000000000])
def test_shift_slice_shift(data, index, axis, periods):
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
    df_equals(
        modin_df.slice_shift(periods=periods, axis=axis),
        pandas_df.slice_shift(periods=periods, axis=axis),
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
def test_to_records(request, data):
    eval_general(
        *create_test_dfs(data),
        lambda df: df.dropna().to_records(),
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
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.truncate(before, after, axis=1)
    else:
        modin_result = modin_df.truncate(before, after, axis=1)
        df_equals(modin_result, pandas_result)

    before = modin_df.columns[1]
    after = modin_df.columns[3]
    try:
        pandas_result = pandas_df.truncate(before, after, axis=1)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.truncate(before, after, axis=1)
    else:
        modin_result = modin_df.truncate(before, after, axis=1)
        df_equals(modin_result, pandas_result)

    before = None
    after = None
    df_equals(modin_df.truncate(before, after), pandas_df.truncate(before, after))
    try:
        pandas_result = pandas_df.truncate(before, after, axis=1)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.truncate(before, after, axis=1)
    else:
        modin_result = modin_df.truncate(before, after, axis=1)
        df_equals(modin_result, pandas_result)


def test_tshift():
    idx = pd.date_range("1/1/2012", periods=5, freq="M")
    data = np.random.randint(0, 100, size=(len(idx), 4))
    modin_df = pd.DataFrame(data, index=idx)
    pandas_df = pandas.DataFrame(data, index=idx)
    df_equals(modin_df.tshift(4), pandas_df.tshift(4))


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


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___bool__(data):
    eval_general(*create_test_dfs(data), lambda df: df.__bool__())


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
