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
import os
import matplotlib
import modin.pandas as pd
from modin.pandas.utils import to_pandas
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
)

pd.DEFAULT_NPARTITIONS = 4

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")


def test_align():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).align(pd.DataFrame(data))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_to_numpy(data):
    modin_frame = pd.DataFrame(data)
    pandas_frame = pandas.DataFrame(data)
    assert_array_equal(modin_frame.values, pandas_frame.values)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_partition_to_numpy(data):
    frame = pd.DataFrame(data)
    for partition in frame._query_compiler._modin_frame._partitions.flatten().tolist():
        assert_array_equal(partition.to_pandas().values, partition.to_numpy())


def test_asfreq():
    index = pd.date_range("1/1/2000", periods=4, freq="T")
    series = pd.Series([0.0, None, 2.0, 3.0], index=index)
    df = pd.DataFrame({"s": series})
    with pytest.warns(UserWarning):
        # We are only testing that this defaults to pandas, so we will just check for
        # the warning
        df.asfreq(freq="30S")


def test_asof():
    df = pd.DataFrame(
        {"a": [10, 20, 30, 40, 50], "b": [None, None, None, None, 500]},
        index=pd.DatetimeIndex(
            [
                "2018-02-27 09:01:00",
                "2018-02-27 09:02:00",
                "2018-02-27 09:03:00",
                "2018-02-27 09:04:00",
                "2018-02-27 09:05:00",
            ]
        ),
    )
    with pytest.warns(UserWarning):
        df.asof(pd.DatetimeIndex(["2018-02-27 09:03:30", "2018-02-27 09:04:30"]))


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


def test_corr():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).corr()


def test_corrwith():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).corrwith(pd.DataFrame(data))


def test_cov():
    data = test_data_values[0]
    modin_result = pd.DataFrame(data).cov()
    pandas_result = pandas.DataFrame(data).cov()
    df_equals(modin_result, pandas_result)


@pytest.mark.skipif(
    os.name == "nt",
    reason="AssertionError: numpy array are different",
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


@pytest.mark.skipif(
    os.name == "nt",
    reason="AssertionError: numpy array are different",
)
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


def test_ewm():
    df = pd.DataFrame({"B": [0, 1, 2, np.nan, 4]})
    with pytest.warns(UserWarning):
        df.ewm(com=0.5).mean()


def test_expanding():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).expanding()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_explode(data):
    modin_df = pd.DataFrame(data)
    with pytest.warns(UserWarning):
        modin_df.explode(modin_df.columns[0])


def test_first():
    i = pd.date_range("2010-04-09", periods=400, freq="2D")
    modin_df = pd.DataFrame({"A": list(range(400)), "B": list(range(400))}, index=i)
    pandas_df = pandas.DataFrame(
        {"A": list(range(400)), "B": list(range(400))}, index=i
    )
    df_equals(modin_df.first("3D"), pandas_df.first("3D"))
    df_equals(modin_df.first("20D"), pandas_df.first("20D"))


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_from_dict(data):
    modin_df = pd.DataFrame(data)  # noqa F841
    pandas_df = pandas.DataFrame(data)  # noqa F841

    with pytest.raises(NotImplementedError):
        pd.DataFrame.from_dict(None)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_from_items(data):
    modin_df = pd.DataFrame(data)  # noqa F841
    pandas_df = pandas.DataFrame(data)  # noqa F841

    with pytest.raises(NotImplementedError):
        pd.DataFrame.from_items(None)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_from_records(data):
    modin_df = pd.DataFrame(data)  # noqa F841
    pandas_df = pandas.DataFrame(data)  # noqa F841

    with pytest.raises(NotImplementedError):
        pd.DataFrame.from_records(None)


def test_hist():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).hist(None)


def test_infer_objects():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).infer_objects()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("verbose", [None, True, False])
@pytest.mark.parametrize("max_cols", [None, 10, 99999999])
@pytest.mark.parametrize("memory_usage", [None, True, False, "deep"])
@pytest.mark.parametrize("null_counts", [None, True, False])
def test_info(data, verbose, max_cols, memory_usage, null_counts):
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


def test_interpolate():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).interpolate()


@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("skipna", bool_arg_values, ids=bool_arg_keys)
@pytest.mark.parametrize("level", [None, -1, 0, 1])
@pytest.mark.parametrize("numeric_only", bool_arg_values, ids=bool_arg_keys)
def test_kurt_kurtosis(axis, skipna, level, numeric_only):
    func_kwargs = {
        "axis": axis,
        "skipna": skipna,
        "level": level,
        "numeric_only": numeric_only,
    }
    data = test_data_values[0]
    df_modin = pd.DataFrame(data)
    df_pandas = pandas.DataFrame(data)

    eval_general(
        df_modin,
        df_pandas,
        lambda df: df.kurtosis(**func_kwargs),
    )

    if level is not None:
        cols_number = len(data.keys())
        arrays = [
            np.random.choice(["bar", "baz", "foo", "qux"], cols_number),
            np.random.choice(["one", "two"], cols_number),
        ]
        index = pd.MultiIndex.from_tuples(list(zip(*arrays)), names=["first", "second"])
        df_modin.columns = index
        df_pandas.columns = index
        eval_general(
            df_modin,
            df_pandas,
            lambda df: df.kurtosis(**func_kwargs),
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


def test_lookup():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).lookup([0, 1], ["col1", "col2"])


@pytest.mark.parametrize("data", test_data_values)
@pytest.mark.parametrize("axis", [None, 0, 1])
@pytest.mark.parametrize("skipna", [None, True, False])
@pytest.mark.parametrize("level", [0, -1, None])
def test_mad(level, data, axis, skipna):
    modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)
    df_equals(
        modin_df.mad(axis=axis, skipna=skipna, level=level),
        pandas_df.mad(axis=axis, skipna=skipna, level=level),
    )


def test_mask():
    df = pd.DataFrame(np.arange(10).reshape(-1, 2), columns=["A", "B"])
    m = df % 3 == 0
    with pytest.warns(UserWarning):
        try:
            df.mask(~m, -df)
        except ValueError:
            pass


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


def test_pct_change():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).pct_change()


def test_pivot():
    df = pd.DataFrame(
        {
            "foo": ["one", "one", "one", "two", "two", "two"],
            "bar": ["A", "B", "C", "A", "B", "C"],
            "baz": [1, 2, 3, 4, 5, 6],
            "zoo": ["x", "y", "z", "q", "w", "t"],
        }
    )
    with pytest.warns(UserWarning):
        df.pivot(index="foo", columns="bar", values="baz")


def test_pivot_table():
    df = pd.DataFrame(
        {
            "A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
            "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
            "C": [
                "small",
                "large",
                "large",
                "small",
                "small",
                "large",
                "small",
                "small",
                "large",
            ],
            "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
            "E": [2, 4, 5, 5, 6, 6, 8, 9, 9],
        }
    )
    with pytest.warns(UserWarning):
        df.pivot_table(values="D", index=["A", "B"], columns=["C"], aggfunc=np.sum)


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
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).replace()


@pytest.mark.parametrize("rule", ["5T", pandas.offsets.Hour()])
@pytest.mark.parametrize("axis", [0, "columns"])
@pytest.mark.parametrize("closed", ["left", "right"])
@pytest.mark.parametrize("label", ["right", "left"])
@pytest.mark.parametrize("on", [None, "DateColumn"])
@pytest.mark.parametrize("level", [None, 1])
def test_resample(rule, axis, closed, label, on, level):
    freq = "H"
    base = 2
    index = pandas.date_range("31/12/2000", periods=12, freq=freq)
    data = {"A": range(12), "B": range(12)}

    pandas_df = pandas.DataFrame(data, index=index)
    modin_df = pd.DataFrame(data, index=index)

    if on is not None and axis == 0:
        pandas_df[on] = pandas.date_range("22/06/1941", periods=12, freq="T")
        modin_df[on] = pandas.date_range("22/06/1941", periods=12, freq="T")
    else:
        on = None

    if axis == "columns":
        pandas_df = pandas_df.T
        modin_df = modin_df.T

    if level is not None and axis == 0 and on is None:
        index = pandas.MultiIndex.from_product(
            [["a", "b", "c"], pandas.date_range("31/12/2000", periods=4, freq=freq)]
        )
        pandas_df.index = index
        modin_df.index = index
    else:
        level = None

    pandas_resampler = pandas_df.resample(
        rule, axis=axis, closed=closed, label=label, base=base, on=on, level=level
    )
    modin_resampler = modin_df.resample(
        rule, axis=axis, closed=closed, label=label, base=base, on=on, level=level
    )

    df_equals(modin_resampler.count(), pandas_resampler.count())
    df_equals(modin_resampler.var(0), pandas_resampler.var(0))
    df_equals(modin_resampler.sum(), pandas_resampler.sum())
    df_equals(modin_resampler.std(), pandas_resampler.std())
    df_equals(modin_resampler.sem(), pandas_resampler.sem())
    df_equals(modin_resampler.size(), pandas_resampler.size())
    df_equals(modin_resampler.prod(), pandas_resampler.prod())
    if on is None:
        df_equals(modin_resampler.ohlc(), pandas_resampler.ohlc())
    df_equals(modin_resampler.min(), pandas_resampler.min())
    df_equals(modin_resampler.median(), pandas_resampler.median())
    df_equals(modin_resampler.mean(), pandas_resampler.mean())
    df_equals(modin_resampler.max(), pandas_resampler.max())
    df_equals(modin_resampler.last(), pandas_resampler.last())
    df_equals(modin_resampler.first(), pandas_resampler.first())
    df_equals(modin_resampler.nunique(), pandas_resampler.nunique())
    df_equals(
        modin_resampler.pipe(lambda x: x.max() - x.min()),
        pandas_resampler.pipe(lambda x: x.max() - x.min()),
    )
    df_equals(
        modin_resampler.transform(lambda x: (x - x.mean()) / x.std()),
        pandas_resampler.transform(lambda x: (x - x.mean()) / x.std()),
    )
    df_equals(
        pandas_resampler.aggregate("max"),
        modin_resampler.aggregate("max"),
    )
    df_equals(
        modin_resampler.apply("sum"),
        pandas_resampler.apply("sum"),
    )
    df_equals(
        modin_resampler.get_group(name=list(modin_resampler.groups)[0]),
        pandas_resampler.get_group(name=list(pandas_resampler.groups)[0]),
    )
    assert pandas_resampler.indices == modin_resampler.indices
    assert pandas_resampler.groups == modin_resampler.groups
    df_equals(modin_resampler.quantile(), pandas_resampler.quantile())
    if axis == 0:
        # Upsampling from level= or on= selection is not supported
        if on is None and level is None:
            df_equals(
                modin_resampler.interpolate(),
                pandas_resampler.interpolate(),
            )
            df_equals(modin_resampler.asfreq(), pandas_resampler.asfreq())
            df_equals(
                modin_resampler.fillna(method="nearest"),
                pandas_resampler.fillna(method="nearest"),
            )
            df_equals(modin_resampler.pad(), pandas_resampler.pad())
            df_equals(modin_resampler.nearest(), pandas_resampler.nearest())
            df_equals(modin_resampler.bfill(), pandas_resampler.bfill())
            df_equals(modin_resampler.backfill(), pandas_resampler.backfill())
            df_equals(modin_resampler.ffill(), pandas_resampler.ffill())
        df_equals(
            pandas_resampler.apply(["sum", "mean", "max"]),
            modin_resampler.apply(["sum", "mean", "max"]),
        )
        df_equals(
            modin_resampler.aggregate(["sum", "mean", "max"]),
            pandas_resampler.aggregate(["sum", "mean", "max"]),
        )


def test_sem():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).sem()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("index", ["default", "ndarray"])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("periods", [0, 1, -1, 10, -10, 1000000000, -1000000000])
def test_shift(data, index, axis, periods):
    if index == "default":
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)
    elif index == "ndarray":
        data_column_length = len(data[next(iter(data))])
        index_data = np.arange(2, data_column_length + 2)
        modin_df = pd.DataFrame(data, index=index_data)
        pandas_df = pandas.DataFrame(data, index=index_data)

    df_equals(
        modin_df.shift(periods=periods, axis=axis),
        pandas_df.shift(periods=periods, axis=axis),
    )
    df_equals(
        modin_df.shift(periods=periods, axis=axis, fill_value=777),
        pandas_df.shift(periods=periods, axis=axis, fill_value=777),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("index", ["default", "ndarray"])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("periods", [0, 1, -1, 10, -10, 1000000000, -1000000000])
def test_slice_shift(data, index, axis, periods):
    if index == "default":
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)
    elif index == "ndarray":
        data_column_length = len(data[next(iter(data))])
        index_data = np.arange(2, data_column_length + 2)
        modin_df = pd.DataFrame(data, index=index_data)
        pandas_df = pandas.DataFrame(data, index=index_data)

    df_equals(
        modin_df.slice_shift(periods=periods, axis=axis),
        pandas_df.slice_shift(periods=periods, axis=axis),
    )


def test_stack():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).stack()


def test_style():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).style


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis1", [0, 1, "columns", "index"])
@pytest.mark.parametrize("axis2", [0, 1, "columns", "index"])
def test_swapaxes(data, axis1, axis2):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    try:
        pandas_result = pandas_df.swapaxes(axis1, axis2)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.swapaxes(axis1, axis2)
    else:
        modin_result = modin_df.swapaxes(axis1, axis2)
        df_equals(modin_result, pandas_result)


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
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    # Skips nan because only difference is nan instead of NaN
    if not name_contains(request.node.name, ["nan"]):
        try:
            pandas_result = pandas_df.to_records()
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.to_records()
        else:
            modin_result = modin_df.to_records()
            assert np.array_equal(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_to_string(request, data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)  # noqa F841

    # Skips nan because only difference is nan instead of NaN
    if not name_contains(request.node.name, ["nan"]):
        assert modin_df.to_string() == to_pandas(modin_df).to_string()


def test_to_timestamp():
    idx = pd.date_range("1/1/2012", periods=5, freq="M")
    df = pd.DataFrame(np.random.randint(0, 100, size=(len(idx), 4)), index=idx)

    with pytest.warns(UserWarning):
        df.to_period().to_timestamp()


def test_to_xarray():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).to_xarray()


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


def test_unstack():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).unstack()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___array__(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    assert_array_equal(modin_df.__array__(), pandas_df.__array__())


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___bool__(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.__bool__()
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.__bool__()
    else:
        modin_result = modin_df.__bool__()
        df_equals(modin_result, pandas_result)


def test___getstate__():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).__getstate__()


def test___setstate__():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        try:
            pd.DataFrame(data).__setstate__(None)
        except TypeError:
            pass


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_hasattr_sparse(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    try:
        pandas_result = hasattr(pandas_df, "sparse")
    except Exception as e:
        with pytest.raises(type(e)):
            hasattr(modin_df, "sparse")
    else:
        modin_result = hasattr(modin_df, "sparse")
        assert modin_result == pandas_result
