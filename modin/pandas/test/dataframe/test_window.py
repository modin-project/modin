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

from modin.pandas.test.utils import (
    random_state,
    df_equals,
    arg_keys,
    name_contains,
    test_data_values,
    test_data_keys,
    no_numeric_dfs,
    quantiles_keys,
    quantiles_values,
    axis_keys,
    axis_values,
    bool_arg_keys,
    bool_arg_values,
    int_arg_keys,
    int_arg_values,
)

pd.DEFAULT_NPARTITIONS = 4

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_cummax(request, data, axis, skipna):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.cummax(axis=axis, skipna=skipna)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.cummax(axis=axis, skipna=skipna)
    else:
        modin_result = modin_df.cummax(axis=axis, skipna=skipna)
        df_equals(modin_result, pandas_result)

    try:
        pandas_result = pandas_df.T.cummax(axis=axis, skipna=skipna)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.T.cummax(axis=axis, skipna=skipna)
    else:
        modin_result = modin_df.T.cummax(axis=axis, skipna=skipna)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
def test_cummax_int_and_float(axis):
    data = {"col1": list(range(1000)), "col2": [i * 0.1 for i in range(1000)]}
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    df_equals(modin_df.cummax(axis=axis), pandas_df.cummax(axis=axis))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_cummin(request, data, axis, skipna):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.cummin(axis=axis, skipna=skipna)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.cummin(axis=axis, skipna=skipna)
    else:
        modin_result = modin_df.cummin(axis=axis, skipna=skipna)
        df_equals(modin_result, pandas_result)

    try:
        pandas_result = pandas_df.T.cummin(axis=axis, skipna=skipna)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.T.cummin(axis=axis, skipna=skipna)
    else:
        modin_result = modin_df.T.cummin(axis=axis, skipna=skipna)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
def test_cummin_int_and_float(axis):
    data = {"col1": list(range(1000)), "col2": [i * 0.1 for i in range(1000)]}
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    df_equals(modin_df.cummin(axis=axis), pandas_df.cummin(axis=axis))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_cumprod(request, data, axis, skipna):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.cumprod(axis=axis, skipna=skipna)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.cumprod(axis=axis, skipna=skipna)
    else:
        modin_result = modin_df.cumprod(axis=axis, skipna=skipna)
        df_equals(modin_result, pandas_result)

    try:
        pandas_result = pandas_df.T.cumprod(axis=axis, skipna=skipna)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.T.cumprod(axis=axis, skipna=skipna)
    else:
        modin_result = modin_df.T.cumprod(axis=axis, skipna=skipna)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_cumsum(request, data, axis, skipna):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    # pandas exhibits weird behavior for this case
    # Remove this case when we can pull the error messages from backend
    if name_contains(request.node.name, ["datetime_timedelta_data"]) and (
        axis == 0 or axis == "rows"
    ):
        with pytest.raises(TypeError):
            modin_df.cumsum(axis=axis, skipna=skipna)
    else:
        try:
            pandas_result = pandas_df.cumsum(axis=axis, skipna=skipna)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.cumsum(axis=axis, skipna=skipna)
        else:
            modin_result = modin_df.cumsum(axis=axis, skipna=skipna)
            df_equals(modin_result, pandas_result)

    if name_contains(request.node.name, ["datetime_timedelta_data"]) and (
        axis == 0 or axis == "rows"
    ):
        with pytest.raises(TypeError):
            modin_df.T.cumsum(axis=axis, skipna=skipna)
    else:
        try:
            pandas_result = pandas_df.T.cumsum(axis=axis, skipna=skipna)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.T.cumsum(axis=axis, skipna=skipna)
        else:
            modin_result = modin_df.T.cumsum(axis=axis, skipna=skipna)
            df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "periods", int_arg_values, ids=arg_keys("periods", int_arg_keys)
)
def test_diff(request, data, axis, periods):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.diff(axis=axis, periods=periods)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.diff(axis=axis, periods=periods)
    else:
        modin_result = modin_df.diff(axis=axis, periods=periods)
        df_equals(modin_result, pandas_result)

    try:
        pandas_result = pandas_df.T.diff(axis=axis, periods=periods)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.T.diff(axis=axis, periods=periods)
    else:
        modin_result = modin_df.T.diff(axis=axis, periods=periods)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize(
    "keep", ["last", "first", False], ids=["last", "first", "False"]
)
def test_duplicated(data, keep):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    pandas_result = pandas_df.duplicated(keep=keep)
    modin_result = modin_df.duplicated(keep=keep)
    df_equals(modin_result, pandas_result)

    import random

    subset = random.sample(
        list(pandas_df.columns), random.randint(1, len(pandas_df.columns))
    )
    pandas_result = pandas_df.duplicated(keep=keep, subset=subset)
    modin_result = modin_df.duplicated(keep=keep, subset=subset)

    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_ffill(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    df_equals(modin_df.ffill(), pandas_df.ffill())


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize(
    "method",
    ["backfill", "bfill", "pad", "ffill", None],
    ids=["backfill", "bfill", "pad", "ffill", "None"],
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("limit", int_arg_values, ids=int_arg_keys)
def test_fillna(data, method, axis, limit):
    # We are not testing when limit is not positive until pandas-27042 gets fixed.
    # We are not testing when axis is over rows until pandas-17399 gets fixed.
    if limit > 0 and axis != 1 and axis != "columns":
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        try:
            pandas_result = pandas_df.fillna(0, method=method, axis=axis, limit=limit)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.fillna(0, method=method, axis=axis, limit=limit)
        else:
            modin_result = modin_df.fillna(0, method=method, axis=axis, limit=limit)
            df_equals(modin_result, pandas_result)


def test_fillna_sanity():
    # with different dtype
    frame_data = [
        ["a", "a", np.nan, "a"],
        ["b", "b", np.nan, "b"],
        ["c", "c", np.nan, "c"],
    ]
    df = pandas.DataFrame(frame_data)

    result = df.fillna({2: "foo"})
    modin_df = pd.DataFrame(frame_data).fillna({2: "foo"})

    df_equals(modin_df, result)

    modin_df = pd.DataFrame(df)
    df.fillna({2: "foo"}, inplace=True)
    modin_df.fillna({2: "foo"}, inplace=True)
    df_equals(modin_df, result)

    frame_data = {
        "Date": [pandas.NaT, pandas.Timestamp("2014-1-1")],
        "Date2": [pandas.Timestamp("2013-1-1"), pandas.NaT],
    }
    df = pandas.DataFrame(frame_data)
    result = df.fillna(value={"Date": df["Date2"]})
    modin_df = pd.DataFrame(frame_data).fillna(value={"Date": df["Date2"]})
    df_equals(modin_df, result)

    frame_data = {"A": [pandas.Timestamp("2012-11-11 00:00:00+01:00"), pandas.NaT]}
    df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)
    df_equals(modin_df.fillna(method="pad"), df.fillna(method="pad"))

    frame_data = {"A": [pandas.NaT, pandas.Timestamp("2012-11-11 00:00:00+01:00")]}
    df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data).fillna(method="bfill")
    df_equals(modin_df, df.fillna(method="bfill"))


def test_fillna_downcast():
    # infer int64 from float64
    frame_data = {"a": [1.0, np.nan]}
    df = pandas.DataFrame(frame_data)
    result = df.fillna(0, downcast="infer")
    modin_df = pd.DataFrame(frame_data).fillna(0, downcast="infer")
    df_equals(modin_df, result)

    # infer int64 from float64 when fillna value is a dict
    df = pandas.DataFrame(frame_data)
    result = df.fillna({"a": 0}, downcast="infer")
    modin_df = pd.DataFrame(frame_data).fillna({"a": 0}, downcast="infer")
    df_equals(modin_df, result)


def test_fillna_inplace():
    frame_data = random_state.randn(10, 4)
    df = pandas.DataFrame(frame_data)
    df[1][:4] = np.nan
    df[3][-4:] = np.nan

    modin_df = pd.DataFrame(df)
    df.fillna(value=0, inplace=True)
    try:
        df_equals(modin_df, df)
    except AssertionError:
        pass
    else:
        assert False

    modin_df.fillna(value=0, inplace=True)
    df_equals(modin_df, df)

    modin_df = pd.DataFrame(df).fillna(value={0: 0}, inplace=True)
    assert modin_df is None

    df[1][:4] = np.nan
    df[3][-4:] = np.nan
    modin_df = pd.DataFrame(df)
    df.fillna(method="ffill", inplace=True)
    try:
        df_equals(modin_df, df)
    except AssertionError:
        pass
    else:
        assert False

    modin_df.fillna(method="ffill", inplace=True)
    df_equals(modin_df, df)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_frame_fillna_limit(data):
    pandas_df = pandas.DataFrame(data)

    index = pandas_df.index

    result = pandas_df[:2].reindex(index)
    modin_df = pd.DataFrame(result)
    df_equals(
        modin_df.fillna(method="pad", limit=2), result.fillna(method="pad", limit=2)
    )

    result = pandas_df[-2:].reindex(index)
    modin_df = pd.DataFrame(result)
    df_equals(
        modin_df.fillna(method="backfill", limit=2),
        result.fillna(method="backfill", limit=2),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_frame_pad_backfill_limit(data):
    pandas_df = pandas.DataFrame(data)

    index = pandas_df.index

    result = pandas_df[:2].reindex(index)
    modin_df = pd.DataFrame(result)
    df_equals(
        modin_df.fillna(method="pad", limit=2), result.fillna(method="pad", limit=2)
    )

    result = pandas_df[-2:].reindex(index)
    modin_df = pd.DataFrame(result)
    df_equals(
        modin_df.fillna(method="backfill", limit=2),
        result.fillna(method="backfill", limit=2),
    )


def test_fillna_dtype_conversion():
    # make sure that fillna on an empty frame works
    df = pandas.DataFrame(index=range(3), columns=["A", "B"], dtype="float64")
    modin_df = pd.DataFrame(index=range(3), columns=["A", "B"], dtype="float64")
    df_equals(modin_df.fillna("nan"), df.fillna("nan"))

    frame_data = {"A": [1, np.nan], "B": [1.0, 2.0]}
    df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)
    for v in ["", 1, np.nan, 1.0]:
        df_equals(modin_df.fillna(v), df.fillna(v))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_fillna_skip_certain_blocks(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    # don't try to fill boolean, int blocks
    df_equals(modin_df.fillna(np.nan), pandas_df.fillna(np.nan))


def test_fillna_dict_series():
    frame_data = {
        "a": [np.nan, 1, 2, np.nan, np.nan],
        "b": [1, 2, 3, np.nan, np.nan],
        "c": [np.nan, 1, 2, 3, 4],
    }
    df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)

    df_equals(modin_df.fillna({"a": 0, "b": 5}), df.fillna({"a": 0, "b": 5}))

    df_equals(
        modin_df.fillna({"a": 0, "b": 5, "d": 7}),
        df.fillna({"a": 0, "b": 5, "d": 7}),
    )

    # Series treated same as dict
    df_equals(modin_df.fillna(modin_df.max()), df.fillna(df.max()))


def test_fillna_dataframe():
    frame_data = {
        "a": [np.nan, 1, 2, np.nan, np.nan],
        "b": [1, 2, 3, np.nan, np.nan],
        "c": [np.nan, 1, 2, 3, 4],
    }
    df = pandas.DataFrame(frame_data, index=list("VWXYZ"))
    modin_df = pd.DataFrame(frame_data, index=list("VWXYZ"))

    # df2 may have different index and columns
    df2 = pandas.DataFrame(
        {
            "a": [np.nan, 10, 20, 30, 40],
            "b": [50, 60, 70, 80, 90],
            "foo": ["bar"] * 5,
        },
        index=list("VWXuZ"),
    )
    modin_df2 = pd.DataFrame(df2)

    # only those columns and indices which are shared get filled
    df_equals(modin_df.fillna(modin_df2), df.fillna(df2))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_fillna_columns(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    df_equals(
        modin_df.fillna(method="ffill", axis=1),
        pandas_df.fillna(method="ffill", axis=1),
    )

    df_equals(
        modin_df.fillna(method="ffill", axis=1),
        pandas_df.fillna(method="ffill", axis=1),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_fillna_invalid_method(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)  # noqa F841

    with pytest.raises(ValueError):
        modin_df.fillna(method="ffil")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_fillna_invalid_value(data):
    modin_df = pd.DataFrame(data)
    # list
    pytest.raises(TypeError, modin_df.fillna, [1, 2])
    # tuple
    pytest.raises(TypeError, modin_df.fillna, (1, 2))
    # frame with series
    pytest.raises(TypeError, modin_df.iloc[:, 0].fillna, modin_df)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_fillna_col_reordering(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    df_equals(modin_df.fillna(method="ffill"), pandas_df.fillna(method="ffill"))


def test_fillna_datetime_columns():
    frame_data = {
        "A": [-1, -2, np.nan],
        "B": pd.date_range("20130101", periods=3),
        "C": ["foo", "bar", None],
        "D": ["foo2", "bar2", None],
    }
    df = pandas.DataFrame(frame_data, index=pd.date_range("20130110", periods=3))
    modin_df = pd.DataFrame(frame_data, index=pd.date_range("20130110", periods=3))
    df_equals(modin_df.fillna("?"), df.fillna("?"))

    frame_data = {
        "A": [-1, -2, np.nan],
        "B": [
            pandas.Timestamp("2013-01-01"),
            pandas.Timestamp("2013-01-02"),
            pandas.NaT,
        ],
        "C": ["foo", "bar", None],
        "D": ["foo2", "bar2", None],
    }
    df = pandas.DataFrame(frame_data, index=pd.date_range("20130110", periods=3))
    modin_df = pd.DataFrame(frame_data, index=pd.date_range("20130110", periods=3))
    df_equals(modin_df.fillna("?"), df.fillna("?"))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
)
def test_median(request, data, axis, skipna, numeric_only):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.median(
            axis=axis, skipna=skipna, numeric_only=numeric_only
        )
    except Exception:
        with pytest.raises(TypeError):
            modin_df.median(axis=axis, skipna=skipna, numeric_only=numeric_only)
    else:
        modin_result = modin_df.median(
            axis=axis, skipna=skipna, numeric_only=numeric_only
        )
        df_equals(modin_result, pandas_result)

    try:
        pandas_result = pandas_df.T.median(
            axis=axis, skipna=skipna, numeric_only=numeric_only
        )
    except Exception:
        with pytest.raises(TypeError):
            modin_df.T.median(axis=axis, skipna=skipna, numeric_only=numeric_only)
    else:
        modin_result = modin_df.T.median(
            axis=axis, skipna=skipna, numeric_only=numeric_only
        )
        df_equals(modin_result, pandas_result)

    # test for issue #1953
    arrays = [["1", "1", "2", "2"], ["1", "2", "3", "4"]]
    modin_df = pd.DataFrame(
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], index=arrays
    )
    pandas_df = pandas.DataFrame(
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], index=arrays
    )
    modin_result = modin_df.median(level=0)
    pandas_result = pandas_df.median(level=0)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
)
def test_mode(request, data, axis, numeric_only):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.mode(axis=axis, numeric_only=numeric_only)
    except Exception:
        with pytest.raises(TypeError):
            modin_df.mode(axis=axis, numeric_only=numeric_only)
    else:
        modin_result = modin_df.mode(axis=axis, numeric_only=numeric_only)
        df_equals(modin_result, pandas_result)


def test_nlargest():
    data = {
        "population": [
            59000000,
            65000000,
            434000,
            434000,
            434000,
            337000,
            11300,
            11300,
            11300,
        ],
        "GDP": [1937894, 2583560, 12011, 4520, 12128, 17036, 182, 38, 311],
        "alpha-2": ["IT", "FR", "MT", "MV", "BN", "IS", "NR", "TV", "AI"],
    }
    index = [
        "Italy",
        "France",
        "Malta",
        "Maldives",
        "Brunei",
        "Iceland",
        "Nauru",
        "Tuvalu",
        "Anguilla",
    ]
    modin_df = pd.DataFrame(data=data, index=index)
    pandas_df = pandas.DataFrame(data=data, index=index)
    df_equals(modin_df.nlargest(3, "population"), pandas_df.nlargest(3, "population"))


def test_nsmallest():
    data = {
        "population": [
            59000000,
            65000000,
            434000,
            434000,
            434000,
            337000,
            11300,
            11300,
            11300,
        ],
        "GDP": [1937894, 2583560, 12011, 4520, 12128, 17036, 182, 38, 311],
        "alpha-2": ["IT", "FR", "MT", "MV", "BN", "IS", "NR", "TV", "AI"],
    }
    index = [
        "Italy",
        "France",
        "Malta",
        "Maldives",
        "Brunei",
        "Iceland",
        "Nauru",
        "Tuvalu",
        "Anguilla",
    ]
    modin_df = pd.DataFrame(data=data, index=index)
    pandas_df = pandas.DataFrame(data=data, index=index)
    df_equals(
        modin_df.nsmallest(n=3, columns="population"),
        pandas_df.nsmallest(n=3, columns="population"),
    )
    df_equals(
        modin_df.nsmallest(n=2, columns=["population", "GDP"], keep="all"),
        pandas_df.nsmallest(n=2, columns=["population", "GDP"], keep="all"),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "dropna", bool_arg_values, ids=arg_keys("dropna", bool_arg_keys)
)
def test_nunique(data, axis, dropna):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    modin_result = modin_df.nunique(axis=axis, dropna=dropna)
    pandas_result = pandas_df.nunique(axis=axis, dropna=dropna)
    df_equals(modin_result, pandas_result)

    modin_result = modin_df.T.nunique(axis=axis, dropna=dropna)
    pandas_result = pandas_df.T.nunique(axis=axis, dropna=dropna)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("q", quantiles_values, ids=quantiles_keys)
def test_quantile(request, data, q):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    if not name_contains(request.node.name, no_numeric_dfs):
        df_equals(modin_df.quantile(q), pandas_df.quantile(q))
        df_equals(modin_df.quantile(q, axis=1), pandas_df.quantile(q, axis=1))

        try:
            pandas_result = pandas_df.quantile(q, axis=1, numeric_only=False)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.quantile(q, axis=1, numeric_only=False)
        else:
            modin_result = modin_df.quantile(q, axis=1, numeric_only=False)
            df_equals(modin_result, pandas_result)
    else:
        with pytest.raises(ValueError):
            modin_df.quantile(q)

    if not name_contains(request.node.name, no_numeric_dfs):
        df_equals(modin_df.T.quantile(q), pandas_df.T.quantile(q))
        df_equals(modin_df.T.quantile(q, axis=1), pandas_df.T.quantile(q, axis=1))

        try:
            pandas_result = pandas_df.T.quantile(q, axis=1, numeric_only=False)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.T.quantile(q, axis=1, numeric_only=False)
        else:
            modin_result = modin_df.T.quantile(q, axis=1, numeric_only=False)
            df_equals(modin_result, pandas_result)
    else:
        with pytest.raises(ValueError):
            modin_df.T.quantile(q)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
)
@pytest.mark.parametrize(
    "na_option", ["keep", "top", "bottom"], ids=["keep", "top", "bottom"]
)
def test_rank(data, axis, numeric_only, na_option):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.rank(
            axis=axis, numeric_only=numeric_only, na_option=na_option
        )
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.rank(axis=axis, numeric_only=numeric_only, na_option=na_option)
    else:
        modin_result = modin_df.rank(
            axis=axis, numeric_only=numeric_only, na_option=na_option
        )
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
)
def test_skew(request, data, axis, skipna, numeric_only):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.skew(
            axis=axis, skipna=skipna, numeric_only=numeric_only
        )
    except Exception:
        with pytest.raises(TypeError):
            modin_df.skew(axis=axis, skipna=skipna, numeric_only=numeric_only)
    else:
        modin_result = modin_df.skew(
            axis=axis, skipna=skipna, numeric_only=numeric_only
        )
        df_equals(modin_result, pandas_result)

    try:
        pandas_result = pandas_df.T.skew(
            axis=axis, skipna=skipna, numeric_only=numeric_only
        )
    except Exception:
        with pytest.raises(TypeError):
            modin_df.T.skew(axis=axis, skipna=skipna, numeric_only=numeric_only)
    else:
        modin_result = modin_df.T.skew(
            axis=axis, skipna=skipna, numeric_only=numeric_only
        )
        df_equals(modin_result, pandas_result)

    # test for issue #1953
    arrays = [["1", "1", "2", "2"], ["1", "2", "3", "4"]]
    modin_df = pd.DataFrame(
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], index=arrays
    )
    pandas_df = pandas.DataFrame(
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], index=arrays
    )
    modin_result = modin_df.skew(level=0)
    pandas_result = pandas_df.skew(level=0)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
)
@pytest.mark.parametrize("ddof", int_arg_values, ids=arg_keys("ddof", int_arg_keys))
def test_std(request, data, axis, skipna, numeric_only, ddof):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.std(
            axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof
        )
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.std(axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof)
    else:
        modin_result = modin_df.std(
            axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof
        )
        df_equals(modin_result, pandas_result)

    try:
        pandas_result = pandas_df.T.std(
            axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof
        )
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.T.std(
                axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof
            )
    else:
        modin_result = modin_df.T.std(
            axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof
        )
        df_equals(modin_result, pandas_result)

    # test for issue #1953
    arrays = [["1", "1", "2", "2"], ["1", "2", "3", "4"]]
    modin_df = pd.DataFrame(
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], index=arrays
    )
    pandas_df = pandas.DataFrame(
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], index=arrays
    )
    modin_result = modin_df.std(level=0)
    pandas_result = pandas_df.std(level=0)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_values(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    np.testing.assert_equal(modin_df.values, pandas_df.values)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
)
@pytest.mark.parametrize("ddof", int_arg_values, ids=arg_keys("ddof", int_arg_keys))
def test_var(request, data, axis, skipna, numeric_only, ddof):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.var(
            axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof
        )
    except Exception:
        with pytest.raises(TypeError):
            modin_df.var(axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof)
    else:
        modin_result = modin_df.var(
            axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof
        )
        df_equals(modin_result, pandas_result)

    try:
        pandas_result = pandas_df.T.var(
            axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof
        )
    except Exception:
        with pytest.raises(TypeError):
            modin_df.T.var(
                axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof
            )
    else:
        modin_result = modin_df.T.var(
            axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof
        )

    # test for issue #1953
    arrays = [["1", "1", "2", "2"], ["1", "2", "3", "4"]]
    modin_df = pd.DataFrame(
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], index=arrays
    )
    pandas_df = pandas.DataFrame(
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], index=arrays
    )
    modin_result = modin_df.var(level=0)
    pandas_result = pandas_df.var(level=0)
    df_equals(modin_result, pandas_result)
