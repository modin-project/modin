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

import contextlib

import matplotlib
import numpy as np
import pandas
import pytest

import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.pandas.test.utils import (
    arg_keys,
    axis_keys,
    axis_values,
    bool_arg_keys,
    bool_arg_values,
    create_test_dfs,
    default_to_pandas_ignore_string,
    df_equals,
    eval_general,
    int_arg_keys,
    int_arg_values,
    name_contains,
    no_numeric_dfs,
    quantiles_keys,
    quantiles_values,
    random_state,
    test_data,
    test_data_diff_dtype,
    test_data_keys,
    test_data_values,
    test_data_with_duplicates_keys,
    test_data_with_duplicates_values,
)

NPartitions.put(4)

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")

if StorageFormat.get() == "Hdk":
    pytestmark = pytest.mark.filterwarnings(default_to_pandas_ignore_string)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize("method", ["cumprod", "cummin", "cummax", "cumsum"])
def test_cumprod_cummin_cummax_cumsum(axis, skipna, method):
    eval_general(
        *create_test_dfs(test_data["float_nan_data"]),
        lambda df: getattr(df, method)(axis=axis, skipna=skipna),
    )


@pytest.mark.parametrize("axis", ["rows", "columns"])
@pytest.mark.parametrize("method", ["cumprod", "cummin", "cummax", "cumsum"])
def test_cumprod_cummin_cummax_cumsum_transposed(axis, method):
    eval_general(
        *create_test_dfs(test_data["int_data"]),
        lambda df: getattr(df.T, method)(axis=axis),
    )


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("method", ["cummin", "cummax"])
def test_cummin_cummax_int_and_float(axis, method):
    data = {"col1": list(range(1000)), "col2": [i * 0.1 for i in range(1000)]}
    eval_general(*create_test_dfs(data), lambda df: getattr(df, method)(axis=axis))


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize(
    "periods", int_arg_values, ids=arg_keys("periods", int_arg_keys)
)
def test_diff(axis, periods):
    eval_general(
        *create_test_dfs(test_data["float_nan_data"]),
        lambda df: df.diff(axis=axis, periods=periods),
    )


def test_diff_with_datetime_types():
    pandas_df = pandas.DataFrame(
        [[1, 2.0, 3], [4, 5.0, 6], [7, np.nan, 9], [10, 11.3, 12], [13, 14.5, 15]]
    )
    data = pandas.date_range("2018-01-01", periods=5, freq="H").values
    pandas_df = pandas.concat([pandas_df, pandas.Series(data)], axis=1)
    modin_df = pd.DataFrame(pandas_df)

    # Test `diff` with datetime type.
    pandas_result = pandas_df.diff()
    modin_result = modin_df.diff()
    df_equals(modin_result, pandas_result)

    # Test `diff` with timedelta type.
    td_pandas_result = pandas_result.diff()
    td_modin_result = modin_result.diff()
    df_equals(td_modin_result, td_pandas_result)


def test_diff_error_handling():
    df = pd.DataFrame([["a", "b", "c"]], columns=["col 0", "col 1", "col 2"])
    with pytest.raises(
        ValueError, match="periods must be an int. got <class 'str'> instead"
    ):
        df.diff(axis=0, periods="1")

    with pytest.raises(TypeError, match="unsupported operand type for -: got object"):
        df.diff()


@pytest.mark.parametrize("axis", ["rows", "columns"])
def test_diff_transposed(axis):
    eval_general(
        *create_test_dfs(test_data["int_data"]),
        lambda df: df.T.diff(axis=axis),
    )


@pytest.mark.parametrize(
    "data", test_data_with_duplicates_values, ids=test_data_with_duplicates_keys
)
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
@pytest.mark.parametrize("limit", [None, 1, 5])
def test_fillna(data, method, axis, limit):
    # We are not testing when axis is over rows until pandas-17399 gets fixed.
    if axis != 1 and axis != "columns":
        with pytest.warns(
            FutureWarning, match=".*DataFrame.fillna with 'method' is deprecated.*"
        ) if method is not None else contextlib.nullcontext():
            eval_general(
                *create_test_dfs(data),
                lambda df: df.fillna(
                    value=0 if method is None else None,
                    method=method,
                    axis=axis,
                    limit=limit,
                ),
            )


@pytest.mark.skipif(
    StorageFormat.get() == "Hdk",
    reason="'datetime64[ns, pytz.FixedOffset(60)]' vs 'datetime64[ns, UTC+01:00]'",
)
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
    with pytest.warns(
        FutureWarning, match=".*DataFrame.fillna with 'method' is deprecated.*"
    ):
        eval_general(
            *create_test_dfs(frame_data),
            lambda df: df.fillna(method="pad"),
        )

    frame_data = {"A": [pandas.NaT, pandas.Timestamp("2012-11-11 00:00:00+01:00")]}
    with pytest.warns(
        FutureWarning, match=".*DataFrame.fillna with 'method' is deprecated.*"
    ):
        eval_general(
            *create_test_dfs(frame_data),
            lambda df: df.fillna(method="bfill"),
        )


def test_fillna_downcast():
    # infer int64 from float64
    frame_data = {"a": [1.0, np.nan]}
    df = pandas.DataFrame(frame_data)
    with pytest.warns(
        FutureWarning, match=".*'downcast' keyword in fillna is deprecated.*"
    ):
        result = df.fillna(0, downcast="infer")
        modin_df = pd.DataFrame(frame_data).fillna(0, downcast="infer")
    df_equals(modin_df, result)

    # infer int64 from float64 when fillna value is a dict
    df = pandas.DataFrame(frame_data)
    with pytest.warns(
        FutureWarning, match=".*'downcast' keyword in fillna is deprecated.*"
    ):
        result = df.fillna({"a": 0}, downcast="infer")
        modin_df = pd.DataFrame(frame_data).fillna({"a": 0}, downcast="infer")
    df_equals(modin_df, result)


def test_fillna_4660():
    eval_general(
        *create_test_dfs({"a": ["a"], "b": ["b"], "c": [pd.NA]}, index=["row1"]),
        lambda df: df["c"].fillna(df["b"]),
    )


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
    with pytest.warns(
        FutureWarning, match=".*DataFrame.fillna with 'method' is deprecated.*"
    ):
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
@pytest.mark.parametrize("limit", [1, 2, 0.5, -1, -2, 1.5])
def test_frame_fillna_limit(data, limit):
    pandas_df = pandas.DataFrame(data)

    replace_pandas_series = pandas_df.columns.to_series().sample(frac=1)
    replace_dict = replace_pandas_series.to_dict()
    replace_pandas_df = pandas.DataFrame(
        {col: pandas_df.index.to_series() for col in pandas_df.columns},
        index=pandas_df.index,
    ).sample(frac=1)
    replace_modin_series = pd.Series(replace_pandas_series)
    replace_modin_df = pd.DataFrame(replace_pandas_df)

    index = pandas_df.index
    result = pandas_df[:2].reindex(index)
    modin_df = pd.DataFrame(result)

    if isinstance(limit, float):
        limit = int(len(modin_df) * limit)
    if limit is not None and limit < 0:
        limit = len(modin_df) + limit

    with pytest.warns(
        FutureWarning, match=".*DataFrame.fillna with 'method' is deprecated.*"
    ):
        df_equals(
            modin_df.fillna(method="pad", limit=limit),
            result.fillna(method="pad", limit=limit),
        )
    df_equals(
        modin_df.fillna(replace_dict, limit=limit),
        result.fillna(replace_dict, limit=limit),
    )
    df_equals(
        modin_df.fillna(replace_modin_series, limit=limit),
        result.fillna(replace_pandas_series, limit=limit),
    )
    df_equals(
        modin_df.fillna(replace_modin_df, limit=limit),
        result.fillna(replace_pandas_df, limit=limit),
    )

    result = pandas_df[-2:].reindex(index)
    modin_df = pd.DataFrame(result)

    with pytest.warns(
        FutureWarning, match=".*DataFrame.fillna with 'method' is deprecated.*"
    ):
        df_equals(
            modin_df.fillna(method="backfill", limit=limit),
            result.fillna(method="backfill", limit=limit),
        )
    df_equals(
        modin_df.fillna(replace_dict, limit=limit),
        result.fillna(replace_dict, limit=limit),
    )
    df_equals(
        modin_df.fillna(replace_modin_series, limit=limit),
        result.fillna(replace_pandas_series, limit=limit),
    )
    df_equals(
        modin_df.fillna(replace_modin_df, limit=limit),
        result.fillna(replace_pandas_df, limit=limit),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_frame_pad_backfill_limit(data):
    pandas_df = pandas.DataFrame(data)

    index = pandas_df.index

    result = pandas_df[:2].reindex(index)
    modin_df = pd.DataFrame(result)

    with pytest.warns(
        FutureWarning, match=".*DataFrame.fillna with 'method' is deprecated.*"
    ):
        df_equals(
            modin_df.fillna(method="pad", limit=2), result.fillna(method="pad", limit=2)
        )

    result = pandas_df[-2:].reindex(index)
    modin_df = pd.DataFrame(result)

    with pytest.warns(
        FutureWarning, match=".*DataFrame.fillna with 'method' is deprecated.*"
    ):
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
        {"a": [np.nan, 10, 20, 30, 40], "b": [50, 60, 70, 80, 90], "foo": ["bar"] * 5},
        index=list("VWXuZ"),
    )
    modin_df2 = pd.DataFrame(df2)

    # only those columns and indices which are shared get filled
    df_equals(modin_df.fillna(modin_df2), df.fillna(df2))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_fillna_columns(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    with pytest.warns(
        FutureWarning, match=".*DataFrame.fillna with 'method' is deprecated.*"
    ):
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

    with pytest.warns(
        FutureWarning, match=".*DataFrame.fillna with 'method' is deprecated.*"
    ):
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


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize("method", ["median", "skew"])
def test_median_skew(axis, skipna, method):
    eval_general(
        *create_test_dfs(test_data["float_nan_data"]),
        lambda df: getattr(df, method)(axis=axis, skipna=skipna),
    )


@pytest.mark.parametrize("axis", ["rows", "columns"])
@pytest.mark.parametrize("method", ["median", "skew"])
def test_median_skew_transposed(axis, method):
    eval_general(
        *create_test_dfs(test_data["int_data"]),
        lambda df: getattr(df.T, method)(axis=axis),
    )


@pytest.mark.parametrize("numeric_only", [True, False, None])
@pytest.mark.parametrize("method", ["median", "skew", "std", "var", "rank", "sem"])
def test_median_skew_std_var_rank_sem_specific(numeric_only, method):
    eval_general(
        *create_test_dfs(test_data_diff_dtype),
        lambda df: getattr(df, method)(numeric_only=numeric_only),
    )


@pytest.mark.parametrize("method", ["median", "skew", "std", "var", "sem"])
def test_median_skew_std_var_sem_1953(method):
    # See #1953 for details
    arrays = [["1", "1", "2", "2"], ["1", "2", "3", "4"]]
    data = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
    modin_df = pd.DataFrame(data, index=arrays)
    pandas_df = pandas.DataFrame(data, index=arrays)

    eval_general(modin_df, pandas_df, lambda df: getattr(df, method)())


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
        except Exception as err:
            with pytest.raises(type(err)):
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
        except Exception as err:
            with pytest.raises(type(err)):
                modin_df.T.quantile(q, axis=1, numeric_only=False)
        else:
            modin_result = modin_df.T.quantile(q, axis=1, numeric_only=False)
            df_equals(modin_result, pandas_result)
    else:
        with pytest.raises(ValueError):
            modin_df.T.quantile(q)


@pytest.mark.parametrize("axis", ["rows", "columns"])
@pytest.mark.parametrize(
    "na_option", ["keep", "top", "bottom"], ids=["keep", "top", "bottom"]
)
def test_rank_transposed(axis, na_option):
    eval_general(
        *create_test_dfs(test_data["int_data"]),
        lambda df: df.rank(axis=axis, na_option=na_option),
    )


@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize("ddof", int_arg_values, ids=arg_keys("ddof", int_arg_keys))
def test_sem_float_nan_only(skipna, ddof):
    eval_general(
        *create_test_dfs(test_data["float_nan_data"]),
        lambda df: df.sem(skipna=skipna, ddof=ddof),
    )


@pytest.mark.parametrize("axis", ["rows", "columns"])
@pytest.mark.parametrize("ddof", int_arg_values, ids=arg_keys("ddof", int_arg_keys))
def test_sem_int_only(axis, ddof):
    eval_general(
        *create_test_dfs(test_data["int_data"]),
        lambda df: df.sem(axis=axis, ddof=ddof),
    )


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize("method", ["std", "var", "rank"])
def test_std_var_rank(axis, skipna, method):
    eval_general(
        *create_test_dfs(test_data["float_nan_data"]),
        lambda df: getattr(df, method)(axis=axis, skipna=skipna),
    )


@pytest.mark.parametrize("axis", ["rows", "columns"])
@pytest.mark.parametrize("ddof", int_arg_values, ids=arg_keys("ddof", int_arg_keys))
@pytest.mark.parametrize("method", ["std", "var"])
def test_std_var_transposed(axis, ddof, method):
    eval_general(
        *create_test_dfs(test_data["int_data"]),
        lambda df: getattr(df.T, method)(axis=axis, ddof=ddof),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_values(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    np.testing.assert_equal(modin_df.values, pandas_df.values)
