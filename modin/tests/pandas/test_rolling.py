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

import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest

import modin.pandas as pd
from modin.config import NPartitions

from .utils import (
    create_test_dfs,
    create_test_series,
    default_to_pandas_ignore_string,
    df_equals,
    eval_general,
    test_data_keys,
    test_data_values,
)

NPartitions.put(4)

# Our configuration in pytest.ini requires that we explicitly catch all
# instances of defaulting to pandas, but some test modules, like this one,
# have too many such instances.
# TODO(https://github.com/modin-project/modin/issues/3655): catch all instances
# of defaulting to pandas.
pytestmark = [
    pytest.mark.filterwarnings(default_to_pandas_ignore_string),
    # TO MAKE SURE ALL FUTUREWARNINGS ARE CONSIDERED
    pytest.mark.filterwarnings("error::FutureWarning"),
    # IGNORE FUTUREWARNINGS MARKS TO CLEANUP OUTPUT
    pytest.mark.filterwarnings(
        "ignore:Support for axis=1 in DataFrame.rolling is deprecated:FutureWarning"
    ),
    # FIXME: these cases inconsistent between modin and pandas
    pytest.mark.filterwarnings(
        "ignore:.*In a future version of pandas, the provided callable will be used directly.*:FutureWarning"
    ),
]


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("window", [5, 100])
@pytest.mark.parametrize("min_periods", [None, 5])
@pytest.mark.parametrize("axis", [lib.no_default, 1])
@pytest.mark.parametrize(
    "method, kwargs",
    [
        ("count", {}),
        ("sum", {}),
        ("mean", {}),
        ("var", {"ddof": 0}),
        ("std", {"ddof": 0}),
        ("min", {}),
        ("max", {}),
        ("skew", {}),
        ("kurt", {}),
        ("apply", {"func": np.sum}),
        ("rank", {}),
        ("sem", {"ddof": 0}),
        ("quantile", {"q": 0.1}),
        ("median", {}),
    ],
)
def test_dataframe_rolling(data, window, min_periods, axis, method, kwargs):
    # Testing of Rolling class
    modin_df, pandas_df = create_test_dfs(data)
    if window > len(pandas_df):
        window = len(pandas_df)
    eval_general(
        modin_df,
        pandas_df,
        lambda df: getattr(
            df.rolling(
                window=window,
                min_periods=min_periods,
                win_type=None,
                center=True,
                axis=axis,
            ),
            method,
        )(**kwargs),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("window", [5, 100])
@pytest.mark.parametrize("min_periods", [None, 5])
@pytest.mark.parametrize("axis", [lib.no_default, 1])
def test_dataframe_agg(data, window, min_periods, axis):
    modin_df, pandas_df = create_test_dfs(data)
    if window > len(pandas_df):
        window = len(pandas_df)
    modin_rolled = modin_df.rolling(
        window=window, min_periods=min_periods, win_type=None, center=True, axis=axis
    )
    pandas_rolled = pandas_df.rolling(
        window=window, min_periods=min_periods, win_type=None, center=True, axis=axis
    )
    df_equals(pandas_rolled.aggregate(np.sum), modin_rolled.aggregate(np.sum))
    # TODO(https://github.com/modin-project/modin/issues/4260): Once pandas
    # allows us to rolling aggregate a list of functions over axis 1, test
    # that, too.
    if axis != 1:
        df_equals(
            pandas_rolled.aggregate([np.sum, np.mean]),
            modin_rolled.aggregate([np.sum, np.mean]),
        )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("window", [5, 100])
@pytest.mark.parametrize("min_periods", [None, 5])
@pytest.mark.parametrize("axis", [lib.no_default, 1])
@pytest.mark.parametrize(
    "method, kwargs",
    [
        ("sum", {}),
        ("mean", {}),
        ("var", {"ddof": 0}),
        ("std", {"ddof": 0}),
    ],
)
def test_dataframe_window(data, window, min_periods, axis, method, kwargs):
    # Testing of Window class
    modin_df, pandas_df = create_test_dfs(data)
    if window > len(pandas_df):
        window = len(pandas_df)
    eval_general(
        modin_df,
        pandas_df,
        lambda df: getattr(
            df.rolling(
                window=window,
                min_periods=min_periods,
                win_type="triang",
                center=True,
                axis=axis,
            ),
            method,
        )(**kwargs),
    )


@pytest.mark.parametrize("axis", [lib.no_default, "columns"])
@pytest.mark.parametrize("on", [None, "DateCol"])
@pytest.mark.parametrize("closed", ["both", "right"])
@pytest.mark.parametrize("window", [3, "3s"])
def test_dataframe_dt_index(axis, on, closed, window):
    index = pandas.date_range("31/12/2000", periods=12, freq="min")
    data = {"A": range(12), "B": range(12)}
    pandas_df = pandas.DataFrame(data, index=index)
    modin_df = pd.DataFrame(data, index=index)
    if on is not None and axis == lib.no_default and isinstance(window, str):
        pandas_df[on] = pandas.date_range("22/06/1941", periods=12, freq="min")
        modin_df[on] = pd.date_range("22/06/1941", periods=12, freq="min")
    else:
        on = None
    if axis == "columns":
        pandas_df = pandas_df.T
        modin_df = modin_df.T
    pandas_rolled = pandas_df.rolling(window=window, on=on, axis=axis, closed=closed)
    modin_rolled = modin_df.rolling(window=window, on=on, axis=axis, closed=closed)
    if isinstance(window, int):
        # This functions are very slowly for data from test_rolling
        df_equals(
            modin_rolled.corr(modin_df, True), pandas_rolled.corr(pandas_df, True)
        )
        df_equals(
            modin_rolled.corr(modin_df, False), pandas_rolled.corr(pandas_df, False)
        )
        df_equals(modin_rolled.cov(modin_df, True), pandas_rolled.cov(pandas_df, True))
        df_equals(
            modin_rolled.cov(modin_df, False), pandas_rolled.cov(pandas_df, False)
        )
        if axis == lib.no_default:
            df_equals(
                modin_rolled.cov(modin_df[modin_df.columns[0]], True),
                pandas_rolled.cov(pandas_df[pandas_df.columns[0]], True),
            )
            df_equals(
                modin_rolled.corr(modin_df[modin_df.columns[0]], True),
                pandas_rolled.corr(pandas_df[pandas_df.columns[0]], True),
            )
    else:
        df_equals(modin_rolled.count(), pandas_rolled.count())
        df_equals(modin_rolled.skew(), pandas_rolled.skew())
        df_equals(
            modin_rolled.apply(np.sum, raw=True),
            pandas_rolled.apply(np.sum, raw=True),
        )
        df_equals(modin_rolled.aggregate(np.sum), pandas_rolled.aggregate(np.sum))
        df_equals(modin_rolled.quantile(0.1), pandas_rolled.quantile(0.1))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("window", [5, 100])
@pytest.mark.parametrize("min_periods", [None, 5])
@pytest.mark.parametrize(
    "method, kwargs",
    [
        ("count", {}),
        ("sum", {}),
        ("mean", {}),
        ("var", {"ddof": 0}),
        ("std", {"ddof": 0}),
        ("min", {}),
        ("max", {}),
        ("skew", {}),
        ("kurt", {}),
        ("apply", {"func": np.sum}),
        ("rank", {}),
        ("sem", {"ddof": 0}),
        ("aggregate", {"func": np.sum}),
        ("agg", {"func": [np.sum, np.mean]}),
        ("quantile", {"q": 0.1}),
        ("median", {}),
    ],
)
def test_series_rolling(data, window, min_periods, method, kwargs):
    # Test of Rolling class
    modin_series, pandas_series = create_test_series(data)
    if window > len(pandas_series):
        window = len(pandas_series)
    eval_general(
        modin_series,
        pandas_series,
        lambda series: getattr(
            series.rolling(
                window=window,
                min_periods=min_periods,
                win_type=None,
                center=True,
            ),
            method,
        )(**kwargs),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("window", [5, 100])
@pytest.mark.parametrize("min_periods", [None, 5])
def test_series_corr_cov(data, window, min_periods):
    modin_series, pandas_series = create_test_series(data)
    if window > len(pandas_series):
        window = len(pandas_series)
    modin_rolled = modin_series.rolling(
        window=window, min_periods=min_periods, win_type=None, center=True
    )
    pandas_rolled = pandas_series.rolling(
        window=window, min_periods=min_periods, win_type=None, center=True
    )
    df_equals(modin_rolled.corr(modin_series), pandas_rolled.corr(pandas_series))
    df_equals(
        modin_rolled.cov(modin_series, True), pandas_rolled.cov(pandas_series, True)
    )
    df_equals(
        modin_rolled.cov(modin_series, False), pandas_rolled.cov(pandas_series, False)
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("window", [5, 100])
@pytest.mark.parametrize("min_periods", [None, 5])
@pytest.mark.parametrize(
    "method, kwargs",
    [
        ("sum", {}),
        ("mean", {}),
        ("var", {"ddof": 0}),
        ("std", {"ddof": 0}),
    ],
)
def test_series_window(data, window, min_periods, method, kwargs):
    # Test of Window class
    modin_series, pandas_series = create_test_series(data)
    if window > len(pandas_series):
        window = len(pandas_series)
    eval_general(
        modin_series,
        pandas_series,
        lambda series: getattr(
            series.rolling(
                window=window,
                min_periods=min_periods,
                win_type="triang",
                center=True,
            ),
            method,
        )(**kwargs),
    )


@pytest.mark.parametrize("closed", ["both", "right"])
def test_series_dt_index(closed):
    index = pandas.date_range("1/1/2000", periods=12, freq="min")
    pandas_series = pandas.Series(range(12), index=index)
    modin_series = pd.Series(range(12), index=index)

    pandas_rolled = pandas_series.rolling("3s", closed=closed)
    modin_rolled = modin_series.rolling("3s", closed=closed)
    df_equals(modin_rolled.count(), pandas_rolled.count())
    df_equals(modin_rolled.skew(), pandas_rolled.skew())
    df_equals(
        modin_rolled.apply(np.sum, raw=True), pandas_rolled.apply(np.sum, raw=True)
    )
    df_equals(modin_rolled.aggregate(np.sum), pandas_rolled.aggregate(np.sum))
    df_equals(modin_rolled.quantile(0.1), pandas_rolled.quantile(0.1))


def test_api_indexer():
    modin_df, pandas_df = create_test_dfs(test_data_values[0])
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=3)
    pandas_rolled = pandas_df.rolling(window=indexer)
    modin_rolled = modin_df.rolling(window=indexer)
    df_equals(modin_rolled.sum(), pandas_rolled.sum())


def test_issue_3512():
    data = np.random.rand(129)
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    modin_ans = modin_df[0:33].rolling(window=21).mean()
    pandas_ans = pandas_df[0:33].rolling(window=21).mean()

    df_equals(modin_ans, pandas_ans)


### TEST ROLLING WARNINGS ###


def test_rolling_axis_1_depr():
    index = pandas.date_range("31/12/2000", periods=12, freq="min")
    data = {"A": range(12), "B": range(12)}
    modin_df = pd.DataFrame(data, index=index)
    with pytest.warns(
        FutureWarning,
        match="Support for axis=1 in DataFrame.rolling is deprecated",
    ):
        modin_df.rolling(window=3, axis=1)
