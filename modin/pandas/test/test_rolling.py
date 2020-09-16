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
import modin.pandas as pd

from .utils import df_equals, test_data_values, test_data_keys

pd.DEFAULT_NPARTITIONS = 4


def create_test_series(vals):
    if isinstance(vals, dict):
        modin_series = pd.Series(vals[next(iter(vals.keys()))])
        pandas_series = pandas.Series(vals[next(iter(vals.keys()))])
    else:
        modin_series = pd.Series(vals)
        pandas_series = pandas.Series(vals)
    return modin_series, pandas_series


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("window", [5, 100])
@pytest.mark.parametrize("min_periods", [None, 5])
@pytest.mark.parametrize("win_type", [None, "triang"])
def test_dataframe(data, window, min_periods, win_type):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    if window > len(pandas_df):
        window = len(pandas_df)
    pandas_rolled = pandas_df.rolling(
        window=window,
        min_periods=min_periods,
        win_type=win_type,
        center=True,
    )
    modin_rolled = modin_df.rolling(
        window=window,
        min_periods=min_periods,
        win_type=win_type,
        center=True,
    )
    # Testing of Window class
    if win_type is not None:
        df_equals(modin_rolled.mean(), pandas_rolled.mean())
        df_equals(modin_rolled.sum(), pandas_rolled.sum())
        df_equals(modin_rolled.var(ddof=0), pandas_rolled.var(ddof=0))
        df_equals(modin_rolled.std(ddof=0), pandas_rolled.std(ddof=0))
    # Testing of Rolling class
    else:
        df_equals(modin_rolled.count(), pandas_rolled.count())
        df_equals(modin_rolled.sum(), pandas_rolled.sum())
        df_equals(modin_rolled.mean(), pandas_rolled.mean())
        df_equals(modin_rolled.median(), pandas_rolled.median())
        df_equals(modin_rolled.var(ddof=0), pandas_rolled.var(ddof=0))
        df_equals(modin_rolled.std(ddof=0), pandas_rolled.std(ddof=0))
        df_equals(modin_rolled.min(), pandas_rolled.min())
        df_equals(modin_rolled.max(), pandas_rolled.max())
        df_equals(modin_rolled.skew(), pandas_rolled.skew())
        df_equals(modin_rolled.kurt(), pandas_rolled.kurt())
        df_equals(modin_rolled.apply(np.sum), pandas_rolled.apply(np.sum))
        df_equals(modin_rolled.aggregate(np.sum), pandas_rolled.aggregate(np.sum))
        df_equals(
            modin_rolled.aggregate([np.sum, np.mean]),
            pandas_rolled.aggregate([np.sum, np.mean]),
        )
        df_equals(modin_rolled.quantile(0.1), pandas_rolled.quantile(0.1))


@pytest.mark.parametrize("axis", [0, "columns"])
@pytest.mark.parametrize("on", [None, "DateCol"])
@pytest.mark.parametrize("closed", ["both", "right"])
@pytest.mark.parametrize("window", [3, "3s"])
def test_dataframe_dt_index(axis, on, closed, window):
    index = pandas.date_range("31/12/2000", periods=12, freq="T")
    data = {"A": range(12), "B": range(12)}
    pandas_df = pandas.DataFrame(data, index=index)
    modin_df = pd.DataFrame(data, index=index)
    if on is not None and axis == 0 and isinstance(window, str):
        pandas_df[on] = pandas.date_range("22/06/1941", periods=12, freq="T")
        modin_df[on] = pd.date_range("22/06/1941", periods=12, freq="T")
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
        if axis == 0:
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
@pytest.mark.parametrize("win_type", [None, "triang"])
def test_series(data, window, min_periods, win_type):
    modin_series, pandas_series = create_test_series(data)
    if window > len(pandas_series):
        window = len(pandas_series)
    pandas_rolled = pandas_series.rolling(
        window=window,
        min_periods=min_periods,
        win_type=win_type,
        center=True,
    )
    modin_rolled = modin_series.rolling(
        window=window,
        min_periods=min_periods,
        win_type=win_type,
        center=True,
    )
    # Testing of Window class
    if win_type is not None:
        df_equals(modin_rolled.mean(), pandas_rolled.mean())
        df_equals(modin_rolled.sum(), pandas_rolled.sum())
        df_equals(modin_rolled.var(ddof=0), pandas_rolled.var(ddof=0))
        df_equals(modin_rolled.std(ddof=0), pandas_rolled.std(ddof=0))
    # Testing of Rolling class
    else:
        df_equals(modin_rolled.count(), pandas_rolled.count())
        df_equals(modin_rolled.sum(), pandas_rolled.sum())
        df_equals(modin_rolled.mean(), pandas_rolled.mean())
        df_equals(modin_rolled.median(), pandas_rolled.median())
        df_equals(modin_rolled.var(ddof=0), pandas_rolled.var(ddof=0))
        df_equals(modin_rolled.std(ddof=0), pandas_rolled.std(ddof=0))
        df_equals(modin_rolled.min(), pandas_rolled.min())
        df_equals(modin_rolled.max(), pandas_rolled.max())
        df_equals(
            modin_rolled.corr(modin_series),
            pandas_rolled.corr(pandas_series),
        )
        df_equals(
            modin_rolled.cov(modin_series, True), pandas_rolled.cov(pandas_series, True)
        )
        df_equals(
            modin_rolled.cov(modin_series, False),
            pandas_rolled.cov(pandas_series, False),
        )
        df_equals(modin_rolled.skew(), pandas_rolled.skew())
        df_equals(modin_rolled.kurt(), pandas_rolled.kurt())
        df_equals(modin_rolled.apply(np.sum), pandas_rolled.apply(np.sum))
        df_equals(modin_rolled.aggregate(np.sum), pandas_rolled.aggregate(np.sum))
        df_equals(
            modin_rolled.agg([np.sum, np.mean]),
            pandas_rolled.agg([np.sum, np.mean]),
        )
        df_equals(modin_rolled.quantile(0.1), pandas_rolled.quantile(0.1))


@pytest.mark.parametrize("closed", ["both", "right"])
def test_series_dt_index(closed):
    index = pandas.date_range("1/1/2000", periods=12, freq="T")
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
