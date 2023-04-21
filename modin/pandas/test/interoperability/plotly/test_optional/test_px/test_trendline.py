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

import plotly.express as px
import numpy as np
import modin.pandas as pd
import pytest
from datetime import datetime


@pytest.mark.parametrize(
    "mode,options",
    [
        ("ols", None),
        ("lowess", None),
        ("lowess", dict(frac=0.3)),
        ("rolling", dict(window=2)),
        ("expanding", None),
        ("ewm", dict(alpha=0.5)),
    ],
)
def test_trendline_results_passthrough(mode, options):
    df = px.data.gapminder().query("continent == 'Oceania'")
    fig = px.scatter(
        df,
        x="year",
        y="pop",
        color="country",
        trendline=mode,
        trendline_options=options,
    )
    assert len(fig.data) == 4
    for trace in fig["data"][0::2]:
        assert "trendline" not in trace.hovertemplate
    for trendline in fig["data"][1::2]:
        assert "trendline" in trendline.hovertemplate
        if mode == "ols":
            assert "R<sup>2</sup>" in trendline.hovertemplate
    results = px.get_trendline_results(fig)
    if mode == "ols":
        assert len(results) == 2
        assert results["country"].values[0] == "Australia"
        au_result = results["px_fit_results"].values[0]
        assert len(au_result.params) == 2
    else:
        assert len(results) == 0


@pytest.mark.parametrize(
    "mode,options",
    [
        ("ols", None),
        ("lowess", None),
        ("lowess", dict(frac=0.3)),
        ("rolling", dict(window=2)),
        ("expanding", None),
        ("ewm", dict(alpha=0.5)),
    ],
)
def test_trendline_enough_values(mode, options):
    fig = px.scatter(x=[0, 1], y=[0, 1], trendline=mode, trendline_options=options)
    assert len(fig.data) == 2
    assert len(fig.data[1].x) == 2
    fig = px.scatter(x=[0], y=[0], trendline=mode, trendline_options=options)
    assert len(fig.data) == 2
    assert fig.data[1].x is None
    fig = px.scatter(x=[0, 1], y=[0, None], trendline=mode, trendline_options=options)
    assert len(fig.data) == 2
    assert fig.data[1].x is None
    fig = px.scatter(
        x=[0, 1], y=np.array([0, np.nan]), trendline=mode, trendline_options=options
    )
    assert len(fig.data) == 2
    assert fig.data[1].x is None
    fig = px.scatter(
        x=[0, 1, None], y=[0, None, 1], trendline=mode, trendline_options=options
    )
    assert len(fig.data) == 2
    assert fig.data[1].x is None
    fig = px.scatter(
        x=np.array([0, 1, np.nan]),
        y=np.array([0, np.nan, 1]),
        trendline=mode,
        trendline_options=options,
    )
    assert len(fig.data) == 2
    assert fig.data[1].x is None
    fig = px.scatter(
        x=[0, 1, None, 2], y=[1, None, 1, 2], trendline=mode, trendline_options=options
    )
    assert len(fig.data) == 2
    assert len(fig.data[1].x) == 2
    fig = px.scatter(
        x=np.array([0, 1, np.nan, 2]),
        y=np.array([1, np.nan, 1, 2]),
        trendline=mode,
        trendline_options=options,
    )
    assert len(fig.data) == 2
    assert len(fig.data[1].x) == 2


@pytest.mark.parametrize(
    "mode,options",
    [
        ("ols", None),
        ("ols", dict(add_constant=False, log_x=True, log_y=True)),
        ("lowess", None),
        ("lowess", dict(frac=0.3)),
        ("rolling", dict(window=2)),
        ("expanding", None),
        ("ewm", dict(alpha=0.5)),
    ],
)
def test_trendline_nan_values(mode, options):
    df = px.data.gapminder().query("continent == 'Oceania'")
    start_date = 1970
    df["pop"][df["year"] < start_date] = np.nan
    fig = px.scatter(
        df,
        x="year",
        y="pop",
        color="country",
        trendline=mode,
        trendline_options=options,
    )
    for trendline in fig["data"][1::2]:
        assert trendline.x[0] >= start_date
        assert len(trendline.x) == len(trendline.y)


def test_ols_trendline_slopes():
    fig = px.scatter(x=[0, 1], y=[0, 1], trendline="ols")
    # should be "y = 1 * x + 0" but sometimes is some tiny number instead
    assert "y = 1 * x + " in fig.data[1].hovertemplate
    results = px.get_trendline_results(fig)
    params = results["px_fit_results"].iloc[0].params
    assert np.all(np.isclose(params, [0, 1]))

    fig = px.scatter(x=[0, 1], y=[1, 2], trendline="ols")
    assert "y = 1 * x + 1<br>" in fig.data[1].hovertemplate
    results = px.get_trendline_results(fig)
    params = results["px_fit_results"].iloc[0].params
    assert np.all(np.isclose(params, [1, 1]))

    fig = px.scatter(
        x=[0, 1], y=[1, 2], trendline="ols", trendline_options=dict(add_constant=False)
    )
    assert "y = 2 * x<br>" in fig.data[1].hovertemplate
    results = px.get_trendline_results(fig)
    params = results["px_fit_results"].iloc[0].params
    assert np.all(np.isclose(params, [2]))

    fig = px.scatter(
        x=[1, 1], y=[0, 0], trendline="ols", trendline_options=dict(add_constant=False)
    )
    assert "y = 0 * x<br>" in fig.data[1].hovertemplate
    results = px.get_trendline_results(fig)
    params = results["px_fit_results"].iloc[0].params
    assert np.all(np.isclose(params, [0]))

    fig = px.scatter(x=[1, 1], y=[0, 0], trendline="ols")
    assert "y = 0<br>" in fig.data[1].hovertemplate
    results = px.get_trendline_results(fig)
    params = results["px_fit_results"].iloc[0].params
    assert np.all(np.isclose(params, [0]))

    fig = px.scatter(x=[1, 2], y=[0, 0], trendline="ols")
    assert "y = 0 * x + 0<br>" in fig.data[1].hovertemplate
    fig = px.scatter(x=[0, 0], y=[1, 1], trendline="ols")
    assert "y = 0 * x + 1<br>" in fig.data[1].hovertemplate
    fig = px.scatter(x=[0, 0], y=[1, 2], trendline="ols")
    assert "y = 0 * x + 1.5<br>" in fig.data[1].hovertemplate


@pytest.mark.parametrize(
    "mode,options",
    [
        ("ols", None),
        ("lowess", None),
        ("lowess", dict(frac=0.3)),
        ("rolling", dict(window=2)),
        ("rolling", dict(window="10d")),
        ("expanding", None),
        ("ewm", dict(alpha=0.5)),
    ],
)
def test_trendline_on_timeseries(mode, options):
    df = px.data.stocks()

    with pytest.raises(ValueError) as err_msg:
        px.scatter(df, x="date", y="GOOG", trendline=mode, trendline_options=options)
    assert "Could not convert value of 'x' ('date') into a numeric type." in str(
        err_msg.value
    )

    df["date"] = pd.to_datetime(df["date"])
    df["date"] = df["date"].dt.tz_localize("CET")  # force a timezone
    fig = px.scatter(df, x="date", y="GOOG", trendline=mode, trendline_options=options)
    assert len(fig.data) == 2
    assert len(fig.data[0].x) == len(fig.data[1].x)
    assert type(fig.data[0].x[0]) == datetime
    assert type(fig.data[1].x[0]) == datetime
    assert np.all(fig.data[0].x == fig.data[1].x)
    assert str(fig.data[0].x[0]) == str(fig.data[1].x[0])


def test_overall_trendline():
    df = px.data.tips()
    fig1 = px.scatter(df, x="total_bill", y="tip", trendline="ols")
    assert len(fig1.data) == 2
    assert "trendline" in fig1.data[1].hovertemplate
    results1 = px.get_trendline_results(fig1)
    params1 = results1["px_fit_results"].iloc[0].params

    fig2 = px.scatter(
        df,
        x="total_bill",
        y="tip",
        color="sex",
        trendline="ols",
        trendline_scope="overall",
    )
    assert len(fig2.data) == 3
    assert "trendline" in fig2.data[2].hovertemplate
    results2 = px.get_trendline_results(fig2)
    params2 = results2["px_fit_results"].iloc[0].params

    assert np.all(np.array_equal(params1, params2))

    fig3 = px.scatter(
        df,
        x="total_bill",
        y="tip",
        facet_row="sex",
        trendline="ols",
        trendline_scope="overall",
    )
    assert len(fig3.data) == 4
    assert "trendline" in fig3.data[3].hovertemplate
    results3 = px.get_trendline_results(fig3)
    params3 = results3["px_fit_results"].iloc[0].params

    assert np.all(np.array_equal(params1, params3))
