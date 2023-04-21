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


@pytest.mark.skip(reason="Failing test")
def test_newdatain_hover_data():
    hover_dicts = [
        {"comment": ["a", "b", "c"]},
        {"comment": (1.234, 45.3455, 5666.234)},
        {"comment": [1.234, 45.3455, 5666.234]},
        {"comment": np.array([1.234, 45.3455, 5666.234])},
        {"comment": pd.Series([1.234, 45.3455, 5666.234])},
    ]
    for hover_dict in hover_dicts:
        fig = px.scatter(x=[1, 2, 3], y=[3, 4, 5], hover_data=hover_dict)
        assert (
            fig.data[0].hovertemplate
            == "x=%{x}<br>y=%{y}<br>comment=%{customdata[0]}<extra></extra>"
        )
    fig = px.scatter(
        x=[1, 2, 3], y=[3, 4, 5], hover_data={"comment": (True, ["a", "b", "c"])}
    )
    assert (
        fig.data[0].hovertemplate
        == "x=%{x}<br>y=%{y}<br>comment=%{customdata[0]}<extra></extra>"
    )
    hover_dicts = [
        {"comment": (":.1f", (1.234, 45.3455, 5666.234))},
        {"comment": (":.1f", [1.234, 45.3455, 5666.234])},
        {"comment": (":.1f", np.array([1.234, 45.3455, 5666.234]))},
        {"comment": (":.1f", pd.Series([1.234, 45.3455, 5666.234]))},
    ]
    for hover_dict in hover_dicts:
        fig = px.scatter(
            x=[1, 2, 3],
            y=[3, 4, 5],
            hover_data=hover_dict,
        )
        assert (
            fig.data[0].hovertemplate
            == "x=%{x}<br>y=%{y}<br>comment=%{customdata[0]:.1f}<extra></extra>"
        )


@pytest.mark.skip(reason="Failing test")
def test_date_in_hover():
    df = pd.DataFrame({"date": ["2015-04-04 19:31:30+1:00"], "value": [3]})
    df["date"] = pd.to_datetime(df["date"])
    fig = px.scatter(df, x="value", y="value", hover_data=["date"])
    assert str(fig.data[0].customdata[0][0]) == str(df["date"][0])
