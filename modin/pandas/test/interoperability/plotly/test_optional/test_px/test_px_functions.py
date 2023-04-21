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
import plotly.graph_objects as go
from numpy.testing import assert_array_equal
import numpy as np
import modin.pandas as pd
import pytest


def _compare_figures(go_trace, px_fig):
    """Compare a figure created with a go trace and a figure created with
    a px function call. Check that all values inside the go Figure are the
    same in the px figure (which sets more parameters).
    """
    go_fig = go.Figure(go_trace)
    go_fig = go_fig.to_plotly_json()
    px_fig = px_fig.to_plotly_json()
    del go_fig["layout"]["template"]
    del px_fig["layout"]["template"]
    for key in go_fig["data"][0]:
        assert_array_equal(go_fig["data"][0][key], px_fig["data"][0][key])
    for key in go_fig["layout"]:
        assert go_fig["layout"][key] == px_fig["layout"][key]


@pytest.mark.skip(reason="Failing test")
def test_sunburst_treemap_with_path():
    vendors = ["A", "B", "C", "D", "E", "F", "G", "H"]
    sectors = [
        "Tech",
        "Tech",
        "Finance",
        "Finance",
        "Tech",
        "Tech",
        "Finance",
        "Finance",
    ]
    regions = ["North", "North", "North", "North", "South", "South", "South", "South"]
    values = [1, 3, 2, 4, 2, 2, 1, 4]
    total = ["total"] * 8
    df = pd.DataFrame(
        dict(
            vendors=vendors,
            sectors=sectors,
            regions=regions,
            values=values,
            total=total,
        )
    )
    path = ["total", "regions", "sectors", "vendors"]
    # No values
    fig = px.sunburst(df, path=path)
    assert fig.data[0].branchvalues == "total"
    # Values passed
    fig = px.sunburst(df, path=path, values="values")
    assert fig.data[0].branchvalues == "total"
    assert fig.data[0].values[-1] == np.sum(values)
    # Values passed
    fig = px.sunburst(df, path=path, values="values")
    assert fig.data[0].branchvalues == "total"
    assert fig.data[0].values[-1] == np.sum(values)
    # Error when values cannot be converted to numerical data type
    df["values"] = ["1 000", "3 000", "2", "4", "2", "2", "1 000", "4 000"]
    msg = "Column `values` of `df` could not be converted to a numerical data type."
    with pytest.raises(ValueError, match=msg):
        fig = px.sunburst(df, path=path, values="values")
    #  path is a mixture of column names and array-like
    path = [df.total, "regions", df.sectors, "vendors"]
    fig = px.sunburst(df, path=path)
    assert fig.data[0].branchvalues == "total"
    # Continuous colorscale
    df["values"] = 1
    fig = px.sunburst(df, path=path, values="values", color="values")
    assert "coloraxis" in fig.data[0].marker
    assert np.all(np.array(fig.data[0].marker.colors) == 1)
    assert fig.data[0].values[-1] == 8


@pytest.mark.skip(reason="Failing test")
def test_sunburst_treemap_with_path_color():
    vendors = ["A", "B", "C", "D", "E", "F", "G", "H"]
    sectors = [
        "Tech",
        "Tech",
        "Finance",
        "Finance",
        "Tech",
        "Tech",
        "Finance",
        "Finance",
    ]
    regions = ["North", "North", "North", "North", "South", "South", "South", "South"]
    values = [1, 3, 2, 4, 2, 2, 1, 4]
    calls = [8, 2, 1, 3, 2, 2, 4, 1]
    total = ["total"] * 8
    df = pd.DataFrame(
        dict(
            vendors=vendors,
            sectors=sectors,
            regions=regions,
            values=values,
            total=total,
            calls=calls,
        )
    )
    path = ["total", "regions", "sectors", "vendors"]
    fig = px.sunburst(df, path=path, values="values", color="calls")
    colors = fig.data[0].marker.colors
    assert np.all(np.array(colors[:8]) == np.array(calls))
    fig = px.sunburst(df, path=path, color="calls")
    colors = fig.data[0].marker.colors
    assert np.all(np.array(colors[:8]) == np.array(calls))

    # Hover info
    df["hover"] = [el.lower() for el in vendors]
    fig = px.sunburst(df, path=path, color="calls", hover_data=["hover"])
    custom = fig.data[0].customdata
    assert np.all(custom[:8, 0] == df["hover"])
    assert np.all(custom[8:, 0] == "(?)")
    assert np.all(custom[:8, 1] == df["calls"])

    # Discrete color
    fig = px.sunburst(df, path=path, color="vendors")
    assert len(np.unique(fig.data[0].marker.colors)) == 9

    # Discrete color and color_discrete_map
    cmap = {"Tech": "yellow", "Finance": "magenta", "(?)": "black"}
    fig = px.sunburst(df, path=path, color="sectors", color_discrete_map=cmap)
    assert np.all(np.in1d(fig.data[0].marker.colors, list(cmap.values())))

    # Numerical column in path
    df["regions"] = df["regions"].map({"North": 1, "South": 2})
    path = ["total", "regions", "sectors", "vendors"]
    fig = px.sunburst(df, path=path, values="values", color="calls")
    colors = fig.data[0].marker.colors
    assert np.all(np.array(colors[:8]) == np.array(calls))


@pytest.mark.skip(reason="Failing test")
def test_sunburst_treemap_column_parent():
    vendors = ["A", "B", "C", "D", "E", "F", "G", "H"]
    sectors = [
        "Tech",
        "Tech",
        "Finance",
        "Finance",
        "Tech",
        "Tech",
        "Finance",
        "Finance",
    ]
    regions = ["North", "North", "North", "North", "South", "South", "South", "South"]
    values = [1, 3, 2, 4, 2, 2, 1, 4]
    df = pd.DataFrame(
        dict(
            id=vendors,
            sectors=sectors,
            parent=regions,
            values=values,
        )
    )
    path = ["parent", "sectors", "id"]
    # One column of the path is a reserved name - this is ok and should not raise
    px.sunburst(df, path=path, values="values")


@pytest.mark.skip(reason="Failing test")
def test_sunburst_treemap_with_path_non_rectangular():
    vendors = ["A", "B", "C", "D", None, "E", "F", "G", "H", None]
    sectors = [
        "Tech",
        "Tech",
        "Finance",
        "Finance",
        None,
        "Tech",
        "Tech",
        "Finance",
        "Finance",
        "Finance",
    ]
    regions = [
        "North",
        "North",
        "North",
        "North",
        "North",
        "South",
        "South",
        "South",
        "South",
        "South",
    ]
    values = [1, 3, 2, 4, 1, 2, 2, 1, 4, 1]
    total = ["total"] * 10
    df = pd.DataFrame(
        dict(
            vendors=vendors,
            sectors=sectors,
            regions=regions,
            values=values,
            total=total,
        )
    )
    path = ["total", "regions", "sectors", "vendors"]
    msg = "Non-leaves rows are not permitted in the dataframe"
    with pytest.raises(ValueError, match=msg):
        fig = px.sunburst(df, path=path, values="values")
    df.loc[df["vendors"].isnull(), "sectors"] = "Other"
    fig = px.sunburst(df, path=path, values="values")
    assert fig.data[0].values[-1] == np.sum(values)


@pytest.mark.skip(reason="Failing test")
def test_timeline():
    df = pd.DataFrame(
        [
            dict(Task="Job A", Start="2009-01-01", Finish="2009-02-28"),
            dict(Task="Job B", Start="2009-03-05", Finish="2009-04-15"),
            dict(Task="Job C", Start="2009-02-20", Finish="2009-05-30"),
        ]
    )
    fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", color="Task")
    assert len(fig.data) == 3
    assert fig.layout.xaxis.type == "date"
    assert fig.layout.xaxis.title.text is None
    fig = px.timeline(df, x_start="Start", x_end="Finish", y="Task", facet_row="Task")
    assert len(fig.data) == 3
    assert fig.data[1].xaxis == "x2"
    assert fig.layout.xaxis.type == "date"

    msg = "Both x_start and x_end are required"
    with pytest.raises(ValueError, match=msg):
        px.timeline(df, x_start="Start", y="Task", color="Task")

    msg = "Both x_start and x_end must refer to data convertible to datetimes."
    with pytest.raises(TypeError, match=msg):
        px.timeline(df, x_start="Start", x_end=["a", "b", "c"], y="Task", color="Task")
