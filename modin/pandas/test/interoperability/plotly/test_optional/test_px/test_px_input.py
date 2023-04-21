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
import numpy as np
import modin.pandas as pd
import pytest
from plotly.express._core import build_dataframe
from pandas.testing import assert_frame_equal


def test_pandas_series():
    tips = px.data.tips()
    before_tip = tips.total_bill - tips.tip
    fig = px.bar(tips, x="day", y=before_tip)
    assert fig.data[0].hovertemplate == "day=%{x}<br>y=%{y}<extra></extra>"
    fig = px.bar(tips, x="day", y=before_tip, labels={"y": "bill"})
    assert fig.data[0].hovertemplate == "day=%{x}<br>bill=%{y}<extra></extra>"
    # lock down that we can pass df.col to facet_*
    fig = px.bar(tips, x="day", y="tip", facet_row=tips.day, facet_col=tips.day)
    assert fig.data[0].hovertemplate == "day=%{x}<br>tip=%{y}<extra></extra>"


@pytest.mark.skip(reason="Failing test")
def test_several_dataframes():
    df = pd.DataFrame(dict(x=[0, 1], y=[1, 10], z=[0.1, 0.8]))
    df2 = pd.DataFrame(dict(time=[23, 26], money=[100, 200]))
    fig = px.scatter(df, x="z", y=df2.money, size="x")
    assert (
        fig.data[0].hovertemplate
        == "z=%{x}<br>y=%{y}<br>x=%{marker.size}<extra></extra>"
    )
    fig = px.scatter(df2, x=df.z, y=df2.money, size=df.z)
    assert (
        fig.data[0].hovertemplate
        == "x=%{x}<br>money=%{y}<br>size=%{marker.size}<extra></extra>"
    )
    # Name conflict
    with pytest.raises(NameError) as err_msg:
        fig = px.scatter(df, x="z", y=df2.money, size="y")
    assert "A name conflict was encountered for argument 'y'" in str(err_msg.value)
    with pytest.raises(NameError) as err_msg:
        fig = px.scatter(df, x="z", y=df2.money, size=df.y)
    assert "A name conflict was encountered for argument 'y'" in str(err_msg.value)

    # No conflict when the dataframe is not given, fields are used
    df = pd.DataFrame(dict(x=[0, 1], y=[3, 4]))
    df2 = pd.DataFrame(dict(x=[3, 5], y=[23, 24]))
    fig = px.scatter(x=df.y, y=df2.y)
    assert np.all(fig.data[0].x == np.array([3, 4]))
    assert np.all(fig.data[0].y == np.array([23, 24]))
    assert fig.data[0].hovertemplate == "x=%{x}<br>y=%{y}<extra></extra>"

    df = pd.DataFrame(dict(x=[0, 1], y=[3, 4]))
    df2 = pd.DataFrame(dict(x=[3, 5], y=[23, 24]))
    df3 = pd.DataFrame(dict(y=[0.1, 0.2]))
    fig = px.scatter(x=df.y, y=df2.y, size=df3.y)
    assert np.all(fig.data[0].x == np.array([3, 4]))
    assert np.all(fig.data[0].y == np.array([23, 24]))
    assert (
        fig.data[0].hovertemplate
        == "x=%{x}<br>y=%{y}<br>size=%{marker.size}<extra></extra>"
    )

    df = pd.DataFrame(dict(x=[0, 1], y=[3, 4]))
    df2 = pd.DataFrame(dict(x=[3, 5], y=[23, 24]))
    df3 = pd.DataFrame(dict(y=[0.1, 0.2]))
    fig = px.scatter(x=df.y, y=df2.y, hover_data=[df3.y])
    assert np.all(fig.data[0].x == np.array([3, 4]))
    assert np.all(fig.data[0].y == np.array([23, 24]))
    assert (
        fig.data[0].hovertemplate
        == "x=%{x}<br>y=%{y}<br>hover_data_0=%{customdata[0]}<extra></extra>"
    )


@pytest.mark.skip(reason="Failing test")
def test_name_heuristics():
    df = pd.DataFrame(dict(x=[0, 1], y=[3, 4], z=[0.1, 0.2]))
    fig = px.scatter(df, x=df.y, y=df.x, size=df.y)
    assert np.all(fig.data[0].x == np.array([3, 4]))
    assert np.all(fig.data[0].y == np.array([0, 1]))
    assert fig.data[0].hovertemplate == "y=%{marker.size}<br>x=%{y}<extra></extra>"


def test_arrayattrable_numpy():
    tips = px.data.tips()
    fig = px.scatter(
        tips, x="total_bill", y="tip", hover_data=[np.random.random(tips.shape[0])]
    )
    assert (
        fig.data[0]["hovertemplate"]
        == "total_bill=%{x}<br>tip=%{y}<br>hover_data_0=%{customdata[0]}<extra></extra>"
    )
    tips = px.data.tips()
    fig = px.scatter(
        tips,
        x="total_bill",
        y="tip",
        hover_data=[np.random.random(tips.shape[0])],
        labels={"hover_data_0": "suppl"},
    )
    assert (
        fig.data[0]["hovertemplate"]
        == "total_bill=%{x}<br>tip=%{y}<br>suppl=%{customdata[0]}<extra></extra>"
    )


@pytest.mark.skip(reason="Failing test")
def test_wrong_dimensions_mixed_case():
    with pytest.raises(ValueError) as err_msg:
        df = pd.DataFrame(dict(time=[1, 2, 3], temperature=[20, 30, 25]))
        px.scatter(df, x="time", y="temperature", color=[1, 3, 9, 5])
    assert "All arguments should have the same length." in str(err_msg.value)


@pytest.mark.skip(reason="Failing test")
def test_multiindex_raise_error():
    index = pd.MultiIndex.from_product(
        [[1, 2, 3], ["a", "b"]], names=["first", "second"]
    )
    df = pd.DataFrame(np.random.random((6, 3)), index=index, columns=["A", "B", "C"])
    # This is ok
    px.scatter(df, x="A", y="B")
    with pytest.raises(TypeError) as err_msg:
        px.scatter(df, x=df.index, y="B")
    assert "pandas MultiIndex is not supported by plotly express" in str(err_msg.value)


def test_build_df_from_lists():
    # Just lists
    args = dict(x=[1, 2, 3], y=[2, 3, 4], color=[1, 3, 9])
    output = {key: key for key in args}
    df = pd.DataFrame(args)
    args["data_frame"] = None
    out = build_dataframe(args, go.Scatter)
    assert_frame_equal(
        df.sort_index(axis=1)._to_pandas(), out["data_frame"].sort_index(axis=1)
    )
    out.pop("data_frame")
    assert out == output

    # Arrays
    args = dict(x=np.array([1, 2, 3]), y=np.array([2, 3, 4]), color=[1, 3, 9])
    output = {key: key for key in args}
    df = pd.DataFrame(args)
    args["data_frame"] = None
    out = build_dataframe(args, go.Scatter)
    assert_frame_equal(
        df.sort_index(axis=1)._to_pandas(), out["data_frame"].sort_index(axis=1)
    )
    out.pop("data_frame")
    assert out == output


@pytest.mark.skip(reason="Failing test")
def test_timezones():
    df = pd.DataFrame({"date": ["2015-04-04 19:31:30+1:00"], "value": [3]})
    df["date"] = pd.to_datetime(df["date"])
    args = dict(data_frame=df, x="date", y="value")
    out = build_dataframe(args, go.Scatter)
    assert str(out["data_frame"]["date"][0]) == str(df["date"][0])


@pytest.mark.skip(reason="Failing test")
def test_non_matching_index():
    df = pd.DataFrame(dict(y=[1, 2, 3]), index=["a", "b", "c"])

    expected = pd.DataFrame(dict(index=["a", "b", "c"], y=[1, 2, 3]))

    args = dict(data_frame=df, x=df.index, y="y")
    out = build_dataframe(args, go.Scatter)
    assert_frame_equal(expected._to_pandas(), out["data_frame"])

    expected = pd.DataFrame(dict(x=["a", "b", "c"], y=[1, 2, 3]))

    args = dict(data_frame=None, x=df.index, y=df.y)

    # args = dict(data_frame=None, x=df.index, y=[1, 2, 3])
    out = build_dataframe(args, go.Scatter)
    assert_frame_equal(expected._to_pandas(), out["data_frame"])

    # args = dict(data_frame=None, x=["a", "b", "c"], y=df.y)
    # out = build_dataframe(args, go.Scatter)
    # assert_frame_equal(expected._to_pandas(), out["data_frame"])


def test_int_col_names():
    # DataFrame with int column names
    lengths = pd.DataFrame(np.random.random(100))
    fig = px.histogram(lengths, x=0)
    assert np.all(np.array(lengths).flatten() == fig.data[0].x)
    # Numpy array
    ar = np.arange(100).reshape((10, 10))
    fig = px.scatter(ar, x=2, y=8)
    assert np.all(fig.data[0].x == ar[:, 2])


@pytest.mark.skip(reason="Failing test")
@pytest.mark.parametrize(
    "fn,mode", [(px.violin, "violinmode"), (px.box, "boxmode"), (px.strip, "boxmode")]
)
@pytest.mark.parametrize(
    "x,y,color,result",
    [
        ("categorical1", "numerical", None, "group"),
        ("categorical1", "numerical", "categorical2", "group"),
        ("categorical1", "numerical", "categorical1", "overlay"),
        ("numerical", "categorical1", None, "group"),
        ("numerical", "categorical1", "categorical2", "group"),
        ("numerical", "categorical1", "categorical1", "overlay"),
    ],
)
def test_auto_boxlike_overlay(fn, mode, x, y, color, result):
    df = pd.DataFrame(
        dict(
            categorical1=["a", "a", "b", "b"],
            categorical2=["a", "a", "b", "b"],
            numerical=[1, 2, 3, 4],
        )
    )
    assert fn(df, x=x, y=y, color=color).layout[mode] == result


@pytest.mark.skip(reason="Failing test")
@pytest.mark.parametrize("fn", [px.scatter, px.line, px.area, px.bar])
def test_x_or_y(fn):
    categorical = ["a", "a", "b", "b"]
    numerical = [1, 2, 3, 4]
    constant = [1, 1, 1, 1]
    range_4 = [0, 1, 2, 3]
    index = [11, 12, 13, 14]
    numerical_df = pd.DataFrame(dict(col=numerical), index=index)
    categorical_df = pd.DataFrame(dict(col=categorical), index=index)

    fig = fn(x=numerical)
    assert list(fig.data[0].x) == numerical
    assert list(fig.data[0].y) == range_4
    assert fig.data[0].orientation == "h"
    fig = fn(y=numerical)
    assert list(fig.data[0].x) == range_4
    assert list(fig.data[0].y) == numerical
    assert fig.data[0].orientation == "v"
    fig = fn(numerical_df, x="col")
    assert list(fig.data[0].x) == numerical
    assert list(fig.data[0].y) == index
    assert fig.data[0].orientation == "h"
    fig = fn(numerical_df, y="col")
    assert list(fig.data[0].x) == index
    assert list(fig.data[0].y) == numerical
    assert fig.data[0].orientation == "v"

    if fn != px.bar:
        fig = fn(x=categorical)
        assert list(fig.data[0].x) == categorical
        assert list(fig.data[0].y) == range_4
        assert fig.data[0].orientation == "h"
        fig = fn(y=categorical)
        assert list(fig.data[0].x) == range_4
        assert list(fig.data[0].y) == categorical
        assert fig.data[0].orientation == "v"
        fig = fn(categorical_df, x="col")
        assert list(fig.data[0].x) == categorical
        assert list(fig.data[0].y) == index
        assert fig.data[0].orientation == "h"
        fig = fn(categorical_df, y="col")
        assert list(fig.data[0].x) == index
        assert list(fig.data[0].y) == categorical
        assert fig.data[0].orientation == "v"

    else:
        fig = fn(x=categorical)
        assert list(fig.data[0].x) == categorical
        assert list(fig.data[0].y) == constant
        assert fig.data[0].orientation == "v"
        fig = fn(y=categorical)
        assert list(fig.data[0].x) == constant
        assert list(fig.data[0].y) == categorical
        assert fig.data[0].orientation == "h"
        fig = fn(categorical_df, x="col")
        assert list(fig.data[0].x) == categorical
        assert list(fig.data[0].y) == constant
        assert fig.data[0].orientation == "v"
        fig = fn(categorical_df, y="col")
        assert list(fig.data[0].x) == constant
        assert list(fig.data[0].y) == categorical
        assert fig.data[0].orientation == "h"
