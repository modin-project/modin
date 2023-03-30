import numpy as np

import matplotlib.pyplot as plt

import modin.pandas as pd

from matplotlib.testing.decorators import (
    check_figures_equal,
)

# Note: Some test cases are run twice: once normally and once with labeled data
#       These two must be defined in the same test function or need to have
#       different baseline images to prevent race conditions when pytest runs
#       the tests with multiple threads.


@check_figures_equal(extensions=["png"])
def test_invisible_axes(fig_test, fig_ref):
    ax = fig_test.subplots()
    ax.set_visible(False)


def test_boxplot_dates_pandas():
    # import modin.pandas as pd

    # smoke test for boxplot and dates in pandas
    data = np.random.rand(5, 2)
    years = pd.date_range("1/1/2000", periods=2, freq=pd.DateOffset(years=1)).year
    plt.figure()
    plt.boxplot(data, positions=years)


def test_bar_pandas():
    # Smoke test for pandas
    df = pd.DataFrame(
        {
            "year": [2018, 2018, 2018],
            "month": [1, 1, 1],
            "day": [1, 2, 3],
            "value": [1, 2, 3],
        }
    )
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])

    monthly = df[["date", "value"]].groupby(["date"]).sum()
    dates = monthly.index
    forecast = monthly["value"]
    baseline = monthly["value"]

    fig, ax = plt.subplots()
    ax.bar(dates, forecast, width=10, align="center")
    ax.plot(dates, baseline, color="orange", lw=4)


def test_bar_pandas_indexed():
    # Smoke test for indexed pandas
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "width": [0.2, 0.4, 0.6]}, index=[1, 2, 3])
    fig, ax = plt.subplots()
    ax.bar(df.x, 1.0, width=df.width)


def test_pandas_minimal_plot():
    # smoke test that series and index objects do not warn
    for x in [pd.Series([1, 2], dtype="float64"), pd.Series([1, 2], dtype="Float64")]:
        plt.plot(x, x)
        plt.plot(x.index, x)
        plt.plot(x)
        plt.plot(x.index)
    df = pd.DataFrame({"col": [1, 2, 3]})
    plt.plot(df)
    plt.plot(df, df)


@check_figures_equal(extensions=["png"])
def test_violinplot_pandas_series(fig_test, fig_ref):
    np.random.seed(110433579)
    s1 = pd.Series(np.random.normal(size=7), index=[9, 8, 7, 6, 5, 4, 3])
    s2 = pd.Series(np.random.normal(size=9), index=list("ABCDEFGHI"))
    s3 = pd.Series(np.random.normal(size=11))
    fig_test.subplots().violinplot([s1, s2, s3])
    fig_ref.subplots().violinplot([s1.values, s2.values, s3.values])


def test_pandas_pcolormesh():
    time = pd.date_range("2000-01-01", periods=10)
    depth = np.arange(20)
    data = np.random.rand(19, 9)

    fig, ax = plt.subplots()
    ax.pcolormesh(time, depth, data)


def test_pandas_indexing_dates():
    dates = np.arange("2005-02", "2005-03", dtype="datetime64[D]")
    values = np.sin(range(len(dates)))
    df = pd.DataFrame({"dates": dates, "values": values})

    ax = plt.gca()

    without_zero_index = df[np.array(df.index) % 2 == 1].copy()
    ax.plot("dates", "values", data=without_zero_index)


def test_pandas_errorbar_indexing():
    df = pd.DataFrame(
        np.random.uniform(size=(5, 4)),
        columns=["x", "y", "xe", "ye"],
        index=[1, 2, 3, 4, 5],
    )
    fig, ax = plt.subplots()
    ax.errorbar("x", "y", xerr="xe", yerr="ye", data=df)


def test_pandas_index_shape():
    df = pd.DataFrame({"XX": [4, 5, 6], "YY": [7, 1, 2]})
    fig, ax = plt.subplots()
    ax.plot(df.index, df["YY"])


def test_pandas_indexing_hist():
    ser_1 = pd.Series(data=[1, 2, 2, 3, 3, 4, 4, 4, 4, 5])
    ser_2 = ser_1.iloc[1:]
    fig, ax = plt.subplots()
    ax.hist(ser_2)


def test_pandas_bar_align_center():
    # Tests fix for issue 8767
    df = pd.DataFrame({"a": range(2), "b": range(2)})

    fig, ax = plt.subplots(1)

    ax.bar(df.loc[df["a"] == 1, "b"], df.loc[df["a"] == 1, "b"], align="center")

    fig.canvas.draw()


def test_scatter_series_non_zero_index():
    # create non-zero index
    ids = range(10, 18)
    x = pd.Series(np.random.uniform(size=8), index=ids)
    y = pd.Series(np.random.uniform(size=8), index=ids)
    c = pd.Series([1, 1, 1, 1, 1, 0, 0, 0], index=ids)
    plt.scatter(x, y, c)
