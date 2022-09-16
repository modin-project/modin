"""
Period benchmarks with non-tslibs dependencies.  See
benchmarks.tslibs.period for benchmarks that rely only on tslibs.
"""
from modin.pandas import (
    DataFrame,
    Period,
    PeriodIndex,
    Series,
    period_range,
)


class DataFramePeriodColumn:
    def setup(self):
        self.rng = period_range(start="1/1/1990", freq="S", periods=20000)
        self.df = DataFrame(index=range(len(self.rng)))

    def time_setitem_period_column(self):
        self.df["col"] = self.rng

    def time_set_index(self):
        # GH#21582 limited by comparisons of Period objects
        self.df["col2"] = self.rng
        self.df.set_index("col2", append=True)


class Algorithms:

    params = ["series"]
    param_names = ["typ"]

    def setup(self, typ):
        data = [
            Period("2011-01", freq="M"),
            Period("2011-02", freq="M"),
            Period("2011-03", freq="M"),
            Period("2011-04", freq="M"),
        ]

        self.vector = Series(data * 1000)

    def time_drop_duplicates(self, typ):
        self.vector.drop_duplicates()

    def time_value_counts(self, typ):
        self.vector.value_counts()


class Indexing:
    def setup(self):
        self.index = period_range(start="1985", periods=1000, freq="D")
        self.series = Series(range(1000), index=self.index)
        self.period = self.index[500]

    def time_series_loc(self):
        self.series.loc[self.period]

    def time_align(self):
        DataFrame({"a": self.series, "b": self.series[:500]})
