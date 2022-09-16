from datetime import timedelta

import dateutil
import numpy as np

from modin.pandas import (
    DataFrame,
    Series,
    date_range,
    period_range,
    timedelta_range,
)


class ResetIndex:

    params = [None, "US/Eastern"]
    param_names = "tz"

    def setup(self, tz):
        idx = date_range(start="1/1/2000", periods=1000, freq="H", tz=tz)
        self.df = DataFrame(np.random.randn(1000, 2), index=idx)

    def time_reset_datetimeindex(self, tz):
        self.df.reset_index()


class AsOf:

    params = ["DataFrame", "Series"]
    param_names = ["constructor"]

    def setup(self, constructor):
        N = 10000
        M = 10
        rng = date_range(start="1/1/1990", periods=N, freq="53s")
        data = {
            "DataFrame": DataFrame(np.random.randn(N, M)),
            "Series": Series(np.random.randn(N)),
        }
        self.ts = data[constructor]
        self.ts.index = rng
        self.ts2 = self.ts.copy()
        self.ts2.iloc[250:5000] = np.nan
        self.ts3 = self.ts.copy()
        self.ts3.iloc[-5000:] = np.nan
        self.dates = date_range(start="1/1/1990", periods=N * 10, freq="5s")
        self.date = self.dates[0]
        self.date_last = self.dates[-1]
        self.date_early = self.date - timedelta(10)

    # test speed of pre-computing NAs.
    def time_asof(self, constructor):
        self.ts.asof(self.dates)

    # should be roughly the same as above.
    def time_asof_nan(self, constructor):
        self.ts2.asof(self.dates)

    # test speed of the code path for a scalar index
    # without *while* loop
    def time_asof_single(self, constructor):
        self.ts.asof(self.date)

    # test speed of the code path for a scalar index
    # before the start. should be the same as above.
    def time_asof_single_early(self, constructor):
        self.ts.asof(self.date_early)

    # test the speed of the code path for a scalar index
    # with a long *while* loop. should still be much
    # faster than pre-computing all the NAs.
    def time_asof_nan_single(self, constructor):
        self.ts3.asof(self.date_last)


class SortIndex:

    params = [True, False]
    param_names = ["monotonic"]

    def setup(self, monotonic):
        N = 10**5
        idx = date_range(start="1/1/2000", periods=N, freq="s")
        self.s = Series(np.random.randn(N), index=idx)
        if not monotonic:
            self.s = self.s.sample(frac=1)

    def time_sort_index(self, monotonic):
        self.s.sort_index()

    def time_get_slice(self, monotonic):
        self.s[:10000]


class Lookup:
    def setup(self):
        N = 1500000
        rng = date_range(start="1/1/2000", periods=N, freq="S")
        self.ts = Series(1, index=rng)
        self.lookup_val = rng[N // 2]

    def time_lookup_and_cleanup(self):
        self.ts[self.lookup_val]
        self.ts.index._cleanup()


class DatetimeAccessor:

    params = [None, "US/Eastern", "UTC", dateutil.tz.tzutc()]
    param_names = "tz"

    def setup(self, tz):
        N = 100000
        self.series = Series(date_range(start="1/1/2000", periods=N, freq="T", tz=tz))

    def time_dt_accessor(self, tz):
        self.series.dt

    def time_dt_accessor_normalize(self, tz):
        self.series.dt.normalize()

    def time_dt_accessor_month_name(self, tz):
        self.series.dt.month_name()

    def time_dt_accessor_day_name(self, tz):
        self.series.dt.day_name()

    def time_dt_accessor_time(self, tz):
        self.series.dt.time

    def time_dt_accessor_date(self, tz):
        self.series.dt.date

    def time_dt_accessor_year(self, tz):
        self.series.dt.year


from .pandas_vb_common import setup  # noqa: F401 isort:skip
