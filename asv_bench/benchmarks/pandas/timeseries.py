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


from .pandas_vb_common import setup  # noqa: F401 isort:skip
