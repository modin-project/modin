import operator

import numpy as np

import modin.pandas as pd
from modin.pandas import (
    DataFrame,
    Series,
    date_range,
)

from .pandas_vb_common import numeric_dtypes


class IntFrameWithScalar:
    params = [
        [np.float64, np.int64],
        [2, 3.0, np.int32(4), np.float64(5)],
        [
            # subset of operators used in the corresponding pandas asv;
            #  include floordiv in particular bc it goes through
            #  its own code-path.
            operator.add,
            operator.floordiv,
            operator.le,
        ],
    ]
    param_names = ["dtype", "scalar", "op"]

    def setup(self, dtype, scalar, op):
        arr = np.random.randn(20000, 100)
        self.df = DataFrame(arr.astype(dtype))

    def time_frame_op_with_scalar(self, dtype, scalar, op):
        op(self.df, scalar)


class Timeseries:

    params = [None, "US/Eastern"]
    param_names = ["tz"]

    def setup(self, tz):
        N = 10**6
        halfway = (N // 2) - 1
        self.s = Series(date_range("20010101", periods=N, freq="T", tz=tz))
        self.ts = self.s[halfway]

        self.s2 = Series(date_range("20010101", periods=N, freq="s", tz=tz))

    def time_series_timestamp_compare(self, tz):
        self.s <= self.ts

    def time_timestamp_series_compare(self, tz):
        self.ts >= self.s

    def time_timestamp_ops_diff(self, tz):
        self.s2.diff()

    def time_timestamp_ops_diff_with_shift(self, tz):
        self.s - self.s.shift()


class IrregularOps:
    def setup(self):
        N = 10**5
        idx = date_range(start="1/1/2000", periods=N, freq="s")
        s = Series(np.random.randn(N), index=idx)
        self.left = s.sample(frac=1)
        self.right = s.sample(frac=1)

    def time_add(self):
        self.left + self.right


class NumericInferOps:
    # from GH 7332
    params = numeric_dtypes
    param_names = ["dtype"]

    def setup(self, dtype):
        N = 5 * 10**5
        self.df = DataFrame(
            {"A": np.arange(N).astype(dtype), "B": np.arange(N).astype(dtype)}
        )

    def time_add(self, dtype):
        self.df["A"] + self.df["B"]

    def time_subtract(self, dtype):
        self.df["A"] - self.df["B"]

    def time_multiply(self, dtype):
        self.df["A"] * self.df["B"]

    def time_divide(self, dtype):
        self.df["A"] / self.df["B"]

    def time_modulo(self, dtype):
        self.df["A"] % self.df["B"]


class DateInferOps:
    # from GH 7332
    def setup_cache(self):
        N = 5 * 10**5
        df = DataFrame({"datetime64": np.arange(N).astype("datetime64[ms]")})
        df["timedelta"] = df["datetime64"] - df["datetime64"]
        return df

    def time_subtract_datetimes(self, df):
        df["datetime64"] - df["datetime64"]

    def time_timedelta_plus_datetime(self, df):
        df["timedelta"] + df["datetime64"]

    def time_add_timedeltas(self, df):
        df["timedelta"] + df["timedelta"]


class BinaryOpsMultiIndex:
    params = ["sub", "add", "mul", "div"]
    param_names = ["func"]

    def setup(self, func):
        array = date_range("20200101 00:00", "20200102 0:00", freq="S")
        level_0_names = [str(i) for i in range(30)]

        index = pd.MultiIndex.from_product([level_0_names, array])
        column_names = ["col_1", "col_2"]

        self.df = DataFrame(
            np.random.rand(len(index), 2), index=index, columns=column_names
        )

        self.arg_df = DataFrame(
            np.random.randint(1, 10, (len(level_0_names), 2)),
            index=level_0_names,
            columns=column_names,
        )

    def time_binary_op_multiindex(self, func):
        getattr(self.df, func)(self.arg_df, level=0)


from .pandas_vb_common import setup  # noqa: F401 isort:skip
