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


class FrameWithFrameWide:
    # Many-columns, mixed dtypes

    params = [
        [
            # GH#32779 has discussion of which operators are included here
            operator.add,
            operator.floordiv,
            operator.gt,
        ],
        [
            # (n_rows, n_columns)
            (1_000_000, 10),
            (100_000, 100),
            (10_000, 1000),
            (1000, 10_000),
        ],
    ]
    param_names = ["op", "shape"]

    def setup(self, op, shape):
        # we choose dtypes so as to make the blocks
        #  a) not perfectly match between right and left
        #  b) appreciably bigger than single columns
        n_rows, n_cols = shape

        if op is operator.floordiv:
            # floordiv is much slower than the other operations -> use less data
            n_rows = n_rows // 10

        # construct dataframe with 2 blocks
        arr1 = np.random.randn(n_rows, n_cols // 2).astype("f8")
        arr2 = np.random.randn(n_rows, n_cols // 2).astype("f4")
        df = pd.concat([DataFrame(arr1), DataFrame(arr2)], axis=1, ignore_index=True)
        # should already be the case, but just to be sure
        df._consolidate_inplace()

        # TODO: GH#33198 the setting here shouldn't need two steps
        arr1 = np.random.randn(n_rows, max(n_cols // 4, 3)).astype("f8")
        arr2 = np.random.randn(n_rows, n_cols // 2).astype("i8")
        arr3 = np.random.randn(n_rows, n_cols // 4).astype("f8")
        df2 = pd.concat(
            [DataFrame(arr1), DataFrame(arr2), DataFrame(arr3)],
            axis=1,
            ignore_index=True,
        )
        # should already be the case, but just to be sure
        df2._consolidate_inplace()

        self.left = df
        self.right = df2

    def time_op_different_blocks(self, op, shape):
        # blocks (and dtypes) are not aligned
        op(self.left, self.right)

    def time_op_same_blocks(self, op, shape):
        # blocks (and dtypes) are aligned
        op(self.left, self.left)


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
