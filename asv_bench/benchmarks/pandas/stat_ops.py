import numpy as np

import modin.pandas as pd

ops = ["mean", "sum", "median", "std", "skew", "kurt", "mad", "prod", "sem", "var"]


class FrameOps:

    params = [ops, ["float", "int", "Int64"], [0, 1]]
    param_names = ["op", "dtype", "axis"]

    def setup(self, op, dtype, axis):
        if op == "mad" and dtype == "Int64":
            # GH-33036, GH#33600
            raise NotImplementedError
        values = np.random.randn(100000, 4)
        if dtype == "Int64":
            values = values.astype(int)
        df = pd.DataFrame(values).astype(dtype)
        self.df_func = getattr(df, op)

    def time_op(self, op, dtype, axis):
        self.df_func(axis=axis)


class FrameMultiIndexOps:

    params = ([0, 1, [0, 1]], ops)
    param_names = ["level", "op"]

    def setup(self, level, op):
        levels = [np.arange(10), np.arange(100), np.arange(100)]
        codes = [
            np.arange(10).repeat(10000),
            np.tile(np.arange(100).repeat(100), 10),
            np.tile(np.tile(np.arange(100), 100), 10),
        ]
        index = pd.MultiIndex(levels=levels, codes=codes)
        df = pd.DataFrame(np.random.randn(len(index), 4), index=index)
        self.df_func = getattr(df, op)

    def time_op(self, level, op):
        self.df_func(level=level)


class Covariance:

    params = []
    param_names = []

    def setup(self):
        self.s = pd.Series(np.random.randn(100000))
        self.s2 = pd.Series(np.random.randn(100000))

    def time_cov_series(self):
        self.s.cov(self.s2)


from .pandas_vb_common import setup  # noqa: F401 isort:skip
