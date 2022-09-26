import string
import warnings

import numpy as np

from modin.pandas import (
    DataFrame,
    MultiIndex,
    NaT,
    Series,
    date_range,
    isnull,
    period_range,
    timedelta_range,
)

import pandas._testing as tm


class Reindex:
    def setup(self):
        N = 10**3
        self.df = DataFrame(np.random.randn(N * 10, N))
        self.idx = np.arange(4 * N, 7 * N)
        self.idx_cols = np.random.randint(0, N, N)
        self.df2 = DataFrame(
            {
                c: {
                    0: np.random.randint(0, 2, N).astype(np.bool_),
                    1: np.random.randint(0, N, N).astype(np.int16),
                    2: np.random.randint(0, N, N).astype(np.int32),
                    3: np.random.randint(0, N, N).astype(np.int64),
                }[np.random.randint(0, 4)]
                for c in range(N)
            }
        )

    def time_reindex_axis0(self):
        self.df.reindex(self.idx)

    def time_reindex_axis1(self):
        self.df.reindex(columns=self.idx_cols)

    def time_reindex_axis1_missing(self):
        self.df.reindex(columns=self.idx)

    def time_reindex_both_axes(self):
        self.df.reindex(index=self.idx, columns=self.idx_cols)

    def time_reindex_upcast(self):
        self.df2.reindex(np.random.permutation(range(1200)))


class Rename:
    def setup(self):
        N = 10**3
        self.df = DataFrame(np.random.randn(N * 10, N))
        self.idx = np.arange(4 * N, 7 * N)
        self.dict_idx = {k: k for k in self.idx}
        self.df2 = DataFrame(
            {
                c: {
                    0: np.random.randint(0, 2, N).astype(np.bool_),
                    1: np.random.randint(0, N, N).astype(np.int16),
                    2: np.random.randint(0, N, N).astype(np.int32),
                    3: np.random.randint(0, N, N).astype(np.int64),
                }[np.random.randint(0, 4)]
                for c in range(N)
            }
        )

    def time_rename_single(self):
        self.df.rename({0: 0})

    def time_rename_axis0(self):
        self.df.rename(self.dict_idx)

    def time_rename_axis1(self):
        self.df.rename(columns=self.dict_idx)

    def time_rename_both_axes(self):
        self.df.rename(index=self.dict_idx, columns=self.dict_idx)

    def time_dict_rename_both_axes(self):
        self.df.rename(index=self.dict_idx, columns=self.dict_idx)


class Repr:
    def setup(self):
        nrows = 10000
        data = np.random.randn(nrows, 10)
        arrays = np.tile(np.random.randn(3, nrows // 100), 100)
        idx = MultiIndex.from_arrays(arrays)
        self.df3 = DataFrame(data, index=idx)
        self.df4 = DataFrame(data, index=np.random.randn(nrows))
        self.df_tall = DataFrame(np.random.randn(nrows, 10))
        self.df_wide = DataFrame(np.random.randn(10, nrows))

    def time_repr_tall(self):
        repr(self.df_tall)

    def time_frame_repr_wide(self):
        repr(self.df_wide)


class MaskBool:
    def setup(self):
        data = np.random.randn(1000, 500)
        df = DataFrame(data)
        df = df.where(df > 0)
        self.bools = df > 0
        self.mask = isnull(df)

    def time_frame_mask_bools(self):
        self.bools.mask(self.mask)

    def time_frame_mask_floats(self):
        self.bools.astype(float).mask(self.mask)


class Isnull:
    def setup(self):
        N = 10**3
        self.df_no_null = DataFrame(np.random.randn(N, N))

        sample = np.array([np.nan, 1.0])
        data = np.random.choice(sample, (N, N))
        self.df = DataFrame(data)

        sample = np.array(list(string.ascii_letters + string.whitespace))
        data = np.random.choice(sample, (N, N))
        self.df_strings = DataFrame(data)

        sample = np.array(
            [
                NaT,
                np.nan,
                None,
                np.datetime64("NaT"),
                np.timedelta64("NaT"),
                0,
                1,
                2.0,
                "",
                "abcd",
            ]
        )
        data = np.random.choice(sample, (N, N))
        self.df_obj = DataFrame(data)

    def time_isnull_floats_no_null(self):
        isnull(self.df_no_null)

    def time_isnull(self):
        isnull(self.df)

    def time_isnull_strngs(self):
        isnull(self.df_strings)

    def time_isnull_obj(self):
        isnull(self.df_obj)


class Dropna:

    params = (["all", "any"], [0, 1])
    param_names = ["how", "axis"]

    def setup(self, how, axis):
        self.df = DataFrame(np.random.randn(10000, 1000))
        self.df.iloc[50:1000, 20:50] = np.nan
        self.df.iloc[2000:3000] = np.nan
        self.df.iloc[:, 60:70] = np.nan
        self.df_mixed = self.df.copy()
        self.df_mixed["foo"] = "bar"

    def time_dropna(self, how, axis):
        self.df.dropna(how=how, axis=axis)

    def time_dropna_axis_mixed_dtypes(self, how, axis):
        self.df_mixed.dropna(how=how, axis=axis)


class Dtypes:
    def setup(self):
        self.df = DataFrame(np.random.randn(1000, 1000))

    def time_frame_dtypes(self):
        self.df.dtypes


class Equals:
    def setup(self):
        N = 10**3
        self.float_df = DataFrame(np.random.randn(N, N))
        self.float_df_nan = self.float_df.copy()
        self.float_df_nan.iloc[-1, -1] = np.nan

        self.object_df = DataFrame("foo", index=range(N), columns=range(N))
        self.object_df_nan = self.object_df.copy()
        self.object_df_nan.iloc[-1, -1] = np.nan

        self.nonunique_cols = self.object_df.copy()
        self.nonunique_cols.columns = ["A"] * len(self.nonunique_cols.columns)
        self.nonunique_cols_nan = self.nonunique_cols.copy()
        self.nonunique_cols_nan.iloc[-1, -1] = np.nan

    def time_frame_float_equal(self):
        self.float_df.equals(self.float_df)

    def time_frame_float_unequal(self):
        self.float_df.equals(self.float_df_nan)

    def time_frame_nonunique_equal(self):
        self.nonunique_cols.equals(self.nonunique_cols)

    def time_frame_nonunique_unequal(self):
        self.nonunique_cols.equals(self.nonunique_cols_nan)

    def time_frame_object_equal(self):
        self.object_df.equals(self.object_df)

    def time_frame_object_unequal(self):
        self.object_df.equals(self.object_df_nan)


class Interpolate:

    params = [None, "infer"]
    param_names = ["downcast"]

    def setup(self, downcast):
        N = 10000
        # this is the worst case, where every column has NaNs.
        arr = np.random.randn(N, 100)
        # NB: we need to set values in array, not in df.values, otherwise
        #  the benchmark will be misleading for ArrayManager
        arr[::2] = np.nan

        self.df = DataFrame(arr)

        self.df2 = DataFrame(
            {
                "A": np.arange(0, N),
                "B": np.random.randint(0, 100, N),
                "C": np.random.randn(N),
                "D": np.random.randn(N),
            }
        )
        self.df2.loc[1::5, "A"] = np.nan
        self.df2.loc[1::5, "C"] = np.nan

    def time_interpolate(self, downcast):
        self.df.interpolate(downcast=downcast)

    def time_interpolate_some_good(self, downcast):
        self.df2.interpolate(downcast=downcast)


class Shift:
    # frame shift speedup issue-5609
    params = [0, 1]
    param_names = ["axis"]

    def setup(self, axis):
        self.df = DataFrame(np.random.rand(10000, 500))

    def time_shift(self, axis):
        self.df.shift(1, axis=axis)

class Rank:
    param_names = ["dtype"]
    params = [
        ["int", "uint", "float", "object"],
    ]

    def setup(self, dtype):
        self.df = DataFrame(
            np.random.randn(10000, 10).astype(dtype), columns=range(10), dtype=dtype
        )

    def time_rank(self, dtype):
        self.df.rank()


from .pandas_vb_common import setup  # noqa: F401 isort:skip
