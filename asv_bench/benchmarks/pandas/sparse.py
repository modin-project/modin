import numpy as np

import modin.pandas as pd
from modin.pandas import (
    MultiIndex,
    Series,
    date_range,
)
from pandas.arrays import SparseArray


def make_array(size, dense_proportion, fill_value, dtype):
    dense_size = int(size * dense_proportion)
    arr = np.full(size, fill_value, dtype)
    indexer = np.random.choice(np.arange(size), dense_size, replace=False)
    arr[indexer] = np.random.choice(np.arange(100, dtype=dtype), dense_size)
    return arr


class SparseSeriesToFrame:
    def setup(self):
        K = 50
        N = 50001
        rng = date_range("1/1/2000", periods=N, freq="T")
        self.series = {}
        for i in range(1, K):
            data = np.random.randn(N)[:-i]
            idx = rng[:-i]
            data[100:] = np.nan
            self.series[i] = Series(SparseArray(data), index=idx)

    def time_series_to_frame(self):
        pd.DataFrame(self.series)


class ToCoo:
    params = [True, False]
    param_names = ["sort_labels"]

    def setup(self, sort_labels):
        s = Series([np.nan] * 10000)
        s[0] = 3.0
        s[100] = -1.0
        s[999] = 12.1

        s_mult_lvl = s.set_axis(MultiIndex.from_product([range(10)] * 4))
        self.ss_mult_lvl = s_mult_lvl.astype("Sparse")

        s_two_lvl = s.set_axis(MultiIndex.from_product([range(100)] * 2))
        self.ss_two_lvl = s_two_lvl.astype("Sparse")

    def time_sparse_series_to_coo(self, sort_labels):
        self.ss_mult_lvl.sparse.to_coo(
            row_levels=[0, 1], column_levels=[2, 3], sort_labels=sort_labels
        )

    def time_sparse_series_to_coo_single_level(self, sort_labels):
        self.ss_two_lvl.sparse.to_coo(sort_labels=sort_labels)


class ToCooFrame:
    def setup(self):
        N = 10000
        k = 10
        arr = np.zeros((N, k), dtype=float)
        arr[0, 0] = 3.0
        arr[12, 7] = -1.0
        arr[0, 9] = 11.2
        self.df = pd.DataFrame(arr, dtype=pd.SparseDtype("float", fill_value=0.0))

    def time_to_coo(self):
        self.df.sparse.to_coo()


from .pandas_vb_common import setup  # noqa: F401 isort:skip
