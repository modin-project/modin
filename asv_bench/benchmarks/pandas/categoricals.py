import string
import sys
import warnings

import numpy as np

import modin.pandas as pd

import pandas._testing as tm


class Concat:
    def setup(self):
        N = 10**5
        self.s = pd.Series(list("aabbcd") * N).astype("category")

        self.a = pd.Categorical(list("aabbcd") * N)
        self.b = pd.Categorical(list("bbcdjk") * N)

        self.idx_a = pd.CategoricalIndex(range(N), range(N))
        self.idx_b = pd.CategoricalIndex(range(N + 1), range(N + 1))
        self.df_a = pd.DataFrame(range(N), columns=["a"], index=self.idx_a)
        self.df_b = pd.DataFrame(range(N + 1), columns=["a"], index=self.idx_b)

    def time_concat(self):
        pd.concat([self.s, self.s])

    def time_concat_overlapping_index(self):
        pd.concat([self.df_a, self.df_a])

    def time_concat_non_overlapping_index(self):
        pd.concat([self.df_a, self.df_b])


class SetCategories:
    def setup(self):
        n = 5 * 10**5
        arr = [f"s{i:04d}" for i in np.random.randint(0, n // 10, size=n)]
        self.ts = pd.Series(arr).astype("category")

    def time_set_categories(self):
        self.ts.cat.set_categories(self.ts.cat.categories[::2])


class RemoveCategories:
    def setup(self):
        n = 5 * 10**5
        arr = [f"s{i:04d}" for i in np.random.randint(0, n // 10, size=n)]
        self.ts = pd.Series(arr).astype("category")

    def time_remove_categories(self):
        self.ts.cat.remove_categories(self.ts.cat.categories[::2])


class Rank:
    def setup(self):
        N = 10**5
        ncats = 15

        self.s_str = pd.Series(tm.makeCategoricalIndex(N, ncats)).astype(str)
        self.s_str_cat = pd.Series(self.s_str, dtype="category")
        with warnings.catch_warnings(record=True):
            str_cat_type = pd.CategoricalDtype(set(self.s_str), ordered=True)
            self.s_str_cat_ordered = self.s_str.astype(str_cat_type)

        self.s_int = pd.Series(np.random.randint(0, ncats, size=N))
        self.s_int_cat = pd.Series(self.s_int, dtype="category")
        with warnings.catch_warnings(record=True):
            int_cat_type = pd.CategoricalDtype(set(self.s_int), ordered=True)
            self.s_int_cat_ordered = self.s_int.astype(int_cat_type)

    def time_rank_string(self):
        self.s_str.rank()

    def time_rank_string_cat(self):
        self.s_str_cat.rank()

    def time_rank_string_cat_ordered(self):
        self.s_str_cat_ordered.rank()

    def time_rank_int(self):
        self.s_int.rank()

    def time_rank_int_cat(self):
        self.s_int_cat.rank()

    def time_rank_int_cat_ordered(self):
        self.s_int_cat_ordered.rank()


class IsMonotonic:
    def setup(self):
        N = 1000
        ci = pd.CategoricalIndex(list("a" * N + "b" * N + "c" * N))
        self.ser = pd.Series(ci)

    def time_categorical_series_is_monotonic_increasing(self):
        self.ser.is_monotonic_increasing

    def time_categorical_series_is_monotonic_decreasing(self):
        self.ser.is_monotonic_decreasing


class Indexing:
    def setup(self):
        N = 10**5
        self.index = pd.CategoricalIndex(range(N), range(N))
        self.series = pd.Series(range(N), index=self.index).sort_index()
        self.category = self.index[500]

    def time_align(self):
        pd.DataFrame({"a": self.series, "b": self.series[:500]})


from .pandas_vb_common import setup  # noqa: F401 isort:skip
