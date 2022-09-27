"""
These benchmarks are for Series and DataFrame indexing methods.  For the
lower-level methods directly on Index and subclasses, see index_object.py,
indexing_engine.py, and index_cached.py
"""
import warnings

import numpy as np

from modin.pandas import (
    DataFrame,
    Float64Index,
    Int64Index,
    MultiIndex,
    Series,
    UInt64Index,
    concat,
    date_range,
    period_range,
)

import pandas._testing as tm


class NumericSeriesIndexing:

    params = [
        (Int64Index, UInt64Index, Float64Index),
        ("unique_monotonic_inc", "nonunique_monotonic_inc"),
    ]
    param_names = ["index_dtype", "index_structure"]

    def setup(self, index, index_structure):
        N = 10**6
        indices = {
            "unique_monotonic_inc": index(range(N)),
            "nonunique_monotonic_inc": index(
                list(range(55)) + [54] + list(range(55, N - 1))
            ),
        }
        self.data = Series(np.random.rand(N), index=indices[index_structure])
        self.array = np.arange(10000)
        self.array_list = self.array.tolist()

    def time_getitem_scalar(self, index, index_structure):
        self.data[800000]

    def time_getitem_slice(self, index, index_structure):
        self.data[:800000]

    def time_getitem_list_like(self, index, index_structure):
        self.data[[800000]]

    def time_getitem_array(self, index, index_structure):
        self.data[self.array]

    def time_getitem_lists(self, index, index_structure):
        self.data[self.array_list]

    def time_iloc_array(self, index, index_structure):
        self.data.iloc[self.array]

    def time_iloc_list_like(self, index, index_structure):
        self.data.iloc[[800000]]

    def time_iloc_scalar(self, index, index_structure):
        self.data.iloc[800000]

    def time_iloc_slice(self, index, index_structure):
        self.data.iloc[:800000]

    def time_loc_array(self, index, index_structure):
        self.data.loc[self.array]

    def time_loc_list_like(self, index, index_structure):
        self.data.loc[[800000]]

    def time_loc_scalar(self, index, index_structure):
        self.data.loc[800000]

    def time_loc_slice(self, index, index_structure):
        self.data.loc[:800000]


class DataFrameNumericIndexing:

    params = [
        (Int64Index, UInt64Index, Float64Index),
        ("unique_monotonic_inc", "nonunique_monotonic_inc"),
    ]
    param_names = ["index_dtype", "index_structure"]

    def setup(self, index, index_structure):
        N = 10**5
        indices = {
            "unique_monotonic_inc": index(range(N)),
            "nonunique_monotonic_inc": index(
                list(range(55)) + [54] + list(range(55, N - 1))
            ),
        }
        self.idx_dupe = np.array(range(30)) * 99
        self.df = DataFrame(np.random.randn(N, 5), index=indices[index_structure])
        self.df_dup = concat([self.df, 2 * self.df, 3 * self.df])
        self.bool_indexer = [True] * (N // 2) + [False] * (N - N // 2)

    def time_iloc_dups(self, index, index_structure):
        self.df_dup.iloc[self.idx_dupe]

    def time_loc_dups(self, index, index_structure):
        self.df_dup.loc[self.idx_dupe]

    def time_iloc(self, index, index_structure):
        self.df.iloc[:100, 0]

    def time_loc(self, index, index_structure):
        self.df.loc[:100, 0]

    def time_bool_indexer(self, index, index_structure):
        self.df[self.bool_indexer]


class Take:

    params = ["int", "datetime"]
    param_names = ["index"]

    def setup(self, index):
        N = 100000
        indexes = {
            "int": Int64Index(np.arange(N)),
            "datetime": date_range("2011-01-01", freq="S", periods=N),
        }
        index = indexes[index]
        self.s = Series(np.random.rand(N), index=index)
        self.indexer = np.random.randint(0, N, size=N)

    def time_take(self, index):
        self.s.take(self.indexer)


class MultiIndexing:

    params = [True, False]
    param_names = ["unique_levels"]

    def setup(self, unique_levels):
        self.nlevels = 2
        if unique_levels:
            mi = MultiIndex.from_arrays([range(1000000)] * self.nlevels)
        else:
            mi = MultiIndex.from_product([range(1000)] * self.nlevels)
        self.df = DataFrame(np.random.randn(len(mi)), index=mi)

        self.tgt_slice = slice(200, 800)
        self.tgt_null_slice = slice(None)
        self.tgt_list = list(range(0, 1000, 10))
        self.tgt_scalar = 500

        bool_indexer = np.zeros(len(mi), dtype=np.bool_)
        bool_indexer[slice(0, len(mi), 100)] = True
        self.tgt_bool_indexer = bool_indexer

    def time_loc_partial_key_slice(self, unique_levels):
        self.df.loc[self.tgt_slice, :]

    def time_loc_partial_key_null_slice(self, unique_levels):
        self.df.loc[self.tgt_null_slice, :]

    def time_loc_partial_key_list(self, unique_levels):
        self.df.loc[self.tgt_list, :]

    def time_loc_partial_key_scalar(self, unique_levels):
        self.df.loc[self.tgt_scalar, :]

    def time_loc_partial_key_bool_indexer(self, unique_levels):
        self.df.loc[self.tgt_bool_indexer, :]

    def time_loc_all_slices(self, unique_levels):
        target = tuple([self.tgt_slice] * self.nlevels)
        self.df.loc[target, :]

    def time_loc_all_null_slices(self, unique_levels):
        target = tuple([self.tgt_null_slice] * self.nlevels)
        self.df.loc[target, :]

    def time_loc_all_lists(self, unique_levels):
        target = tuple([self.tgt_list] * self.nlevels)
        self.df.loc[target, :]

    def time_loc_all_scalars(self, unique_levels):
        target = tuple([self.tgt_scalar] * self.nlevels)
        self.df.loc[target, :]

    def time_loc_all_bool_indexers(self, unique_levels):
        target = tuple([self.tgt_bool_indexer] * self.nlevels)
        self.df.loc[target, :]

    def time_loc_slice_plus_null_slice(self, unique_levels):
        target = (self.tgt_slice, self.tgt_null_slice)
        self.df.loc[target, :]

    def time_loc_null_slice_plus_slice(self, unique_levels):
        target = (self.tgt_null_slice, self.tgt_slice)
        self.df.loc[target, :]

    def time_xs_level_0(self, unique_levels):
        target = self.tgt_scalar
        self.df.xs(target, level=0)

    def time_xs_level_1(self, unique_levels):
        target = self.tgt_scalar
        self.df.xs(target, level=1)

    def time_xs_full_key(self, unique_levels):
        target = tuple([self.tgt_scalar] * self.nlevels)
        self.df.xs(target)


class GetItemSingleColumn:
    def setup(self):
        self.df_string_col = DataFrame(np.random.randn(3000, 1), columns=["A"])
        self.df_int_col = DataFrame(np.random.randn(3000, 1))

    def time_frame_getitem_single_column_label(self):
        self.df_string_col["A"]

    def time_frame_getitem_single_column_int(self):
        self.df_int_col[0]


class IndexSingleRow:
    params = [True, False]
    param_names = ["unique_cols"]

    def setup(self, unique_cols):
        arr = np.arange(10**7).reshape(-1, 10)
        df = DataFrame(arr)
        dtypes = ["u1", "u2", "u4", "u8", "i1", "i2", "i4", "i8", "f8", "f4"]
        for i, d in enumerate(dtypes):
            df[i] = df[i].astype(d)

        if not unique_cols:
            # GH#33032 single-row lookups with non-unique columns were
            #  15x slower than with unique columns
            df.columns = ["A", "A"] + list(df.columns[2:])

        self.df = df

    def time_iloc_row(self, unique_cols):
        self.df.iloc[10000]

    def time_loc_row(self, unique_cols):
        self.df.loc[10000]


class AssignTimeseriesIndex:
    def setup(self):
        N = 100000
        idx = date_range("1/1/2000", periods=N, freq="H")
        self.df = DataFrame(np.random.randn(N, 1), columns=["A"], index=idx)

    def time_frame_assign_timeseries_index(self):
        self.df["date"] = self.df.index


class InsertColumns:
    def setup(self):
        self.N = 10**3
        self.df = DataFrame(index=range(self.N))
        self.df2 = DataFrame(np.random.randn(self.N, 2))

    def time_insert(self):
        for i in range(100):
            self.df.insert(0, i, np.random.randn(self.N), allow_duplicates=True)

    def time_insert_middle(self):
        # same as time_insert but inserting to a middle column rather than
        #  front or back (which have fast-paths)
        for _ in range(100):
            self.df2.insert(
                1, "colname", np.random.randn(self.N), allow_duplicates=True
            )

    def time_assign_with_setitem(self):
        for i in range(100):
            self.df[i] = np.random.randn(self.N)

    def time_assign_list_like_with_setitem(self):
        self.df[list(range(100))] = np.random.randn(self.N, 100)

    def time_assign_list_of_columns_concat(self):
        df = DataFrame(np.random.randn(self.N, 100))
        concat([self.df, df], axis=1)


from .pandas_vb_common import setup  # noqa: F401 isort:skip
