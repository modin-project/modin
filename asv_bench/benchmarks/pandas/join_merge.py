import numpy as np

from modin.pandas import (
    DataFrame,
    MultiIndex,
    Series,
    concat,
    date_range,
    merge,
)

import pandas._testing as tm


class ConcatDataFrames:

    params = ([0, 1], [True, False])
    param_names = ["axis", "ignore_index"]

    def setup(self, axis, ignore_index):
        frame_c = DataFrame(np.zeros((10000, 200), dtype=np.float32, order="C"))
        self.frame_c = [frame_c] * 20
        frame_f = DataFrame(np.zeros((10000, 200), dtype=np.float32, order="F"))
        self.frame_f = [frame_f] * 20

    def time_c_ordered(self, axis, ignore_index):
        concat(self.frame_c, axis=axis, ignore_index=ignore_index)

    def time_f_ordered(self, axis, ignore_index):
        concat(self.frame_f, axis=axis, ignore_index=ignore_index)


class Join:

    params = [True, False]
    param_names = ["sort"]

    def setup(self, sort):
        level1 = tm.makeStringIndex(10).values
        level2 = tm.makeStringIndex(1000).values
        codes1 = np.arange(10).repeat(1000)
        codes2 = np.tile(np.arange(1000), 10)
        index2 = MultiIndex(levels=[level1, level2], codes=[codes1, codes2])
        self.df_multi = DataFrame(
            np.random.randn(len(index2), 4), index=index2, columns=["A", "B", "C", "D"]
        )

        self.key1 = np.tile(level1.take(codes1), 10)
        self.key2 = np.tile(level2.take(codes2), 10)
        self.df = DataFrame(
            {
                "data1": np.random.randn(100000),
                "data2": np.random.randn(100000),
                "key1": self.key1,
                "key2": self.key2,
            }
        )

        self.df_key1 = DataFrame(
            np.random.randn(len(level1), 4), index=level1, columns=["A", "B", "C", "D"]
        )
        self.df_key2 = DataFrame(
            np.random.randn(len(level2), 4), index=level2, columns=["A", "B", "C", "D"]
        )

        shuf = np.arange(100000)
        np.random.shuffle(shuf)
        self.df_shuf = self.df.reindex(self.df.index[shuf])

    def time_join_dataframe_index_multi(self, sort):
        self.df.join(self.df_multi, on=["key1", "key2"], sort=sort)

    def time_join_dataframe_index_single_key_bigger(self, sort):
        self.df.join(self.df_key2, on="key2", sort=sort)

    def time_join_dataframe_index_single_key_small(self, sort):
        self.df.join(self.df_key1, on="key1", sort=sort)

    def time_join_dataframe_index_shuffle_key_bigger_sort(self, sort):
        self.df_shuf.join(self.df_key2, on="key2", sort=sort)


class JoinNonUnique:
    # outer join of non-unique
    def setup(self):
        date_index = date_range("01-Jan-2013", "23-Jan-2013", freq="T")
        daily_dates = date_index.to_period("D").to_timestamp("S", "S")
        self.fracofday = date_index.values - daily_dates.values
        self.fracofday = self.fracofday.astype("timedelta64[ns]")
        self.fracofday = self.fracofday.astype(np.float64) / 86_400_000_000_000
        self.fracofday = Series(self.fracofday, daily_dates)
        index = date_range(date_index.min(), date_index.max(), freq="D")
        self.temp = Series(1.0, index)[self.fracofday.index]

    def time_join_non_unique_equal(self):
        self.fracofday * self.temp


class Merge:

    params = [True, False]
    param_names = ["sort"]

    def setup(self, sort):
        N = 10000
        indices = tm.makeStringIndex(N).values
        indices2 = tm.makeStringIndex(N).values
        key = np.tile(indices[:8000], 10)
        key2 = np.tile(indices2[:8000], 10)
        self.left = DataFrame(
            {"key": key, "key2": key2, "value": np.random.randn(80000)}
        )
        self.right = DataFrame(
            {
                "key": indices[2000:],
                "key2": indices2[2000:],
                "value2": np.random.randn(8000),
            }
        )

        self.df = DataFrame(
            {
                "key1": np.tile(np.arange(500).repeat(10), 2),
                "key2": np.tile(np.arange(250).repeat(10), 4),
                "value": np.random.randn(10000),
            }
        )
        self.df2 = DataFrame({"key1": np.arange(500), "value2": np.random.randn(500)})
        self.df3 = self.df[:5000]

    def time_merge_2intkey(self, sort):
        merge(self.left, self.right, sort=sort)

    def time_merge_dataframe_integer_2key(self, sort):
        merge(self.df, self.df3, sort=sort)

    def time_merge_dataframe_integer_key(self, sort):
        merge(self.df, self.df2, on="key1", sort=sort)

    def time_merge_dataframe_empty_right(self, sort):
        merge(self.left, self.right.iloc[:0], sort=sort)

    def time_merge_dataframe_empty_left(self, sort):
        merge(self.left.iloc[:0], self.right, sort=sort)

    def time_merge_dataframes_cross(self, sort):
        merge(self.left.loc[:2000], self.right.loc[:2000], how="cross", sort=sort)


class MergeCategoricals:
    def setup(self):
        self.left_object = DataFrame(
            {
                "X": np.random.choice(range(0, 10), size=(10000,)),
                "Y": np.random.choice(["one", "two", "three"], size=(10000,)),
            }
        )

        self.right_object = DataFrame(
            {
                "X": np.random.choice(range(0, 10), size=(10000,)),
                "Z": np.random.choice(["jjj", "kkk", "sss"], size=(10000,)),
            }
        )

        self.left_cat = self.left_object.assign(
            Y=self.left_object["Y"].astype("category")
        )
        self.right_cat = self.right_object.assign(
            Z=self.right_object["Z"].astype("category")
        )

    def time_merge_object(self):
        merge(self.left_object, self.right_object, on="X")

    def time_merge_cat(self):
        merge(self.left_cat, self.right_cat, on="X")


from .pandas_vb_common import setup  # noqa: E402, F401
