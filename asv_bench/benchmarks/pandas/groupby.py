from functools import partial
from itertools import product
from string import ascii_letters

import numpy as np

from modin.pandas import (
    NA,
    Categorical,
    DataFrame,
    Index,
    MultiIndex,
    Series,
    Timestamp,
    date_range,
    period_range,
)

import pandas._testing as tm


method_blocklist = {
    "object": {
        "diff",
        "median",
        "prod",
        "sem",
        "cumsum",
        "sum",
        "cummin",
        "mean",
        "max",
        "skew",
        "cumprod",
        "cummax",
        "pct_change",
        "min",
        "var",
        "mad",
        "describe",
        "std",
        "quantile",
    },
    "datetime": {
        "median",
        "prod",
        "sem",
        "cumsum",
        "sum",
        "mean",
        "skew",
        "cumprod",
        "cummax",
        "pct_change",
        "var",
        "mad",
        "describe",
        "std",
    },
}


class Apply:

    param_names = ["factor"]
    params = [4, 5]

    def setup(self, factor):
        N = 10**factor
        # two cases:
        # - small groups: small data (N**4) + many labels (2000) -> average group
        #   size of 5 (-> larger overhead of slicing method)
        # - larger groups: larger data (N**5) + fewer labels (20) -> average group
        #   size of 5000
        labels = np.random.randint(0, 2000 if factor == 4 else 20, size=N)
        labels2 = np.random.randint(0, 3, size=N)
        df = DataFrame(
            {
                "key": labels,
                "key2": labels2,
                "value1": np.random.randn(N),
                "value2": ["foo", "bar", "baz", "qux"] * (N // 4),
            }
        )
        self.df = df

    def time_scalar_function_multi_col(self, factor):
        self.df.groupby(["key", "key2"]).apply(lambda x: 1)

    def time_scalar_function_single_col(self, factor):
        self.df.groupby("key").apply(lambda x: 1)

    @staticmethod
    def df_copy_function(g):
        # ensure that the group name is available (see GH #15062)
        g.name
        return g.copy()

    def time_copy_function_multi_col(self, factor):
        self.df.groupby(["key", "key2"]).apply(self.df_copy_function)

    def time_copy_overhead_single_col(self, factor):
        self.df.groupby("key").apply(self.df_copy_function)


class ApplyNonUniqueUnsortedIndex:
    def setup(self):
        # GH 46527
        # unsorted and non-unique index
        idx = np.arange(100)[::-1]
        idx = Index(np.repeat(idx, 200), name="key")
        self.df = DataFrame(np.random.randn(len(idx), 10), index=idx)

    def time_groupby_apply_non_unique_unsorted_index(self):
        self.df.groupby("key", group_keys=False).apply(lambda x: x)


class Groups:

    param_names = ["key"]
    params = ["int64_small", "int64_large", "object_small", "object_large"]

    def setup_cache(self):
        size = 10**6
        data = {
            "int64_small": Series(np.random.randint(0, 100, size=size)),
            "int64_large": Series(np.random.randint(0, 10000, size=size)),
            "object_small": Series(
                tm.makeStringIndex(100).take(np.random.randint(0, 100, size=size))
            ),
            "object_large": Series(
                tm.makeStringIndex(10000).take(np.random.randint(0, 10000, size=size))
            ),
        }
        return data

    def setup(self, data, key):
        self.ser = data[key]

    def time_series_groups(self, data, key):
        self.ser.groupby(self.ser).groups

    def time_series_indices(self, data, key):
        self.ser.groupby(self.ser).indices


class GroupManyLabels:

    params = [1, 1000]
    param_names = ["ncols"]

    def setup(self, ncols):
        N = 1000
        data = np.random.randn(N, ncols)
        self.labels = np.random.randint(0, 100, size=N)
        self.df = DataFrame(data)

    def time_sum(self, ncols):
        self.df.groupby(self.labels).sum()


class DateAttributes:
    def setup(self):
        rng = date_range("1/1/2000", "12/31/2005", freq="H")
        self.year, self.month, self.day = rng.year, rng.month, rng.day
        self.ts = Series(np.random.randn(len(rng)), index=rng)

    def time_len_groupby_object(self):
        len(self.ts.groupby([self.year, self.month, self.day]))


class Int64:
    def setup(self):
        arr = np.random.randint(-1 << 12, 1 << 12, (1 << 17, 5))
        i = np.random.choice(len(arr), len(arr) * 5)
        arr = np.vstack((arr, arr[i]))
        i = np.random.permutation(len(arr))
        arr = arr[i]
        self.cols = list("abcde")
        self.df = DataFrame(arr, columns=self.cols)
        self.df["jim"], self.df["joe"] = np.random.randn(2, len(self.df)) * 10

    def time_overflow(self):
        self.df.groupby(self.cols).max()


class CountMultiInt:
    def setup_cache(self):
        n = 10000
        df = DataFrame(
            {
                "key1": np.random.randint(0, 500, size=n),
                "key2": np.random.randint(0, 100, size=n),
                "ints": np.random.randint(0, 1000, size=n),
                "ints2": np.random.randint(0, 1000, size=n),
            }
        )
        return df

    def time_multi_int_count(self, df):
        df.groupby(["key1", "key2"]).count()

    def time_multi_int_nunique(self, df):
        df.groupby(["key1", "key2"]).nunique()


class AggFunctions:
    def setup_cache(self):
        N = 10**5
        fac1 = np.array(["A", "B", "C"], dtype="O")
        fac2 = np.array(["one", "two"], dtype="O")
        df = DataFrame(
            {
                "key1": fac1.take(np.random.randint(0, 3, size=N)),
                "key2": fac2.take(np.random.randint(0, 2, size=N)),
                "value1": np.random.randn(N),
                "value2": np.random.randn(N),
                "value3": np.random.randn(N),
            }
        )
        return df

    def time_different_str_functions(self, df):
        df.groupby(["key1", "key2"]).agg(
            {"value1": "mean", "value2": "var", "value3": "sum"}
        )

    def time_different_numpy_functions(self, df):
        df.groupby(["key1", "key2"]).agg(
            {"value1": np.mean, "value2": np.var, "value3": np.sum}
        )

    def time_different_python_functions_multicol(self, df):
        df.groupby(["key1", "key2"]).agg([sum, min, max])

    def time_different_python_functions_singlecol(self, df):
        df.groupby("key1").agg([sum, min, max])


class MultiColumn:
    def setup_cache(self):
        N = 10**5
        key1 = np.tile(np.arange(100, dtype=object), 1000)
        key2 = key1.copy()
        np.random.shuffle(key1)
        np.random.shuffle(key2)
        df = DataFrame(
            {
                "key1": key1,
                "key2": key2,
                "data1": np.random.randn(N),
                "data2": np.random.randn(N),
            }
        )
        return df

    def time_lambda_sum(self, df):
        df.groupby(["key1", "key2"]).agg(lambda x: x.values.sum())

    def time_cython_sum(self, df):
        df.groupby(["key1", "key2"]).sum()

    def time_col_select_lambda_sum(self, df):
        df.groupby(["key1", "key2"])["data1"].agg(lambda x: x.values.sum())

    def time_col_select_numpy_sum(self, df):
        df.groupby(["key1", "key2"])["data1"].agg(np.sum)


class Shift:
    def setup(self):
        N = 18
        self.df = DataFrame({"g": ["a", "b"] * 9, "v": list(range(N))})

    def time_defaults(self):
        self.df.groupby("g").shift()

    def time_fill_value(self):
        self.df.groupby("g").shift(fill_value=99)


class FillNA:
    def setup(self):
        N = 100
        self.df = DataFrame(
            {"group": [1] * N + [2] * N, "value": [np.nan, 1.0] * N}
        ).set_index("group")

    def time_df_ffill(self):
        self.df.groupby("group").fillna(method="ffill")

    def time_df_bfill(self):
        self.df.groupby("group").fillna(method="bfill")

    def time_srs_ffill(self):
        self.df.groupby("group")["value"].fillna(method="ffill")

    def time_srs_bfill(self):
        self.df.groupby("group")["value"].fillna(method="bfill")


class GroupByCythonAgg:
    """
    Benchmarks specifically targeting our cython aggregation algorithms
    (using a big enough dataframe with simple key, so a large part of the
    time is actually spent in the grouped aggregation).
    """

    param_names = ["dtype", "method"]
    params = [
        ["float64"],
        [
            "sum",
            "prod",
            "min",
            "max",
            "mean",
            "median",
            "var",
            "first",
            "last",
            "any",
            "all",
        ],
    ]

    def setup(self, dtype, method):
        N = 1_000_000
        df = DataFrame(np.random.randn(N, 10), columns=list("abcdefghij"))
        df["key"] = np.random.randint(0, 100, size=N)
        self.df = df

    def time_frame_agg(self, dtype, method):
        self.df.groupby("key").agg(method)


class GroupByCythonAggEaDtypes:
    """
    Benchmarks specifically targeting our cython aggregation algorithms
    (using a big enough dataframe with simple key, so a large part of the
    time is actually spent in the grouped aggregation).
    """

    param_names = ["dtype", "method"]
    params = [
        ["Float64", "Int64", "Int32"],
        [
            "sum",
            "prod",
            "min",
            "max",
            "mean",
            "median",
            "var",
            "first",
            "last",
            "any",
            "all",
        ],
    ]

    def setup(self, dtype, method):
        N = 1_000_000
        df = DataFrame(
            np.random.randint(0, high=100, size=(N, 10)),
            columns=list("abcdefghij"),
            dtype=dtype,
        )
        df.loc[list(range(1, N, 5)), list("abcdefghij")] = NA
        df["key"] = np.random.randint(0, 100, size=N)
        self.df = df

    def time_frame_agg(self, dtype, method):
        self.df.groupby("key").agg(method)

class Datelike:
    # GH 14338
    params = ["period_range", "date_range", "date_range_tz"]
    param_names = ["grouper"]

    def setup(self, grouper):
        N = 10**4
        rng_map = {
            "period_range": period_range,
            "date_range": date_range,
            "date_range_tz": partial(date_range, tz="US/Central"),
        }
        self.grouper = rng_map[grouper]("1900-01-01", freq="D", periods=N)
        self.df = DataFrame(np.random.randn(10**4, 2))

    def time_sum(self, grouper):
        self.df.groupby(self.grouper).sum()


class Transform:
    def setup(self):
        n1 = 400
        n2 = 250
        index = MultiIndex(
            levels=[np.arange(n1), tm.makeStringIndex(n2)],
            codes=[np.repeat(range(n1), n2).tolist(), list(range(n2)) * n1],
            names=["lev1", "lev2"],
        )
        arr = np.random.randn(n1 * n2, 3)
        arr[::10000, 0] = np.nan
        arr[1::10000, 1] = np.nan
        arr[2::10000, 2] = np.nan
        data = DataFrame(arr, index=index, columns=["col1", "col20", "col3"])
        self.df = data

        n = 1000
        self.df_wide = DataFrame(
            np.random.randn(n, n),
            index=np.random.choice(range(10), n),
        )

        n = 1_000_000
        self.df_tall = DataFrame(
            np.random.randn(n, 3),
            index=np.random.randint(0, 5, n),
        )

        n = 20000
        self.df1 = DataFrame(
            np.random.randint(1, n, (n, 3)), columns=["jim", "joe", "jolie"]
        )
        self.df2 = self.df1.copy()
        self.df2["jim"] = self.df2["joe"]

        self.df3 = DataFrame(
            np.random.randint(1, (n / 10), (n, 3)), columns=["jim", "joe", "jolie"]
        )
        self.df4 = self.df3.copy()
        self.df4["jim"] = self.df4["joe"]

    def time_transform_lambda_max(self):
        self.df.groupby(level="lev1").transform(lambda x: max(x))

    def time_transform_ufunc_max(self):
        self.df.groupby(level="lev1").transform(np.max)

    def time_transform_lambda_max_tall(self):
        self.df_tall.groupby(level=0).transform(lambda x: np.max(x, axis=0))

    def time_transform_lambda_max_wide(self):
        self.df_wide.groupby(level=0).transform(lambda x: np.max(x, axis=0))

    def time_transform_multi_key1(self):
        self.df1.groupby(["jim", "joe"])["jolie"].transform("max")

    def time_transform_multi_key2(self):
        self.df2.groupby(["jim", "joe"])["jolie"].transform("max")

    def time_transform_multi_key3(self):
        self.df3.groupby(["jim", "joe"])["jolie"].transform("max")

    def time_transform_multi_key4(self):
        self.df4.groupby(["jim", "joe"])["jolie"].transform("max")


class Sample:
    def setup(self):
        N = 10**3
        self.df = DataFrame({"a": np.zeros(N)})
        self.groups = np.arange(0, N)
        self.weights = np.ones(N)

    def time_sample(self):
        self.df.groupby(self.groups).sample(n=1)

    def time_sample_weights(self):
        self.df.groupby(self.groups).sample(n=1, weights=self.weights)


from .pandas_vb_common import setup  # noqa: F401 isort:skip
