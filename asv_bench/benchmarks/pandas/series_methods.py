from datetime import datetime

import numpy as np

from modin.pandas import (
    NA,
    Index,
    NaT,
    Series,
    date_range,
)

import pandas._testing as tm



class NSort:

    params = ["first", "last", "all"]
    param_names = ["keep"]

    def setup(self, keep):
        self.s = Series(np.random.randint(1, 10, 100000))

    def time_nlargest(self, keep):
        self.s.nlargest(3, keep=keep)

    def time_nsmallest(self, keep):
        self.s.nsmallest(3, keep=keep)


class Dropna:

    params = ["int", "datetime"]
    param_names = ["dtype"]

    def setup(self, dtype):
        N = 10**6
        data = {
            "int": np.random.randint(1, 10, N),
            "datetime": date_range("2000-01-01", freq="S", periods=N),
        }
        self.s = Series(data[dtype])
        if dtype == "datetime":
            self.s[np.random.randint(1, N, 100)] = NaT

    def time_dropna(self, dtype):
        self.s.dropna()


class SearchSorted:

    goal_time = 0.2
    params = [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "str",
    ]
    param_names = ["dtype"]

    def setup(self, dtype):
        N = 10**5
        data = np.array([1] * N + [2] * N + [3] * N).astype(dtype)
        self.s = Series(data)

    def time_searchsorted(self, dtype):
        key = "2" if dtype == "str" else 2
        self.s.searchsorted(key)


class Map:

    params = (["dict", "Series", "lambda"], ["object", "category", "int"])
    param_names = "mapper"

    def setup(self, mapper, dtype):
        map_size = 1000
        map_data = Series(map_size - np.arange(map_size), dtype=dtype)

        # construct mapper
        if mapper == "Series":
            self.map_data = map_data
        elif mapper == "dict":
            self.map_data = map_data.to_dict()
        elif mapper == "lambda":
            map_dict = map_data.to_dict()
            self.map_data = lambda x: map_dict[x]
        else:
            raise NotImplementedError

        self.s = Series(np.random.randint(0, map_size, 10000), dtype=dtype)

    def time_map(self, mapper, *args, **kwargs):
        self.s.map(self.map_data)


class Clip:
    params = [50, 1000, 10**5]
    param_names = ["n"]

    def setup(self, n):
        self.s = Series(np.random.randn(n))

    def time_clip(self, n):
        self.s.clip(0, 1)


class ClipDt:
    def setup(self):
        dr = date_range("20220101", periods=100_000, freq="s", tz="UTC")
        self.clipper_dt = dr[0:1_000].repeat(100)
        self.s = Series(dr)

    def time_clip(self):
        self.s.clip(upper=self.clipper_dt)


class ValueCounts:

    params = [[10**3, 10**4, 10**5], ["int", "uint", "float", "object"]]
    param_names = ["N", "dtype"]

    def setup(self, N, dtype):
        self.s = Series(np.random.randint(0, N, size=10 * N)).astype(dtype)

    def time_value_counts(self, N, dtype):
        self.s.value_counts()


class ValueCountsEA:

    params = [[10**3, 10**4, 10**5], [True, False]]
    param_names = ["N", "dropna"]

    def setup(self, N, dropna):
        self.s = Series(np.random.randint(0, N, size=10 * N), dtype="Int64")
        self.s.loc[1] = NA

    def time_value_counts(self, N, dropna):
        self.s.value_counts(dropna=dropna)


class ValueCountsObjectDropNAFalse:

    params = [10**3, 10**4, 10**5]
    param_names = ["N"]

    def setup(self, N):
        self.s = Series(np.random.randint(0, N, size=10 * N)).astype("object")

    def time_value_counts(self, N):
        self.s.value_counts(dropna=False)


class Mode:

    params = [[10**3, 10**4, 10**5], ["int", "uint", "float", "object"]]
    param_names = ["N", "dtype"]

    def setup(self, N, dtype):
        self.s = Series(np.random.randint(0, N, size=10 * N)).astype(dtype)

    def time_mode(self, N, dtype):
        self.s.mode()


class ModeObjectDropNAFalse:

    params = [10**3, 10**4, 10**5]
    param_names = ["N"]

    def setup(self, N):
        self.s = Series(np.random.randint(0, N, size=10 * N)).astype("object")

    def time_mode(self, N):
        self.s.mode(dropna=False)


class Dir:
    def setup(self):
        self.s = Series(index=tm.makeStringIndex(10000))

    def time_dir_strings(self):
        dir(self.s)


class SeriesGetattr:
    # https://github.com/pandas-dev/pandas/issues/19764
    def setup(self):
        self.s = Series(1, index=date_range("2012-01-01", freq="s", periods=10**6))

    def time_series_datetimeindex_repr(self):
        getattr(self.s, "a", None)


class All:

    params = [[10**3, 10**6], ["fast", "slow"], ["bool", "boolean"]]
    param_names = ["N", "case", "dtype"]

    def setup(self, N, case, dtype):
        val = case != "fast"
        self.s = Series([val] * N, dtype=dtype)

    def time_all(self, N, case, dtype):
        self.s.all()


class Any:

    params = [[10**3, 10**6], ["fast", "slow"], ["bool", "boolean"]]
    param_names = ["N", "case", "dtype"]

    def setup(self, N, case, dtype):
        val = case == "fast"
        self.s = Series([val] * N, dtype=dtype)

    def time_any(self, N, case, dtype):
        self.s.any()


class NanOps:

    params = [
        [
            "var",
            "mean",
            "median",
            "max",
            "min",
            "sum",
            "std",
            "sem",
            "argmax",
            "skew",
            "kurt",
            "prod",
        ],
        [10**3, 10**6],
        ["int8", "int32", "int64", "float64", "Int64", "boolean"],
    ]
    param_names = ["func", "N", "dtype"]

    def setup(self, func, N, dtype):
        if func == "argmax" and dtype in {"Int64", "boolean"}:
            # Skip argmax for nullable int since this doesn't work yet (GH-24382)
            raise NotImplementedError
        self.s = Series([1] * N, dtype=dtype)
        self.func = getattr(self.s, func)

    def time_func(self, func, N, dtype):
        self.func()


class Rank:

    param_names = ["dtype"]
    params = [
        ["int", "uint", "float", "object"],
    ]

    def setup(self, dtype):
        self.s = Series(np.random.randint(0, 1000, size=100000), dtype=dtype)

    def time_rank(self, dtype):
        self.s.rank()


from .pandas_vb_common import setup  # noqa: F401 isort:skip
