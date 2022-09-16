from itertools import product
import string

import numpy as np

import modin.pandas as pd
from modin.pandas import (
    DataFrame,
    MultiIndex,
    date_range,
)
from pandas.api.types import CategoricalDtype


class SimpleReshape:
    def setup(self):
        arrays = [np.arange(100).repeat(100), np.roll(np.tile(np.arange(100), 100), 25)]
        index = MultiIndex.from_arrays(arrays)
        self.df = DataFrame(np.random.randn(10000, 4), index=index)
        self.udf = self.df.unstack(1)

    def time_stack(self):
        self.udf.stack()

    def time_unstack(self):
        self.df.unstack(1)


class ReshapeExtensionDtype:

    params = ["datetime64[ns, US/Pacific]", "Period[s]"]
    param_names = ["dtype"]

    def setup(self, dtype):
        lev = pd.Index(list("ABCDEFGHIJ"))
        ri = pd.Index(range(1000))
        mi = MultiIndex.from_product([lev, ri], names=["foo", "bar"])

        index = date_range("2016-01-01", periods=10000, freq="s", tz="US/Pacific")
        if dtype == "Period[s]":
            index = index.tz_localize(None).to_period("s")

        ser = pd.Series(index, index=mi)
        df = ser.unstack("bar")
        # roundtrips -> df.stack().equals(ser)

        self.ser = ser
        self.df = df

    def time_stack(self, dtype):
        self.df.stack()

    def time_unstack_fast(self, dtype):
        # last level -> doesn't have to make copies
        self.ser.unstack("bar")

    def time_unstack_slow(self, dtype):
        # first level -> must make copies
        self.ser.unstack("foo")

    def time_transpose(self, dtype):
        self.df.T


class Unstack:

    params = ["int", "category"]

    def setup(self, dtype):
        m = 100
        n = 1000

        levels = np.arange(m)
        index = MultiIndex.from_product([levels] * 2)
        columns = np.arange(n)
        if dtype == "int":
            values = np.arange(m * m * n).reshape(m * m, n)
            self.df = DataFrame(values, index, columns)
        else:
            # the category branch is ~20x slower than int. So we
            # cut down the size a bit. Now it's only ~3x slower.
            n = 50
            columns = columns[:n]
            indices = np.random.randint(0, 52, size=(m * m, n))
            values = np.take(list(string.ascii_letters), indices)
            values = [pd.Categorical(v) for v in values.T]

            self.df = DataFrame(
                {i: cat for i, cat in enumerate(values)}, index, columns
            )

        self.df2 = self.df.iloc[:-1]

    def time_full_product(self, dtype):
        self.df.unstack()

    def time_without_last_row(self, dtype):
        self.df2.unstack()


class Explode:
    param_names = ["n_rows", "max_list_length"]
    params = [[100, 1000, 10000], [3, 5, 10]]

    def setup(self, n_rows, max_list_length):

        data = [np.arange(np.random.randint(max_list_length)) for _ in range(n_rows)]
        self.series = pd.Series(data)

    def time_explode(self, n_rows, max_list_length):
        self.series.explode()


from .pandas_vb_common import setup  # noqa: F401 isort:skip
