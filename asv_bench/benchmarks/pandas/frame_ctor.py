import numpy as np

import modin.pandas as pd
from modin.pandas import (
    NA,
    Categorical,
    DataFrame,
    Float64Dtype,
    MultiIndex,
    Series,
    Timestamp,
    date_range,
)

import pandas._testing as tm

try:
    from pandas.tseries.offsets import (
        Hour,
        Nano,
    )
except ImportError:
    # For compatibility with older versions
    from pandas.core.datetools import (
        Hour,
        Nano,
    )


class FromDicts:
    def setup(self):
        N, K = 5000, 50
        self.index = tm.makeStringIndex(N)
        self.columns = tm.makeStringIndex(K)
        frame = DataFrame(np.random.randn(N, K), index=self.index, columns=self.columns)
        self.data = frame.to_dict()
        self.dict_list = frame.to_dict(orient="records")
        self.data2 = {i: {j: float(j) for j in range(100)} for i in range(2000)}

        # arrays which we won't consolidate
        self.dict_of_categoricals = {i: Categorical(np.arange(N)) for i in range(K)}

    def time_list_of_dict(self):
        DataFrame(self.dict_list)

    def time_nested_dict(self):
        DataFrame(self.data)

    def time_nested_dict_index(self):
        DataFrame(self.data, index=self.index)

    def time_nested_dict_columns(self):
        DataFrame(self.data, columns=self.columns)

    def time_nested_dict_index_columns(self):
        DataFrame(self.data, index=self.index, columns=self.columns)

    def time_nested_dict_int64(self):
        # nested dict, integer indexes, regression described in #621
        DataFrame(self.data2)

    def time_dict_of_categoricals(self):
        # dict of arrays that we won't consolidate
        DataFrame(self.dict_of_categoricals)


class FromSeries:
    def setup(self):
        mi = MultiIndex.from_product([range(100), range(100)])
        self.s = Series(np.random.randn(10000), index=mi)

    def time_mi_series(self):
        DataFrame(self.s)


class FromRecords:

    params = [None, 1000]
    param_names = ["nrows"]

    # Generators get exhausted on use, so run setup before every call
    number = 1
    repeat = (3, 250, 10)

    def setup(self, nrows):
        N = 100000
        self.gen = ((x, (x * 20), (x * 100)) for x in range(N))

    def time_frame_from_records_generator(self, nrows):
        # issue-6700
        self.df = DataFrame.from_records(self.gen, nrows=nrows)


class FromNDArray:
    def setup(self):
        N = 100000
        self.data = np.random.randn(N)

    def time_frame_from_ndarray(self):
        self.df = DataFrame(self.data)


class FromLists:

    goal_time = 0.2

    def setup(self):
        N = 1000
        M = 100
        self.data = [list(range(M)) for i in range(N)]

    def time_frame_from_lists(self):
        self.df = DataFrame(self.data)


class FromRange:

    goal_time = 0.2

    def setup(self):
        N = 1_000_000
        self.data = range(N)

    def time_frame_from_range(self):
        self.df = DataFrame(self.data)


class FromScalar:
    def setup(self):
        self.nrows = 100_000

    def time_frame_from_scalar_ea_float64(self):
        DataFrame(
            1.0,
            index=range(self.nrows),
            columns=list("abc"),
            dtype=Float64Dtype(),
        )

    def time_frame_from_scalar_ea_float64_na(self):
        DataFrame(
            NA,
            index=range(self.nrows),
            columns=list("abc"),
            dtype=Float64Dtype(),
        )


class From3rdParty:
    # GH#44616

    def setup(self):
        try:
            import torch
        except ImportError:
            raise NotImplementedError

        row = 700000
        col = 64
        self.val_tensor = torch.randn(row, col)

    def time_from_torch(self):
        DataFrame(self.val_tensor)


from .pandas_vb_common import setup  # noqa: F401 isort:skip
