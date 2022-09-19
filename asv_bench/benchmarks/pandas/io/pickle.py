import numpy as np

from modin.pandas import (
    DataFrame,
    date_range,
    read_pickle,
)

from ..pandas_vb_common import (
    BaseIO,
)
import pandas._testing as tm


class Pickle(BaseIO):
    def setup(self):
        self.fname = "__test__.pkl"
        N = 100000
        C = 5
        self.df = DataFrame(
            np.random.randn(N, C),
            columns=[f"float{i}" for i in range(C)],
            index=date_range("20000101", periods=N, freq="H"),
        )
        self.df["object"] = tm.makeStringIndex(N)
        self.df.to_pickle(self.fname)

    def time_read_pickle(self):
        read_pickle(self.fname)

    def time_write_pickle(self):
        self.df.to_pickle(self.fname)

    def peakmem_read_pickle(self):
        read_pickle(self.fname)

    def peakmem_write_pickle(self):
        self.df.to_pickle(self.fname)


from ..pandas_vb_common import setup  # noqa: F401 isort:skip
