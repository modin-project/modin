# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import modin.pandas as pd
import numpy as np

pd.DEFAULT_NPARTITIONS = 4


class TimeGroupBy:
    param_names = ["rows_cols"]
    params = [
        [
            (100, 1000),
            (10000, 1000),
        ]
    ]

    def setup(self, rows_cols):
        rows, cols = rows_cols
        # workaround for #2482
        columns = [str(x) for x in range(cols)]
        self.df = pd.DataFrame(
            np.random.randint(0, 100, size=(rows, cols)), columns=columns
        )

    # add case for multiple by
    def time_groupby_sum(self, rows_cols):
        self.df.groupby(by="1").sum()

    def time_groupby_mean(self, rows_cols):
        self.df.groupby(by="1").mean()

    def time_groupby_count(self, rows_cols):
        self.df.groupby(by="1").count()


class TimeJoin:
    param_names = ["rows_cols", "how"]
    params = [
        [
            (100, 1000),
            (10000, 1000),
        ],
        ["outer", "inner", "left", "right"],
    ]

    def setup(self, rows_cols, how):
        rows, cols = rows_cols
        # workaround for #2482
        columns = [str(x) for x in range(cols)]
        numpy_data = np.random.randint(0, 100, size=(rows, cols)), columns=columns
        self.df_left = pd.DataFrame(numpy_data)
        self.df_right = pd.DataFrame(numpy_data)

    def time_join(self, rows_cols, how):
        self.df_left.join(self.df_right, how=how, lsuffix="left_")


class TimeMerge:
    param_names = ["rows_cols", "how"]
    params = [
        [
            (100, 1000),
            (10000, 1000),
        ],
        ["outer", "inner", "left", "right"],
    ]

    def setup(self, rows_cols, how):
        rows, cols = rows_cols
        # workaround for #2482
        columns = [str(x) for x in range(cols)]
        numpy_data = np.random.randint(0, 100, size=(rows, cols)), columns=columns
        self.df_left = pd.DataFrame(numpy_data)
        self.df_right = pd.DataFrame(numpy_data)

    def time_merge(self, rows_cols, how):
        self.df_left.merge(self.df_right, how=how, left_index=True, right_index=True)


class TimeArithmetic:
    param_names = ["rows_cols"]
    params = [
        [
            (100, 1000),
            (10000, 1000),
        ]
    ]

    def setup(self, rows_cols):
        rows, cols = rows_cols
        # workaround for #2482
        columns = [str(x) for x in range(cols)]
        self.df = pd.DataFrame(
            np.random.randint(0, 100, size=(rows, cols)), columns=columns
        )

    def time_transpose_lazy(self, rows_cols):
        self.df.T

    def time_transpose(self, rows_cols):
        repr(self.df.T)

    def time_sum(self, rows_cols):
        self.df.sum()

    def time_sum_axis_1(self, rows_cols):
        self.df.sum(axis=1)

    def time_median(self, rows_cols):
        self.df.median()

    def time_median_axis_1(self, rows_cols):
        self.df.median(axis=1)

    def time_nunique(self, rows_cols):
        self.df.nunique()

    def time_nunique_axis_1(self, rows_cols):
        self.df.nunique(axis=1)

    def time_apply(self, rows_cols):
        self.df.apply(lambda df: df.sum())

    def time_apply(self, rows_cols):
        self.df.apply(lambda df: df.sum(), axis=1)
