# Licensed to Modin Development Team under one or more contributor license agreements.
# See the NOTICE file distributed with this work for additional information regarding
# copyright ownership.  The Modin Development Team licenses this file to you under the
# Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import modin.pandas as pd
import numpy as np
from .utils import generate_dataframe, RAND_LOW, RAND_HIGH

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
        numpy_data = np.random.randint(0, 100, size=(rows, cols))
        self.df_left = pd.DataFrame(numpy_data, columns=columns)
        self.df_right = pd.DataFrame(numpy_data, columns=columns)

    def time_join(self, rows_cols, how):
        self.df_left.join(self.df_right, how=how, lsuffix="left_")


class TimeMerge:
    param_names = ["impl", "data_type", "data_size", "how", "sort"]
    params = [
        ["modin", "pandas"],
        ["int"],
        [
            (5000, 5000, 5000, 5000),
            (10, 1_000_00, 10, 1_000_00),
            (1_000_00, 10, 1_000_00, 10),
        ],
        ["left", "right", "outer", "inner"],
        [False, True],
    ]

    def setup(self, impl, data_type, data_size, how, sort):
        self.df1 = generate_dataframe(
            impl, data_type, data_size[0], data_size[1], RAND_LOW, RAND_HIGH
        )
        self.df2 = generate_dataframe(
            impl, data_type, data_size[2], data_size[3], RAND_LOW, RAND_HIGH
        )

    def time_merge(self, impl, data_type, data_size, how, sort):
        self.df1.merge(self.df2, on=self.df1.columns[0], how=how, sort=sort)


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

    def time_apply_axis_1(self, rows_cols):
        self.df.apply(lambda df: df.sum(), axis=1)
