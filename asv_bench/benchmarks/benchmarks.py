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

import modin.pandas as pd
from modin.config import TestDatasetSize
from .utils import generate_dataframe, RAND_LOW, RAND_HIGH

pd.DEFAULT_NPARTITIONS = 4

if TestDatasetSize.get() == "Big":
    MERGE_DATA_SIZE = [
        (5000, 5000, 5000, 5000),
        (10, 1_000_000, 10, 1_000_000),
        (1_000_000, 10, 1_000_000, 10),
    ]
    GROUPBY_DATA_SIZE = [
        (5000, 5000),
        (10, 1_000_000),
        (1_000_000, 10),
    ]
else:
    MERGE_DATA_SIZE = [
        (2000, 100, 2000, 100),
    ]
    GROUPBY_DATA_SIZE = [
        (2000, 100),
    ]

JOIN_DATA_SIZE = MERGE_DATA_SIZE
ARITHMETIC_DATA_SIZE = GROUPBY_DATA_SIZE


class TimeGroupBy:
    param_names = ["impl", "data_type", "data_size"]
    params = [
        ["modin", "pandas"],
        ["int"],
        GROUPBY_DATA_SIZE,
    ]

    def setup(self, impl, data_type, data_size):
        self.df = generate_dataframe(
            impl, data_type, data_size[0], data_size[1], RAND_LOW, RAND_HIGH
        )

    def time_groupby_sum(self, impl, data_type, data_size):
        self.df.groupby(by=self.df.columns[0]).sum()

    def time_groupby_mean(self, impl, data_type, data_size):
        self.df.groupby(by=self.df.columns[0]).mean()

    def time_groupby_count(self, impl, data_type, data_size):
        self.df.groupby(by=self.df.columns[0]).count()


class TimeJoin:
    param_names = ["impl", "data_type", "data_size", "how", "sort"]
    params = [
        ["modin", "pandas"],
        ["int"],
        JOIN_DATA_SIZE,
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

    def time_join(self, impl, data_type, data_size, how, sort):
        self.df1.join(
            self.df2, on=self.df1.columns[0], how=how, lsuffix="left_", sort=sort
        )


class TimeMerge:
    param_names = ["impl", "data_type", "data_size", "how", "sort"]
    params = [
        ["modin", "pandas"],
        ["int"],
        MERGE_DATA_SIZE,
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
    param_names = ["impl", "data_type", "data_size", "axis"]
    params = [
        ["modin", "pandas"],
        ["int"],
        ARITHMETIC_DATA_SIZE,
        [0, 1],
    ]

    def setup(self, impl, data_type, data_size, axis):
        self.df = generate_dataframe(
            impl, data_type, data_size[0], data_size[1], RAND_LOW, RAND_HIGH
        )

    def time_sum(self, impl, data_type, data_size, axis):
        self.df.sum(axis=axis)

    def time_median(self, impl, data_type, data_size, axis):
        self.df.median(axis=axis)

    def time_nunique(self, impl, data_type, data_size, axis):
        self.df.nunique(axis=axis)

    def time_apply(self, impl, data_type, data_size, axis):
        self.df.apply(lambda df: df.sum(), axis=axis)
