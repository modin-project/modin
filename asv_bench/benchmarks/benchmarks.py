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

# define `MODIN_CPUS` env var to control the number of partitions
# it should be defined before modin.pandas import (in case of using os.environ)

# define `MODIN_ASV_USE_IMPL` env var to choose library for using in performance
# measurements

import modin.pandas as pd
import numpy as np
import pandas

from .utils import generate_dataframe, RAND_LOW, RAND_HIGH, random_string

try:
    from modin.config import TestDatasetSize, AsvImplementation

    ASV_USE_IMPL = AsvImplementation.get()
    ASV_DATASET_SIZE = TestDatasetSize.get()
except ImportError:
    # The same benchmarking code can be run for different versions of Modin, so in
    # case of an error importing important variables, we'll just use predefined values
    ASV_USE_IMPL = "modin"
    ASV_DATASET_SIZE = "Big" if pd.DEFAULT_NPARTITIONS >= 32 else "Small"

if ASV_DATASET_SIZE == "Big":
    BINARY_OP_DATA_SIZE = [
        (5000, 5000, 5000, 5000),
        # the case extremely inefficient
        # (20, 500_000, 10, 1_000_000),
        (500_000, 20, 1_000_000, 10),
    ]
    UNARY_OP_DATA_SIZE = [
        (5000, 5000),
        # the case extremely inefficient
        # (10, 1_000_000),
        (1_000_000, 10),
    ]
else:
    BINARY_OP_DATA_SIZE = [
        (256, 256, 256, 256),
        (20, 10_000, 10, 25_000),
        (10_000, 20, 25_000, 10),
    ]
    UNARY_OP_DATA_SIZE = [
        (256, 256),
        (10, 10_000),
        (10_000, 10),
    ]


def execute(df):
    "Make sure the calculations are done."
    return df.shape


class BaseTimeGroupBy:
    def setup(self, data_size, ncols=1):
        self.df = generate_dataframe(
            ASV_USE_IMPL, "int", data_size[1], data_size[0], RAND_LOW, RAND_HIGH
        )
        self.groupby_columns = self.df.columns[:ncols].tolist()


class TimeMultiColumnGroupby(BaseTimeGroupBy):
    param_names = ["data_size", "ncols"]
    params = [UNARY_OP_DATA_SIZE, [6]]

    def time_groupby_agg_quan(self, data_size, ncols):
        execute(self.df.groupby(by=self.groupby_columns).agg("quantile"))

    def time_groupby_agg_mean(self, data_size, ncols):
        execute(self.df.groupby(by=self.groupby_columns).apply(lambda df: df.mean()))


class TimeGroupByDefaultAggregations(BaseTimeGroupBy):
    param_names = ["data_size"]
    params = [
        UNARY_OP_DATA_SIZE,
    ]

    def time_groupby_count(self, data_size):
        execute(self.df.groupby(by=self.groupby_columns).count())

    def time_groupby_size(self, data_size):
        execute(self.df.groupby(by=self.groupby_columns).size())

    def time_groupby_sum(self, data_size):
        execute(self.df.groupby(by=self.groupby_columns).sum())

    def time_groupby_mean(self, data_size):
        execute(self.df.groupby(by=self.groupby_columns).mean())


class TimeGroupByDictionaryAggregation(BaseTimeGroupBy):
    param_names = ["data_size", "operation_type"]
    params = [UNARY_OP_DATA_SIZE, ["reduction", "aggregation"]]
    operations = {
        "reduction": ["sum", "count", "prod"],
        "aggregation": ["quantile", "std", "median"],
    }

    def setup(self, data_size, operation_type):
        super().setup(data_size)
        self.cols_to_agg = self.df.columns[1:4]
        operations = self.operations[operation_type]
        self.agg_dict = {
            c: operations[i % len(operations)] for i, c in enumerate(self.cols_to_agg)
        }

    def time_groupby_dict_agg(self, data_size, operation_type):
        execute(self.df.groupby(by=self.groupby_columns).agg(self.agg_dict))


class TimeJoin:
    param_names = ["data_size", "how", "sort"]
    params = [
        BINARY_OP_DATA_SIZE,
        ["left", "inner"],
        [False],
    ]

    def setup(self, data_size, how, sort):
        self.df1 = generate_dataframe(
            ASV_USE_IMPL, "int", data_size[1], data_size[0], RAND_LOW, RAND_HIGH
        )
        self.df2 = generate_dataframe(
            ASV_USE_IMPL, "int", data_size[3], data_size[2], RAND_LOW, RAND_HIGH
        )

    def time_join(self, data_size, how, sort):
        execute(
            self.df1.join(
                self.df2, on=self.df1.columns[0], how=how, lsuffix="left_", sort=sort
            )
        )


class TimeMerge:
    param_names = ["data_size", "how", "sort"]
    params = [
        BINARY_OP_DATA_SIZE,
        ["left", "inner"],
        [False],
    ]

    def setup(self, data_size, how, sort):
        self.df1 = generate_dataframe(
            ASV_USE_IMPL, "int", data_size[1], data_size[0], RAND_LOW, RAND_HIGH
        )
        self.df2 = generate_dataframe(
            ASV_USE_IMPL, "int", data_size[3], data_size[2], RAND_LOW, RAND_HIGH
        )

    def time_merge(self, data_size, how, sort):
        execute(self.df1.merge(self.df2, on=self.df1.columns[0], how=how, sort=sort))


class TimeConcat:
    param_names = ["data_size", "how", "axis"]
    params = [
        BINARY_OP_DATA_SIZE,
        ["inner"],
        [0, 1],
    ]

    def setup(self, data_size, how, axis):
        # shape for generate_dataframe: first - ncols, second - nrows
        self.df1 = generate_dataframe(
            ASV_USE_IMPL, "int", data_size[1], data_size[0], RAND_LOW, RAND_HIGH
        )
        self.df2 = generate_dataframe(
            ASV_USE_IMPL, "int", data_size[3], data_size[2], RAND_LOW, RAND_HIGH
        )

    def time_concat(self, data_size, how, axis):
        if ASV_USE_IMPL == "modin":
            execute(pd.concat([self.df1, self.df2], axis=axis, join=how))
        elif ASV_USE_IMPL == "pandas":
            execute(pandas.concat([self.df1, self.df2], axis=axis, join=how))
        else:
            raise NotImplementedError


class TimeBinaryOp:
    param_names = ["data_size", "binary_op", "axis"]
    params = [
        BINARY_OP_DATA_SIZE,
        ["mul"],
        [0, 1],
    ]

    def setup(self, data_size, binary_op, axis):
        # shape for generate_dataframe: first - ncols, second - nrows
        self.df1 = generate_dataframe(
            ASV_USE_IMPL, "int", data_size[1], data_size[0], RAND_LOW, RAND_HIGH
        )
        self.df2 = generate_dataframe(
            ASV_USE_IMPL, "int", data_size[3], data_size[2], RAND_LOW, RAND_HIGH
        )
        self.op = getattr(self.df1, binary_op)

    def time_binary_op(self, data_size, binary_op, axis):
        execute(self.op(self.df2, axis=axis))


class BaseTimeSetItem:
    param_names = ["data_size", "item_length", "loc", "is_equal_indices"]

    @staticmethod
    def get_loc(df, loc, axis, item_length):
        locs_dict = {
            "zero": 0,
            "middle": len(df.axes[axis]) // 2,
            "last": len(df.axes[axis]) - 1,
        }
        base_loc = locs_dict[loc]
        range_based_loc = np.arange(
            base_loc, min(len(df.axes[axis]), base_loc + item_length)
        )
        return (
            (df.axes[axis][base_loc], base_loc)
            if len(range_based_loc) == 1
            else (df.axes[axis][range_based_loc], range_based_loc)
        )

    def setup(self, data_size, item_length, loc, is_equal_indices):
        self.df = generate_dataframe(
            ASV_USE_IMPL, "int", data_size[1], data_size[0], RAND_LOW, RAND_HIGH
        ).copy()
        self.loc, self.iloc = self.get_loc(
            self.df, loc, item_length=item_length, axis=1
        )

        self.item = self.df[self.loc] + 1
        self.item_raw = self.item.to_numpy()
        if not is_equal_indices:
            self.item.index = reversed(self.item.index)


class TimeSetItem(BaseTimeSetItem):
    params = [
        UNARY_OP_DATA_SIZE,
        [1],
        ["zero", "middle", "last"],
        [True, False],
    ]

    def time_setitem_qc(self, *args, **kwargs):
        self.df[self.loc] = self.item
        execute(self.df)

    def time_setitem_raw(self, *args, **kwargs):
        self.df[self.loc] = self.item_raw
        execute(self.df)


class TimeInsert(BaseTimeSetItem):
    params = [
        UNARY_OP_DATA_SIZE,
        [1],
        ["zero", "middle", "last"],
        [True, False],
    ]

    def time_insert_qc(self, *args, **kwargs):
        self.df.insert(loc=self.iloc, column=random_string(), value=self.item)
        execute(self.df)

    def time_insert_raw(self, *args, **kwargs):
        self.df.insert(loc=self.iloc, column=random_string(), value=self.item_raw)
        execute(self.df)


class TimeArithmetic:
    param_names = ["data_size", "axis"]
    params = [
        UNARY_OP_DATA_SIZE,
        [0, 1],
    ]

    def setup(self, data_size, axis):
        self.df = generate_dataframe(
            ASV_USE_IMPL, "int", data_size[1], data_size[0], RAND_LOW, RAND_HIGH
        )

    def time_sum(self, data_size, axis):
        execute(self.df.sum(axis=axis))

    def time_median(self, data_size, axis):
        execute(self.df.median(axis=axis))

    def time_nunique(self, data_size, axis):
        execute(self.df.nunique(axis=axis))

    def time_apply(self, data_size, axis):
        execute(self.df.apply(lambda df: df.sum(), axis=axis))

    def time_mean(self, data_size, axis):
        execute(self.df.mean(axis=axis))
