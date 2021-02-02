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

from .utils import (
    generate_dataframe,
    RAND_LOW,
    RAND_HIGH,
    random_string,
    random_columns,
    random_booleans,
)

try:
    from modin import config  # noqa: F401

    OLD_MODIN_VERSION = False
except ImportError:
    OLD_MODIN_VERSION = True

try:
    from modin.config import NPartitions

    NPARTITIONS = NPartitions.get()
except ImportError:
    if not OLD_MODIN_VERSION:
        from modin.config import CpuCount

        NPARTITIONS = CpuCount.get()
    else:
        import os
        import multiprocessing

        NPARTITIONS = int(os.environ.get("MODIN_CPUS", multiprocessing.cpu_count()))

try:
    from modin.config import TestDatasetSize, AsvImplementation

    ASV_USE_IMPL = AsvImplementation.get()
    ASV_DATASET_SIZE = TestDatasetSize.get() or "Small"
except ImportError:
    # The same benchmarking code can be run for different versions of Modin, so in
    # case of an error importing important variables, we'll just use predefined values
    ASV_USE_IMPL = "modin"
    ASV_DATASET_SIZE = "Big" if NPARTITIONS >= 32 else "Small"

BINARY_OP_DATA_SIZE = {
    "Big": [
        ((5000, 5000), (5000, 5000)),
        # the case extremely inefficient
        # ((20, 500_000), (10, 1_000_000)),
        ((500_000, 20), (1_000_000, 10)),
    ],
    "Small": [
        ((250, 250), (250, 250)),
        ((20, 10_000), (10, 25_000)),
        ((10_000, 20), (25_000, 10)),
    ],
}

UNARY_OP_DATA_SIZE = {
    "Big": [
        (5000, 5000),
        # the case extremely inefficient
        # (10, 1_000_000),
        (1_000_000, 10),
    ],
    "Small": [
        (250, 250),
        (10, 10_000),
        (10_000, 10),
    ],
}

GROUPBY_NGROUPS = {
    "Big": 100,
    "Small": 5,
}


def execute(df):
    "Make sure the calculations are done."
    return df.shape


class BaseTimeGroupBy:
    def setup(self, shape, groupby_ncols=1):
        self.df, self.groupby_columns = generate_dataframe(
            ASV_USE_IMPL,
            "int",
            *shape,
            RAND_LOW,
            RAND_HIGH,
            groupby_ncols,
            count_groups=GROUPBY_NGROUPS[ASV_DATASET_SIZE],
        )


class TimeGroupByMultiColumn(BaseTimeGroupBy):
    param_names = ["shape", "groupby_ncols"]
    params = [
        UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
        [6],
    ]

    def time_groupby_agg_quan(self, shape, groupby_ncols):
        execute(self.df.groupby(by=self.groupby_columns).agg("quantile"))

    def time_groupby_agg_mean(self, shape, groupby_ncols):
        execute(self.df.groupby(by=self.groupby_columns).apply(lambda df: df.mean()))


class TimeGroupByDefaultAggregations(BaseTimeGroupBy):
    param_names = ["shape"]
    params = [
        UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
    ]

    def time_groupby_count(self, shape):
        execute(self.df.groupby(by=self.groupby_columns).count())

    def time_groupby_size(self, shape):
        execute(self.df.groupby(by=self.groupby_columns).size())

    def time_groupby_sum(self, shape):
        execute(self.df.groupby(by=self.groupby_columns).sum())

    def time_groupby_mean(self, shape):
        execute(self.df.groupby(by=self.groupby_columns).mean())


class TimeGroupByDictionaryAggregation(BaseTimeGroupBy):
    param_names = ["shape", "operation_type"]
    params = [UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE], ["reduction", "aggregation"]]
    operations = {
        "reduction": ["sum", "count", "prod"],
        "aggregation": ["quantile", "std", "median"],
    }

    def setup(self, shape, operation_type):
        super().setup(shape)
        self.cols_to_agg = self.df.columns[1:4]
        operations = self.operations[operation_type]
        self.agg_dict = {
            c: operations[i % len(operations)] for i, c in enumerate(self.cols_to_agg)
        }

    def time_groupby_dict_agg(self, shape, operation_type):
        execute(self.df.groupby(by=self.groupby_columns).agg(self.agg_dict))


class TimeJoin:
    param_names = ["shapes", "how", "sort"]
    params = [
        BINARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
        ["left", "inner"],
        [False],
    ]

    def setup(self, shapes, how, sort):
        self.df1 = generate_dataframe(
            ASV_USE_IMPL, "int", *shapes[0], RAND_LOW, RAND_HIGH
        )
        self.df2 = generate_dataframe(
            ASV_USE_IMPL, "int", *shapes[1], RAND_LOW, RAND_HIGH
        )

    def time_join(self, shapes, how, sort):
        # join dataframes on index to get the predictable shape
        execute(self.df1.join(self.df2, how=how, lsuffix="left_", sort=sort))


class TimeMerge:
    param_names = ["shapes", "how", "sort"]
    params = [
        BINARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
        ["left", "inner"],
        [False],
    ]

    def setup(self, shapes, how, sort):
        self.df1 = generate_dataframe(
            ASV_USE_IMPL, "int", *shapes[0], RAND_LOW, RAND_HIGH
        )
        self.df2 = generate_dataframe(
            ASV_USE_IMPL, "int", *shapes[1], RAND_LOW, RAND_HIGH
        )

    def time_merge(self, shapes, how, sort):
        # merge dataframes by index to get the predictable shape
        execute(
            self.df1.merge(
                self.df2, left_index=True, right_index=True, how=how, sort=sort
            )
        )


class TimeConcat:
    param_names = ["shapes", "how", "axis"]
    params = [
        BINARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
        ["inner"],
        [0, 1],
    ]

    def setup(self, shapes, how, axis):
        self.df1 = generate_dataframe(
            ASV_USE_IMPL, "int", *shapes[0], RAND_LOW, RAND_HIGH
        )
        self.df2 = generate_dataframe(
            ASV_USE_IMPL, "int", *shapes[1], RAND_LOW, RAND_HIGH
        )

    def time_concat(self, shapes, how, axis):
        if ASV_USE_IMPL == "modin":
            execute(pd.concat([self.df1, self.df2], axis=axis, join=how))
        elif ASV_USE_IMPL == "pandas":
            execute(pandas.concat([self.df1, self.df2], axis=axis, join=how))
        else:
            raise NotImplementedError


class TimeBinaryOp:
    param_names = ["shapes", "binary_op", "axis"]
    params = [
        BINARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
        ["mul"],
        [0, 1],
    ]

    def setup(self, shapes, binary_op, axis):
        self.df1 = generate_dataframe(
            ASV_USE_IMPL, "int", *shapes[0], RAND_LOW, RAND_HIGH
        )
        self.df2 = generate_dataframe(
            ASV_USE_IMPL, "int", *shapes[1], RAND_LOW, RAND_HIGH
        )
        self.op = getattr(self.df1, binary_op)

    def time_binary_op(self, shapes, binary_op, axis):
        execute(self.op(self.df2, axis=axis))


class BaseTimeSetItem:
    param_names = ["shape", "item_length", "loc", "is_equal_indices"]

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

    def setup(self, shape, item_length, loc, is_equal_indices):
        self.df = generate_dataframe(
            ASV_USE_IMPL, "int", *shape, RAND_LOW, RAND_HIGH
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
        UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
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
        UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
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
    param_names = ["shape", "axis"]
    params = [
        UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
        [0, 1],
    ]

    def setup(self, shape, axis):
        self.df = generate_dataframe(ASV_USE_IMPL, "int", *shape, RAND_LOW, RAND_HIGH)

    def time_sum(self, shape, axis):
        execute(self.df.sum(axis=axis))

    def time_median(self, shape, axis):
        execute(self.df.median(axis=axis))

    def time_nunique(self, shape, axis):
        execute(self.df.nunique(axis=axis))

    def time_apply(self, shape, axis):
        execute(self.df.apply(lambda df: df.sum(), axis=axis))

    def time_mean(self, shape, axis):
        execute(self.df.mean(axis=axis))


class TimeSortValues:
    param_names = ["shape", "columns_number", "ascending_list"]
    params = [
        UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
        [1, 2, 10, 100],
        [False, True],
    ]

    def setup(self, shape, columns_number, ascending_list):
        self.df = generate_dataframe(ASV_USE_IMPL, "int", *shape, RAND_LOW, RAND_HIGH)
        self.columns = random_columns(self.df.columns, columns_number)
        self.ascending = (
            random_booleans(columns_number)
            if ascending_list
            else bool(random_booleans(1)[0])
        )

    def time_sort_values(self, shape, columns_number, ascending_list):
        execute(self.df.sort_values(self.columns, ascending=self.ascending))
