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

import os
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
    from modin.config import NPartitions

    NPARTITIONS = NPartitions.get()
except ImportError:
    NPARTITIONS = pd.DEFAULT_NPARTITIONS

try:
    from modin.config import TestDatasetSize, AsvImplementation

    ASV_USE_IMPL = AsvImplementation.get()
    ASV_DATASET_SIZE = TestDatasetSize.get() or "Small"
except ImportError:
    # The same benchmarking code can be run for different versions of Modin, so in
    # case of an error importing important variables, we'll just use predefined values
    ASV_USE_IMPL = os.environ.get("MODIN_ASV_USE_IMPL", "modin")
    ASV_DATASET_SIZE = os.environ.get("MODIN_TEST_DATASET_SIZE", "Small")

assert ASV_USE_IMPL in ("modin", "pandas")

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

IMPL = {
    "modin": pd,
    "pandas": pandas,
}


def execute(df):
    "Make sure the calculations are done."
    return df.shape, df.dtypes


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
        execute(IMPL[ASV_USE_IMPL].concat([self.df1, self.df2], axis=axis, join=how))


class TimeAppend:
    param_names = ["shapes", "sort"]
    params = [
        BINARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
        [False, True],
    ]

    def setup(self, shapes, sort):
        self.df1 = generate_dataframe(
            ASV_USE_IMPL, "int", *shapes[0], RAND_LOW, RAND_HIGH
        )
        self.df2 = generate_dataframe(
            ASV_USE_IMPL, "int", *shapes[1], RAND_LOW, RAND_HIGH
        )
        if sort:
            self.df1.columns = self.df1.columns[::-1]

    def time_append(self, shapes, sort):
        execute(self.df1.append(self.df2, sort=sort))


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


class TimeDrop:
    param_names = ["shape", "axis", "drop_ncols"]
    params = [
        UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
        [0, 1],
        [1, 0.8],
    ]

    def setup(self, shape, axis, drop_ncols):
        self.df = generate_dataframe(ASV_USE_IMPL, "int", *shape, RAND_LOW, RAND_HIGH)
        drop_count = (
            int(len(self.df.axes[axis]) * drop_ncols)
            if isinstance(drop_ncols, float)
            else drop_ncols
        )
        self.labels = self.df.axes[axis][:drop_count]

    def time_drop(self, shape, axis, drop_ncols):
        execute(self.df.drop(self.labels, axis))


class TimeHead:
    param_names = ["shape", "head_count"]
    params = [
        UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
        [5, 0.8],
    ]

    def setup(self, shape, head_count):
        self.df = generate_dataframe(ASV_USE_IMPL, "int", *shape, RAND_LOW, RAND_HIGH)
        self.head_count = (
            int(head_count * len(self.df.index))
            if isinstance(head_count, float)
            else head_count
        )

    def time_head(self, shape, head_count):
        execute(self.df.head(self.head_count))


class TimeFillna:
    param_names = ["shape", "limit", "inplace"]
    params = [UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE], [None, 0.8], [False, True]]

    def setup(self, shape, limit, inplace):
        pd = IMPL[ASV_USE_IMPL]
        columns = [f"col{x}" for x in range(shape[1])]
        self.df = pd.DataFrame(np.nan, index=pd.RangeIndex(shape[0]), columns=columns)
        self.limit = int(limit * shape[0]) if limit else None

    def time_fillna(self, shape, limit, inplace):
        kw = {"value": 0.0, "limit": self.limit, "inplace": inplace}
        if inplace:
            self.df.fillna(**kw)
            execute(self.df)
        else:
            execute(self.df.fillna(**kw))


class BaseTimeValueCounts:
    subset_params = {
        "all": lambda shape: shape[1],
        "half": lambda shape: shape[1] // 2,
    }

    def setup(self, shape, subset="all"):
        try:
            subset = self.subset_params[subset]
        except KeyError:
            raise KeyError(
                f"Invalid value for 'subset={subset}'. Allowed: {list(self.subset_params.keys())}"
            )
        ncols = subset(shape)
        self.df, _ = generate_dataframe(
            ASV_USE_IMPL,
            "int",
            *shape,
            RAND_LOW,
            RAND_HIGH,
            groupby_ncols=ncols,
            count_groups=GROUPBY_NGROUPS[ASV_DATASET_SIZE],
        )
        self.subset = self.df.columns[:ncols].tolist()


class TimeValueCountsFrame(BaseTimeValueCounts):
    param_names = ["shape", "subset"]
    params = [UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE], ["all", "half"]]

    def time_value_counts(self, *args, **kwargs):
        execute(self.df.value_counts(subset=self.subset))


class TimeValueCountsSeries(BaseTimeValueCounts):
    param_names = ["shape", "bins"]
    params = [UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE], [None, 3]]

    def setup(self, shape, bins):
        super().setup(shape=shape)
        self.df = self.df.iloc[:, 0]

    def time_value_counts(self, shape, bins):
        execute(self.df.value_counts(bins=bins))


class TimeAstype:
    param_names = ["shape", "dtype", "astype_ncolumns"]
    params = [
        UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
        ["float64", "category"],
        ["one", "all"],
    ]

    def setup(self, shape, dtype, astype_ncolumns):
        self.df = generate_dataframe(ASV_USE_IMPL, "int", *shape, RAND_LOW, RAND_HIGH)
        if astype_ncolumns == "all":
            self.astype_arg = dtype
        elif astype_ncolumns == "one":
            self.astype_arg = {"col1": dtype}
        else:
            raise ValueError("astype_ncolumns: {astype_ncolumns} isn't supported")

    def time_astype(self, shape, dtype, astype_ncolumns):
        execute(self.df.astype(self.astype_arg))
