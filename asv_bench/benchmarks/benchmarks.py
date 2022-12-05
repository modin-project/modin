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

"""General Modin benchmarks."""

# define `MODIN_CPUS` env var to control the number of partitions
# it should be defined before modin.pandas import (in case of using os.environ)

# define `MODIN_ASV_USE_IMPL` env var to choose library for using in performance
# measurements

import numpy as np
import pandas._testing as tm
import math

from .utils import (
    generate_dataframe,
    gen_nan_data,
    RAND_LOW,
    RAND_HIGH,
    random_string,
    random_columns,
    random_booleans,
    GROUPBY_NGROUPS,
    IMPL,
    execute,
    translator_groupby_ngroups,
    get_benchmark_shapes,
    trigger_import,
)


class BaseTimeGroupBy:
    def setup(self, shape, ngroups=5, groupby_ncols=1):
        ngroups = translator_groupby_ngroups(ngroups, shape)
        self.df, self.groupby_columns = generate_dataframe(
            "int",
            *shape,
            RAND_LOW,
            RAND_HIGH,
            groupby_ncols,
            count_groups=ngroups,
        )


class TimeGroupByMultiColumn(BaseTimeGroupBy):
    param_names = ["shape", "ngroups", "groupby_ncols"]
    params = [
        get_benchmark_shapes("TimeGroupByMultiColumn"),
        GROUPBY_NGROUPS,
        [6],
    ]

    def time_groupby_agg_quan(self, *args, **kwargs):
        execute(self.df.groupby(by=self.groupby_columns).agg("quantile"))

    def time_groupby_agg_mean(self, *args, **kwargs):
        execute(self.df.groupby(by=self.groupby_columns).apply(lambda df: df.mean()))


class TimeGroupByDefaultAggregations(BaseTimeGroupBy):
    param_names = ["shape", "ngroups"]
    params = [
        get_benchmark_shapes("TimeGroupByDefaultAggregations"),
        GROUPBY_NGROUPS,
    ]

    def time_groupby_count(self, *args, **kwargs):
        execute(self.df.groupby(by=self.groupby_columns).count())

    def time_groupby_size(self, *args, **kwargs):
        execute(self.df.groupby(by=self.groupby_columns).size())

    def time_groupby_sum(self, *args, **kwargs):
        execute(self.df.groupby(by=self.groupby_columns).sum())

    def time_groupby_mean(self, *args, **kwargs):
        execute(self.df.groupby(by=self.groupby_columns).mean())


class TimeGroupByDictionaryAggregation(BaseTimeGroupBy):
    param_names = ["shape", "ngroups", "operation_type"]
    params = [
        get_benchmark_shapes("TimeGroupByDictionaryAggregation"),
        GROUPBY_NGROUPS,
        ["reduce", "aggregation"],
    ]
    operations = {
        "reduce": ["sum", "count", "prod"],
        "aggregation": ["quantile", "std", "median"],
    }

    def setup(self, shape, ngroups, operation_type):
        super().setup(shape, ngroups)
        self.cols_to_agg = self.df.columns[1:4]
        operations = self.operations[operation_type]
        self.agg_dict = {
            c: operations[i % len(operations)] for i, c in enumerate(self.cols_to_agg)
        }

    def time_groupby_dict_agg(self, *args, **kwargs):
        execute(self.df.groupby(by=self.groupby_columns).agg(self.agg_dict))


class TimeJoin:
    param_names = ["shapes", "how", "sort"]
    params = [
        get_benchmark_shapes("TimeJoin"),
        ["left", "inner"],
        [False],
    ]

    def setup(self, shapes, how, sort):
        self.df1 = generate_dataframe("int", *shapes[0], RAND_LOW, RAND_HIGH)
        self.df2 = generate_dataframe("int", *shapes[1], RAND_LOW, RAND_HIGH)

    def time_join(self, shapes, how, sort):
        # join dataframes on index to get the predictable shape
        execute(self.df1.join(self.df2, how=how, lsuffix="left_", sort=sort))


class TimeJoinStringIndex:
    param_names = ["shapes", "sort"]
    params = [
        get_benchmark_shapes("TimeJoinStringIndex"),
        [True, False],
    ]

    def setup(self, shapes, sort):
        assert shapes[0] % 100 == 0, "implementation restriction"
        level1 = tm.makeStringIndex(10).values
        level2 = tm.makeStringIndex(shapes[0] // 100).values
        codes1 = np.arange(10).repeat(shapes[0] // 100)
        codes2 = np.tile(np.arange(shapes[0] // 100), 10)
        index2 = IMPL.MultiIndex(levels=[level1, level2], codes=[codes1, codes2])
        self.df_multi = IMPL.DataFrame(
            np.random.randn(len(index2), 4), index=index2, columns=["A", "B", "C", "D"]
        )

        self.key1 = np.tile(level1.take(codes1), 10)
        self.key2 = np.tile(level2.take(codes2), 10)
        self.df = generate_dataframe("int", *shapes, RAND_LOW, RAND_HIGH)
        # just to keep source shape
        self.df = self.df.drop(columns=self.df.columns[-2:])
        self.df["key1"] = self.key1
        self.df["key2"] = self.key2
        execute(self.df)

        self.df_key1 = IMPL.DataFrame(
            np.random.randn(len(level1), 4), index=level1, columns=["A", "B", "C", "D"]
        )
        self.df_key2 = IMPL.DataFrame(
            np.random.randn(len(level2), 4), index=level2, columns=["A", "B", "C", "D"]
        )

    def time_join_dataframe_index_multi(self, shapes, sort):
        execute(self.df.join(self.df_multi, on=["key1", "key2"], sort=sort))

    def time_join_dataframe_index_single_key_bigger(self, shapes, sort):
        execute(self.df.join(self.df_key2, on="key2", sort=sort))

    def time_join_dataframe_index_single_key_small(self, shapes, sort):
        execute(self.df.join(self.df_key1, on="key1", sort=sort))


class TimeMerge:
    param_names = ["shapes", "how", "sort"]
    params = [
        get_benchmark_shapes("TimeMerge"),
        ["left", "inner"],
        [True, False],
    ]

    def setup(self, shapes, how, sort):
        self.df1 = generate_dataframe("int", *shapes[0], RAND_LOW, RAND_HIGH)
        self.df2 = generate_dataframe("int", *shapes[1], RAND_LOW, RAND_HIGH)

    def time_merge(self, shapes, how, sort):
        # merge dataframes by index to get the predictable shape
        execute(
            self.df1.merge(
                self.df2, left_index=True, right_index=True, how=how, sort=sort
            )
        )

    def time_merge_default(self, shapes, how, sort):
        execute(IMPL.merge(self.df1, self.df2, how=how, sort=sort))

    def time_merge_dataframe_empty_right(self, shapes, how, sort):
        # Getting an empty dataframe using `iloc` should be very fast,
        # so the impact on the time of the merge operation should be negligible.
        execute(IMPL.merge(self.df1, self.df2.iloc[:0], how=how, sort=sort))

    def time_merge_dataframe_empty_left(self, shapes, how, sort):
        # Getting an empty dataframe using `iloc` should be very fast,
        # so the impact on the time of the merge operation should be negligible.
        execute(IMPL.merge(self.df1.iloc[:0], self.df2, how=how, sort=sort))


class TimeMergeCategoricals:
    param_names = ["shapes", "data_type"]
    params = [
        get_benchmark_shapes("MergeCategoricals"),
        ["object", "category"],
    ]

    def setup(self, shapes, data_type):
        assert len(shapes) == 2
        assert shapes[1] == 2
        size = (shapes[0],)
        self.left = IMPL.DataFrame(
            {
                "X": np.random.choice(range(0, 10), size=size),
                "Y": np.random.choice(["one", "two", "three"], size=size),
            }
        )

        self.right = IMPL.DataFrame(
            {
                "X": np.random.choice(range(0, 10), size=size),
                "Z": np.random.choice(["jjj", "kkk", "sss"], size=size),
            }
        )

        if data_type == "category":
            self.left = self.left.assign(Y=self.left["Y"].astype("category"))
            execute(self.left)
            self.right = self.right.assign(Z=self.right["Z"].astype("category"))
            execute(self.right)

    def time_merge_categoricals(self, shapes, data_type):
        execute(IMPL.merge(self.left, self.right, on="X"))


class TimeConcat:
    param_names = ["shapes", "how", "axis", "ignore_index"]
    params = [
        get_benchmark_shapes("TimeConcat"),
        ["inner", "outer"],
        [0, 1],
        [True, False],
    ]

    def setup(self, shapes, how, axis, ignore_index):
        self.df1 = generate_dataframe("int", *shapes[0], RAND_LOW, RAND_HIGH)
        self.df2 = generate_dataframe("int", *shapes[1], RAND_LOW, RAND_HIGH)

    def time_concat(self, shapes, how, axis, ignore_index):
        execute(
            IMPL.concat(
                [self.df1, self.df2], axis=axis, join=how, ignore_index=ignore_index
            )
        )


class TimeAppend:
    param_names = ["shapes", "sort"]
    params = [
        get_benchmark_shapes("TimeAppend"),
        [False, True],
    ]

    def setup(self, shapes, sort):
        self.df1 = generate_dataframe("int", *shapes[0], RAND_LOW, RAND_HIGH)
        self.df2 = generate_dataframe("int", *shapes[1], RAND_LOW, RAND_HIGH)
        if sort:
            self.df1.columns = self.df1.columns[::-1]

    def time_append(self, shapes, sort):
        execute(self.df1.append(self.df2, sort=sort))


class TimeBinaryOp:
    param_names = ["shapes", "binary_op", "axis"]
    params = [
        get_benchmark_shapes("TimeBinaryOp"),
        ["mul"],
        [0, 1],
    ]

    def setup(self, shapes, binary_op, axis):
        self.df1 = generate_dataframe("int", *shapes[0], RAND_LOW, RAND_HIGH)
        self.df2 = generate_dataframe("int", *shapes[1], RAND_LOW, RAND_HIGH)
        self.op = getattr(self.df1, binary_op)

    def time_binary_op(self, shapes, binary_op, axis):
        execute(self.op(self.df2, axis=axis))


class TimeBinaryOpSeries:
    param_names = ["shapes", "binary_op"]
    params = [
        get_benchmark_shapes("TimeBinaryOpSeries"),
        ["mul"],
    ]

    def setup(self, shapes, binary_op):
        df1 = generate_dataframe("int", *shapes[0], RAND_LOW, RAND_HIGH)
        df2 = generate_dataframe("int", *shapes[1], RAND_LOW, RAND_HIGH)
        self.series1 = df1[df1.columns[0]]
        self.series2 = df2[df2.columns[0]]
        self.op = getattr(self.series1, binary_op)
        execute(self.series1)
        execute(self.series2)

    def time_binary_op_series(self, shapes, binary_op):
        execute(self.op(self.series2))


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
        self.df = generate_dataframe("int", *shape, RAND_LOW, RAND_HIGH).copy()
        self.loc, self.iloc = self.get_loc(
            self.df, loc, item_length=item_length, axis=1
        )

        self.item = self.df[self.loc] + 1
        self.item_raw = self.item.to_numpy()
        if not is_equal_indices:
            self.item.index = reversed(self.item.index)


class TimeSetItem(BaseTimeSetItem):
    params = [
        get_benchmark_shapes("TimeSetItem"),
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
        get_benchmark_shapes("TimeInsert"),
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
        get_benchmark_shapes("TimeArithmetic"),
        [0, 1],
    ]

    def setup(self, shape, axis):
        self.df = generate_dataframe("int", *shape, RAND_LOW, RAND_HIGH)

    def time_sum(self, shape, axis):
        execute(self.df.sum(axis=axis))

    def time_count(self, shape, axis):
        execute(self.df.count(axis=axis))

    def time_median(self, shape, axis):
        execute(self.df.median(axis=axis))

    def time_nunique(self, shape, axis):
        execute(self.df.nunique(axis=axis))

    def time_apply(self, shape, axis):
        execute(self.df.apply(lambda df: df.sum(), axis=axis))

    def time_mean(self, shape, axis):
        execute(self.df.mean(axis=axis))

    def time_mode(self, shape, axis):
        execute(self.df.mode(axis=axis))

    def time_add(self, shape, axis):
        execute(self.df.add(2, axis=axis))

    def time_mul(self, shape, axis):
        execute(self.df.mul(2, axis=axis))

    def time_mod(self, shape, axis):
        execute(self.df.mod(2, axis=axis))

    def time_abs(self, shape, axis):
        execute(self.df.abs())

    def time_aggregate(self, shape, axis):
        execute(self.df.aggregate(lambda df: df.sum(), axis=axis))

    def time_is_in(self, shape, axis):
        execute(self.df.isin([0, 2]))

    def time_transpose(self, shape, axis):
        execute(self.df.transpose())


class TimeSortValues:
    param_names = ["shape", "columns_number", "ascending_list"]
    params = [
        get_benchmark_shapes("TimeSortValues"),
        [1, 2, 10, 100],
        [False, True],
    ]

    def setup(self, shape, columns_number, ascending_list):
        self.df = generate_dataframe("int", *shape, RAND_LOW, RAND_HIGH)
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
        get_benchmark_shapes("TimeDrop"),
        [0, 1],
        [1, 0.8],
    ]

    def setup(self, shape, axis, drop_ncols):
        self.df = generate_dataframe("int", *shape, RAND_LOW, RAND_HIGH)
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
        get_benchmark_shapes("TimeHead"),
        [5, 0.8],
    ]

    def setup(self, shape, head_count):
        self.df = generate_dataframe("int", *shape, RAND_LOW, RAND_HIGH)
        self.head_count = (
            int(head_count * len(self.df.index))
            if isinstance(head_count, float)
            else head_count
        )

    def time_head(self, shape, head_count):
        execute(self.df.head(self.head_count))


class TimeTail:
    param_names = ["shape", "tail_count"]
    params = [
        get_benchmark_shapes("TimeTail"),
        [5, 0.8],
    ]

    def setup(self, shape, tail_count):
        self.df = generate_dataframe("int", *shape, RAND_LOW, RAND_HIGH)
        self.tail_count = (
            int(tail_count * len(self.df.index))
            if isinstance(tail_count, float)
            else tail_count
        )

    def time_tail(self, shape, tail_count):
        execute(self.df.tail(self.tail_count))


class TimeExplode:
    param_names = ["shape"]
    params = [
        get_benchmark_shapes("TimeExplode"),
    ]

    def setup(self, shape):
        self.df = generate_dataframe(
            "int", *shape, RAND_LOW, RAND_HIGH, gen_unique_key=True
        )

    def time_explode(self, shape):
        execute(self.df.explode("col1"))


class TimeFillnaSeries:
    param_names = ["value_type", "shape", "limit"]
    params = [
        ["scalar", "dict", "Series"],
        get_benchmark_shapes("TimeFillnaSeries"),
        [None, 0.8],
    ]

    def setup(self, value_type, shape, limit):
        self.series = gen_nan_data(*shape)

        if value_type == "scalar":
            self.value = 18.19
        elif value_type == "dict":
            self.value = {k: k * 1.23 for k in range(shape[0])}
        elif value_type == "Series":
            self.value = IMPL.Series(
                [k * 1.23 for k in range(shape[0])], index=IMPL.RangeIndex(shape[0])
            )
        else:
            assert False
        limit = int(limit * shape[0]) if limit else None
        self.kw = {"value": self.value, "limit": limit}

    def time_fillna(self, value_type, shape, limit):
        execute(self.series.fillna(**self.kw))

    def time_fillna_inplace(self, value_type, shape, limit):
        self.series.fillna(inplace=True, **self.kw)
        execute(self.series)


class TimeFillnaDataFrame:
    param_names = ["value_type", "shape", "limit"]
    params = [
        ["scalar", "dict", "DataFrame", "Series"],
        get_benchmark_shapes("TimeFillnaDataFrame"),
        [None, 0.8],
    ]

    def setup(self, value_type, shape, limit):
        self.df = gen_nan_data(*shape)
        columns = self.df.columns

        if value_type == "scalar":
            self.value = 18.19
        elif value_type == "dict":
            self.value = {k: i * 1.23 for i, k in enumerate(columns)}
        elif value_type == "Series":
            self.value = IMPL.Series(
                [i * 1.23 for i in range(len(columns))], index=columns
            )
        elif value_type == "DataFrame":
            self.value = IMPL.DataFrame(
                {
                    k: [i + j * 1.23 for j in range(shape[0])]
                    for i, k in enumerate(columns)
                },
                index=IMPL.RangeIndex(shape[0]),
                columns=columns,
            )
        else:
            assert False
        limit = int(limit * shape[0]) if limit else None
        self.kw = {"value": self.value, "limit": limit}

    def time_fillna(self, value_type, shape, limit):
        execute(self.df.fillna(**self.kw))

    def time_fillna_inplace(self, value_type, shape, limit):
        self.df.fillna(inplace=True, **self.kw)
        execute(self.df)


class BaseTimeValueCounts:
    def setup(self, shape, ngroups=5, subset=1):
        ngroups = translator_groupby_ngroups(ngroups, shape)
        self.df, self.subset = generate_dataframe(
            "int",
            *shape,
            RAND_LOW,
            RAND_HIGH,
            groupby_ncols=subset,
            count_groups=ngroups,
        )


class TimeValueCountsFrame(BaseTimeValueCounts):
    param_names = ["shape", "ngroups", "subset"]
    params = [
        get_benchmark_shapes("TimeValueCountsFrame"),
        GROUPBY_NGROUPS,
        [2, 10],
    ]

    def time_value_counts(self, *args, **kwargs):
        execute(self.df.value_counts(subset=self.subset))


class TimeValueCountsSeries(BaseTimeValueCounts):
    param_names = ["shape", "ngroups", "bins"]
    params = [
        get_benchmark_shapes("TimeValueCountsSeries"),
        GROUPBY_NGROUPS,
        [None, 3],
    ]

    def setup(self, shape, ngroups, bins):
        super().setup(ngroups=ngroups, shape=shape)
        self.df = self.df[self.subset[0]]

    def time_value_counts(self, shape, ngroups, bins):
        execute(self.df.value_counts(bins=bins))


class TimeIndexing:
    param_names = ["shape", "indexer_type"]
    params = [
        get_benchmark_shapes("TimeIndexing"),
        [
            "bool_array",
            "bool_series",
            "scalar",
            "slice",
            "continuous_slice",
            "numpy_array_take_all_values",
            "python_list_take_10_values",
            "function",
        ],
    ]

    indexer_getters = {
        "bool_array": lambda df: np.array([False, True] * (len(df) // 2)),
        # This boolean-Series is a projection of the source frame, it shouldn't
        # be reimported or triggered to execute:
        "bool_series": lambda df: df.iloc[:, 0] > 50,
        "scalar": lambda df: len(df) // 2,
        "slice": lambda df: slice(0, len(df), 2),
        "continuous_slice": lambda df: slice(len(df) // 2),
        "numpy_array_take_all_values": lambda df: np.arange(len(df)),
        "python_list_take_10_values": lambda df: list(range(min(10, len(df)))),
        "function": lambda df: (lambda df: df.index[::-2]),
    }

    def setup(self, shape, indexer_type):
        self.df = generate_dataframe("int", *shape, RAND_LOW, RAND_HIGH)
        trigger_import(self.df)

        self.indexer = self.indexer_getters[indexer_type](self.df)
        if isinstance(self.indexer, (IMPL.Series, IMPL.DataFrame)):
            # HACK: Triggering `dtypes` meta-data computation in advance,
            # so it won't affect the `loc/iloc` time:
            self.indexer.dtypes

    def time_iloc(self, shape, indexer_type):
        # Pandas doesn't implement `df.iloc[series boolean_mask]` and raises an exception on it.
        # Replacing this with the semantically equivalent construction:
        if indexer_type != "bool_series":
            execute(self.df.iloc[self.indexer])
        else:
            execute(self.df[self.indexer])

    def time_loc(self, shape, indexer_type):
        execute(self.df.loc[self.indexer])


class TimeIndexingColumns:
    param_names = ["shape"]
    params = [get_benchmark_shapes("TimeIndexing")]

    def setup(self, shape):
        self.df = generate_dataframe("int", *shape, RAND_LOW, RAND_HIGH)
        trigger_import(self.df)
        self.numeric_indexer = [0, 1]
        self.labels_indexer = self.df.columns[self.numeric_indexer].tolist()

    def time_iloc(self, shape):
        execute(self.df.iloc[:, self.numeric_indexer])

    def time_loc(self, shape):
        execute(self.df.loc[:, self.labels_indexer])

    def time___getitem__(self, shape):
        execute(self.df[self.labels_indexer])


class TimeMultiIndexing:
    param_names = ["shape"]
    params = [get_benchmark_shapes("TimeMultiIndexing")]

    def setup(self, shape):
        df = generate_dataframe("int", *shape, RAND_LOW, RAND_HIGH)

        index = IMPL.MultiIndex.from_product(
            [df.index[: shape[0] // 2], ["bar", "foo"]]
        )
        columns = IMPL.MultiIndex.from_product(
            [df.columns[: shape[1] // 2], ["buz", "fuz"]]
        )

        df.index = index
        df.columns = columns

        self.df = df.sort_index(axis=1)

    def time_multiindex_loc(self, shape):
        execute(
            self.df.loc[
                self.df.index[2] : self.df.index[-2],
                self.df.columns[2] : self.df.columns[-2],
            ]
        )


class TimeResetIndex:
    param_names = ["shape", "drop", "level"]
    params = [
        get_benchmark_shapes("TimeResetIndex"),
        [False, True],
        [None, "level_1"],
    ]

    def setup(self, shape, drop, level):
        self.df = generate_dataframe("int", *shape, RAND_LOW, RAND_HIGH)

        if level:
            index = IMPL.MultiIndex.from_product(
                [self.df.index[: shape[0] // 2], ["bar", "foo"]],
                names=["level_1", "level_2"],
            )
            self.df.index = index

    def time_reset_index(self, shape, drop, level):
        execute(self.df.reset_index(drop=drop, level=level))


class TimeAstype:
    param_names = ["shape", "dtype", "astype_ncolumns"]
    params = [
        get_benchmark_shapes("TimeAstype"),
        ["float64", "category"],
        ["one", "all"],
    ]

    def setup(self, shape, dtype, astype_ncolumns):
        self.df = generate_dataframe("int", *shape, RAND_LOW, RAND_HIGH)
        if astype_ncolumns == "all":
            self.astype_arg = dtype
        elif astype_ncolumns == "one":
            self.astype_arg = {"col1": dtype}
        else:
            raise ValueError(f"astype_ncolumns: {astype_ncolumns} isn't supported")

    def time_astype(self, shape, dtype, astype_ncolumns):
        execute(self.df.astype(self.astype_arg))


class TimeDescribe:
    param_names = ["shape"]
    params = [
        get_benchmark_shapes("TimeDescribe"),
    ]

    def setup(self, shape):
        self.df = generate_dataframe("int", *shape, RAND_LOW, RAND_HIGH)

    def time_describe(self, shape):
        execute(self.df.describe())


class TimeProperties:
    param_names = ["shape"]
    params = [
        get_benchmark_shapes("TimeProperties"),
    ]

    def setup(self, shape):
        self.df = generate_dataframe("int", *shape, RAND_LOW, RAND_HIGH)

    def time_shape(self, shape):
        return self.df.shape

    def time_columns(self, shape):
        return self.df.columns

    def time_index(self, shape):
        return self.df.index


class TimeIndexingNumericSeries:

    param_names = ["shape", "dtype", "index_structure"]
    params = [
        get_benchmark_shapes("TimeIndexingNumericSeries"),
        (np.int64, np.uint64, np.float64),
        ("unique_monotonic_inc", "nonunique_monotonic_inc"),
    ]

    def setup(self, shape, dtype, index_structure):
        N = shape[0]
        indices = {
            "unique_monotonic_inc": IMPL.Index(range(N), dtype=dtype),
            "nonunique_monotonic_inc": IMPL.Index(
                list(range(N // 100)) + [(N // 100) - 1] + list(range(N // 100, N - 1)),
                dtype=dtype,
            ),
        }
        self.data = IMPL.Series(np.random.rand(N), index=indices[index_structure])
        self.array = np.arange(N // 2)
        self.index_to_query = N // 2
        self.array_list = self.array.tolist()
        execute(self.data)

    def time_getitem_scalar(self, shape, index, index_structure):
        # not calling execute as execute function fails for scalar
        self.data[self.index_to_query]

    def time_getitem_slice(self, shape, index, index_structure):
        execute(self.data[: self.index_to_query])

    def time_getitem_list_like(self, shape, index, index_structure):
        execute(self.data[[self.index_to_query]])

    def time_getitem_array(self, shape, index, index_structure):
        execute(self.data[self.array])

    def time_getitem_lists(self, shape, index, index_structure):
        execute(self.data[self.array_list])

    def time_iloc_array(self, shape, index, index_structure):
        execute(self.data.iloc[self.array])

    def time_iloc_list_like(self, shape, index, index_structure):
        execute(self.data.iloc[[self.index_to_query]])

    def time_iloc_scalar(self, shape, index, index_structure):
        # not calling execute as execute function fails for scalar
        self.data.iloc[self.index_to_query]

    def time_iloc_slice(self, shape, index, index_structure):
        execute(self.data.iloc[: self.index_to_query])

    def time_loc_array(self, shape, index, index_structure):
        execute(self.data.loc[self.array])

    def time_loc_list_like(self, shape, index, index_structure):
        execute(self.data.loc[[self.index_to_query]])

    def time_loc_scalar(self, shape, index, index_structure):
        self.data.loc[self.index_to_query]

    def time_loc_slice(self, shape, index, index_structure):
        execute(self.data.loc[: self.index_to_query])


class TimeReindex:
    param_names = ["shape"]
    params = [get_benchmark_shapes("TimeReindex")]

    def setup(self, shape):
        rows, cols = shape
        rng = IMPL.date_range(start="1/1/1970", periods=rows, freq="1min")
        self.df = IMPL.DataFrame(
            np.random.rand(rows, cols), index=rng, columns=range(cols)
        )
        self.df["foo"] = "bar"
        self.rng_subset = IMPL.Index(rng[::2])
        self.df2 = IMPL.DataFrame(
            index=range(rows), data=np.random.rand(rows, cols), columns=range(cols)
        )
        level1 = tm.makeStringIndex(rows).values.repeat(cols)
        level2 = np.tile(tm.makeStringIndex(cols).values, rows)
        index = IMPL.MultiIndex.from_arrays([level1, level2])
        self.s = IMPL.Series(np.random.randn(rows * cols), index=index)
        self.s_subset = self.s[::2]
        self.s_subset_no_cache = self.s[::2].copy()

        mi = IMPL.MultiIndex.from_product([rng, range(100)])
        self.s2 = IMPL.Series(np.random.randn(len(mi)), index=mi)
        self.s2_subset = self.s2[::2].copy()
        execute(self.df), execute(self.df2)
        execute(self.s), execute(self.s_subset)
        execute(self.s2), execute(self.s2_subset)
        execute(self.s_subset_no_cache)

    def time_reindex_dates(self, shape):
        execute(self.df.reindex(self.rng_subset))

    def time_reindex_columns(self, shape):
        execute(self.df2.reindex(columns=self.df.columns[1:5]))

    def time_reindex_multiindex_with_cache(self, shape):
        # MultiIndex._values gets cached (pandas specific)
        execute(self.s.reindex(self.s_subset.index))

    def time_reindex_multiindex_no_cache(self, shape):
        # Copy to avoid MultiIndex._values getting cached (pandas specific)
        execute(self.s.reindex(self.s_subset_no_cache.index.copy()))

    def time_reindex_multiindex_no_cache_dates(self, shape):
        # Copy to avoid MultiIndex._values getting cached (pandas specific)
        execute(self.s2_subset.reindex(self.s2.index.copy()))


class TimeReindexMethod:

    params = [
        get_benchmark_shapes("TimeReindexMethod"),
        ["pad", "backfill"],
        [IMPL.date_range, IMPL.period_range],
    ]
    param_names = ["shape", "method", "constructor"]

    def setup(self, shape, method, constructor):
        N = shape[0]
        self.idx = constructor("1/1/2000", periods=N, freq="1min")
        self.ts = IMPL.Series(np.random.randn(N), index=self.idx)[::2]
        execute(self.ts)

    def time_reindex_method(self, shape, method, constructor):
        execute(self.ts.reindex(self.idx, method=method))


class TimeFillnaMethodSeries:

    params = [get_benchmark_shapes("TimeFillnaMethodSeries"), ["pad", "backfill"]]
    param_names = ["shape", "method"]

    def setup(self, shape, method):
        N = shape[0]
        self.idx = IMPL.date_range("1/1/2000", periods=N, freq="1min")
        ts = IMPL.Series(np.random.randn(N), index=self.idx)[::2]
        self.ts_reindexed = ts.reindex(self.idx)
        self.ts_float32 = self.ts_reindexed.astype("float32")
        execute(self.ts_reindexed), execute(self.ts_float32)

    def time_reindexed(self, shape, method):
        execute(self.ts_reindexed.fillna(method=method))

    def time_float_32(self, shape, method):
        execute(self.ts_float32.fillna(method=method))


class TimeFillnaMethodDataframe:

    params = [get_benchmark_shapes("TimeFillnaMethodDataframe"), ["pad", "backfill"]]
    param_names = ["shape", "method"]

    def setup(self, shape, method):
        self.idx = IMPL.date_range("1/1/2000", periods=shape[0], freq="1min")
        df_ts = IMPL.DataFrame(np.random.randn(*shape), index=self.idx)[::2]
        self.df_ts_reindexed = df_ts.reindex(self.idx)
        self.df_ts_float32 = self.df_ts_reindexed.astype("float32")
        execute(self.df_ts_reindexed), execute(self.df_ts_float32)

    def time_reindexed(self, shape, method):
        execute(self.df_ts_reindexed.fillna(method=method))

    def time_float_32(self, shape, method):
        execute(self.df_ts_float32.fillna(method=method))


class TimeLevelAlign:
    params = [get_benchmark_shapes("TimeLevelAlign")]
    param_names = ["shape"]

    def setup(self, shape):
        row1, col1 = shape[0]
        row2, col2 = shape[1]
        Nroot = round(math.sqrt(row1))
        N = Nroot * Nroot
        self.index = IMPL.MultiIndex(
            levels=[np.arange(10), np.arange(Nroot), np.arange(Nroot)],
            codes=[
                np.arange(10).repeat(N),
                np.tile(np.arange(Nroot).repeat(Nroot), 10),
                np.tile(np.tile(np.arange(Nroot), Nroot), 10),
            ],
        )
        self.df = IMPL.DataFrame(
            np.random.randn(len(self.index), col1), index=self.index
        )
        self.df_level = IMPL.DataFrame(np.random.randn(row2, col2))
        execute(self.df)
        execute(self.df_level)

    def time_align_level(self, shape):
        left, right = self.df.align(self.df_level, level=1, copy=False)
        execute(left)
        execute(right)

    def time_reindex_level(self, shape):
        execute(self.df_level.reindex(self.index, level=1))


class TimeDropDuplicates:

    params = [get_benchmark_shapes("TimeDropDuplicates")]
    param_names = ["shape"]

    def setup(self, shape):
        N = shape[0]
        K = shape[1]
        key1 = tm.makeStringIndex(N).values.repeat(K)
        key2 = tm.makeStringIndex(N).values.repeat(K)
        self.df = IMPL.DataFrame(
            {"key1": key1, "key2": key2, "value": np.random.randn(N * K)}
        )
        self.df_nan = self.df.copy()
        self.df_nan.iloc[:N, :] = np.nan
        self.s = IMPL.Series(np.random.randint(0, N // 10, size=N))
        self.s_str = IMPL.Series(np.tile(tm.makeStringIndex(N // 10).values, 10))
        key1 = np.random.randint(0, K, size=N)
        self.df_int = IMPL.DataFrame({"key1": key1})
        execute(self.df), execute(self.df_nan)
        execute(self.s), execute(self.s_str)
        execute(self.df_int)

    def time_frame_drop_dups(self, shape):
        execute(self.df.drop_duplicates(["key1", "key2"]))

    def time_frame_drop_dups_inplace(self, shape):
        self.df.drop_duplicates(["key1", "key2"], inplace=True)
        execute(self.df)

    def time_frame_drop_dups_na(self, shape):
        execute(self.df_nan.drop_duplicates(["key1", "key2"]))

    def time_frame_drop_dups_na_inplace(self, shape):
        self.df_nan.drop_duplicates(["key1", "key2"], inplace=True)
        execute(self.df_nan)

    def time_series_drop_dups_int(self, shape):
        execute(self.s.drop_duplicates())

    def time_series_drop_dups_int_inplace(self, shape):
        self.s.drop_duplicates(inplace=True)
        execute(self.s)

    def time_series_drop_dups_string(self, shape):
        execute(self.s_str.drop_duplicates())

    def time_series_drop_dups_string_inplace(self, shape):
        self.s_str.drop_duplicates(inplace=True)
        execute(self.s_str)

    def time_frame_drop_dups_int(self, shape):
        execute(self.df_int.drop_duplicates())

    def time_frame_drop_dups_int_inplace(self, shape):
        self.df_int.drop_duplicates(inplace=True)
        execute(self.df_int)


from .utils import setup  # noqa: E402, F401
