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

"""General Modin on OmniSci backend benchmarks."""

from ..utils.utils import (
    generate_dataframe,
    gen_nan_data,
    ASV_USE_IMPL,
    ASV_DATASET_SIZE,
    GROUPBY_NGROUPS,
    IMPL,
    execute,
    translator_groupby_ngroups,
    random_columns,
    random_booleans,
    trigger_import,
    get_benchmark_shapes,
)

from .utils import (
    BINARY_OP_DATA_SIZE,
    UNARY_OP_DATA_SIZE,
    SERIES_DATA_SIZE,
    RAND_LOW,
    RAND_HIGH,
)


class TimeJoin:
    param_names = ["shape", "how"]
    params = [
        get_benchmark_shapes("omisci.TimeJoin", UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE]),
        ["left", "inner"],
    ]

    def setup(self, shape, how):
        self.df1 = generate_dataframe(ASV_USE_IMPL, "int", *shape, RAND_LOW, RAND_HIGH)
        self.df2 = generate_dataframe(ASV_USE_IMPL, "int", *shape, RAND_LOW, RAND_HIGH)
        trigger_import(self.df1, self.df2)

    def time_join(self, shape, how):
        # join dataframes on index to get the predictable shape
        execute(self.df1.join(self.df2, how=how, lsuffix="left_"))


class TimeMerge:
    param_names = ["shapes", "how"]
    params = [
        get_benchmark_shapes("omisci.TimeMerge", BINARY_OP_DATA_SIZE[ASV_DATASET_SIZE]),
        ["left"],
    ]

    def setup(self, shapes, how):
        self.df1 = generate_dataframe(
            ASV_USE_IMPL, "int", *shapes[0], RAND_LOW, RAND_HIGH
        )
        self.df2 = generate_dataframe(
            ASV_USE_IMPL, "int", *shapes[1], RAND_LOW, RAND_HIGH
        )
        trigger_import(self.df1, self.df2)

    def time_merge(self, shapes, how):
        # merging dataframes by index is not supported, therefore we merge by column
        # with arbitrary values, which leads to an unpredictable form of the operation result;
        # it's need to get the predictable shape to get consistent performance results
        execute(
            self.df1.merge(self.df2, on="col1", how=how, suffixes=("left_", "right_"))
        )


class TimeAppend:
    param_names = ["shapes"]
    params = [
        get_benchmark_shapes(
            "omisci.TimeAppend", BINARY_OP_DATA_SIZE[ASV_DATASET_SIZE]
        ),
    ]

    def setup(self, shapes):
        self.df1 = generate_dataframe(
            ASV_USE_IMPL, "int", *shapes[0], RAND_LOW, RAND_HIGH
        )
        self.df2 = generate_dataframe(
            ASV_USE_IMPL, "int", *shapes[1], RAND_LOW, RAND_HIGH
        )
        trigger_import(self.df1, self.df2)

    def time_append(self, shapes):
        execute(self.df1.append(self.df2))


class TimeBinaryOpDataFrame:
    param_names = ["shape", "binary_op"]
    params = [
        get_benchmark_shapes(
            "omisci.TimeBinaryOpDataFrame", UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE]
        ),
        ["mul"],
    ]

    def setup(self, shape, binary_op):
        self.df1 = generate_dataframe(ASV_USE_IMPL, "int", *shape, RAND_LOW, RAND_HIGH)
        trigger_import(self.df1)
        self.op = getattr(self.df1, binary_op)

    def time_mul_scalar(self, shape, binary_op):
        execute(self.op(2))

    def time_mul_dataframes(self, shape, binary_op):
        execute(self.op(self.df1))


class TimeBinaryOpSeries:
    param_names = ["shape", "binary_op"]
    params = [
        get_benchmark_shapes(
            "omisci.TimeBinaryOpSeries", SERIES_DATA_SIZE[ASV_DATASET_SIZE]
        ),
        ["mul"],
    ]

    def setup(self, shape, binary_op):
        self.series = generate_dataframe(
            ASV_USE_IMPL, "int", *shape, RAND_LOW, RAND_HIGH
        )["col0"]
        trigger_import(self.series)
        self.op = getattr(self.series, binary_op)

    def time_mul_series(self, shape, binary_op):
        execute(self.op(self.series))


class TimeArithmetic:
    param_names = ["shape"]
    params = [
        get_benchmark_shapes(
            "omisci.TimeArithmetic", UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE]
        ),
    ]

    def setup(self, shape):
        self.df = generate_dataframe(ASV_USE_IMPL, "int", *shape, RAND_LOW, RAND_HIGH)
        trigger_import(self.df)

    def time_sum(self, shape):
        execute(self.df.sum())

    def time_median(self, shape):
        execute(self.df.median())

    def time_nunique(self, shape):
        execute(self.df.nunique())

    def time_apply(self, shape):
        execute(self.df.apply(lambda df: df.sum()))

    def time_mean(self, shape):
        execute(self.df.mean())


class TimeSortValues:
    param_names = ["shape", "columns_number", "ascending_list"]
    params = [
        get_benchmark_shapes(
            "omisci.TimeSortValues", UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE]
        ),
        [1, 5],
        [False, True],
    ]

    def setup(self, shape, columns_number, ascending_list):
        self.df = generate_dataframe(ASV_USE_IMPL, "int", *shape, RAND_LOW, RAND_HIGH)
        trigger_import(self.df)
        self.columns = random_columns(self.df.columns, columns_number)
        self.ascending = (
            random_booleans(columns_number)
            if ascending_list
            else bool(random_booleans(1)[0])
        )

    def time_sort_values(self, shape, columns_number, ascending_list):
        execute(self.df.sort_values(self.columns, ascending=self.ascending))


class TimeDrop:
    param_names = ["shape", "drop_ncols"]
    params = [
        get_benchmark_shapes("omisci.TimeDrop", UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE]),
        [1, 0.8],
    ]

    def setup(self, shape, drop_ncols):
        self.df = generate_dataframe(ASV_USE_IMPL, "int", *shape, RAND_LOW, RAND_HIGH)
        trigger_import(self.df)
        drop_count = (
            int(len(self.df.axes[1]) * drop_ncols)
            if isinstance(drop_ncols, float)
            else drop_ncols
        )
        self.labels = self.df.axes[1][:drop_count]

    def time_drop(self, shape, drop_ncols):
        execute(self.df.drop(self.labels, axis=1))


class TimeHead:
    param_names = ["shape", "head_count"]
    params = [
        get_benchmark_shapes("omisci.TimeHead", UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE]),
        [5, 0.8],
    ]

    def setup(self, shape, head_count):
        self.df = generate_dataframe(ASV_USE_IMPL, "int", *shape, RAND_LOW, RAND_HIGH)
        trigger_import(self.df)
        self.head_count = (
            int(head_count * len(self.df.index))
            if isinstance(head_count, float)
            else head_count
        )

    def time_head(self, shape, head_count):
        execute(self.df.head(self.head_count))


class TimeFillna:
    param_names = ["value_type", "shape", "limit"]
    params = [
        ["scalar", "dict"],
        get_benchmark_shapes("omisci.TimeFillna", UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE]),
        [None],
    ]

    def setup(self, value_type, shape, limit):
        self.df = gen_nan_data(ASV_USE_IMPL, *shape)
        columns = self.df.columns
        trigger_import(self.df)

        value = self.create_fillna_value(value_type, columns)
        limit = int(limit * shape[0]) if limit else None
        self.kw = {"value": value, "limit": limit}

    def time_fillna(self, value_type, shape, limit):
        execute(self.df.fillna(**self.kw))

    @staticmethod
    def create_fillna_value(value_type: str, columns: list):
        if value_type == "scalar":
            value = 18.19
        elif value_type == "dict":
            value = {k: i * 1.23 for i, k in enumerate(columns)}
        else:
            assert False
        return value


class TimeValueCountsSeries:
    param_names = ["shape", "ngroups"]
    params = [
        get_benchmark_shapes(
            "omisci.TimeValueCountsSeries", SERIES_DATA_SIZE[ASV_DATASET_SIZE]
        ),
        GROUPBY_NGROUPS[ASV_DATASET_SIZE],
    ]

    def setup(self, shape, ngroups):
        ngroups = translator_groupby_ngroups(ngroups, shape)
        self.df, self.column_names = generate_dataframe(
            ASV_USE_IMPL,
            "int",
            *shape,
            RAND_LOW,
            RAND_HIGH,
            groupby_ncols=1,
            count_groups=ngroups,
        )
        self.series = self.df[self.column_names[0]]
        trigger_import(self.series)

    def time_value_counts(self, shape, ngroups):
        execute(self.series.value_counts())


class TimeIndexing:
    param_names = ["shape", "indexer_type"]
    params = [
        get_benchmark_shapes(
            "omisci.TimeIndexing", UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE]
        ),
        [
            "scalar",
            "bool",
            "slice",
            "list",
            "function",
        ],
    ]

    def setup(self, shape, indexer_type):
        self.df = generate_dataframe(ASV_USE_IMPL, "int", *shape, RAND_LOW, RAND_HIGH)
        trigger_import(self.df)
        self.indexer = {
            "bool": [False, True] * (shape[0] // 2),
            "scalar": shape[0] // 2,
            "slice": slice(0, shape[0], 2),
            "list": list(range(shape[0])),
            "function": lambda df: df.index[::-2],
        }[indexer_type]

    def time_iloc(self, shape, indexer_type):
        execute(self.df.iloc[self.indexer])

    def time_loc(self, shape, indexer_type):
        execute(self.df.loc[self.indexer])


class TimeResetIndex:
    param_names = ["shape", "drop", "level"]
    params = [
        get_benchmark_shapes(
            "omisci.TimeResetIndex", UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE]
        ),
        [False, True],
        [None, "level_1"],
    ]

    def setup(self, shape, drop, level):
        pd = IMPL[ASV_USE_IMPL]
        if not drop or level == "level_1":
            raise NotImplementedError

        self.df = generate_dataframe(ASV_USE_IMPL, "int", *shape, RAND_LOW, RAND_HIGH)
        if level:
            index = pd.MultiIndex.from_product(
                [self.df.index[: shape[0] // 2], ["bar", "foo"]],
                names=["level_1", "level_2"],
            )
            self.df.index = index
        trigger_import(self.df)

    def time_reset_index(self, shape, drop, level):
        execute(self.df.reset_index(drop=drop, level=level))


class TimeAstype:
    param_names = ["shape", "dtype", "astype_ncolumns"]
    params = [
        get_benchmark_shapes("omisci.TimeAstype", UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE]),
        ["float64"],
        ["one", "all"],
    ]

    def setup(self, shape, dtype, astype_ncolumns):
        self.df = generate_dataframe(ASV_USE_IMPL, "int", *shape, RAND_LOW, RAND_HIGH)
        trigger_import(self.df)
        self.astype_arg = self.create_astype_arg(dtype, astype_ncolumns)

    def time_astype(self, shape, dtype, astype_ncolumns):
        execute(self.df.astype(self.astype_arg))

    @staticmethod
    def create_astype_arg(dtype, astype_ncolumns):
        if astype_ncolumns == "all":
            astype_arg = dtype
        elif astype_ncolumns == "one":
            astype_arg = {"col1": dtype}
        else:
            assert False
        return astype_arg


class TimeDescribe:
    param_names = ["shape"]
    params = [
        get_benchmark_shapes(
            "omisci.TimeDescribe", UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE]
        ),
    ]

    def setup(self, shape):
        self.df = generate_dataframe(ASV_USE_IMPL, "int", *shape, RAND_LOW, RAND_HIGH)
        trigger_import(self.df)

    def time_describe(self, shape):
        execute(self.df.describe())


class TimeProperties:
    param_names = ["shape"]
    params = [
        get_benchmark_shapes(
            "omisci.TimeProperties", UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE]
        ),
    ]

    def setup(self, shape):
        self.df = generate_dataframe(ASV_USE_IMPL, "int", *shape, RAND_LOW, RAND_HIGH)
        trigger_import(self.df)

    def time_shape(self, shape):
        return self.df.shape

    def time_columns(self, shape):
        return self.df.columns

    def time_index(self, shape):
        return self.df.index


class BaseTimeGroupBy:
    def setup(self, shape, ngroups=5, groupby_ncols=1):
        ngroups = translator_groupby_ngroups(ngroups, shape)
        self.df, self.groupby_columns = generate_dataframe(
            ASV_USE_IMPL,
            "int",
            *shape,
            RAND_LOW,
            RAND_HIGH,
            groupby_ncols,
            count_groups=ngroups,
        )
        trigger_import(self.df)


class TimeGroupByDefaultAggregations(BaseTimeGroupBy):
    param_names = ["shape", "ngroups"]
    params = [
        get_benchmark_shapes(
            "omisci.TimeGroupByDefaultAggregations",
            UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
        ),
        GROUPBY_NGROUPS[ASV_DATASET_SIZE],
    ]

    def time_groupby_count(self, *args, **kwargs):
        execute(self.df.groupby(by=self.groupby_columns).count())

    def time_groupby_sum(self, *args, **kwargs):
        execute(self.df.groupby(by=self.groupby_columns).sum())


class TimeGroupByMultiColumn(BaseTimeGroupBy):
    param_names = ["shape", "ngroups", "groupby_ncols"]
    params = [
        UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
        GROUPBY_NGROUPS[ASV_DATASET_SIZE],
        [6],
    ]

    def time_groupby_sum(self, *args, **kwargs):
        execute(self.df.groupby(by=self.groupby_columns).sum())

    def time_groupby_agg_mean(self, *args, **kwargs):
        execute(self.df.groupby(by=self.groupby_columns).agg("mean"))

    def time_groupby_agg_nunique(self, *args, **kwargs):
        execute(self.df.groupby(by=self.groupby_columns).agg("nunique"))
