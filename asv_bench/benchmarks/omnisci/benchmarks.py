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

from time import time

import modin.pandas as pd
import numpy as np

from ..utils import (
    generate_dataframe,
    RAND_LOW,
    RAND_HIGH,
    ASV_USE_IMPL,
    ASV_DATASET_SIZE,
    GROUPBY_NGROUPS,
    IMPL,
    execute,
    translator_groupby_ngroups,
    get_shape_id,
    random_columns,
    random_booleans,
)

BINARY_OP_DATA_SIZE = {
    "big": [
        ((500_000, 20), (1_000_000, 10)),
    ],
    "small": [
        ((10_000, 20), (25_000, 10)),
    ],
}

UNARY_OP_DATA_SIZE = {
    "big": [
        (1_000_000, 10),
    ],
    "small": [
        (10_000, 10),
    ],
}


def trigger_import(*dfs):
    print("triger_import start")
    from modin.experimental.engines.omnisci_on_ray.frame.omnisci_worker import (
        OmnisciServer,
    )

    for df in dfs:
        df.shape  # to trigger real execution
        df._query_compiler._modin_frame._partitions[0][
            0
        ].frame_id = OmnisciServer().put_arrow_to_omnisci(
            df._query_compiler._modin_frame._partitions[0][0].get()
        )  # to trigger real execution
    print("triger_import end")


class TimeMerge:
    param_names = ["shapes", "how"]
    params = [
        BINARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
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


class TimeIndexing:
    param_names = ["shape", "indexer_type"]
    params = [
        UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
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


class TimeHead:
    param_names = ["shape", "head_count"]
    params = [
        UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
        [5, 0.8],
    ]

    def setup(self, shape, head_count):
        start = time()
        self.df = generate_dataframe(ASV_USE_IMPL, "int", *shape, RAND_LOW, RAND_HIGH)
        trigger_import(self.df)
        self.head_count = (
            int(head_count * len(self.df.index))
            if isinstance(head_count, float)
            else head_count
        )
        print(f"setup time: {time()-start}")

    def time_head(self, shape, head_count):
        start = time()
        execute(self.df.head(self.head_count))
        print(f"time_head time: {time()-start}")


class TimeProperties:
    param_names = ["shape"]
    params = [
        UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
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


def create_fillna_value(value_type: str, shape: tuple):
    if value_type == "scalar":
        value = 18.19
    elif value_type == "dict":
        value = {k: k * 1.23 for k in range(shape[0])}
    else:
        assert False
    return value


class TimeFillna:
    param_names = ["value_type", "shape", "limit"]
    params = [
        ["scalar", "dict"],
        UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
        [None],
    ]

    def setup(self, value_type, shape, limit):
        pd = IMPL[ASV_USE_IMPL]
        columns = [f"col{x}" for x in range(shape[1])]
        self.df = pd.DataFrame(np.nan, index=pd.RangeIndex(shape[0]), columns=columns)
        trigger_import(self.df)

        value = create_fillna_value(value_type, shape)
        limit = int(limit * shape[0]) if limit else None
        self.kw = {"value": value, "limit": limit}

    def time_fillna(self, value_type, shape, limit):
        execute(self.df.fillna(**self.kw))


class TimeAstype:
    param_names = ["shape", "dtype", "astype_ncolumns"]
    params = [
        UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
        ["float64"],
        ["one", "all"],
    ]

    def setup(self, shape, dtype, astype_ncolumns):
        self.df = generate_dataframe(ASV_USE_IMPL, "int", *shape, RAND_LOW, RAND_HIGH)
        trigger_import(self.df)

        if astype_ncolumns == "all":
            self.astype_arg = dtype
        elif astype_ncolumns == "one":
            self.astype_arg = {"col1": dtype}
        else:
            assert False

    def time_astype(self, shape, dtype, astype_ncolumns):
        execute(self.df.astype(self.astype_arg))


class TimeArithmetic:
    param_names = ["shape", "axis"]
    params = [
        UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
        [0],
    ]

    def setup(self, shape, axis):
        self.df = generate_dataframe(ASV_USE_IMPL, "int", *shape, RAND_LOW, RAND_HIGH)
        trigger_import(self.df)

    def time_mean(self, shape, axis):
        execute(self.df.agg("sum"))


class TimeReadCsvNames:
    param_names = ["shape"]
    params = [
        UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
    ]

    def setup_cache(self, test_filename="io_test_file_csv_names"):
        # filenames with a metadata of saved dataframes
        cache = {}
        for shape in UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE]:
            df = generate_dataframe("pandas", "int", *shape, RAND_LOW, RAND_HIGH)
            file_id = get_shape_id(shape)
            cache[file_id] = (
                f"{test_filename}_{file_id}.csv",
                df.columns.to_list(),
                df.dtypes.to_dict(),
            )
            df.to_csv(cache[file_id][0], index=False)
        return cache

    def setup(self, cache, shape):
        # ray init
        if ASV_USE_IMPL == "modin":
            pd.DataFrame([])
        file_id = get_shape_id(shape)
        self.filename, self.names, self.dtype = cache[file_id]

    def time_read_csv_names(self, cache, shape):
        df = IMPL[ASV_USE_IMPL].read_csv(
            self.filename,
            names=self.names,
            header=0,
            dtype=self.dtype,
        )
        trigger_import(df)


class TimeValueCountsSeries:
    param_names = ["shape", "ngroups"]
    params = [
        UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
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


class TimeResetIndex:
    param_names = ["shape", "drop", "level"]
    params = [UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE], [False, True], [None, "level_1"]]

    def setup(self, shape, drop, level):
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


class TimeSortValues:
    param_names = ["shape", "columns_number", "ascending_list"]
    params = [
        UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
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


class TimeDescribe:
    param_names = ["shape"]
    params = [
        UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
    ]

    def setup(self, shape):
        self.df = generate_dataframe(ASV_USE_IMPL, "int", *shape, RAND_LOW, RAND_HIGH)
        trigger_import(self.df)

    def time_describe(self, shape):
        execute(self.df.describe())


class TimeBinaryOp:
    param_names = ["shape", "binary_op"]
    params = [
        UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
        ["mul"],
    ]

    def setup(self, shape, binary_op):
        self.df1 = generate_dataframe(ASV_USE_IMPL, "int", *shape, RAND_LOW, RAND_HIGH)
        trigger_import(self.df1)

    def time_mul_scalar(self, shape, binary_op):
        execute(self.df1 * 2)

    def time_mul_series(self, shape, binary_op):
        execute(self.df1["col1"] * self.df1["col2"])

    def time_mul_dataframes(self, shape, binary_op):
        execute(self.df1 * self.df1)


class TimeDrop:
    param_names = ["shape", "drop_ncols"]
    params = [
        UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
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


class TimeJoin:
    param_names = ["shape", "how"]
    params = [
        UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
        ["left", "inner"],
    ]

    def setup(self, shape, how):
        self.df1 = generate_dataframe(ASV_USE_IMPL, "int", *shape, RAND_LOW, RAND_HIGH)
        self.df2 = generate_dataframe(ASV_USE_IMPL, "int", *shape, RAND_LOW, RAND_HIGH)
        trigger_import(self.df1, self.df2)

    def time_join(self, shape, how):
        # join dataframes on index to get the predictable shape
        execute(self.df1.join(self.df2, how=how, lsuffix="left_"))


class TimeAppend:
    param_names = ["shapes"]
    params = [
        BINARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
    ]

    def setup(self, shapes):
        self.df1 = generate_dataframe(
            ASV_USE_IMPL, "int", *shapes[0], RAND_LOW, RAND_HIGH
        )
        self.df2 = generate_dataframe(
            ASV_USE_IMPL, "int", *shapes[1], RAND_LOW, RAND_HIGH
        )

    def time_append(self, shapes):
        execute(self.df1.append(self.df2))
