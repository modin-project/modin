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


from ..utils import (
    generate_dataframe,
    RAND_LOW,
    RAND_HIGH,
    ASV_USE_IMPL,
    ASV_DATASET_SIZE,
    execute,
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
        trigger_import((self.df1, self.df2))

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
        trigger_import((self.df,))
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
        self.df = generate_dataframe(ASV_USE_IMPL, "int", *shape, RAND_LOW, RAND_HIGH)
        trigger_import((self.df,))
        self.head_count = (
            int(head_count * len(self.df.index))
            if isinstance(head_count, float)
            else head_count
        )

    def time_head(self, shape, head_count):
        execute(self.df.head(self.head_count))


class TimeProperties:
    param_names = ["shape"]
    params = [
        UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
    ]

    def setup(self, shape):
        self.df = generate_dataframe(ASV_USE_IMPL, "int", *shape, RAND_LOW, RAND_HIGH)
        trigger_import((self.df,))

    def time_shape(self, shape):
        return self.df.shape

    def time_columns(self, shape):
        return self.df.columns

    def time_index(self, shape):
        return self.df.index
