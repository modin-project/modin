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

"""Define data shapes."""

import os
import json

from .compatibility import ASV_USE_STORAGE_FORMAT, ASV_DATASET_SIZE

RAND_LOW = 0
RAND_HIGH = 1_000_000_000 if ASV_USE_STORAGE_FORMAT == "hdk" else 100

BINARY_OP_DATA_SIZE = {
    "big": [
        [[5000, 5000], [5000, 5000]],
        # the case extremely inefficient
        # [[20, 500_000], [10, 1_000_000]],
        [[500_000, 20], [1_000_000, 10]],
    ],
    "small": [[[250, 250], [250, 250]], [[10_000, 20], [25_000, 10]]],
}
UNARY_OP_DATA_SIZE = {
    "big": [
        [5000, 5000],
        # the case extremely inefficient
        # [10, 1_000_000],
        [1_000_000, 10],
    ],
    "small": [[250, 250], [10_000, 10]],
}
SERIES_DATA_SIZE = {
    "big": [[100_000, 1]],
    "small": [[10_000, 1]],
}
BINARY_OP_SERIES_DATA_SIZE = {
    "big": [
        [[500_000, 1], [1_000_000, 1]],
        [[500_000, 1], [500_000, 1]],
    ],
    "small": [[[5_000, 1], [10_000, 1]]],
}


HDK_BINARY_OP_DATA_SIZE = {
    "big": [[[500_000, 20], [1_000_000, 10]]],
    "small": [[[10_000, 20], [25_000, 10]]],
}
HDK_UNARY_OP_DATA_SIZE = {
    "big": [[1_000_000, 10]],
    "small": [[10_000, 10]],
}
HDK_SERIES_DATA_SIZE = {
    "big": [[10_000_000, 1]],
    "small": [[100_000, 1]],
}

DEFAULT_GROUPBY_NGROUPS = {
    "big": [100, "huge_amount_groups"],
    "small": [5],
}
GROUPBY_NGROUPS = DEFAULT_GROUPBY_NGROUPS[ASV_DATASET_SIZE]

_DEFAULT_CONFIG_T = [
    (
        UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
        [
            # Pandas storage format benchmarks
            "TimeGroupByMultiColumn",
            "TimeGroupByDefaultAggregations",
            "TimeGroupByDictionaryAggregation",
            "TimeSetItem",
            "TimeInsert",
            "TimeArithmetic",
            "TimeSortValues",
            "TimeDrop",
            "TimeHead",
            "TimeTail",
            "TimeExplode",
            "TimeFillna",
            "TimeFillnaDataFrame",
            "TimeValueCountsFrame",
            "TimeValueCountsSeries",
            "TimeIndexing",
            "TimeMultiIndexing",
            "TimeResetIndex",
            "TimeAstype",
            "TimeDescribe",
            "TimeProperties",
            "TimeReindex",
            "TimeReindexMethod",
            "TimeFillnaMethodDataframe",
            "TimeDropDuplicates",
            # IO benchmarks
            "TimeReadCsvSkiprows",
            "TimeReadCsvTrueFalseValues",
            "TimeReadCsvNamesDtype",
            "TimeReadParquet",
            # Scalability benchmarks
            "TimeFromPandas",
            "TimeToPandas",
        ],
    ),
    (
        BINARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
        [
            # Pandas storage format benchmarks
            "TimeJoin",
            "TimeMerge",
            "TimeConcat",
            "TimeAppend",
            "TimeBinaryOp",
            "TimeLevelAlign",
        ],
    ),
    (
        SERIES_DATA_SIZE[ASV_DATASET_SIZE],
        [
            # Pandas storage format benchmarks
            "TimeFillnaSeries",
            "TimeIndexingNumericSeries",
            "TimeFillnaMethodSeries",
        ],
    ),
    (
        BINARY_OP_SERIES_DATA_SIZE[ASV_DATASET_SIZE],
        [
            # Pandas storage format benchmarks
            "TimeBinaryOpSeries",
        ],
    ),
]

_DEFAULT_HDK_CONFIG_T = [
    (
        HDK_UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
        [
            "hdk.TimeJoin",
            "hdk.TimeBinaryOpDataFrame",
            "hdk.TimeArithmetic",
            "hdk.TimeSortValues",
            "hdk.TimeDrop",
            "hdk.TimeHead",
            "hdk.TimeFillna",
            "hdk.TimeIndexing",
            "hdk.TimeResetIndex",
            "hdk.TimeAstype",
            "hdk.TimeDescribe",
            "hdk.TimeProperties",
            "hdk.TimeGroupByDefaultAggregations",
            "hdk.TimeGroupByMultiColumn",
            "hdk.TimeValueCountsDataFrame",
            "hdk.TimeReadCsvNames",
        ],
    ),
    (
        HDK_BINARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
        ["hdk.TimeMerge", "hdk.TimeAppend"],
    ),
    (
        HDK_SERIES_DATA_SIZE[ASV_DATASET_SIZE],
        ["hdk.TimeBinaryOpSeries", "hdk.TimeValueCountsSeries"],
    ),
]
DEFAULT_CONFIG = {}
DEFAULT_CONFIG["MergeCategoricals"] = (
    [[10_000, 2]] if ASV_DATASET_SIZE == "big" else [[1_000, 2]]
)
DEFAULT_CONFIG["TimeJoinStringIndex"] = (
    [[100_000, 64]] if ASV_DATASET_SIZE == "big" else [[1_000, 4]]
)
for config in (_DEFAULT_CONFIG_T, _DEFAULT_HDK_CONFIG_T):
    for _shape, _names in config:
        DEFAULT_CONFIG.update({_name: _shape for _name in _names})

CONFIG_FROM_FILE = None


def get_benchmark_shapes(bench_id: str):
    """
    Get custom benchmark shapes from a json file stored in MODIN_ASV_DATASIZE_CONFIG.

    If `bench_id` benchmark is not found in the file, then the default value will
    be used.

    Parameters
    ----------
    bench_id : str
        Unique benchmark identifier that is used to get shapes.

    Returns
    -------
    list
        Benchmark shapes.
    """
    global CONFIG_FROM_FILE
    if not CONFIG_FROM_FILE:
        try:
            from modin.config import AsvDataSizeConfig

            filename = AsvDataSizeConfig.get()
        except ImportError:
            filename = os.environ.get("MODIN_ASV_DATASIZE_CONFIG", None)
        if filename:
            # should be json
            with open(filename) as _f:
                CONFIG_FROM_FILE = json.load(_f)

    if CONFIG_FROM_FILE and bench_id in CONFIG_FROM_FILE:
        # example: "hdk.TimeReadCsvNames": [[5555, 55], [3333, 33]]
        return CONFIG_FROM_FILE[bench_id]
    return DEFAULT_CONFIG[bench_id]
