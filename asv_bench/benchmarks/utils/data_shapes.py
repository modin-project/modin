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

from .compatibility import ASV_USE_BACKEND, ASV_DATASET_SIZE

RAND_LOW = 0
RAND_HIGH = 1_000_000_000 if ASV_USE_BACKEND == "omnisci" else 100

BINARY_OP_DATA_SIZE = {
    "big": [
        [[5000, 5000], [5000, 5000]],
        # the case extremely inefficient
        # [[20, 500_000], [10, 1_000_000]],
        [[500_000, 20], [1_000_000, 10]],
    ],
    "small": [
        [[250, 250], [250, 250]],
        [[20, 10_000], [10, 25_000]],
        [[10_000, 20], [25_000, 10]],
    ],
}
UNARY_OP_DATA_SIZE = {
    "big": [
        [5000, 5000],
        # the case extremely inefficient
        # [10, 1_000_000],
        [1_000_000, 10],
    ],
    "small": [
        [250, 250],
        [10, 10_000],
        [10_000, 10],
    ],
}
SERIES_DATA_SIZE = {
    "big": [
        (100_000, 1),
    ],
    "small": [
        (10_000, 1),
    ],
}


OMNISCI_BINARY_OP_DATA_SIZE = {
    "big": [
        [[500_000, 20], [1_000_000, 10]],
    ],
    "small": [
        [[10_000, 20], [25_000, 10]],
    ],
}
OMNISCI_UNARY_OP_DATA_SIZE = {
    "big": [
        [1_000_000, 10],
    ],
    "small": [
        [10_000, 10],
    ],
}
OMNISCI_SERIES_DATA_SIZE = {
    "big": [
        [10_000_000, 1],
    ],
    "small": [
        [100_000, 1],
    ],
}

BINARY_SHAPES = (
    OMNISCI_BINARY_OP_DATA_SIZE[ASV_DATASET_SIZE]
    if ASV_USE_BACKEND == "omnisci"
    else BINARY_OP_DATA_SIZE[ASV_DATASET_SIZE]
)
UNARY_SHAPES = (
    OMNISCI_UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE]
    if ASV_USE_BACKEND == "omnisci"
    else UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE]
)
SERIES_SHAPES = (
    OMNISCI_SERIES_DATA_SIZE[ASV_DATASET_SIZE]
    if ASV_USE_BACKEND == "omnisci"
    else SERIES_DATA_SIZE[ASV_DATASET_SIZE]
)

DEFAULT_GROUPBY_NGROUPS = {
    "big": [100, "huge_amount_groups"],
    "small": [5],
}
GROUPBY_NGROUPS = DEFAULT_GROUPBY_NGROUPS[ASV_DATASET_SIZE]

_DEFAULT_CONFIG_T = [
    (
        UNARY_SHAPES,
        [
            # Pandas backend benchmarks
            "TimeGroupByMultiColumn",
            "TimeGroupByDefaultAggregations",
            "TimeGroupByDictionaryAggregation",
            "TimeSetItem",
            "TimeInsert",
            "TimeArithmetic",
            "TimeSortValues",
            "TimeDrop",
            "TimeHead",
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
            # IO benchmarks
            "TimeReadCsvSkiprows",
            "TimeReadCsvTrueFalseValues",
            "TimeReadCsvNamesDtype",
            # Scalability benchmarks
            "TimeFromPandas",
            "TimeToPandas",
            # OmniSci backend benchmarks
            "omnisci.TimeJoin",
            "omnisci.TimeBinaryOpDataFrame",
            "omnisci.TimeArithmetic",
            "omnisci.TimeSortValues",
            "omnisci.TimeDrop",
            "omnisci.TimeHead",
            "omnisci.TimeFillna",
            "omnisci.TimeIndexing",
            "omnisci.TimeResetIndex",
            "omnisci.TimeAstype",
            "omnisci.TimeDescribe",
            "omnisci.TimeProperties",
            "omnisci.TimeGroupByDefaultAggregations",
            "omnisci.TimeGroupByMultiColumn",
            # OmniSci backend IO benchmarks
            "omnisci.TimeReadCsvNames",
        ],
    ),
    (
        BINARY_SHAPES,
        [
            # Pandas backend benchmarks
            "TimeJoin",
            "TimeMerge",
            "TimeConcat",
            "TimeAppend",
            "TimeBinaryOp",
            # OmniSci backend benchmarks
            "omnisci.TimeMerge",
            "omnisci.TimeAppend",
        ],
    ),
    (
        SERIES_SHAPES,
        [
            # Pandas backend benchmarks
            "TimeFillnaSeries",
            # OmniSci backend benchmarks
            "omnisci.TimeBinaryOpSeries",
            "omnisci.TimeValueCountsSeries",
        ],
    ),
]
DEFAULT_CONFIG = {}
for _shape, _names in _DEFAULT_CONFIG_T:
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
        # example: "omnisci.TimeReadCsvNames": [[5555, 55], [3333, 33]]
        return CONFIG_FROM_FILE[bench_id]
    return DEFAULT_CONFIG[bench_id]
