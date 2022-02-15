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
RAND_HIGH = 1_000_000_000 if ASV_USE_STORAGE_FORMAT == "omnisci" else 100

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


OMNISCI_BINARY_OP_DATA_SIZE = {
    "big": [[[500_000, 20], [1_000_000, 10]]],
    "small": [[[10_000, 20], [25_000, 10]]],
}
OMNISCI_UNARY_OP_DATA_SIZE = {
    "big": [[1_000_000, 10]],
    "small": [[10_000, 10]],
}
OMNISCI_SERIES_DATA_SIZE = {
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
            # IO benchmarks
            "TimeReadCsvSkiprows",
            "TimeReadCsvTrueFalseValues",
            "TimeReadCsvNamesDtype",
            "TimeReadParquetSkiprows",
            "TimeReadParquetTrueFalseValues",
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
        ],
    ),
    (
        SERIES_DATA_SIZE[ASV_DATASET_SIZE],
        [
            # Pandas storage format benchmarks
            "TimeFillnaSeries",
        ],
    ),
]

_DEFAULT_OMNISCI_CONFIG_T = [
    (
        OMNISCI_UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
        [
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
            "omnisci.TimeValueCountsDataFrame",
            "omnisci.TimeReadCsvNames",
        ],
    ),
    (
        OMNISCI_BINARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
        ["omnisci.TimeMerge", "omnisci.TimeAppend"],
    ),
    (
        OMNISCI_SERIES_DATA_SIZE[ASV_DATASET_SIZE],
        ["omnisci.TimeBinaryOpSeries", "omnisci.TimeValueCountsSeries"],
    ),
]
DEFAULT_CONFIG = {}
for config in (_DEFAULT_CONFIG_T, _DEFAULT_OMNISCI_CONFIG_T):
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
        # example: "omnisci.TimeReadCsvNames": [[5555, 55], [3333, 33]]
        return CONFIG_FROM_FILE[bench_id]
    return DEFAULT_CONFIG[bench_id]
