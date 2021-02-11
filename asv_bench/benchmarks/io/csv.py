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

import modin.pandas as pd
import numpy as np

from ..utils import (
    generate_dataframe,
    RAND_LOW,
    RAND_HIGH,
    ASV_USE_IMPL,
    ASV_DATASET_SIZE,
    UNARY_OP_DATA_SIZE,
    IMPL,
    execute,
)

# ray init
if ASV_USE_IMPL == "modin":
    pd.DataFrame([])


class BaseReadCsv:
    # test data file can de created only once
    def setup_cache(self, test_filename="io_test_file"):
        test_filenames = {}
        for shape in UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE]:
            data_id = "_".join([str(size) for size in shape])
            test_filenames[data_id] = f"{test_filename}_{data_id}.csv"
            df = generate_dataframe("pandas", "str_int", *shape, RAND_LOW, RAND_HIGH)
            df.to_csv(test_filenames[data_id], index=False)

        return test_filenames


class TimeReadCsvSkiprows(BaseReadCsv):
    param_names = ["shape", "skiprows"]
    params = [
        UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
        [
            None,
            lambda x: x % 2,
            np.arange(1, UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE][0][0] // 10),
            np.arange(1, UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE][0][0], 2),
        ],
    ]

    def time_skiprows(self, test_filenames, shape, skiprows):
        data_id = "_".join([str(size) for size in shape])
        execute(IMPL[ASV_USE_IMPL].read_csv(test_filenames[data_id], skiprows=skiprows))
