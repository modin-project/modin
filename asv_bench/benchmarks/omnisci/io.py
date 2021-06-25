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

"""IO Modin on OmniSci backend benchmarks."""

import modin.pandas as pd

from ..utils import (
    generate_dataframe,
    RAND_LOW,
    RAND_HIGH,
    ASV_USE_IMPL,
    ASV_DATASET_SIZE,
    IMPL,
    get_shape_id,
)

from .utils import UNARY_OP_DATA_SIZE, trigger_import


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
