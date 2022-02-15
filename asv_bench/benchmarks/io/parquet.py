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
    ASV_USE_IMPL,
    ASV_USE_STORAGE_FORMAT,
    IMPL,
    execute,
    get_shape_id,
    prepare_io_data_parquet,
    get_benchmark_shapes,
)


class BaseReadParquet:
    # test data file should be created only once
    def setup_cache(self, test_filename="io_test_file"):
        test_filenames = prepare_io_data_parquet(
            test_filename, self.data_type, get_benchmark_shapes(self.__class__.__name__)
        )
        return test_filenames

    def setup(self, test_filenames, shape, *args, **kwargs):
        # ray init
        if ASV_USE_IMPL == "modin":
            pd.DataFrame([])
        self.shape_id = get_shape_id(shape)


class TimeReadParquetSkiprows(BaseReadParquet):
    shapes = get_benchmark_shapes("TimeReadParquetSkiprows")
    skiprows_mapping = {
        "lambda_even_rows": lambda x: x % 2,
        "range_uniform": np.arange(1, shapes[0][0] // 10),
        "range_step2": np.arange(1, shapes[0][0], 2),
    }
    data_type = "str_int"

    param_names = ["shape", "skiprows"]
    params = [
        shapes,
        [None, "lambda_even_rows", "range_uniform", "range_step2"],
    ]

    def setup(self, test_filenames, shape, skiprows):
        super().setup(test_filenames, shape, skiprows)
        self.skiprows = self.skiprows_mapping[skiprows] if skiprows else None

    def time_skiprows(self, test_filenames, shape, skiprows):
        execute(
            IMPL[ASV_USE_IMPL].read_parquet(
                test_filenames[self.shape_id], skiprows=self.skiprows
            )
        )


class TimeReadParquetTrueFalseValues(BaseReadParquet):
    data_type = "true_false_int"

    param_names = ["shape"]
    params = [get_benchmark_shapes("TimeReadParquetTrueFalseValues")]

    def time_true_false_values(self, test_filenames, shape):
        execute(
            IMPL[ASV_USE_IMPL].read_csv(
                test_filenames[self.shape_id],
                true_values=["Yes", "true"],
                false_values=["No", "false"],
            ),
            trigger_omnisci_import=ASV_USE_STORAGE_FORMAT == "omnisci",
        )
