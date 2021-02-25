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
    get_shape_id,
)

# ray init
if ASV_USE_IMPL == "modin":
    pd.DataFrame([])


class BaseReadCsv:
    # test data file can de created only once
    def setup_cache(self, test_filename="io_test_file"):
        test_filenames = {}
        for shape in UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE]:
            shape_id = get_shape_id(shape)
            test_filenames[shape_id] = f"{test_filename}_{shape_id}.csv"
            df = generate_dataframe("pandas", "str_int", *shape, RAND_LOW, RAND_HIGH)
            df.to_csv(test_filenames[shape_id], index=False)

        return test_filenames

    def setup(self, test_filenames, shape, *args, **kwargs):
        self.shape_id = get_shape_id(shape)


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
        execute(
            IMPL[ASV_USE_IMPL].read_csv(
                test_filenames[self.shape_id], skiprows=skiprows
            )
        )


class TimeReadCsvGeneral(BaseReadCsv):
    _dtypes_params = ["Int64", "Int64_Timestamp"]
    _timestamp_column = "col2"

    param_names = ["shape", "names", "dtype"]
    params = [
        UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
        ["array-like"],
        _dtypes_params,
    ]

    def _get_file_id(self, shape, dtype):
        return get_shape_id(shape) + dtype

    def _add_timestamp_column(self, df):
        df = df.copy()
        df.loc[:, self._timestamp_column] = IMPL["pandas"].date_range(
            "2000",
            periods=df.shape[0],
            freq="ms",
        )
        return df

    def setup_cache(self, test_filename="io_test_file_csv_general"):
        test_filenames = {}
        for shape in UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE]:
            for dtype in self._dtypes_params:
                df = generate_dataframe("pandas", "int", *shape, RAND_LOW, RAND_HIGH)
                if dtype == "Int64_Timestamp":
                    df = self._add_timestamp_column(df)

                file_id = self._get_file_id(shape, dtype)
                test_filenames[file_id] = (
                    f"{test_filename}_{file_id}.csv",
                    df.columns.to_list(),
                    df.dtypes.to_dict(),
                )
                df.to_csv(test_filenames[file_id][0], index=False)
        return test_filenames

    def setup(self, test_filenames, shape, names, dtype):
        file_id = self._get_file_id(shape, dtype)
        self.filename, self.names, self.dtype = test_filenames[file_id]

        self.parse_dates = None
        if dtype == "Int64_Timestamp":
            self.dtype = self.dtype.copy()
            del self.dtype[self._timestamp_column]
            self.parse_dates = [self._timestamp_column]

    def time_read_csv_general(self, test_filenames, shape, names, dtype):
        execute(
            IMPL[ASV_USE_IMPL].read_csv(
                self.filename,
                names=self.names,
                header=0,
                dtype=self.dtype,
                parse_dates=self.parse_dates,
            )
        )
