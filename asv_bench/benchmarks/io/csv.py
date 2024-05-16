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

import numpy as np

from ..utils import (
    ASV_USE_IMPL,
    IMPL,
    RAND_HIGH,
    RAND_LOW,
    execute,
    generate_dataframe,
    get_benchmark_shapes,
    get_shape_id,
    prepare_io_data,
)


class BaseReadCsv:
    # test data file should be created only once
    def setup_cache(self, test_filename="io_test_file"):
        test_filenames = prepare_io_data(
            test_filename, self.data_type, get_benchmark_shapes(self.__class__.__name__)
        )
        return test_filenames

    def setup(self, test_filenames, shape, *args, **kwargs):
        # ray init
        if ASV_USE_IMPL == "modin":
            IMPL.DataFrame([])
        self.shape_id = get_shape_id(shape)


class TimeReadCsvSkiprows(BaseReadCsv):
    shapes = get_benchmark_shapes("TimeReadCsvSkiprows")
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
        execute(IMPL.read_csv(test_filenames[self.shape_id], skiprows=self.skiprows))


class TimeReadCsvTrueFalseValues(BaseReadCsv):
    data_type = "true_false_int"

    param_names = ["shape"]
    params = [get_benchmark_shapes("TimeReadCsvTrueFalseValues")]

    def time_true_false_values(self, test_filenames, shape):
        execute(
            IMPL.read_csv(
                test_filenames[self.shape_id],
                true_values=["Yes", "true"],
                false_values=["No", "false"],
            ),
        )


class TimeReadCsvNamesDtype:
    shapes = get_benchmark_shapes("TimeReadCsvNamesDtype")
    _dtypes_params = ["Int64", "Int64_Timestamp"]
    _timestamp_columns = ["col1", "col2"]

    param_names = ["shape", "names", "dtype"]
    params = [
        shapes,
        ["array-like"],
        _dtypes_params,
    ]

    def _get_file_id(self, shape, dtype):
        return get_shape_id(shape) + dtype

    def _add_timestamp_columns(self, df):
        df = df.copy()
        date_column = IMPL.date_range("2000", periods=df.shape[0], freq="ms")
        for col in self._timestamp_columns:
            df[col] = date_column
        return df

    def setup_cache(self, test_filename="io_test_file_csv_names_dtype"):
        # filenames with a metadata of saved dataframes
        cache = {}
        for shape in self.shapes:
            for dtype in self._dtypes_params:
                df = generate_dataframe(
                    "int", *shape, RAND_LOW, RAND_HIGH, impl="pandas"
                )
                if dtype == "Int64_Timestamp":
                    df = self._add_timestamp_columns(df)

                file_id = self._get_file_id(shape, dtype)
                cache[file_id] = (
                    f"{test_filename}_{file_id}.csv",
                    df.columns.to_list(),
                    df.dtypes.to_dict(),
                )
                df.to_csv(cache[file_id][0], index=False)
        return cache

    def setup(self, cache, shape, names, dtype):
        # ray init
        if ASV_USE_IMPL == "modin":
            IMPL.DataFrame([])
        file_id = self._get_file_id(shape, dtype)
        self.filename, self.names, self.dtype = cache[file_id]

        self.parse_dates = None
        if dtype == "Int64_Timestamp":
            # cached version of dtype should not change
            self.dtype = self.dtype.copy()
            for col in self._timestamp_columns:
                del self.dtype[col]
            self.parse_dates = self._timestamp_columns

    def time_read_csv_names_dtype(self, cache, shape, names, dtype):
        execute(
            IMPL.read_csv(
                self.filename,
                names=self.names,
                header=0,
                dtype=self.dtype,
                parse_dates=self.parse_dates,
            )
        )


from ..utils import setup  # noqa: E402, F401
