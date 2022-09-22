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

"""IO Modin on HDK storage format benchmarks."""

from ..utils import (
    generate_dataframe,
    RAND_LOW,
    RAND_HIGH,
    ASV_USE_IMPL,
    IMPL,
    get_shape_id,
    trigger_import,
    get_benchmark_shapes,
)

from ..io.csv import TimeReadCsvTrueFalseValues  # noqa: F401


class TimeReadCsvNames:
    shapes = get_benchmark_shapes("hdk.TimeReadCsvNames")
    param_names = ["shape"]
    params = [shapes]

    def setup_cache(self, test_filename="io_test_file_csv_names"):
        # filenames with a metadata of saved dataframes
        cache = {}
        for shape in self.shapes:
            df = generate_dataframe("int", *shape, RAND_LOW, RAND_HIGH, impl="pandas")
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
            IMPL.DataFrame([])
        file_id = get_shape_id(shape)
        self.filename, self.names, self.dtype = cache[file_id]

    def time_read_csv_names(self, cache, shape):
        df = IMPL.read_csv(
            self.filename,
            names=self.names,
            header=0,
            dtype=self.dtype,
        )
        trigger_import(df)
