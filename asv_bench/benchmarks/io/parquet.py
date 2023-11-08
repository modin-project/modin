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

from ..utils import (
    ASV_USE_IMPL,
    IMPL,
    execute,
    get_benchmark_shapes,
    get_shape_id,
    prepare_io_data_parquet,
)


class TimeReadParquet:
    shapes = get_benchmark_shapes("TimeReadParquet")
    data_type = "str_int"

    param_names = ["shape"]
    params = [
        shapes,
    ]

    # test data file should be created only once
    def setup_cache(self, test_filename="io_test_file"):
        test_filenames = prepare_io_data_parquet(
            test_filename, self.data_type, get_benchmark_shapes(self.__class__.__name__)
        )
        return test_filenames

    def setup(self, test_filenames, shape):
        # ray init
        if ASV_USE_IMPL == "modin":
            IMPL.DataFrame([])
        self.shape_id = get_shape_id(shape)

    def time_read_parquet(self, test_filenames, shape):
        execute(
            IMPL.read_parquet(
                test_filenames[self.shape_id],
            )
        )


from ..utils import setup  # noqa: E402, F401
