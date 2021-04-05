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
from modin.pandas.utils import from_pandas

try:
    from modin.utils import to_pandas
except ImportError:
    from modin.pandas.utils import to_pandas
import pandas

from ..utils import (
    gen_data,
    generate_dataframe,
    RAND_LOW,
    RAND_HIGH,
    ASV_DATASET_SIZE,
    UNARY_OP_DATA_SIZE,
    execute,
)


class TimeFromPandas:
    param_names = ["shape", "cpus"]
    params = [
        UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
        [4, 16, 32],
    ]

    def setup(self, shape, cpus):
        self.data = pandas.DataFrame(gen_data("int", *shape, RAND_LOW, RAND_HIGH))
        from modin.config import NPartitions

        NPartitions.get = lambda: cpus
        # trigger ray init
        pd.DataFrame([])

    def time_from_pandas(self, shape, cpus):
        execute(from_pandas(self.data))


class TimeToPandas:
    param_names = ["shape", "cpus"]
    params = [
        UNARY_OP_DATA_SIZE[ASV_DATASET_SIZE],
        [4, 16, 32],
    ]

    def setup(self, shape, cpus):
        from modin.config import NPartitions

        NPartitions.get = lambda: cpus
        self.data = generate_dataframe("modin", "int", *shape, RAND_LOW, RAND_HIGH)

    def time_to_pandas(self, shape, cpus):
        # to_pandas is already synchronous
        to_pandas(self.data)
