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

# NOTE: do not use pytest in parallel mode for this file

import modin.pandas as pd
from modin.config import BenchmarkMode
from modin.pandas.test.utils import test_data_values


def test_wait_func_result():
    if BenchmarkMode.get():
        BenchmarkMode.put(False)
    df = pd.DataFrame(test_data_values[0])

    @df._query_compiler._modin_frame.wait_func_result
    def compute_mean():
        return df.mean()

    compute_mean()


def test_syncronous_mode():
    if not BenchmarkMode.get():
        BenchmarkMode.put(True)
    df = pd.DataFrame(test_data_values[0])
    df.mean()
