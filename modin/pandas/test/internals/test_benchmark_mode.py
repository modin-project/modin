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

# test BenchmarkMode == True

from contextlib import nullcontext
import modin.pandas as pd
from modin.pandas.test.utils import test_data_values, df_equals
from modin.config import BenchmarkMode, StorageFormat
from modin.test.test_utils import warns_that_defaulting_to_pandas


def test_syncronous_mode():
    assert BenchmarkMode.get()
    # On Omnisci storage, transpose() defaults to Pandas.
    with (
        warns_that_defaulting_to_pandas()
        if StorageFormat.get() == "Omnisci"
        else nullcontext()
    ):
        pd.DataFrame(test_data_values[0]).mean()


def test_serialization():
    assert BenchmarkMode.get()

    sr1 = pd.Series(range(10))
    constructor, args = sr1.__reduce__()
    sr2 = constructor(*args)
    df_equals(sr1, sr2)

    df1 = pd.DataFrame(range(10))
    constructor, args = df1.__reduce__()
    df2 = constructor(*args)
    df_equals(df1, df2)
