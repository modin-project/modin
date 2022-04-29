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

import pytest

from contextlib import nullcontext
import modin.pandas as pd
from modin.pandas.test.utils import test_data_values, df_equals
from modin.config import BenchmarkMode, StorageFormat
from modin.test.test_utils import warns_that_defaulting_to_pandas


default_to_pandas_context = (
    warns_that_defaulting_to_pandas()
    if StorageFormat.get() == "Omnisci"
    else nullcontext()
)


def test_syncronous_mode():
    assert BenchmarkMode.get()
    # On Omnisci storage, transpose() defaults to Pandas.
    with default_to_pandas_context:
        pd.DataFrame(test_data_values[0]).mean()


@pytest.mark.parametrize("data_type", [pd.Series, pd.DataFrame])
def test_serialization(data_type):
    assert BenchmarkMode.get()

    dataset1 = data_type(range(10))
    # On Omnisci `__finalize__` raises a warning
    with default_to_pandas_context:
        constructor, args = dataset1.__reduce__()
    dataset2 = constructor(*args)
    df_equals(dataset1, dataset2)


@pytest.mark.parametrize("data_type", [pd.Series, pd.DataFrame])
def test_finalize(data_type):
    assert BenchmarkMode.get()

    dataset = data_type(range(10))
    # On Omnisci `__finalize__` raises a warning
    with default_to_pandas_context:
        dataset._query_compiler.finalize()
