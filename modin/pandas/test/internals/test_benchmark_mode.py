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
import numpy as np

import modin.pandas as pd
from modin.pandas.test.utils import (
    test_data_values,
    df_equals,
    catch_default_to_pandas_warnings_if_omnisci,
)
from modin.config import BenchmarkMode
from modin.core.dataframe.pandas.partitioning.partition_manager import (
    wait_computations_if_benchmark_mode,
)


class Waitable:
    """Emulates a partition class that can be awaited."""

    def __init__(self):
        self.is_wait_called = False

    def wait(self):
        self.is_wait_called = True


def test_wait_decorator_on_arbitrary_func():
    """
    Test that ``wait_computations_if_benchmark_mode`` decorator raises
    a proper error when applied to non-partitions-related function.
    """
    assert BenchmarkMode.get()

    @wait_computations_if_benchmark_mode
    def func(return_value):
        return return_value

    with pytest.raises(Exception, match="No partitions to wait.*"):
        func(None)

    with pytest.raises(Exception, match="No partitions to wait.*"):
        func("Not partitions")

    with pytest.raises(Exception, match="No partitions to wait.*"):
        func(("Not partitions", "Not partitions"))


def test_wait_decorator_on_partitions_returning_function():
    """
    Test that ``wait_computations_if_benchmark_mode`` decorator indeed calls ``.wait()``
    on partitions that were returned from the wrapped function.
    """
    assert BenchmarkMode.get()

    @wait_computations_if_benchmark_mode
    def partitions_returning_func(partitions_loc_in_result, num_returns):
        """Return `num_returns` objects and waitable objects are located in the `partitions_loc_in_result` position of the return value."""
        parts = np.array([[Waitable(), Waitable()], [Waitable(), Waitable()]])
        assert all(not part.is_wait_called for part in parts.flatten())
        result = ["Not partitions"] * (num_returns - 1)
        result.insert(partitions_loc_in_result, parts)
        return tuple(result) if len(result) > 1 else result[0]

    parts_location_in_res = 0
    parts = partitions_returning_func(parts_location_in_res, num_returns=1)
    assert all(part.is_wait_called for part in parts.flatten())

    result = partitions_returning_func(parts_location_in_res, num_returns=4)
    assert all(part.is_wait_called for part in result[parts_location_in_res].flatten())

    parts_location_in_res = 2
    result = partitions_returning_func(parts_location_in_res, num_returns=4)
    assert all(part.is_wait_called for part in result[parts_location_in_res].flatten())


def test_syncronous_mode():
    assert BenchmarkMode.get()
    # On Omnisci storage, transpose() defaults to Pandas.
    with catch_default_to_pandas_warnings_if_omnisci():
        pd.DataFrame(test_data_values[0]).mean()


@pytest.mark.parametrize("data_type", [pd.Series, pd.DataFrame])
def test_serialization(data_type):
    assert BenchmarkMode.get()

    dataset1 = data_type(range(10))
    # On Omnisci `__finalize__` raises a warning
    with catch_default_to_pandas_warnings_if_omnisci():
        constructor, args = dataset1.__reduce__()
    dataset2 = constructor(*args)
    df_equals(dataset1, dataset2)


@pytest.mark.parametrize("data_type", [pd.Series, pd.DataFrame])
def test_finalize(data_type):
    assert BenchmarkMode.get()

    dataset = data_type(range(10))
    # On Omnisci `abs` and `__finalize__` raise a warning
    with catch_default_to_pandas_warnings_if_omnisci():
        dataset = dataset.abs()
        dataset._query_compiler.finalize()
