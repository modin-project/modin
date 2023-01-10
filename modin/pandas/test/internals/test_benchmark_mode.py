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

import unittest.mock as mock

import modin.pandas as pd
from modin.pandas.test.utils import test_data_values
from modin.config import BenchmarkMode, Engine


engine = Engine.get()

# We have to explicitly mock subclass implementations of wait_partitions.
if engine == "Ray":
    wait_method = (
        "modin.core.execution.ray.implementations."
        + "pandas_on_ray.partitioning."
        + "PandasOnRayDataframePartitionManager.wait_partitions"
    )
elif engine == "Dask":
    wait_method = (
        "modin.core.execution.dask.implementations."
        + "pandas_on_dask.partitioning."
        + "PandasOnDaskDataframePartitionManager.wait_partitions"
    )
elif engine == "Unidist":
    wait_method = (
        "modin.core.execution.unidist.implementations."
        + "pandas_on_unidist.partitioning."
        + "PandasOnUnidistDataframePartitionManager.wait_partitions"
    )
else:
    wait_method = (
        "modin.core.dataframe.pandas.partitioning."
        + "partition_manager.PandasDataframePartitionManager.wait_partitions"
    )


def test_from_environment_variable():
    assert BenchmarkMode.get()
    with mock.patch(wait_method) as wait:
        pd.DataFrame(test_data_values[0]).mean()

    wait.assert_called()


def test_turn_off():
    df = pd.DataFrame([0])
    BenchmarkMode.put(False)
    with mock.patch(wait_method) as wait:
        df.dropna()
    wait.assert_not_called()


def test_turn_on():
    BenchmarkMode.put(False)
    df = pd.DataFrame([0])
    BenchmarkMode.put(True)
    with mock.patch(wait_method) as wait:
        df.dropna()
    wait.assert_called()
