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
import pandas
import pytest

import modin.pandas as pd
from modin.distributed.dataframe.pandas import unwrap_partitions, from_partitions
from modin.config import Engine, NPartitions
from modin.pandas.test.utils import df_equals


if Engine.get() == "Ray":
    import ray
if Engine.get() == "Dask":
    from distributed.client import get_client

NPartitions.put(4)


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_unwrap_partitions(axis):
    data = np.random.randint(0, 100, size=(2 ** 16, 2 ** 8))
    df = pd.DataFrame(data)

    if axis is None:
        expected_partitions = df._query_compiler._modin_frame._partitions
        actual_partitions = np.array(unwrap_partitions(df, axis=axis))
        assert (
            expected_partitions.shape[0] == actual_partitions.shape[0]
            and expected_partitions.shape[1] == expected_partitions.shape[1]
        )
        for row_idx in range(expected_partitions.shape[0]):
            for col_idx in range(expected_partitions.shape[1]):
                if Engine.get() == "Ray":
                    assert (
                        expected_partitions[row_idx][col_idx].oid
                        == actual_partitions[row_idx][col_idx]
                    )
                if Engine.get() == "Dask":
                    assert (
                        expected_partitions[row_idx][col_idx].future
                        == actual_partitions[row_idx][col_idx]
                    )
    else:
        expected_axis_partitions = (
            df._query_compiler._modin_frame._frame_mgr_cls.axis_partition(
                df._query_compiler._modin_frame._partitions, axis ^ 1
            )
        )
        expected_axis_partitions = [
            axis_partition.force_materialization().unwrap(squeeze=True)
            for axis_partition in expected_axis_partitions
        ]
        actual_axis_partitions = unwrap_partitions(df, axis=axis)
        assert len(expected_axis_partitions) == len(actual_axis_partitions)
        for item_idx in range(len(expected_axis_partitions)):
            if Engine.get() == "Ray":
                df_equals(
                    ray.get(expected_axis_partitions[item_idx]),
                    ray.get(actual_axis_partitions[item_idx]),
                )
            if Engine.get() == "Dask":
                df_equals(
                    expected_axis_partitions[item_idx].result(),
                    actual_axis_partitions[item_idx].result(),
                )


@pytest.mark.parametrize("axis", [None, 0, 1])
def test_from_partitions(axis):
    data = np.random.randint(0, 100, size=(2 ** 16, 2 ** 8))
    df1, df2 = pandas.DataFrame(data), pandas.DataFrame(data)
    expected_df = pandas.concat([df1, df2], axis=1 if axis is None else axis)
    if Engine.get() == "Ray":
        if axis is None:
            futures = [[ray.put(df1), ray.put(df2)]]
        else:
            futures = [ray.put(df1), ray.put(df2)]
    if Engine.get() == "Dask":
        client = get_client()
        if axis is None:
            futures = [client.scatter([df1, df2], hash=False)]
        else:
            futures = client.scatter([df1, df2], hash=False)
    actual_df = from_partitions(futures, axis)
    df_equals(expected_df, actual_df)
