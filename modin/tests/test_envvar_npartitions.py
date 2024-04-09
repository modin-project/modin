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
import pytest

import modin.pandas as pd
from modin.config import NPartitions


@pytest.mark.parametrize("num_partitions", [2, 4, 6, 8, 10])
def test_set_npartitions(num_partitions):
    NPartitions.put(num_partitions)
    data = np.random.randint(0, 100, size=(2**16, 2**8))
    df = pd.DataFrame(data)
    part_shape = df._query_compiler._modin_frame._partitions.shape
    assert part_shape[0] == num_partitions and part_shape[1] == min(num_partitions, 8)


@pytest.mark.parametrize("left_num_partitions", [2, 4, 6, 8, 10])
@pytest.mark.parametrize("right_num_partitions", [2, 4, 6, 8, 10])
def test_runtime_change_npartitions(left_num_partitions, right_num_partitions):
    NPartitions.put(left_num_partitions)
    data = np.random.randint(0, 100, size=(2**16, 2**8))
    left_df = pd.DataFrame(data)
    part_shape = left_df._query_compiler._modin_frame._partitions.shape
    assert part_shape[0] == left_num_partitions and part_shape[1] == min(
        left_num_partitions, 8
    )

    NPartitions.put(right_num_partitions)
    right_df = pd.DataFrame(data)
    part_shape = right_df._query_compiler._modin_frame._partitions.shape
    assert part_shape[0] == right_num_partitions and part_shape[1] == min(
        right_num_partitions, 8
    )
