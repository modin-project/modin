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
from modin.config import NPartitions, context

NPartitions.put(4)


@pytest.mark.parametrize("axis", [0, 1, None])
@pytest.mark.parametrize("dtype", ["DataFrame", "Series"])
def test_repartition(axis, dtype):
    if axis in (1, None) and dtype == "Series":
        # no sense for Series
        return

    df = pd.DataFrame({"col1": [1, 2], "col2": [5, 6]})
    df2 = pd.DataFrame({"col3": [9, 4]})

    df = pd.concat([df, df2], axis=1)
    df = pd.concat([df, df], axis=0)

    obj = df if dtype == "DataFrame" else df["col1"]

    source_shapes = {
        "DataFrame": (2, 2),
        "Series": (2, 1),
    }
    # check that the test makes sense
    assert obj._query_compiler._modin_frame._partitions.shape == source_shapes[dtype]

    kwargs = {"axis": axis} if dtype == "DataFrame" else {}
    obj = obj._repartition(**kwargs)

    if dtype == "DataFrame":
        results = {
            None: (1, 1),
            0: (1, 2),
            1: (2, 1),
        }
    else:
        results = {
            None: (1, 1),
            0: (1, 1),
            1: (2, 1),
        }

    assert obj._query_compiler._modin_frame._partitions.shape == results[axis]


def test_repartition_7170():
    with context(MinColumnPartitionSize=102, NPartitions=5):
        df = pd.DataFrame(np.random.rand(10000, 100))
        _ = df._repartition(axis=1).to_numpy()
