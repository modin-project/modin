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

import re

import numpy as np
import pytest

import modin.pandas as pd
from modin.config import context
from modin.core.storage_formats.pandas.native_query_compiler import (
    _NO_REPARTITION_ON_NATIVE_EXECUTION_EXCEPTION_MESSAGE,
)
from modin.tests.test_utils import current_execution_is_native
from modin.utils import get_current_execution


@pytest.fixture(autouse=True)
def set_npartitions():
    with context(NPartitions=4):
        yield


@pytest.mark.skipif(
    current_execution_is_native(), reason="Native execution does not have partitions."
)
@pytest.mark.skipif(
    get_current_execution() == "BaseOnPython",
    reason="BaseOnPython chooses partition numbers differently",
)
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


@pytest.mark.skipif(
    current_execution_is_native(), reason="Native execution does not have partitions."
)
def test_repartition_7170():
    with context(MinColumnPartitionSize=102, NPartitions=5):
        df = pd.DataFrame(np.random.rand(10000, 100))
        _ = df._repartition(axis=1).to_numpy()


@pytest.mark.skipif(
    not current_execution_is_native(), reason="This is a native execution test."
)
def test_repartition_not_valid_on_native_execution():
    df = pd.DataFrame()
    with pytest.raises(
        Exception,
        match=re.escape(_NO_REPARTITION_ON_NATIVE_EXECUTION_EXCEPTION_MESSAGE),
    ):
        df._repartition()
