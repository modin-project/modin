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

"""Tests the reduce and tree_reduce Modin algebra operators."""


import pytest
import numpy as np
import pandas
import modin.pandas as pd

from modin.pandas.test.utils import (
    arg_keys,
    bool_arg_keys,
    bool_arg_values,
)
from modin.config import NPartitions
from modin.utils import MODIN_UNNAMED_SERIES_LABEL

NPartitions.put(4)


@pytest.mark.parametrize(
    "hint_dtypes", bool_arg_values, ids=arg_keys("hint_dtypes", bool_arg_keys)
)
def test_reduce_small(hint_dtypes):
    modin_df = pd.DataFrame({"a": range(33), "b": range(33, 66)})
    modin_internal_df = modin_df._query_compiler._modin_frame
    # dtypes optional argument specifies expected dtype to avoid recomputing after
    dtypes = pandas.Series({"a": np.float64, "b": np.float64}) if hint_dtypes else None
    result = modin_internal_df.reduce(0, pandas.DataFrame.mean, dtypes=dtypes)
    result_as_pandas = result.to_pandas()
    expected = pandas.DataFrame(
        {"a": [16.0], "b": [49.0]}, index=[MODIN_UNNAMED_SERIES_LABEL]
    )
    assert result_as_pandas.equals(expected)
    assert result.dtypes.equals(expected.dtypes)


@pytest.mark.parametrize(
    "hint_dtypes", bool_arg_values, ids=arg_keys("hint_dtypes", bool_arg_keys)
)
def test_tree_reduce_small(hint_dtypes):
    data = {"a": (["a"] * 5) + ([np.nan] * 6), "b": ([np.nan] * 3) + (["c"] * 8)}
    modin_df = pd.DataFrame(data)
    modin_internal_df = modin_df._query_compiler._modin_frame
    # dtypes optional argument specifies expected dtype to avoid recomputing after
    dtypes = pandas.Series({"a": np.int64, "b": np.int64}) if hint_dtypes else None
    result = modin_internal_df.tree_reduce(
        0, pandas.DataFrame.count, pandas.DataFrame.sum, dtypes=dtypes
    )
    result_as_pandas = result.to_pandas()
    expected = pandas.DataFrame(
        {"a": [5], "b": [8]}, index=[MODIN_UNNAMED_SERIES_LABEL]
    )
    assert result_as_pandas.equals(expected)
    assert result.dtypes.equals(expected.dtypes)
