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

NPartitions.put(4)


@pytest.mark.parametrize(
    "hint_dtypes", bool_arg_values, ids=arg_keys("hint_dtypes", bool_arg_keys)
)
def test_map_elementwise(hint_dtypes):
    modin_df = pd.DataFrame({"a": range(33), "b": range(33, 66)})
    modin_internal_df = modin_df._query_compiler._modin_frame
    # dtypes optional argument specifies expected dtype to avoid recomputing after
    dtypes = pandas.Series({"a": np.bool, "b": np.bool}) if hint_dtypes else None
    result = modin_internal_df.map(lambda df: df < 32, dtypes=dtypes)
    result_as_pandas = result.to_pandas()
    expected = pandas.DataFrame({"a": [True] * 32 + [False], "b": [False] * 33})
    assert result_as_pandas.equals(expected)
    assert result.dtypes.equals(expected.dtypes)
    # The result of the following operation should be the same, even though
    # an `axis` argument is provided
    result = modin_internal_df.map(lambda df: df < 32, dtypes=dtypes, axis=0)
    result_as_pandas = result.to_pandas()
    assert result_as_pandas.equals(expected)
    assert result.dtypes.equals(expected.dtypes)
