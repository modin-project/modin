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
import modin.pandas as pd
from matplotlib import cbook


def test_reshape2d_pandas():
    # separate to allow the rest of the tests to run if no pandas...
    X = np.arange(30).reshape(10, 3)
    x = pd.DataFrame(X, columns=["a", "b", "c"])
    Xnew = cbook._reshape_2D(x, "x")
    # Need to check each row because _reshape_2D returns a list of arrays:
    for x, xnew in zip(X.T, Xnew):
        np.testing.assert_array_equal(x, xnew)


def test_index_of_pandas():
    # separate to allow the rest of the tests to run if no pandas...
    X = np.arange(30).reshape(10, 3)
    x = pd.DataFrame(X, columns=["a", "b", "c"])
    Idx, Xnew = cbook.index_of(x)
    np.testing.assert_array_equal(X, Xnew)
    IdxRef = np.arange(10)
    np.testing.assert_array_equal(Idx, IdxRef)


def test_safe_first_element_pandas_series():
    # deliberately create a pandas series with index not starting from 0
    s = pd.Series(range(5), index=range(10, 15))
    actual = cbook._safe_first_finite(s)
    assert actual == 0
