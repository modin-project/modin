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

from numpy.testing import assert_array_equal
import matplotlib.colors as mcolors
import modin.pandas as pd


def test_pandas_iterable():
    # Using a list or series yields equivalent
    # colormaps, i.e the series isn't seen as
    # a single color
    lst = ["red", "blue", "green"]
    s = pd.Series(lst)
    cm1 = mcolors.ListedColormap(lst, N=5)
    cm2 = mcolors.ListedColormap(s, N=5)
    assert_array_equal(cm1.colors, cm2.colors)
