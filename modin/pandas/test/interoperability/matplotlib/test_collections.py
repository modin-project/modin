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

import modin.pandas as pd
from matplotlib.collections import Collection


def test_pandas_indexing():
    # Should not fail break when faced with a
    # non-zero indexed series
    index = [11, 12, 13]
    ec = fc = pd.Series(["red", "blue", "green"], index=index)
    lw = pd.Series([1, 2, 3], index=index)
    ls = pd.Series(["solid", "dashed", "dashdot"], index=index)
    aa = pd.Series([True, False, True], index=index)

    Collection(edgecolors=ec)
    Collection(facecolors=fc)
    Collection(linewidths=lw)
    Collection(linestyles=ls)
    Collection(antialiaseds=aa)
