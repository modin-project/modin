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


"""Collection of utility functions for the PandasDataFrame."""

import pandas
from pandas.api.types import union_categoricals


def concatenate(dfs):
    """
    Concatenate pandas DataFrames with saving 'category' dtype.

    All dataframes' columns must be equal to each other.

    Parameters
    ----------
    dfs : list
        List of pandas DataFrames to concatenate.

    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame.
    """
    for df in dfs:
        assert df.columns.equals(dfs[0].columns)
    for i in range(len(dfs[0].columns)):
        if dfs[0].dtypes.iloc[i].name != "category":
            continue
        columns = [df.iloc[:, i] for df in dfs]
        union = union_categoricals(columns)
        for df in dfs:
            df.iloc[:, i] = pandas.Categorical(
                df.iloc[:, i], categories=union.categories
            )
    return pandas.concat(dfs)
