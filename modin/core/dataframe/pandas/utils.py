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

    Parameters
    ----------
    dfs : list
        List of pandas DataFrames to concatenate.

    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame.
    """
    categoricals_columns = set.intersection(
        *[set(df.select_dtypes("category").columns.tolist()) for df in dfs]
    )

    for col in categoricals_columns:
        uc = union_categoricals([df[col] for df in dfs])
        for df in dfs:
            df[col] = pandas.Categorical(df[col], categories=uc.categories)

    return pandas.concat(dfs)
