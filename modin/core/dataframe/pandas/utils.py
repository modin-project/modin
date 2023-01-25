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
from math import ceil

from modin.config import NPartitions


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
    for i in dfs[0].columns.get_indexer_for(dfs[0].select_dtypes("category").columns):
        columns = [df.iloc[:, i] for df in dfs]
        union = union_categoricals(columns)
        for df in dfs:
            df.isetitem(
                i, pandas.Categorical(df.iloc[:, i], categories=union.categories)
            )
    return pandas.concat(dfs)


def merge_partitioning(left, right, axis=1):
    """
    Get the number of splits across the `axis` for the two dataframes being concatenated.

    Parameters
    ----------
    left : PandasDataframe
    right : PandasDataframe
    axis : int, default: 1

    Returns
    -------
    int
    """
    # Avoiding circular imports from pandas query compiler
    from modin.core.storage_formats.pandas.utils import compute_chunksize

    lsplits = left._partitions.shape[axis]
    rsplits = right._partitions.shape[axis]

    lshape = left._row_lengths_cache if axis == 0 else left._column_widths_cache
    rshape = right._row_lengths_cache if axis == 0 else right._column_widths_cache

    if lshape is not None and rshape is not None:
        res_shape = sum(lshape) + sum(rshape)
        chunk_size = compute_chunksize(axis_len=res_shape, num_splits=NPartitions.get())
        return ceil(res_shape / chunk_size)
    else:
        return min(lsplits + rsplits, NPartitions.get())
