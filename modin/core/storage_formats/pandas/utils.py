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

"""Contains utility functions for frame partitioning."""

from modin.config import MinPartitionSize
import numpy as np
import pandas


def compute_chunksize(axis_len, num_splits, min_block_size=None):
    """
    Compute the number of elements (rows/columns) to include in each partition.

    Chunksize is defined the same for both axes.

    Parameters
    ----------
    axis_len : int
        Element count in an axis.
    num_splits : int
        The number of splits.
    min_block_size : int, optional
        Minimum number of rows/columns in a single split.
        If not specified, the value is assumed equal to ``MinPartitionSize``.

    Returns
    -------
    int
        Integer number of rows/columns to split the DataFrame will be returned.
    """
    if min_block_size is None:
        min_block_size = MinPartitionSize.get()

    assert min_block_size > 0, "`min_block_size` should be > 0"

    chunksize = axis_len // num_splits
    if axis_len % num_splits:
        chunksize += 1
    # chunksize shouldn't be less than `min_block_size` to avoid a
    # large amount of small partitions.
    return max(chunksize, min_block_size)


def split_result_of_axis_func_pandas(axis, num_splits, result, length_list=None):
    """
    Split pandas DataFrame evenly based on the provided number of splits.

    Parameters
    ----------
    axis : {0, 1}
        Axis to split across. 0 means index axis when 1 means column axis.
    num_splits : int
        Number of splits to separate the DataFrame into.
        This parameter is ignored if `length_list` is specified.
    result : pandas.DataFrame
        DataFrame to split.
    length_list : list of ints, optional
        List of slice lengths to split DataFrame into. This is used to
        return the DataFrame to its original partitioning schema.

    Returns
    -------
    list of pandas.DataFrames
        Splitted dataframe represented by list of frames.
    """
    if length_list is not None:
        length_list.insert(0, 0)
        sums = np.cumsum(length_list)
        if axis == 0 or isinstance(result, pandas.Series):
            return [result.iloc[sums[i] : sums[i + 1]] for i in range(len(sums) - 1)]
        else:
            return [result.iloc[:, sums[i] : sums[i + 1]] for i in range(len(sums) - 1)]

    if num_splits == 1:
        return [result]
    # We do this to restore block partitioning
    chunksize = compute_chunksize(result.shape[axis], num_splits)
    if axis == 0 or isinstance(result, pandas.Series):
        return [
            result.iloc[chunksize * i : chunksize * (i + 1)] for i in range(num_splits)
        ]
    else:
        return [
            result.iloc[:, chunksize * i : chunksize * (i + 1)]
            for i in range(num_splits)
        ]


def length_fn_pandas(df):
    """
    Compute number of rows of passed `pandas.DataFrame`.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    int
    """
    assert isinstance(df, pandas.DataFrame)
    return len(df) if len(df) > 0 else 0


def width_fn_pandas(df):
    """
    Compute number of columns of passed `pandas.DataFrame`.

    Parameters
    ----------
    df : pandas.DataFrame

    Returns
    -------
    int
    """
    assert isinstance(df, pandas.DataFrame)
    return len(df.columns) if len(df.columns) > 0 else 0
