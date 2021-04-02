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
import pandas


def get_default_chunksize(length, num_splits):
    """
    Create the most equal chunksize possible based on length and number of splits.

    Parameters
    ----------
    length: int,
        The integer length to split (number of rows/columns).
    num_splits: int,
        The integer number of splits.

    Returns
    -------
    int,
        An integer chunksize.
    """
    return (
        length // num_splits if length % num_splits == 0 else length // num_splits + 1
    )


def compute_chunksize(df, num_splits, default_block_size=32, axis=None):
    """Compute the number of rows and/or columns to include in each partition.

    Parameters
    ----------
    df: pandas.DataFrame
        The DataFrame to split.
    num_splits: int,
        The maximum number of splits to separate the DataFrame into.
    default_block_size: int, (default 32)
        Minimum number of rows/columns (default set to 32x32).
    axis: int, {0: Index, 1: Columns, None: Both}
        The axis to split.

    Returns
    -------
    int if axis was specified, tuple of ints otherwise,
        If axis is 1 or 0, returns an integer number of rows/columns to split the
        DataFrame. If axis is None, return a tuple containing both.
    """
    if axis == 0 or axis is None:
        row_chunksize = get_default_chunksize(len(df.index), num_splits)
        # Take the min of the default and the memory-usage chunksize first to avoid a
        # large amount of small partitions.
        row_chunksize = max(1, row_chunksize, default_block_size)
        if axis == 0:
            return row_chunksize
    # We always execute this because we can only get here if axis is 1 or None.
    col_chunksize = get_default_chunksize(len(df.columns), num_splits)
    # Take the min of the default and the memory-usage chunksize first to avoid a
    # large amount of small partitions.
    col_chunksize = max(1, col_chunksize, default_block_size)
    if axis == 1:
        return col_chunksize

    return row_chunksize, col_chunksize


def split_result_of_axis_func_pandas(axis, num_splits, result, length_list=None):
    """
    Split the Pandas DataFrame evenly based on the provided number of splits.

    Parameters
    ----------
    axis: int {0: Index, 1: Columns}
        The axis to split across.
    num_splits: int,
        The number of even splits to create. Ignored if `lenght_list` is specified.
    result: pandas.DataFrame,
        DataFrame to split.
    length_list: list of ints (optional)
        The list of lengths to split this DataFrame into. This is used to
        return the DataFrame to its original partitioning schema.

    Returns
    -------
    list of pandas.DataFrames,
        Splitted frame represented by list of frames.
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
    chunksize = compute_chunksize(result, num_splits, axis=axis)
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
    """Compute number of rows of passed pandas.DataFrame"""
    assert isinstance(df, pandas.DataFrame)
    return len(df) if len(df) > 0 else 0


def width_fn_pandas(df):
    """Compute number of columns of passed pandas.DataFrame"""
    assert isinstance(df, pandas.DataFrame)
    return len(df.columns) if len(df.columns) > 0 else 0
