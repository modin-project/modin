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

import numpy as np
import pandas

from modin.config import NPartitions


def get_default_chunksize(length, num_splits):
    """
    Get the most equal chunksize possible based on length and number of splits.

    Parameters
    ----------
    length : int
        The length to split (number of rows/columns).
    num_splits : int
        The number of splits.

    Returns
    -------
    int
        Computed chunksize.
    """
    return (
        length // num_splits if length % num_splits == 0 else length // num_splits + 1
    )


def compute_chunksize(
    row_count=None, col_count=None, num_splits=None, default_block_size=32
):
    """
    Compute the number of rows and/or columns to include in each partition.

    Parameters
    ----------
    row_count : int, optional
        Rows count.
    col_count : int, optional
        Columns count.
    num_splits : int, optional
        Number of splits to separate the DataFrame into. `NPartitions` by default.
    default_block_size : int, default: 32
        Minimum number of rows/columns in a single split.

    Returns
    -------
    int
        - `row_count` and `col_count` are specified: returns tuple of ints otherwise,
        - `row_count`/`col_count` are specified: returns an integer number of rows/columns to split the
        DataFrame.
    """
    assert row_count is not None or col_count is not None

    if num_splits is None:
        num_splits = NPartitions.get()

    if row_count is not None:
        row_chunksize = get_default_chunksize(row_count, num_splits)
        # Take the min of the default and the memory-usage chunksize first to avoid a
        # large amount of small partitions.
        row_chunksize = max(1, row_chunksize, default_block_size)
        if col_count is None:
            return row_chunksize

    col_chunksize = get_default_chunksize(col_count, num_splits)
    # Take the min of the default and the memory-usage chunksize first to avoid a
    # large amount of small partitions.
    col_chunksize = max(1, col_chunksize, default_block_size)
    if row_count is None:
        return col_chunksize

    return row_chunksize, col_chunksize


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
    shape_part = {"row_count" if axis == 0 else "col_count": result.shape[axis]}
    chunksize = compute_chunksize(**shape_part, num_splits=num_splits)
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
