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

import re
from typing import Hashable, List
import contextlib

import numpy as np
import pandas

from modin.config import MinPartitionSize, NPartitions
from math import ceil


@contextlib.contextmanager
def _nullcontext(dummy_value=None):  # noqa: PR01
    """
    Act as a replacement for contextlib.nullcontext missing in older Python.

    Notes
    -----
    contextlib.nullcontext is only available from Python 3.7.
    """
    yield dummy_value


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
    if num_splits == 1:
        return [result]

    if length_list is None:
        length_list = get_length_list(result.shape[axis], num_splits)
    # Inserting the first "zero" to properly compute cumsum indexing slices
    length_list = np.insert(length_list, obj=0, values=[0])

    sums = np.cumsum(length_list)
    axis = 0 if isinstance(result, pandas.Series) else axis
    # We do this to restore block partitioning
    if axis == 0:
        chunked = [result.iloc[sums[i] : sums[i + 1]] for i in range(len(sums) - 1)]
    else:
        chunked = [result.iloc[:, sums[i] : sums[i + 1]] for i in range(len(sums) - 1)]

    return [
        # Sliced MultiIndex still stores all encoded values of the original index, explicitly
        # asking it to drop unused values in order to save memory.
        chunk.set_axis(chunk.axes[axis].remove_unused_levels(), axis=axis, copy=False)
        if isinstance(chunk.axes[axis], pandas.MultiIndex)
        else chunk
        for chunk in chunked
    ]


def get_length_list(axis_len: int, num_splits: int) -> list:
    """
    Compute partitions lengths along the axis with the specified number of splits.

    Parameters
    ----------
    axis_len : int
        Element count in an axis.
    num_splits : int
        Number of splits along the axis.

    Returns
    -------
    list of ints
        List of integer lengths of partitions.
    """
    chunksize = compute_chunksize(axis_len, num_splits)
    return [
        chunksize
        if (i + 1) * chunksize <= axis_len
        else max(0, axis_len - i * chunksize)
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


def get_group_names(regex: "re.Pattern") -> "List[Hashable]":
    """
    Get named groups from compiled regex.

    Unnamed groups are numbered.

    Parameters
    ----------
    regex : compiled regex

    Returns
    -------
    list of column labels
    """
    names = {v: k for k, v in regex.groupindex.items()}
    return [names.get(1 + i, i) for i in range(regex.groups)]


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
    lshape = left._row_lengths_cache if axis == 0 else left._column_widths_cache
    rshape = right._row_lengths_cache if axis == 0 else right._column_widths_cache

    if lshape is not None and rshape is not None:
        res_shape = sum(lshape) + sum(rshape)
        chunk_size = compute_chunksize(axis_len=res_shape, num_splits=NPartitions.get())
        return ceil(res_shape / chunk_size)
    else:
        lsplits = left._partitions.shape[axis]
        rsplits = right._partitions.shape[axis]
        return min(lsplits + rsplits, NPartitions.get())
