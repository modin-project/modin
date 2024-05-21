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

from __future__ import annotations

import re
from math import ceil
from typing import Generator, Hashable, List, Optional

import numpy as np
import pandas

from modin.config import MinColumnPartitionSize, MinRowPartitionSize, NPartitions


def compute_chunksize(axis_len: int, num_splits: int, min_block_size: int) -> int:
    """
    Compute the number of elements (rows/columns) to include in each partition.

    Chunksize is defined the same for both axes.

    Parameters
    ----------
    axis_len : int
        Element count in an axis.
    num_splits : int
        The number of splits.
    min_block_size : int
        Minimum number of rows/columns in a single split.

    Returns
    -------
    int
        Integer number of rows/columns to split the DataFrame will be returned.
    """
    if not isinstance(min_block_size, int) or min_block_size <= 0:
        raise ValueError(
            f"'min_block_size' should be int > 0, passed: {min_block_size=}"
        )

    chunksize = axis_len // num_splits
    if axis_len % num_splits:
        chunksize += 1
    # chunksize shouldn't be less than `min_block_size` to avoid a
    # large amount of small partitions.
    return max(chunksize, min_block_size)


def split_result_of_axis_func_pandas(
    axis: int,
    num_splits: int,
    result: pandas.DataFrame,
    min_block_size: int,
    length_list: Optional[list] = None,
) -> list[pandas.DataFrame]:
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
    min_block_size : int
        Minimum number of rows/columns in a single split.
    length_list : list of ints, optional
        List of slice lengths to split DataFrame into. This is used to
        return the DataFrame to its original partitioning schema.

    Returns
    -------
    list of pandas.DataFrames
        Splitted dataframe represented by list of frames.
    """
    return list(
        generate_result_of_axis_func_pandas(
            axis, num_splits, result, min_block_size, length_list
        )
    )


def generate_result_of_axis_func_pandas(
    axis: int,
    num_splits: int,
    result: pandas.DataFrame,
    min_block_size: int,
    length_list: Optional[list] = None,
) -> Generator:
    """
    Generate pandas DataFrame evenly based on the provided number of splits.

    Parameters
    ----------
    axis : {0, 1}
        Axis to split across. 0 means index axis when 1 means column axis.
    num_splits : int
        Number of splits to separate the DataFrame into.
        This parameter is ignored if `length_list` is specified.
    result : pandas.DataFrame
        DataFrame to split.
    min_block_size : int
        Minimum number of rows/columns in a single split.
    length_list : list of ints, optional
        List of slice lengths to split DataFrame into. This is used to
        return the DataFrame to its original partitioning schema.

    Yields
    ------
    Generator
        Generates 'num_splits' dataframes as a result of axis function.
    """
    if num_splits == 1:
        yield result
    else:
        if length_list is None:
            length_list = get_length_list(
                result.shape[axis], num_splits, min_block_size
            )
        # Inserting the first "zero" to properly compute cumsum indexing slices
        length_list = np.insert(length_list, obj=0, values=[0])
        sums = np.cumsum(length_list)
        axis = 0 if isinstance(result, pandas.Series) else axis

        for i in range(len(sums) - 1):
            # We do this to restore block partitioning
            if axis == 0:
                chunk = result.iloc[sums[i] : sums[i + 1]]
            else:
                chunk = result.iloc[:, sums[i] : sums[i + 1]]

            # Sliced MultiIndex still stores all encoded values of the original index, explicitly
            # asking it to drop unused values in order to save memory.
            if isinstance(chunk.axes[axis], pandas.MultiIndex):
                chunk = chunk.set_axis(
                    chunk.axes[axis].remove_unused_levels(), axis=axis, copy=False
                )
            yield chunk


def get_length_list(axis_len: int, num_splits: int, min_block_size: int) -> list:
    """
    Compute partitions lengths along the axis with the specified number of splits.

    Parameters
    ----------
    axis_len : int
        Element count in an axis.
    num_splits : int
        Number of splits along the axis.
    min_block_size : int
        Minimum number of rows/columns in a single split.

    Returns
    -------
    list of ints
        List of integer lengths of partitions.
    """
    chunksize = compute_chunksize(axis_len, num_splits, min_block_size)
    return [
        (
            chunksize
            if (i + 1) * chunksize <= axis_len
            else max(0, axis_len - i * chunksize)
        )
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
        chunk_size = compute_chunksize(
            axis_len=res_shape,
            num_splits=NPartitions.get(),
            min_block_size=(
                MinRowPartitionSize.get() if axis == 0 else MinColumnPartitionSize.get()
            ),
        )
        return ceil(res_shape / chunk_size)
    else:
        lsplits = left._partitions.shape[axis]
        rsplits = right._partitions.shape[axis]
        return min(lsplits + rsplits, NPartitions.get())
