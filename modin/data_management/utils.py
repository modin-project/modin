from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

import numpy as np
import pandas


def get_default_chunksize(length, num_splits):
    """Creates the most equal chunksize possible based on length and number of splits.

    Args:
        length: The integer length to split (number of rows/columns).
        num_splits: The integer number of splits.

    Returns:
        An integer chunksize.
    """
    return (
        length // num_splits if length % num_splits == 0 else length // num_splits + 1
    )


def compute_chunksize(df, num_splits, default_block_size=32, axis=None):
    """Computes the number of rows and/or columns to include in each partition.

    Args:
        df: The DataFrame to split.
        num_splits: The maximum number of splits to separate the DataFrame into.
        default_block_size: Minimum number of rows/columns (default set to 32x32).
        axis: The axis to split. (0: Index, 1: Columns, None: Both)

    Returns:
         If axis is 1 or 0, returns an integer number of rows/columns to split the
         DataFrame. If axis is None, return a tuple containing both.
    """
    # TODO(williamma12): Change this to not use an empty dataframe.
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


def compute_lengths(df, axis, num_splits, default_block_size=32):
    """Computes lengths and/or widths of the dataframe partitions

    Args:
        df: The DataFrame to split.
        axis: The axis to split. (0: Index, 1: Columns)
        num_splits: The maximum number of splits to separate the DataFrame into.
        default_block_size: Minimum number of rows/columns (default set to 32x32).

    Returns:
         Returns a list of the lengths/widths of the partitioned DataFrame.
    """
    total_length = len(df.columns) if axis == 1 else len(df)
    chunksize = compute_chunksize(df, num_splits, axis=axis)
    if chunksize > total_length:
        lengths = [total_length]
    else:
        lengths = [
            chunksize
            if total_length > (chunksize * (i + 1))
            else 0
            if total_length < (chunksize * i)
            else total_length - (chunksize * i)
            for i in range(num_splits)
        ]
    return lengths


def _get_nan_block_id(partition_class, n_row=1, n_col=1, transpose=False):
    """A memory efficient way to get a block of NaNs.

    Args:
        partition_class (BaseFramePartition): The class to use to put the object
            in the remote format.
        n_row(int): The number of rows.
        n_col(int): The number of columns.
        transpose(bool): If true, swap rows and columns.
    Returns:
        ObjectID of the NaN block.
    """
    global _NAN_BLOCKS
    if transpose:
        n_row, n_col = n_col, n_row
    shape = (n_row, n_col)
    if shape not in _NAN_BLOCKS:
        arr = np.tile(np.array(np.NaN), shape)
        # TODO Not use pandas.DataFrame here, but something more general.
        _NAN_BLOCKS[shape] = partition_class.put(pandas.DataFrame(data=arr))
    return _NAN_BLOCKS[shape]


def split_result_of_axis_func_pandas(axis, num_splits, result, length_list=None):
    """Split the Pandas result evenly based on the provided number of splits.

    Args:
        axis: The axis to split across.
        num_splits: The number of even splits to create.
        result: The result of the computation. This should be a Pandas
            DataFrame.
        length_list: The list of lengths to split this DataFrame into. This is used to
            return the DataFrame to its original partitioning schema.

    Returns:
        A list of Pandas DataFrames.
    """
    if num_splits == 1:
        return result
    if length_list is not None:
        length_list.insert(0, 0)
        sums = np.cumsum(length_list)
        if axis == 0:
            return [result.iloc[sums[i] : sums[i + 1]] for i in range(len(sums) - 1)]
        else:
            return [result.iloc[:, sums[i] : sums[i + 1]] for i in range(len(sums) - 1)]
    # We do this to restore block partitioning
    chunksize = compute_chunksize(result, num_splits, axis=axis)
    if axis == 0:
        return [
            result.iloc[chunksize * i : chunksize * (i + 1)] for i in range(num_splits)
        ]
    else:
        return [
            result.iloc[:, chunksize * i : chunksize * (i + 1)]
            for i in range(num_splits)
        ]


def length_fn_pandas(df):
    assert isinstance(df, pandas.DataFrame)
    return len(df)


def width_fn_pandas(df):
    assert isinstance(df, pandas.DataFrame)
    return len(df.columns)


def set_indices_for_pandas_concat(df, transposed=False):
    df.index = pandas.RangeIndex(len(df))
    df.columns = pandas.RangeIndex(len(df.columns))
    return df.T if transposed else df


def compute_partition_shuffle(old_lengths, new_lengths, old_index=None, new_index=None):
    """Calculates split old_length partitions into new_lengths partitions.

    Args:
        old_lengths: Lengths of the old partitions.
        new_lengths: Lengths of the new partitions.
        old_index: Current ordering of the labels.
        new_index: New ordering of the labels of the data.

    Returns:
        Tuple containing (1) how the new blocks are built (a list of tuples
        that each contain the old block index and the old block split index) and
        (2) how the old blocks are split (dictionary key-value pairs of old block
        indices and how they should be split). Any -1's indicate a dataframe of NaNs.
    """
    # Create a numpy arrays of index values to facilitate the shuffle calculations.
    # We use -1 to represent NaN indices.
    # old_index  old_block_index  new_block_index
    #     0            0                0
    #     1            0                0
    #     2            1                0
    #     3            1                1
    #    -1           -1                1

    # Create list for the old/new block index and internal indices of the old blocks.
    old_block_index = np.repeat(np.arange(len(old_lengths)), old_lengths)
    new_block_index = np.repeat(np.arange(len(new_lengths)), new_lengths)

    # Pad old block and internal indices to the new index length.
    if new_index is not None and len(new_index) > len(old_index):
        diff = len(new_index) - len(old_index)
        empty_rows = np.repeat(-1, diff)
        old_block_index = np.append(old_block_index, empty_rows)
        old_index = pandas.Index(old_index).append(
            pandas.Index(np.arange(-1, -(diff + 1), -1))
        )

    index = pandas.Index(old_index).get_indexer(new_index)

    # Using the index dataframe, we iterate through to calculate how we split
    # the old blocks and how the new blocks are built out of the old blocks.
    new_partitions = []
    old_partition_counts = defaultdict(int)
    old_partition_splits = defaultdict(list)
    new_partition_block = []
    prev_old_block_idx = -2
    prev_new_block_idx = -2

    for i in range(len(index)):
        old_block_idx = old_block_index[i]
        new_block_idx = new_block_index[i]
        # Calculate the old partition splits.
        if new_block_idx != prev_new_block_idx or old_block_idx != prev_old_block_idx:
            # Create a old partitions split if either of the block indices are not equal to the last one.
            old_partition_split_idx = old_partition_counts[old_block_idx]
            old_partition_counts[old_block_idx] += 1
            old_partition_splits[old_block_idx].append([index[i]])
        else:
            # Otherwise, add to the previous split.
            old_partition_split_idx = old_partition_counts[old_block_idx]
            old_partition_splits[old_block_idx][-1].append(index[i])
        # Save new block data
        if len(new_partition_block) > 0 and new_block_idx != prev_new_block_idx:
            # Save new partition block to new block if we are on a new block
            new_partitions.append(new_partition_block)
            new_partition_block = [(old_block_idx, old_partition_split_idx)]
        elif old_block_idx != prev_old_block_idx:
            # Add old partition split index if it we created a new split
            new_partition_block.append((old_block_idx, old_partition_split_idx))
        prev_old_block_idx = old_block_idx
        prev_new_block_idx = new_block_idx
    # Add last block to result
    new_partitions.append(new_partition_block)

    return new_partitions, old_partition_splits
