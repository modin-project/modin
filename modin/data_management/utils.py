from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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


def compute_chunksize(df, num_splits, min_block_size=4096, axis=None):
    """Computes the number of rows and/or columns to include in each partition.

    Args:
        df: The DataFrame to split.
        num_splits: The maximum number of splits to separate the DataFrame into.
        min_block_size: The minimum number of bytes for a single partition.
        axis: The axis to split. (0: Index, 1: Columns, None: Both)

    Returns:
         If axis is 1 or 0, returns an integer number of rows/columns to split the
         DataFrame. If axis is None, return a tuple containing both.
    """
    if axis is not None:
        # If we're only chunking one axis, based on the math below we can create
        # extremely large partitions without this.
        # TODO: Make the math not require this for single axis
        min_block_size /= 2
    # We use the memory usage to compute the partitioning.
    # TODO: Create a filter for computing this, since it may be expensive.
    mem_usage = df.memory_usage().sum()

    # This happens in the case of a small DataFrame
    if mem_usage <= min_block_size:
        return df.shape[axis if axis is not None else slice(None)]
    else:
        # The chunksize based on the memory usage
        mem_usage_chunksize = np.sqrt(mem_usage // min_block_size)
        # We can run into some issues with the division if we don't correct for very
        # small memory chunksizes.
        if mem_usage_chunksize < 1:
            mem_usage_chunksize = 1
        if axis == 0 or axis is None:
            row_chunksize = get_default_chunksize(len(df.index), num_splits)
            # If we don't add 1 in the case where the mod is not 0, we end up leaving
            # out data.
            row_mem_usage_chunksize = len(df.index) // (
                int(len(df.index) // mem_usage_chunksize)
                if len(df.index) % mem_usage_chunksize == 0
                else int(len(df.index) // mem_usage_chunksize) + 1
            )
            # Take the min of the default and the memory-usage chunksize first to avoid a
            # large amount of small partitions.
            row_chunksize = max(1, row_chunksize, row_mem_usage_chunksize)
            if axis == 0:
                return row_chunksize

        # We always execute this because we can only get here if axis is 1 or None.
        col_chunksize = get_default_chunksize(len(df.columns), num_splits)
        # Adjust mem_usage_chunksize for non-perfect square roots to have better
        # partitioning.
        if mem_usage_chunksize - int(mem_usage_chunksize) != 0:
            mem_usage_chunksize += 1
        # Add 1 when mod is not 0 to avoid leaving out data.
        col_mem_usage_chunksize = len(df.columns) // (
            int(len(df.columns) // mem_usage_chunksize)
            if len(df.columns) % mem_usage_chunksize == 0
            else int(len(df.columns) // mem_usage_chunksize) + 1
        )
        # Take the min of the default and the memory-usage chunksize first to avoid a
        # large amount of small partitions.
        col_chunksize = max(1, col_chunksize, col_mem_usage_chunksize)
        if axis == 1:
            return col_chunksize

        return row_chunksize, col_chunksize


def _get_nan_block_id(partition_class, n_row=1, n_col=1, transpose=False):
    """A memory efficient way to get a block of NaNs.

    Args:
        partition_class (BaseRemotePartition): The class to use to put the object
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
    if axis == 0 or type(result) is pandas.Series:
        return [
            result.iloc[chunksize * i : chunksize * (i + 1)] for i in range(num_splits)
        ]
    else:
        return [
            result.iloc[:, chunksize * i : chunksize * (i + 1)]
            for i in range(num_splits)
        ]


def length_fn_pandas(df):
    assert isinstance(df, (pandas.DataFrame, pandas.Series)), "{}".format(df)
    return len(df)


def width_fn_pandas(df):
    assert isinstance(df, (pandas.DataFrame, pandas.Series)), "{}".format((df))
    if isinstance(df, pandas.DataFrame):
        return len(df.columns)
    else:
        return 1
