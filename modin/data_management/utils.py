from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas


def compute_chunksize(length, num_splits):
    # We do this to avoid zeros and having an extremely large last partition
    return (
        length // num_splits if length % num_splits == 0 else length // num_splits + 1
    )


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
    if axis == 0 or type(result) is pandas.Series:
        chunksize = compute_chunksize(len(result), num_splits)
        return [
            result.iloc[chunksize * i : chunksize * (i + 1)] for i in range(num_splits)
        ]
    else:
        chunksize = compute_chunksize(len(result.columns), num_splits)
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
