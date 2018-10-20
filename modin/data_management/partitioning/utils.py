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
