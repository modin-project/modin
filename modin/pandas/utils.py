from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas

import collections
import numpy as np
import ray
import time
import gc

from . import get_npartitions

_NAN_BLOCKS = {}
_MEMOIZER_CAPACITY = 1000  # Capacity per function


class LRUCache(object):
    """A LRUCache implemented with collections.OrderedDict

    Notes:
        - OrderedDict will record the order each item is inserted.
        - The head of the queue will be LRU items.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = collections.OrderedDict()

    def __contains__(self, key):
        return key in self.cache

    def __getitem__(self, key):
        """Retrieve item from cache and re-insert it to the back of the queue
        """
        value = self.cache.pop(key)
        self.cache[key] = value
        return value

    def __setitem__(self, key, value):
        if key in self.cache:
            self.cache.pop(key)

        if len(self.cache) >= self.capacity:
            # Pop oldest items at the beginning of the queue
            self.cache.popitem(last=False)

        self.cache[key] = value


class memoize(object):
    """A basic memoizer that cache the input and output of the remote function

    Notes:
        - How is this implemented?
          This meoizer is implemented by adding a caching layer to the remote
          function's remote attribute. When user call f.remote(*args), we will
          first check against the cache, and then call the ray remote function
          if we can't find the return value in the cache.
        - When should this be used?
          This should be used when we anticipate temporal locality for the
          function. For example, we can reasonally assume users will perform
          columnar operation repetitively over time (like sum() or loc[]).
        - Caveat
          Don't use this decorator if the any argument to the remote function
          will mutate. Following snippet will fail
          ```py
              @memoize
              @ray.remote
              def f(obj):
                ...

              mutable_obj = [1]
              oid_1 = f.remote(mutable_obj) # will be cached

              mutable_obj.append(3)
              oid_2 = f.remote(mutable_obj) # cache hit!

              oid_1 == oid_2 # True!
           ```
           In short, use this function sparingly. The ideal case is that all
           inputs are ray ObjectIDs because they are immutable objects.
        - Future Development
          - Fix the mutability bug
          - Dynamic cache size (Fixed as 1000 for now)
    """

    def __init__(self, f):
        # Save of remote function
        self.old_remote_func = f.remote
        self.cache = LRUCache(capacity=_MEMOIZER_CAPACITY)

    def remote(self, *args):
        """Return cached result if the arguments are cached
        """
        args = tuple(args)

        if args in self.cache:
            cached_result = self.cache[args]
            return cached_result

        result = self.old_remote_func(*args)
        self.cache[args] = result
        return result


def post_task_gc(func):
    """Perform garbage collection after the task is executed.

    Usage:
        ```
        @ray.remote
        @post_task_gc
        def memory_hungry_op():
            ...
        ```
    Note:
        - This will invoke the GC for the entire process. Expect
          About 100ms latency.
        - We have a basic herustic in place to balance of tradeoff between
          speed and memory. If the task takes more than 500ms to run, we
          will do the GC.
    """

    def wrapped(*args):
        start_time = time.time()

        result = func(*args)

        duration_s = time.time() - start_time
        duration_ms = duration_s * 1000
        if duration_ms > 500:
            gc.collect()

        return result

    return wrapped


def _get_nan_block_id(n_row=1, n_col=1, transpose=False):
    """A memory efficent way to get a block of NaNs.

    Args:
        n_rows(int): number of rows
        n_col(int): number of columns
        transpose(bool): if true, swap rows and columns
    Returns:
        ObjectID of the NaN block
    """
    global _NAN_BLOCKS
    if transpose:
        n_row, n_col = n_col, n_row
    shape = (n_row, n_col)
    if shape not in _NAN_BLOCKS:
        arr = np.tile(np.array(np.NaN), shape)
        _NAN_BLOCKS[shape] = ray.put(pandas.DataFrame(data=arr))
    return _NAN_BLOCKS[shape]


def _get_lengths(df):
    """Gets the length of the DataFrame.
    Args:
        df: A remote pandas.DataFrame object.
    Returns:
        Returns an integer length of the DataFrame object. If the attempt
            fails, returns 0 as the length.
    """
    try:
        return len(df)
    # Because we sometimes have cases where we have summary statistics in our
    # DataFrames
    except TypeError:
        return 0


def _get_widths(df):
    """Gets the width (number of columns) of the DataFrame.
    Args:
        df: A remote pandas.DataFrame object.
    Returns:
        Returns an integer width of the DataFrame object. If the attempt
            fails, returns 0 as the length.
    """
    try:
        return len(df.columns)
    # Because we sometimes have cases where we have summary statistics in our
    # DataFrames
    except TypeError:
        return 0


def _partition_pandas_dataframe(df, num_partitions=None, row_chunksize=None):
    """Partitions a Pandas DataFrame object.
    Args:
        df (pandas.DataFrame): The pandas DataFrame to convert.
        npartitions (int): The number of partitions to split the DataFrame
            into. Has priority over chunksize.
        row_chunksize (int): The number of rows to put in each partition.
    Returns:
        [ObjectID]: A list of object IDs corresponding to the DataFrame
        partitions
    """
    if num_partitions is not None:
        row_chunksize = len(df) // num_partitions \
            if len(df) % num_partitions == 0 \
            else len(df) // num_partitions + 1
    else:
        assert row_chunksize is not None

    temp_df = df

    row_partitions = []
    while len(temp_df) > row_chunksize:
        t_df = temp_df[:row_chunksize]
        # reset_index here because we want a pandas.RangeIndex
        # within the partitions. It is smaller and sometimes faster.
        t_df.reset_index(drop=True, inplace=True)
        t_df.columns = pandas.RangeIndex(0, len(t_df.columns))
        top = ray.put(t_df)
        row_partitions.append(top)
        temp_df = temp_df[row_chunksize:]
    else:
        # Handle the last chunk correctly.
        # This call is necessary to prevent modifying original df
        temp_df = temp_df[:]
        temp_df.reset_index(drop=True, inplace=True)
        temp_df.columns = pandas.RangeIndex(0, len(temp_df.columns))
        row_partitions.append(ray.put(temp_df))

    return row_partitions


def from_pandas(df, num_partitions=None, chunksize=None):
    """Converts a pandas DataFrame to a Ray DataFrame.
    Args:
        df (pandas.DataFrame): The pandas DataFrame to convert.
        num_partitions (int): The number of partitions to split the DataFrame
            into. Has priority over chunksize.
        chunksize (int): The number of rows to put in each partition.
    Returns:
        A new Ray DataFrame object.
    """
    from .dataframe import DataFrame

    row_partitions = \
        _partition_pandas_dataframe(df, num_partitions, chunksize)

    return DataFrame(
        row_partitions=row_partitions, columns=df.columns, index=df.index)


def to_pandas(df):
    """Converts a Ray DataFrame to a pandas DataFrame/Series.
    Args:
        df (modin.DataFrame): The Ray DataFrame to convert.
    Returns:
        A new pandas DataFrame.
    """
    pandas_df = pandas.concat(ray.get(df._row_partitions), copy=False)
    pandas_df.index = df.index
    pandas_df.columns = df.columns
    return pandas_df


"""
Indexing Section
    Generate View Copy Helpers
    Function list:
        - `extract_block` (ray.remote function, move to EOF)
        - `_generate_block`
        - `_repartition_coord_df`
    Call Dependency:
        - _generate_block calls extract_block remote
    Pipeline:
        - Repartition the dataframe by npartition
        - Use case:
              The dataframe is a DataFrameView, the two coord_dfs only
              describe the subset of the block partition data. We want
              to create a new copy of this subset and re-partition
              the new dataframe.
"""


def _repartition_coord_df(old_coord_df, npartition):
    """Repartition the (view of) coord_df by npartition

    This function is best used when old_coord_df is not contigous.
    For example, it turns:

        partition index_within_partition
    i0  0         0
    i6  3         2

    into

        partition index_within_partition
    i0  0         0
    i6  0         1

    Note(simon):
        The resulting npartition will be <= npartition
        passed in.
    """
    length = len(old_coord_df)
    chunksize = (len(old_coord_df) // npartition if len(old_coord_df) %
                 npartition == 0 else len(old_coord_df) // npartition + 1)

    # genereate array([0, 0, 0, 1, 1, 1, 2])
    partitions = np.repeat(np.arange(npartition), chunksize)[:length]

    # generate array([0, 1, 2, 0, 1, 2, 0])
    final_n_partition = np.max(partitions)
    idx_in_part = np.tile(np.arange(chunksize), final_n_partition + 1)[:length]

    final_df = pandas.DataFrame({
        'partition': partitions,
        'index_within_partition': idx_in_part
    },
                                index=old_coord_df.index)

    return final_df


def _generate_blocks(old_row, new_row, old_col, new_col,
                     block_partition_2d_oid_arr):
    """
    Given the four coord_dfs:
        - Old Row Coord df
        - New Row Coord df
        - Old Col Coord df
        - New Col Coord df
    and the block partition array, this function will generate the new
    block partition array.
    """

    # We join the old and new coord_df to find out which chunk in the old
    # partition belongs to the chunk in the new partition. The new coord df
    # should have the same index as the old coord df in order to align the
    # row/column. This is guaranteed by _repartition_coord_df.
    def join(old, new):
        return new.merge(
            old, left_index=True, right_index=True, suffixes=('_new', '_old'))

    row_grouped = join(old_row, new_row).groupby('partition_new')
    col_grouped = join(old_col, new_col).groupby('partition_new')

    oid_lst = []
    for row_idx, row_lookup in row_grouped:
        for col_idx, col_lookup in col_grouped:
            oid = extract_block.remote(
                block_partition_2d_oid_arr,
                row_lookup,
                col_lookup,
                col_name_suffix='_old')
            oid_lst.append(oid)
    return np.array(oid_lst).reshape(len(row_grouped), len(col_grouped))


# Indexing
#  Generate View Copy Helpers
# END


def _mask_block_partitions(blk_partitions, row_metadata, col_metadata):
    """Return the squeezed/expanded block partitions as defined by
    row_metadata and col_metadata.

    Note:
        Very naive implementation. Extract one scaler at a time in a double
        for loop.
    """
    col_df = col_metadata._coord_df
    row_df = row_metadata._coord_df

    result_oids = []
    shape = (len(row_df.index), len(col_df.index))

    for _, row_partition_data in row_df.iterrows():
        for _, col_partition_data in col_df.iterrows():
            row_part = row_partition_data.partition
            col_part = col_partition_data.partition
            block_oid = blk_partitions[row_part, col_part]

            row_idx = row_partition_data['index_within_partition']
            col_idx = col_partition_data['index_within_partition']

            result_oid = extractor.remote(block_oid, [row_idx], [col_idx])
            result_oids.append(result_oid)
    return np.array(result_oids).reshape(shape)


def _map_partitions(func, partitions, *argslists):
    """Apply a function across the specified axis

    Args:
        func (callable): The function to apply
        partitions ([ObjectID]): The list of partitions to map func on.

    Returns:
        A list of partitions ([ObjectID]) with the result of the function
    """
    if partitions is None:
        return None

    assert (callable(func))
    if len(argslists) == 0:
        return [_deploy_func.remote(func, part) for part in partitions]
    elif len(argslists) == 1:
        return [
            _deploy_func.remote(func, part, argslists[0])
            for part in partitions
        ]
    else:
        assert (all(len(args) == len(partitions) for args in argslists))
        return [
            _deploy_func.remote(func, *args)
            for args in zip(partitions, *argslists)
        ]


def _create_block_partitions(partitions, axis=0, length=None):

    if length is not None and length != 0 and get_npartitions() > length:
        npartitions = length
    elif length == 0:
        npartitions = 1
    else:
        npartitions = get_npartitions()

    x = [
        create_blocks._submit(
            args=(partition, npartitions, axis), num_return_vals=npartitions)
        for partition in partitions
    ]

    # In the case that axis is 1 we have to transpose because we build the
    # columns into rows. Fortunately numpy is efficient at this.
    blocks = np.array(x) if axis == 0 else np.array(x).T

    # Sometimes we only get a single column or row, which is
    # problematic for building blocks from the partitions, so we
    # add whatever dimension we're missing from the input.
    return _fix_blocks_dimensions(blocks, axis)


def _create_blocks_helper(df, npartitions, axis):
    # Single partition dataframes don't need to be repartitioned
    if npartitions == 1:
        return df
    # In the case that the size is not a multiple of the number of partitions,
    # we need to add one to each partition to avoid losing data off the end
    block_size = df.shape[axis ^ 1] // npartitions \
        if df.shape[axis ^ 1] % npartitions == 0 \
        else df.shape[axis ^ 1] // npartitions + 1

    # if not isinstance(df.columns, pandas.RangeIndex):
    #     df.columns = pandas.RangeIndex(0, len(df.columns))

    blocks = [
        df.iloc[:, i * block_size:(i + 1) * block_size]
        if axis == 0 else df.iloc[i * block_size:(i + 1) * block_size, :]
        for i in range(npartitions)
    ]

    for block in blocks:
        block.columns = pandas.RangeIndex(0, len(block.columns))
        block.reset_index(inplace=True, drop=True)
    return blocks


def _inherit_docstrings(parent, excluded=[]):
    """Creates a decorator which overwrites a decorated class' __doc__
    attribute with parent's __doc__ attribute. Also overwrites __doc__ of
    methods and properties defined in the class with the __doc__ of matching
    methods and properties in parent.

    Args:
        parent (object): Class from which the decorated class inherits __doc__.
        excluded (list): List of parent objects from which the class does not
            inherit docstrings.

    Returns:
        function: decorator which replaces the decorated class' documentation
            parent's documentation.
    """

    def decorator(cls):
        if parent not in excluded:
            cls.__doc__ = parent.__doc__
        for attr, obj in cls.__dict__.items():
            parent_obj = getattr(parent, attr, None)
            if parent_obj in excluded or \
                    (not callable(parent_obj) and
                     not isinstance(parent_obj, property)):
                continue
            if callable(obj):
                obj.__doc__ = parent_obj.__doc__
            elif isinstance(obj, property) and obj.fget is not None:
                p = property(obj.fget, obj.fset, obj.fdel, parent_obj.__doc__)
                setattr(cls, attr, p)

        return cls

    return decorator


def _fix_blocks_dimensions(blocks, axis):
    """Checks that blocks is 2D, and adds a dimension if not.
    """
    if blocks.ndim < 2:
        return np.expand_dims(blocks, axis=axis ^ 1)
    return blocks


@ray.remote
def _deploy_func(func, dataframe, *args):
    """Deploys a function for the _map_partitions call.
    Args:
        dataframe (pandas.DataFrame): The pandas DataFrame for this partition.
    Returns:
        A futures object representing the return value of the function
        provided.
    """
    if len(args) == 0:
        return func(dataframe)
    else:
        return func(dataframe, *args)


@ray.remote
def extractor(df_chunk, row_loc, col_loc):
    """Retrieve an item from remote block
    """
    # We currently have to do the writable flag trick because a pandas bug
    # https://github.com/pandas-dev/pandas/issues/17192
    try:
        row_loc.flags.writeable = True
        col_loc.flags.writeable = True
    except AttributeError:
        # Locators might be scaler or python list
        pass
    # Python2 doesn't allow writable flag to be set on this object. Copying
    # into a list allows it to be used by iloc.
    except ValueError:
        row_loc = list(row_loc)
        col_loc = list(col_loc)
    return df_chunk.iloc[row_loc, col_loc]


@ray.remote
def writer(df_chunk, row_loc, col_loc, item):
    """Make a copy of the block and write new item to it
    """
    df_chunk = df_chunk.copy()
    df_chunk.iloc[row_loc, col_loc] = item
    return df_chunk


@ray.remote
def _build_col_widths(df_col):
    """Compute widths (# of columns) for each partition."""
    widths = np.array(
        ray.get([_deploy_func.remote(_get_widths, d) for d in df_col]))

    return widths


@ray.remote
def _build_row_lengths(df_row):
    """Compute lengths (# of rows) for each partition."""
    lengths = np.array(
        ray.get([_deploy_func.remote(_get_lengths, d) for d in df_row]))

    return lengths


@ray.remote
def _build_coord_df(lengths, index):
    """Build the coordinate DataFrame over all partitions."""
    filtered_lengths = [x for x in lengths if x > 0]
    coords = None
    if len(filtered_lengths) > 0:
        coords = np.vstack([
            np.column_stack((np.full(l, i), np.arange(l)))
            for i, l in enumerate(filtered_lengths)
        ])
    col_names = ("partition", "index_within_partition")
    return pandas.DataFrame(coords, index=index, columns=col_names)


@ray.remote
def create_blocks(df, npartitions, axis):
    return _create_blocks_helper(df, npartitions, axis)


@memoize
@ray.remote
def _blocks_to_series(*partition):
    """Used in indexing, concatenating blocks in a flexible way
    """
    if len(partition) == 0:
        return pandas.Series()

    partition = [pandas.Series(p.squeeze()) for p in partition]
    series = pandas.concat(partition)
    return series


@memoize
@ray.remote
def _blocks_to_col(*partition):
    if len(partition):
        return pandas.concat(partition, axis=0, copy=False)\
            .reset_index(drop=True)
    else:
        return pandas.DataFrame()


@memoize
@ray.remote
def _blocks_to_row(*partition):
    if len(partition):
        row_part = pandas.concat(partition, axis=1, copy=False)\
            .reset_index(drop=True)
        # Because our block partitions contain different indices (for the
        # columns), this change is needed to ensure correctness.
        row_part.columns = pandas.RangeIndex(0, len(row_part.columns))
        return row_part
    else:
        return pandas.DataFrame()


@ray.remote
def _reindex_helper(old_index, new_index, axis, npartitions, *df):
    """Reindexes a DataFrame to prepare for join/concat.

    Args:
        df: The DataFrame partition
        old_index: The index/column for this partition.
        new_index: The new index/column to assign.
        axis: Which axis to reindex over.

    Returns:
        A new set of blocks made up of DataFrames.
    """
    df = pandas.concat(df, axis=axis ^ 1)
    if axis == 1:
        df.index = old_index
    elif axis == 0:
        df.columns = old_index

    df = df.reindex(new_index, copy=False, axis=axis ^ 1)
    return _create_blocks_helper(df, npartitions, axis)


@ray.remote
def _co_op_helper(func, left_columns, right_columns, left_df_len, left_idx,
                  *zipped):
    """Copartition operation where two DataFrames must have aligned indexes.

    NOTE: This function assumes things are already copartitioned. Requires that
        row partitions are passed in as blocks.

    Args:
        func: The operation to conduct between two DataFrames.
        left_columns: The column names for the left DataFrame.
        right_columns: The column names for the right DataFrame.
        left_df_len: The length of the left. This is used so we can split up
            the zipped partitions.
        zipped: The DataFrame partitions (in blocks).

    Returns:
         A new set of blocks for the partitioned DataFrame.
    """
    left = pandas.concat(zipped[:left_df_len], axis=1, copy=False).copy()
    left.columns = left_columns
    if left_idx is not None:
        left.index = left_idx

    right = pandas.concat(zipped[left_df_len:], axis=1, copy=False).copy()
    right.columns = right_columns

    new_rows = func(left, right)

    new_blocks = _create_blocks_helper(new_rows, left_df_len, 0)

    if left_idx is not None:
        new_blocks.append(new_rows.index)

    return new_blocks


@ray.remote
def _match_partitioning(column_partition, lengths, index):
    """Match the number of rows on each partition. Used in df.merge().

    NOTE: This function can cause problems when there are empty column
        partitions.

        The way this function is intended to be used is as follows: Align the
        right partitioning with the left. The left will remain unchanged. Then,
        you are free to perform actions on a per-partition basis with the
        partitioning.

        The index objects must already be identical for this to work correctly.

    Args:
        column_partition: The column partition to change.
        lengths: The lengths of each row partition to match to.
        index: The index index of the column_partition. This is used to push
            down to the inner frame for correctness in the merge.

    Returns:
         A list of blocks created from this column partition.
    """
    partitioned_list = []

    columns = column_partition.columns
    # We set this because this is the only place we can guarantee correct
    # placement. We use it in the case the user wants to join on the index.
    column_partition.index = index
    for length in lengths:
        if len(column_partition) == 0:
            partitioned_list.append(pandas.DataFrame(columns=columns))
            continue

        partitioned_list.append(column_partition.iloc[:length, :])
        column_partition = column_partition.iloc[length:, :]
    return partitioned_list


@ray.remote
def _concat_index(*index_parts):
    return index_parts[0].append(index_parts[1:])


@ray.remote
def _compile_remote_dtypes(*column_of_blocks):
    small_dfs = [df.loc[0:0] for df in column_of_blocks]
    return pandas.concat(small_dfs, copy=False).dtypes


@ray.remote
def extract_block(blk_partitions, row_lookup, col_lookup, col_name_suffix):
    """
    This function extracts a single block from blk_partitions using
    the row_lookup and col_lookup.

    Pass in col_name_suffix='_old' when operate on a joined df.
    """

    def apply_suffix(s):
        return s + col_name_suffix

    # Address Arrow Error:
    #   Buffer source array is read-only
    row_lookup = row_lookup.copy()
    col_lookup = col_lookup.copy()

    df_columns = []
    for row_idx, row_df in row_lookup.groupby(apply_suffix('partition')):
        this_column = []
        for col_idx, col_df in col_lookup.groupby(apply_suffix('partition')):
            block_df_oid = blk_partitions[row_idx, col_idx]
            block_df = ray.get(block_df_oid)
            chunk = block_df.iloc[row_df[apply_suffix(
                'index_within_partition'
            )], col_df[apply_suffix('index_within_partition')]]
            this_column.append(chunk)
        df_columns.append(pandas.concat(this_column, axis=1))
    final_df = pandas.concat(df_columns)
    final_df.index = pandas.RangeIndex(0, final_df.shape[0])
    final_df.columns = pandas.RangeIndex(0, final_df.shape[1])

    return final_df
