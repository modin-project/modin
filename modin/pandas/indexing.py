from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas
from pandas.api.types import (is_scalar, is_list_like, is_bool)
from pandas.core.dtypes.common import is_integer
from pandas.core.indexing import IndexingError

import numpy as np
import ray
from warnings import warn

from .utils import (_get_nan_block_id, extractor, _repartition_coord_df,
                    _generate_blocks, _mask_block_partitions, writer,
                    _blocks_to_series)
from .index_metadata import _IndexMetadata
from .dataframe import DataFrame
from . import get_npartitions
"""Indexing Helper Class works as follows:

_Location_Indexer_Base provide methods framework for __getitem__
  and __setitem__ that work with Ray DataFrame's internal index. Base
  class's __{get,set}item__ takes in partitions & idx_in_partition data
  and perform lookup/item write.

_LocIndexer and _iLocIndexer is responsible for indexer specific logic and
  lookup computation. Loc will take care of enlarge DataFrame. Both indexer
  will take care of translating pandas's lookup to Ray DataFrame's internal
  lookup.

An illustration is available at
https://github.com/ray-project/ray/pull/1955#issuecomment-386781826
"""


def is_slice(x):
    return isinstance(x, slice)


def is_2d(x):
    return is_list_like(x) or is_slice(x)


def is_tuple(x):
    return isinstance(x, tuple)


def is_boolean_array(x):
    return is_list_like(x) and all(map(is_bool, x))


def is_integer_slice(x):
    if not is_slice(x):
        return False
    for pos in [x.start, x.stop, x.step]:
        if not ((pos is None) or is_integer(pos)):
            return False  # one position is neither None nor int
    return True


_ENLARGEMENT_WARNING = """
Passing list-likes to .loc or [] with any missing label will raise
KeyError in the future, you can use .reindex() as an alternative.

See the documentation here:
http://pandas.pydata.org/pandas-docs/stable/indexing.html#deprecate-loc-reindex-listlike
"""

_ILOC_INT_ONLY_ERROR = """
Location based indexing can only have [integer, integer slice (START point is
INCLUDED, END point is EXCLUDED), listlike of integers, boolean array] types.
"""

_VIEW_IS_COPY_WARNING = """
Modin is making a copy of of the DataFrame. This behavior diverges from Pandas.
This will be fixed in future releases.
"""


def _parse_tuple(tup):
    """Unpack the user input for getitem and setitem and compute ndim

    loc[a] -> ([a], :), 1D
    loc[[a,b],] -> ([a,b], :),
    loc[a,b] -> ([a], [b]), 0D
    """
    row_loc, col_loc = slice(None), slice(None)

    if is_tuple(tup):
        row_loc = tup[0]
        if len(tup) == 2:
            col_loc = tup[1]
        if len(tup) > 2:
            raise IndexingError('Too many indexers')
    else:
        row_loc = tup

    ndim = _compute_ndim(row_loc, col_loc)
    row_loc = [row_loc] if is_scalar(row_loc) else row_loc
    col_loc = [col_loc] if is_scalar(col_loc) else col_loc

    return row_loc, col_loc, ndim


def _is_enlargement(locator, coord_df):
    """Determine if a locator will enlarge the corrd_df.

    Enlargement happens when you trying to locate using labels isn't in the
    original index. In other words, enlargement == adding NaNs !
    """
    if is_list_like(locator) and not is_slice(
            locator) and len(locator) > 0 and not is_boolean_array(locator):
        n_diff_elems = len(pandas.Index(locator).difference(coord_df.index))
        is_enlargement_boolean = n_diff_elems > 0
        return is_enlargement_boolean
    return False


def _warn_enlargement():
    warn(FutureWarning(_ENLARGEMENT_WARNING))


def _compute_ndim(row_loc, col_loc):
    """Compute the ndim of result from locators
    """
    row_scaler = is_scalar(row_loc)
    col_scaler = is_scalar(col_loc)

    if row_scaler and col_scaler:
        ndim = 0
    elif row_scaler ^ col_scaler:
        ndim = 1
    else:
        ndim = 2

    return ndim


class _Location_Indexer_Base(object):
    """Base class for location indexer like loc and iloc
    """

    def __init__(self, ray_df):
        self.df = ray_df
        self.col_coord_df = ray_df._col_metadata._coord_df
        self.row_coord_df = ray_df._row_metadata._coord_df
        self.block_oids = ray_df._block_partitions

        self.is_view = False
        if isinstance(ray_df, DataFrameView):
            self.block_oids = ray_df._block_partitions_data
            self.is_view = True

    def __getitem__(self, row_lookup, col_lookup, ndim):
        """
        Args:
            row_lookup: A pandas DataFrame, a partial view from row_coord_df
            col_lookup: A pandas DataFrame, a partial view from col_coord_df
            ndim: the dimension of returned data
        """
        if ndim == 2:
            return self._generate_view_copy(row_lookup, col_lookup)

        extracted = self._retrive_items(row_lookup, col_lookup)
        if ndim == 1:
            result = ray.get(_blocks_to_series.remote(*extracted)).squeeze()

            if is_scalar(result):
                result = pandas.Series(result)

            scaler_axis = row_lookup if len(row_lookup) == 1 else col_lookup
            series_name = scaler_axis.iloc[0].name
            result.name = series_name

            index_axis = row_lookup if len(col_lookup) == 1 else col_lookup
            result.index = index_axis.index

        if ndim == 0:
            result = ray.get(extracted[0]).squeeze()

        return result

    def _retrive_items(self, row_lookup, col_lookup):
        """Given lookup dataframes, return a list of result oids
        """
        result_oids = []

        # We have to copy before we groupby because
        # https://github.com/pandas-dev/pandas/issues/10043
        row_groups = row_lookup.copy().groupby('partition')
        col_groups = col_lookup.copy().groupby('partition')
        for row_blk, row_data in row_groups:
            for col_blk, col_data in col_groups:
                block_oid = self.block_oids[row_blk, col_blk]
                row_idx = row_data['index_within_partition']
                col_idx = col_data['index_within_partition']

                result_oid = extractor.remote(block_oid, row_idx, col_idx)
                result_oids.append(result_oid)
        return result_oids

    def _generate_view_copy(self, row_lookup, col_lookup):
        """Generate a new DataFrame by making copies.

        Note (simon):
            - This is a temporary replacement for _generate_view
              function below.
        """
        warn(_VIEW_IS_COPY_WARNING)

        row_lookup_new = _repartition_coord_df(row_lookup, get_npartitions())
        col_lookup_new = _repartition_coord_df(col_lookup, get_npartitions())

        new_blocks = _generate_blocks(row_lookup, row_lookup_new, col_lookup,
                                      col_lookup_new, self.block_oids)

        row_lengths_oid = ray.put(np.bincount(row_lookup_new['partition']))
        col_lengths_oid = ray.put(np.bincount(col_lookup_new['partition']))

        new_row_metadata = _IndexMetadata(
            coord_df_oid=row_lookup_new, lengths_oid=row_lengths_oid)

        new_col_metadata = _IndexMetadata(
            coord_df_oid=col_lookup_new, lengths_oid=col_lengths_oid)

        df_view = DataFrame(
            block_partitions=new_blocks,
            row_metadata=new_row_metadata,
            col_metadata=new_col_metadata,
            index=row_lookup.index,
            columns=col_lookup.index)

        return df_view

    def _generate_view(self, row_lookup, col_lookup):
        """Generate a DataFrameView from lookup

        Note (simon):
            - This is not used because of index metadata was broken
        """
        row_lengths = [0] * len(self.df._row_metadata._lengths)
        for i in row_lookup["partition"]:
            row_lengths[i] += 1
        col_lengths = [0] * len(self.df._col_metadata._lengths)
        for i in col_lookup["partition"]:
            col_lengths[i] += 1

        row_lengths_oid = ray.put(np.array(row_lengths))
        col_lengths_oid = ray.put(np.array(col_lengths))

        row_metadata_view = _IndexMetadata(
            coord_df_oid=row_lookup, lengths_oid=row_lengths_oid)

        col_metadata_view = _IndexMetadata(
            coord_df_oid=col_lookup, lengths_oid=col_lengths_oid)

        df_view = DataFrameView(
            block_partitions=self.block_oids,
            row_metadata=row_metadata_view,
            col_metadata=col_metadata_view,
            index=row_metadata_view.index,
            columns=col_metadata_view.index)

        return df_view

    def __setitem__(self, row_lookup, col_lookup, item):
        """
        Args:
            row_lookup: A pandas DataFrame, a partial view from row_coord_df
            col_lookup: A pandas DataFrame, a partial view from col_coord_df
            item: The new item needs to be set. It can be any shape that's
                broadcastable to the product of the lookup tables.
        """
        to_shape = (len(row_lookup), len(col_lookup))
        item = self._broadcast_item(item, to_shape)
        self._write_items(row_lookup, col_lookup, item)

    def _broadcast_item(self, item, to_shape):
        """Use numpy to broadcast or reshape item.

        Notes:
            - Numpy is memory efficent, there shouldn't be performance issue.
        """
        try:
            item = np.array(item)
            if np.prod(to_shape) == np.prod(item.shape):
                return item.reshape(to_shape)
            else:
                return np.broadcast_to(item, to_shape)
        except ValueError:
            from_shape = np.array(item).shape
            raise ValueError("could not broadcast input array from \
                shape {from_shape} into shape {to_shape}".format(
                from_shape=from_shape, to_shape=to_shape))

    def _write_items(self, row_lookup, col_lookup, item):
        """Perform remote write and replace blocks.
        """

        # We have to copy before we groupby because
        # https://github.com/pandas-dev/pandas/issues/10043
        row_groups = row_lookup.copy().groupby('partition')
        col_groups = col_lookup.copy().groupby('partition')

        row_item_index = 0
        for row_blk, row_data in row_groups:
            row_len = len(row_data)

            col_item_index = 0
            for col_blk, col_data in col_groups:
                col_len = len(col_data)

                block_oid = self.block_oids[row_blk, col_blk]
                row_idx = row_data['index_within_partition']
                col_idx = col_data['index_within_partition']

                item_to_write = item[row_item_index:row_item_index +
                                     row_len, col_item_index:col_item_index +
                                     col_len]

                result_oid = writer.remote(block_oid, row_idx, col_idx,
                                           item_to_write)

                if self.is_view:
                    self.df._block_partitions_data[row_blk,
                                                   col_blk] = result_oid
                else:
                    self.df._block_partitions[row_blk, col_blk] = result_oid

                col_item_index += col_len
            row_item_index += row_len


class _Loc_Indexer(_Location_Indexer_Base):
    """A indexer for ray_df.loc[] functionality"""

    def __getitem__(self, key):
        row_loc, col_loc, ndim = _parse_tuple(key)
        self._handle_enlargement(row_loc, col_loc)
        row_lookup, col_lookup = self._compute_lookup(row_loc, col_loc)
        ndim = self._expand_dim(row_lookup, col_lookup, ndim)
        result = super(_Loc_Indexer, self).__getitem__(row_lookup, col_lookup,
                                                       ndim)
        return result

    def __setitem__(self, key, item):
        row_loc, col_loc, _ = _parse_tuple(key)
        self._handle_enlargement(row_loc, col_loc)
        row_lookup, col_lookup = self._compute_lookup(row_loc, col_loc)
        super(_Loc_Indexer, self).__setitem__(row_lookup, col_lookup, item)

    def _handle_enlargement(self, row_loc, col_loc):
        """Handle Enlargement (if there is one).

        Returns:
            None
        """
        locators = [row_loc, col_loc]
        coord_dfs = [self.row_coord_df, self.col_coord_df]
        axis = ['row', 'col']
        metadata = {'row': self.df._row_metadata, 'col': self.df._col_metadata}

        for loc, coord, axis in zip(locators, coord_dfs, axis):
            if _is_enlargement(loc, coord):
                new_meta = self._enlarge_axis(loc, axis=axis)
                _warn_enlargement()
                metadata[axis] = new_meta

        self.row_coord_df = metadata['row']._coord_df
        self.col_coord_df = metadata['col']._coord_df

    def _enlarge_axis(self, locator, axis):
        """Add rows/columns to block partitions according to locator.

        Returns:
            metadata (_IndexMetadata)
        """
        # 1. Prepare variables
        row_based_bool = axis == 'row'
        # major == the axis of the locator
        major_meta = self.df._row_metadata if row_based_bool \
            else self.df._col_metadata
        minor_meta = self.df._col_metadata if row_based_bool \
            else self.df._row_metadata

        # 2. Compute the nan labels and add blocks
        nan_labels = self._compute_enlarge_labels(locator, major_meta.index)
        num_nan_labels = len(nan_labels)
        blk_part_n_row, blk_part_n_col = self.block_oids.shape

        nan_blk_lens = minor_meta._lengths
        nan_blks = np.array([[
            _get_nan_block_id(
                num_nan_labels, n_cols, transpose=not row_based_bool)
            for n_cols in nan_blk_lens
        ]])
        nan_blks = nan_blks.T if not row_based_bool else nan_blks

        self.block_oids = np.concatenate([self.block_oids, nan_blks],
                                         axis=0 if row_based_bool else 1)

        # 3. Prepare metadata to return
        nan_coord_df = pandas.DataFrame(data=[
            {
                '': name,
                'partition':
                blk_part_n_row if row_based_bool else blk_part_n_col,
                'index_within_partition': i
            } for name, i in zip(nan_labels, np.arange(num_nan_labels))
        ]).set_index('')

        coord_df = pandas.concat([major_meta._coord_df, nan_coord_df])
        coord_df = coord_df.loc[locator]  # Re-index that allows duplicates

        lens = major_meta._lengths
        lens = np.concatenate([lens, np.array([num_nan_labels])])
        lens_oid = ray.put(np.array(lens))

        metadata_view = _IndexMetadata(
            coord_df_oid=coord_df, lengths_oid=lens_oid)
        return metadata_view

    def _compute_enlarge_labels(self, locator, base_index):
        """Helper for _enlarge_axis, compute common labels and extra labels.

        Returns:
             nan_labels: The labels needs to be added
        """
        # base_index_type can be pd.Index or pd.DatetimeIndex
        # depending on user input and pandas behavior
        # See issue #2264
        base_index_type = type(base_index)
        locator_as_index = base_index_type(locator)

        nan_labels = locator_as_index.difference(base_index)
        common_labels = locator_as_index.intersection(base_index)

        if len(common_labels) == 0:
            raise KeyError(
                'None of [{labels}] are in the [{base_index_name}]'.format(
                    labels=list(locator_as_index), base_index_name=base_index))

        return nan_labels

    def _expand_dim(self, row_lookup, col_lookup, ndim):
        """Expand the dimension if necessary.
        This method is for cases like duplicate labels.
        """
        many_rows = len(row_lookup) > 1
        many_cols = len(col_lookup) > 1

        if ndim == 0 and (many_rows or many_cols):
            ndim = 1
        if ndim == 1 and (many_rows and many_cols):
            ndim = 2

        return ndim

    def _compute_lookup(self, row_loc, col_loc):
        # We use reindex for list to avoid duplicates.
        row_lookup = self.row_coord_df.loc[row_loc]
        col_lookup = self.col_coord_df.loc[col_loc]
        return row_lookup, col_lookup


class _iLoc_Indexer(_Location_Indexer_Base):
    """A indexer for ray_df.iloc[] functionality"""

    def __getitem__(self, key):
        row_loc, col_loc, ndim = _parse_tuple(key)

        self._check_dtypes(row_loc)
        self._check_dtypes(col_loc)

        row_lookup, col_lookup = self._compute_lookup(row_loc, col_loc)
        result = super(_iLoc_Indexer, self).__getitem__(
            row_lookup, col_lookup, ndim)
        return result

    def __setitem__(self, key, item):
        row_loc, col_loc, _ = _parse_tuple(key)

        self._check_dtypes(row_loc)
        self._check_dtypes(col_loc)

        row_lookup, col_lookup = self._compute_lookup(row_loc, col_loc)
        super(_iLoc_Indexer, self).__setitem__(row_lookup, col_lookup, item)

    def _compute_lookup(self, row_loc, col_loc):
        # We use reindex for list to avoid duplicates.
        return self.row_coord_df.iloc[row_loc], self.col_coord_df.iloc[col_loc]

    def _check_dtypes(self, locator):
        is_int = is_integer(locator)
        is_int_slice = is_integer_slice(locator)
        is_int_list = is_list_like(locator) and all(map(is_integer, locator))
        is_bool_arr = is_boolean_array(locator)

        if not any([is_int, is_int_slice, is_int_list, is_bool_arr]):
            raise ValueError(_ILOC_INT_ONLY_ERROR)


class DataFrameView(DataFrame):
    """A subclass of DataFrame where the index can be smaller than blocks.

    Deprecated because _generate_view_copy is used instead of _generate_view
    """

    def __init__(self, block_partitions, row_metadata, col_metadata, index,
                 columns):
        self._block_partitions = block_partitions
        self._row_metadata = row_metadata
        self._col_metadata = col_metadata
        self.index = index
        self.columns = columns

    def _get_block_partitions(self):
        oid_arr = _mask_block_partitions(self._block_partitions_data,
                                         self._row_metadata,
                                         self._col_metadata)
        return oid_arr

    def _set_block_partitions(self, new_block_partitions):
        self._block_partitions_data = new_block_partitions

    _block_partitions = property(_get_block_partitions, _set_block_partitions)
