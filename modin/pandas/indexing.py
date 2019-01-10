from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas
from pandas.api.types import is_scalar, is_list_like, is_bool
from pandas.core.dtypes.common import is_integer
from pandas.core.indexing import IndexingError
from typing import Tuple
from warnings import warn

from .dataframe import DataFrame
from .series import SeriesView

"""Indexing Helper Class works as follows:

_LocationIndexerBase provide methods framework for __getitem__
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
            raise IndexingError("Too many indexers")
    else:
        row_loc = tup

    ndim = _compute_ndim(row_loc, col_loc)
    row_scaler = is_scalar(row_loc)
    col_scaler = is_scalar(col_loc)
    row_loc = [row_loc] if row_scaler else row_loc
    col_loc = [col_loc] if col_scaler else col_loc

    return row_loc, col_loc, ndim, row_scaler, col_scaler


def _is_enlargement(locator, global_index):
    """Determine if a locator will enlarge the global index.

    Enlargement happens when you trying to locate using labels isn't in the
    original index. In other words, enlargement == adding NaNs !
    """
    if (
        is_list_like(locator)
        and not is_slice(locator)
        and len(locator) > 0
        and not is_boolean_array(locator)
        and (isinstance(locator, type(global_index[0])) and locator not in global_index)
    ):
        n_diff_elems = len(pandas.Index(locator).difference(global_index))
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


class _LocationIndexerBase(object):
    """Base class for location indexer like loc and iloc
    """

    def __init__(self, ray_df: DataFrame):
        self.df = ray_df
        self.qc = ray_df._query_compiler
        self.row_scaler = False
        self.col_scaler = False

    def __getitem__(
        self, row_lookup: pandas.Index, col_lookup: pandas.Index, ndim: int
    ):
        qc_view = self.qc.view(row_lookup, col_lookup)

        if ndim == 2:
            return DataFrame(query_compiler=qc_view)
        elif ndim == 0:
            return qc_view.squeeze(ndim=0)
        else:
            single_axis = 1 if self.col_scaler else 0
            return SeriesView(
                qc_view.squeeze(ndim=1, axis=single_axis),
                self.df,
                (row_lookup, col_lookup),
            )

    def __setitem__(self, row_lookup: pandas.Index, col_lookup: pandas.Index, item):
        """
        Args:
            row_lookup: the global row index to write item to
            col_lookup: the global col index to write item to
            item: The new item needs to be set. It can be any shape that's
                broadcast-able to the product of the lookup tables.
        """
        to_shape = (len(row_lookup), len(col_lookup))
        item = self._broadcast_item(row_lookup, col_lookup, item, to_shape)
        self._write_items(row_lookup, col_lookup, item)

    def _broadcast_item(self, row_lookup, col_lookup, item, to_shape):
        """Use numpy to broadcast or reshape item.

        Notes:
            - Numpy is memory efficient, there shouldn't be performance issue.
        """
        # It is valid to pass a DataFrame or Series to __setitem__ that is larger than
        # the target the user is trying to overwrite. This
        if isinstance(item, (pandas.Series, pandas.DataFrame, DataFrame)):
            if not all(idx in item.index for idx in row_lookup):
                raise ValueError(
                    "Must have equal len keys and value when setting with "
                    "an iterable"
                )
            if hasattr(item, "columns"):
                if not all(idx in item.columns for idx in col_lookup):
                    raise ValueError(
                        "Must have equal len keys and value when setting "
                        "with an iterable"
                    )
                item = item.reindex(index=row_lookup, columns=col_lookup)
            else:
                item = item.reindex(index=row_lookup)
        try:
            item = np.array(item)
            if np.prod(to_shape) == np.prod(item.shape):
                return item.reshape(to_shape)
            else:
                return np.broadcast_to(item, to_shape)
        except ValueError:
            from_shape = np.array(item).shape
            raise ValueError(
                "could not broadcast input array from shape {from_shape} into shape "
                "{to_shape}".format(from_shape=from_shape, to_shape=to_shape)
            )

    def _write_items(self, row_lookup, col_lookup, item):
        """Perform remote write and replace blocks.
        """
        row_numeric_idx = self.qc.global_idx_to_numeric_idx("row", row_lookup)
        col_numeric_idx = self.qc.global_idx_to_numeric_idx("col", col_lookup)
        self.qc.write_items(row_numeric_idx, col_numeric_idx, item)


class _LocIndexer(_LocationIndexerBase):
    """A indexer for ray_df.loc[] functionality"""

    def __getitem__(self, key):
        row_loc, col_loc, ndim, self.row_scaler, self.col_scaler = _parse_tuple(key)
        self._handle_enlargement(row_loc, col_loc)
        row_lookup, col_lookup = self._compute_lookup(row_loc, col_loc)
        ndim = self._expand_dim(row_lookup, col_lookup, ndim)
        result = super(_LocIndexer, self).__getitem__(row_lookup, col_lookup, ndim)
        return result

    def __setitem__(self, key, item):
        row_loc, col_loc, _, __, ___ = _parse_tuple(key)
        if isinstance(row_loc, list) and len(row_loc) == 1:
            if row_loc[0] not in self.qc.index:
                index = self.qc.index.insert(len(self.qc.index), row_loc[0])
                self.qc = self.qc.reindex(labels=index, axis=0)
                self.df._update_inplace(new_query_compiler=self.qc)

        if (
            isinstance(col_loc, list)
            and len(col_loc) == 1
            and col_loc[0] not in self.qc.columns
        ):
            new_col = pandas.Series(index=self.df.index)
            new_col[row_loc] = item
            self.df.insert(loc=len(self.df.columns), column=col_loc[0], value=new_col)
            self.qc = self.df._query_compiler
        else:
            row_lookup, col_lookup = self._compute_lookup(row_loc, col_loc)
            super(_LocIndexer, self).__setitem__(row_lookup, col_lookup, item)

    def _handle_enlargement(self, row_loc, col_loc):
        """Handle Enlargement (if there is one).

        Returns:
            None
        """
        if _is_enlargement(row_loc, self.qc.index) or _is_enlargement(
            col_loc, self.qc.columns
        ):
            _warn_enlargement()
            self.qc.enlarge_partitions(
                new_row_labels=self._compute_enlarge_labels(row_loc, self.qc.index),
                new_col_labels=self._compute_enlarge_labels(col_loc, self.qc.columns),
            )

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
                "None of [{labels}] are in the [{base_index_name}]".format(
                    labels=list(locator_as_index), base_index_name=base_index
                )
            )
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

    def _compute_lookup(self, row_loc, col_loc) -> Tuple[pandas.Index, pandas.Index]:
        if isinstance(row_loc, list) and len(row_loc) == 1:
            if (
                isinstance(self.qc.index.values[0], np.datetime64)
                and type(row_loc[0]) != np.datetime64
            ):
                row_loc = [pandas.to_datetime(row_loc[0])]
        row_lookup = self.qc.index.to_series().loc[row_loc].index
        col_lookup = self.qc.columns.to_series().loc[col_loc].index
        return row_lookup, col_lookup


class _iLocIndexer(_LocationIndexerBase):
    """A indexer for ray_df.iloc[] functionality"""

    def __getitem__(self, key):
        row_loc, col_loc, ndim, self.row_scaler, self.col_scaler = _parse_tuple(key)
        self._check_dtypes(row_loc)
        self._check_dtypes(col_loc)

        row_lookup, col_lookup = self._compute_lookup(row_loc, col_loc)
        result = super(_iLocIndexer, self).__getitem__(row_lookup, col_lookup, ndim)
        return result

    def __setitem__(self, key, item):
        row_loc, col_loc, _, __, ___ = _parse_tuple(key)
        self._check_dtypes(row_loc)
        self._check_dtypes(col_loc)

        row_lookup, col_lookup = self._compute_lookup(row_loc, col_loc)
        super(_iLocIndexer, self).__setitem__(row_lookup, col_lookup, item)

    def _compute_lookup(self, row_loc, col_loc) -> Tuple[pandas.Index, pandas.Index]:
        row_lookup = self.qc.index.to_series().iloc[row_loc].index
        col_lookup = self.qc.columns.to_series().iloc[col_loc].index
        return row_lookup, col_lookup

    def _check_dtypes(self, locator):
        is_int = is_integer(locator)
        is_int_slice = is_integer_slice(locator)
        is_int_list = is_list_like(locator) and all(map(is_integer, locator))
        is_bool_arr = is_boolean_array(locator)

        if not any([is_int, is_int_slice, is_int_list, is_bool_arr]):
            raise ValueError(_ILOC_INT_ONLY_ERROR)
