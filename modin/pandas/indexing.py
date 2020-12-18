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

"""
Details about how Indexing Helper Class works.

_LocationIndexerBase provide methods framework for __getitem__
  and __setitem__ that work with Modin DataFrame's internal index. Base
  class's __{get,set}item__ takes in partitions & idx_in_partition data
  and perform lookup/item write.

_LocIndexer and _iLocIndexer is responsible for indexer specific logic and
  lookup computation. Loc will take care of enlarge DataFrame. Both indexer
  will take care of translating pandas's lookup to Modin DataFrame's internal
  lookup.

An illustration is available at
https://github.com/ray-project/ray/pull/1955#issuecomment-386781826
"""

import numpy as np
import pandas
from pandas.api.types import is_list_like, is_bool
from pandas.core.dtypes.common import is_integer
from pandas.core.indexing import IndexingError

from .dataframe import DataFrame
from .series import Series
from .utils import is_scalar


def is_slice(x):
    """
    Implement [METHOD_NAME].

    TODO: Add more details for this docstring template.

    Parameters
    ----------
    What arguments does this function have.
    [
    PARAMETER_NAME: PARAMETERS TYPES
        Description.
    ]

    Returns
    -------
    What this returns (if anything)
    """
    return isinstance(x, slice)


def compute_sliced_len(slc, sequence_len):
    """
    Compute length of sliced object.

    Parameters
    ----------
    slc: slice
        Slice object
    sequence_len: int
        Length of sequence, to which slice will be applied

    Returns
    -------
    int
        Length of object after applying slice object on it.
    """
    # This will translate slice to a range, from which we can retrieve length
    return len(range(*slc.indices(sequence_len)))


def is_2d(x):
    """
    Implement [METHOD_NAME].

    TODO: Add more details for this docstring template.

    Parameters
    ----------
    What arguments does this function have.
    [
    PARAMETER_NAME: PARAMETERS TYPES
        Description.
    ]

    Returns
    -------
    What this returns (if anything)
    """
    return is_list_like(x) or is_slice(x)


def is_tuple(x):
    """
    Implement [METHOD_NAME].

    TODO: Add more details for this docstring template.

    Parameters
    ----------
    What arguments does this function have.
    [
    PARAMETER_NAME: PARAMETERS TYPES
        Description.
    ]

    Returns
    -------
    What this returns (if anything)
    """
    return isinstance(x, tuple)


def is_boolean_array(x):
    """
    Implement [METHOD_NAME].

    TODO: Add more details for this docstring template.

    Parameters
    ----------
    What arguments does this function have.
    [
    PARAMETER_NAME: PARAMETERS TYPES
        Description.
    ]

    Returns
    -------
    What this returns (if anything)
    """
    return is_list_like(x) and all(map(is_bool, x))


def is_integer_slice(x):
    """
    Implement [METHOD_NAME].

    TODO: Add more details for this docstring template.

    Parameters
    ----------
    What arguments does this function have.
    [
    PARAMETER_NAME: PARAMETERS TYPES
        Description.
    ]

    Returns
    -------
    What this returns (if anything)
    """
    if not is_slice(x):
        return False
    for pos in [x.start, x.stop, x.step]:
        if not ((pos is None) or is_integer(pos)):
            return False  # one position is neither None nor int
    return True


_ILOC_INT_ONLY_ERROR = """
Location based indexing can only have [integer, integer slice (START point is
INCLUDED, END point is EXCLUDED), listlike of integers, boolean array] types.
"""

_VIEW_IS_COPY_WARNING = """
Modin is making a copy of of the DataFrame. This behavior diverges from Pandas.
This will be fixed in future releases.
"""


def _parse_tuple(tup):
    """
    Unpack the user input for getitem and setitem and compute ndim.

    TODO: Add more details for this docstring template.

    loc[a] -> ([a], :), 1D
    loc[[a,b],] -> ([a,b], :),
    loc[a,b] -> ([a], [b]), 0D

    Parameters
    ----------
    tup: tuple
        [Descsription]

    Returns
    -------
    What this returns (if anything)
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


def _compute_ndim(row_loc, col_loc):
    """
    Compute the ndim of result from locators.

    TODO: Add more details for this docstring template.

    Parameters
    ----------
    What arguments does this function have.
    [
    PARAMETER_NAME: PARAMETERS TYPES
        Description.
    ]

    Returns
    -------
    What this returns (if anything)
    """
    row_scaler = is_scalar(row_loc) or is_tuple(row_loc)
    col_scaler = is_scalar(col_loc) or is_tuple(col_loc)

    if row_scaler and col_scaler:
        ndim = 0
    elif row_scaler ^ col_scaler:
        ndim = 1
    else:
        ndim = 2

    return ndim


class _LocationIndexerBase(object):
    """Base class for location indexer like loc and iloc."""

    def __init__(self, modin_df):
        """
        Implement [METHOD_NAME].

        TODO: Add more details for this docstring template.

        Parameters
        ----------
        What arguments does this function have.
        [
        PARAMETER_NAME: PARAMETERS TYPES
            Description.
        ]

        Returns
        -------
        What this returns (if anything)
        """
        self.df = modin_df
        self.qc = modin_df._query_compiler
        self.row_scaler = False
        self.col_scaler = False

    def __getitem__(self, row_lookup, col_lookup, ndim):
        """
        Implement [METHOD_NAME].

        TODO: Add more details for this docstring template.

        Parameters
        ----------
        What arguments does this function have.
        [
        PARAMETER_NAME: PARAMETERS TYPES
            Description.
        ]

        Returns
        -------
        What this returns (if anything)
        """
        qc_view = self.qc.view(row_lookup, col_lookup)
        if ndim == 2:
            return self.df.__constructor__(query_compiler=qc_view)
        if isinstance(self.df, Series) and not self.row_scaler:
            return self.df.__constructor__(query_compiler=qc_view)
        if isinstance(self.df, Series):
            axis = 0
        elif ndim == 0:
            axis = None
        else:
            axis = (
                None
                if self.col_scaler and self.row_scaler
                else 1
                if self.col_scaler
                else 0
            )
        return self.df.__constructor__(query_compiler=qc_view).squeeze(axis=axis)

    def __setitem__(self, row_lookup, col_lookup, item, axis=None):
        """
        Implement [METHOD_NAME].

        TODO: Add more details for this docstring template.

        Parameters
        ----------
        row_lookup:
            the global row index to write item to
        col_lookup:
            the global col index to write item to
        item:
            The new item needs to be set. It can be any shape that's
            broadcast-able to the product of the lookup tables.
        """
        # Convert slices to indices for the purposes of application.
        # TODO (devin-petersohn): Apply to slice without conversion to list
        if isinstance(row_lookup, slice):
            row_lookup = range(len(self.qc.index))[row_lookup]
        if isinstance(col_lookup, slice):
            col_lookup = range(len(self.qc.columns))[col_lookup]
        # This is True when we dealing with assignment of a full column. This case
        # should be handled in a fastpath with `df[col] = item`.
        if axis == 0:
            self.df[self.df.columns[col_lookup][0]] = item
        # This is True when we are assigning to a full row. We want to reuse the setitem
        # mechanism to operate along only one axis for performance reasons.
        elif axis == 1:
            if hasattr(item, "_query_compiler"):
                item = item._query_compiler
            new_qc = self.qc.setitem(1, self.qc.index[row_lookup[0]], item)
            self.df._create_or_update_from_compiler(new_qc, inplace=True)
        # Assignment to both axes.
        else:
            if isinstance(row_lookup, slice):
                new_row_len = len(self.df.index[row_lookup])
            else:
                new_row_len = len(row_lookup)
            if isinstance(col_lookup, slice):
                new_col_len = len(self.df.columns[col_lookup])
            else:
                new_col_len = len(col_lookup)
            to_shape = new_row_len, new_col_len
            item = self._broadcast_item(row_lookup, col_lookup, item, to_shape)
            self._write_items(row_lookup, col_lookup, item)

    def _broadcast_item(self, row_lookup, col_lookup, item, to_shape):
        """
        Use numpy to broadcast or reshape item.

        TODO: Add more details for this docstring template.

        Parameters
        ----------
        What arguments does this function have.
        [
        PARAMETER_NAME: PARAMETERS TYPES
            Description.
        ]

        Returns
        -------
        What this returns (if anything)

        Notes
        -----
        Numpy is memory efficient, there shouldn't be performance issue.
        """
        # It is valid to pass a DataFrame or Series to __setitem__ that is larger than
        # the target the user is trying to overwrite. This
        if isinstance(item, (pandas.Series, pandas.DataFrame, Series, DataFrame)):
            # convert indices in lookups to names, as Pandas reindex expects them to be so
            index_values = self.qc.index[row_lookup]
            if not all(idx in item.index for idx in index_values):
                raise ValueError(
                    "Must have equal len keys and value when setting with "
                    "an iterable"
                )
            if hasattr(item, "columns"):
                column_values = self.qc.columns[col_lookup]
                if not all(col in item.columns for col in column_values):
                    # TODO: think if it is needed to handle cases when columns have duplicate names
                    raise ValueError(
                        "Must have equal len keys and value when setting "
                        "with an iterable"
                    )
                item = item.reindex(index=index_values, columns=column_values)
            else:
                item = item.reindex(index=index_values)
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
        """
        Perform remote write and replace blocks.

        TODO: Add more details for this docstring template.

        Parameters
        ----------
        What arguments does this function have.
        [
        PARAMETER_NAME: PARAMETERS TYPES
            Description.
        ]

        Returns
        -------
        What this returns (if anything)
        """
        new_qc = self.qc.write_items(row_lookup, col_lookup, item)
        self.df._create_or_update_from_compiler(new_qc, inplace=True)

    def _determine_setitem_axis(self, row_lookup, col_lookup, row_scaler, col_scaler):
        """
        Determine an axis along which we should do an assignment.

        Parameters
        ----------
        row_lookup: slice or list
            Indexer for rows
        col_lookup: slice or list
            Indexer for columns
        row_scaler: bool
            Whether indexer for rows was slacar or not
        col_scaler: bool
            Whether indexer for columns was slacer or not

        Returns
        -------
        int or None
            None if this will be a both axis assignment, number of axis to assign in other cases.

        Notes
        -----
            axis = 0: column assignment df[col] = item
            axis = 1: row assignment df.loc[row] = item
            axis = None: assignment along both axes
        """
        if self.df.shape == (1, 1):
            return None if not (row_scaler ^ col_scaler) else 1 if row_scaler else 0

        def get_axis(axis):
            return self.qc.index if axis == 0 else self.qc.columns

        row_lookup_len, col_lookup_len = [
            len(lookup)
            if not isinstance(lookup, slice)
            else compute_sliced_len(lookup, len(get_axis(i)))
            for i, lookup in enumerate([row_lookup, col_lookup])
        ]

        if (
            row_lookup_len == len(self.qc.index)
            and col_lookup_len == 1
            and isinstance(self.df, DataFrame)
        ):
            axis = 0
        elif col_lookup_len == len(self.qc.columns) and row_lookup_len == 1:
            axis = 1
        else:
            axis = None
        return axis


class _LocIndexer(_LocationIndexerBase):
    """An indexer for modin_df.loc[] functionality."""

    def __getitem__(self, key):
        """
        Implement [METHOD_NAME].

        TODO: Add more details for this docstring template.

        Parameters
        ----------
        What arguments does this function have.
        [
        PARAMETER_NAME: PARAMETERS TYPES
            Description.
        ]

        Returns
        -------
        What this returns (if anything)
        """
        if callable(key):
            return self.__getitem__(key(self.df))
        row_loc, col_loc, ndim, self.row_scaler, self.col_scaler = _parse_tuple(key)
        if isinstance(row_loc, slice) and row_loc == slice(None):
            # If we're only slicing columns, handle the case with `__getitem__`
            if not isinstance(col_loc, slice):
                # Boolean indexers can just be sliced into the columns object and
                # then passed to `__getitem__`
                if is_boolean_array(col_loc):
                    col_loc = self.df.columns[col_loc]
                return self.df.__getitem__(col_loc)
            else:
                result_slice = self.df.columns.slice_locs(col_loc.start, col_loc.stop)
                return self.df.iloc[:, slice(*result_slice)]

        row_lookup, col_lookup = self._compute_lookup(row_loc, col_loc)
        if any(i == -1 for i in row_lookup) or any(i == -1 for i in col_lookup):
            raise KeyError(
                "Passing list-likes to .loc or [] with any missing labels is no longer "
                "supported, see https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#deprecate-loc-reindex-listlike"
            )
        result = super(_LocIndexer, self).__getitem__(row_lookup, col_lookup, ndim)
        if isinstance(result, Series):
            result._parent = self.df
            result._parent_axis = 0
        # Pandas drops the levels that are in the `loc`, so we have to as well.
        if hasattr(result, "index") and isinstance(result.index, pandas.MultiIndex):
            if (
                isinstance(result, Series)
                and not isinstance(col_loc, slice)
                and all(
                    col_loc[i] in result.index.levels[i] for i in range(len(col_loc))
                )
            ):
                result.index = result.index.droplevel(list(range(len(col_loc))))
            elif all(
                not isinstance(row_loc[i], slice)
                and row_loc[i] in result.index.levels[i]
                for i in range(len(row_loc))
            ):
                result.index = result.index.droplevel(list(range(len(row_loc))))
        if (
            hasattr(result, "columns")
            and not isinstance(col_loc, slice)
            and isinstance(result.columns, pandas.MultiIndex)
            and all(col_loc[i] in result.columns.levels[i] for i in range(len(col_loc)))
        ):
            result.columns = result.columns.droplevel(list(range(len(col_loc))))
        # This is done for cases where the index passed in has other state, like a
        # frequency in the case of DateTimeIndex.
        if (
            row_lookup is not None
            and isinstance(col_loc, slice)
            and col_loc == slice(None)
            and isinstance(key, pandas.Index)
        ):
            result.index = key
        return result

    def __setitem__(self, key, item):
        """
        Implement [METHOD_NAME].

        TODO: Add more details for this docstring template.

        Parameters
        ----------
        What arguments does this function have.
        [
        PARAMETER_NAME: PARAMETERS TYPES
            Description.
        ]

        Returns
        -------
        What this returns (if anything)
        """
        row_loc, col_loc, _, row_scaler, col_scaler = _parse_tuple(key)
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
            super(_LocIndexer, self).__setitem__(
                row_lookup,
                col_lookup,
                item,
                axis=self._determine_setitem_axis(
                    row_lookup, col_lookup, row_scaler, col_scaler
                ),
            )

    def _compute_enlarge_labels(self, locator, base_index):
        """
        Help to _enlarge_axis, compute common labels and extra labels.

        TODO: add types.

        Returns
        -------
        nan_labels:
            The labels needs to be added
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

    def _compute_lookup(self, row_loc, col_loc):
        """
        Implement [METHOD_NAME].

        TODO: Add more details for this docstring template.

        Parameters
        ----------
        What arguments does this function have.
        [
        PARAMETER_NAME: PARAMETERS TYPES
            Description.
        ]

        Returns
        -------
        What this returns (if anything)
        """
        if is_list_like(row_loc) and len(row_loc) == 1:
            if (
                isinstance(self.qc.index.values[0], np.datetime64)
                and type(row_loc[0]) != np.datetime64
            ):
                row_loc = [pandas.to_datetime(row_loc[0])]

        if isinstance(row_loc, slice):
            row_lookup = self.qc.index.get_indexer_for(
                self.qc.index.to_series().loc[row_loc]
            )
        elif self.qc.has_multiindex():
            if isinstance(row_loc, pandas.MultiIndex):
                row_lookup = self.qc.index.get_indexer_for(row_loc)
            else:
                row_lookup = self.qc.index.get_locs(row_loc)
        elif is_boolean_array(row_loc):
            # If passed in a list of booleans, we return the index of the true values
            row_lookup = [i for i, row_val in enumerate(row_loc) if row_val]
        else:
            row_lookup = self.qc.index.get_indexer_for(row_loc)
        if isinstance(col_loc, slice):
            col_lookup = self.qc.columns.get_indexer_for(
                self.qc.columns.to_series().loc[col_loc]
            )
        elif self.qc.has_multiindex(axis=1):
            if isinstance(col_loc, pandas.MultiIndex):
                col_lookup = self.qc.columns.get_indexer_for(col_loc)
            else:
                col_lookup = self.qc.columns.get_locs(col_loc)
        elif is_boolean_array(col_loc):
            # If passed in a list of booleans, we return the index of the true values
            col_lookup = [i for i, col_val in enumerate(col_loc) if col_val]
        else:
            col_lookup = self.qc.columns.get_indexer_for(col_loc)
        return row_lookup, col_lookup


class _iLocIndexer(_LocationIndexerBase):
    """An indexer for modin_df.iloc[] functionality."""

    def __getitem__(self, key):
        """
        Implement [METHOD_NAME].

        TODO: Add more details for this docstring template.

        Parameters
        ----------
        What arguments does this function have.
        [
        PARAMETER_NAME: PARAMETERS TYPES
            Description.
        ]

        Returns
        -------
        What this returns (if anything)
        """
        if callable(key):
            return self.__getitem__(key(self.df))
        row_loc, col_loc, ndim, self.row_scaler, self.col_scaler = _parse_tuple(key)
        self._check_dtypes(row_loc)
        self._check_dtypes(col_loc)

        row_lookup, col_lookup = self._compute_lookup(row_loc, col_loc)
        result = super(_iLocIndexer, self).__getitem__(row_lookup, col_lookup, ndim)
        if isinstance(result, Series):
            result._parent = self.df
            result._parent_axis = 0
        return result

    def __setitem__(self, key, item):
        """
        Implement [METHOD_NAME].

        TODO: Add more details for this docstring template.

        Parameters
        ----------
        What arguments does this function have.
        [
        PARAMETER_NAME: PARAMETERS TYPES
            Description.
        ]

        Returns
        -------
        What this returns (if anything)
        """
        row_loc, col_loc, _, row_scaler, col_scaler = _parse_tuple(key)
        self._check_dtypes(row_loc)
        self._check_dtypes(col_loc)

        row_lookup, col_lookup = self._compute_lookup(row_loc, col_loc)
        super(_iLocIndexer, self).__setitem__(
            row_lookup,
            col_lookup,
            item,
            axis=self._determine_setitem_axis(
                row_lookup, col_lookup, row_scaler, col_scaler
            ),
        )

    def _compute_lookup(self, row_loc, col_loc):
        """
        Implement [METHOD_NAME].

        TODO: Add more details for this docstring template.

        Parameters
        ----------
        What arguments does this function have.
        [
        PARAMETER_NAME: PARAMETERS TYPES
            Description.
        ]

        Returns
        -------
        What this returns (if anything)
        """
        if (
            not isinstance(row_loc, slice)
            or isinstance(row_loc, slice)
            and row_loc.step is not None
        ):
            row_lookup = (
                pandas.RangeIndex(len(self.qc.index)).to_series().iloc[row_loc].index
            )
        else:
            row_lookup = row_loc
        if (
            not isinstance(col_loc, slice)
            or isinstance(col_loc, slice)
            and col_loc.step is not None
        ):
            col_lookup = (
                pandas.RangeIndex(len(self.qc.columns)).to_series().iloc[col_loc].index
            )
        else:
            col_lookup = col_loc
        return row_lookup, col_lookup

    def _check_dtypes(self, locator):
        """
        Implement [METHOD_NAME].

        TODO: Add more details for this docstring template.

        Parameters
        ----------
        What arguments does this function have.
        [
        PARAMETER_NAME: PARAMETERS TYPES
            Description.
        ]

        Returns
        -------
        What this returns (if anything)
        """
        is_int = is_integer(locator)
        is_int_slice = is_integer_slice(locator)
        is_int_list = is_list_like(locator) and all(map(is_integer, locator))
        is_bool_arr = is_boolean_array(locator)

        if not any([is_int, is_int_slice, is_int_list, is_bool_arr]):
            raise ValueError(_ILOC_INT_ONLY_ERROR)
