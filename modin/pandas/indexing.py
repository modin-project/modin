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

# noqa: MD02
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
import itertools
from pandas.api.types import is_list_like, is_bool
from pandas.core.dtypes.common import is_integer, is_bool_dtype, is_integer_dtype
from pandas.core.indexing import IndexingError
from modin.error_message import ErrorMessage
from modin.logging import LoggerMetaClass, metaclass_resolver

from .dataframe import DataFrame
from .series import Series
from .utils import is_scalar


def is_slice(x):
    """
    Check that argument is an instance of slice.

    Parameters
    ----------
    x : object
        Object to check.

    Returns
    -------
    bool
        True if argument is a slice, False otherwise.
    """
    return isinstance(x, slice)


def compute_sliced_len(slc, sequence_len):
    """
    Compute length of sliced object.

    Parameters
    ----------
    slc : slice
        Slice object.
    sequence_len : int
        Length of sequence, to which slice will be applied.

    Returns
    -------
    int
        Length of object after applying slice object on it.
    """
    # This will translate slice to a range, from which we can retrieve length
    return len(range(*slc.indices(sequence_len)))


def is_2d(x):
    """
    Check that argument is a list or a slice.

    Parameters
    ----------
    x : object
        Object to check.

    Returns
    -------
    bool
        `True` if argument is a list or slice, `False` otherwise.
    """
    return is_list_like(x) or is_slice(x)


def is_tuple(x):
    """
    Check that argument is a tuple.

    Parameters
    ----------
    x : object
        Object to check.

    Returns
    -------
    bool
        True if argument is a tuple, False otherwise.
    """
    return isinstance(x, tuple)


def is_boolean_array(x):
    """
    Check that argument is an array of bool.

    Parameters
    ----------
    x : object
        Object to check.

    Returns
    -------
    bool
        True if argument is an array of bool, False otherwise.
    """
    if isinstance(x, (np.ndarray, Series, pandas.Series, pandas.Index)):
        return is_bool_dtype(x.dtype)
    elif isinstance(x, (DataFrame, pandas.DataFrame)):
        return all(map(is_bool_dtype, x.dtypes))
    return is_list_like(x) and all(map(is_bool, x))


def is_integer_array(x):
    """
    Check that argument is an array of integers.

    Parameters
    ----------
    x : object
        Object to check.

    Returns
    -------
    bool
        True if argument is an array of integers, False otherwise.
    """
    if isinstance(x, (np.ndarray, Series, pandas.Series, pandas.Index)):
        return is_integer_dtype(x.dtype)
    elif isinstance(x, (DataFrame, pandas.DataFrame)):
        return all(map(is_integer_dtype, x.dtypes))
    return is_list_like(x) and all(map(is_integer, x))


def is_integer_slice(x):
    """
    Check that argument is an array of int.

    Parameters
    ----------
    x : object
        Object to check.

    Returns
    -------
    bool
        True if argument is an array of int, False otherwise.
    """
    if not is_slice(x):
        return False
    for pos in [x.start, x.stop, x.step]:
        if not ((pos is None) or is_integer(pos)):
            return False  # one position is neither None nor int
    return True


def is_range_like(obj):
    """
    Check if the object is range-like.

    Objects that are considered range-like have information about the range (start and
    stop positions, and step) and also have to be iterable. Examples of range-like
    objects are: Python range, pandas.RangeIndex.

    Parameters
    ----------
    obj : object

    Returns
    -------
    bool
    """
    return (
        hasattr(obj, "__iter__")
        and hasattr(obj, "start")
        and hasattr(obj, "stop")
        and hasattr(obj, "step")
    )


def boolean_mask_to_numeric(indexer):
    """
    Convert boolean mask to numeric indices.

    Parameters
    ----------
    indexer : list-like of booleans

    Returns
    -------
    np.ndarray of ints
        Numerical positions of ``True`` elements in the passed `indexer`.
    """
    if isinstance(indexer, (np.ndarray, Series, pandas.Series)):
        return np.where(indexer)[0]
    else:
        # It's faster to build the resulting numpy array from the reduced amount of data via
        # `compress` iterator than convert non-numpy-like `indexer` to numpy and apply `np.where`.
        return np.fromiter(
            # `itertools.compress` masks `data` with the `selectors` mask,
            # works about ~10% faster than a pure list comprehension
            itertools.compress(data=range(len(indexer)), selectors=indexer),
            dtype=np.int64,
        )


_ILOC_INT_ONLY_ERROR = """
Location based indexing can only have [integer, integer slice (START point is
INCLUDED, END point is EXCLUDED), listlike of integers, boolean array] types.
"""

_VIEW_IS_COPY_WARNING = """
Modin is making a copy of of the DataFrame. This behavior diverges from Pandas.
This will be fixed in future releases.
"""


def _compute_ndim(row_loc, col_loc):
    """
    Compute the number of dimensions of result from locators.

    Parameters
    ----------
    row_loc : list or scalar
        Row locator.
    col_loc : list or scalar
        Column locator.

    Returns
    -------
    {0, 1, 2}
        Number of dimensions in located dataset.
    """
    row_scalar = is_scalar(row_loc) or is_tuple(row_loc)
    col_scalar = is_scalar(col_loc) or is_tuple(col_loc)

    if row_scalar and col_scalar:
        ndim = 0
    elif row_scalar ^ col_scalar:
        ndim = 1
    else:
        ndim = 2

    return ndim


class _LocationIndexerBase(object, metaclass=LoggerMetaClass):
    """
    Base class for location indexer like loc and iloc.

    Parameters
    ----------
    modin_df : modin.pandas.DataFrame
        DataFrame to operate on.
    """

    def __init__(self, modin_df):
        self.df = modin_df
        self.qc = modin_df._query_compiler
        self.row_scalar = False
        self.col_scalar = False

    def __getitem__(self, row_lookup, col_lookup, ndim):
        """
        Retrieve dataset according to `row_lookup` and `col_lookup`.

        Parameters
        ----------
        row_lookup : slice(None), range or np.ndarray
            The global row index to retrieve data from.
        col_lookup : slice(None), range or np.ndarray
            The global col index to retrieve data from.
        ndim : {0, 1, 2}
            Number of dimensions in dataset to be retrieved.

        Returns
        -------
        modin.pandas.DataFrame or modin.pandas.Series
            Located dataset.

        Notes
        -----
        Usage of `slice(None)` as a lookup is a hack to pass information about
        full-axis grab without computing actual indices that triggers lazy computations.
        Ideally, this API should get rid of using slices as indexers and either use a
        common ``Indexer`` object or range and ``np.ndarray`` only.
        """
        if isinstance(row_lookup, slice):
            ErrorMessage.catch_bugs_and_request_email(
                failure_condition=row_lookup != slice(None),
                extra_log=f"Only None-slices are acceptable as a slice argument in masking, got: {row_lookup}",
            )
            row_lookup = None
        if isinstance(col_lookup, slice):
            ErrorMessage.catch_bugs_and_request_email(
                failure_condition=col_lookup != slice(None),
                extra_log=f"Only None-slices are acceptable as a slice argument in masking, got: {col_lookup}",
            )
            col_lookup = None
        qc_view = self.qc.view(row_lookup, col_lookup)
        if ndim == 2:
            return self.df.__constructor__(query_compiler=qc_view)
        if isinstance(self.df, Series) and not self.row_scalar:
            return self.df.__constructor__(query_compiler=qc_view)
        if isinstance(self.df, Series):
            axis = 0
        elif ndim == 0:
            axis = None
        else:
            axis = (
                None
                if self.col_scalar and self.row_scalar
                else 1
                if self.col_scalar
                else 0
            )
        return self.df.__constructor__(query_compiler=qc_view).squeeze(axis=axis)

    def __setitem__(self, row_lookup, col_lookup, item, axis=None):
        """
        Assign `item` value to located dataset.

        Parameters
        ----------
        row_lookup : slice or scalar
            The global row index to write item to.
        col_lookup : slice or scalar
            The global col index to write item to.
        item : DataFrame, Series or scalar
            The new item needs to be set. It can be any shape that's
            broadcast-able to the product of the lookup tables.
        axis : {None, 0, 1}, default: None
            If not None, it means that whole axis is used to assign a value.
            0 means assign to whole column, 1 means assign to whole row.
            If None, it means that partial assignment is done on both axes.
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
            if not is_scalar(item):
                item = self._broadcast_item(row_lookup, col_lookup, item, to_shape)
            self._write_items(row_lookup, col_lookup, item)

    def _broadcast_item(self, row_lookup, col_lookup, item, to_shape):
        """
        Use NumPy to broadcast or reshape item.

        Parameters
        ----------
        row_lookup : slice or scalar
            The global row index to locate inside of `item`.
        col_lookup : slice or scalar
            The global col index to locate inside of `item`.
        item : DataFrame, Series, or query_compiler
            Value that should be broadcast to a new shape of `to_shape`.
        to_shape : tuple of two int
            Shape of dataset that `item` should be broadcasted to.

        Returns
        -------
        numpy.ndarray
            `item` after it was broadcasted to `to_shape`.

        Raises
        ------
        ValueError
            If `row_lookup` or `col_lookup` contain values missing in
            `self` index or columns correspondingly.
            If `item` cannot be broadcast from its own shape to `to_shape`.

        Notes
        -----
        NumPy is memory efficient, there shouldn't be performance issue.
        """
        # It is valid to pass a DataFrame or Series to __setitem__ that is larger than
        # the target the user is trying to overwrite. This
        if isinstance(item, (pandas.Series, pandas.DataFrame, Series, DataFrame)):
            # convert indices in lookups to names, as Pandas reindex expects them to be so
            axes_to_reindex = {}
            index_values = self.qc.index[row_lookup]
            if not index_values.equals(item.index):
                axes_to_reindex["index"] = index_values
            if hasattr(item, "columns"):
                column_values = self.qc.columns[col_lookup]
                if not column_values.equals(item.columns):
                    axes_to_reindex["columns"] = column_values
            # New value for columns/index make that reindex add NaN values
            if axes_to_reindex:
                item = item.reindex(**axes_to_reindex)
        try:
            item = np.array(item)
            if np.prod(to_shape) == np.prod(item.shape):
                return item.reshape(to_shape)
            else:
                return np.broadcast_to(item, to_shape)
        except ValueError:
            from_shape = np.array(item).shape
            raise ValueError(
                f"could not broadcast input array from shape {from_shape} into shape "
                + f"{to_shape}"
            )

    def _write_items(self, row_lookup, col_lookup, item):
        """
        Perform remote write and replace blocks.

        Parameters
        ----------
        row_lookup : slice or scalar
            The global row index to write item to.
        col_lookup : slice or scalar
            The global col index to write item to.
        item : numpy.ndarray
            The new item value that needs to be assigned to `self`.
        """
        new_qc = self.qc.write_items(row_lookup, col_lookup, item)
        self.df._create_or_update_from_compiler(new_qc, inplace=True)

    def _determine_setitem_axis(self, row_lookup, col_lookup, row_scalar, col_scalar):
        """
        Determine an axis along which we should do an assignment.

        Parameters
        ----------
        row_lookup : slice or list
            Indexer for rows.
        col_lookup : slice or list
            Indexer for columns.
        row_scalar : bool
            Whether indexer for rows is scalar or not.
        col_scalar : bool
            Whether indexer for columns is scalar or not.

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
            return None if not (row_scalar ^ col_scalar) else 1 if row_scalar else 0

        def get_axis(axis):
            return self.qc.index if axis == 0 else self.qc.columns

        row_lookup_len, col_lookup_len = [
            len(lookup)
            if not isinstance(lookup, slice)
            else compute_sliced_len(lookup, len(get_axis(i)))
            for i, lookup in enumerate([row_lookup, col_lookup])
        ]

        if col_lookup_len == 1 and row_lookup_len == 1:
            axis = None
        elif (
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

    def _parse_row_and_column_locators(self, tup):
        """
        Unpack the user input for getitem and setitem and compute ndim.

        loc[a] -> ([a], :), 1D
        loc[[a,b]] -> ([a,b], :),
        loc[a,b] -> ([a], [b]), 0D

        Parameters
        ----------
        tup : tuple
            User input to unpack.

        Returns
        -------
        row_loc : scalar or list
            Row locator(s) as a scalar or List.
        col_list : scalar or list
            Column locator(s) as a scalar or List.
        ndim : {0, 1, 2}
            Number of dimensions of located dataset.
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

        row_loc = row_loc(self.df) if callable(row_loc) else row_loc
        col_loc = col_loc(self.df) if callable(col_loc) else col_loc
        return row_loc, col_loc, _compute_ndim(row_loc, col_loc)

    # HACK: This method bypasses regular ``loc/iloc.__getitem__`` flow in order to ensure better
    # performance in the case of boolean masking. The only purpose of this method is to compensate
    # for a lack of backend's indexing API, there is no Query Compiler method allowing masking
    # along both axis when any of the indexers is a boolean. That's why rows and columns masking
    # phases are separate in this case.
    # TODO: Remove this method and handle this case naturally via ``loc/iloc.__getitem__`` flow
    # when QC API would support both-axis masking with boolean indexers.
    def _handle_boolean_masking(self, row_loc, col_loc):
        """
        Retrieve dataset according to the boolean mask for rows and an indexer for columns.

        In comparison with the regular ``loc/iloc.__getitem__`` flow this method efficiently
        masks rows with a Modin Series boolean mask without materializing it (if the selected
        execution implements such masking).

        Parameters
        ----------
        row_loc : modin.pandas.Series of bool dtype
            Boolean mask to index rows with.
        col_loc : object
            An indexer along column axis.

        Returns
        -------
        modin.pandas.DataFrame or modin.pandas.Series
            Located dataset.
        """
        ErrorMessage.catch_bugs_and_request_email(
            failure_condition=not isinstance(row_loc, Series),
            extra_log=f"Only ``modin.pandas.Series`` boolean masks are acceptable, got: {type(row_loc)}",
        )
        masked_df = self.df.__constructor__(
            query_compiler=self.qc.getitem_array(row_loc._query_compiler)
        )
        # Passing `slice(None)` as a row indexer since we've just applied it
        return type(self)(masked_df)[(slice(None), col_loc)]


class _LocIndexer(metaclass_resolver(_LocationIndexerBase)):
    """
    An indexer for modin_df.loc[] functionality.

    Parameters
    ----------
    modin_df : modin.pandas.DataFrame
        DataFrame to operate on.
    """

    def __getitem__(self, key):
        """
        Retrieve dataset according to `key`.

        Parameters
        ----------
        key : callable, scalar, or tuple
            The global row index to retrieve data from.

        Returns
        -------
        modin.pandas.DataFrame or modin.pandas.Series
            Located dataset.

        See Also
        --------
        pandas.DataFrame.loc
        """
        if self.df.empty:
            return self.df._default_to_pandas(lambda df: df.loc[key])
        row_loc, col_loc, ndim = self._parse_row_and_column_locators(key)
        self.row_scalar = is_scalar(row_loc)
        self.col_scalar = is_scalar(col_loc)

        if isinstance(row_loc, Series) and is_boolean_array(row_loc):
            return self._handle_boolean_masking(row_loc, col_loc)

        row_lookup, col_lookup = self._compute_lookup(row_loc, col_loc)
        result = super(_LocIndexer, self).__getitem__(row_lookup, col_lookup, ndim)
        if isinstance(result, Series):
            result._parent = self.df
            result._parent_axis = 0
        col_loc_as_list = [col_loc] if self.col_scalar else col_loc
        row_loc_as_list = [row_loc] if self.row_scalar else row_loc
        # Pandas drops the levels that are in the `loc`, so we have to as well.
        if (
            isinstance(result, (Series, DataFrame))
            and result._query_compiler.has_multiindex()
        ):
            if (
                isinstance(result, Series)
                and not isinstance(col_loc_as_list, slice)
                and all(
                    col_loc_as_list[i] in result.index.levels[i]
                    for i in range(len(col_loc_as_list))
                )
            ):
                result.index = result.index.droplevel(list(range(len(col_loc_as_list))))
            elif not isinstance(row_loc_as_list, slice) and all(
                not isinstance(row_loc_as_list[i], slice)
                and row_loc_as_list[i] in result.index.levels[i]
                for i in range(len(row_loc_as_list))
            ):
                result.index = result.index.droplevel(list(range(len(row_loc_as_list))))
        if (
            hasattr(result, "columns")
            and not isinstance(col_loc_as_list, slice)
            and result._query_compiler.has_multiindex(axis=1)
            and all(
                col_loc_as_list[i] in result.columns.levels[i]
                for i in range(len(col_loc_as_list))
            )
        ):
            result.columns = result.columns.droplevel(list(range(len(col_loc_as_list))))
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
        Assign `item` value to dataset located by `key`.

        Parameters
        ----------
        key : callable or tuple
            The global row index to assign data to.
        item : modin.pandas.DataFrame, modin.pandas.Series or scalar
            Value that should be assigned to located dataset.

        See Also
        --------
        pandas.DataFrame.loc
        """
        row_loc, col_loc, _ = self._parse_row_and_column_locators(key)
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
                    row_lookup, col_lookup, is_scalar(row_loc), is_scalar(col_loc)
                ),
            )

    def _compute_enlarge_labels(self, locator, base_index):
        """
        Help to _enlarge_axis, compute common labels and extra labels.

        Parameters
        ----------
        locator : pandas.Index
            Index from locator.
        base_index : pandas.Index
            Current index.

        Returns
        -------
        nan_labels : pandas.Index
            The labels that need to be added.
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
        Compute index and column labels from index and column locators.

        Parameters
        ----------
        row_loc : scalar, slice, list, array or tuple
            Row locator.
        col_loc : scalar, slice, list, array or tuple
            Columns locator.

        Returns
        -------
        row_lookup : slice(None) if full axis grab, pandas.RangeIndex if repetition is detected, numpy.ndarray otherwise
            List of index labels.
        col_lookup : slice(None) if full axis grab, pandas.RangeIndex if repetition is detected, numpy.ndarray otherwise
            List of columns labels.

        Notes
        -----
        Usage of `slice(None)` as a resulting lookup is a hack to pass information about
        full-axis grab without computing actual indices that triggers lazy computations.
        Ideally, this API should get rid of using slices as indexers and either use a
        common ``Indexer`` object or range and ``np.ndarray`` only.
        """
        lookups = []
        for axis, axis_loc in enumerate((row_loc, col_loc)):
            if is_scalar(axis_loc):
                axis_loc = np.array([axis_loc])
            if isinstance(axis_loc, slice) or is_range_like(axis_loc):
                if isinstance(axis_loc, slice) and axis_loc == slice(None):
                    axis_lookup = axis_loc
                else:
                    axis_labels = self.qc.get_axis(axis)
                    # `slice_indexer` returns a fully-defined numeric slice for a non-fully-defined labels-based slice
                    axis_lookup = axis_labels.slice_indexer(
                        axis_loc.start, axis_loc.stop, axis_loc.step
                    )
                    # Converting negative indices to their actual positions:
                    axis_lookup = pandas.RangeIndex(
                        start=(
                            axis_lookup.start
                            if axis_lookup.start >= 0
                            else axis_lookup.start + len(axis_labels)
                        ),
                        stop=(
                            axis_lookup.stop
                            if axis_lookup.stop >= 0
                            else axis_lookup.stop + len(axis_labels)
                        ),
                        step=axis_lookup.step,
                    )
            elif self.qc.has_multiindex(axis):
                # `Index.get_locs` raises an IndexError by itself if missing labels were provided,
                # we don't have to do missing-check for the received `axis_lookup`.
                if isinstance(axis_loc, pandas.MultiIndex):
                    axis_lookup = self.qc.get_axis(axis).get_indexer_for(axis_loc)
                else:
                    axis_lookup = self.qc.get_axis(axis).get_locs(axis_loc)
            elif is_boolean_array(axis_loc):
                axis_lookup = boolean_mask_to_numeric(axis_loc)
            else:
                axis_labels = self.qc.get_axis(axis)
                if is_list_like(axis_loc) and not isinstance(
                    axis_loc, (np.ndarray, pandas.Index)
                ):
                    # `Index.get_indexer_for` works much faster with numpy arrays than with python lists,
                    # so although we lose some time here on converting to numpy, `Index.get_indexer_for`
                    # speedup covers the loss that we gain here.
                    axis_loc = np.array(axis_loc, dtype=axis_labels.dtype)
                axis_lookup = axis_labels.get_indexer_for(axis_loc)
                # `Index.get_indexer_for` sets -1 value for missing labels, we have to verify whether
                # there are any -1 in the received indexer to raise a KeyError here.
                missing_mask = axis_lookup < 0
                if missing_mask.any():
                    missing_labels = (
                        # Converting `axis_loc` to maskable `np.array` to not fail
                        # on masking non-maskable list-like
                        np.array(axis_loc)[missing_mask]
                        if is_list_like(axis_loc)
                        # If `axis_loc` is not a list-like then we can't select certain
                        # labels that are missing and so printing the whole indexer
                        else axis_loc
                    )
                    raise KeyError(missing_labels)

            if isinstance(axis_lookup, pandas.Index) and not is_range_like(axis_lookup):
                axis_lookup = axis_lookup.values

            lookups.append(axis_lookup)
        return lookups


class _iLocIndexer(metaclass_resolver(_LocationIndexerBase)):
    """
    An indexer for modin_df.iloc[] functionality.

    Parameters
    ----------
    modin_df : modin.pandas.DataFrame
        DataFrame to operate on.
    """

    def __getitem__(self, key):
        """
        Retrieve dataset according to `key`.

        Parameters
        ----------
        key : callable or tuple
            The global row numbers to retrieve data from.

        Returns
        -------
        DataFrame or Series
            Located dataset.

        See Also
        --------
        pandas.DataFrame.iloc
        """
        if self.df.empty:
            return self.df._default_to_pandas(lambda df: df.iloc[key])
        row_loc, col_loc, ndim = self._parse_row_and_column_locators(key)
        self.row_scalar = is_scalar(row_loc)
        self.col_scalar = is_scalar(col_loc)
        self._check_dtypes(row_loc)
        self._check_dtypes(col_loc)

        if isinstance(row_loc, Series) and is_boolean_array(row_loc):
            return self._handle_boolean_masking(row_loc, col_loc)

        row_lookup, col_lookup = self._compute_lookup(row_loc, col_loc)
        result = super(_iLocIndexer, self).__getitem__(row_lookup, col_lookup, ndim)
        if isinstance(result, Series):
            result._parent = self.df
            result._parent_axis = 0
        return result

    def __setitem__(self, key, item):
        """
        Assign `item` value to dataset located by `key`.

        Parameters
        ----------
        key : callable or tuple
            The global row numbers to assign data to.
        item : modin.pandas.DataFrame, modin.pandas.Series or scalar
            Value that should be assigned to located dataset.

        See Also
        --------
        pandas.DataFrame.iloc
        """
        row_loc, col_loc, _ = self._parse_row_and_column_locators(key)
        row_scalar = is_scalar(row_loc)
        col_scalar = is_scalar(col_loc)
        self._check_dtypes(row_loc)
        self._check_dtypes(col_loc)

        row_lookup, col_lookup = self._compute_lookup(row_loc, col_loc)
        super(_iLocIndexer, self).__setitem__(
            row_lookup,
            col_lookup,
            item,
            axis=self._determine_setitem_axis(
                row_lookup, col_lookup, row_scalar, col_scalar
            ),
        )

    def _compute_lookup(self, row_loc, col_loc):
        """
        Compute index and column labels from index and column integer locators.

        Parameters
        ----------
        row_loc : slice, list, array or tuple
            Row locator.
        col_loc : slice, list, array or tuple
            Columns locator.

        Returns
        -------
        row_lookup : slice(None) if full axis grab, pandas.RangeIndex if repetition is detected, numpy.ndarray otherwise
            List of index labels.
        col_lookup : slice(None) if full axis grab, pandas.RangeIndex if repetition is detected, numpy.ndarray otherwise
            List of columns labels.

        Notes
        -----
        Usage of `slice(None)` as a resulting lookup is a hack to pass information about
        full-axis grab without computing actual indices that triggers lazy computations.
        Ideally, this API should get rid of using slices as indexers and either use a
        common ``Indexer`` object or range and ``np.ndarray`` only.
        """
        lookups = []
        for axis, axis_loc in enumerate((row_loc, col_loc)):
            if is_scalar(axis_loc):
                axis_loc = np.array([axis_loc])
            if isinstance(axis_loc, slice):
                axis_lookup = (
                    axis_loc
                    if axis_loc == slice(None)
                    else pandas.RangeIndex(
                        *axis_loc.indices(len(self.qc.get_axis(axis)))
                    )
                )
            elif is_range_like(axis_loc):
                axis_lookup = pandas.RangeIndex(
                    axis_loc.start, axis_loc.stop, axis_loc.step
                )
            elif is_boolean_array(axis_loc):
                axis_lookup = boolean_mask_to_numeric(axis_loc)
            else:
                if isinstance(axis_loc, pandas.Index):
                    axis_loc = axis_loc.values
                elif is_list_like(axis_loc) and not isinstance(axis_loc, np.ndarray):
                    # `Index.__getitem__` works much faster with numpy arrays than with python lists,
                    # so although we lose some time here on converting to numpy, `Index.__getitem__`
                    # speedup covers the loss that we gain here.
                    axis_loc = np.array(axis_loc, dtype=np.int64)
                # Relatively fast check allows us to not trigger `self.qc.get_axis()` computation
                # if there're no negative indices and so they don't not depend on the axis length.
                if isinstance(axis_loc, np.ndarray) and not (axis_loc < 0).any():
                    axis_lookup = axis_loc
                else:
                    axis_lookup = pandas.RangeIndex(len(self.qc.get_axis(axis)))[
                        axis_loc
                    ]

            if isinstance(axis_lookup, pandas.Index) and not is_range_like(axis_lookup):
                axis_lookup = axis_lookup.values
            lookups.append(axis_lookup)
        return lookups

    def _check_dtypes(self, locator):
        """
        Check that `locator` is an integer scalar, integer slice, integer list or array of booleans.

        Parameters
        ----------
        locator : scalar, list, slice or array
            Object to check.

        Raises
        ------
        ValueError
            If check fails.
        """
        is_int = is_integer(locator)
        is_int_slice = is_integer_slice(locator)
        is_int_arr = is_integer_array(locator)
        is_bool_arr = is_boolean_array(locator)

        if not any([is_int, is_int_slice, is_int_arr, is_bool_arr]):
            raise ValueError(_ILOC_INT_ONLY_ERROR)
