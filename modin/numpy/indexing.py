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
  and __setitem__ that work with Modin NumPy Array's internal index. Base
  class's __{get,set}item__ takes in partitions & idx_in_partition data
  and perform lookup/item write.

_iLocIndexer is responsible for indexer specific logic and
  lookup computation. Loc will take care of enlarge DataFrame. Both indexer
  will take care of translating pandas's lookup to Modin DataFrame's internal
  lookup.

An illustration is available at
https://github.com/ray-project/ray/pull/1955#issuecomment-386781826
"""

import itertools

import numpy as np
import pandas
from pandas.api.types import is_bool, is_list_like
from pandas.core.dtypes.common import is_bool_dtype, is_integer, is_integer_dtype
from pandas.core.indexing import IndexingError

from modin.error_message import ErrorMessage
from modin.pandas.indexing import compute_sliced_len, is_range_like, is_slice, is_tuple
from modin.pandas.utils import is_scalar

from .arr import array


def broadcast_item(
    obj,
    row_lookup,
    col_lookup,
    item,
    need_columns_reindex=True,
):
    """
    Use NumPy to broadcast or reshape item with reindexing.

    Parameters
    ----------
    obj : DataFrame or Series
        The object containing the necessary information about the axes.
    row_lookup : slice or scalar
        The global row index to locate inside of `item`.
    col_lookup : range, array, list, slice or scalar
        The global col index to locate inside of `item`.
    item : DataFrame, Series, or query_compiler
        Value that should be broadcast to a new shape of `to_shape`.
    need_columns_reindex : bool, default: True
        In the case of assigning columns to a dataframe (broadcasting is
        part of the flow), reindexing is not needed.

    Returns
    -------
    np.ndarray
        `item` after it was broadcasted to `to_shape`.

    Raises
    ------
    ValueError
        1) If `row_lookup` or `col_lookup` contains values missing in
        DataFrame/Series index or columns correspondingly.
        2) If `item` cannot be broadcast from its own shape to `to_shape`.

    Notes
    -----
    NumPy is memory efficient, there shouldn't be performance issue.
    """
    new_row_len = (
        len(obj._query_compiler.index[row_lookup])
        if isinstance(row_lookup, slice)
        else len(row_lookup)
    )
    new_col_len = (
        len(obj._query_compiler.columns[col_lookup])
        if isinstance(col_lookup, slice)
        else len(col_lookup)
    )
    to_shape = new_row_len, new_col_len

    if isinstance(item, array):
        # convert indices in lookups to names, as pandas reindex expects them to be so
        axes_to_reindex = {}
        index_values = obj._query_compiler.index[row_lookup]
        if not index_values.equals(item._query_compiler.index):
            axes_to_reindex["index"] = index_values
        if need_columns_reindex and isinstance(item, array) and item._ndim == 2:
            column_values = obj._query_compiler.columns[col_lookup]
            if not column_values.equals(item._query_compiler.columns):
                axes_to_reindex["columns"] = column_values
        # New value for columns/index make that reindex add NaN values
        if axes_to_reindex:
            row_axes = axes_to_reindex.get("index", None)
            if row_axes is not None:
                item._query_compiler = item._query_compiler.reindex(
                    axis=0, labels=row_axes, copy=None
                )
            col_axes = axes_to_reindex.get("columns", None)
            if col_axes is not None:
                item._query_compiler = item._query_compiler.reindex(
                    axis=1, labels=col_axes, copy=None
                )
    try:
        item = np.array(item) if not isinstance(item, array) else item._to_numpy()
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
    if isinstance(x, (np.ndarray, array, pandas.Series, pandas.Index)):
        return is_bool_dtype(x.dtype)
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
    if isinstance(x, (np.ndarray, array, pandas.Series, pandas.Index)):
        return is_integer_dtype(x.dtype)
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
    if isinstance(indexer, (np.ndarray, array, pandas.Series)):
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


class ArrayIndexer(object):
    """
    An indexer for modin_arr.__{get|set}item__ functionality.

    Parameters
    ----------
    array : modin.numpy.array
        Array to operate on.
    """

    def __init__(self, array):
        self.arr = array

    def _get_numpy_object_from_qc_view(
        self,
        qc_view,
        row_scalar: bool,
        col_scalar: bool,
        ndim: int,
    ):
        """
        Convert the query compiler view to the appropriate NumPy object.

        Parameters
        ----------
        qc_view : BaseQueryCompiler
            Query compiler to convert.
        row_scalar : bool
            Whether indexer for rows is scalar.
        col_scalar : bool
            Whether indexer for columns is scalar.
        ndim : {0, 1, 2}
            Number of dimensions in dataset to be retrieved.

        Returns
        -------
        modin.numpy.array
            The array object with the data from the query compiler view.

        Notes
        -----
        Usage of `slice(None)` as a lookup is a hack to pass information about
        full-axis grab without computing actual indices that triggers lazy computations.
        Ideally, this API should get rid of using slices as indexers and either use a
        common ``Indexer`` object or range and ``np.ndarray`` only.
        """
        if ndim == 2:
            return array(_query_compiler=qc_view, _ndim=self.arr._ndim)
        if self.arr._ndim == 1 and not row_scalar:
            return array(_query_compiler=qc_view, _ndim=1)

        if self.arr._ndim == 1:
            _ndim = 0
        elif ndim == 0:
            _ndim = 0
        else:
            # We are in the case where ndim == 1
            # The axis we squeeze on depends on whether we are looking for an exact
            # value or a subset of rows and columns. Knowing if we have a full MultiIndex
            # lookup or scalar lookup can help us figure out whether we need to squeeze
            # on the row or column index.
            if row_scalar and col_scalar:
                _ndim = 0
            elif not any([row_scalar, col_scalar]):
                _ndim = 2
            else:
                _ndim = 1
                if row_scalar:
                    qc_view = qc_view.transpose()

        if _ndim == 0:
            return qc_view.to_numpy()[0, 0]

        res_arr = array(_query_compiler=qc_view, _ndim=_ndim)
        return res_arr

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

        row_loc = row_loc(self.arr) if callable(row_loc) else row_loc
        col_loc = col_loc(self.arr) if callable(col_loc) else col_loc
        row_loc = row_loc._to_numpy() if isinstance(row_loc, array) else row_loc
        col_loc = col_loc._to_numpy() if isinstance(col_loc, array) else col_loc
        return row_loc, col_loc, _compute_ndim(row_loc, col_loc)

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
        row_loc, col_loc, ndim = self._parse_row_and_column_locators(key)
        row_scalar = is_scalar(row_loc)
        col_scalar = is_scalar(col_loc)
        self._check_dtypes(row_loc)
        self._check_dtypes(col_loc)

        row_lookup, col_lookup = self._compute_lookup(row_loc, col_loc)
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
        qc_view = self.arr._query_compiler.take_2d_positional(row_lookup, col_lookup)
        result = self._get_numpy_object_from_qc_view(
            qc_view,
            row_scalar=row_scalar,
            col_scalar=col_scalar,
            ndim=ndim,
        )
        return result

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
        if self.arr.shape == (1, 1):
            return None if not (row_scalar ^ col_scalar) else 1 if row_scalar else 0

        def get_axis(axis):
            return (
                self.arr._query_compiler.index
                if axis == 0
                else self.arr._query_compiler.columns
            )

        row_lookup_len, col_lookup_len = [
            (
                len(lookup)
                if not isinstance(lookup, slice)
                else compute_sliced_len(lookup, len(get_axis(i)))
            )
            for i, lookup in enumerate([row_lookup, col_lookup])
        ]

        if col_lookup_len == 1 and row_lookup_len == 1:
            axis = None
        elif (
            row_lookup_len == len(self.arr._query_compiler.index)
            and col_lookup_len == 1
            and self.arr._ndim == 2
        ):
            axis = 0
        elif (
            col_lookup_len == len(self.arr._query_compiler.columns)
            and row_lookup_len == 1
        ):
            axis = 1
        else:
            axis = None
        return axis

    def _setitem_positional(self, row_lookup, col_lookup, item, axis=None):
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
            row_lookup = range(len(self.arr._query_compiler.index))[row_lookup]
        if isinstance(col_lookup, slice):
            col_lookup = range(len(self.arr._query_compiler.columns))[col_lookup]

        new_qc = self.arr._query_compiler.write_items(row_lookup, col_lookup, item)
        self.arr._update_inplace(new_qc)

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
        self._setitem_positional(
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
                        *axis_loc.indices(len(self.arr._query_compiler.get_axis(axis)))
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
                    axis_lookup = pandas.RangeIndex(
                        len(self.arr._query_compiler.get_axis(axis))
                    )[axis_loc]

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
