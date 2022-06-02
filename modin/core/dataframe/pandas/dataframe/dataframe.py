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
Module contains class PandasDataframe.

PandasDataframe is a parent abstract class for any dataframe class
for pandas storage format.
"""
from collections import OrderedDict
import numpy as np
import pandas
import datetime
from pandas.core.indexes.api import ensure_index, Index, RangeIndex
from pandas.core.dtypes.common import is_numeric_dtype, is_list_like
from pandas._libs.lib import no_default
from typing import List, Hashable, Optional, Callable, Union, Dict

from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.error_message import ErrorMessage
from modin.core.storage_formats.pandas.parsers import (
    find_common_type_cat as find_common_type,
)
from modin.core.dataframe.base.dataframe.dataframe import ModinDataframe
from modin.core.dataframe.base.dataframe.utils import (
    Axis,
    JoinType,
)
from modin.pandas.indexing import is_range_like
from modin.pandas.utils import is_full_grab_slice, check_both_not_none
from modin.logging import LoggerMetaClass


def lazy_metadata_decorator(apply_axis=None, axis_arg=-1, transpose=False):
    """
    Lazily propagate metadata for the ``PandasDataframe``.

    This decorator first adds the minimum required reindexing operations
    to each partition's queue of functions to be lazily applied for
    each PandasDataframe in the arguments by applying the function
    run_f_on_minimally_updated_metadata. The decorator also sets the
    flags for deferred metadata synchronization on the function result
    if necessary.

    Parameters
    ----------
    apply_axis : str, default: None
        The axes on which to apply the reindexing operations to the `self._partitions` lazily.
        Case None: No lazy metadata propagation.
        Case "both": Add reindexing operations on both axes to partition queue.
        Case "opposite": Add reindexing operations complementary to given axis.
        Case "rows": Add reindexing operations on row axis to partition queue.
    axis_arg : int, default: -1
        The index or column axis.
    transpose : bool, default: False
        Boolean for if a transpose operation is being used.

    Returns
    -------
    Wrapped Function.
    """

    def decorator(f):
        from functools import wraps

        @wraps(f)
        def run_f_on_minimally_updated_metadata(self, *args, **kwargs):
            for obj in (
                [self]
                + [o for o in args if isinstance(o, PandasDataframe)]
                + [v for v in kwargs.values() if isinstance(v, PandasDataframe)]
                + [
                    d
                    for o in args
                    if isinstance(o, list)
                    for d in o
                    if isinstance(d, PandasDataframe)
                ]
                + [
                    d
                    for _, o in kwargs.items()
                    if isinstance(o, list)
                    for d in o
                    if isinstance(d, PandasDataframe)
                ]
            ):
                if apply_axis == "both":
                    if obj._deferred_index and obj._deferred_column:
                        obj._propagate_index_objs(axis=None)
                    elif obj._deferred_index:
                        obj._propagate_index_objs(axis=0)
                    elif obj._deferred_column:
                        obj._propagate_index_objs(axis=1)
                elif apply_axis == "opposite":
                    if "axis" not in kwargs:
                        axis = args[axis_arg]
                    else:
                        axis = kwargs["axis"]
                    if axis == 0 and obj._deferred_column:
                        obj._propagate_index_objs(axis=1)
                    elif axis == 1 and obj._deferred_index:
                        obj._propagate_index_objs(axis=0)
                elif apply_axis == "rows":
                    obj._propagate_index_objs(axis=0)
            result = f(self, *args, **kwargs)
            if apply_axis is None and not transpose:
                result._deferred_index = self._deferred_index
                result._deferred_column = self._deferred_column
            elif apply_axis is None and transpose:
                result._deferred_index = self._deferred_column
                result._deferred_column = self._deferred_index
            elif apply_axis == "opposite":
                if axis == 0:
                    result._deferred_index = self._deferred_index
                else:
                    result._deferred_column = self._deferred_column
            elif apply_axis == "rows":
                result._deferred_column = self._deferred_column
            return result

        return run_f_on_minimally_updated_metadata

    return decorator


class PandasDataframe(object, metaclass=LoggerMetaClass):
    """
    An abstract class that represents the parent class for any pandas storage format dataframe class.

    This class provides interfaces to run operations on dataframe partitions.

    Parameters
    ----------
    partitions : np.ndarray
        A 2D NumPy array of partitions.
    index : sequence
        The index for the dataframe. Converted to a ``pandas.Index``.
    columns : sequence
        The columns object for the dataframe. Converted to a ``pandas.Index``.
    row_lengths : list, optional
        The length of each partition in the rows. The "height" of
        each of the block partitions. Is computed if not provided.
    column_widths : list, optional
        The width of each partition in the columns. The "width" of
        each of the block partitions. Is computed if not provided.
    dtypes : pandas.Series, optional
        The data types for the dataframe columns.
    """

    _partition_mgr_cls = None
    _query_compiler_cls = PandasQueryCompiler
    # These properties flag whether or not we are deferring the metadata synchronization
    _deferred_index = False
    _deferred_column = False

    @property
    def __constructor__(self):
        """
        Create a new instance of this object.

        Returns
        -------
        PandasDataframe
        """
        return type(self)

    def __init__(
        self,
        partitions,
        index,
        columns,
        row_lengths=None,
        column_widths=None,
        dtypes=None,
    ):
        self._partitions = partitions
        self._index_cache = ensure_index(index)
        self._columns_cache = ensure_index(columns)
        if row_lengths is not None and len(self.index) > 0:
            # An empty frame can have 0 rows but a nonempty index. If the frame
            # does have rows, the number of rows must equal the size of the
            # index.
            num_rows = sum(row_lengths)
            if num_rows > 0:
                ErrorMessage.catch_bugs_and_request_email(
                    num_rows != len(self._index_cache),
                    "Row lengths: {} != {}".format(num_rows, len(self._index_cache)),
                )
            ErrorMessage.catch_bugs_and_request_email(
                any(val < 0 for val in row_lengths),
                "Row lengths cannot be negative: {}".format(row_lengths),
            )
        self._row_lengths_cache = row_lengths
        if column_widths is not None and len(self.columns) > 0:
            # An empty frame can have 0 column but a nonempty column index. If
            # the frame does have columns, the number of columns must equal the
            # size of the columns.
            num_columns = sum(column_widths)
            if num_columns > 0:
                ErrorMessage.catch_bugs_and_request_email(
                    num_columns != len(self._columns_cache),
                    "Column widths: {} != {}".format(
                        num_columns, len(self._columns_cache)
                    ),
                )
            ErrorMessage.catch_bugs_and_request_email(
                any(val < 0 for val in column_widths),
                "Column widths cannot be negative: {}".format(column_widths),
            )
        self._column_widths_cache = column_widths
        self._dtypes = dtypes
        self._filter_empties()

    @property
    def _row_lengths(self):
        """
        Compute the row partitions lengths if they are not cached.

        Returns
        -------
        list
            A list of row partitions lengths.
        """
        if self._row_lengths_cache is None:
            if len(self._partitions.T) > 0:
                self._row_lengths_cache = [
                    obj.length() for obj in self._partitions.T[0]
                ]
            else:
                self._row_lengths_cache = []
        return self._row_lengths_cache

    @property
    def _column_widths(self):
        """
        Compute the column partitions widths if they are not cached.

        Returns
        -------
        list
            A list of column partitions widths.
        """
        if self._column_widths_cache is None:
            if len(self._partitions) > 0:
                self._column_widths_cache = [obj.width() for obj in self._partitions[0]]
            else:
                self._column_widths_cache = []
        return self._column_widths_cache

    @property
    def _axes_lengths(self):
        """
        Get a pair of row partitions lengths and column partitions widths.

        Returns
        -------
        list
            The pair of row partitions lengths and column partitions widths.
        """
        return [self._row_lengths, self._column_widths]

    @property
    def dtypes(self):
        """
        Compute the data types if they are not cached.

        Returns
        -------
        pandas.Series
            A pandas Series containing the data types for this dataframe.
        """
        if self._dtypes is None:
            self._dtypes = self._compute_dtypes()
        return self._dtypes

    def _compute_dtypes(self):
        """
        Compute the data types via TreeReduce pattern.

        Returns
        -------
        pandas.Series
            A pandas Series containing the data types for this dataframe.
        """

        def dtype_builder(df):
            return df.apply(lambda col: find_common_type(col.values), axis=0)

        map_func = self._build_treereduce_func(0, lambda df: df.dtypes)
        reduce_func = self._build_treereduce_func(0, dtype_builder)
        # For now we will use a pandas Series for the dtypes.
        if len(self.columns) > 0:
            dtypes = self.tree_reduce(0, map_func, reduce_func).to_pandas().iloc[0]
        else:
            dtypes = pandas.Series([])
        # reset name to None because we use "__reduced__" internally
        dtypes.name = None
        return dtypes

    _index_cache = None
    _columns_cache = None

    def _validate_set_axis(self, new_labels, old_labels):
        """
        Validate the possibility of replacement of old labels with the new labels.

        Parameters
        ----------
        new_labels : list-like
            The labels to replace with.
        old_labels : list-like
            The labels to replace.

        Returns
        -------
        list-like
            The validated labels.
        """
        new_labels = ensure_index(new_labels)
        old_len = len(old_labels)
        new_len = len(new_labels)
        if old_len != new_len:
            raise ValueError(
                f"Length mismatch: Expected axis has {old_len} elements, "
                + f"new values have {new_len} elements"
            )
        return new_labels

    def _get_index(self):
        """
        Get the index from the cache object.

        Returns
        -------
        pandas.Index
            An index object containing the row labels.
        """
        return self._index_cache

    def _get_columns(self):
        """
        Get the columns from the cache object.

        Returns
        -------
        pandas.Index
            An index object containing the column labels.
        """
        return self._columns_cache

    def _set_index(self, new_index):
        """
        Replace the current row labels with new labels.

        Parameters
        ----------
        new_index : list-like
            The new row labels.
        """
        if self._index_cache is None:
            self._index_cache = ensure_index(new_index)
        else:
            new_index = self._validate_set_axis(new_index, self._index_cache)
            self._index_cache = new_index
        self.synchronize_labels(axis=0)

    def _set_columns(self, new_columns):
        """
        Replace the current column labels with new labels.

        Parameters
        ----------
        new_columns : list-like
           The new column labels.
        """
        if self._columns_cache is None:
            self._columns_cache = ensure_index(new_columns)
        else:
            new_columns = self._validate_set_axis(new_columns, self._columns_cache)
            self._columns_cache = new_columns
            if self._dtypes is not None:
                self._dtypes.index = new_columns
        self.synchronize_labels(axis=1)

    columns = property(_get_columns, _set_columns)
    index = property(_get_index, _set_index)

    @property
    def axes(self):
        """
        Get index and columns that can be accessed with an `axis` integer.

        Returns
        -------
        list
            List with two values: index and columns.
        """
        return [self.index, self.columns]

    def _compute_axis_labels(self, axis: int, partitions=None):
        """
        Compute the labels for specific `axis`.

        Parameters
        ----------
        axis : int
            Axis to compute labels along.
        partitions : np.ndarray, optional
            A 2D NumPy array of partitions from which labels will be grabbed.
            If not specified, partitions will be taken from `self._partitions`.

        Returns
        -------
        pandas.Index
            Labels for the specified `axis`.
        """
        if partitions is None:
            partitions = self._partitions
        return self._partition_mgr_cls.get_indices(
            axis, partitions, lambda df: df.axes[axis]
        )

    def _filter_empties(self):
        """Remove empty partitions from `self._partitions` to avoid triggering excess computation."""
        if len(self.axes[0]) == 0 or len(self.axes[1]) == 0:
            # This is the case for an empty frame. We don't want to completely remove
            # all metadata and partitions so for the moment, we won't prune if the frame
            # is empty.
            # TODO: Handle empty dataframes better
            return
        self._partitions = np.array(
            [
                [
                    self._partitions[i][j]
                    for j in range(len(self._partitions[i]))
                    if j < len(self._column_widths) and self._column_widths[j] != 0
                ]
                for i in range(len(self._partitions))
                if i < len(self._row_lengths) and self._row_lengths[i] != 0
            ]
        )
        self._column_widths_cache = [w for w in self._column_widths if w != 0]
        self._row_lengths_cache = [r for r in self._row_lengths if r != 0]

    def synchronize_labels(self, axis=None):
        """
        Set the deferred axes variables for the ``PandasDataframe``.

        Parameters
        ----------
        axis : int, default: None
            The deferred axis.
            0 for the index, 1 for the columns.
        """
        if axis is None:
            self._deferred_index = True
            self._deferred_column = True
        elif axis == 0:
            self._deferred_index = True
        else:
            self._deferred_column = True

    def _propagate_index_objs(self, axis=None):
        """
        Synchronize labels by applying the index object for specific `axis` to the `self._partitions` lazily.

        Adds `set_axis` function to call-queue of each partition from `self._partitions`
        to apply new axis.

        Parameters
        ----------
        axis : int, default: None
            The axis to apply to. If it's None applies to both axes.
        """
        self._filter_empties()
        if axis is None or axis == 0:
            cum_row_lengths = np.cumsum([0] + self._row_lengths)
        if axis is None or axis == 1:
            cum_col_widths = np.cumsum([0] + self._column_widths)

        if axis is None:

            def apply_idx_objs(df, idx, cols):
                return df.set_axis(idx, axis="index", inplace=False).set_axis(
                    cols, axis="columns", inplace=False
                )

            self._partitions = np.array(
                [
                    [
                        self._partitions[i][j].add_to_apply_calls(
                            apply_idx_objs,
                            idx=self.index[
                                slice(cum_row_lengths[i], cum_row_lengths[i + 1])
                            ],
                            cols=self.columns[
                                slice(cum_col_widths[j], cum_col_widths[j + 1])
                            ],
                        )
                        for j in range(len(self._partitions[i]))
                    ]
                    for i in range(len(self._partitions))
                ]
            )
            self._deferred_index = False
            self._deferred_column = False
        elif axis == 0:

            def apply_idx_objs(df, idx):
                return df.set_axis(idx, axis="index", inplace=False)

            self._partitions = np.array(
                [
                    [
                        self._partitions[i][j].add_to_apply_calls(
                            apply_idx_objs,
                            idx=self.index[
                                slice(cum_row_lengths[i], cum_row_lengths[i + 1])
                            ],
                        )
                        for j in range(len(self._partitions[i]))
                    ]
                    for i in range(len(self._partitions))
                ]
            )
            self._deferred_index = False
        elif axis == 1:

            def apply_idx_objs(df, cols):
                return df.set_axis(cols, axis="columns", inplace=False)

            self._partitions = np.array(
                [
                    [
                        self._partitions[i][j].add_to_apply_calls(
                            apply_idx_objs,
                            cols=self.columns[
                                slice(cum_col_widths[j], cum_col_widths[j + 1])
                            ],
                        )
                        for j in range(len(self._partitions[i]))
                    ]
                    for i in range(len(self._partitions))
                ]
            )
            self._deferred_column = False
        else:
            ErrorMessage.catch_bugs_and_request_email(
                axis is not None and axis not in [0, 1]
            )

    @lazy_metadata_decorator(apply_axis=None)
    def mask(
        self,
        row_labels: Optional[List[Hashable]] = None,
        row_positions: Optional[List[int]] = None,
        col_labels: Optional[List[Hashable]] = None,
        col_positions: Optional[List[int]] = None,
    ) -> "PandasDataframe":
        """
        Lazily select columns or rows from given indices.

        Parameters
        ----------
        row_labels : list of hashable, optional
            The row labels to extract.
        row_positions : list-like of ints, optional
            The row positions to extract.
        col_labels : list of hashable, optional
            The column labels to extract.
        col_positions : list-like of ints, optional
            The column positions to extract.

        Returns
        -------
        PandasDataframe
             A new PandasDataframe from the mask provided.

        Notes
        -----
        If both `row_labels` and `row_positions` are provided, a ValueError is raised.
        The same rule applies for `col_labels` and `col_positions`.
        """
        if check_both_not_none(row_labels, row_positions):
            raise ValueError(
                "Both row_labels and row_positions were provided - please provide only one of row_labels and row_positions."
            )
        if check_both_not_none(col_labels, col_positions):
            raise ValueError(
                "Both col_labels and col_positions were provided - please provide only one of col_labels and col_positions."
            )
        indexers = []
        for axis, indexer in enumerate((row_positions, col_positions)):
            if is_range_like(indexer):
                if indexer.step == 1 and len(indexer) == len(self.axes[axis]):
                    # By this function semantics, `None` indexer is a full-axis access
                    indexer = None
                elif indexer is not None and not isinstance(indexer, pandas.RangeIndex):
                    # Pure python's range is not fully compatible with a list of ints,
                    # converting it to ``pandas.RangeIndex``` that is compatible.
                    indexer = pandas.RangeIndex(
                        indexer.start, indexer.stop, indexer.step
                    )
            else:
                ErrorMessage.catch_bugs_and_request_email(
                    failure_condition=not (indexer is None or is_list_like(indexer)),
                    extra_log=f"Mask takes only list-like numeric indexers, received: {type(indexer)}",
                )
            indexers.append(indexer)
        row_positions, col_positions = indexers

        if (
            col_labels is None
            and col_positions is None
            and row_labels is None
            and row_positions is None
        ):
            return self.copy()
        # Get numpy array of positions of values from `row_labels`
        if row_labels is not None:
            row_positions = self.index.get_indexer_for(row_labels)
        if row_positions is not None:
            sorted_row_positions = (
                row_positions
                if (
                    (is_range_like(row_positions) and row_positions.step > 0)
                    # `np.sort` of empty list returns an array with `float` dtype,
                    # which doesn't work well as an indexer
                    or len(row_positions) == 0
                )
                else np.sort(row_positions)
            )
            # Get dict of row_parts as {row_index: row_internal_indices}
            # TODO: Rename `row_partitions_list`->`row_partitions_dict`
            row_partitions_list = self._get_dict_of_block_index(
                0, sorted_row_positions, are_indices_sorted=True
            )
            new_row_lengths = [
                len(
                    # Row lengths for slice are calculated as the length of the slice
                    # on the partition. Often this will be the same length as the current
                    # length, but sometimes it is different, thus the extra calculation.
                    range(*part_indexer.indices(self._row_lengths[part_idx]))
                    if isinstance(part_indexer, slice)
                    else part_indexer
                )
                for part_idx, part_indexer in row_partitions_list.items()
            ]
            new_index = self.index[
                # pandas Index is more likely to preserve its metadata if the indexer is slice
                slice(row_positions.start, row_positions.stop, row_positions.step)
                # TODO: Fast range processing of non-positive-step ranges is not yet supported
                if is_range_like(row_positions) and row_positions.step > 0
                else sorted_row_positions
            ]
        else:
            row_partitions_list = {
                i: slice(None) for i in range(len(self._row_lengths))
            }
            new_row_lengths = self._row_lengths
            new_index = self.index

        # Get numpy array of positions of values from `col_labels`
        if col_labels is not None:
            col_positions = self.columns.get_indexer_for(col_labels)
        if col_positions is not None:
            sorted_col_positions = (
                col_positions
                if (
                    (is_range_like(col_positions) and col_positions.step > 0)
                    # `np.sort` of empty list returns an array with `float` dtype,
                    # which doesn't work well as an indexer
                    or len(col_positions) == 0
                )
                else np.sort(col_positions)
            )
            # Get dict of col_parts as {col_index: col_internal_indices}
            col_partitions_list = self._get_dict_of_block_index(
                1, sorted_col_positions, are_indices_sorted=True
            )
            new_col_widths = [
                len(
                    # Column widths for slice are calculated as the length of the slice
                    # on the partition. Often this will be the same length as the current
                    # length, but sometimes it is different, thus the extra calculation.
                    range(*part_indexer.indices(self._column_widths[part_idx]))
                    if isinstance(part_indexer, slice)
                    else part_indexer
                )
                for part_idx, part_indexer in col_partitions_list.items()
            ]
            # Use the slice to calculate the new columns
            # TODO: Support fast processing of negative-step ranges
            if is_range_like(col_positions) and col_positions.step > 0:
                # pandas Index is more likely to preserve its metadata if the indexer is slice
                monotonic_col_idx = slice(
                    col_positions.start, col_positions.stop, col_positions.step
                )
            else:
                monotonic_col_idx = sorted_col_positions
            new_columns = self.columns[monotonic_col_idx]
            ErrorMessage.catch_bugs_and_request_email(
                failure_condition=sum(new_col_widths) != len(new_columns),
                extra_log=f"{sum(new_col_widths)} != {len(new_columns)}.\n{col_positions}\n{self._column_widths}\n{col_partitions_list}",
            )
            if self._dtypes is not None:
                new_dtypes = self.dtypes.iloc[monotonic_col_idx]
            else:
                new_dtypes = None
        else:
            col_partitions_list = {
                i: slice(None) for i in range(len(self._column_widths))
            }
            new_col_widths = self._column_widths
            new_columns = self.columns
            if self._dtypes is not None:
                new_dtypes = self.dtypes
            else:
                new_dtypes = None
        new_partitions = np.array(
            [
                [
                    self._partitions[row_idx][col_idx].mask(
                        row_internal_indices, col_internal_indices
                    )
                    for col_idx, col_internal_indices in col_partitions_list.items()
                    if isinstance(col_internal_indices, slice)
                    or len(col_internal_indices) > 0
                ]
                for row_idx, row_internal_indices in row_partitions_list.items()
                if isinstance(row_internal_indices, slice)
                or len(row_internal_indices) > 0
            ]
        )
        intermediate = self.__constructor__(
            new_partitions,
            new_index,
            new_columns,
            new_row_lengths,
            new_col_widths,
            new_dtypes,
        )
        # Check if monotonically increasing, return if it is. Fast track code path for
        # common case to keep it fast.
        if (
            row_positions is None
            # Fast range processing of non-positive-step ranges is not yet supported
            or (is_range_like(row_positions) and row_positions.step > 0)
            or len(row_positions) == 1
            or np.all(row_positions[1:] >= row_positions[:-1])
        ) and (
            col_positions is None
            # Fast range processing of non-positive-step ranges is not yet supported
            or (is_range_like(col_positions) and col_positions.step > 0)
            or len(col_positions) == 1
            or np.all(col_positions[1:] >= col_positions[:-1])
        ):
            return intermediate
        # The new labels are often smaller than the old labels, so we can't reuse the
        # original order values because those were mapped to the original data. We have
        # to reorder here based on the expected order from within the data.
        # We create a dictionary mapping the position of the numeric index with respect
        # to all others, then recreate that order by mapping the new order values from
        # the old. This information is sent to `_reorder_labels`.
        if row_positions is not None:
            row_order_mapping = dict(
                zip(sorted_row_positions, range(len(row_positions)))
            )
            new_row_order = [row_order_mapping[idx] for idx in row_positions]
        else:
            new_row_order = None
        if col_positions is not None:
            col_order_mapping = dict(
                zip(sorted_col_positions, range(len(col_positions)))
            )
            new_col_order = [col_order_mapping[idx] for idx in col_positions]
        else:
            new_col_order = None
        return intermediate._reorder_labels(
            row_positions=new_row_order, col_positions=new_col_order
        )

    @lazy_metadata_decorator(apply_axis="rows")
    def from_labels(self) -> "PandasDataframe":
        """
        Convert the row labels to a column of data, inserted at the first position.

        Gives result by similar way as `pandas.DataFrame.reset_index`. Each level
        of `self.index` will be added as separate column of data.

        Returns
        -------
        PandasDataframe
            A PandasDataframe with new columns from index labels.
        """
        new_row_labels = pandas.RangeIndex(len(self.index))

        if self.index.nlevels > 1:
            level_names = [
                self.index.names[i]
                if self.index.names[i] is not None
                else "level_{}".format(i)
                for i in range(self.index.nlevels)
            ]
        else:
            level_names = [
                self.index.names[0]
                if self.index.names[0] is not None
                else "index"
                if "index" not in self.columns
                else "level_{}".format(0)
            ]

        # We will also use the `new_column_names` in the calculation of the internal metadata, so this is a
        # lightweight way of ensuring the metadata matches.
        if self.columns.nlevels > 1:
            # Column labels are different for multilevel index.
            new_column_names = pandas.MultiIndex.from_tuples(
                # Set level names on the 1st columns level and fill up empty level names with empty string.
                # Expand tuples in level names. This is how reset_index works when col_level col_fill are not specified.
                [
                    tuple(
                        list(level) + [""] * (self.columns.nlevels - len(level))
                        if isinstance(level, tuple)
                        else [level] + [""] * (self.columns.nlevels - 1)
                    )
                    for level in level_names
                ],
                names=self.columns.names,
            )
        else:
            new_column_names = pandas.Index(level_names, tupleize_cols=False)
        new_columns = new_column_names.append(self.columns)

        def from_labels_executor(df, **kwargs):
            # Setting the names here ensures that external and internal metadata always match.
            df.index.names = new_column_names

            # Handling of a case when columns have the same name as one of index levels names.
            # In this case `df.reset_index` provides errors related to columns duplication.
            # This case is possible because columns metadata updating is deferred. To workaround
            # `df.reset_index` error we allow columns duplication in "if" branch via `concat`.
            if any(name_level in df.columns for name_level in df.index.names):
                columns_to_add = df.index.to_frame()
                columns_to_add.reset_index(drop=True, inplace=True)
                df = df.reset_index(drop=True)
                result = pandas.concat([columns_to_add, df], axis=1, copy=False)
            else:
                result = df.reset_index()
            # Put the index back to the original due to GH#4394
            result.index = df.index
            return result

        new_parts = self._partition_mgr_cls.apply_func_to_select_indices(
            0,
            self._partitions,
            from_labels_executor,
            [0],
            keep_remaining=True,
        )
        new_column_widths = [
            self.index.nlevels + self._column_widths[0]
        ] + self._column_widths[1:]
        result = self.__constructor__(
            new_parts,
            new_row_labels,
            new_columns,
            row_lengths=self._row_lengths_cache,
            column_widths=new_column_widths,
        )
        # Set flag for propagating deferred row labels across dataframe partitions
        result.synchronize_labels(axis=0)
        return result

    def to_labels(self, column_list: List[Hashable]) -> "PandasDataframe":
        """
        Move one or more columns into the row labels. Previous labels are dropped.

        Parameters
        ----------
        column_list : list of hashable
            The list of column names to place as the new row labels.

        Returns
        -------
        PandasDataframe
            A new PandasDataframe that has the updated labels.
        """
        extracted_columns = self.mask(col_labels=column_list).to_pandas()
        if len(column_list) == 1:
            new_labels = pandas.Index(extracted_columns.squeeze(axis=1))
        else:
            new_labels = pandas.MultiIndex.from_frame(extracted_columns)
        result = self.mask(col_labels=[i for i in self.columns if i not in column_list])
        result.index = new_labels
        return result

    @lazy_metadata_decorator(apply_axis="both")
    def _reorder_labels(self, row_positions=None, col_positions=None):
        """
        Reorder the column and or rows in this DataFrame.

        Parameters
        ----------
        row_positions : list of int, optional
            The ordered list of new row orders such that each position within the list
            indicates the new position.
        col_positions : list of int, optional
            The ordered list of new column orders such that each position within the
            list indicates the new position.

        Returns
        -------
        PandasDataframe
            A new PandasDataframe with reordered columns and/or rows.
        """
        if row_positions is not None:
            ordered_rows = self._partition_mgr_cls.map_axis_partitions(
                0, self._partitions, lambda df: df.iloc[row_positions]
            )
            row_idx = self.index[row_positions]
        else:
            ordered_rows = self._partitions
            row_idx = self.index
        if col_positions is not None:
            ordered_cols = self._partition_mgr_cls.map_axis_partitions(
                1, ordered_rows, lambda df: df.iloc[:, col_positions]
            )
            col_idx = self.columns[col_positions]
        else:
            ordered_cols = ordered_rows
            col_idx = self.columns
        return self.__constructor__(ordered_cols, row_idx, col_idx)

    @lazy_metadata_decorator(apply_axis=None)
    def copy(self):
        """
        Copy this object.

        Returns
        -------
        PandasDataframe
            A copied version of this object.
        """
        return self.__constructor__(
            self._partitions,
            self.index.copy(),
            self.columns.copy(),
            self._row_lengths,
            self._column_widths,
            self._dtypes,
        )

    @classmethod
    def combine_dtypes(cls, list_of_dtypes, column_names):
        """
        Describe how data types should be combined when they do not match.

        Parameters
        ----------
        list_of_dtypes : list
            A list of pandas Series with the data types.
        column_names : list
            The names of the columns that the data types map to.

        Returns
        -------
        pandas.Series
             A pandas Series containing the finalized data types.
        """
        # Compute dtypes by getting collecting and combining all of the partitions. The
        # reported dtypes from differing rows can be different based on the inference in
        # the limited data seen by each worker. We use pandas to compute the exact dtype
        # over the whole column for each column.
        dtypes = (
            pandas.concat(list_of_dtypes, axis=1)
            .apply(lambda row: find_common_type(row.values), axis=1)
            .squeeze(axis=0)
        )
        dtypes.index = column_names
        return dtypes

    @lazy_metadata_decorator(apply_axis="both")
    def astype(self, col_dtypes):
        """
        Convert the columns dtypes to given dtypes.

        Parameters
        ----------
        col_dtypes : dictionary of {col: dtype,...}
            Where col is the column name and dtype is a NumPy dtype.

        Returns
        -------
        BaseDataFrame
            Dataframe with updated dtypes.
        """
        columns = col_dtypes.keys()
        # Create Series for the updated dtypes
        new_dtypes = self.dtypes.copy()
        for i, column in enumerate(columns):
            dtype = col_dtypes[column]
            if (
                not isinstance(dtype, type(self.dtypes[column]))
                or dtype != self.dtypes[column]
            ):
                # Update the new dtype series to the proper pandas dtype
                try:
                    new_dtype = np.dtype(dtype)
                except TypeError:
                    new_dtype = dtype

                if dtype != np.int32 and new_dtype == np.int32:
                    new_dtypes[column] = np.dtype("int64")
                elif dtype != np.float32 and new_dtype == np.float32:
                    new_dtypes[column] = np.dtype("float64")
                # We cannot infer without computing the dtype if
                elif isinstance(new_dtype, str) and new_dtype == "category":
                    new_dtypes = None
                    break
                else:
                    new_dtypes[column] = new_dtype

        def astype_builder(df):
            """Compute new partition frame with dtypes updated."""
            return df.astype({k: v for k, v in col_dtypes.items() if k in df})

        new_frame = self._partition_mgr_cls.map_partitions(
            self._partitions, astype_builder
        )
        return self.__constructor__(
            new_frame,
            self.index,
            self.columns,
            self._row_lengths,
            self._column_widths,
            new_dtypes,
        )

    # Metadata modification methods
    def add_prefix(self, prefix, axis):
        """
        Add a prefix to the current row or column labels.

        Parameters
        ----------
        prefix : str
            The prefix to add.
        axis : int
            The axis to update.

        Returns
        -------
        PandasDataframe
            A new dataframe with the updated labels.
        """

        def new_labels_mapper(x, prefix=str(prefix)):
            return prefix + str(x)

        if axis == 0:
            return self.rename(new_row_labels=new_labels_mapper)
        return self.rename(new_col_labels=new_labels_mapper)

    def add_suffix(self, suffix, axis):
        """
        Add a suffix to the current row or column labels.

        Parameters
        ----------
        suffix : str
            The suffix to add.
        axis : int
            The axis to update.

        Returns
        -------
        PandasDataframe
            A new dataframe with the updated labels.
        """

        def new_labels_mapper(x, suffix=str(suffix)):
            return str(x) + suffix

        if axis == 0:
            return self.rename(new_row_labels=new_labels_mapper)
        return self.rename(new_col_labels=new_labels_mapper)

    # END Metadata modification methods

    def numeric_columns(self, include_bool=True):
        """
        Return the names of numeric columns in the frame.

        Parameters
        ----------
        include_bool : bool, default: True
            Whether to consider boolean columns as numeric.

        Returns
        -------
        list
            List of column names.
        """
        columns = []
        for col, dtype in zip(self.columns, self.dtypes):
            if is_numeric_dtype(dtype) and (
                include_bool or (not include_bool and dtype != np.bool_)
            ):
                columns.append(col)
        return columns

    def _get_dict_of_block_index(self, axis, indices, are_indices_sorted=False):
        """
        Convert indices to an ordered dict mapping partition (or block) index to internal indices in said partition.

        Parameters
        ----------
        axis : {0, 1}
            The axis along which to get the indices (0 - rows, 1 - columns).
        indices : list of int, slice
            A list of global indices to convert.
        are_indices_sorted : bool, default: False
            Flag indicating whether the `indices` sequence is sorted by ascending or not.
            Note: the internal algorithm requires for the `indices` to be sorted, this
            flag is used for optimization in order to not sort already sorted data.
            Be careful when passing ``True`` for this flag, if the data appears to be unsorted
            with the flag set to ``True`` this would lead to undefined behavior.

        Returns
        -------
        OrderedDict
            A mapping from partition index to list of internal indices which correspond to `indices` in each
            partition.
        """
        # TODO: Support handling of slices with specified 'step'. For now, converting them into a range
        if isinstance(indices, slice) and (
            indices.step is not None and indices.step != 1
        ):
            indices = range(*indices.indices(len(self.axes[axis])))
        # Fasttrack slices
        if isinstance(indices, slice) or (is_range_like(indices) and indices.step == 1):
            # Converting range-like indexer to slice
            indices = slice(indices.start, indices.stop, indices.step)
            if is_full_grab_slice(indices, sequence_len=len(self.axes[axis])):
                return OrderedDict(
                    zip(
                        range(self._partitions.shape[axis]),
                        [slice(None)] * self._partitions.shape[axis],
                    )
                )
            # Empty selection case
            if indices.start == indices.stop and indices.start is not None:
                return OrderedDict()
            if indices.start is None or indices.start == 0:
                last_part, last_idx = list(
                    self._get_dict_of_block_index(axis, [indices.stop]).items()
                )[0]
                dict_of_slices = OrderedDict(
                    zip(range(last_part), [slice(None)] * last_part)
                )
                dict_of_slices.update({last_part: slice(last_idx[0])})
                return dict_of_slices
            elif indices.stop is None or indices.stop >= len(self.axes[axis]):
                first_part, first_idx = list(
                    self._get_dict_of_block_index(axis, [indices.start]).items()
                )[0]
                dict_of_slices = OrderedDict({first_part: slice(first_idx[0], None)})
                num_partitions = np.size(self._partitions, axis=axis)
                part_list = range(first_part + 1, num_partitions)
                dict_of_slices.update(
                    OrderedDict(zip(part_list, [slice(None)] * len(part_list)))
                )
                return dict_of_slices
            else:
                first_part, first_idx = list(
                    self._get_dict_of_block_index(axis, [indices.start]).items()
                )[0]
                last_part, last_idx = list(
                    self._get_dict_of_block_index(axis, [indices.stop]).items()
                )[0]
                if first_part == last_part:
                    return OrderedDict({first_part: slice(first_idx[0], last_idx[0])})
                else:
                    if last_part - first_part == 1:
                        return OrderedDict(
                            # FIXME: this dictionary creation feels wrong - it might not maintain the order
                            {
                                first_part: slice(first_idx[0], None),
                                last_part: slice(None, last_idx[0]),
                            }
                        )
                    else:
                        dict_of_slices = OrderedDict(
                            {first_part: slice(first_idx[0], None)}
                        )
                        part_list = range(first_part + 1, last_part)
                        dict_of_slices.update(
                            OrderedDict(zip(part_list, [slice(None)] * len(part_list)))
                        )
                        dict_of_slices.update({last_part: slice(None, last_idx[0])})
                        return dict_of_slices
        if isinstance(indices, list):
            # Converting python list to numpy for faster processing
            indices = np.array(indices, dtype=np.int64)
        negative_mask = np.less(indices, 0)
        has_negative = np.any(negative_mask)
        if has_negative:
            # We're going to modify 'indices' inplace in a numpy way, so doing a copy/converting indices to numpy.
            indices = (
                indices.copy()
                if isinstance(indices, np.ndarray)
                else np.array(indices, dtype=np.int64)
            )
            indices[negative_mask] = indices[negative_mask] % len(self.axes[axis])
        # If the `indices` array was modified because of the negative indices conversion
        # then the original order was broken and so we have to sort anyway:
        if has_negative or not are_indices_sorted:
            indices = np.sort(indices)
        if axis == 0:
            bins = np.array(self._row_lengths)
        else:
            bins = np.array(self._column_widths)
        # INT_MAX to make sure we don't try to compute on partitions that don't exist.
        cumulative = np.append(bins[:-1].cumsum(), np.iinfo(bins.dtype).max)

        def internal(block_idx, global_index):
            """Transform global index to internal one for given block (identified by its index)."""
            return (
                global_index
                if not block_idx
                else np.subtract(
                    global_index, cumulative[min(block_idx, len(cumulative) - 1) - 1]
                )
            )

        partition_ids = np.digitize(indices, cumulative)
        count_for_each_partition = np.array(
            [(partition_ids == i).sum() for i in range(len(cumulative))]
        ).cumsum()
        # Compute the internal indices and pair those with the partition index.
        # If the first partition has any values we need to return, compute those
        # first to make the list comprehension easier. Otherwise, just append the
        # rest of the values to an empty list.
        if count_for_each_partition[0] > 0:
            first_partition_indices = [
                (0, internal(0, indices[slice(count_for_each_partition[0])]))
            ]
        else:
            first_partition_indices = []
        partition_ids_with_indices = first_partition_indices + [
            (
                i,
                internal(
                    i,
                    indices[
                        slice(
                            count_for_each_partition[i - 1],
                            count_for_each_partition[i],
                        )
                    ],
                ),
            )
            for i in range(1, len(count_for_each_partition))
            if count_for_each_partition[i] > count_for_each_partition[i - 1]
        ]
        return OrderedDict(partition_ids_with_indices)

    @staticmethod
    def _join_index_objects(axis, indexes, how, sort):
        """
        Join the pair of index objects (columns or rows) by a given strategy.

        Unlike Index.join() in pandas, if `axis` is 1, `sort` is False,
        and `how` is "outer", the result will _not_ be sorted.

        Parameters
        ----------
        axis : {0, 1}
            The axis index object to join (0 - rows, 1 - columns).
        indexes : list(Index)
            The indexes to join on.
        how : {'left', 'right', 'inner', 'outer', None}
            The type of join to join to make. If `None` then joined index
            considered to be the first index in the `indexes` list.
        sort : boolean
            Whether or not to sort the joined index.

        Returns
        -------
        (Index, func)
            Joined index with make_reindexer func.
        """
        assert isinstance(indexes, list)

        # define helper functions
        def merge(left_index, right_index):
            """Combine a pair of indices depending on `axis`, `how` and `sort` from outside."""
            if axis == 1 and how == "outer" and not sort:
                return left_index.union(right_index, sort=False)
            else:
                return left_index.join(right_index, how=how, sort=sort)

        # define condition for joining indexes
        all_indices_equal = all(indexes[0].equals(index) for index in indexes[1:])
        do_join_index = how is not None and not all_indices_equal

        # define condition for joining indexes with getting indexers
        need_indexers = (
            axis == 0
            and not all_indices_equal
            and any(not index.is_unique for index in indexes)
        )
        indexers = None

        # perform joining indexes
        if do_join_index:
            if len(indexes) == 2 and need_indexers:
                # in case of count of indexes > 2 we should perform joining all indexes
                # after that get indexers
                # in the fast path we can obtain joined_index and indexers in one call
                indexers = [None, None]
                joined_index, indexers[0], indexers[1] = indexes[0].join(
                    indexes[1], how=how, sort=sort, return_indexers=True
                )
            else:
                joined_index = indexes[0]
                # TODO: revisit for performance
                for index in indexes[1:]:
                    joined_index = merge(joined_index, index)
        else:
            joined_index = indexes[0].copy()

        if need_indexers and indexers is None:
            indexers = [index.get_indexer_for(joined_index) for index in indexes]

        def make_reindexer(do_reindex: bool, frame_idx: int):
            """Create callback that reindexes the dataframe using newly computed index."""
            # the order of the frames must match the order of the indexes
            if not do_reindex:
                return lambda df: df

            if need_indexers:
                assert indexers is not None

                return lambda df: df._reindex_with_indexers(
                    {0: [joined_index, indexers[frame_idx]]},
                    copy=True,
                    allow_dups=True,
                )
            return lambda df: df.reindex(joined_index, axis=axis)

        return joined_index, make_reindexer

    # Internal methods
    # These methods are for building the correct answer in a modular way.
    # Please be careful when changing these!

    def _build_treereduce_func(self, axis, func):
        """
        Properly formats a TreeReduce result so that the partitioning is correct.

        Parameters
        ----------
        axis : int
            The axis along which to apply the function.
        func : callable
            The function to apply.

        Returns
        -------
        callable
            A function to be shipped to the partitions to be executed.

        Notes
        -----
        This should be used for any TreeReduce style operation that results in a
        reduced data dimensionality (dataframe -> series).
        """

        def _tree_reduce_func(df, *args, **kwargs):
            """Tree-reducer function itself executing `func`, presenting the resulting pandas.Series as pandas.DataFrame."""
            series_result = func(df, *args, **kwargs)
            if axis == 0 and isinstance(series_result, pandas.Series):
                # In the case of axis=0, we need to keep the shape of the data
                # consistent with what we have done. In the case of a reduce, the
                # data for axis=0 should be a single value for each column. By
                # transposing the data after we convert to a DataFrame, we ensure that
                # the columns of the result line up with the columns from the data.
                # axis=1 does not have this requirement because the index already will
                # line up with the index of the data based on how pandas creates a
                # DataFrame from a Series.
                result = pandas.DataFrame(series_result).T
                result.index = ["__reduced__"]
            else:
                result = pandas.DataFrame(series_result)
                if isinstance(series_result, pandas.Series):
                    result.columns = ["__reduced__"]
            return result

        return _tree_reduce_func

    def _compute_tree_reduce_metadata(self, axis, new_parts):
        """
        Compute the metadata for the result of reduce function.

        Parameters
        ----------
        axis : int
            The axis on which reduce function was applied.
        new_parts : NumPy 2D array
            Partitions with the result of applied function.

        Returns
        -------
        PandasDataframe
            Modin series (1xN frame) containing the reduced data.
        """
        new_axes, new_axes_lengths = [0, 0], [0, 0]

        new_axes[axis] = ["__reduced__"]
        new_axes[axis ^ 1] = self.axes[axis ^ 1]

        new_axes_lengths[axis] = [1]
        new_axes_lengths[axis ^ 1] = self._axes_lengths[axis ^ 1]

        new_dtypes = None
        result = self.__constructor__(
            new_parts,
            *new_axes,
            *new_axes_lengths,
            new_dtypes,
        )
        return result

    @lazy_metadata_decorator(apply_axis="both")
    def reduce(
        self,
        axis: Union[int, Axis],
        function: Callable,
        dtypes: Optional[str] = None,
    ) -> "PandasDataframe":
        """
        Perform a user-defined aggregation on the specified axis, where the axis reduces down to a singleton. Requires knowledge of the full axis for the reduction.

        Parameters
        ----------
        axis : int or modin.core.dataframe.base.utils.Axis
            The axis to perform the reduce over.
        function : callable(row|col) -> single value
            The reduce function to apply to each column.
        dtypes : str, optional
            The data types for the result. This is an optimization
            because there are functions that always result in a particular data
            type, and this allows us to avoid (re)computing it.

        Returns
        -------
        PandasDataframe
            Modin series (1xN frame) containing the reduced data.

        Notes
        -----
        The user-defined function must reduce to a single value.
        """
        axis = Axis(axis)
        function = self._build_treereduce_func(axis.value, function)
        new_parts = self._partition_mgr_cls.map_axis_partitions(
            axis.value, self._partitions, function
        )
        return self._compute_tree_reduce_metadata(axis.value, new_parts)

    @lazy_metadata_decorator(apply_axis="opposite", axis_arg=0)
    def tree_reduce(
        self,
        axis: Union[int, Axis],
        map_func: Callable,
        reduce_func: Optional[Callable] = None,
        dtypes: Optional[str] = None,
    ) -> "PandasDataframe":
        """
        Apply function that will reduce the data to a pandas Series.

        Parameters
        ----------
        axis : int or modin.core.dataframe.base.utils.Axis
            The axis to perform the tree reduce over.
        map_func : callable(row|col) -> row|col
            Callable function to map the dataframe.
        reduce_func : callable(row|col) -> single value, optional
            Callable function to reduce the dataframe.
            If none, then apply map_func twice.
        dtypes : str, optional
            The data types for the result. This is an optimization
            because there are functions that always result in a particular data
            type, and this allows us to avoid (re)computing it.

        Returns
        -------
        PandasDataframe
            A new dataframe.
        """
        axis = Axis(axis)
        map_func = self._build_treereduce_func(axis.value, map_func)
        if reduce_func is None:
            reduce_func = map_func
        else:
            reduce_func = self._build_treereduce_func(axis.value, reduce_func)

        map_parts = self._partition_mgr_cls.map_partitions(self._partitions, map_func)
        reduce_parts = self._partition_mgr_cls.map_axis_partitions(
            axis.value, map_parts, reduce_func
        )
        return self._compute_tree_reduce_metadata(axis.value, reduce_parts)

    @lazy_metadata_decorator(apply_axis=None)
    def map(self, func: Callable, dtypes: Optional[str] = None) -> "PandasDataframe":
        """
        Perform a function that maps across the entire dataset.

        Parameters
        ----------
        func : callable(row|col|cell) -> row|col|cell
            The function to apply.
        dtypes : dtypes of the result, optional
            The data types for the result. This is an optimization
            because there are functions that always result in a particular data
            type, and this allows us to avoid (re)computing it.

        Returns
        -------
        PandasDataframe
            A new dataframe.
        """
        new_partitions = self._partition_mgr_cls.map_partitions(self._partitions, func)
        if dtypes == "copy":
            dtypes = self._dtypes
        elif dtypes is not None:
            dtypes = pandas.Series(
                [np.dtype(dtypes)] * len(self.columns), index=self.columns
            )
        return self.__constructor__(
            new_partitions,
            self.axes[0],
            self.axes[1],
            self._row_lengths_cache,
            self._column_widths_cache,
            dtypes=dtypes,
        )

    def window(
        self,
        axis: Union[int, Axis],
        reduce_fn: Callable,
        window_size: int,
        result_schema: Optional[Dict[Hashable, type]] = None,
    ) -> "PandasDataframe":
        """
        Apply a sliding window operator that acts as a GROUPBY on each window, and reduces down to a single row (column) per window.

        Parameters
        ----------
        axis : int or modin.core.dataframe.base.utils.Axis
            The axis to slide over.
        reduce_fn : callable(rowgroup|colgroup) -> row|col
            The reduce function to apply over the data.
        window_size : int
            The number of row/columns to pass to the function.
            (The size of the sliding window).
        result_schema : dict, optional
            Mapping from column labels to data types that represents the types of the output dataframe.

        Returns
        -------
        PandasDataframe
            A new PandasDataframe with the reduce function applied over windows of the specified
                axis.

        Notes
        -----
        The user-defined reduce function must reduce each window’s column
        (row if axis=1) down to a single value.
        """
        pass

    @lazy_metadata_decorator(apply_axis="both")
    def fold(self, axis, func):
        """
        Perform a function across an entire axis.

        Parameters
        ----------
        axis : int
            The axis to apply over.
        func : callable
            The function to apply.

        Returns
        -------
        PandasDataframe
            A new dataframe.

        Notes
        -----
        The data shape is not changed (length and width of the table).
        """
        new_partitions = self._partition_mgr_cls.map_axis_partitions(
            axis, self._partitions, func, keep_partitioning=True
        )
        return self.__constructor__(
            new_partitions,
            self.index,
            self.columns,
            self._row_lengths,
            self._column_widths,
        )

    def infer_types(self, columns_list: List[str]) -> "PandasDataframe":
        """
        Determine the compatible type shared by all values in the specified columns, and coerce them to that type.

        Parameters
        ----------
        columns_list : list
            List of column labels to infer and induce types over.

        Returns
        -------
        PandasDataframe
            A new PandasDataframe with the inferred schema.
        """
        pass

    def join(
        self,
        axis: Union[int, Axis],
        condition: Callable,
        other: ModinDataframe,
        join_type: Union[str, JoinType],
    ) -> "PandasDataframe":
        """
        Join this dataframe with the other.

        Parameters
        ----------
        axis : int or modin.core.dataframe.base.utils.Axis
            The axis to perform the join on.
        condition : callable
            Function that determines which rows should be joined. The condition can be a
            simple equality, e.g. "left.col1 == right.col1" or can be arbitrarily complex.
        other : ModinDataframe
            The other data to join with, i.e. the right dataframe.
        join_type : string {"inner", "left", "right", "outer"} or modin.core.dataframe.base.utils.JoinType
            The type of join to perform.

        Returns
        -------
        PandasDataframe
            A new PandasDataframe that is the result of applying the specified join over the two
            dataframes.

        Notes
        -----
        During the join, this dataframe is considered the left, while the other is
        treated as the right.

        Only inner joins, left outer, right outer, and full outer joins are currently supported.
        Support for other join types (e.g. natural join) may be implemented in the future.
        """
        pass

    def rename(
        self,
        new_row_labels: Optional[Union[Dict[Hashable, Hashable], Callable]] = None,
        new_col_labels: Optional[Union[Dict[Hashable, Hashable], Callable]] = None,
        level: Optional[Union[int, List[int]]] = None,
    ) -> "PandasDataframe":
        """
        Replace the row and column labels with the specified new labels.

        Parameters
        ----------
        new_row_labels : dictionary or callable, optional
            Mapping or callable that relates old row labels to new labels.
        new_col_labels : dictionary or callable, optional
            Mapping or callable that relates old col labels to new labels.
        level : int, optional
            Level whose row labels to replace.

        Returns
        -------
        PandasDataframe
            A new PandasDataframe with the new row and column labels.

        Notes
        -----
        If level is not specified, the default behavior is to replace row labels in all levels.
        """
        new_index = self.index.copy()

        def make_label_swapper(label_dict):
            if isinstance(label_dict, dict):
                return lambda label: label_dict.get(label, label)
            return label_dict

        def swap_labels_levels(index_tuple):
            if isinstance(new_row_labels, dict):
                return tuple(new_row_labels.get(label, label) for label in index_tuple)
            return tuple(new_row_labels(label) for label in index_tuple)

        if new_row_labels:
            swap_row_labels = make_label_swapper(new_row_labels)
            if isinstance(self.index, pandas.MultiIndex):
                if level is not None:
                    new_index.set_levels(
                        new_index.levels[level].map(swap_row_labels), level
                    )
                else:
                    new_index = new_index.map(swap_labels_levels)
            else:
                new_index = new_index.map(swap_row_labels)
        new_cols = self.columns.copy()
        if new_col_labels:
            new_cols = new_cols.map(make_label_swapper(new_col_labels))

        def map_fn(df):
            return df.rename(index=new_row_labels, columns=new_col_labels, level=level)

        new_parts = self._partition_mgr_cls.map_partitions(self._partitions, map_fn)
        return self.__constructor__(
            new_parts,
            new_index,
            new_cols,
            self._row_lengths,
            self._column_widths,
            self._dtypes,
        )

    def sort_by(
        self,
        axis: Union[int, Axis],
        columns: Union[str, List[str]],
        ascending: bool = True,
    ) -> "PandasDataframe":
        """
        Logically reorder rows (columns if axis=1) lexicographically by the data in a column or set of columns.

        Parameters
        ----------
        axis : int or modin.core.dataframe.base.utils.Axis
            The axis to perform the sort over.
        columns : string or list
            Column label(s) to use to determine lexicographical ordering.
        ascending : boolean, default: True
            Whether to sort in ascending or descending order.

        Returns
        -------
        PandasDataframe
            A new PandasDataframe sorted into lexicographical order by the specified column(s).
        """
        pass

    @lazy_metadata_decorator(apply_axis="both")
    def filter(self, axis: Union[Axis, int], condition: Callable) -> "PandasDataframe":
        """
        Filter data based on the function provided along an entire axis.

        Parameters
        ----------
        axis : int or modin.core.dataframe.base.utils.Axis
            The axis to filter over.
        condition : callable(row|col) -> bool
            The function to use for the filter. This function should filter the
            data itself.

        Returns
        -------
        PandasDataframe
            A new filtered dataframe.
        """
        axis = Axis(axis)
        assert axis in (
            Axis.ROW_WISE,
            Axis.COL_WISE,
        ), "Axis argument to filter operator must be 0 (rows) or 1 (columns)"

        new_partitions = self._partition_mgr_cls.map_axis_partitions(
            axis.value, self._partitions, condition, keep_partitioning=True
        )
        new_axes, new_lengths = [0, 0], [0, 0]

        new_axes[axis.value] = self.axes[axis.value]
        new_axes[axis.value ^ 1] = self._compute_axis_labels(
            axis.value ^ 1, new_partitions
        )

        new_lengths[axis.value] = self._axes_lengths[axis.value]
        new_lengths[
            axis.value ^ 1
        ] = None  # We do not know what the resulting widths will be

        return self.__constructor__(
            new_partitions,
            *new_axes,
            *new_lengths,
            self.dtypes if axis == 0 else None,
        )

    def filter_by_types(self, types: List[Hashable]) -> "PandasDataframe":
        """
        Allow the user to specify a type or set of types by which to filter the columns.

        Parameters
        ----------
        types : list
            The types to filter columns by.

        Returns
        -------
        PandasDataframe
             A new PandasDataframe from the filter provided.
        """
        return self.mask(
            col_positions=[i for i, dtype in enumerate(self.dtypes) if dtype in types]
        )

    @lazy_metadata_decorator(apply_axis="both")
    def explode(self, axis: Union[int, Axis], func: Callable) -> "PandasDataframe":
        """
        Explode list-like entries along an entire axis.

        Parameters
        ----------
        axis : int or modin.core.dataframe.base.utils.Axis
            The axis specifying how to explode. If axis=1, explode according
            to columns.
        func : callable
            The function to use to explode a single element.

        Returns
        -------
        PandasFrame
            A new filtered dataframe.
        """
        axis = Axis(axis)
        partitions = self._partition_mgr_cls.map_axis_partitions(
            axis.value, self._partitions, func, keep_partitioning=True
        )
        if axis == Axis.COL_WISE:
            new_index = self._compute_axis_labels(0, partitions)
            new_columns = self.columns
        else:
            new_index = self.index
            new_columns = self._compute_axis_labels(1, partitions)
        return self.__constructor__(partitions, new_index, new_columns)

    @lazy_metadata_decorator(apply_axis="both")
    def apply_full_axis(
        self,
        axis,
        func,
        new_index=None,
        new_columns=None,
        dtypes=None,
    ):
        """
        Perform a function across an entire axis.

        Parameters
        ----------
        axis : {0, 1}
            The axis to apply over (0 - rows, 1 - columns).
        func : callable
            The function to apply.
        new_index : list-like, optional
            The index of the result. We may know this in advance,
            and if not provided it must be computed.
        new_columns : list-like, optional
            The columns of the result. We may know this in
            advance, and if not provided it must be computed.
        dtypes : list-like, optional
            The data types of the result. This is an optimization
            because there are functions that always result in a particular data
            type, and allows us to avoid (re)computing it.

        Returns
        -------
        PandasDataframe
            A new dataframe.

        Notes
        -----
        The data shape may change as a result of the function.
        """
        return self.broadcast_apply_full_axis(
            axis=axis,
            func=func,
            new_index=new_index,
            new_columns=new_columns,
            dtypes=dtypes,
            other=None,
        )

    @lazy_metadata_decorator(apply_axis="both")
    def apply_full_axis_select_indices(
        self,
        axis,
        func,
        apply_indices=None,
        numeric_indices=None,
        new_index=None,
        new_columns=None,
        keep_remaining=False,
    ):
        """
        Apply a function across an entire axis for a subset of the data.

        Parameters
        ----------
        axis : int
            The axis to apply over.
        func : callable
            The function to apply.
        apply_indices : list-like, default: None
            The labels to apply over.
        numeric_indices : list-like, default: None
            The indices to apply over.
        new_index : list-like, optional
            The index of the result. We may know this in advance,
            and if not provided it must be computed.
        new_columns : list-like, optional
            The columns of the result. We may know this in
            advance, and if not provided it must be computed.
        keep_remaining : boolean, default: False
            Whether or not to drop the data that is not computed over.

        Returns
        -------
        PandasDataframe
            A new dataframe.
        """
        assert apply_indices is not None or numeric_indices is not None
        # Convert indices to numeric indices
        old_index = self.index if axis else self.columns
        if apply_indices is not None:
            numeric_indices = old_index.get_indexer_for(apply_indices)
        # Get the indices for the axis being applied to (it is the opposite of axis
        # being applied over)
        dict_indices = self._get_dict_of_block_index(axis ^ 1, numeric_indices)
        new_partitions = (
            self._partition_mgr_cls.apply_func_to_select_indices_along_full_axis(
                axis,
                self._partitions,
                func,
                dict_indices,
                keep_remaining=keep_remaining,
            )
        )
        # TODO Infer columns and index from `keep_remaining` and `apply_indices`
        if new_index is None:
            new_index = self.index if axis == 1 else None
        if new_columns is None:
            new_columns = self.columns if axis == 0 else None
        return self.__constructor__(new_partitions, new_index, new_columns, None, None)

    @lazy_metadata_decorator(apply_axis="both")
    def apply_select_indices(
        self,
        axis,
        func,
        apply_indices=None,
        row_labels=None,
        col_labels=None,
        new_index=None,
        new_columns=None,
        keep_remaining=False,
        item_to_distribute=no_default,
    ):
        """
        Apply a function for a subset of the data.

        Parameters
        ----------
        axis : {0, 1}
            The axis to apply over.
        func : callable
            The function to apply.
        apply_indices : list-like, default: None
            The labels to apply over. Must be given if axis is provided.
        row_labels : list-like, default: None
            The row labels to apply over. Must be provided with
            `col_labels` to apply over both axes.
        col_labels : list-like, default: None
            The column labels to apply over. Must be provided
            with `row_labels` to apply over both axes.
        new_index : list-like, optional
            The index of the result. We may know this in advance,
            and if not provided it must be computed.
        new_columns : list-like, optional
            The columns of the result. We may know this in
            advance, and if not provided it must be computed.
        keep_remaining : boolean, default: False
            Whether or not to drop the data that is not computed over.
        item_to_distribute : np.ndarray or scalar, default: no_default
            The item to split up so it can be applied over both axes.

        Returns
        -------
        PandasDataframe
            A new dataframe.
        """
        # TODO Infer columns and index from `keep_remaining` and `apply_indices`
        if new_index is None:
            new_index = self.index if axis == 1 else None
        if new_columns is None:
            new_columns = self.columns if axis == 0 else None
        if axis is not None:
            assert apply_indices is not None
            # Convert indices to numeric indices
            old_index = self.index if axis else self.columns
            numeric_indices = old_index.get_indexer_for(apply_indices)
            # Get indices being applied to (opposite of indices being applied over)
            dict_indices = self._get_dict_of_block_index(axis ^ 1, numeric_indices)
            new_partitions = self._partition_mgr_cls.apply_func_to_select_indices(
                axis,
                self._partitions,
                func,
                dict_indices,
                keep_remaining=keep_remaining,
            )
            # Length objects for new object creation. This is shorter than if..else
            # This object determines the lengths and widths based on the given
            # parameters and builds a dictionary used in the constructor below. 0 gives
            # the row lengths and 1 gives the column widths. Since the dimension of
            # `axis` given may have changed, we currently just recompute it.
            # TODO Determine lengths from current lengths if `keep_remaining=False`
            lengths_objs = {
                axis: [len(apply_indices)]
                if not keep_remaining
                else [self._row_lengths, self._column_widths][axis],
                axis ^ 1: [self._row_lengths, self._column_widths][axis ^ 1],
            }
            return self.__constructor__(
                new_partitions, new_index, new_columns, lengths_objs[0], lengths_objs[1]
            )
        else:
            # We are applying over both axes here, so make sure we have all the right
            # variables set.
            assert row_labels is not None and col_labels is not None
            assert keep_remaining
            assert item_to_distribute is not no_default
            row_partitions_list = self._get_dict_of_block_index(0, row_labels).items()
            col_partitions_list = self._get_dict_of_block_index(1, col_labels).items()
            new_partitions = self._partition_mgr_cls.apply_func_to_indices_both_axis(
                self._partitions,
                func,
                row_partitions_list,
                col_partitions_list,
                item_to_distribute,
                # Passing caches instead of values in order to not trigger shapes recomputation
                # if they are not used inside this function.
                self._row_lengths_cache,
                self._column_widths_cache,
            )
            return self.__constructor__(
                new_partitions,
                new_index,
                new_columns,
                self._row_lengths_cache,
                self._column_widths_cache,
            )

    @lazy_metadata_decorator(apply_axis="both")
    def broadcast_apply(
        self, axis, func, other, join_type="left", preserve_labels=True, dtypes=None
    ):
        """
        Broadcast axis partitions of `other` to partitions of `self` and apply a function.

        Parameters
        ----------
        axis : {0, 1}
            Axis to broadcast over.
        func : callable
            Function to apply.
        other : PandasDataframe
            Modin DataFrame to broadcast.
        join_type : str, default: "left"
            Type of join to apply.
        preserve_labels : bool, default: True
            Whether keep labels from `self` Modin DataFrame or not.
        dtypes : "copy" or None, default: None
            Whether keep old dtypes or infer new dtypes from data.

        Returns
        -------
        PandasDataframe
            New Modin DataFrame.
        """
        # Only sort the indices if they do not match
        (
            left_parts,
            right_parts,
            joined_index,
            partition_sizes_along_axis,
        ) = self._copartition(
            axis, other, join_type, sort=not self.axes[axis].equals(other.axes[axis])
        )
        # unwrap list returned by `copartition`.
        right_parts = right_parts[0]
        new_frame = self._partition_mgr_cls.broadcast_apply(
            axis, func, left_parts, right_parts
        )
        if dtypes == "copy":
            dtypes = self._dtypes
        new_index = self.index
        new_columns = self.columns
        # Pass shape caches instead of values in order to not trigger shape
        # computation.
        new_row_lengths = self._row_lengths_cache
        new_column_widths = self._column_widths_cache
        if not preserve_labels:
            if axis == 1:
                new_columns = joined_index
                new_column_widths = partition_sizes_along_axis
            else:
                new_index = joined_index
                new_row_lengths = partition_sizes_along_axis
        return self.__constructor__(
            new_frame,
            new_index,
            new_columns,
            new_row_lengths,
            new_column_widths,
            dtypes=dtypes,
        )

    def _prepare_frame_to_broadcast(self, axis, indices, broadcast_all):
        """
        Compute the indices to broadcast `self` considering `indices`.

        Parameters
        ----------
        axis : {0, 1}
            Axis to broadcast along.
        indices : dict
            Dict of indices and internal indices of partitions where `self` must
            be broadcasted.
        broadcast_all : bool
            Whether broadcast the whole axis of `self` frame or just a subset of it.

        Returns
        -------
        dict
            Dictionary with indices of partitions to broadcast.

        Notes
        -----
        New dictionary of indices of `self` partitions represents that
        you want to broadcast `self` at specified another partition named `other`. For example,
        Dictionary {key: {key1: [0, 1], key2: [5]}} means, that in `other`[key] you want to
        broadcast [self[key1], self[key2]] partitions and internal indices for `self` must be [[0, 1], [5]]
        """
        if broadcast_all:
            sizes = self._row_lengths if axis else self._column_widths
            return {key: dict(enumerate(sizes)) for key in indices.keys()}
        passed_len = 0
        result_dict = {}
        for part_num, internal in indices.items():
            result_dict[part_num] = self._get_dict_of_block_index(
                axis ^ 1, np.arange(passed_len, passed_len + len(internal))
            )
            passed_len += len(internal)
        return result_dict

    @lazy_metadata_decorator(apply_axis="both")
    def broadcast_apply_select_indices(
        self,
        axis,
        func,
        other,
        apply_indices=None,
        numeric_indices=None,
        keep_remaining=False,
        broadcast_all=True,
        new_index=None,
        new_columns=None,
    ):
        """
        Apply a function to select indices at specified axis and broadcast partitions of `other` Modin DataFrame.

        Parameters
        ----------
        axis : {0, 1}
            Axis to apply function along.
        func : callable
            Function to apply.
        other : PandasDataframe
            Partitions of which should be broadcasted.
        apply_indices : list, default: None
            List of labels to apply (if `numeric_indices` are not specified).
        numeric_indices : list, default: None
            Numeric indices to apply (if `apply_indices` are not specified).
        keep_remaining : bool, default: False
            Whether drop the data that is not computed over or not.
        broadcast_all : bool, default: True
            Whether broadcast the whole axis of right frame to every
            partition or just a subset of it.
        new_index : pandas.Index, optional
            Index of the result. We may know this in advance,
            and if not provided it must be computed.
        new_columns : pandas.Index, optional
            Columns of the result. We may know this in advance,
            and if not provided it must be computed.

        Returns
        -------
        PandasDataframe
            New Modin DataFrame.
        """
        assert (
            apply_indices is not None or numeric_indices is not None
        ), "Indices to apply must be specified!"

        if other is None:
            if apply_indices is None:
                apply_indices = self.axes[axis][numeric_indices]
            return self.apply_select_indices(
                axis=axis,
                func=func,
                apply_indices=apply_indices,
                keep_remaining=keep_remaining,
                new_index=new_index,
                new_columns=new_columns,
            )

        if numeric_indices is None:
            old_index = self.index if axis else self.columns
            numeric_indices = old_index.get_indexer_for(apply_indices)

        dict_indices = self._get_dict_of_block_index(axis ^ 1, numeric_indices)
        broadcasted_dict = other._prepare_frame_to_broadcast(
            axis, dict_indices, broadcast_all=broadcast_all
        )
        new_partitions = self._partition_mgr_cls.broadcast_apply_select_indices(
            axis,
            func,
            self._partitions,
            other._partitions,
            dict_indices,
            broadcasted_dict,
            keep_remaining,
        )

        new_axes = [
            self._compute_axis_labels(i, new_partitions)
            if new_axis is None
            else new_axis
            for i, new_axis in enumerate([new_index, new_columns])
        ]

        return self.__constructor__(new_partitions, *new_axes)

    @lazy_metadata_decorator(apply_axis="both")
    def broadcast_apply_full_axis(
        self,
        axis,
        func,
        other,
        new_index=None,
        new_columns=None,
        apply_indices=None,
        enumerate_partitions=False,
        dtypes=None,
    ):
        """
        Broadcast partitions of `other` Modin DataFrame and apply a function along full axis.

        Parameters
        ----------
        axis : {0, 1}
            Axis to apply over (0 - rows, 1 - columns).
        func : callable
            Function to apply.
        other : PandasDataframe or list
            Modin DataFrame(s) to broadcast.
        new_index : list-like, optional
            Index of the result. We may know this in advance,
            and if not provided it must be computed.
        new_columns : list-like, optional
            Columns of the result. We may know this in
            advance, and if not provided it must be computed.
        apply_indices : list-like, default: None
            Indices of `axis ^ 1` to apply function over.
        enumerate_partitions : bool, default: False
            Whether pass partition index into applied `func` or not.
            Note that `func` must be able to obtain `partition_idx` kwarg.
        dtypes : list-like, default: None
            Data types of the result. This is an optimization
            because there are functions that always result in a particular data
            type, and allows us to avoid (re)computing it.

        Returns
        -------
        PandasDataframe
            New Modin DataFrame.
        """
        if other is not None:
            if not isinstance(other, list):
                other = [other]
            other = [o._partitions for o in other] if len(other) else None

        if apply_indices is not None:
            numeric_indices = self.axes[axis ^ 1].get_indexer_for(apply_indices)
            apply_indices = self._get_dict_of_block_index(
                axis ^ 1, numeric_indices
            ).keys()

        new_partitions = self._partition_mgr_cls.broadcast_axis_partitions(
            axis=axis,
            left=self._partitions,
            right=other,
            apply_func=self._build_treereduce_func(axis, func),
            apply_indices=apply_indices,
            enumerate_partitions=enumerate_partitions,
            keep_partitioning=True,
        )
        # Index objects for new object creation. This is shorter than if..else
        new_axes = [
            self._compute_axis_labels(i, new_partitions)
            if new_axis is None
            else new_axis
            for i, new_axis in enumerate([new_index, new_columns])
        ]
        if dtypes == "copy":
            dtypes = self._dtypes
        elif dtypes is not None:
            dtypes = pandas.Series(
                [np.dtype(dtypes)] * len(new_axes[1]), index=new_axes[1]
            )
        result = self.__constructor__(
            new_partitions,
            *new_axes,
            None,
            None,
            dtypes,
        )
        if new_index is not None:
            result.synchronize_labels(axis=0)
        if new_columns is not None:
            result.synchronize_labels(axis=1)
        return result

    def _copartition(self, axis, other, how, sort, force_repartition=False):
        """
        Copartition two Modin DataFrames.

        Perform aligning of partitions, index and partition blocks.

        Parameters
        ----------
        axis : {0, 1}
            Axis to copartition along (0 - rows, 1 - columns).
        other : PandasDataframe
            Other Modin DataFrame(s) to copartition against.
        how : str
            How to manage joining the index object ("left", "right", etc.).
        sort : bool
            Whether sort the joined index or not.
        force_repartition : bool, default: False
            Whether force the repartitioning or not. By default,
            this method will skip repartitioning if it is possible. This is because
            reindexing is extremely inefficient. Because this method is used to
            `join` or `append`, it is vital that the internal indices match.

        Returns
        -------
        tuple
            Tuple containing:
                1) 2-d NumPy array of aligned left partitions
                2) list of 2-d NumPy arrays of aligned right partitions
                3) joined index along ``axis``
                4) List with sizes of partitions along axis that partitioning
                   was done on. This list will be empty if and only if all
                   the frames are empty.
        """
        if isinstance(other, type(self)):
            other = [other]

        self_index = self.axes[axis]
        others_index = [o.axes[axis] for o in other]
        joined_index, make_reindexer = self._join_index_objects(
            axis, [self_index] + others_index, how, sort
        )

        frames = [self] + other
        non_empty_frames_idx = [
            i for i, o in enumerate(frames) if o._partitions.size != 0
        ]

        # If all frames are empty
        if len(non_empty_frames_idx) == 0:
            return (
                self._partitions,
                [o._partitions for o in other],
                joined_index,
                # There are no partition sizes because the resulting dataframe
                # has no partitions.
                [],
            )

        base_frame_idx = non_empty_frames_idx[0]
        other_frames = frames[base_frame_idx + 1 :]

        # Picking first non-empty frame
        base_frame = frames[non_empty_frames_idx[0]]
        base_index = base_frame.axes[axis]

        # define conditions for reindexing and repartitioning `self` frame
        do_reindex_base = not base_index.equals(joined_index)
        do_repartition_base = force_repartition or do_reindex_base

        # Perform repartitioning and reindexing for `base_frame` if needed.
        # Also define length of base and frames. We will need to know the
        # lengths for alignment.
        if do_repartition_base:
            reindexed_base = base_frame._partition_mgr_cls.map_axis_partitions(
                axis,
                base_frame._partitions,
                make_reindexer(do_reindex_base, base_frame_idx),
            )
            if axis:
                base_lengths = [obj.width() for obj in reindexed_base[0]]
            else:
                base_lengths = [obj.length() for obj in reindexed_base.T[0]]
        else:
            reindexed_base = base_frame._partitions
            base_lengths = self._column_widths if axis else self._row_lengths

        others_lengths = [o._axes_lengths[axis] for o in other_frames]

        # define conditions for reindexing and repartitioning `other` frames
        do_reindex_others = [
            not o.axes[axis].equals(joined_index) for o in other_frames
        ]

        do_repartition_others = [None] * len(other_frames)
        for i in range(len(other_frames)):
            do_repartition_others[i] = (
                force_repartition
                or do_reindex_others[i]
                or others_lengths[i] != base_lengths
            )

        # perform repartitioning and reindexing for `other_frames` if needed
        reindexed_other_list = [None] * len(other_frames)
        for i in range(len(other_frames)):
            if do_repartition_others[i]:
                # indices of others frame start from `base_frame_idx` + 1
                reindexed_other_list[i] = other_frames[
                    i
                ]._partition_mgr_cls.map_axis_partitions(
                    axis,
                    other_frames[i]._partitions,
                    make_reindexer(do_repartition_others[i], base_frame_idx + 1 + i),
                    lengths=base_lengths,
                )
            else:
                reindexed_other_list[i] = other_frames[i]._partitions
        reindexed_frames = (
            [frames[i]._partitions for i in range(base_frame_idx)]
            + [reindexed_base]
            + reindexed_other_list
        )
        return (reindexed_frames[0], reindexed_frames[1:], joined_index, base_lengths)

    @lazy_metadata_decorator(apply_axis="both")
    def binary_op(self, op, right_frame, join_type="outer"):
        """
        Perform an operation that requires joining with another Modin DataFrame.

        Parameters
        ----------
        op : callable
            Function to apply after the join.
        right_frame : PandasDataframe
            Modin DataFrame to join with.
        join_type : str, default: "outer"
            Type of join to apply.

        Returns
        -------
        PandasDataframe
            New Modin DataFrame.
        """
        left_parts, right_parts, joined_index, row_lengths = self._copartition(
            0, right_frame, join_type, sort=True
        )
        # unwrap list returned by `copartition`.
        right_parts = right_parts[0]
        new_frame = self._partition_mgr_cls.binary_operation(
            1, left_parts, lambda l, r: op(l, r), right_parts
        )
        new_columns = self.columns.join(right_frame.columns, how=join_type)
        return self.__constructor__(
            new_frame,
            joined_index,
            new_columns,
            row_lengths,
            column_widths=self._column_widths_cache,
        )

    @lazy_metadata_decorator(apply_axis="both")
    def concat(
        self,
        axis: Union[int, Axis],
        others: Union["PandasDataframe", List["PandasDataframe"]],
        how,
        sort,
    ) -> "PandasDataframe":
        """
        Concatenate `self` with one or more other Modin DataFrames.

        Parameters
        ----------
        axis : int or modin.core.dataframe.base.utils.Axis
            Axis to concatenate over.
        others : list
            List of Modin DataFrames to concatenate with.
        how : str
            Type of join to use for the axis.
        sort : bool
            Whether sort the result or not.

        Returns
        -------
        PandasDataframe
            New Modin DataFrame.
        """
        axis = Axis(axis)
        # Fast path for equivalent columns and partitioning
        if (
            axis == Axis.ROW_WISE
            and all(o.columns.equals(self.columns) for o in others)
            and all(o._column_widths == self._column_widths for o in others)
        ):
            joined_index = self.columns
            left_parts = self._partitions
            right_parts = [o._partitions for o in others]
            new_widths = self._column_widths_cache
        elif (
            axis == Axis.COL_WISE
            and all(o.index.equals(self.index) for o in others)
            and all(o._row_lengths == self._row_lengths for o in others)
        ):
            joined_index = self.index
            left_parts = self._partitions
            right_parts = [o._partitions for o in others]
            new_lengths = self._row_lengths_cache
        else:
            (
                left_parts,
                right_parts,
                joined_index,
                partition_sizes_along_axis,
            ) = self._copartition(
                axis.value ^ 1, others, how, sort, force_repartition=False
            )
            if axis == Axis.COL_WISE:
                new_lengths = partition_sizes_along_axis
            else:
                new_widths = partition_sizes_along_axis
        new_partitions = self._partition_mgr_cls.concat(
            axis.value, left_parts, right_parts
        )
        if axis == Axis.ROW_WISE:
            new_index = self.index.append([other.index for other in others])
            new_columns = joined_index
            # TODO: Can optimize by combining if all dtypes are materialized
            new_dtypes = None
            # If we have already cached the length of each row in at least one
            # of the row's partitions, we can build new_lengths for the new
            # frame. Typically, if we know the length for any partition in a
            # row, we know the length for the first partition in the row. So
            # just check the lengths of the first column of partitions.
            new_lengths = []
            if new_partitions.size > 0:
                for part in new_partitions.T[0]:
                    if part._length_cache is not None:
                        new_lengths.append(part.length())
                    else:
                        new_lengths = None
                        break
        else:
            new_columns = self.columns.append([other.columns for other in others])
            new_index = joined_index
            if self._dtypes is not None and all(o._dtypes is not None for o in others):
                new_dtypes = self.dtypes.append([o.dtypes for o in others])
            else:
                new_dtypes = None
            # If we have already cached the width of each column in at least one
            # of the column's partitions, we can build new_widths for the new
            # frame. Typically, if we know the width for any partition in a
            # column, we know the width for the first partition in the column.
            # So just check the widths of the first row of partitions.
            new_widths = []
            if new_partitions.size > 0:
                for part in new_partitions[0]:
                    if part._width_cache is not None:
                        new_widths.append(part.width())
                    else:
                        new_widths = None
                        break

        return self.__constructor__(
            new_partitions, new_index, new_columns, new_lengths, new_widths, new_dtypes
        )

    def groupby(
        self,
        axis: Union[int, Axis],
        by: Union[str, List[str]],
        operator: Callable,
        result_schema: Optional[Dict[Hashable, type]] = None,
    ) -> "PandasDataframe":
        """
        Generate groups based on values in the input column(s) and perform the specified operation on each.

        Parameters
        ----------
        axis : int or modin.core.dataframe.base.utils.Axis
            The axis to apply the grouping over.
        by : string or list of strings
            One or more column labels to use for grouping.
        operator : callable
            The operation to carry out on each of the groups. The operator is another
            algebraic operator with its own user-defined function parameter, depending
            on the output desired by the user.
        result_schema : dict, optional
            Mapping from column labels to data types that represents the types of the output dataframe.

        Returns
        -------
        PandasDataframe
            A new PandasDataframe containing the groupings specified, with the operator
                applied to each group.

        Notes
        -----
        No communication between groups is allowed in this algebra implementation.

        The number of rows (columns if axis=1) returned by the user-defined function
        passed to the groupby may be at most the number of rows in the group, and
        may be as small as a single row.

        Unlike the pandas API, an intermediate “GROUP BY” object is not present in this
        algebra implementation.
        """
        pass

    @lazy_metadata_decorator(apply_axis="opposite", axis_arg=0)
    def groupby_reduce(
        self,
        axis,
        by,
        map_func,
        reduce_func,
        new_index=None,
        new_columns=None,
        apply_indices=None,
    ):
        """
        Groupby another Modin DataFrame dataframe and aggregate the result.

        Parameters
        ----------
        axis : {0, 1}
            Axis to groupby and aggregate over.
        by : PandasDataframe or None
            A Modin DataFrame to group by.
        map_func : callable
            Map component of the aggregation.
        reduce_func : callable
            Reduce component of the aggregation.
        new_index : pandas.Index, optional
            Index of the result. We may know this in advance,
            and if not provided it must be computed.
        new_columns : pandas.Index, optional
            Columns of the result. We may know this in advance,
            and if not provided it must be computed.
        apply_indices : list-like, default: None
            Indices of `axis ^ 1` to apply groupby over.

        Returns
        -------
        PandasDataframe
            New Modin DataFrame.
        """
        by_parts = by if by is None else by._partitions
        if by is None:
            self._propagate_index_objs(axis=0)

        if apply_indices is not None:
            numeric_indices = self.axes[axis ^ 1].get_indexer_for(apply_indices)
            apply_indices = list(
                self._get_dict_of_block_index(axis ^ 1, numeric_indices).keys()
            )

        new_partitions = self._partition_mgr_cls.groupby_reduce(
            axis, self._partitions, by_parts, map_func, reduce_func, apply_indices
        )
        new_axes = [
            self._compute_axis_labels(i, new_partitions)
            if new_axis is None
            else new_axis
            for i, new_axis in enumerate([new_index, new_columns])
        ]

        return self.__constructor__(new_partitions, *new_axes)

    @classmethod
    def from_pandas(cls, df):
        """
        Create a Modin DataFrame from a pandas DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            A pandas DataFrame.

        Returns
        -------
        PandasDataframe
            New Modin DataFrame.
        """
        new_index = df.index
        new_columns = df.columns
        new_dtypes = df.dtypes
        new_frame, new_lengths, new_widths = cls._partition_mgr_cls.from_pandas(
            df, True
        )
        return cls(
            new_frame,
            new_index,
            new_columns,
            new_lengths,
            new_widths,
            dtypes=new_dtypes,
        )

    @classmethod
    def from_arrow(cls, at):
        """
        Create a Modin DataFrame from an Arrow Table.

        Parameters
        ----------
        at : pyarrow.table
            Arrow Table.

        Returns
        -------
        PandasDataframe
            New Modin DataFrame.
        """
        new_frame, new_lengths, new_widths = cls._partition_mgr_cls.from_arrow(
            at, return_dims=True
        )
        new_columns = Index.__new__(Index, data=at.column_names, dtype="O")
        new_index = Index.__new__(RangeIndex, data=range(at.num_rows))
        new_dtypes = pandas.Series(
            [cls._arrow_type_to_dtype(col.type) for col in at.columns],
            index=at.column_names,
        )
        return cls(
            partitions=new_frame,
            index=new_index,
            columns=new_columns,
            row_lengths=new_lengths,
            column_widths=new_widths,
            dtypes=new_dtypes,
        )

    @classmethod
    def _arrow_type_to_dtype(cls, arrow_type):
        """
        Convert an arrow data type to a pandas data type.

        Parameters
        ----------
        arrow_type : arrow dtype
            Arrow data type to be converted to a pandas data type.

        Returns
        -------
        object
            Any dtype compatible with pandas.
        """
        import pyarrow

        try:
            res = arrow_type.to_pandas_dtype()
        # Conversion to pandas is not implemented for some arrow types,
        # perform manual conversion for them:
        except NotImplementedError:
            if pyarrow.types.is_time(arrow_type):
                res = np.dtype(datetime.time)
            else:
                raise

        if not isinstance(res, (np.dtype, str)):
            return np.dtype(res)
        return res

    @lazy_metadata_decorator(apply_axis="both")
    def to_pandas(self):
        """
        Convert this Modin DataFrame to a pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
        """
        df = self._partition_mgr_cls.to_pandas(self._partitions)
        if df.empty:
            df = pandas.DataFrame(columns=self.columns, index=self.index)
        else:
            for axis in [0, 1]:
                ErrorMessage.catch_bugs_and_request_email(
                    not df.axes[axis].equals(self.axes[axis]),
                    f"Internal and external indices on axis {axis} do not match.",
                )
            df.index = self.index
            df.columns = self.columns

        return df

    def to_numpy(self, **kwargs):
        """
        Convert this Modin DataFrame to a NumPy array.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to be passed in `to_numpy`.

        Returns
        -------
        np.ndarray
        """
        return self._partition_mgr_cls.to_numpy(self._partitions, **kwargs)

    @lazy_metadata_decorator(apply_axis=None, transpose=True)
    def transpose(self):
        """
        Transpose the index and columns of this Modin DataFrame.

        Reflect this Modin DataFrame over its main diagonal
        by writing rows as columns and vice-versa.

        Returns
        -------
        PandasDataframe
            New Modin DataFrame.
        """
        new_partitions = self._partition_mgr_cls.lazy_map_partitions(
            self._partitions, lambda df: df.T
        ).T
        if self._dtypes is not None:
            new_dtypes = pandas.Series(
                np.full(len(self.index), find_common_type(self.dtypes.values)),
                index=self.index,
            )
        else:
            new_dtypes = None
        return self.__constructor__(
            new_partitions,
            self.columns,
            self.index,
            self._column_widths,
            self._row_lengths,
            dtypes=new_dtypes,
        )

    def finalize(self):
        """
        Perform all deferred calls on partitions.

        This makes `self` Modin Dataframe independent of a history of queries
        that were used to build it.
        """
        self._partition_mgr_cls.finalize(self._partitions)

    def __dataframe__(self, nan_as_null: bool = False, allow_copy: bool = True):
        """
        Get a Modin DataFrame that implements the dataframe exchange protocol.

        See more about the protocol in https://data-apis.org/dataframe-protocol/latest/index.html.

        Parameters
        ----------
        nan_as_null : bool, default: False
            A keyword intended for the consumer to tell the producer
            to overwrite null values in the data with ``NaN`` (or ``NaT``).
            This currently has no effect; once support for nullable extension
            dtypes is added, this value should be propagated to columns.
        allow_copy : bool, default: True
            A keyword that defines whether or not the library is allowed
            to make a copy of the data. For example, copying data would be necessary
            if a library supports strided buffers, given that this protocol
            specifies contiguous buffers. Currently, if the flag is set to ``False``
            and a copy is needed, a ``RuntimeError`` will be raised.

        Returns
        -------
        ProtocolDataframe
            A dataframe object following the dataframe protocol specification.
        """
        from modin.core.dataframe.pandas.exchange.dataframe_protocol.dataframe import (
            PandasProtocolDataframe,
        )

        return PandasProtocolDataframe(
            self, nan_as_null=nan_as_null, allow_copy=allow_copy
        )

    @classmethod
    def from_dataframe(cls, df: "ProtocolDataframe") -> "PandasDataframe":
        """
        Convert a DataFrame implementing the dataframe exchange protocol to a Core Modin Dataframe.

        See more about the protocol in https://data-apis.org/dataframe-protocol/latest/index.html.

        Parameters
        ----------
        df : ProtocolDataframe
            The DataFrame object supporting the dataframe exchange protocol.

        Returns
        -------
        PandasDataframe
            A new Core Modin Dataframe object.
        """
        if type(df) == cls:
            return df

        if not hasattr(df, "__dataframe__"):
            raise ValueError(
                "`df` does not support DataFrame exchange protocol, i.e. `__dataframe__` method"
            )

        from modin.core.dataframe.pandas.exchange.dataframe_protocol.from_dataframe import (
            from_dataframe_to_pandas,
        )

        ErrorMessage.default_to_pandas(message="`from_dataframe`")
        pandas_df = from_dataframe_to_pandas(df)
        return cls.from_pandas(pandas_df)
