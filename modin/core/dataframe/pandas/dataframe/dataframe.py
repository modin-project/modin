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
from pandas.api.types import is_object_dtype
from pandas.core.indexes.api import Index, RangeIndex
from pandas.core.dtypes.common import (
    is_numeric_dtype,
    is_list_like,
    is_categorical_dtype,
)
from pandas._libs.lib import no_default
from typing import List, Hashable, Optional, Callable, Union, Dict, TYPE_CHECKING

from modin.config import Engine
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.core.storage_formats.pandas.utils import get_length_list
from modin.error_message import ErrorMessage
from modin.core.storage_formats.pandas.parsers import (
    find_common_type_cat as find_common_type,
)
from modin.core.dataframe.base.dataframe.dataframe import ModinDataframe
from modin.core.dataframe.base.dataframe.utils import (
    Axis,
    JoinType,
)
from modin.core.dataframe.pandas.dataframe.utils import (
    build_sort_functions,
    lazy_metadata_decorator,
)
from modin.core.dataframe.pandas.metadata import (
    ModinDtypes,
    ModinIndex,
    LazyProxyCategoricalDtype,
)

if TYPE_CHECKING:
    from modin.core.dataframe.base.interchange.dataframe_protocol.dataframe import (
        ProtocolDataframe,
    )
    from pandas._typing import npt

from modin.pandas.indexing import is_range_like
from modin.pandas.utils import is_full_grab_slice, check_both_not_none
from modin.logging import ClassLogger
from modin.utils import MODIN_UNNAMED_SERIES_LABEL


class PandasDataframe(ClassLogger):
    """
    An abstract class that represents the parent class for any pandas storage format dataframe class.

    This class provides interfaces to run operations on dataframe partitions.

    Parameters
    ----------
    partitions : np.ndarray
        A 2D NumPy array of partitions.
    index : sequence or callable, optional
        The index for the dataframe. Converted to a ``pandas.Index``.
        Is computed from partitions on demand if not specified.
        If ``callable() -> (pandas.Index, list of row lengths or None)`` type,
        then the calculation will be delayed until `self.index` is called.
    columns : sequence, optional
        The columns object for the dataframe. Converted to a ``pandas.Index``.
        Is computed from partitions on demand if not specified.
    row_lengths : list, optional
        The length of each partition in the rows. The "height" of
        each of the block partitions. Is computed if not provided.
    column_widths : list, optional
        The width of each partition in the columns. The "width" of
        each of the block partitions. Is computed if not provided.
    dtypes : pandas.Series or callable, optional
        The data types for the dataframe columns.
    """

    _partition_mgr_cls = None
    _query_compiler_cls = PandasQueryCompiler
    # These properties flag whether or not we are deferring the metadata synchronization
    _deferred_index = False
    _deferred_column = False

    @pandas.util.cache_readonly
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
        index=None,
        columns=None,
        row_lengths=None,
        column_widths=None,
        dtypes=None,
    ):
        self._partitions = partitions
        self.set_index_cache(index)
        self.set_columns_cache(columns)
        self._row_lengths_cache = row_lengths
        self._column_widths_cache = column_widths
        self.set_dtypes_cache(dtypes)

        self._validate_axes_lengths()
        self._filter_empties(compute_metadata=False)

    def _validate_axes_lengths(self):
        """Validate that labels are split correctly if split is known."""
        if self._row_lengths_cache is not None and len(self.index) > 0:
            # An empty frame can have 0 rows but a nonempty index. If the frame
            # does have rows, the number of rows must equal the size of the
            # index.
            num_rows = sum(self._row_lengths_cache)
            if num_rows > 0:
                ErrorMessage.catch_bugs_and_request_email(
                    num_rows != len(self.index),
                    f"Row lengths: {num_rows} != {len(self.index)}",
                )
            ErrorMessage.catch_bugs_and_request_email(
                any(val < 0 for val in self._row_lengths_cache),
                f"Row lengths cannot be negative: {self._row_lengths_cache}",
            )
        if self._column_widths_cache is not None and len(self.columns) > 0:
            # An empty frame can have 0 column but a nonempty column index. If
            # the frame does have columns, the number of columns must equal the
            # size of the columns.
            num_columns = sum(self._column_widths_cache)
            if num_columns > 0:
                ErrorMessage.catch_bugs_and_request_email(
                    num_columns != len(self.columns),
                    f"Column widths: {num_columns} != {len(self.columns)}",
                )
            ErrorMessage.catch_bugs_and_request_email(
                any(val < 0 for val in self._column_widths_cache),
                f"Column widths cannot be negative: {self._column_widths_cache}",
            )

    @property
    def row_lengths(self):
        """
        Compute the row partitions lengths if they are not cached.

        Returns
        -------
        list
            A list of row partitions lengths.
        """
        if self._row_lengths_cache is None:
            if len(self._partitions.T) > 0:
                row_parts = self._partitions.T[0]
                self._row_lengths_cache = [part.length() for part in row_parts]
            else:
                self._row_lengths_cache = []
        return self._row_lengths_cache

    @property
    def column_widths(self):
        """
        Compute the column partitions widths if they are not cached.

        Returns
        -------
        list
            A list of column partitions widths.
        """
        if self._column_widths_cache is None:
            if len(self._partitions) > 0:
                col_parts = self._partitions[0]
                self._column_widths_cache = [part.width() for part in col_parts]
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
        return [self.row_lengths, self.column_widths]

    def _set_axis_lengths_cache(self, value, axis=0):
        """
        Set the row/column lengths cache for the specified axis.

        Parameters
        ----------
        value : list of ints
        axis : int, default: 0
            0 for row lengths and 1 for column widths.
        """
        if axis == 0:
            self._row_lengths_cache = value
        else:
            self._column_widths_cache = value

    @property
    def has_dtypes_cache(self):
        """
        Check if the dtypes cache exists.

        Returns
        -------
        bool
        """
        return self._dtypes is not None

    @property
    def has_materialized_dtypes(self):
        """
        Check if dataframe has materialized index cache.

        Returns
        -------
        bool
        """
        return self.has_dtypes_cache and self._dtypes.is_materialized

    def copy_dtypes_cache(self):
        """
        Copy the dtypes cache.

        Returns
        -------
        pandas.Series, callable or None
            If there is an pandas.Series in the cache, then copying occurs.
        """
        dtypes_cache = None
        if self.has_dtypes_cache:
            dtypes_cache = self._dtypes.copy()
        return dtypes_cache

    def _maybe_update_proxies(self, dtypes, new_parent=None):
        """
        Update lazy proxies stored inside of `dtypes` with a new parent inplace.

        Parameters
        ----------
        dtypes : pandas.Series, ModinDtypes or callable
        new_parent : object, optional
            A new parent to link the proxies to. If not specified
            will consider the `self` to be a new parent.
        """
        new_parent = new_parent or self
        if isinstance(dtypes, pandas.Series) or (
            isinstance(dtypes, ModinDtypes) and dtypes.is_materialized
        ):
            for key, value in dtypes.items():
                if isinstance(value, LazyProxyCategoricalDtype):
                    dtypes[key] = value._update_proxy(new_parent, column_name=key)

    def set_dtypes_cache(self, dtypes):
        """
        Set dtypes cache.

        Parameters
        ----------
        dtypes : pandas.Series, ModinDtypes or callable
        """
        self._maybe_update_proxies(dtypes)
        if isinstance(dtypes, ModinDtypes) or dtypes is None:
            self._dtypes = dtypes
        else:
            self._dtypes = ModinDtypes(dtypes)

    @property
    def dtypes(self):
        """
        Compute the data types if they are not cached.

        Returns
        -------
        pandas.Series
            A pandas Series containing the data types for this dataframe.
        """
        if self.has_dtypes_cache:
            dtypes = self._dtypes.get()
        else:
            dtypes = self._compute_dtypes()
            self.set_dtypes_cache(dtypes)
        return dtypes

    def _compute_dtypes(self, columns=None):
        """
        Compute the data types via TreeReduce pattern for the specified columns.

        Parameters
        ----------
        columns : list-like, default: None
            Columns to compute dtypes for. If not specified compute dtypes
            for all the columns in the dataframe.

        Returns
        -------
        pandas.Series
            A pandas Series containing the data types for this dataframe.
        """

        def dtype_builder(df):
            return df.apply(lambda col: find_common_type(col.values), axis=0)

        if columns is not None:
            # Sorting positions to request columns in the order they're stored (it's more efficient)
            numeric_indices = sorted(self.columns.get_indexer_for(columns))
            obj = self._take_2d_positional(col_positions=numeric_indices)
        else:
            obj = self

        # For now we will use a pandas Series for the dtypes.
        if len(obj.columns) > 0:
            dtypes = (
                obj.tree_reduce(0, lambda df: df.dtypes, dtype_builder)
                .to_pandas()
                .iloc[0]
            )
        else:
            dtypes = pandas.Series([])
        # reset name to None because we use MODIN_UNNAMED_SERIES_LABEL internally
        dtypes.name = None
        return dtypes

    _index_cache = None
    _columns_cache = None

    def set_index_cache(self, index):
        """
        Set index cache.

        Parameters
        ----------
        index : sequence, callable or None
        """
        if isinstance(index, ModinIndex) or index is None:
            self._index_cache = index
        else:
            self._index_cache = ModinIndex(index)

    def set_columns_cache(self, columns):
        """
        Set columns cache.

        Parameters
        ----------
        columns : sequence, callable or None
        """
        if isinstance(columns, ModinIndex) or columns is None:
            self._columns_cache = columns
        else:
            self._columns_cache = ModinIndex(columns)

    def set_axis_cache(self, value, axis=0):
        """
        Set cache for the specified axis (index or columns).

        Parameters
        ----------
        value : sequence, callable or None
        axis : int, default: 0
        """
        if axis == 0:
            self.set_index_cache(value)
        else:
            self.set_columns_cache(value)

    @property
    def has_index_cache(self):
        """
        Check if the index cache exists.

        Returns
        -------
        bool
        """
        return self._index_cache is not None

    def copy_index_cache(self):
        """
        Copy the index cache.

        Returns
        -------
        pandas.Index, callable or None
            If there is an pandas.Index in the cache, then copying occurs.
        """
        idx_cache = self._index_cache
        if self.has_index_cache:
            idx_cache = self._index_cache.copy()
        return idx_cache

    @property
    def has_columns_cache(self):
        """
        Check if the columns cache exists.

        Returns
        -------
        bool
        """
        return self._columns_cache is not None

    def copy_columns_cache(self):
        """
        Copy the columns cache.

        Returns
        -------
        pandas.Index or None
            If there is an pandas.Index in the cache, then copying occurs.
        """
        columns_cache = self._columns_cache
        if columns_cache is not None:
            columns_cache = columns_cache.copy()
        return columns_cache

    def copy_axis_cache(self, axis=0):
        """
        Copy the axis cache (index or columns).

        Parameters
        ----------
        axis : int, default: 0

        Returns
        -------
        pandas.Index, callable or None
            If there is an pandas.Index in the cache, then copying occurs.
        """
        if axis == 0:
            return self.copy_index_cache()
        else:
            return self.copy_columns_cache()

    @property
    def has_materialized_index(self):
        """
        Check if dataframe has materialized index cache.

        Returns
        -------
        bool
        """
        return self.has_index_cache and self._index_cache.is_materialized

    @property
    def has_materialized_columns(self):
        """
        Check if dataframe has materialized columns cache.

        Returns
        -------
        bool
        """
        return self.has_columns_cache and self._columns_cache.is_materialized

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
        new_labels = (
            ModinIndex(new_labels)
            if not isinstance(new_labels, ModinIndex)
            else new_labels
        )
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
        if self.has_index_cache:
            index, row_lengths = self._index_cache.get(return_lengths=True)
        else:
            index, row_lengths = self._compute_axis_labels_and_lengths(0)
            self.set_index_cache(index)
        if self._row_lengths_cache is None:
            self._row_lengths_cache = row_lengths
        return index

    def _get_columns(self):
        """
        Get the columns from the cache object.

        Returns
        -------
        pandas.Index
            An index object containing the column labels.
        """
        if self.has_columns_cache:
            columns, column_widths = self._columns_cache.get(return_lengths=True)
        else:
            columns, column_widths = self._compute_axis_labels_and_lengths(1)
            self.set_columns_cache(columns)
        if self._column_widths_cache is None:
            self._column_widths_cache = column_widths
        return columns

    def _set_index(self, new_index):
        """
        Replace the current row labels with new labels.

        Parameters
        ----------
        new_index : list-like
            The new row labels.
        """
        if self.has_materialized_index:
            new_index = self._validate_set_axis(new_index, self._index_cache)
        self.set_index_cache(new_index)
        self.synchronize_labels(axis=0)

    def _set_columns(self, new_columns):
        """
        Replace the current column labels with new labels.

        Parameters
        ----------
        new_columns : list-like
           The new column labels.
        """
        if self.has_materialized_columns:
            new_columns = self._validate_set_axis(new_columns, self._columns_cache)
            if self.has_materialized_dtypes:
                self.dtypes.index = new_columns
        self.set_columns_cache(new_columns)
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

    def get_axis(self, axis: int = 0) -> pandas.Index:
        """
        Get index object for the requested axis.

        Parameters
        ----------
        axis : {0, 1}, default: 0

        Returns
        -------
        pandas.Index
        """
        return self.index if axis == 0 else self.columns

    def _compute_axis_labels_and_lengths(self, axis: int, partitions=None):
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
        List of int
            Size of partitions alongside specified `axis`.
        """
        if partitions is None:
            partitions = self._partitions
        new_index, internal_idx = self._partition_mgr_cls.get_indices(axis, partitions)
        return new_index, list(map(len, internal_idx))

    def _filter_empties(self, compute_metadata=True):
        """
        Remove empty partitions from `self._partitions` to avoid triggering excess computation.

        Parameters
        ----------
        compute_metadata : bool, default: True
            Trigger the computations for partition sizes and labels if they're not done already.
        """
        if not compute_metadata and (
            not self.has_materialized_index
            or not self.has_materialized_columns
            or self._row_lengths_cache is None
            or self._column_widths_cache is None
        ):
            # do not trigger the computations
            return

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
                    if j < len(self.column_widths) and self.column_widths[j] != 0
                ]
                for i in range(len(self._partitions))
                if i < len(self.row_lengths) and self.row_lengths[i] != 0
            ]
        )
        self._column_widths_cache = [w for w in self.column_widths if w != 0]
        self._row_lengths_cache = [r for r in self.row_lengths if r != 0]

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
            cum_row_lengths = np.cumsum([0] + self.row_lengths)
        if axis is None or axis == 1:
            cum_col_widths = np.cumsum([0] + self.column_widths)

        if axis is None:

            def apply_idx_objs(df, idx, cols):
                return df.set_axis(idx, axis="index").set_axis(cols, axis="columns")

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
                            length=self.row_lengths[i],
                            width=self.column_widths[j],
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
                return df.set_axis(idx, axis="index")

            self._partitions = np.array(
                [
                    [
                        self._partitions[i][j].add_to_apply_calls(
                            apply_idx_objs,
                            idx=self.index[
                                slice(cum_row_lengths[i], cum_row_lengths[i + 1])
                            ],
                            length=self.row_lengths[i],
                            width=self.column_widths[j],
                        )
                        for j in range(len(self._partitions[i]))
                    ]
                    for i in range(len(self._partitions))
                ]
            )
            self._deferred_index = False
        elif axis == 1:

            def apply_idx_objs(df, cols):
                return df.set_axis(cols, axis="columns")

            self._partitions = np.array(
                [
                    [
                        self._partitions[i][j].add_to_apply_calls(
                            apply_idx_objs,
                            cols=self.columns[
                                slice(cum_col_widths[j], cum_col_widths[j + 1])
                            ],
                            length=self.row_lengths[i],
                            width=self.column_widths[j],
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
    def take_2d_labels_or_positional(
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
                "Both row_labels and row_positions were provided - "
                + "please provide only one of row_labels and row_positions."
            )
        if check_both_not_none(col_labels, col_positions):
            raise ValueError(
                "Both col_labels and col_positions were provided - "
                + "please provide only one of col_labels and col_positions."
            )

        if row_labels is not None:
            # Get numpy array of positions of values from `row_labels`
            if isinstance(self.index, pandas.MultiIndex):
                row_positions = np.zeros(len(row_labels), dtype="int64")
                # we can't use .get_locs(row_labels) because the function
                # requires a different format for row_labels
                for idx, label in enumerate(row_labels):
                    if isinstance(label, str):
                        label = [label]
                    # get_loc can return slice that _take_2d_positional can't handle
                    row_positions[idx] = self.index.get_locs(label)[0]
            else:
                row_positions = self.index.get_indexer_for(row_labels)

        if col_labels is not None:
            # Get numpy array of positions of values from `col_labels`
            if isinstance(self.columns, pandas.MultiIndex):
                col_positions = np.zeros(len(col_labels), dtype="int64")
                # we can't use .get_locs(col_labels) because the function
                # requires a different format for row_labels
                for idx, label in enumerate(col_labels):
                    if isinstance(label, str):
                        label = [label]
                    # get_loc can return slice that _take_2d_positional can't handle
                    col_positions[idx] = self.columns.get_locs(label)[0]
            else:
                col_positions = self.columns.get_indexer_for(col_labels)

        return self._take_2d_positional(row_positions, col_positions)

    def _get_sorted_positions(self, positions):
        """
        Sort positions if necessary.

        Parameters
        ----------
        positions : Sequence[int]

        Returns
        -------
        Sequence[int]
        """
        # Helper for take_2d_positional
        if is_range_like(positions) and positions.step > 0:
            sorted_positions = positions
        else:
            sorted_positions = np.sort(positions)
        return sorted_positions

    def _get_new_lengths(self, partitions_dict, *, axis: int) -> List[int]:
        """
        Find lengths of new partitions.

        Parameters
        ----------
        partitions_dict : dict
        axis : int

        Returns
        -------
        list[int]
        """
        # Helper for take_2d_positional
        if axis == 0:
            axis_lengths = self.row_lengths
        else:
            axis_lengths = self.column_widths

        new_lengths = [
            len(
                # Row lengths for slice are calculated as the length of the slice
                # on the partition. Often this will be the same length as the current
                # length, but sometimes it is different, thus the extra calculation.
                range(*part_indexer.indices(axis_lengths[part_idx]))
                if isinstance(part_indexer, slice)
                else part_indexer
            )
            for part_idx, part_indexer in partitions_dict.items()
        ]
        return new_lengths

    def _get_new_index_obj(
        self, positions, sorted_positions, axis: int
    ) -> "tuple[pandas.Index, slice | npt.NDArray[np.intp]]":
        """
        Find the new Index object for take_2d_positional result.

        Parameters
        ----------
        positions : Sequence[int]
        sorted_positions : Sequence[int]
        axis : int

        Returns
        -------
        pandas.Index
        slice or Sequence[int]
        """
        # Helper for take_2d_positional
        # Use the slice to calculate the new columns
        if axis == 0:
            idx = self.index
        else:
            idx = self.columns

        # TODO: Support fast processing of negative-step ranges
        if is_range_like(positions) and positions.step > 0:
            # pandas Index is more likely to preserve its metadata if the indexer
            #  is slice
            monotonic_idx = slice(positions.start, positions.stop, positions.step)
        else:
            monotonic_idx = np.asarray(sorted_positions, dtype=np.intp)

        new_idx = idx[monotonic_idx]
        return new_idx, monotonic_idx

    def _take_2d_positional(
        self,
        row_positions: Optional[List[int]] = None,
        col_positions: Optional[List[int]] = None,
    ) -> "PandasDataframe":
        """
        Lazily select columns or rows from given indices.

        Parameters
        ----------
        row_positions : list-like of ints, optional
            The row positions to extract.
        col_positions : list-like of ints, optional
            The column positions to extract.

        Returns
        -------
        PandasDataframe
             A new PandasDataframe from the mask provided.
        """
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
                    extra_log="Mask takes only list-like numeric indexers, "
                    + f"received: {type(indexer)}",
                )
            indexers.append(indexer)
        row_positions, col_positions = indexers

        if col_positions is None and row_positions is None:
            return self.copy()

        sorted_row_positions = sorted_col_positions = None

        if row_positions is not None:
            sorted_row_positions = self._get_sorted_positions(row_positions)
            # Get dict of row_parts as {row_index: row_internal_indices}
            row_partitions_dict = self._get_dict_of_block_index(
                0, sorted_row_positions, are_indices_sorted=True
            )
            new_row_lengths = self._get_new_lengths(row_partitions_dict, axis=0)
            new_index, _ = self._get_new_index_obj(
                row_positions, sorted_row_positions, axis=0
            )
        else:
            row_partitions_dict = {i: slice(None) for i in range(len(self._partitions))}
            new_row_lengths = self._row_lengths_cache
            new_index = self.copy_index_cache()

        if col_positions is not None:
            sorted_col_positions = self._get_sorted_positions(col_positions)
            # Get dict of col_parts as {col_index: col_internal_indices}
            col_partitions_dict = self._get_dict_of_block_index(
                1, sorted_col_positions, are_indices_sorted=True
            )
            new_col_widths = self._get_new_lengths(col_partitions_dict, axis=1)
            new_columns, monotonic_col_idx = self._get_new_index_obj(
                col_positions, sorted_col_positions, axis=1
            )

            ErrorMessage.catch_bugs_and_request_email(
                failure_condition=sum(new_col_widths) != len(new_columns),
                extra_log=f"{sum(new_col_widths)} != {len(new_columns)}.\n"
                + f"{col_positions}\n{self.column_widths}\n{col_partitions_dict}",
            )

            if self.has_materialized_dtypes:
                new_dtypes = self.dtypes.iloc[monotonic_col_idx]
            else:
                new_dtypes = None
        else:
            col_partitions_dict = {
                i: slice(None) for i in range(len(self._partitions.T))
            }
            new_col_widths = self._column_widths_cache
            new_columns = self.copy_columns_cache()
            new_dtypes = self.copy_dtypes_cache()

        new_partitions = np.array(
            [
                [
                    self._partitions[row_idx][col_idx].mask(
                        row_internal_indices, col_internal_indices
                    )
                    for col_idx, col_internal_indices in col_partitions_dict.items()
                ]
                for row_idx, row_internal_indices in row_partitions_dict.items()
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

        return self._maybe_reorder_labels(
            intermediate,
            row_positions,
            sorted_row_positions,
            col_positions,
            sorted_col_positions,
        )

    def _maybe_reorder_labels(
        self,
        intermediate: "PandasDataframe",
        row_positions,
        sorted_row_positions,
        col_positions,
        sorted_col_positions,
    ) -> "PandasDataframe":
        """
        Call re-order labels on take_2d_labels_or_positional result if necessary.

        Parameters
        ----------
        intermediate : PandasDataFrame
        row_positions : list-like of ints, optional
            The row positions to extract.
        sorted_row_positions : list-like of ints, optional
            Sorted version of row_positions.
        col_positions : list-like of ints, optional
            The column positions to extract.
        sorted_col_positions : list-like of ints, optional
            Sorted version of col_positions.

        Returns
        -------
        PandasDataframe
        """
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
        new_dtypes = None
        if self.has_materialized_dtypes:
            names = tuple(level_names) if len(level_names) > 1 else level_names[0]
            new_dtypes = self.index.to_frame(name=names).dtypes
            new_dtypes = pandas.concat([new_dtypes, self.dtypes])

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
            self.index.nlevels + self.column_widths[0]
        ] + self.column_widths[1:]
        result = self.__constructor__(
            new_parts,
            new_row_labels,
            new_columns,
            row_lengths=self._row_lengths_cache,
            column_widths=new_column_widths,
            dtypes=new_dtypes,
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
        extracted_columns = self.take_2d_labels_or_positional(
            col_labels=column_list
        ).to_pandas()

        if len(column_list) == 1:
            new_labels = pandas.Index(
                extracted_columns.squeeze(axis=1), name=column_list[0]
            )
        else:
            new_labels = pandas.MultiIndex.from_frame(
                extracted_columns, names=column_list
            )
        result = self.take_2d_labels_or_positional(
            col_labels=[i for i in self.columns if i not in extracted_columns.columns]
        )
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
        new_dtypes = self.copy_dtypes_cache()
        if row_positions is not None:
            # We want to preserve the frame's partitioning so passing in ``keep_partitioning=True``
            # in order to use the cached `row_lengths` values for the new frame.
            # If the frame's is re-partitioned using the "standard" partitioning,
            # then knowing that, we can compute new row lengths.
            ordered_rows = self._partition_mgr_cls.map_axis_partitions(
                0,
                self._partitions,
                lambda df: df.iloc[row_positions],
                keep_partitioning=True,
            )
            row_idx = self.index[row_positions]

            if len(row_idx) != len(self.index):
                # The frame was re-partitioned along the 0 axis during reordering using
                # the "standard" partitioning. Knowing the standard partitioning scheme
                # we are able to compute new row lengths.
                new_lengths = get_length_list(
                    axis_len=len(row_idx), num_splits=ordered_rows.shape[0]
                )
            else:
                # If the frame's partitioning was preserved then
                # we can use previous row lengths cache
                new_lengths = self._row_lengths_cache
        else:
            ordered_rows = self._partitions
            row_idx = self.index
            new_lengths = self._row_lengths_cache
        if col_positions is not None:
            # We want to preserve the frame's partitioning so passing in ``keep_partitioning=True``
            # in order to use the cached `column_widths` values for the new frame.
            # If the frame's is re-partitioned using the "standard" partitioning,
            # then knowing that, we can compute new column widths.
            ordered_cols = self._partition_mgr_cls.map_axis_partitions(
                1,
                ordered_rows,
                lambda df: df.iloc[:, col_positions],
                keep_partitioning=True,
            )
            col_idx = self.columns[col_positions]
            if self.has_materialized_dtypes:
                new_dtypes = self.dtypes.iloc[col_positions]

            if len(col_idx) != len(self.columns):
                # The frame was re-partitioned along the 1 axis during reordering using
                # the "standard" partitioning. Knowing the standard partitioning scheme
                # we are able to compute new column widths.
                new_widths = get_length_list(
                    axis_len=len(col_idx), num_splits=ordered_cols.shape[1]
                )
            else:
                # If the frame's partitioning was preserved then
                # we can use previous column widths cache
                new_widths = self._column_widths_cache
        else:
            ordered_cols = ordered_rows
            col_idx = self.columns
            new_widths = self._column_widths_cache
        return self.__constructor__(
            ordered_cols, row_idx, col_idx, new_lengths, new_widths, new_dtypes
        )

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
            self.copy_index_cache(),
            self.copy_columns_cache(),
            self._row_lengths_cache,
            self._column_widths_cache,
            self.copy_dtypes_cache(),
        )

    @lazy_metadata_decorator(apply_axis="both")
    def astype(self, col_dtypes, errors: str = "raise"):
        """
        Convert the columns dtypes to given dtypes.

        Parameters
        ----------
        col_dtypes : dictionary of {col: dtype,...}
            Where col is the column name and dtype is a NumPy dtype.
        errors : {'raise', 'ignore'}, default: 'raise'
            Control raising of exceptions on invalid data for provided dtype.

        Returns
        -------
        BaseDataFrame
            Dataframe with updated dtypes.
        """
        columns = col_dtypes.keys()
        # Create Series for the updated dtypes
        new_dtypes = self.dtypes.copy()
        # When casting to "category" we have to make up the whole axis partition
        # to get the properly encoded table of categories. Every block partition
        # will store the encoded table. That can lead to higher memory footprint.
        # TODO: Revisit if this hurts users.
        use_full_axis_cast = False
        has_categorical_cast = False
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
                    new_dtypes[column] = LazyProxyCategoricalDtype._build_proxy(
                        # Actual parent will substitute `None` at `.set_dtypes_cache`
                        parent=None,
                        column_name=column,
                        materializer=lambda parent, column: parent._compute_dtypes(
                            columns=[column]
                        )[column],
                    )
                    use_full_axis_cast = has_categorical_cast = True
                else:
                    new_dtypes[column] = new_dtype

        def astype_builder(df):
            """Compute new partition frame with dtypes updated."""
            # TODO(https://github.com/modin-project/modin/issues/6266): Remove this
            # copy, which is a workaround for https://github.com/pandas-dev/pandas/issues/53658
            df_for_astype = (
                df.copy(deep=True)
                if Engine.get() == "Ray" and has_categorical_cast
                else df
            )
            return df_for_astype.astype(
                {k: v for k, v in col_dtypes.items() if k in df}, errors=errors
            )

        if use_full_axis_cast:
            new_frame = self._partition_mgr_cls.map_axis_partitions(
                0, self._partitions, astype_builder, keep_partitioning=True
            )
        else:
            new_frame = self._partition_mgr_cls.map_partitions(
                self._partitions, astype_builder
            )
        return self.__constructor__(
            new_frame,
            self.copy_index_cache(),
            self.copy_columns_cache(),
            self._row_lengths_cache,
            self._column_widths_cache,
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
        # Fasttrack empty numpy array
        if isinstance(indices, np.ndarray) and indices.size == 0:
            # This will help preserve metadata stored in empty dataframes (indexes and dtypes)
            # Otherwise, we will get an empty `new_partitions` array, from which it will
            #  no longer be possible to obtain metadata
            return OrderedDict([(0, np.array([], dtype=np.int64))])
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
            bins = np.array(self.row_lengths)
        else:
            bins = np.array(self.column_widths)
        # INT_MAX to make sure we don't try to compute on partitions that don't exist.
        cumulative = np.append(bins[:-1].cumsum(), np.iinfo(bins.dtype).max)

        def internal(block_idx: int, global_index):
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
                result.index = [MODIN_UNNAMED_SERIES_LABEL]
            else:
                result = pandas.DataFrame(series_result)
                if isinstance(series_result, pandas.Series):
                    result.columns = [MODIN_UNNAMED_SERIES_LABEL]
            return result

        return _tree_reduce_func

    def _compute_tree_reduce_metadata(self, axis, new_parts, dtypes=None):
        """
        Compute the metadata for the result of reduce function.

        Parameters
        ----------
        axis : int
            The axis on which reduce function was applied.
        new_parts : NumPy 2D array
            Partitions with the result of applied function.
        dtypes : str, optional
            The data types for the result. This is an optimization
            because there are functions that always result in a particular data
            type, and this allows us to avoid (re)computing it.

        Returns
        -------
        PandasDataframe
            Modin series (1xN frame) containing the reduced data.
        """
        new_axes, new_axes_lengths = [0, 0], [0, 0]

        new_axes[axis] = [MODIN_UNNAMED_SERIES_LABEL]
        new_axes[axis ^ 1] = self.axes[axis ^ 1]

        new_axes_lengths[axis] = [1]
        new_axes_lengths[axis ^ 1] = self._axes_lengths[axis ^ 1]

        if dtypes == "copy":
            dtypes = self.copy_dtypes_cache()
        elif dtypes is not None:
            dtypes = pandas.Series(
                [np.dtype(dtypes)] * len(new_axes[1]), index=new_axes[1]
            )

        result = self.__constructor__(
            new_parts,
            *new_axes,
            *new_axes_lengths,
            dtypes,
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
        return self._compute_tree_reduce_metadata(axis.value, new_parts, dtypes=dtypes)

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
        return self._compute_tree_reduce_metadata(
            axis.value, reduce_parts, dtypes=dtypes
        )

    @lazy_metadata_decorator(apply_axis=None)
    def map(
        self,
        func: Callable,
        dtypes: Optional[str] = None,
        new_columns: Optional[pandas.Index] = None,
    ) -> "PandasDataframe":
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
        new_columns : pandas.Index, optional
            New column labels of the result, its length has to be identical
            to the older columns. If not specified, old column labels are preserved.

        Returns
        -------
        PandasDataframe
            A new dataframe.
        """
        new_partitions = self._partition_mgr_cls.map_partitions(self._partitions, func)
        if new_columns is not None and self.has_materialized_columns:
            assert len(new_columns) == len(
                self.columns
            ), "New column's length must be identical to the previous columns"
        elif new_columns is None:
            new_columns = self.copy_columns_cache()
        if isinstance(dtypes, str) and dtypes == "copy":
            dtypes = self.copy_dtypes_cache()
        elif dtypes is not None and not isinstance(dtypes, pandas.Series):
            if isinstance(new_columns, ModinIndex):
                # Materializing lazy columns in order to build dtype's index
                new_columns = new_columns.get(return_lengths=False)
            dtypes = pandas.Series(
                [np.dtype(dtypes)] * len(new_columns), index=new_columns
            )
        return self.__constructor__(
            new_partitions,
            self.copy_index_cache(),
            new_columns,
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
    def fold(self, axis, func, new_columns=None):
        """
        Perform a function across an entire axis.

        Parameters
        ----------
        axis : int
            The axis to apply over.
        func : callable
            The function to apply.
        new_columns : list-like, optional
            The columns of the result.
            Must be the same length as the columns' length of `self`.
            The column labels of `self` may change during an operation so
            we may want to pass the new column labels in (e.g., see `cat.codes`).

        Returns
        -------
        PandasDataframe
            A new dataframe.

        Notes
        -----
        The data shape is not changed (length and width of the table).
        """
        if new_columns is not None:
            if self.has_materialized_columns:
                assert len(self.columns) == len(
                    new_columns
                ), "The length of `new_columns` doesn't match the columns' length of `self`"
            self.set_columns_cache(new_columns)

        new_partitions = self._partition_mgr_cls.map_axis_partitions(
            axis, self._partitions, func, keep_partitioning=True
        )
        return self.__constructor__(
            new_partitions,
            self.copy_index_cache(),
            self.copy_columns_cache(),
            self._row_lengths_cache,
            self._column_widths_cache,
        )

    def infer_objects(self) -> "PandasDataframe":
        """
        Attempt to infer better dtypes for object columns.

        Attempts soft conversion of object-dtyped columns, leaving non-object and unconvertible
        columns unchanged. The inference rules are the same as during normal Series/DataFrame
        construction.

        Returns
        -------
        PandasDataframe
            A new PandasDataframe with the inferred schema.
        """
        obj_cols = [
            col for col, dtype in enumerate(self.dtypes) if is_object_dtype(dtype)
        ]
        return self.infer_types(obj_cols)

    def infer_types(self, col_labels: List[str]) -> "PandasDataframe":
        """
        Determine the compatible type shared by all values in the specified columns, and coerce them to that type.

        Parameters
        ----------
        col_labels : list
            List of column labels to infer and induce types over.

        Returns
        -------
        PandasDataframe
            A new PandasDataframe with the inferred schema.
        """
        # Compute dtypes on the specified columns, and then set those dtypes on a new frame
        new_cols = self.take_2d_labels_or_positional(col_labels=col_labels)
        new_cols_dtypes = new_cols.tree_reduce(0, pandas.DataFrame.infer_objects).dtypes
        new_dtypes = self.dtypes.copy()
        new_dtypes[col_labels] = new_cols_dtypes
        return self.__constructor__(
            self._partitions,
            self.copy_index_cache(),
            self.copy_columns_cache(),
            self._row_lengths_cache,
            self._column_widths_cache,
            new_dtypes,
        )

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
        new_dtypes = None
        if self.has_materialized_dtypes:
            new_dtypes = self.dtypes.set_axis(new_cols)
        return self.__constructor__(
            new_parts,
            new_index,
            new_cols,
            self._row_lengths_cache,
            self._column_widths_cache,
            new_dtypes,
        )

    def combine_and_apply(
        self, func, new_index=None, new_columns=None, new_dtypes=None
    ):
        """
        Combine all partitions into a single big one and apply the passed function to it.

        Use this method with care as it collects all the data on the same worker,
        it's only recommended to use this method on small or reduced datasets.

        Parameters
        ----------
        func : callable(pandas.DataFrame) -> pandas.DataFrame
            A function to apply to the combined partition.
        new_index : sequence, optional
            Index of the result.
        new_columns : sequence, optional
            Columns of the result.
        new_dtypes : dict-like, optional
            Dtypes of the result.

        Returns
        -------
        PandasDataframe
        """
        if self._partitions.shape[1] > 1:
            new_partitions = self._partition_mgr_cls.row_partitions(self._partitions)
            new_partitions = np.array([[partition] for partition in new_partitions])
            modin_frame = self.__constructor__(
                new_partitions,
                self.copy_index_cache(),
                self.copy_columns_cache(),
                self._row_lengths_cache,
                [len(self.columns)] if self.has_materialized_columns else None,
                self.copy_dtypes_cache(),
            )
        else:
            modin_frame = self
        return modin_frame.apply_full_axis(
            axis=0,
            func=func,
            new_index=new_index,
            new_columns=new_columns,
            dtypes=new_dtypes,
        )

    def _apply_func_to_range_partitioning(
        self, key_column, func, ascending=True, **kwargs
    ):
        """
        Reshuffle data so it would be range partitioned and then apply the passed function row-wise.

        Parameters
        ----------
        key_column : hashable
            Column name to build the range partitioning for.
        func : callable(pandas.DataFrame) -> pandas.DataFrame
            Function to apply against partitions.
        ascending : bool, default: True
            Whether the range should be built in ascending or descending order.
        **kwargs : dict
            Additional arguments to forward to the range builder function.

        Returns
        -------
        PandasDataframe
            A new dataframe.
        """
        # If there's only one row partition can simply apply the function row-wise without the need to reshuffle
        if self._partitions.shape[0] == 1:
            return self.apply_full_axis(axis=1, func=func)

        ideal_num_new_partitions = len(self._partitions)
        m = len(self.index) / ideal_num_new_partitions
        sampling_probability = (1 / m) * np.log(
            ideal_num_new_partitions * len(self.index)
        )
        # If this df is overpartitioned, we try to sample each partition with probability
        # greater than 1, which leads to an error. In this case, we can do one of the following
        # two things. If there is only enough rows for one partition, and we have only 1 column
        # partition, we can just combine the overpartitioned df into one partition, and sort that
        # partition. If there is enough data for more than one partition, we can tell the sorting
        # algorithm how many partitions we want to end up with, so it samples and finds pivots
        # according to that.
        if sampling_probability >= 1:
            from modin.config import MinPartitionSize

            ideal_num_new_partitions = round(len(self.index) / MinPartitionSize.get())
            if len(self.index) < MinPartitionSize.get() or ideal_num_new_partitions < 2:
                # If the data is too small, we shouldn't try reshuffling/repartitioning but rather
                # simply combine all partitions and apply the sorting to the whole dataframe
                return self.combine_and_apply(func=func)

        if self.dtypes[key_column] == object:
            # This means we are not sorting numbers, so we need our quantiles to not try
            # arithmetic on the values.
            method = "inverted_cdf"
        else:
            method = "linear"

        shuffling_functions = build_sort_functions(
            self,
            key_column,
            method,
            ascending[0] if is_list_like(ascending) else ascending,
            ideal_num_new_partitions,
            **kwargs,
        )
        if ideal_num_new_partitions < len(self._partitions):
            if len(self._partitions) % ideal_num_new_partitions == 0:
                joining_partitions = np.split(
                    self._partitions, ideal_num_new_partitions
                )
            else:
                joining_partitions = np.split(
                    self._partitions,
                    range(
                        0,
                        len(self._partitions),
                        round(len(self._partitions) / ideal_num_new_partitions),
                    )[1:],
                )

            new_partitions = np.array(
                [
                    self._partition_mgr_cls.column_partitions(ptn_grp, full_axis=False)
                    for ptn_grp in joining_partitions
                ]
            )
        else:
            new_partitions = self._partitions

        major_col_partition_index = self.columns.get_loc(key_column)
        cols_seen = 0
        index = -1
        for i, length in enumerate(self.column_widths):
            cols_seen += length
            if major_col_partition_index < cols_seen:
                index = i
                break
        new_partitions = self._partition_mgr_cls.shuffle_partitions(
            new_partitions,
            index,
            shuffling_functions,
            func,
        )

        return self.__constructor__(new_partitions)

    @lazy_metadata_decorator(apply_axis="both")
    def sort_by(
        self,
        axis: Union[int, Axis],
        columns: Union[str, List[str]],
        ascending: bool = True,
        **kwargs,
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
        **kwargs : dict
            Keyword arguments to pass when sorting partitions.

        Returns
        -------
        PandasDataframe
            A new PandasDataframe sorted into lexicographical order by the specified column(s).
        """
        if not isinstance(columns, list):
            columns = [columns]

        def sort_function(df):  # pragma: no cover
            # When we do a sort on the result of Series.value_counts, we don't rename the index until
            # after everything is done, which causes an error when sorting the partitions, since the
            # index and the column share the same name, when in actuality, the index's name should be
            # None. This fixes the indexes name beforehand in that case, so that the sort works.
            index_renaming = None
            if any(name in df.columns for name in df.index.names):
                index_renaming = df.index.names
                df.index = df.index.set_names([None] * len(df.index.names))
            df = df.sort_values(by=columns, ascending=ascending, **kwargs)
            if index_renaming is not None:
                df.index = df.index.set_names(index_renaming)
            return df

        # If this df is empty, we don't want to try and shuffle or sort.
        if len(self.axes[0]) == 0 or len(self.axes[1]) == 0:
            return self.copy()

        axis = Axis(axis)
        if axis != Axis.ROW_WISE:
            raise NotImplementedError(
                f"Algebra sort only implemented row-wise. {axis.name} sort not implemented yet!"
            )

        result = self._apply_func_to_range_partitioning(
            key_column=columns[0], func=sort_function, ascending=ascending, **kwargs
        )

        result.set_axis_cache(self.copy_axis_cache(axis.value ^ 1), axis=axis.value ^ 1)
        result.set_dtypes_cache(self.copy_dtypes_cache())
        # We perform the final steps of the sort on full axis partitions, so we know that the
        # length of each partition is the full length of the dataframe.
        if self.has_materialized_columns:
            self._set_axis_lengths_cache([len(self.columns)], axis=axis.value ^ 1)

        if kwargs.get("ignore_index", False):
            result.index = RangeIndex(len(self.get_axis(axis.value)))

        # Since the strategy to pick our pivots involves random sampling
        # we could end up picking poor pivots, leading to skew in our partitions.
        # We should add a fix to check if there is skew in the partitions and rebalance
        # them if necessary. Calling `rebalance_partitions` won't do this, since it only
        # resolves the case where there isn't the right amount of partitions - not where
        # there is skew across the lengths of partitions.
        return result

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

        new_axes[axis.value] = (
            self.copy_index_cache() if axis.value == 0 else self.copy_columns_cache()
        )
        new_lengths[axis.value] = (
            self._row_lengths_cache if axis.value == 0 else self._column_widths_cache
        )
        new_axes[axis.value ^ 1], new_lengths[axis.value ^ 1] = None, None

        return self.__constructor__(
            new_partitions,
            *new_axes,
            *new_lengths,
            self.copy_dtypes_cache() if axis == Axis.COL_WISE else None,
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
        return self.take_2d_labels_or_positional(
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
            new_index, row_lengths = self._compute_axis_labels_and_lengths(
                0, partitions
            )
            new_columns, column_widths = self.columns, self._column_widths_cache
        else:
            new_index, row_lengths = self.index, self._row_lengths_cache
            new_columns, column_widths = self._compute_axis_labels_and_lengths(
                1, partitions
            )
        return self.__constructor__(
            partitions, new_index, new_columns, row_lengths, column_widths
        )

    @lazy_metadata_decorator(apply_axis="both")
    def apply_full_axis(
        self,
        axis,
        func,
        new_index=None,
        new_columns=None,
        apply_indices=None,
        enumerate_partitions: bool = False,
        dtypes=None,
        keep_partitioning=True,
        num_splits=None,
        sync_labels=True,
        pass_axis_lengths_to_partitions=False,
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
        apply_indices : list-like, default: None
            Indices of `axis ^ 1` to apply function over.
        enumerate_partitions : bool, default: False
            Whether pass partition index into applied `func` or not.
            Note that `func` must be able to obtain `partition_idx` kwarg.
        dtypes : list-like, optional
            The data types of the result. This is an optimization
            because there are functions that always result in a particular data
            type, and allows us to avoid (re)computing it.
        keep_partitioning : boolean, default: True
            The flag to keep partition boundaries for Modin Frame if possible.
            Setting it to True disables shuffling data from one partition to another in case the resulting
            number of splits is equal to the initial number of splits.
        num_splits : int, optional
            The number of partitions to split the result into across the `axis`. If None, then the number
            of splits will be infered automatically. If `num_splits` is None and `keep_partitioning=True`
            then the number of splits is preserved.
        sync_labels : boolean, default: True
            Synchronize external indexes (`new_index`, `new_columns`) with internal indexes.
            This could be used when you're certain that the indices in partitions are equal to
            the provided hints in order to save time on syncing them.
        pass_axis_lengths_to_partitions : bool, default: False
            Whether pass partition lengths along `axis ^ 1` to the kernel `func`.
            Note that `func` must be able to obtain `df, *axis_lengths`.

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
            apply_indices=apply_indices,
            enumerate_partitions=enumerate_partitions,
            dtypes=dtypes,
            other=None,
            keep_partitioning=keep_partitioning,
            num_splits=num_splits,
            sync_labels=sync_labels,
            pass_axis_lengths_to_partitions=pass_axis_lengths_to_partitions,
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
                else [self.row_lengths, self.column_widths][axis],
                axis ^ 1: [self.row_lengths, self.column_widths][axis ^ 1],
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
        self, axis, func, other, join_type="left", labels="keep", dtypes=None
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
        labels : {"keep", "replace", "drop"}, default: "keep"
            Whether keep labels from `self` Modin DataFrame, replace them with labels
            from joined DataFrame or drop altogether to make them be computed lazily later.
        dtypes : "copy", pandas.Series or None, default: None
            Dtypes of the result. "copy" to keep old dtypes and None to compute them on demand.

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
        if isinstance(dtypes, str) and dtypes == "copy":
            dtypes = self.copy_dtypes_cache()

        def _pick_axis(get_axis, sizes_cache):
            if labels == "keep":
                return get_axis(), sizes_cache
            if labels == "replace":
                return joined_index, partition_sizes_along_axis
            assert labels == "drop", f"Unexpected `labels`: {labels}"
            return None, None

        if axis == 0:
            # Pass shape caches instead of values in order to not trigger shape computation.
            new_index, new_row_lengths = _pick_axis(
                self._get_index, self._row_lengths_cache
            )
            new_columns, new_column_widths = self.columns, self._column_widths_cache
        else:
            new_index, new_row_lengths = self.index, self._row_lengths_cache
            new_columns, new_column_widths = _pick_axis(
                self._get_columns, self._column_widths_cache
            )

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
            sizes = self.row_lengths if axis else self.column_widths
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
        return self.__constructor__(
            new_partitions, index=new_index, columns=new_columns
        )

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
        keep_partitioning=True,
        num_splits=None,
        sync_labels=True,
        pass_axis_lengths_to_partitions=False,
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
        keep_partitioning : boolean, default: True
            The flag to keep partition boundaries for Modin Frame if possible.
            Setting it to True disables shuffling data from one partition to another in case the resulting
            number of splits is equal to the initial number of splits.
        num_splits : int, optional
            The number of partitions to split the result into across the `axis`. If None, then the number
            of splits will be infered automatically. If `num_splits` is None and `keep_partitioning=True`
            then the number of splits is preserved.
        sync_labels : boolean, default: True
            Synchronize external indexes (`new_index`, `new_columns`) with internal indexes.
            This could be used when you're certain that the indices in partitions are equal to
            the provided hints in order to save time on syncing them.
        pass_axis_lengths_to_partitions : bool, default: False
            Whether pass partition lengths along `axis ^ 1` to the kernel `func`.
            Note that `func` must be able to obtain `df, *axis_lengths`.

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

        apply_func_args = None
        if pass_axis_lengths_to_partitions:
            if axis == 0:
                apply_func_args = (
                    self._column_widths_cache
                    if self._column_widths_cache is not None
                    else [part.width(materialize=False) for part in self._partitions[0]]
                )
            else:
                apply_func_args = (
                    self._row_lengths_cache
                    if self._row_lengths_cache is not None
                    else [
                        part.length(materialize=False) for part in self._partitions.T[0]
                    ]
                )

        new_partitions = self._partition_mgr_cls.broadcast_axis_partitions(
            axis=axis,
            left=self._partitions,
            right=other,
            apply_func=self._build_treereduce_func(axis, func),
            apply_indices=apply_indices,
            enumerate_partitions=enumerate_partitions,
            keep_partitioning=keep_partitioning,
            num_splits=num_splits,
            apply_func_args=apply_func_args,
        )
        kw = {"row_lengths": None, "column_widths": None}
        if isinstance(dtypes, str) and dtypes == "copy":
            kw["dtypes"] = self.copy_dtypes_cache()
        elif dtypes is not None:
            if isinstance(dtypes, (pandas.Series, ModinDtypes)):
                kw["dtypes"] = dtypes.copy()
            else:
                if new_columns is None:
                    (
                        new_columns,
                        kw["column_widths"],
                    ) = self._compute_axis_labels_and_lengths(1, new_partitions)
                kw["dtypes"] = (
                    pandas.Series(dtypes, index=new_columns)
                    if is_list_like(dtypes)
                    else pandas.Series(
                        [np.dtype(dtypes)] * len(new_columns), index=new_columns
                    )
                )

        if not keep_partitioning:
            if kw["row_lengths"] is None and new_index is not None:
                if axis == 0:
                    kw["row_lengths"] = get_length_list(
                        axis_len=len(new_index), num_splits=new_partitions.shape[0]
                    )
                elif axis == 1:
                    if self._row_lengths_cache is not None and len(new_index) == sum(
                        self._row_lengths_cache
                    ):
                        kw["row_lengths"] = self._row_lengths_cache
                    elif len(new_index) == 1 and new_partitions.shape[0] == 1:
                        kw["row_lengths"] = [1]
            if kw["column_widths"] is None and new_columns is not None:
                if axis == 1:
                    kw["column_widths"] = get_length_list(
                        axis_len=len(new_columns),
                        num_splits=new_partitions.shape[1],
                    )
                elif axis == 0:
                    if self._column_widths_cache is not None and len(
                        new_columns
                    ) == sum(self._column_widths_cache):
                        kw["column_widths"] = self._column_widths_cache
                    elif len(new_columns) == 1 and new_partitions.shape[1] == 1:
                        kw["column_widths"] = [1]

        result = self.__constructor__(
            new_partitions, index=new_index, columns=new_columns, **kw
        )
        if sync_labels and new_index is not None:
            result.synchronize_labels(axis=0)
        if sync_labels and new_columns is not None:
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
            base_lengths = base_frame.column_widths if axis else base_frame.row_lengths

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
    def n_ary_op(
        self,
        op,
        right_frames: list,
        join_type="outer",
        copartition_along_columns=True,
        labels="replace",
        dtypes=None,
    ):
        """
        Perform an n-opary operation by joining with other Modin DataFrame(s).

        Parameters
        ----------
        op : callable
            Function to apply after the join.
        right_frames : list of PandasDataframe
            Modin DataFrames to join with.
        join_type : str, default: "outer"
            Type of join to apply.
        copartition_along_columns : bool, default: True
            Whether to perform copartitioning along columns or not.
            For some ops this isn't needed (e.g., `fillna`).
        labels : {"replace", "drop"}, default: "replace"
            Whether use labels from joined DataFrame or drop altogether to make
            them be computed lazily later.
        dtypes : series, default: None
            Dtypes of the resultant dataframe, this argument will be
            received if the resultant dtypes of n-opary operation is precomputed.

        Returns
        -------
        PandasDataframe
            New Modin DataFrame.
        """
        left_parts, list_of_right_parts, joined_index, row_lengths = self._copartition(
            0, right_frames, join_type, sort=True
        )
        if copartition_along_columns:
            new_left_frame = self.__constructor__(
                left_parts, joined_index, self.columns, row_lengths, self.column_widths
            )
            new_right_frames = [
                self.__constructor__(
                    right_parts,
                    joined_index,
                    right_frame.columns,
                    row_lengths,
                    right_frame.column_widths,
                )
                for right_parts, right_frame in zip(list_of_right_parts, right_frames)
            ]

            (
                left_parts,
                list_of_right_parts,
                joined_columns,
                column_widths,
            ) = new_left_frame._copartition(
                1,
                new_right_frames,
                join_type,
                sort=True,
            )
        else:
            joined_columns = self.copy_columns_cache()
            column_widths = self._column_widths_cache

        new_frame = (
            np.array([])
            if len(left_parts) == 0
            or any(len(right_parts) == 0 for right_parts in list_of_right_parts)
            else self._partition_mgr_cls.n_ary_operation(
                left_parts, op, list_of_right_parts
            )
        )
        if labels == "drop":
            joined_index = joined_columns = row_lengths = column_widths = None

        return self.__constructor__(
            new_frame,
            joined_index,
            joined_columns,
            row_lengths,
            column_widths,
            dtypes,
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
        new_widths = None
        new_lengths = None

        def _compute_new_widths():
            widths = None
            if self._column_widths_cache is not None and all(
                o._column_widths_cache is not None for o in others
            ):
                widths = self._column_widths_cache + [
                    width for o in others for width in o._column_widths_cache
                ]
            return widths

        # Fast path for equivalent columns and partitioning
        if (
            axis == Axis.ROW_WISE
            and all(o.columns.equals(self.columns) for o in others)
            and all(o.column_widths == self.column_widths for o in others)
        ):
            joined_index = self.columns
            left_parts = self._partitions
            right_parts = [o._partitions for o in others]
            new_widths = self._column_widths_cache
        elif (
            axis == Axis.COL_WISE
            and all(o.index.equals(self.index) for o in others)
            and all(o.row_lengths == self.row_lengths for o in others)
        ):
            joined_index = self.index
            left_parts = self._partitions
            right_parts = [o._partitions for o in others]
            new_lengths = self._row_lengths_cache
            # we can only do this for COL_WISE because `concat` might rebalance partitions for ROW_WISE
            new_widths = _compute_new_widths()
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
                new_widths = _compute_new_widths()
            else:
                new_widths = partition_sizes_along_axis
        new_partitions, new_lengths2 = self._partition_mgr_cls.concat(
            axis.value, left_parts, right_parts
        )
        if new_lengths is None:
            new_lengths = new_lengths2
        new_dtypes = None
        if axis == Axis.ROW_WISE:
            new_index = self.index.append([other.index for other in others])
            new_columns = joined_index
            frames = [self] + others
            if all(frame.has_materialized_dtypes for frame in frames):
                all_dtypes = [frame.dtypes for frame in frames]
                new_dtypes = pandas.concat(all_dtypes, axis=1)
                # 'nan' value will be placed in a row if a column doesn't exist in all frames;
                # this value is np.float64 type so we need an explicit conversion
                new_dtypes.fillna(np.dtype("float64"), inplace=True)
                new_dtypes = new_dtypes.apply(
                    lambda row: find_common_type(row.values), axis=1
                )
            # If we have already cached the length of each row in at least one
            # of the row's partitions, we can build new_lengths for the new
            # frame. Typically, if we know the length for any partition in a
            # row, we know the length for the first partition in the row. So
            # just check the lengths of the first column of partitions.
            if not new_lengths:
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
            if self.has_materialized_dtypes and all(
                o.has_materialized_dtypes for o in others
            ):
                new_dtypes = pandas.concat([self.dtypes] + [o.dtypes for o in others])
            # If we have already cached the width of each column in at least one
            # of the column's partitions, we can build new_widths for the new
            # frame. Typically, if we know the width for any partition in a
            # column, we know the width for the first partition in the column.
            # So just check the widths of the first row of partitions.
            if not new_widths:
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

    @lazy_metadata_decorator(apply_axis="both")
    def groupby(
        self,
        axis: Union[int, Axis],
        by: Union[str, List[str]],
        operator: Callable,
        result_schema: Optional[Dict[Hashable, type]] = None,
        **kwargs: dict,
    ) -> "PandasDataframe":
        """
        Generate groups based on values in the input column(s) and perform the specified operation on each.

        Parameters
        ----------
        axis : int or modin.core.dataframe.base.utils.Axis
            The axis to apply the grouping over.
        by : string or list of strings
            One or more column labels to use for grouping.
        operator : callable(pandas.core.groupby.DataFrameGroupBy) -> pandas.DataFrame
            The operation to carry out on each of the groups. The operator is another
            algebraic operator with its own user-defined function parameter, depending
            on the output desired by the user.
        result_schema : dict, optional
            Mapping from column labels to data types that represents the types of the output dataframe.
        **kwargs : dict
            Additional arguments to pass to the ``df.groupby`` method (besides the 'by' argument).

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

        Unlike the pandas API, an intermediate "GROUP BY" object is not present in this
        algebra implementation.
        """
        axis = Axis(axis)
        if axis != Axis.ROW_WISE:
            raise NotImplementedError(
                f"Algebra groupby only implemented row-wise. {axis.name} axis groupby not implemented yet!"
            )

        if not isinstance(by, list):
            by = [by]

        def apply_func(df):  # pragma: no cover
            if any(is_categorical_dtype(dtype) for dtype in df.dtypes[by].values):
                raise NotImplementedError(
                    "Reshuffling groupby is not yet supported when grouping on a categorical column. "
                    + "https://github.com/modin-project/modin/issues/5925"
                )
            return operator(df.groupby(by, **kwargs))

        result = self._apply_func_to_range_partitioning(
            key_column=by[0],
            func=apply_func,
        )

        if result_schema is not None:
            new_dtypes = pandas.Series(result_schema)

            result.set_dtypes_cache(new_dtypes)
            result.set_columns_cache(new_dtypes.index)

        return result

    @lazy_metadata_decorator(apply_axis="both")
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

        if by_parts is not None:
            # inplace operation
            if by_parts.shape[axis] != self._partitions.shape[axis]:
                self._filter_empties(compute_metadata=False)
        new_partitions = self._partition_mgr_cls.groupby_reduce(
            axis, self._partitions, by_parts, map_func, reduce_func, apply_indices
        )
        return self.__constructor__(
            new_partitions, index=new_index, columns=new_columns
        )

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
            if len(df.columns) and self.has_materialized_dtypes:
                df = df.astype(self.dtypes)
        else:
            for axis, has_external_index in enumerate(
                ["has_materialized_index", "has_materialized_columns"]
            ):
                # no need to check external and internal axes since in that case
                # external axes will be computed from internal partitions
                if getattr(self, has_external_index):
                    external_index = self.columns if axis else self.index
                    ErrorMessage.catch_bugs_and_request_email(
                        not df.axes[axis].equals(external_index),
                        f"Internal and external indices on axis {axis} do not match.",
                    )
                    # have to do this in order to assign some potentially missing metadata,
                    # the ones that were set to the external index but were never propagated
                    # into the internal ones
                    df = df.set_axis(axis=axis, labels=external_index, copy=False)

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
        if self.has_materialized_dtypes:
            new_dtypes = pandas.Series(
                np.full(len(self.index), find_common_type(self.dtypes.values)),
                index=self.index,
            )
        else:
            new_dtypes = None
        return self.__constructor__(
            new_partitions,
            self.copy_columns_cache(),
            self.copy_index_cache(),
            self._column_widths_cache,
            self._row_lengths_cache,
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
        from modin.core.dataframe.pandas.interchange.dataframe_protocol.dataframe import (
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
        if type(df) is cls:
            return df

        if not hasattr(df, "__dataframe__"):
            raise ValueError(
                "`df` does not support DataFrame exchange protocol, i.e. `__dataframe__` method"
            )

        from modin.core.dataframe.pandas.interchange.dataframe_protocol.from_dataframe import (
            from_dataframe_to_pandas,
        )

        ErrorMessage.default_to_pandas(message="`from_dataframe`")
        pandas_df = from_dataframe_to_pandas(df)
        return cls.from_pandas(pandas_df)
