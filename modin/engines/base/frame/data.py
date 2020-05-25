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

from collections import OrderedDict
import numpy as np
import pandas
from pandas.core.indexes.api import ensure_index
from pandas.core.dtypes.common import is_numeric_dtype

from modin.backends.pandas.query_compiler import PandasQueryCompiler
from modin.error_message import ErrorMessage
from modin.backends.pandas.parsers import find_common_type_cat as find_common_type


class BasePandasFrame(object):

    _frame_mgr_cls = None
    _query_compiler_cls = PandasQueryCompiler

    @property
    def __constructor__(self):
        """The constructor for this object. A convenience method"""
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
        """Initialize a dataframe.

        Args:
            partitions: A 2D NumPy array of partitions. Must contain partition objects.
            index: The index object for the dataframe. Converts to a pandas.Index.
            columns: The columns object for the dataframe. Converts to a pandas.Index.
            row_lengths: (optional) The lengths of each partition in the rows. The
                "height" of each of the block partitions. Is computed if not provided.
            column_widths: (optional) The width of each partition in the columns. The
                "width" of each of the block partitions. Is computed if not provided.
            dtypes: (optional) The data types for the dataframe.
        """
        self._partitions = partitions
        self._index_cache = ensure_index(index)
        self._columns_cache = ensure_index(columns)
        if row_lengths is not None and len(self.index) > 0:
            ErrorMessage.catch_bugs_and_request_email(
                sum(row_lengths) != len(self._index_cache),
                "Row lengths: {} != {}".format(
                    sum(row_lengths), len(self._index_cache)
                ),
            )
        self._row_lengths_cache = row_lengths
        if column_widths is not None and len(self.columns) > 0:
            ErrorMessage.catch_bugs_and_request_email(
                sum(column_widths) != len(self._columns_cache),
                "Column widths: {} != {}".format(
                    sum(column_widths), len(self._columns_cache)
                ),
            )
        self._column_widths_cache = column_widths
        self._dtypes = dtypes
        self._filter_empties()

    @property
    def _row_lengths(self):
        """Compute the row lengths if they are not cached.

        Returns:
            A list of row lengths.
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
        """Compute the column widths if they are not cached.

        Returns:
            A list of column widths.
        """
        if self._column_widths_cache is None:
            if len(self._partitions) > 0:
                self._column_widths_cache = [obj.width() for obj in self._partitions[0]]
            else:
                self._column_widths_cache = []
        return self._column_widths_cache

    @property
    def dtypes(self):
        """Compute the data types if they are not cached.

        Returns:
            A pandas Series containing the data types for this dataframe.
        """
        if self._dtypes is None:
            self._dtypes = self._compute_dtypes()
        return self._dtypes

    def _compute_dtypes(self):
        """Compute the dtypes via MapReduce.

        Returns:
            The data types of this dataframe.
        """

        def dtype_builder(df):
            return df.apply(lambda col: find_common_type(col.values), axis=0)

        map_func = self._build_mapreduce_func(0, lambda df: df.dtypes)
        reduce_func = self._build_mapreduce_func(0, dtype_builder)
        # For now we will use a pandas Series for the dtypes.
        if len(self.columns) > 0:
            dtypes = self._map_reduce(0, map_func, reduce_func).to_pandas().iloc[0]
        else:
            dtypes = pandas.Series([])
        # reset name to None because we use "__reduced__" internally
        dtypes.name = None
        return dtypes

    _index_cache = None
    _columns_cache = None

    def _validate_set_axis(self, new_labels, old_labels):
        """Validates the index or columns replacement against the old labels.

        Args:
            new_labels: The labels to replace with.
            old_labels: The labels to replace.

        Returns:
            The validated labels.
        """
        new_labels = ensure_index(new_labels)
        old_len = len(old_labels)
        new_len = len(new_labels)
        if old_len != new_len:
            raise ValueError(
                "Length mismatch: Expected axis has %d elements, "
                "new values have %d elements" % (old_len, new_len)
            )
        return new_labels

    def _get_index(self):
        """Gets the index from the cache object.

        Returns:
            A pandas.Index object containing the row labels.
        """
        return self._index_cache

    def _get_columns(self):
        """Gets the columns from the cache object.

        Returns:
            A pandas.Index object containing the column labels.
        """
        return self._columns_cache

    def _set_index(self, new_index):
        """Replaces the current row labels with new labels.

        Args:
            new_index: The replacement row labels.
        """
        if self._index_cache is None:
            self._index_cache = ensure_index(new_index)
        else:
            new_index = self._validate_set_axis(new_index, self._index_cache)
            self._index_cache = new_index
        self._apply_index_objs(axis=0)

    def _set_columns(self, new_columns):
        """Replaces the current column labels with new labels.

        Args:
            new_columns: The replacement column labels.
        """
        if self._columns_cache is None:
            self._columns_cache = ensure_index(new_columns)
        else:
            new_columns = self._validate_set_axis(new_columns, self._columns_cache)
            self._columns_cache = new_columns
            if self._dtypes is not None:
                self._dtypes.index = new_columns
        self._apply_index_objs(axis=1)

    columns = property(_get_columns, _set_columns)
    index = property(_get_index, _set_index)

    @property
    def axes(self):
        """The index, columns that can be accessed with an `axis` integer."""
        return [self.index, self.columns]

    def _filter_empties(self):
        """Removes empty partitions to avoid triggering excess computation."""
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
                    if j < len(self._column_widths) and self._column_widths[j] > 0
                ]
                for i in range(len(self._partitions))
                if i < len(self._row_lengths) and self._row_lengths[i] > 0
            ]
        )
        self._column_widths_cache = [w for w in self._column_widths if w > 0]
        self._row_lengths_cache = [r for r in self._row_lengths if r > 0]

    def _apply_index_objs(self, axis=None):
        """Lazily applies the index object (Index or Columns) to the partitions.

        Args:
            axis: The axis to apply to, None applies to both axes.

        Returns:
            A new 2D array of partitions that have the index assignment added to the
            call queue.
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
            ErrorMessage.catch_bugs_and_request_email(
                axis is not None and axis not in [0, 1]
            )

    def mask(
        self,
        row_indices=None,
        row_numeric_idx=None,
        col_indices=None,
        col_numeric_idx=None,
    ):
        """Lazily select columns or rows from given indices.

        Note: If both row_indices and row_numeric_idx are set, row_indices will be used.
            The same rule applied to col_indices and col_numeric_idx.

        Parameters
        ----------
        row_indices : list of hashable
            The row labels to extract.
        row_numeric_idx : list of int
            The row indices to extract.
        col_indices : list of hashable
            The column labels to extract.
        col_numeric_idx : list of int
            The column indices to extract.

        Returns
        -------
        BasePandasFrame
             A new BasePandasFrame from the mask provided.
        """
        if (
            row_indices is None
            and row_numeric_idx is None
            and col_indices is None
            and col_numeric_idx is None
        ):
            return self.copy()
        if row_indices is not None:
            row_numeric_idx = self.index.get_indexer_for(row_indices)
        if row_numeric_idx is not None:
            row_partitions_list = self._get_dict_of_block_index(1, row_numeric_idx)
            new_row_lengths = [
                len(indices) for _, indices in row_partitions_list.items()
            ]
            new_index = self.index[sorted(row_numeric_idx)]
        else:
            row_partitions_list = {
                i: slice(None) for i in range(len(self._row_lengths))
            }
            new_row_lengths = self._row_lengths
            new_index = self.index

        if col_indices is not None:
            col_numeric_idx = self.columns.get_indexer_for(col_indices)
        if col_numeric_idx is not None:
            col_partitions_list = self._get_dict_of_block_index(0, col_numeric_idx)
            new_col_widths = [
                len(indices) for _, indices in col_partitions_list.items()
            ]
            new_columns = self.columns[sorted(col_numeric_idx)]
            if self._dtypes is not None:
                new_dtypes = self.dtypes[sorted(col_numeric_idx)]
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
            row_numeric_idx is None
            or len(row_numeric_idx) == 1
            or np.all(row_numeric_idx[1:] >= row_numeric_idx[:-1])
        ) and (
            col_numeric_idx is None
            or len(col_numeric_idx) == 1
            or np.all(col_numeric_idx[1:] >= col_numeric_idx[:-1])
        ):
            return intermediate
        # The new labels are often smaller than the old labels, so we can't reuse the
        # original order values because those were mapped to the original data. We have
        # to reorder here based on the expected order from within the data.
        # We create a dictionary mapping the position of the numeric index with respect
        # to all others, then recreate that order by mapping the new order values from
        # the old. This information is sent to `reorder_labels`.
        if row_numeric_idx is not None:
            row_order_mapping = dict(
                zip(sorted(row_numeric_idx), range(len(row_numeric_idx)))
            )
            new_row_order = [row_order_mapping[idx] for idx in row_numeric_idx]
        else:
            new_row_order = None
        if col_numeric_idx is not None:
            col_order_mapping = dict(
                zip(sorted(col_numeric_idx), range(len(col_numeric_idx)))
            )
            new_col_order = [col_order_mapping[idx] for idx in col_numeric_idx]
        else:
            new_col_order = None
        return intermediate.reorder_labels(
            row_numeric_idx=new_row_order, col_numeric_idx=new_col_order
        )

    def reorder_labels(self, row_numeric_idx=None, col_numeric_idx=None):
        """Reorder the column and or rows in this DataFrame.

        Parameters
        ----------
        row_numeric_idx : list of int, optional
            The ordered list of new row orders such that each position within the list
            indicates the new position.
        col_numeric_idx : list of int, optional
            The ordered list of new column orders such that each position within the
            list indicates the new position.

        Returns
        -------
        BasePandasFrame
            A new BasePandasFrame with reordered columns and/or rows.
        """
        if row_numeric_idx is not None:
            ordered_rows = self._frame_mgr_cls.map_axis_partitions(
                0, self._partitions, lambda df: df.iloc[row_numeric_idx]
            )
            row_idx = self.index[row_numeric_idx]
        else:
            ordered_rows = self._partitions
            row_idx = self.index
        if col_numeric_idx is not None:
            ordered_cols = self._frame_mgr_cls.map_axis_partitions(
                1, ordered_rows, lambda df: df.iloc[:, col_numeric_idx]
            )
            col_idx = self.columns[col_numeric_idx]
        else:
            ordered_cols = ordered_rows
            col_idx = self.columns
        return self.__constructor__(ordered_cols, row_idx, col_idx)

    def copy(self):
        """Copy this object.

        Returns:
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
        """Describes how data types should be combined when they do not match.

        Args:
            list_of_dtypes: A list of pandas Series with the data types.
            column_names: The names of the columns that the data types map to.

        Returns:
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

    def astype(self, col_dtypes):
        """Converts columns dtypes to given dtypes.

        Args:
            col_dtypes: Dictionary of {col: dtype,...} where col is the column
                name and dtype is a numpy dtype.

        Returns:
            dataframe with updated dtypes.
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

        def astype_builder(df):
            return df.astype({k: v for k, v in col_dtypes.items() if k in df})

        new_frame = self._frame_mgr_cls.map_partitions(self._partitions, astype_builder)
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
        """Add a prefix to the current row or column labels.

        Args:
            prefix: The prefix to add.
            axis: The axis to update.

        Returns:
            A new dataframe with the updated labels.
        """
        new_labels = self.axes[axis].map(lambda x: str(prefix) + str(x))
        new_frame = self.copy()
        if axis == 0:
            new_frame.index = new_labels
        else:
            new_frame.columns = new_labels
        return new_frame

    def add_suffix(self, suffix, axis):
        """Add a suffix to the current row or column labels.

        Args:
            suffix: The suffix to add.
            axis: The axis to update.

        Returns:
            A new dataframe with the updated labels.
        """
        new_labels = self.axes[axis].map(lambda x: str(x) + str(suffix))
        new_frame = self.copy()
        if axis == 0:
            new_frame.index = new_labels
        else:
            new_frame.columns = new_labels
        return new_frame

    # END Metadata modification methods

    def _numeric_columns(self, include_bool=True):
        """Returns the numeric columns of the Manager.

        Returns:
            List of index names.
        """
        columns = []
        for col, dtype in zip(self.columns, self.dtypes):
            if is_numeric_dtype(dtype) and (
                include_bool or (not include_bool and dtype != np.bool_)
            ):
                columns.append(col)
        return columns

    def _get_dict_of_block_index(self, axis, indices):
        """Convert indices to a dict of block index to internal index mapping.

        Parameters
        ----------
        axis : (0 - columns, 1 - rows)
               The axis along which to get the indices
        indices : list of int
                A list of global indices to convert.

        Returns
        -------
        dictionary mapping int to list of int
            A mapping from partition to list of internal indices to extract from that
            partition.
        """
        indices = np.sort(indices)
        if not axis:
            bins = np.array(self._column_widths)
        else:
            bins = np.array(self._row_lengths)
        # INT_MAX to make sure we don't try to compute on partitions that don't exist.
        cumulative = np.append(bins[:-1].cumsum(), np.iinfo(bins.dtype).max)

        def internal(block_idx, global_index):
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

    def _join_index_objects(self, axis, other_index, how, sort):
        """Joins a pair of index objects (columns or rows) by a given strategy.

        Args:
            axis: The axis index object to join (0 for columns, 1 for index).
            other_index: The other_index to join on.
            how: The type of join to join to make (e.g. right, left).

        Returns:
            Joined indices.
        """
        if isinstance(other_index, list):
            joined_obj = self.columns if not axis else self.index
            # TODO: revisit for performance
            for obj in other_index:
                joined_obj = joined_obj.join(obj, how=how, sort=sort)
            return joined_obj
        if not axis:
            return self.columns.join(other_index, how=how, sort=sort)
        else:
            return self.index.join(other_index, how=how, sort=sort)

    # Internal methods
    # These methods are for building the correct answer in a modular way.
    # Please be careful when changing these!

    def _build_mapreduce_func(self, axis, func):
        """Properly formats a MapReduce result so that the partitioning is correct.

        Note: This should be used for any MapReduce style operation that results in a
            reduced data dimensionality (dataframe -> series).

        Args:
            axis: The axis along which to apply the function.
            func: The function to apply.

        Returns:
            A function to be shipped to the partitions to be executed.
        """

        def _map_reduce_func(df):
            series_result = func(df)
            if axis == 0 and isinstance(series_result, pandas.Series):
                # In the case of axis=0, we need to keep the shape of the data
                # consistent with what we have done. In the case of a reduction, the
                # data for axis=0 should be a single value for each column. By
                # transposing the data after we convert to a DataFrame, we ensure that
                # the columns of the result line up with the columns from the data.
                # axis=1 does not have this requirement because the index already will
                # line up with the index of the data based on how pandas creates a
                # DataFrame from a Series.
                return pandas.DataFrame(series_result).T
            return pandas.DataFrame(series_result)

        return _map_reduce_func

    def _compute_map_reduce_metadata(self, axis, new_parts):
        if axis == 0:
            columns = self.columns
            index = ["__reduced__"]
            new_lengths = [1]
            new_widths = self._column_widths
            new_dtypes = self._dtypes
        else:
            columns = ["__reduced__"]
            index = self.index
            new_lengths = self._row_lengths
            new_widths = [1]
            if self._dtypes is not None:
                new_dtypes = pandas.Series(
                    np.full(1, find_common_type(self.dtypes.values)),
                    index=["__reduced__"],
                )
            else:
                new_dtypes = self._dtypes
        return self.__constructor__(
            new_parts, index, columns, new_lengths, new_widths, new_dtypes
        )

    def _fold_reduce(self, axis, func):
        """Applies map that reduce Manager to series but require knowledge of full axis.

        Args:
            func: Function to reduce the Manager by. This function takes in a Manager.
            axis: axis to apply the function to.

        Return:
            Pandas series containing the reduced data.
        """
        func = self._build_mapreduce_func(axis, func)
        new_parts = self._frame_mgr_cls.map_axis_partitions(
            axis, self._partitions, func
        )
        return self._compute_map_reduce_metadata(axis, new_parts)

    def _map_reduce(self, axis, map_func, reduce_func=None):
        """Apply function that will reduce the data to a Pandas Series.

        Args:
            axis: 0 for columns and 1 for rows. Default is 0.
            map_func: Callable function to map the dataframe.
            reduce_func: Callable function to reduce the dataframe. If none,
                then apply map_func twice.

        Return:
            A new dataframe.
        """
        map_func = self._build_mapreduce_func(axis, map_func)
        if reduce_func is None:
            reduce_func = map_func
        else:
            reduce_func = self._build_mapreduce_func(axis, reduce_func)

        map_parts = self._frame_mgr_cls.map_partitions(self._partitions, map_func)
        reduce_parts = self._frame_mgr_cls.map_axis_partitions(
            axis, map_parts, reduce_func
        )
        return self._compute_map_reduce_metadata(axis, reduce_parts)

    def _map(self, func, dtypes=None):
        """Perform a function that maps across the entire dataset.

        Args:
            func: The function to apply.
            dtypes: (optional) The data types for the result. This is an optimization
                because there are functions that always result in a particular data
                type, and allows us to avoid (re)computing it.
        Returns:
            A new dataframe.
        """
        new_partitions = self._frame_mgr_cls.map_partitions(self._partitions, func)
        if dtypes == "copy":
            dtypes = self._dtypes
        elif dtypes is not None:
            dtypes = pandas.Series(
                [np.dtype(dtypes)] * len(self.columns), index=self.columns
            )
        return self.__constructor__(
            new_partitions,
            self.index,
            self.columns,
            self._row_lengths,
            self._column_widths,
            dtypes=dtypes,
        )

    def _fold(self, axis, func):
        """Perform a function across an entire axis.

        Note: The data shape is not changed (length and width of the table).

        Args:
            axis: The axis to apply over.
            func: The function to apply.

        Returns:
             A new dataframe.
        """
        new_partitions = self._frame_mgr_cls.map_axis_partitions(
            axis, self._partitions, func
        )
        return self.__constructor__(
            new_partitions,
            self.index,
            self.columns,
            self._row_lengths,
            self._column_widths,
        )

    def filter_full_axis(self, axis, func):
        """Filter data based on the function provided along an entire axis.

        Args:
            axis: The axis to filter over.
            func: The function to use for the filter. This function should filter the
                data itself.

        Returns:
            A new dataframe.
        """
        new_partitions = self._frame_mgr_cls.map_axis_partitions(
            axis, self._partitions, func, keep_partitioning=True
        )
        if axis == 0:
            new_index = self.index
            new_lengths = self._row_lengths
            new_widths = None  # We do not know what the resulting widths will be
            new_columns = self._frame_mgr_cls.get_indices(
                1, new_partitions, lambda df: df.columns
            )
        else:
            new_columns = self.columns
            new_lengths = None  # We do not know what the resulting lengths will be
            new_widths = self._column_widths
            new_index = self._frame_mgr_cls.get_indices(
                0, new_partitions, lambda df: df.index
            )
        return self.__constructor__(
            new_partitions,
            new_index,
            new_columns,
            new_lengths,
            new_widths,
            self.dtypes if axis == 0 else None,
        )

    def _apply_full_axis(
        self, axis, func, new_index=None, new_columns=None, dtypes=None
    ):
        """Perform a function across an entire axis.

        Note: The data shape may change as a result of the function.

        Args:
            axis: The axis to apply over.
            func: The function to apply.
            new_index: (optional) The index of the result. We may know this in advance,
                and if not provided it must be computed.
            new_columns: (optional) The columns of the result. We may know this in
                advance, and if not provided it must be computed.
            dtypes: (optional) The data types of the result. This is an optimization
                because there are functions that always result in a particular data
                type, and allows us to avoid (re)computing it.

        Returns:
            A new dataframe.
        """
        new_partitions = self._frame_mgr_cls.map_axis_partitions(
            axis,
            self._partitions,
            self._build_mapreduce_func(axis, func),
            keep_partitioning=True,
        )
        # Index objects for new object creation. This is shorter than if..else
        if new_columns is None:
            new_columns = self._frame_mgr_cls.get_indices(
                1, new_partitions, lambda df: df.columns
            )
        if new_index is None:
            new_index = self._frame_mgr_cls.get_indices(
                0, new_partitions, lambda df: df.index
            )
        if dtypes == "copy":
            dtypes = self._dtypes
        elif dtypes is not None:
            dtypes = pandas.Series(
                [np.dtype(dtypes)] * len(new_columns), index=new_columns
            )
        return self.__constructor__(
            new_partitions, new_index, new_columns, None, None, dtypes
        )

    def _apply_full_axis_select_indices(
        self,
        axis,
        func,
        apply_indices=None,
        numeric_indices=None,
        new_index=None,
        new_columns=None,
        keep_remaining=False,
    ):
        """Apply a function across an entire axis for a subset of the data.

        Args:
            axis: The axis to apply over.
            func: The function to apply
            apply_indices: The labels to apply over.
            numeric_indices: The indices to apply over.
            new_index: (optional) The index of the result. We may know this in advance,
                and if not provided it must be computed.
            new_columns: (optional) The columns of the result. We may know this in
                advance, and if not provided it must be computed.
            keep_remaining: Whether or not to drop the data that is not computed over.

        Returns:
            A new dataframe.
        """ 
        assert apply_indices is not None or numeric_indices is not None
        # Convert indices to numeric indices
        old_index = self.index if axis else self.columns
        if apply_indices is not None:
            numeric_indices = old_index.get_indexer_for(apply_indices)
        dict_indices = self._get_dict_of_block_index(axis, numeric_indices)
        new_partitions = self._frame_mgr_cls.apply_func_to_select_indices_along_full_axis(
            axis, self._partitions, func, dict_indices, keep_remaining=keep_remaining
        )
        # TODO Infer columns and index from `keep_remaining` and `apply_indices`
        if new_index is None:
            new_index = self.index if axis == 1 else None
        if new_columns is None:
            new_columns = self.columns if axis == 0 else None
        return self.__constructor__(new_partitions, new_index, new_columns, None, None)

    def _apply_select_indices(
        self,
        axis,
        func,
        apply_indices=None,
        row_indices=None,
        col_indices=None,
        new_index=None,
        new_columns=None,
        keep_remaining=False,
        item_to_distribute=None,
    ):
        """Apply a function for a subset of the data.

        Args:
            axis: The axis to apply over.
            func: The function to apply
            apply_indices: (optional) The labels to apply over. Must be given if axis is
                provided.
            row_indices: (optional) The row indices to apply over. Must be provided with
                `col_indices` to apply over both axes.
            col_indices: (optional) The column indices to apply over. Must be provided
                with `row_indices` to apply over both axes.
            new_index: (optional) The index of the result. We may know this in advance,
                and if not provided it must be computed.
            new_columns: (optional) The columns of the result. We may know this in
                advance, and if not provided it must be computed.
            keep_remaining: Whether or not to drop the data that is not computed over.
            item_to_distribute: (optional) The item to split up so it can be applied
                over both axes.

        Returns:
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
            dict_indices = self._get_dict_of_block_index(axis, numeric_indices)
            new_partitions = self._frame_mgr_cls.apply_func_to_select_indices(
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
            # `axis` given may have changed, we current just recompute it.
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
            # We are apply over both axes here, so make sure we have all the right
            # variables set.
            assert row_indices is not None and col_indices is not None
            assert keep_remaining
            assert item_to_distribute is not None
            row_partitions_list = self._get_dict_of_block_index(1, row_indices).items()
            col_partitions_list = self._get_dict_of_block_index(0, col_indices).items()
            new_partitions = self._frame_mgr_cls.apply_func_to_indices_both_axis(
                self._partitions,
                func,
                row_partitions_list,
                col_partitions_list,
                item_to_distribute,
            )
            return self.__constructor__(
                new_partitions,
                new_index,
                new_columns,
                self._row_lengths_cache,
                self._column_widths_cache,
            )

    def broadcast_apply(self, axis, func, other, preserve_labels=True, dtypes=None):
        """Broadcast partitions of other dataframe partitions and apply a function.

        Args:
            axis: The axis to broadcast over.
            func: The function to apply.
            other: The Modin DataFrame to broadcast.
            preserve_labels: Whether or not to keep labels from this Modin DataFrame.
            dtypes: "copy" or None. Whether to keep old dtypes or infer new dtypes from
                data.

        Returns:
             A new Modin DataFrame
        """
        assert preserve_labels, "`preserve_labels=False` Not Yet Implemented"
        # Only sort the indices if they do not match
        left_parts, right_parts, joined_index = self._copartition(
            axis, other, "left", sort=not self.axes[axis].equals(other.axes[axis])
        )
        # unwrap list returned by `copartition`.
        right_parts = right_parts[0]
        new_frame = self._frame_mgr_cls.broadcast_apply(
            axis, func, left_parts, right_parts
        )
        if dtypes == "copy":
            dtypes = self._dtypes
        new_index = self.index
        new_columns = self.columns
        return self.__constructor__(
            new_frame, new_index, new_columns, None, None, dtypes=dtypes
        )

    def _copartition(self, axis, other, how, sort, force_repartition=False):
        """Copartition two dataframes.

        Args:
            axis: The axis to copartition along.
            other: The other dataframes(s) to copartition against.
            how: How to manage joining the index object ("left", "right", etc.)
            sort: Whether or not to sort the joined index.
            force_repartition: Whether or not to force the repartitioning. By default,
                this method will skip repartitioning if it is possible. This is because
                reindexing is extremely inefficient. Because this method is used to
                `join` or `append`, it is vital that the internal indices match.

        Returns:
            A tuple (left data, right data list, joined index).
        """
        if isinstance(other, type(self)):
            other = [other]

        index_obj = [o.axes[axis] for o in other]
        joined_index = self._join_index_objects(axis ^ 1, index_obj, how, sort)
        # We have to set these because otherwise when we perform the functions it may
        # end up serializing this entire object.
        left_old_idx = self.axes[axis]
        right_old_idxes = index_obj

        # Start with this and we'll repartition the first time, and then not again.
        if not left_old_idx.equals(joined_index) or force_repartition:
            reindexed_self = self._frame_mgr_cls.map_axis_partitions(
                axis, self._partitions, lambda df: df.reindex(joined_index, axis=axis)
            )
        else:
            reindexed_self = self._partitions
        reindexed_other_list = []

        for i in range(len(other)):
            if right_old_idxes[i].equals(joined_index) and not force_repartition:
                reindexed_other = other[i]._partitions
            else:
                reindexed_other = other[i]._frame_mgr_cls.map_axis_partitions(
                    axis,
                    other[i]._partitions,
                    lambda df: df.reindex(joined_index, axis=axis),
                )
            reindexed_other_list.append(reindexed_other)
        return reindexed_self, reindexed_other_list, joined_index

    def _binary_op(self, op, right_frame, join_type="outer"):
        """Perform an operation that requires joining with another dataframe.

        Args:
            op: The function to apply after the join.
            right_frame: The dataframe to join with.
            join_type: (optional) The type of join to apply.

        Returns:
             A new dataframe.
        """
        left_parts, right_parts, joined_index = self._copartition(
            0, right_frame, join_type, sort=True
        )
        # unwrap list returned by `copartition`.
        right_parts = right_parts[0]
        new_frame = self._frame_mgr_cls.binary_operation(
            1, left_parts, lambda l, r: op(l, r), right_parts
        )
        new_columns = self.columns.join(right_frame.columns, how=join_type)
        return self.__constructor__(new_frame, self.index, new_columns, None, None)

    def _concat(self, axis, others, how, sort):
        """Concatenate this dataframe with one or more others.

        Args:
            axis: The axis to concatenate over.
            others: The list of dataframes to concatenate with.
            how: The type of join to use for the axis.
            sort: Whether or not to sort the result.

        Returns:
            A new dataframe.
        """
        # Fast path for equivalent columns and partitioning
        if (
            axis == 0
            and all(o.columns.equals(self.columns) for o in others)
            and all(o._column_widths == self._column_widths for o in others)
        ):
            joined_index = self.columns
            left_parts = self._partitions
            right_parts = [o._partitions for o in others]
            new_lengths = self._row_lengths + [
                length for o in others for length in o._row_lengths
            ]
            new_widths = self._column_widths
        elif (
            axis == 1
            and all(o.index.equals(self.index) for o in others)
            and all(o._row_lengths == self._row_lengths for o in others)
        ):
            joined_index = self.index
            left_parts = self._partitions
            right_parts = [o._partitions for o in others]
            new_lengths = self._row_lengths
            new_widths = self._column_widths + [
                length for o in others for length in o._column_widths
            ]
        else:
            left_parts, right_parts, joined_index = self._copartition(
                axis ^ 1, others, how, sort, force_repartition=True
            )
            new_lengths = None
            new_widths = None
        new_partitions = self._frame_mgr_cls.concat(axis, left_parts, right_parts)
        if axis == 0:
            new_index = self.index.append([other.index for other in others])
            new_columns = joined_index
            # TODO: Can optimize by combining if all dtypes are materialized
            new_dtypes = None
        else:
            new_columns = self.columns.append([other.columns for other in others])
            new_index = joined_index
            if self._dtypes is not None and all(o._dtypes is not None for o in others):
                new_dtypes = self.dtypes.append([o.dtypes for o in others])
            else:
                new_dtypes = None
        return self.__constructor__(
            new_partitions, new_index, new_columns, new_lengths, new_widths, new_dtypes
        )

    def groupby_reduce(
        self, axis, by, map_func, reduce_func, new_index=None, new_columns=None
    ):
        """Groupby another dataframe and aggregate the result.

        Args:
            axis: The axis to groupby and aggregate over.
            by: The dataframe to group by.
            map_func: The map component of the aggregation.
            reduce_func: The reduce component of the aggregation.
            new_index: (optional) The index of the result. We may know this in advance,
                and if not provided it must be computed.
            new_columns: (optional) The columns of the result. We may know this in
                advance, and if not provided it must be computed.

        Returns:
             A new dataframe.
        """
        new_partitions = self._frame_mgr_cls.groupby_reduce(
            axis, self._partitions, by._partitions, map_func, reduce_func
        )
        if new_columns is None:
            new_columns = self._frame_mgr_cls.get_indices(
                1, new_partitions, lambda df: df.columns
            )
        if new_index is None:
            new_index = self._frame_mgr_cls.get_indices(
                0, new_partitions, lambda df: df.index
            )
        return self.__constructor__(new_partitions, new_index, new_columns)

    @classmethod
    def from_pandas(cls, df):
        """Improve simple Pandas DataFrame to an advanced and superior Modin DataFrame.

        Args:
            df: Pandas DataFrame object.

        Returns:
            A new dataframe.
        """
        new_index = df.index
        new_columns = df.columns
        new_dtypes = df.dtypes
        new_frame, new_lengths, new_widths = cls._frame_mgr_cls.from_pandas(df, True)
        return cls(
            new_frame,
            new_index,
            new_columns,
            new_lengths,
            new_widths,
            dtypes=new_dtypes,
        )

    def to_pandas(self):
        """Converts Modin DataFrame to Pandas DataFrame.

        Returns:
            Pandas DataFrame.
        """
        df = self._frame_mgr_cls.to_pandas(self._partitions)
        if df.empty:
            if len(self.columns) != 0:
                df = pandas.DataFrame(columns=self.columns)
            else:
                df = pandas.DataFrame(columns=self.columns, index=self.index)
        df.index.name = self.index.name
        return df

    def to_numpy(self):
        """Converts Modin DataFrame to a 2D NumPy array.

        Returns:
            NumPy array.
        """
        return self._frame_mgr_cls.to_numpy(self._partitions)

    def transpose(self):
        """Transpose the index and columns of this dataframe.

        Returns:
            A new dataframe.
        """
        new_partitions = self._frame_mgr_cls.lazy_map_partitions(
            self._partitions, lambda df: df.T
        ).T
        new_dtypes = pandas.Series(
            np.full(len(self.index), find_common_type(self.dtypes.values)),
            index=self.index,
        )
        return self.__constructor__(
            new_partitions,
            self.columns,
            self.index,
            self._column_widths,
            self._row_lengths,
            dtypes=new_dtypes,
        )
