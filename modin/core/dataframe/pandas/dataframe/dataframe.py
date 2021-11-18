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
from pandas.core.indexes.api import ensure_index, Index, RangeIndex
from pandas.core.dtypes.common import is_numeric_dtype, is_list_like
from typing import List, Hashable

from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.error_message import ErrorMessage
from modin.core.storage_formats.pandas.parsers import (
    find_common_type_cat as find_common_type,
)
from modin.pandas.indexing import is_range_like


class PandasDataframe(object):
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
        Compute the data types via MapReduce pattern.

        Returns
        -------
        pandas.Series
            A pandas Series containing the data types for this dataframe.
        """

        def dtype_builder(df):
            return df.apply(lambda col: find_common_type(col.values), axis=0)

        map_func = self._build_mapreduce_func(0, lambda df: df.dtypes)
        reduce_func = self._build_mapreduce_func(0, dtype_builder)
        # For now we will use a pandas Series for the dtypes.
        if len(self.columns) > 0:
            dtypes = self.map_reduce(0, map_func, reduce_func).to_pandas().iloc[0]
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
                "Length mismatch: Expected axis has %d elements, "
                "new values have %d elements" % (old_len, new_len)
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
        """
        Lazily select columns or rows from given indices.

        Parameters
        ----------
        row_indices : list of hashable, optional
            The row labels to extract.
        row_numeric_idx : list-like of ints, optional
            The row indices to extract.
        col_indices : list of hashable, optional
            The column labels to extract.
        col_numeric_idx : list-like of ints, optional
            The column indices to extract.

        Returns
        -------
        PandasDataframe
             A new PandasDataframe from the mask provided.

        Notes
        -----
        If both `row_indices` and `row_numeric_idx` are set, `row_indices` will be used.
        The same rule applied to `col_indices` and `col_numeric_idx`.
        """
        indexers = []
        for axis, indexer in enumerate((row_numeric_idx, col_numeric_idx)):
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
        row_numeric_idx, col_numeric_idx = indexers

        if (
            row_indices is None
            and row_numeric_idx is None
            and col_indices is None
            and col_numeric_idx is None
        ):
            return self.copy()
        # Get numpy array of positions of values from `row_indices`
        if row_indices is not None:
            row_numeric_idx = self.index.get_indexer_for(row_indices)
        if row_numeric_idx is not None:
            # Get dict of row_parts as {row_index: row_internal_indices}
            # TODO: Rename `row_partitions_list`->`row_partitions_dict`
            row_partitions_list = self._get_dict_of_block_index(0, row_numeric_idx)
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
                slice(row_numeric_idx.start, row_numeric_idx.stop, row_numeric_idx.step)
                # TODO: Fast range processing of non-1-step ranges is not yet supported
                if is_range_like(row_numeric_idx) and row_numeric_idx.step > 0
                else sorted(row_numeric_idx)
            ]
        else:
            row_partitions_list = {
                i: slice(None) for i in range(len(self._row_lengths))
            }
            new_row_lengths = self._row_lengths
            new_index = self.index

        # Get numpy array of positions of values from `col_indices`
        if col_indices is not None:
            col_numeric_idx = self.columns.get_indexer_for(col_indices)
        if col_numeric_idx is not None:
            # Get dict of col_parts as {col_index: col_internal_indices}
            col_partitions_list = self._get_dict_of_block_index(1, col_numeric_idx)
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
            if is_range_like(col_numeric_idx) and col_numeric_idx.step > 0:
                # pandas Index is more likely to preserve its metadata if the indexer is slice
                monotonic_col_idx = slice(
                    col_numeric_idx.start, col_numeric_idx.stop, col_numeric_idx.step
                )
            else:
                monotonic_col_idx = sorted(col_numeric_idx)
            new_columns = self.columns[monotonic_col_idx]
            ErrorMessage.catch_bugs_and_request_email(
                failure_condition=sum(new_col_widths) != len(new_columns),
                extra_log=f"{sum(new_col_widths)} != {len(new_columns)}.\n{col_numeric_idx}\n{self._column_widths}\n{col_partitions_list}",
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
            row_numeric_idx is None
            # Fast range processing of non-1-step ranges is not yet supported
            or (is_range_like(row_numeric_idx) and row_numeric_idx.step > 0)
            or len(row_numeric_idx) == 1
            or np.all(row_numeric_idx[1:] >= row_numeric_idx[:-1])
        ) and (
            col_numeric_idx is None
            # Fast range processing of non-1-step ranges is not yet supported
            or (is_range_like(col_numeric_idx) and col_numeric_idx.step > 0)
            or len(col_numeric_idx) == 1
            or np.all(col_numeric_idx[1:] >= col_numeric_idx[:-1])
        ):
            return intermediate
        # The new labels are often smaller than the old labels, so we can't reuse the
        # original order values because those were mapped to the original data. We have
        # to reorder here based on the expected order from within the data.
        # We create a dictionary mapping the position of the numeric index with respect
        # to all others, then recreate that order by mapping the new order values from
        # the old. This information is sent to `_reorder_labels`.
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
        return intermediate._reorder_labels(
            row_numeric_idx=new_row_order, col_numeric_idx=new_col_order
        )

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
            return df.reset_index()

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
        # Propagate the new row labels to the all dataframe partitions
        result.synchronize_labels(0)
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
        extracted_columns = self.mask(col_indices=column_list).to_pandas()
        if len(column_list) == 1:
            new_labels = pandas.Index(extracted_columns.squeeze(axis=1))
        else:
            new_labels = pandas.MultiIndex.from_frame(extracted_columns)
        result = self.mask(
            col_indices=[i for i in self.columns if i not in column_list]
        )
        result.index = new_labels
        return result

    def _reorder_labels(self, row_numeric_idx=None, col_numeric_idx=None):
        """
        Reorder the column and or rows in this DataFrame.

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
        PandasDataframe
            A new PandasDataframe with reordered columns and/or rows.
        """
        if row_numeric_idx is not None:
            ordered_rows = self._partition_mgr_cls.map_axis_partitions(
                0, self._partitions, lambda df: df.iloc[row_numeric_idx]
            )
            row_idx = self.index[row_numeric_idx]
        else:
            ordered_rows = self._partitions
            row_idx = self.index
        if col_numeric_idx is not None:
            ordered_cols = self._partition_mgr_cls.map_axis_partitions(
                1, ordered_rows, lambda df: df.iloc[:, col_numeric_idx]
            )
            col_idx = self.columns[col_numeric_idx]
        else:
            ordered_cols = ordered_rows
            col_idx = self.columns
        return self.__constructor__(ordered_cols, row_idx, col_idx)

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
        new_labels = self.axes[axis].map(lambda x: str(prefix) + str(x))
        new_frame = self.copy()
        if axis == 0:
            new_frame.index = new_labels
        else:
            new_frame.columns = new_labels
        return new_frame

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
        new_labels = self.axes[axis].map(lambda x: str(x) + str(suffix))
        new_frame = self.copy()
        if axis == 0:
            new_frame.index = new_labels
        else:
            new_frame.columns = new_labels
        return new_frame

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

    def _get_dict_of_block_index(self, axis, indices):
        """
        Convert indices to an ordered dict mapping partition (or block) index to internal indices in said partition.

        Parameters
        ----------
        axis : {0, 1}
            The axis along which to get the indices (0 - rows, 1 - columns).
        indices : list of int, slice
            A list of global indices to convert.

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
            if indices == slice(None) or indices == slice(0, None):
                return OrderedDict(
                    zip(
                        range(self._partitions.shape[axis]),
                        [slice(None)] * self._partitions.shape[axis],
                    )
                )
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
        # Sort and convert negative indices to positive
        indices = np.sort(
            [i if i >= 0 else max(0, len(self.axes[axis]) + i) for i in indices]
        )
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

    def _build_mapreduce_func(self, axis, func):
        """
        Properly formats a MapReduce result so that the partitioning is correct.

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
        This should be used for any MapReduce style operation that results in a
        reduced data dimensionality (dataframe -> series).
        """

        def _map_reduce_func(df, *args, **kwargs):
            """Map-reducer function itself executing `func`, presenting the resulting pandas.Series as pandas.DataFrame."""
            series_result = func(df, *args, **kwargs)
            if axis == 0 and isinstance(series_result, pandas.Series):
                # In the case of axis=0, we need to keep the shape of the data
                # consistent with what we have done. In the case of a reduction, the
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

        return _map_reduce_func

    def _compute_map_reduce_metadata(self, axis, new_parts):
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

    def fold_reduce(self, axis, func):
        """
        Apply function that reduces Frame Manager to series but requires knowledge of full axis.

        Parameters
        ----------
        axis : {0, 1}
            The axis to apply the function to (0 - index, 1 - columns).
        func : callable
            The function to reduce the Manager by. This function takes in a Manager.

        Returns
        -------
        PandasDataframe
            Modin series (1xN frame) containing the reduced data.
        """
        func = self._build_mapreduce_func(axis, func)
        new_parts = self._partition_mgr_cls.map_axis_partitions(
            axis, self._partitions, func
        )
        return self._compute_map_reduce_metadata(axis, new_parts)

    def map_reduce(self, axis, map_func, reduce_func=None):
        """
        Apply function that will reduce the data to a pandas Series.

        Parameters
        ----------
        axis : {0, 1}
            0 for columns and 1 for rows.
        map_func : callable
            Callable function to map the dataframe.
        reduce_func : callable, default: None
            Callable function to reduce the dataframe.
            If none, then apply map_func twice.

        Returns
        -------
        PandasDataframe
            A new dataframe.
        """
        map_func = self._build_mapreduce_func(axis, map_func)
        if reduce_func is None:
            reduce_func = map_func
        else:
            reduce_func = self._build_mapreduce_func(axis, reduce_func)

        map_parts = self._partition_mgr_cls.map_partitions(self._partitions, map_func)
        reduce_parts = self._partition_mgr_cls.map_axis_partitions(
            axis, map_parts, reduce_func
        )
        return self._compute_map_reduce_metadata(axis, reduce_parts)

    def map(self, func, dtypes=None):
        """
        Perform a function that maps across the entire dataset.

        Parameters
        ----------
        func : callable
            The function to apply.
        dtypes : dtypes of the result, default: None
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

    def filter_full_axis(self, axis, func):
        """
        Filter data based on the function provided along an entire axis.

        Parameters
        ----------
        axis : int
            The axis to filter over.
        func : callable
            The function to use for the filter. This function should filter the
            data itself.

        Returns
        -------
        PandasDataframe
            A new filtered dataframe.
        """
        new_partitions = self._partition_mgr_cls.map_axis_partitions(
            axis, self._partitions, func, keep_partitioning=True
        )
        new_axes, new_lengths = [0, 0], [0, 0]

        new_axes[axis] = self.axes[axis]
        new_axes[axis ^ 1] = self._compute_axis_labels(axis ^ 1, new_partitions)

        new_lengths[axis] = self._axes_lengths[axis]
        new_lengths[axis ^ 1] = None  # We do not know what the resulting widths will be

        return self.__constructor__(
            new_partitions,
            *new_axes,
            *new_lengths,
            self.dtypes if axis == 0 else None,
        )

    def explode(self, axis, func):
        """
        Explode list-like entries along an entire axis.

        Parameters
        ----------
        axis : int
            The axis specifying how to explode. If axis=1, explode according
            to columns.
        func : callable
            The function to use to explode a single element.

        Returns
        -------
        PandasFrame
            A new filtered dataframe.
        """
        partitions = self._partition_mgr_cls.map_axis_partitions(
            axis, self._partitions, func, keep_partitioning=True
        )
        if axis == 1:
            new_index = self._compute_axis_labels(0, partitions)
            new_columns = self.columns
        else:
            new_index = self.index
            new_columns = self._compute_axis_labels(1, partitions)
        return self.__constructor__(partitions, new_index, new_columns)

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

    def apply_select_indices(
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
        row_indices : list-like, default: None
            The row indices to apply over. Must be provided with
            `col_indices` to apply over both axes.
        col_indices : list-like, default: None
            The column indices to apply over. Must be provided
            with `row_indices` to apply over both axes.
        new_index : list-like, optional
            The index of the result. We may know this in advance,
            and if not provided it must be computed.
        new_columns : list-like, optional
            The columns of the result. We may know this in
            advance, and if not provided it must be computed.
        keep_remaining : boolean, default: False
            Whether or not to drop the data that is not computed over.
        item_to_distribute : (optional)
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
            assert row_indices is not None and col_indices is not None
            assert keep_remaining
            assert item_to_distribute is not None
            row_partitions_list = self._get_dict_of_block_index(0, row_indices).items()
            col_partitions_list = self._get_dict_of_block_index(1, col_indices).items()
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
        left_parts, right_parts, joined_index = self._copartition(
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
        if not preserve_labels:
            if axis == 1:
                new_columns = joined_index
            else:
                new_index = joined_index
        return self.__constructor__(
            new_frame, new_index, new_columns, None, None, dtypes=dtypes
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

            def get_len(part):
                return part.width() if not axis else part.length()

            parts = self._partitions if not axis else self._partitions.T
            return {
                key: {
                    i: np.arange(get_len(parts[0][i])) for i in np.arange(len(parts[0]))
                }
                for key in indices.keys()
            }
        passed_len = 0
        result_dict = {}
        for part_num, internal in indices.items():
            result_dict[part_num] = self._get_dict_of_block_index(
                axis ^ 1, np.arange(passed_len, passed_len + len(internal))
            )
            passed_len += len(internal)
        return result_dict

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
            apply_func=self._build_mapreduce_func(axis, func),
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
            result.synchronize_labels(0)
        if new_columns is not None:
            result.synchronize_labels(1)
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
            Tuple of (left data, right data list, joined index).
        """
        if isinstance(other, type(self)):
            other = [other]

        # define helper functions
        def get_axis_lengths(partitions, axis):
            if axis:
                return [obj.width() for obj in partitions[0]]
            return [obj.length() for obj in partitions.T[0]]

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
            return self._partitions, [o._partitions for o in other], joined_index

        base_frame_idx = non_empty_frames_idx[0]
        other_frames = frames[base_frame_idx + 1 :]

        # Picking first non-empty frame
        base_frame = frames[non_empty_frames_idx[0]]
        base_index = base_frame.axes[axis]

        # define conditions for reindexing and repartitioning `self` frame
        do_reindex_base = not base_index.equals(joined_index)
        do_repartition_base = force_repartition or do_reindex_base

        # perform repartitioning and reindexing for `base_frame` if needed
        if do_repartition_base:
            reindexed_base = base_frame._partition_mgr_cls.map_axis_partitions(
                axis,
                base_frame._partitions,
                make_reindexer(do_reindex_base, base_frame_idx),
            )
        else:
            reindexed_base = base_frame._partitions

        # define length of base and `other` frames to aligning purpose
        base_lengths = get_axis_lengths(reindexed_base, axis)
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
        return reindexed_frames[0], reindexed_frames[1:], joined_index

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
        left_parts, right_parts, joined_index = self._copartition(
            0, right_frame, join_type, sort=True
        )
        # unwrap list returned by `copartition`.
        right_parts = right_parts[0]
        new_frame = self._partition_mgr_cls.binary_operation(
            1, left_parts, lambda l, r: op(l, r), right_parts
        )
        new_columns = self.columns.join(right_frame.columns, how=join_type)
        return self.__constructor__(new_frame, joined_index, new_columns, None, None)

    def concat(self, axis, others, how, sort):
        """
        Concatenate `self` with one or more other Modin DataFrames.

        Parameters
        ----------
        axis : {0, 1}
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
                axis ^ 1, others, how, sort, force_repartition=False
            )
            new_lengths = None
            new_widths = None
        new_partitions = self._partition_mgr_cls.concat(axis, left_parts, right_parts)
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
        res = arrow_type.to_pandas_dtype()
        if not isinstance(res, (np.dtype, str)):
            return np.dtype(res)
        return res

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

    def finalize(self):
        """
        Perform all deferred calls on partitions.

        This makes `self` Modin Dataframe independent of a history of queries
        that were used to build it.
        """
        self._partition_mgr_cls.finalize(self._partitions)
