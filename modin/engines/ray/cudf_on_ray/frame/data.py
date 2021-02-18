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

import numpy as np

from .partition import cuDFOnRayFramePartition
from .partition_manager import cuDFOnRayFrameManager

from modin.engines.base.frame.data import BasePandasFrame
from modin.error_message import ErrorMessage


class cuDFOnRayFrame(BasePandasFrame):

    _frame_mgr_cls = cuDFOnRayFrameManager

    def _apply_index_objs(self, axis=None):
        """Eagerly applies the index object (Index or Columns) to the partitions.

        Args:
            axis: The axis to apply to, None applies to both axes.

        Returns
        -------
            A new 2D array of partitions that have the index assignment added to the
            call queue.
        """
        ErrorMessage.catch_bugs_and_request_email(
            axis is not None and axis not in [0, 1]
        )

        cum_row_lengths = np.cumsum([0] + self._row_lengths)
        cum_col_widths = np.cumsum([0] + self._column_widths)

        def apply_idx_objs(df, idx, cols, axis):
            # cudf does not support set_axis. It only supports rename with 1-to-1 mapping.
            # Therefore, we need to create the dictionary that have the relationship between
            # current index and new ones.
            idx = {df.index[i]: idx[i] for i in range(len(idx))}
            cols = {df.index[i]: cols[i] for i in range(len(cols))}

            if axis == 0:
                return df.rename(index=idx)
            elif axis == 1:
                return df.rename(columns=cols)
            else:
                return df.rename(index=idx, columns=cols)

        keys = np.array(
            [
                [
                    self._partitions[i][j].apply(
                        apply_idx_objs,
                        idx=self.index[
                            slice(cum_row_lengths[i], cum_row_lengths[i + 1])
                        ],
                        cols=self.columns[
                            slice(cum_col_widths[j], cum_col_widths[j + 1])
                        ],
                        axis=axis
                    )
                    for j in range(len(self._partitions[i]))
                ]
                for i in range(len(self._partitions))
            ]
        )

        self._partitions = np.array(
            [
                [
                    cuDFOnRayFramePartition(
                        self._partitions[i][j].get_gpu_manager(),
                        keys[i][j],
                        self._partitions[i][j]._length_cache,
                        self._partitions[i][j]._width_cache
                    )
                    for j in range(len(keys[i]))
                ]
                for i in range(len(keys))
            ]
        )

    # _query_compiler_cls = cuDFQueryCompiler

    # @property
    # def __constructor__(self):
    #     """The constructor for this object. A convenience method"""
    #     return type(self)

    # def __init__(
    #     self,
    #     partitions,
    #     index,
    #     columns,
    #     row_lengths=None,
    #     column_widths=None,
    #     dtypes=None,
    #     validate_axes: Union[bool, str] = False,
    # ):
    #     self._partitions = partitions

    #     self._index_cache = ensure_index(index)
    #     self._columns_cache = ensure_index(columns)

    #     if row_lengths is not None and len(self.index) > 0:
    #         ErrorMessage.catch_bugs_and_request_email(
    #             sum(row_lengths) != len(self._index_cache),
    #             "Row lengths: {} != {}".format(
    #                 sum(row_lengths), len(self._index_cache)
    #             ),
    #         )
    #     self._row_lengths_cache = row_lengths
    #     if column_widths is not None and len(self.columns) > 0:
    #         ErrorMessage.catch_bugs_and_request_email(
    #             sum(column_widths) != len(self._columns_cache),
    #             "Column widths: {} != {}".format(
    #                 sum(column_widths), len(self._columns_cache)
    #             ),
    #         )
    #     self._column_widths_cache = column_widths
    #     self._dtypes = dtypes
    #     # self._filter_empties()
    #     if validate_axes is not False:
    #         self._validate_internal_indices(mode=validate_axes)

    # def groupby_reduce(
    #     self, axis, by, map_func, reduce_func, new_index=None, new_columns=None
    # ):
    #     new_partitions = self._frame_mgr_cls.groupby_reduce(
    #         axis, self._partitions, by._partitions, map_func, reduce_func
    #     )
    #     if new_columns is None:
    #         new_columns = self._frame_mgr_cls.get_indices(
    #             1, new_partitions, lambda df: df.columns
    #         )
    #     if new_index is None:
    #         new_index = self._frame_mgr_cls.get_indices(
    #             0, new_partitions, lambda df: df.index.to_pandas()
    #         )
    #     return self.__constructor__(new_partitions, new_index, new_columns)

    # @classmethod
    # def from_pandas(cls, pandas_dataframe):
    #     new_index = pandas_dataframe.index
    #     new_columns = pandas_dataframe.columns
    #     new_dtypes = pandas_dataframe.dtypes
    #     new_frame, new_lengths, new_widths = cls._frame_mgr_cls.from_pandas(pandas_dataframe, True)
    #     return cls(
    #         new_frame,
    #         new_index,
    #         new_columns,
    #         new_lengths,
    #         new_widths,
    #         dtypes=new_dtypes,
    #     )

    # def to_pandas(self):
    #     df = self._frame_mgr_cls.to_pandas(self._partitions)
    #     if df.empty:
    #         if len(self.columns) != 0:
    #             df = pandas.DataFrame(columns=self.columns)
    #         else:
    #             df = pandas.DataFrame(columns=self.columns, index=self.index)
    #     else:
    #         # TODO (kvu35): Enable error checking
    #         #ErrorMessage.catch_bugs_and_request_email(
    #         #    not df.index.equals(self.index) or not df.columns.equals(self.columns),
    #         #    "Internal and external indices do not match.",
    #         #)
    #         df.index = self.index
    #         df.columns = self.columns
    #     return df

    # def isna(self):
    #     new_partitions = self._frame_mgr_cls.isna(self._partitions)
    #     new_index = self.index
    #     new_row_lengths = self._row_lengths
    #     new_columns = self.columns
    #     new_column_widths = self._column_widths
    #     return self.__constructor__(
    #         new_partitions,
    #         new_index,
    #         new_columns,
    #         new_row_lengths,
    #         new_column_widths,
    #         dtypes=None,
    #     )

    # def mask(
    #     self,
    #     row_indices=None,
    #     row_numeric_idx=None,
    #     col_indices=None,
    #     col_numeric_idx=None,
    # ):
    #     if isinstance(row_numeric_idx, slice) and (
    #             row_numeric_idx == slice(None) or row_numeric_idx == slice(0, None)
    #     ):
    #         row_numeric_idx = None
    #     if isinstance(col_numeric_idx, slice) and (
    #         col_numeric_idx == slice(None) or col_numeric_idx == slice(0, None)
    #     ):
    #         col_numeric_idx = None
    #     if (
    #         row_indices is None
    #         and row_numeric_idx is None
    #         and col_indices is None
    #         and col_numeric_idx is None
    #     ):
    #         return self.copy()
    #     if row_indices is not None:
    #         row_numeric_idx = self.index.get_indexer_for(row_indices)
    #     if row_numeric_idx is not None:
    #         row_partitions_list = self._get_dict_of_block_index(0, row_numeric_idx)
    #         if isinstance(row_numeric_idx, slice):
    #             # Row lengths for slice are calculated as the length of the slice
    #             # on the partition. Often this will be the same length as the current
    #             # length, but sometimes it is different, thus the extra calculation.
    #             new_row_lengths = [
    #                 len(range(*idx.indices(self._row_lengths[p])))
    #                 for p, idx in row_partitions_list.items()
    #             ]
    #             # Use the slice to calculate the new row index
    #             new_index = self.index[row_numeric_idx]
    #         else:
    #             new_row_lengths = [len(idx) for _, idx in row_partitions_list.items()]
    #             new_index = self.index[sorted(row_numeric_idx)]
    #     else:
    #         row_partitions_list = {
    #             i: slice(None) for i in range(len(self._row_lengths))
    #         }
    #         new_row_lengths = self._row_lengths
    #         new_index = self.index

    #     if col_indices is not None:
    #         col_numeric_idx = self.columns.get_indexer_for(col_indices)
    #     if col_numeric_idx is not None:
    #         col_partitions_list = self._get_dict_of_block_index(1, col_numeric_idx)
    #         if isinstance(col_numeric_idx, slice):
    #             # Column widths for slice are calculated as the length of the slice
    #             # on the partition. Often this will be the same length as the current
    #             # length, but sometimes it is different, thus the extra calculation.
    #             new_col_widths = [
    #                 len(range(*idx.indices(self._column_widths[p])))
    #                 for p, idx in col_partitions_list.items()
    #             ]
    #             # Use the slice to calculate the new columns
    #             new_columns = self.columns[col_numeric_idx]
    #             assert sum(new_col_widths) == len(
    #                 new_columns
    #             ), "{} != {}.\n{}\n{}\n{}".format(
    #                 sum(new_col_widths),
    #                 len(new_columns),
    #                 col_numeric_idx,
    #                 self._column_widths,
    #                 col_partitions_list,
    #             )
    #             if self._dtypes is not None:
    #                 new_dtypes = self.dtypes[col_numeric_idx]
    #             else:
    #                 new_dtypes = None
    #         else:
    #             new_col_widths = [len(idx) for _, idx in col_partitions_list.items()]
    #             new_columns = self.columns[sorted(col_numeric_idx)]
    #             if self._dtypes is not None:
    #                 new_dtypes = self.dtypes.iloc[sorted(col_numeric_idx)]
    #             else:
    #                 new_dtypes = None
    #     else:
    #         col_partitions_list = {
    #             i: slice(None) for i in range(len(self._column_widths))
    #         }
    #         new_col_widths = self._column_widths
    #         new_columns = self.columns
    #         if self._dtypes is not None:
    #             new_dtypes = self.dtypes
    #         else:
    #             new_dtypes = None

    #     get_key_gpu = lambda row_idx, col_idx, row_internal_indices, col_internal_indices : \
    #         [
    #             self._partitions[row_idx][col_idx].mask(
    #                 row_internal_indices, col_internal_indices
    #             ),
    #             self._partitions[row_idx][col_idx].get_gpu_manager()
    #         ]

    #     new_partitions = np.array(
    #         [
    #             [
    #                 get_key_gpu(row_idx, col_idx, row_internal_indices, col_internal_indices)
    #                 for col_idx, col_internal_indices in col_partitions_list.items()
    #                 if isinstance(col_internal_indices, slice)
    #                 or len(col_internal_indices) > 0
    #             ]
    #             for row_idx, row_internal_indices in row_partitions_list.items()
    #             if isinstance(row_internal_indices, slice)
    #             or len(row_internal_indices) > 0
    #         ]
    #     )


    #     num_rows,num_cols = new_partitions.shape[:-1]
    #     keys = list(new_partitions[:,:,0].flatten())
    #     keys = ray.get(keys)
    #     gpu_managers = list(new_partitions[:,:,1].flatten())
    #     new_partitions = np.array([
    #         [cuDFOnRayFramePartition(gpu_manager, key)]
    #         for gpu_manager, key in zip(gpu_managers, keys)
    #     ], dtype=object).reshape(num_rows, num_cols)

    #     intermediate = self.__constructor__(
    #         new_partitions,
    #         new_index,
    #         new_columns,
    #         new_row_lengths,
    #         new_col_widths,
    #         new_dtypes,
    #     )
    #     # Check if monotonically increasing, return if it is. Fast track code path for
    #     # common case to keep it fast.
    #     if (
    #         row_numeric_idx is None
    #         or isinstance(row_numeric_idx, slice)
    #         or len(row_numeric_idx) == 1
    #         or np.all(row_numeric_idx[1:] >= row_numeric_idx[:-1])
    #     ) and (
    #         col_numeric_idx is None
    #         or isinstance(col_numeric_idx, slice)
    #         or len(col_numeric_idx) == 1
    #         or np.all(col_numeric_idx[1:] >= col_numeric_idx[:-1])
    #     ):
    #         return intermediate
    #     # The new labels are often smaller than the old labels, so we can't reuse the
    #     # original order values because those were mapped to the original data. We have
    #     # to reorder here based on the expected order from within the data.
    #     # We create a dictionary mapping the position of the numeric index with respect
    #     # to all others, then recreate that order by mapping the new order values from
    #     # the old. This information is sent to `reorder_labels`.
    #     if row_numeric_idx is not None:
    #         row_order_mapping = dict(
    #             zip(sorted(row_numeric_idx), range(len(row_numeric_idx)))
    #         )
    #         new_row_order = [row_order_mapping[idx] for idx in row_numeric_idx]
    #     else:
    #         new_row_order = None
    #     if col_numeric_idx is not None:
    #         col_order_mapping = dict(
    #             zip(sorted(col_numeric_idx), range(len(col_numeric_idx)))
    #         )
    #         new_col_order = [col_order_mapping[idx] for idx in col_numeric_idx]
    #     else:
    #         new_col_order = None
    #     return intermediate.reorder_labels(
    #         row_numeric_idx=new_row_order, col_numeric_idx=new_col_order
    #     )

    # def _map(
    #     self,
    #     func,
    #     dtypes=None,
    #     validate_index=False,
    #     validate_columns=False,
    #     persistent=True
    # ):
    #     """Perform a function that maps across the entire dataset.

    #     Pamareters
    #     ----------
    #         func : callable
    #             The function to apply.
    #         dtypes :
    #             (optional) The data types for the result. This is an optimization
    #             because there are functions that always result in a particular data
    #             type, and allows us to avoid (re)computing it.
    #         validate_index : bool, (default False)
    #             Is index validation required after performing `func` on partitions.
    #     Returns
    #     -------
    #         A new dataframe.
    #     """
    #     new_partitions = self._frame_mgr_cls.map_partitions(
    #         self._partitions,
    #         func,
    #         persistent=persistent,
    #     )
    #     if dtypes is "copy":
    #         dtypes = self._dtypes
    #     elif dtypes is not None:
    #         dtypes = pandas.Series(
    #             [np.dtype(dtypes)] * len(self.columns), index=self.columns
    #         )
    #     if validate_index:
    #         new_index = self._frame_mgr_cls.get_indices(
    #             0, new_partitions, lambda df: df.index.to_pandas()
    #         )
    #     else:
    #         new_index = self.index
    #     if len(new_index) != len(self.index):
    #         new_row_lengths = None
    #     else:
    #         new_row_lengths = self._row_lengths

    #     if validate_columns:
    #         new_columns = self._frame_mgr_cls.get_indices(
    #             1, new_partitions, lambda df: df.columns
    #         )
    #     else:
    #         new_columns = self.columns
    #     if len(new_columns) != len(self.columns):
    #         new_column_widths = None
    #     else:
    #         new_column_widths = self._column_widths
    #     return self.__constructor__(
    #         new_partitions,
    #         new_index,
    #         new_columns,
    #         new_row_lengths,
    #         new_column_widths,
    #         dtypes=dtypes,
    #     )

    # @property
    # def _row_lengths(self):
    #     """Compute the row lengths if they are not cached.

    #     Returns:
    #         A list of row lengths.
    #     """
    #     if self._row_lengths_cache is None:
    #         if len(self._partitions.T) > 0:
    #             self._row_lengths_cache = [obj.length() for obj in self._partitions.T[0]]
    #             self._row_lengths_cache = ray.get(self._row_lengths_cache)
    #         else:
    #             self._row_lengths_cache = []
    #     return self._row_lengths_cache

    # @property
    # def _column_widths(self):
    #     """Compute the column widths if they are not cached.

    #     Returns:
    #         A list of column widths.
    #     """
    #     if self._column_widths_cache is None:
    #         if len(self._partitions) > 0:
    #             self._column_widths_cache = [obj.width() for obj in self._partitions[0]]
    #             self._column_widths_cache = ray.get(self._column_widths_cache)
    #         else:
    #             self._column_widths_cache = []
    #     return self._column_widths_cache

    # @property
    # def dtypes(self):
    #     """Compute the data types if they are not cached.

    #     Returns:
    #         A pandas Series containing the data types for this dataframe.
    #     """
    #     if self._dtypes is None:
    #         self._dtypes = self._compute_dtypes()
    #     return self._dtypes

    # def _compute_dtypes(self):
    #     """Compute the dtypes via MapReduce.

    #     Returns:
    #         The data types of this dataframe.
    #     """
    #     # TODO: does not provide full axis information without the reduce step
    #     first_row = self.mask(row_numeric_idx=[0]).to_pandas()
    #     # For now we will use a pandas Series for the dtypes.
    #     if len(self.columns) > 0:
    #         dtypes = first_row.dtypes
    #     else:
    #         dtypes = pandas.Series([])
    #     # reset name to None because we use "__reduced__" internally
    #     dtypes.name = None
    #     return dtypes

    # _index_cache = None
    # _columns_cache = None

    # def _validate_set_axis(self, new_labels, old_labels):
    #     """Validates the index or columns replacement against the old labels.

    #     Args:
    #         new_labels: The labels to replace with.
    #         old_labels: The labels to replace.

    #     Returns:
    #         The validated labels.
    #     """
    #     new_labels = ensure_index(new_labels)
    #     old_len = len(old_labels)
    #     new_len = len(new_labels)
    #     if old_len != new_len:
    #         raise ValueError(
    #             "Length mismatch: Expected axis has %d elements, "
    #             "new values have %d elements" % (old_len, new_len)
    #         )
    #     return new_labels

    # def _get_index(self):
    #     """Gets the index from the cache object.

    #     Returns:
    #         A pandas.Index object containing the row labels.
    #     """
    #     return self._index_cache

    # def _get_columns(self):
    #     """Gets the columns from the cache object.

    #     Returns:
    #         A pandas.Index object containing the column labels.
    #     """
    #     return self._columns_cache

    # def _set_index(self, new_index):
    #     """Replaces the current row labels with new labels.

    #     Args:
    #         new_index: The replacement row labels.
    #     """
    #     if self._index_cache is None:
    #         self._index_cache = ensure_index(new_index)
    #     else:
    #         new_index = self._validate_set_axis(new_index, self._index_cache)
    #         self._index_cache = new_index
    #     self._apply_index_objs(axis=0)

    # def _set_columns(self, new_columns):
    #     """Replaces the current column labels with new labels.

    #     Args:
    #         new_columns: The replacement column labels.
    #     """
    #     if self._columns_cache is None:
    #         self._columns_cache = ensure_index(new_columns)
    #     else:
    #         new_columns = self._validate_set_axis(new_columns, self._columns_cache)
    #         self._columns_cache = new_columns
    #         if self._dtypes is not None:
    #             self._dtypes.index = new_columns
    #     self._apply_index_objs(axis=1)

    # def _set_axis(self, axis, new_axis, cache_only=False):
    #     """Replaces the current labels at the specified axis with the new one

    #     Parameters
    #     ----------
    #         axis : int,
    #             Axis to set labels along
    #         new_axis : Index,
    #             The replacement labels
    #         cache_only : bool,
    #             Whether to change only external indices, or propagate it
    #             into partitions
    #     """
    #     if axis:
    #         if not cache_only:
    #             self._set_columns(new_axis)
    #         else:
    #             self._columns_cache = ensure_index(new_axis)
    #     else:
    #         if not cache_only:
    #             self._set_index(new_axis)
    #         else:
    #             self._index_cache = ensure_index(new_axis)

    # columns = property(_get_columns, _set_columns)
    # index = property(_get_index, _set_index)

    # @property
    # def axes(self):
    #     """The index, columns that can be accessed with an `axis` integer."""
    #     return [self.index, self.columns]

    # def _filter_empties(self):
    #     """Removes empty partitions to avoid triggering excess computation."""
    #     if len(self.axes[0]) == 0 or len(self.axes[1]) == 0:
    #         # This is the case for an empty frame. We don't want to completely remove
    #         # all metadata and partitions so for the moment, we won't prune if the frame
    #         # is empty.
    #         # TODO: Handle empty dataframes better
    #         return
    #     self._partitions = np.array(
    #         [
    #             [
    #                 self._partitions[i][j]
    #                 for j in range(len(self._partitions[i]))
    #                 if j < len(self._column_widths) and self._column_widths[j] > 0
    #             ]
    #             for i in range(len(self._partitions))
    #             if i < len(self._row_lengths) and self._row_lengths[i] > 0
    #         ]
    #     )
    #     self._column_widths_cache = [w for w in self._column_widths if w > 0]
    #     self._row_lengths_cache = [r for r in self._row_lengths if r > 0]

    # def _validate_axis_equality(self, axis: int, force: bool = False):
    #     """
    #     Validates internal and external indices of modin_frame at the specified axis.

    #     Parameters
    #     ----------
    #         axis : int,
    #             Axis to validate indices along
    #         force : bool,
    #             Whether to update external indices with internal if their lengths
    #             do not match or raise an exception in that case.
    #     """
    #     internal_axis = self._frame_mgr_cls.get_indices(
    #         axis, self._partitions, lambda df: [df.index, df.columns][axis]
    #     )
    #     is_equals = self.axes[axis].equals(internal_axis)
    #     is_lenghts_matches = len(self.axes[axis]) == len(internal_axis)
    #     if not is_equals:
    #         if force:
    #             new_axis = self.axes[axis] if is_lenghts_matches else internal_axis
    #             self._set_axis(axis, new_axis, cache_only=not is_lenghts_matches)
    #         else:
    #             self._set_axis(
    #                 axis, self.axes[axis],
    #             )

    # def _validate_internal_indices(self, mode=None, **kwargs):
    #     """
    #     Validates and optionally updates internal and external indices
    #     of modin_frame in specified mode. There is 3 modes supported:
    #         1. "reduced" - force validates on that axes
    #             where external indices is ["__reduced__"]
    #         2. "all" - validates indices at all axes, optionally force
    #             if `force` parameter specified in kwargs
    #         3. "custom" - validation follows arguments specified in kwargs.

    #     Parameters
    #     ----------
    #         mode : str or bool, default None
    #         validate_index : bool, (optional, could be specified via `mode`)
    #         validate_columns : bool, (optional, could be specified via `mode`)
    #         force : bool (optional, could be specified via `mode`)
    #             Whether to update external indices with internal if their lengths
    #             do not match or raise an exception in that case.
    #     """

    #     if isinstance(mode, bool):
    #         is_force = mode
    #         mode = "all"
    #     else:
    #         is_force = kwargs.get("force", False)

    #     reduced_sample = cudf.Index(["__reduced__"])
    #     args_dict = {
    #         "custom": kwargs,
    #         "reduced": {
    #             "validate_index": self.index.equals(reduced_sample),
    #             "validate_columns": self.columns.equals(reduced_sample),
    #             "force": True,
    #         },
    #         "all": {
    #             "validate_index": True,
    #             "validate_columns": True,
    #             "force": is_force,
    #         },
    #     }

    #     args = args_dict.get(mode, args_dict["custom"])

    #     if args.get("validate_index", True):
    #         self._validate_axis_equality(axis=0)
    #     if args.get("validate_columns", True):
    #         self._validate_axis_equality(axis=1)

    # def _apply_index_objs(self, axis=None):
    #     """Applies the index object (Index or Columns) to the partitions.

    #     Args:
    #         axis: The axis to apply to, None applies to both axes.

    #     Returns:
    #         A new 2D array of partitions that have the index assignment added to the
    #         call queue.
    #     """
    #     def set_axis(df, labels, axis="index", inplace=False):
    #         idx = cudf.Index(labels)
    #         if not inplace:
    #             df = df.copy()
    #         if axis == 'index':
    #             df.index = idx
    #         else:
    #             df.columns = idx
    #         return df

    #     self._filter_empties()
    #     if axis is None or axis == 0:
    #         cum_row_lengths = np.cumsum([0] + self._row_lengths)
    #     if axis is None or axis == 1:
    #         cum_col_widths = np.cumsum([0] + self._column_widths)

    #     if axis is None:

    #         def apply_idx_objs(df, idx, cols):
    #             # FIXME not garbage collected in gpu memory
    #             df = set_axis(df, idx, axis='index', inplace=False)
    #             df = set_axis(df, cols, axis='columns', inplace=False)
    #             return df

    #         keys = np.array(
    #             [
    #                 [
    #                     [
    #                         self._partitions[i][j].apply(
    #                             apply_idx_objs,
    #                             idx=self.index[
    #                                 slice(cum_row_lengths[i], cum_row_lengths[i + 1])
    #                             ],
    #                             cols=self.columns[
    #                                 slice(cum_col_widths[j], cum_col_widths[j + 1])
    #                             ],
    #                         ),
    #                         self._partitions[i][j].get_gpu_manager(),
    #                     ]
    #                     for j in range(len(self._partitions[i]))
    #                 ]
    #                 for i in range(len(self._partitions))
    #             ]
    #         )
    #     elif axis == 0:

    #         def apply_idx_objs(df, idx):
    #             return set_axis(df, idx, axis='index', inplace=False)

    #         keys_and_gpus = np.array(
    #             [
    #                 [
    #                     [
    #                         self._partitions[i][j].apply(
    #                             apply_idx_objs,
    #                             idx=self.index[
    #                                 slice(cum_row_lengths[i], cum_row_lengths[i + 1])
    #                             ],
    #                         ),
    #                         self._partitions[i][j].get_gpu_manager(),
    #                     ]
    #                     for j in range(len(self._partitions[i]))
    #                 ]
    #                 for i in range(len(self._partitions))
    #             ]
    #         )

    #     elif axis == 1:

    #         def apply_idx_objs(df, cols):
    #             return set_axis(df, cols, axis='columns', inplace=False)

    #         keys_and_gpus = np.array(
    #             [
    #                 [
    #                     [
    #                         self._partitions[i][j].apply(
    #                             apply_idx_objs,
    #                             cols=self.columns[
    #                                 slice(cum_col_widths[j], cum_col_widths[j + 1])
    #                             ],
    #                         ),
    #                         self._partitions[i][j].get_gpu_manager(),
    #                     ]
    #                     for j in range(len(self._partitions[i]))
    #                 ]
    #                 for i in range(len(self._partitions))
    #             ]
    #         )
    #         ErrorMessage.catch_bugs_and_request_email(
    #             axis is not None and axis not in [0, 1]
    #         )
    #     keys = ray.get(list(keys_and_gpus[:,:,0].flatten()))
    #     gpus = list(keys_and_gpus[:,:,1].flatten())
    #     original_shape = self._partitions.shape
    #     self._partitions = np.array(
    #         [
    #             [cuDFOnRayFramePartition(gpu_manager, key)]
    #             for key, gpu_manager in zip(keys, gpus)
    #         ]
    #     ).reshape(original_shape)

    # def reorder_labels(self, row_numeric_idx=None, col_numeric_idx=None):
    #     """Reorder the column and or rows in this DataFrame.

    #     Parameters
    #     ----------
    #     row_numeric_idx : list of int, optional
    #         The ordered list of new row orders such that each position within the list
    #         indicates the new position.
    #     col_numeric_idx : list of int, optional
    #         The ordered list of new column orders such that each position within the
    #         list indicates the new position.

    #     Returns
    #     -------
    #     BasePandasFrame
    #         A new BasePandasFrame with reordered columns and/or rows.
    #     """
    #     if row_numeric_idx is not None:
    #         ordered_rows = self._frame_mgr_cls.map_axis_partitions(
    #             0, self._partitions, lambda df: df.iloc[row_numeric_idx]
    #         )
    #         ordered_rows = self._frame_mgr_cls.map_axis_partitions(
    #             0, self._partitions, f
    #         )
    #         row_idx = self.index[row_numeric_idx]
    #     else:
    #         ordered_rows = self._partitions
    #         row_idx = self.index
    #     if col_numeric_idx is not None:
    #         ordered_cols = self._frame_mgr_cls.map_axis_partitions(
    #             1, ordered_rows, lambda df: df.iloc[:, col_numeric_idx]
    #         )
    #         col_idx = self.columns[col_numeric_idx]
    #     else:
    #         ordered_cols = ordered_rows
    #         col_idx = self.columns
    #     return self.__constructor__(ordered_cols, row_idx, col_idx)

    # def copy(self):
    #     """Copy this object.

    #     Returns:
    #         A copied version of this object.
    #     """
    #     return self.__constructor__(
    #         self._frame_mgr_cls.copy(self._partitions),
    #         self.index.copy(),
    #         self.columns.copy(),
    #         self._row_lengths,
    #         self._column_widths,
    #         self._dtypes,
    #     )

    # @classmethod
    # def combine_dtypes(cls, list_of_dtypes, column_names):
    #     """Describes how data types should be combined when they do not match.

    #     Args:
    #         list_of_dtypes: A list of pandas Series with the data types.
    #         column_names: The names of the columns that the data types map to.

    #     Returns:
    #          A pandas Series containing the finalized data types.
    #     """
    #     # Compute dtypes by getting collecting and combining all of the partitions. The
    #     # reported dtypes from differing rows can be different based on the inference in
    #     # the limited data seen by each worker. We use pandas to compute the exact dtype
    #     # over the whole column for each column.
    #     dtypes = (
    #         pandas.concat(list_of_dtypes, axis=1)
    #         .apply(lambda row: find_common_type(row.values), axis=1)
    #         .squeeze(axis=0)
    #     )
    #     dtypes.index = column_names
    #     return dtypes

    # def astype(self, col_dtypes, **kwargs):
    #     """Converts columns dtypes to given dtypes.

    #     Args:
    #         col_dtypes: Dictionary of {col: dtype,...} where col is the column
    #             name and dtype is a numpy dtype.

    #     Returns:
    #         dataframe with updated dtypes.
    #     """
    #     columns = col_dtypes.keys()
    #     # Create Series for the updated dtypes
    #     new_dtypes = self.dtypes.copy()
    #     for i, column in enumerate(columns):
    #         dtype = col_dtypes[column]
    #         if (
    #             not isinstance(dtype, type(self.dtypes[column]))
    #             or dtype != self.dtypes[column]
    #         ):
    #             # Update the new dtype series to the proper pandas dtype
    #             try:
    #                 new_dtype = cp.dtype(dtype)
    #             except TypeError:
    #                 new_dtype = dtype

    #             if dtype != np.int32 and new_dtype == np.int32:
    #                 new_dtypes[column] = cp.dtype("int64")
    #             elif dtype != np.float32 and new_dtype == np.float32:
    #                 new_dtypes[column] = np.dtype("float64")
    #             elif isinstance(new_dtype, str) and new_dtype == "category":
    #                 new_dtypes[column] = cudf.core.dtypes.CategoricalDtype
    #             else:
    #                 new_dtypes[column] = new_dtype
    #     copy = kwargs.get('copy', True)
    #     new_frame = self._frame_mgr_cls.map_partitions(
    #         self._partitions,
    #         lambda df : df.astype({k: v for k, v in col_dtypes.items() if k in df}, copy=copy),
    #     )
    #     return self.__constructor__(
    #         new_frame,
    #         self.index,
    #         self.columns,
    #         self._row_lengths,
    #         self._column_widths,
    #         new_dtypes,
    #     )

    # # Metadata modification methods
    # def add_prefix(self, prefix, axis):
    #     """Add a prefix to the current row or column labels.

    #     Args:
    #         prefix: The prefix to add.
    #         axis: The axis to update.

    #     Returns:
    #         A new dataframe with the updated labels.
    #     """
    #     new_labels = self.axes[axis].map(lambda x: str(prefix) + str(x))
    #     new_frame = self.copy()
    #     if axis == 0:
    #         new_frame.index = new_labels
    #     else:
    #         new_frame.columns = new_labels
    #     return new_frame

    # def add_suffix(self, suffix, axis):
    #     """Add a suffix to the current row or column labels.

    #     Args:
    #         suffix: The suffix to add.
    #         axis: The axis to update.

    #     Returns:
    #         A new dataframe with the updated labels.
    #     """
    #     new_labels = self.axes[axis].map(lambda x: str(x) + str(suffix))
    #     new_frame = self.copy()
    #     if axis == 0:
    #         new_frame.index = new_labels
    #     else:
    #         new_frame.columns = new_labels
    #     return new_frame

    # # END Metadata modification methods

    # def _numeric_columns(self, include_bool=True):
    #     """Returns the numeric columns of the Manager.

    #     Returns:
    #         List of index names.
    #     """
    #     columns = []
    #     for col, dtype in zip(self.columns, self.dtypes):
    #         if is_numeric_dtype(dtype) and (
    #             include_bool or (not include_bool and dtype != np.bool_)
    #         ):
    #             columns.append(col)
    #     return columns

    # def _get_dict_of_block_index(self, axis, indices):
    #     """Convert indices to a dict of block index to internal index mapping.

    #     Parameters
    #     ----------
    #     axis : (0 - rows, 1 - columns)
    #            The axis along which to get the indices
    #     indices : list of int, slice
    #             A list of global indices to convert.

    #     Returns
    #     -------
    #     dictionary mapping int to list of int
    #         A mapping from partition to list of internal indices to extract from that
    #         partition.
    #     """
    #     # Fasttrack slices
    #     if isinstance(indices, slice):
    #         if indices == slice(None) or indices == slice(0, None):
    #             return OrderedDict(
    #                 zip(
    #                     range(len(self.axes[axis])),
    #                     [slice(None)] * len(self.axes[axis]),
    #                 )
    #             )
    #         if indices.start is None or indices.start == 0:
    #             last_part, last_idx = list(
    #                 self._get_dict_of_block_index(axis, [indices.stop]).items()
    #             )[0]
    #             dict_of_slices = OrderedDict(
    #                 zip(range(last_part), [slice(None)] * last_part)
    #             )
    #             dict_of_slices.update({last_part: slice(last_idx[0])})
    #             return dict_of_slices
    #         elif indices.stop is None or indices.stop >= len(self.axes[axis]):
    #             first_part, first_idx = list(
    #                 self._get_dict_of_block_index(axis, [indices.start]).items()
    #             )[0]
    #             dict_of_slices = OrderedDict({first_part: slice(first_idx[0], None)})
    #             num_partitions = np.size(self._partitions, axis=axis)
    #             part_list = range(first_part + 1, num_partitions)
    #             dict_of_slices.update(
    #                 OrderedDict(zip(part_list, [slice(None)] * len(part_list)))
    #             )
    #             return dict_of_slices
    #         else:
    #             first_part, first_idx = list(
    #                 self._get_dict_of_block_index(axis, [indices.start]).items()
    #             )[0]
    #             last_part, last_idx = list(
    #                 self._get_dict_of_block_index(axis, [indices.stop]).items()
    #             )[0]
    #             if first_part == last_part:
    #                 return OrderedDict({first_part: slice(first_idx[0], last_idx[0])})
    #             else:
    #                 if last_part - first_part == 1:
    #                     return OrderedDict(
    #                         {
    #                             first_part: slice(first_idx[0], None),
    #                             last_part: slice(None, last_idx[0]),
    #                         }
    #                     )
    #                 else:
    #                     dict_of_slices = OrderedDict(
    #                         {first_part: slice(first_idx[0], None)}
    #                     )
    #                     part_list = range(first_part + 1, last_part)
    #                     dict_of_slices.update(
    #                         OrderedDict(zip(part_list, [slice(None)] * len(part_list)))
    #                     )
    #                     dict_of_slices.update({last_part: slice(None, last_idx[0])})
    #                     return dict_of_slices
    #     # Sort and convert negative indices to positive
    #     indices = np.sort(
    #         [i if i >= 0 else max(0, len(self.axes[axis]) + i) for i in indices]
    #     )
    #     if axis == 0:
    #         bins = np.array(self._row_lengths)
    #     else:
    #         bins = np.array(self._column_widths)
    #     # INT_MAX to make sure we don't try to compute on partitions that don't exist.
    #     cumulative = np.append(bins[:-1].cumsum(), np.iinfo(bins.dtype).max)

    #     def internal(block_idx, global_index):
    #         return (
    #             global_index
    #             if not block_idx
    #             else np.subtract(
    #                 global_index, cumulative[min(block_idx, len(cumulative) - 1) - 1]
    #             )
    #         )

    #     partition_ids = np.digitize(indices, cumulative)
    #     count_for_each_partition = np.array(
    #         [(partition_ids == i).sum() for i in range(len(cumulative))]
    #     ).cumsum()
    #     # Compute the internal indices and pair those with the partition index.
    #     # If the first partition has any values we need to return, compute those
    #     # first to make the list comprehension easier. Otherwise, just append the
    #     # rest of the values to an empty list.
    #     if count_for_each_partition[0] > 0:
    #         first_partition_indices = [
    #             (0, internal(0, indices[slice(count_for_each_partition[0])]))
    #         ]
    #     else:
    #         first_partition_indices = []
    #     partition_ids_with_indices = first_partition_indices + [
    #         (
    #             i,
    #             internal(
    #                 i,
    #                 indices[
    #                     slice(
    #                         count_for_each_partition[i - 1],
    #                         count_for_each_partition[i],
    #                     )
    #                 ],
    #             ),
    #         )
    #         for i in range(1, len(count_for_each_partition))
    #         if count_for_each_partition[i] > count_for_each_partition[i - 1]
    #     ]
    #     return OrderedDict(partition_ids_with_indices)

    # def _join_index_objects(self, axis, other_index, how, sort):
    #     """
    #     Joins a pair of index objects (columns or rows) by a given strategy.

    #     Parameters
    #     ----------
    #         axis : 0 or 1
    #             The axis index object to join (0 - rows, 1 - columns).
    #         other_index : Index
    #             The other_index to join on.
    #         how : {'left', 'right', 'inner', 'outer'}
    #             The type of join to join to make.
    #         sort : boolean
    #             Whether or not to sort the joined index

    #     Returns
    #     -------
    #     Index
    #         Joined indices.
    #     """
    #     if isinstance(other_index, list):
    #         joined_obj = self.columns if axis else self.index
    #         # TODO (kvu35): revisit for performance
    #         for obj in other_index:
    #             joined_obj = joined_obj.join(obj, how=how, sort=sort)
    #         return joined_obj
    #     if axis:
    #         return self.columns.join(other_index, how=how, sort=sort)
    #     else:
    #         return self.index.join(other_index, how=how, sort=sort)

    # # Internal methods
    # # These methods are for building the correct answer in a modular way.
    # # Please be careful when changing these!
    # def _build_mapreduce_func(self, axis, func):
    #     def _map_reduce_func(df):
    #         series_result = func(df)
    #         # If the result is actually a series, we have to use to_frame api in order to preserve
    #         # the index objects.
    #         if isinstance(series_result, cudf.Series):
    #             if axis == 0:
    #                 # In the case of axis=0, we need to keep the shape of the data
    #                 # consistent with what we have done. In the case of a reduction, the
    #                 # data for axis=0 should be a single value for each column. By
    #                 # transposing the data after we convert to a DataFrame, we ensure that
    #                 # the columns of the result line up with the columns from the data.
    #                 # axis=1 does not have this requirement because the index already will
    #                 # line up with the index of the data based on how pandas creates a
    #                 # DataFrame from a Series.
    #                 new_df = series_result.to_frame().transpose()
    #                 # CuDF does not support categorical index objects as the columns therefore we
    #                 # have to save it as a hidden attribute
    #                 if isinstance(series_result.index, cudf.core.index.CategoricalIndex):
    #                     new_df.__columns__ = series_result.index
    #                 return new_df
    #             return series_result.to_frame()
    #         return cudf.DataFrame(series_result)

    #     return _map_reduce_func

    # def _compute_map_reduce_metadata(self, axis, new_parts):
    #     if axis == 0:
    #         columns = self.columns
    #         index = ["__reduced__"]
    #         new_lengths = [1]
    #         new_widths = self._column_widths
    #         new_dtypes = self._dtypes
    #     else:
    #         columns = ["__reduced__"]
    #         index = self.index
    #         new_lengths = self._row_lengths
    #         new_widths = [1]
    #         if self._dtypes is not None:
    #             new_dtypes = pandas.Series(
    #                 np.full(1, find_common_type(self.dtypes.values)),
    #                 index=["__reduced__"],
    #             )
    #         else:
    #             new_dtypes = self._dtypes
    #     return self.__constructor__(
    #         new_parts,
    #         index,
    #         columns,
    #         new_lengths,
    #         new_widths,
    #         new_dtypes,
    #         validate_axes="reduced",
    #     )

    # def _fold_reduce(self, axis, func, persistent=True):
    #     """Applies map that reduce Manager to series but require knowledge of full axis.

    #     Args:
    #         func: Function to reduce the Manager by. This function takes in a Manager.
    #         axis: axis to apply the function to.

    #     Return:
    #         Pandas series containing the reduced data.
    #     """
    #     func = self._build_mapreduce_func(axis, func)
    #     new_parts = self._frame_mgr_cls.map_axis_partitions(
    #         axis, self._partitions, func, persistent=persistent
    #     )
    #     return self._compute_map_reduce_metadata(axis, new_parts)

    # def _map_reduce(self, axis, map_func, reduce_func=None, preserve_index=True):
    #     """
    #     Apply function that will reduce the data to a Pandas Series.

    #     Parameters
    #     ----------
    #         axis : 0 or 1
    #             0 for columns and 1 for rows.
    #         map_func : callable
    #             Callable function to map the dataframe.
    #         reduce_func : callable
    #             Callable function to reduce the dataframe.
    #             If none, then apply map_func twice. Default is None.
    #         preserve_index : boolean
    #             The flag to preserve index for default behavior
    #             map and reduce operations. Default is True.

    #     Returns
    #     -------
    #     BasePandasFrame
    #         A new dataframe.
    #     """
    #     map_func = self._build_mapreduce_func(axis, map_func)
    #     if reduce_func is None:
    #         reduce_func = map_func
    #     else:
    #         reduce_func = self._build_mapreduce_func(axis, reduce_func)

    #     map_parts = self._frame_mgr_cls.map_partitions(self._partitions, map_func)
    #     reduce_parts = self._frame_mgr_cls.map_axis_partitions(
    #         axis, map_parts, reduce_func
    #     )
    #     if preserve_index:
    #         return self._compute_map_reduce_metadata(axis, reduce_parts)
    #     else:
    #         if axis == 0:
    #             new_index = ["__reduced__"]
    #             # We have to add this here because CuDF only supports Index objects as columns.
    #             new_columns = self._frame_mgr_cls.get_indices(
    #                 1,
    #                 reduce_parts,
    #                 lambda df: \
    #                     df.__columns__.to_pandas()
    #                     if hasattr(df, "__columns__")
    #                     else df.columns
    #             )
    #         else:
    #             new_index = self._frame_mgr_cls.get_indices(
    #                 0, reduce_parts, lambda df: df.index.to_pandas()
    #             )
    #             new_columns = ["__reduced__"]
    #         return self.__constructor__(
    #             reduce_parts, new_index, new_columns, validate_axes="reduced"
    #         )

    # def _fold(self, axis, func):
    #     """Perform a function across an entire axis.

    #     Note: The data shape is not changed (length and width of the table).

    #     Args:
    #         axis: The axis to apply over.
    #         func: The function to apply.

    #     Returns:
    #          A new dataframe.
    #     """
    #     new_partitions = self._frame_mgr_cls.map_axis_partitions(
    #         axis, self._partitions, func
    #     )
    #     return self.__constructor__(
    #         new_partitions,
    #         self.index,
    #         self.columns,
    #         self._row_lengths,
    #         self._column_widths,
    #     )

    # def filter_full_axis(self, axis, func):
    #     """Filter data based on the function provided along an entire axis.

    #     Args:
    #         axis: The axis to filter over.
    #         func: The function to use for the filter. This function should filter the
    #             data itself.

    #     Returns:
    #         A new dataframe.
    #     """
    #     new_partitions = self._frame_mgr_cls.map_axis_partitions(
    #         axis, self._partitions, func, keep_partitioning=True
    #     )
    #     if axis == 0:
    #         new_index = self.index
    #         new_lengths = self._row_lengths
    #         new_widths = None  # We do not know what the resulting widths will be
    #         new_columns = self._frame_mgr_cls.get_indices(
    #             1, new_partitions, lambda df: df.columns
    #         )
    #     else:
    #         new_columns = self.columns
    #         new_lengths = None  # We do not know what the resulting lengths will be
    #         new_widths = self._column_widths
    #         new_index = self._frame_mgr_cls.get_indices(
    #             0, new_partitions, lambda df: df.index.to_pandas()
    #         )
    #     return self.__constructor__(
    #         new_partitions,
    #         new_index,
    #         new_columns,
    #         new_lengths,
    #         new_widths,
    #         self.dtypes if axis == 0 else None,
    #     )

    # def _apply_full_axis(
    #     self, axis, func, new_index=None, new_columns=None, dtypes=None,
    # ):
    #     new_partitions = self._frame_mgr_cls.map_axis_partitions(
    #         axis,
    #         self._partitions,
    #         self._build_mapreduce_func(axis, func),
    #         keep_partitioning=True,
    #     )
    #     # Index objects for new object creation. This is shorter than if..else
    #     if new_columns is None:
    #         new_columns = self._frame_mgr_cls.get_indices(
    #             1, new_partitions, lambda df: df.columns
    #         )
    #     if new_index is None:
    #         new_index = self._frame_mgr_cls.get_indices(
    #             0, new_partitions, lambda df: df.index.to_pandas()
    #         )
    #     if dtypes == "copy":
    #         dtypes = self._dtypes
    #     elif dtypes is not None:
    #         dtypes = pandas.Series(
    #             [np.dtype(dtypes)] * len(new_columns), index=new_columns
    #         )
    #     return self.__constructor__(
    #         new_partitions,
    #         new_index,
    #         new_columns,
    #         None,
    #         None,
    #         dtypes,
    #         validate_axes="reduced",
    #     )

    # def _apply_full_axis_select_indices(
    #     self,
    #     axis,
    #     func,
    #     apply_indices=None,
    #     numeric_indices=None,
    #     new_index=None,
    #     new_columns=None,
    #     keep_remaining=False,
    # ):
    #     """Apply a function across an entire axis for a subset of the data.

    #     Args:
    #         axis: The axis to apply over.
    #         func: The function to apply
    #         apply_indices: The labels to apply over.
    #         numeric_indices: The indices to apply over.
    #         new_index: (optional) The index of the result. We may know this in advance,
    #             and if not provided it must be computed.
    #         new_columns: (optional) The columns of the result. We may know this in
    #             advance, and if not provided it must be computed.
    #         keep_remaining: Whether or not to drop the data that is not computed over.

    #     Returns:
    #         A new dataframe.
    #     """
    #     assert apply_indices is not None or numeric_indices is not None
    #     # Convert indices to numeric indices
    #     old_index = self.index if axis else self.columns
    #     if apply_indices is not None:
    #         numeric_indices = old_index.get_indexer_for(apply_indices)
    #     # Get the indices for the axis being applied to (it is the opposite of axis
    #     # being applied over)
    #     dict_indices = self._get_dict_of_block_index(axis ^ 1, numeric_indices)
    #     new_partitions = self._frame_mgr_cls.apply_func_to_select_indices_along_full_axis(
    #         axis, self._partitions, func, dict_indices, keep_remaining=keep_remaining
    #     )
    #     # TODO Infer columns and index from `keep_remaining` and `apply_indices`
    #     if new_index is None:
    #         new_index = self.index if axis == 1 else None
    #     if new_columns is None:
    #         new_columns = self.columns if axis == 0 else None
    #     return self.__constructor__(new_partitions, new_index, new_columns, None, None)

    # def _apply_select_indices(
    #     self,
    #     axis,
    #     func,
    #     apply_indices=None,
    #     row_indices=None,
    #     col_indices=None,
    #     new_index=None,
    #     new_columns=None,
    #     keep_remaining=False,
    #     item_to_distribute=None,
    # ):
    #     """Apply a function for a subset of the data.

    #     Args:
    #         axis: The axis to apply over.
    #         func: The function to apply
    #         apply_indices: (optional) The labels to apply over. Must be given if axis is
    #             provided.
    #         row_indices: (optional) The row indices to apply over. Must be provided with
    #             `col_indices` to apply over both axes.
    #         col_indices: (optional) The column indices to apply over. Must be provided
    #             with `row_indices` to apply over both axes.
    #         new_index: (optional) The index of the result. We may know this in advance,
    #             and if not provided it must be computed.
    #         new_columns: (optional) The columns of the result. We may know this in
    #             advance, and if not provided it must be computed.
    #         keep_remaining: Whether or not to drop the data that is not computed over.
    #         item_to_distribute: (optional) The item to split up so it can be applied
    #             over both axes.

    #     Returns:
    #         A new dataframe.
    #     """
    #     # TODO Infer columns and index from `keep_remaining` and `apply_indices`
    #     if new_index is None:
    #         new_index = self.index if axis == 1 else None
    #     if new_columns is None:
    #         new_columns = self.columns if axis == 0 else None
    #     if axis is not None:
    #         assert apply_indices is not None
    #         # Convert indices to numeric indices
    #         old_index = self.index if axis else self.columns
    #         numeric_indices = old_index.get_indexer_for(apply_indices)
    #         # Get indices being applied to (opposite of indices being applied over)
    #         dict_indices = self._get_dict_of_block_index(axis ^ 1, numeric_indices)
    #         new_partitions = self._frame_mgr_cls.apply_func_to_select_indices(
    #             axis,
    #             self._partitions,
    #             func,
    #             dict_indices,
    #             keep_remaining=keep_remaining,
    #         )
    #         # Length objects for new object creation. This is shorter than if..else
    #         # This object determines the lengths and widths based on the given
    #         # parameters and builds a dictionary used in the constructor below. 0 gives
    #         # the row lengths and 1 gives the column widths. Since the dimension of
    #         # `axis` given may have changed, we current just recompute it.
    #         # TODO Determine lengths from current lengths if `keep_remaining=False`
    #         lengths_objs = {
    #             axis: [len(apply_indices)]
    #             if not keep_remaining
    #             else [self._row_lengths, self._column_widths][axis],
    #             axis ^ 1: [self._row_lengths, self._column_widths][axis ^ 1],
    #         }
    #         return self.__constructor__(
    #             new_partitions, new_index, new_columns, lengths_objs[0], lengths_objs[1]
    #         )
    #     else:
    #         # We are apply over both axes here, so make sure we have all the right
    #         # variables set.
    #         assert row_indices is not None and col_indices is not None
    #         assert keep_remaining
    #         assert item_to_distribute is not None
    #         row_partitions_list = self._get_dict_of_block_index(0, row_indices).items()
    #         col_partitions_list = self._get_dict_of_block_index(1, col_indices).items()
    #         new_partitions = self._frame_mgr_cls.apply_func_to_indices_both_axis(
    #             self._partitions,
    #             func,
    #             row_partitions_list,
    #             col_partitions_list,
    #             item_to_distribute,
    #         )
    #         return self.__constructor__(
    #             new_partitions,
    #             new_index,
    #             new_columns,
    #             self._row_lengths_cache,
    #             self._column_widths_cache,
    #         )

    # def broadcast_apply(self, axis, func, other, preserve_labels=True, dtypes=None):
    #     # Only sort the indices if they do not match
    #     left_parts, right_parts, joined_index = self._copartition(
    #         axis, other, "left", sort=not self.axes[axis].equals(other.axes[axis])
    #     )
    #     # unwrap list returned by `copartition`.
    #     right_parts = right_parts[0]
    #     new_frame = self._frame_mgr_cls.broadcast_apply(
    #         axis, func, left_parts, right_parts
    #     )
    #     if dtypes == "copy":
    #         dtypes = self._dtypes
    #     new_index = self.index
    #     new_columns = self.columns
    #     if not preserve_labels:
    #         if axis == 1:
    #             new_columns = joined_index
    #         else:
    #             new_index = joined_index
    #     return self.__constructor__(
    #         new_frame, new_index, new_columns, None, None, dtypes=dtypes
    #     )

    # def _prepare_frame_to_broadcast(self, axis, indices, broadcast_all):
    #     """
    #     Computes indices to broadcast `self` with considering of `indices`

    #     Parameters
    #     ----------
    #         axis : int,
    #             axis to broadcast along
    #         indices : dict,
    #             Dict of indices and internal indices of partitions where `self` must
    #             be broadcasted
    #         broadcast_all : bool,
    #             Whether broadcast the whole axis of `self` frame or just a subset of it

    #     Returns
    #     -------
    #         Dictianary with indices of partitions to broadcast
    #     """
    #     if broadcast_all:

    #         def get_len(part):
    #             return part.width() if not axis else part.length()

    #         parts = self._partitions if not axis else self._partitions.T
    #         return {
    #             key: {
    #                 i: np.arange(get_len(parts[0][i])) for i in np.arange(len(parts[0]))
    #             }
    #             for key in indices.keys()
    #         }
    #     passed_len = 0
    #     result_dict = {}
    #     for part_num, internal in indices.items():
    #         result_dict[part_num] = self._get_dict_of_block_index(
    #             axis ^ 1, np.arange(passed_len, passed_len + len(internal))
    #         )
    #         passed_len += len(internal)
    #     return result_dict

    # def broadcast_apply_select_indices(
    #     self,
    #     axis,
    #     func,
    #     other,
    #     apply_indices=None,
    #     numeric_indices=None,
    #     keep_remaining=False,
    #     broadcast_all=True,
    #     new_index=None,
    #     new_columns=None,
    # ):
    #     """
    #     Applyies `func` to select indices at specified axis and broadcasts
    #     partitions of `other` frame.

    #     Parameters
    #     ----------
    #         axis : int,
    #             Axis to apply function along
    #         func : callable,
    #             Function to apply
    #         other : BasePandasFrame,
    #             Partitions of which should be broadcasted
    #         apply_indices : list,
    #             List of labels to apply (if `numeric_indices` are not specified)
    #         numeric_indices : list,
    #             Numeric indices to apply (if `apply_indices` are not specified)
    #         keep_remaining : Whether or not to drop the data that is not computed over.
    #         broadcast_all : Whether broadcast the whole axis of right frame to every
    #             partition or just a subset of it.
    #         new_index : Index, (optional)
    #             The index of the result. We may know this in advance,
    #             and if not provided it must be computed
    #         new_columns : Index, (optional)
    #             The columns of the result. We may know this in advance,
    #             and if not provided it must be computed.

    #     Returns
    #     -------
    #         BasePandasFrame
    #     """
    #     assert (
    #         apply_indices is not None or numeric_indices is not None
    #     ), "Indices to apply must be specified!"

    #     if other is None:
    #         if apply_indices is None:
    #             apply_indices = self.axes[axis][numeric_indices]
    #         return self._apply_select_indices(
    #             axis=axis,
    #             func=func,
    #             apply_indices=apply_indices,
    #             keep_remaining=keep_remaining,
    #             new_index=new_index,
    #             new_columns=new_columns,
    #         )

    #     if numeric_indices is None:
    #         old_index = self.index if axis else self.columns
    #         numeric_indices = old_index.get_indexer_for(apply_indices)

    #     dict_indices = self._get_dict_of_block_index(axis ^ 1, numeric_indices)
    #     broadcasted_dict = other._prepare_frame_to_broadcast(
    #         axis, dict_indices, broadcast_all=broadcast_all
    #     )
    #     new_partitions = self._frame_mgr_cls.broadcast_apply_select_indices(
    #         axis,
    #         func,
    #         self._partitions,
    #         other._partitions,
    #         dict_indices,
    #         broadcasted_dict,
    #         keep_remaining,
    #     )
    #     if new_index is None:
    #         new_index = self._frame_mgr_cls.get_indices(
    #             0, new_partitions, lambda df: df.index.to_pandas()
    #         )
    #     if new_columns is None:
    #         new_columns = self._frame_mgr_cls.get_indices(
    #             1, new_partitions, lambda df: df.columns
    #         )
    #     return self.__constructor__(new_partitions, new_index, new_columns)

    # def _copartition(self, axis, other, how, sort, force_repartition=False):
    #     """
    #     Copartition two dataframes.

    #     Parameters
    #     ----------
    #         axis : 0 or 1
    #             The axis to copartition along (0 - rows, 1 - columns).
    #         other : BasePandasFrame
    #             The other dataframes(s) to copartition against.
    #         how : str
    #             How to manage joining the index object ("left", "right", etc.)
    #         sort : boolean
    #             Whether or not to sort the joined index.
    #         force_repartition : boolean
    #             Whether or not to force the repartitioning. By default,
    #             this method will skip repartitioning if it is possible. This is because
    #             reindexing is extremely inefficient. Because this method is used to
    #             `join` or `append`, it is vital that the internal indices match.

    #     Returns
    #     -------
    #     Tuple
    #         A tuple (left data, right data list, joined index).
    #     """
    #     if isinstance(other, type(self)):
    #         other = [other]

    #     index_other_obj = [o.axes[axis] for o in other]
    #     joined_index = self._join_index_objects(axis, index_other_obj, how, sort)
    #     # We have to set these because otherwise when we perform the functions it may
    #     # end up serializing this entire object.
    #     left_old_idx = self.axes[axis]
    #     right_old_idxes = index_other_obj

    #     # Start with this and we'll repartition the first time, and then not again.
    #     if not left_old_idx.equals(joined_index) or force_repartition:
    #         reindexed_self = self._frame_mgr_cls.map_axis_partitions(
    #             axis, self._partitions, lambda df: df.reindex(joined_index, axis=axis)
    #         )
    #     else:
    #         reindexed_self = self._partitions
    #     reindexed_other_list = []

    #     for i in range(len(other)):
    #         if right_old_idxes[i].equals(joined_index) and not force_repartition:
    #             reindexed_other = other[i]._partitions
    #         else:
    #             reindexed_other = other[i]._frame_mgr_cls.map_axis_partitions(
    #                 axis,
    #                 other[i]._partitions,
    #                 lambda df: df.reindex(joined_index, axis=axis),
    #             )
    #         reindexed_other_list.append(reindexed_other)
    #     return reindexed_self, reindexed_other_list, joined_index

    # def _binary_op(self, op, right_frame, join_type="outer"):
    #     """
    #     Perform an operation that requires joining with another dataframe.

    #     Parameters
    #     ----------
    #         op : callable
    #             The function to apply after the join.
    #         right_frame : BasePandasFrame
    #             The dataframe to join with.
    #         join_type : str (optional)
    #             The type of join to apply.

    #     Returns
    #     -------
    #     BasePandasFrame
    #         A new dataframe.
    #     """
    #     if self.index.equals(right_frame.index):
    #         new_frame = self._frame_mgr_cls.binary_operation(1, self._partitions, lambda l, r: op(l, r), right_frame._partitions)
    #         return self.__constructor__(new_frame, self.index, self.columns, None, None)


    #     ## TODO(lepl3): Find an optimal method to join different shape
    #     left_parts, right_parts, joined_index = self._copartition(
    #         0, right_frame, join_type, sort=True
    #     )
    #     # unwrap list returned by `copartition`.
    #     right_parts = right_parts[0]
    #     new_frame = self._frame_mgr_cls.binary_operation(
    #         1, left_parts, lambda l, r: op(l, r), right_parts
    #     )
    #     new_columns = self.columns.join(right_frame.columns, how=join_type)
    #     return self.__constructor__(new_frame, self.index, new_columns, None, None)

    # def _join_op(self, op, right_frame, new_columns=None, new_index=None):
    #     new_partitions = self._frame_mgr_cls.join_operation(self._partitions, right_frame._partitions, lambda l, r: op(l, r))
    #     if new_index is None:
    #         new_index = self._frame_mgr_cls.get_indices(
    #             0, new_partitions, lambda df: df.index.to_pandas()
    #         )
    #     if new_columns is None:
    #         new_columns = self._frame_mgr_cls.get_indices(
    #             1, new_partitions, lambda df: df.columns
    #         )
    #     return self.__constructor__(new_partitions, new_index, new_columns)

    # def _concat(self, axis, others, how, sort):
    #     """Concatenate this dataframe with one or more others.

    #     Args:
    #         axis: The axis to concatenate over.
    #         others: The list of dataframes to concatenate with.
    #         how: The type of join to use for the axis.
    #         sort: Whether or not to sort the result.

    #     Returns:
    #         A new dataframe.
    #     """
    #     # Fast path for equivalent columns and partitioning
    #     if (
    #         axis == 0
    #         and all(o.columns.equals(self.columns) for o in others)
    #         and all(o._column_widths == self._column_widths for o in others)
    #     ):
    #         joined_index = self.columns
    #         left_parts = self._frame_mgr_cls.copy(self._partitions)
    #         right_parts = [self._frame_mgr_cls.copy(o._partitions) for o in others]
    #         new_lengths = self._row_lengths + [
    #             length for o in others for length in o._row_lengths
    #         ]
    #         new_widths = self._column_widths
    #     elif (
    #         axis == 1
    #         and all(o.index.equals(self.index) for o in others)
    #         and all(o._row_lengths == self._row_lengths for o in others)
    #     ):
    #         joined_index = self.index
    #         left_parts = self._frame_mgr_cls.copy(self._partitions)
    #         right_parts = [self._frame_mgr_cls.copy(o._partitions) for o in others]
    #         new_lengths = self._row_lengths
    #         new_widths = self._column_widths + [
    #             length for o in others for length in o._column_widths
    #         ]
    #     else:
    #         left_parts, right_parts, joined_index = self._copartition(
    #             axis ^ 1, others, how, sort, force_repartition=True
    #         )
    #         new_lengths = None
    #         new_widths = None
    #     new_partitions = self._frame_mgr_cls.concat(axis, left_parts, right_parts)
    #     if axis == 0:
    #         new_index = self.index.append([other.index for other in others])
    #         new_columns = joined_index
    #         # TODO: Can optimize by combining if all dtypes are materialized
    #         new_dtypes = None
    #     else:
    #         new_columns = self.columns.append([other.columns for other in others])
    #         new_index = joined_index
    #         if self._dtypes is not None and all(o._dtypes is not None for o in others):
    #             new_dtypes = self.dtypes.append([o.dtypes for o in others])
    #         else:
    #             new_dtypes = None
    #     return self.__constructor__(
    #         new_partitions, new_index, new_columns, new_lengths, new_widths, new_dtypes
    #     )

    # def to_numpy(self):
    #     """Converts Modin DataFrame to a 2D NumPy array.

    #     Returns:
    #         NumPy array.
    #     """
    #     return self._frame_mgr_cls.to_numpy(self._partitions)

    # def transpose(self):
    #     """Transpose the index and columns of this dataframe.

    #     Returns:
    #         A new dataframe.
    #     """
    #     new_partitions = self._frame_mgr_cls.map_partitions(
    #         self._partitions, lambda df: df.T
    #     ).T
    #     new_dtypes = pandas.Series(
    #         np.full(len(self.index), find_common_type(self.dtypes.values)),
    #         index=self.index,
    #     )
    #     return self.__constructor__(
    #         new_partitions,
    #         self.columns,
    #         self.index,
    #         self._column_widths,
    #         self._row_lengths,
    #         dtypes=new_dtypes,
    #     )

    # def get_gpu_keys(self):
    #     partitions = list(self._partitions.flatten())
    #     gpu_ids = [obj.get_gpu_manager().get_id.remote() for obj in partitions]
    #     gpu_ids = ray.get(gpu_ids)
    #     gpu_key_pair = zip(gpu_ids, [obj.key for obj in partitions])
    #     return list(gpu_key_pair)

    # def free(self):
    #     for obj in list(self._partitions.flatten()):
    #         obj.free()

    # # Implement the easiest method for boolean indexing which assumes that the boolean indexor
    # # is generated and kept in the same gpu as the dataframe that it will be indexing into.
    # def bool_indexor(self, mask, keepna=True):
    #     gpu_and_keys = np.array([
    #         [
    #             [
    #                 p.get_gpu_manager(),
    #                 p.get_gpu_manager().apply_with_two_keys.remote(
    #                     p.get_key(),
    #                     mask._partitions[row, 0].get_key(),
    #                     lambda df, bool_series : df[bool_series.iloc[:,0]],
    #                 )
    #             ]
    #             for p in self._partitions[row]
    #         ]
    #         for row in range(self._partitions.shape[0])
    #     ])
    #     keys = ray.get(list(gpu_and_keys[:,:,1].flatten()))
    #     gpu_and_keys[:,:,1] = np.array(keys).reshape(gpu_and_keys[:,:,0].shape)
    #     new_partitions = np.array([
    #         [
    #             self._frame_mgr_cls._partition_class(
    #                 gpu_and_keys[row, col, 0],
    #                 gpu_and_keys[row, col, 1],
    #             )
    #             for col in range(self._partitions.shape[1])
    #         ]
    #         for row in range(self._partitions.shape[0])
    #     ])
    #     row_lengths = ray.get([row[0].length() for row in new_partitions])
    #     new_index = self._frame_mgr_cls.get_indices(
    #         0, new_partitions, lambda df: df.index.to_pandas()
    #     )
    #     return self.__constructor__(
    #         new_partitions,
    #         new_index,
    #         self.columns,
    #         row_lengths,
    #         self._column_widths,
    #         dtypes=self._dtypes,
    #     )

    # def filter_full_axis(self, axis, func):
    #     """Filter data based on the function provided along an entire axis.

    #     Args:
    #         axis: The axis to filter over.
    #         func: The function to use for the filter. This function should filter the
    #             data itself.

    #     Returns:
    #         A new dataframe.
    #     """
    #     new_partitions = self._frame_mgr_cls.map_axis_partitions(
    #         axis, self._partitions, func, keep_partitioning=True
    #     )
    #     if axis == 0:
    #         new_index = self.index
    #         new_lengths = self._row_lengths
    #         new_widths = None  # We do not know what the resulting widths will be
    #         new_columns = self._frame_mgr_cls.get_indices(
    #             1, new_partitions, lambda df: df.columns
    #         )
    #     else:
    #         new_columns = self.columns
    #         new_lengths = None  # We do not know what the resulting lengths will be
    #         new_widths = self._column_widths
    #         new_index = self._frame_mgr_cls.get_indices(
    #             0, new_partitions, lambda df: df.index.to_pandas()
    #         )
    #     return self.__constructor__(
    #         new_partitions,
    #         new_index,
    #         new_columns,
    #         new_lengths,
    #         new_widths,
    #         self.dtypes if axis == 0 else None,
    #     )

    # # TODO (kvu35): add multiaxis support and utilize axis partition class
    # def trickle_down(self, func, upstream=None):
    #     nrows, ncols = self._partitions.shape
    #     new_partitions = self._frame_mgr_cls.trickle_down(func, self._partitions, upstream)
    #     new_partitions = np.array(new_partitions).reshape((nrows, 1))
    #     new_columns = [0]
    #     new_lengths = None
    #     new_widths = None
    #     new_index = self._frame_mgr_cls.get_indices(
    #         0, new_partitions, lambda df: df.index.to_pandas()
    #     )
    #     return self.__constructor__(
    #         new_partitions,
    #         new_index,
    #         new_columns,
    #         new_lengths,
    #         new_widths,
    #         pandas.Series([bool]),
    #     )
