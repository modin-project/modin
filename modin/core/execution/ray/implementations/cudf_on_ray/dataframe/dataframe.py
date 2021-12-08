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

"""Module houses class that implements ``PandasOnRayDataframe`` class using cuDF."""

import numpy as np
import ray

from ..partitioning.partition import cuDFOnRayDataframePartition
from ..partitioning.partition_manager import cuDFOnRayDataframePartitionManager

from modin.core.execution.ray.implementations.pandas_on_ray.dataframe.dataframe import (
    PandasOnRayDataframe,
)
from modin.error_message import ErrorMessage


class cuDFOnRayDataframe(PandasOnRayDataframe):
    """
    The class implements the interface in ``PandasOnRayDataframe`` using cuDF.

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

    _partition_mgr_cls = cuDFOnRayDataframePartitionManager

    def synchronize_labels(self, axis=None):
        """
        Synchronize labels by applying the index object (Index or Columns) to the partitions eagerly.

        Parameters
        ----------
        axis : {0, 1, None}, default: None
            The axis to apply to. If None, it applies to both axes.
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
                        axis=axis,
                    )
                    for j in range(len(self._partitions[i]))
                ]
                for i in range(len(self._partitions))
            ]
        )

        self._partitions = np.array(
            [
                [
                    cuDFOnRayDataframePartition(
                        self._partitions[i][j].get_gpu_manager(),
                        keys[i][j],
                        self._partitions[i][j]._length_cache,
                        self._partitions[i][j]._width_cache,
                    )
                    for j in range(len(keys[i]))
                ]
                for i in range(len(keys))
            ]
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
        row_numeric_idx : list of int, optional
            The row indices to extract.
        col_indices : list of hashable, optional
            The column labels to extract.
        col_numeric_idx : list of int, optional
            The column indices to extract.

        Returns
        -------
        cuDFOnRayDataframe
             A new ``cuDFOnRayDataframe`` from the mask provided.

        Notes
        -----
        If both `row_indices` and `row_numeric_idx` are set, `row_indices` will be used.
        The same rule applied to `col_indices` and `col_numeric_idx`.
        """
        if isinstance(row_numeric_idx, slice) and (
            row_numeric_idx == slice(None) or row_numeric_idx == slice(0, None)
        ):
            row_numeric_idx = None
        if isinstance(col_numeric_idx, slice) and (
            col_numeric_idx == slice(None) or col_numeric_idx == slice(0, None)
        ):
            col_numeric_idx = None
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
            row_partitions_list = self._get_dict_of_block_index(0, row_numeric_idx)
            if isinstance(row_numeric_idx, slice):
                # Row lengths for slice are calculated as the length of the slice
                # on the partition. Often this will be the same length as the current
                # length, but sometimes it is different, thus the extra calculation.
                new_row_lengths = [
                    len(range(*idx.indices(self._row_lengths[p])))
                    for p, idx in row_partitions_list.items()
                ]
                # Use the slice to calculate the new row index
                new_index = self.index[row_numeric_idx]
            else:
                new_row_lengths = [len(idx) for _, idx in row_partitions_list.items()]
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
            col_partitions_list = self._get_dict_of_block_index(1, col_numeric_idx)
            if isinstance(col_numeric_idx, slice):
                # Column widths for slice are calculated as the length of the slice
                # on the partition. Often this will be the same length as the current
                # length, but sometimes it is different, thus the extra calculation.
                new_col_widths = [
                    len(range(*idx.indices(self._column_widths[p])))
                    for p, idx in col_partitions_list.items()
                ]
                # Use the slice to calculate the new columns
                new_columns = self.columns[col_numeric_idx]
                assert sum(new_col_widths) == len(
                    new_columns
                ), "{} != {}.\n{}\n{}\n{}".format(
                    sum(new_col_widths),
                    len(new_columns),
                    col_numeric_idx,
                    self._column_widths,
                    col_partitions_list,
                )
                if self._dtypes is not None:
                    new_dtypes = self.dtypes[col_numeric_idx]
                else:
                    new_dtypes = None
            else:
                new_col_widths = [len(idx) for _, idx in col_partitions_list.items()]
                new_columns = self.columns[sorted(col_numeric_idx)]
                if self._dtypes is not None:
                    new_dtypes = self.dtypes.iloc[sorted(col_numeric_idx)]
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

        key_and_gpus = np.array(
            [
                [
                    [
                        self._partitions[row_idx][col_idx].mask(
                            row_internal_indices, col_internal_indices
                        ),
                        self._partitions[row_idx][col_idx].get_gpu_manager(),
                    ]
                    for col_idx, col_internal_indices in col_partitions_list.items()
                    if isinstance(col_internal_indices, slice)
                    or len(col_internal_indices) > 0
                ]
                for row_idx, row_internal_indices in row_partitions_list.items()
                if isinstance(row_internal_indices, slice)
                or len(row_internal_indices) > 0
            ]
        )

        shape = key_and_gpus.shape[:2]
        keys = ray.get(key_and_gpus[:, :, 0].flatten().tolist())
        gpu_managers = key_and_gpus[:, :, 1].flatten().tolist()
        new_partitions = self._partition_mgr_cls._create_partitions(
            keys, gpu_managers
        ).reshape(shape)
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
            or isinstance(row_numeric_idx, slice)
            or len(row_numeric_idx) == 1
            or np.all(row_numeric_idx[1:] >= row_numeric_idx[:-1])
        ) and (
            col_numeric_idx is None
            or isinstance(col_numeric_idx, slice)
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
