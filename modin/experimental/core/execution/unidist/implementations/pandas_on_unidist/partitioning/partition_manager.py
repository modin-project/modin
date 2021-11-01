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

"""Module houses class that implements ``GenericUnidistDataframePartitionManager`` using unidist."""

import numpy as np

from modin.experimental.core.execution.unidist.generic.partitioning.partition_manager import (
    GenericUnidistDataframePartitionManager,
)
from .axis_partition import (
    PandasOnUnidistDataframeColumnPartition,
    PandasOnUnidistDataframeRowPartition,
)
from .partition import PandasOnUnidistDataframePartition
from modin.error_message import ErrorMessage
import pandas

import unidist


class PandasOnUnidistDataframePartitionManager(GenericUnidistDataframePartitionManager):
    """The class implements the interface in `GenericUnidistDataframePartitionManager`."""

    # This object uses PandasOnUnidistDataframePartition objects as the underlying store.
    _partition_class = PandasOnUnidistDataframePartition
    _column_partitions_class = PandasOnUnidistDataframeColumnPartition
    _row_partition_class = PandasOnUnidistDataframeRowPartition

    @classmethod
    def get_indices(cls, axis, partitions, index_func=None):
        """
        Get the internal indices stored in the partitions.

        Parameters
        ----------
        axis : {0, 1}
            Axis to extract the labels over.
        partitions : np.ndarray
            NumPy array with ``PandasFramePartition``-s.
        index_func : callable, default: None
            The function to be used to extract the indices.

        Returns
        -------
        pandas.Index
            A ``pandas.Index`` object.

        Notes
        -----
        These are the global indices of the object. This is mostly useful
        when you have deleted rows/columns internally, but do not know
        which ones were deleted.
        """
        ErrorMessage.catch_bugs_and_request_email(not callable(index_func))
        func = cls.preprocess_func(index_func)
        if axis == 0:
            # We grab the first column of blocks and extract the indices
            new_idx = (
                [idx.apply(func).oid for idx in partitions.T[0]]
                if len(partitions.T)
                else []
            )
        else:
            new_idx = (
                [idx.apply(func).oid for idx in partitions[0]]
                if len(partitions)
                else []
            )
        new_idx = unidist.get(new_idx)
        return new_idx[0].append(new_idx[1:]) if len(new_idx) else new_idx

    @classmethod
    def broadcast_apply(cls, axis, apply_func, left, right, other_name="r"):
        """
        Broadcast the `right` partitions to `left` and apply `apply_func` to selected indices.

        Parameters
        ----------
        axis : {0, 1}
            Axis to apply and broadcast over.
        apply_func : callable
            Function to apply.
        left : np.ndarray
            NumPy 2D array of left partitions.
        right : np.ndarray
            NumPy 2D array of right partitions.
        other_name : str, default: "r"
            Name of key-value argument for `apply_func` that
            is used to pass `right` to `apply_func`.

        Returns
        -------
        np.ndarray
            An array of partition objects.
        """

        def map_func(df, *others):
            other = pandas.concat(others, axis=axis ^ 1)
            return apply_func(df, **{other_name: other})

        map_func = cls.preprocess_func(map_func)
        rt_axis_parts = cls.axis_partition(right, axis ^ 1)
        return np.array(
            [
                [
                    part.apply(
                        map_func,
                        *(
                            rt_axis_parts[col_idx].list_of_blocks
                            if axis
                            else rt_axis_parts[row_idx].list_of_blocks
                        ),
                    )
                    for col_idx, part in enumerate(left[row_idx])
                ]
                for row_idx in range(len(left))
            ]
        )
