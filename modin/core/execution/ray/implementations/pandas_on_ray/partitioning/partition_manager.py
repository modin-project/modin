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

"""Module houses class that implements ``GenericRayDataframePartitionManager`` using Ray."""

import numpy as np
import pandas
import ray

from modin.config import NPartitions
from modin.core.execution.ray.generic.partitioning.partition_manager import (
    GenericRayDataframePartitionManager,
)
from .virtual_partition import (
    PandasOnRayDataframeColumnPartition,
    PandasOnRayDataframeRowPartition,
)
from .partition import PandasOnRayDataframePartition
from modin.core.storage_formats.pandas.utils import compute_chunksize
from modin.error_message import ErrorMessage


class PandasOnRayDataframePartitionManager(GenericRayDataframePartitionManager):
    """The class implements the interface in `PandasDataframePartitionManager`."""

    # This object uses RayRemotePartition objects as the underlying store.
    _partition_class = PandasOnRayDataframePartition
    _column_partitions_class = PandasOnRayDataframeColumnPartition
    _row_partition_class = PandasOnRayDataframeRowPartition

    @classmethod
    def get_indices(cls, axis, partitions, index_func=None):
        """
        Get the internal indices stored in the partitions.

        Parameters
        ----------
        axis : {0, 1}
            Axis to extract the labels over.
        partitions : np.ndarray
            NumPy array with ``PandasDataframePartition``-s.
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
        new_idx = ray.get(new_idx)
        return new_idx[0].append(new_idx[1:]) if len(new_idx) else new_idx

    @classmethod
    def concat(cls, axis, left_parts, right_parts):
        """
        Concatenate the blocks of partitions with another set of blocks.

        Parameters
        ----------
        axis : int
            The axis to concatenate to.
        left_parts : np.ndarray
            NumPy array of partitions to concatenate with.
        right_parts : np.ndarray or list
            NumPy array of partitions to be concatenated.

        Returns
        -------
        np.ndarray
            A new NumPy array with concatenated partitions.

        Notes
        -----
        Assumes that the `left_parts` and `right_parts` blocks are already the same
        shape on the dimension (opposite `axis`) as the one being concatenated. A
        ``ValueError`` will be thrown if this condition is not met.
        """
        result = super(PandasOnRayDataframePartitionManager, cls).concat(
            axis, left_parts, right_parts
        )
        if axis == 0:
            return cls.rebalance_partitions(result)
        else:
            return result

    @classmethod
    def rebalance_partitions(cls, partitions):
        """
        Rebalance a 2-d array of partitions.

        Rebalance the partitions by building a new array
        of partitions out of the original ones so that:
          - If all partitions have a length, each new partition has roughly the
            same number of rows.
          - Otherwise, each new partition spans roughly the same number of old
            partitions.

        Parameters
        ----------
        partitions : np.ndarray
            The 2-d array of partitions to rebalance.

        Returns
        -------
        np.ndarray
            A new NumPy array with rebalanced partitions.
        """
        # We rebalance when the ratio of the number of existing partitions to
        # the ideal number of partitions is larger than this threshold. The
        # threshold is a heuristic that may need to be tuned for performance.
        max_excess_of_num_partitions = 1.5
        num_existing_partitions = partitions.shape[0]
        ideal_num_new_partitions = NPartitions.get()
        if (
            num_existing_partitions
            <= ideal_num_new_partitions * max_excess_of_num_partitions
        ):
            return partitions
        # If any partition has an unknown length, give each axis partition
        # roughly the same number of row partitions. We use `_length_cache` here
        # to avoid materializing any unmaterialized lengths.
        if any(
            partition._length_cache is None for row in partitions for partition in row
        ):
            # We need each partition to go into an axis partition, but the
            # number of axis partitions may not evenly divide the number of
            # partitions.
            chunk_size = compute_chunksize(
                num_existing_partitions, ideal_num_new_partitions, min_block_size=1
            )
            return np.array(
                [
                    cls.column_partitions(
                        partitions[i : i + chunk_size],
                        full_axis=False,
                    )
                    for i in range(
                        0,
                        num_existing_partitions,
                        chunk_size,
                    )
                ]
            )

        # If we know the number of rows in every partition, then we should try
        # instead to give each new partition roughly the same number of rows.
        new_partitions = []
        # `start` is the index of the first existing partition that we want to
        # put into the current new partition.
        start = 0
        total_rows = sum(part.length() for part in partitions[:, 0])
        ideal_partition_size = compute_chunksize(
            total_rows, ideal_num_new_partitions, min_block_size=1
        )
        for _ in range(ideal_num_new_partitions):
            # We might pick up old partitions too quickly and exhaust all of them.
            if start >= len(partitions):
                break
            # `stop` is the index of the last existing partition so far that we
            # want to put into the current new partition.
            stop = start
            partition_size = partitions[start][0].length()
            # Add existing partitions into the current new partition until the
            # number of rows in the new partition hits `ideal_partition_size`.
            while stop < len(partitions) and partition_size < ideal_partition_size:
                stop += 1
                if stop < len(partitions):
                    partition_size += partitions[stop][0].length()
            # If the new partition is larger than we want, split the last
            # current partition that it contains into two partitions, where
            # the first partition has just enough rows to make the current
            # new partition have length `ideal_partition_size`, and the second
            # partition has the remainder.
            if partition_size > ideal_partition_size * max_excess_of_num_partitions:
                new_last_partition_size = ideal_partition_size - sum(
                    row[0].length() for row in partitions[start:stop]
                )
                partitions = np.insert(
                    partitions,
                    stop + 1,
                    [
                        obj.mask(slice(new_last_partition_size, None), slice(None))
                        for obj in partitions[stop]
                    ],
                    0,
                )
                partitions[stop, :] = [
                    obj.mask(slice(None, new_last_partition_size), slice(None))
                    for obj in partitions[stop]
                ]
                partition_size = ideal_partition_size
            new_partitions.append(
                cls.column_partitions(
                    (partitions[start : stop + 1]),
                    full_axis=partition_size == total_rows,
                )
            )
            start = stop + 1
        return np.array(new_partitions)

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
