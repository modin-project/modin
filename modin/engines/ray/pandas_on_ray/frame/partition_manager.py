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

from modin.engines.ray.generic.frame.partition_manager import RayFrameManager
from .axis_partition import (
    PandasOnRayFrameColumnPartition,
    PandasOnRayFrameRowPartition,
)
from .partition import PandasOnRayFramePartition
from modin.error_message import ErrorMessage
from modin import __execution_engine__

if __execution_engine__ == "Ray":
    import ray

    @ray.remote
    def func(df, other, apply_func, call_queue_df=None, call_queue_other=None):
        if call_queue_df is not None and len(call_queue_df) > 0:
            for call, kwargs in call_queue_df:
                if isinstance(call, ray.ObjectID):
                    call = ray.get(call)
                if isinstance(kwargs, ray.ObjectID):
                    kwargs = ray.get(kwargs)
                df = call(df, **kwargs)
        if call_queue_other is not None and len(call_queue_other) > 0:
            for call, kwargs in call_queue_other:
                if isinstance(call, ray.ObjectID):
                    call = ray.get(call)
                if isinstance(kwargs, ray.ObjectID):
                    kwargs = ray.get(kwargs)
                other = call(other, **kwargs)
        return apply_func(df, other)


class PandasOnRayFrameManager(RayFrameManager):
    """This method implements the interface in `BaseFrameManager`."""

    # This object uses RayRemotePartition objects as the underlying store.
    _partition_class = PandasOnRayFramePartition
    _column_partitions_class = PandasOnRayFrameColumnPartition
    _row_partition_class = PandasOnRayFrameRowPartition

    @classmethod
    def get_indices(cls, axis, partitions, index_func=None):
        """This gets the internal indices stored in the partitions.

        Note: These are the global indices of the object. This is mostly useful
            when you have deleted rows/columns internally, but do not know
            which ones were deleted.

        Args:
            axis: This axis to extract the labels. (0 - index, 1 - columns).
            index_func: The function to be used to extract the function.
            old_blocks: An optional previous object that this object was
                created from. This is used to compute the correct offsets.

        Returns:
            A Pandas Index object.
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
    def groupby_reduce(
        cls, axis, partitions, by, map_func, reduce_func
    ):  # pragma: no cover
        map_func = ray.put(map_func)
        by_parts = np.squeeze(by)
        if len(by_parts.shape) == 0:
            by_parts = np.array([by_parts.item()])
        new_partitions = np.array(
            [
                [
                    PandasOnRayFramePartition(
                        func.remote(
                            part.oid,
                            by_parts[col_idx].oid if axis else by_parts[row_idx].oid,
                            map_func,
                            part.call_queue,
                            by_parts[col_idx].call_queue
                            if axis
                            else by_parts[row_idx].call_queue,
                        )
                    )
                    for col_idx, part in enumerate(partitions[row_idx])
                ]
                for row_idx in range(len(partitions))
            ]
        )
        return cls.map_axis_partitions(axis, new_partitions, reduce_func)

    @classmethod
    def broadcast_apply(cls, axis, apply_func, left, right):
        map_func = ray.put(apply_func)
        right_parts = np.squeeze(right)
        if len(right_parts.shape) == 0:
            right_parts = np.array([right_parts.item()])
        assert (
            len(right_parts.shape) == 1
        ), "Invalid broadcast partitions shape {}\n{}".format(
            right_parts.shape, [[i.get() for i in j] for j in right_parts]
        )
        return np.array(
            [
                [
                    PandasOnRayFramePartition(
                        func.remote(
                            part.oid,
                            right_parts[col_idx].oid
                            if axis
                            else right_parts[row_idx].oid,
                            map_func,
                            part.call_queue,
                            right_parts[col_idx].call_queue
                            if axis
                            else right_parts[row_idx].call_queue,
                        )
                    )
                    for col_idx, part in enumerate(left[row_idx])
                ]
                for row_idx in range(len(left))
            ]
        )
