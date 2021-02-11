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

import ray


class PandasOnRayFrameManager(RayFrameManager):
    """This method implements the interface in `BaseFrameManager`."""

    # This object uses RayRemotePartition objects as the underlying store.
    _partition_class = PandasOnRayFramePartition
    _column_partitions_class = PandasOnRayFrameColumnPartition
    _row_partition_class = PandasOnRayFrameRowPartition

    @classmethod
    def get_indices(cls, axis, partitions, index_func=None):
        """
        This gets the internal indices stored in the partitions.
        Parameters
        ----------
            axis : 0 or 1
                This axis to extract the labels (0 - index, 1 - columns).
            partitions : NumPy array
                The array of partitions from which need to extract the labels.
            index_func : callable
                The function to be used to extract the function.
        Returns
        -------
        Index
            A Pandas Index object.
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
    def broadcast_apply(cls, axis, apply_func, left, right, other_name="r"):
        lt_axis_parts = cls.axis_partition(left, 1)
        rt_axis_parts = cls.axis_partition(right, axis ^ 1)
        return np.array(
            [
                lt_axis_part.broadcast_apply(
                    rt_axis_parts if axis else rt_axis_parts[i],
                    axis,
                    apply_func,
                    other_name,
                )
                for i, lt_axis_part in enumerate(lt_axis_parts)
            ]
        )
