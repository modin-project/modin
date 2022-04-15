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

"""Module houses class that implements ``PandasDataframePartitionManager``."""

import numpy as np

from modin.core.dataframe.pandas.partitioning.partition_manager import (
    PandasDataframePartitionManager,
)
from modin.core.execution.dask.common.engine_wrapper import DaskWrapper
from .virtual_partition import (
    PandasOnDaskDataframeColumnPartition,
    PandasOnDaskDataframeRowPartition,
)
from .partition import PandasOnDaskDataframePartition


class PandasOnDaskDataframePartitionManager(PandasDataframePartitionManager):
    """The class implements the interface in `PandasDataframePartitionManager`."""

    # This object uses PandasOnDaskDataframePartition objects as the underlying store.
    _partition_class = PandasOnDaskDataframePartition
    _column_partitions_class = PandasOnDaskDataframeColumnPartition
    _row_partition_class = PandasOnDaskDataframeRowPartition

    @classmethod
    def get_objects_from_partitions(cls, partitions):
        """
        Get the objects wrapped by `partitions` in parallel.

        Parameters
        ----------
        partitions : np.ndarray
            NumPy array with ``PandasDataframePartition``-s.

        Returns
        -------
        list
            The objects wrapped by `partitions`.
        """
        return DaskWrapper.materialize([partition.future for partition in partitions])

    @classmethod
    def binary_operation(cls, left, func, right, axis=1):
        """
        Apply a function that requires partitions of two ``PandasOnRayDataframe`` objects.

        Parameters
        ----------
        left : np.ndarray
            The partitions of left ``PandasOnRayDataframe``.
        func : callable
            The function to apply.
        right : np.ndarray
            The partitions of right ``PandasOnRayDataframe``.
        axis : {0, 1}, default: 1
            The axis to apply the function over (0 - rows, 1 - columns).

        Returns
        -------
        np.ndarray
            A NumPy array with new partitions.
        """
        func = cls.preprocess_func(func)
        return np.array(
            [
                [
                    part.apply(func, right[row_idx][col_idx].future)
                    for col_idx, part in enumerate(left[row_idx])
                ]
                for row_idx in range(len(left))
            ]
        )
