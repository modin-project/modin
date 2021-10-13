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

"""The module holds Modin partition manager implemented for Ray."""

import numpy as np

from modin.core.dataframe.pandas.partitioning.partition_manager import (
    PandasDataframePartitionManager,
)

import ray


class GenericRayDataframePartitionManager(PandasDataframePartitionManager):
    """The class implements the interface in `PandasDataframePartitionManager`."""

    @classmethod
    def to_numpy(cls, partitions, **kwargs):
        """
        Convert `partitions` into a NumPy array.

        Parameters
        ----------
        partitions : NumPy array
            A 2-D array of partitions to convert to local NumPy array.
        **kwargs : dict
            Keyword arguments to pass to each partition ``.to_numpy()`` call.

        Returns
        -------
        NumPy array
        """
        parts = ray.get(
            [
                obj.apply(lambda df, **kwargs: df.to_numpy(**kwargs)).oid
                for row in partitions
                for obj in row
            ]
        )
        n = partitions.shape[1]
        parts = [parts[i * n : (i + 1) * n] for i in list(range(partitions.shape[0]))]

        arr = np.block(parts)
        return arr
