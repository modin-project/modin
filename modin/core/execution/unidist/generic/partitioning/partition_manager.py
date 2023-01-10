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

"""The module holds Modin partition manager implemented for unidist."""

import numpy as np

from modin.core.dataframe.pandas.partitioning.partition_manager import (
    PandasDataframePartitionManager,
)
from modin.core.execution.unidist.common import UnidistWrapper


class GenericUnidistDataframePartitionManager(PandasDataframePartitionManager):
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
        parts = UnidistWrapper.materialize(
            [
                obj.apply(lambda df, **kwargs: df.to_numpy(**kwargs)).list_of_blocks[0]
                for row in partitions
                for obj in row
            ]
        )
        rows, cols = partitions.shape
        parts = [parts[i * cols : (i + 1) * cols] for i in range(rows)]
        return np.block(parts)
