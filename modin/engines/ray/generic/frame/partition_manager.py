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

from modin.engines.base.frame.partition_manager import BaseFrameManager
from modin import __execution_engine__

if __execution_engine__ == "Ray":
    import ray


class RayFrameManager(BaseFrameManager):
    """This method implements the interface in `BaseFrameManager`."""

    @classmethod
    def to_numpy(cls, partitions):
        """Convert this object into a NumPy Array from the partitions.

        Returns:
            A NumPy Array
        """
        parts = ray.get(
            [
                obj.apply(lambda df: df.to_numpy()).oid
                for row in partitions
                for obj in row
            ]
        )
        n = partitions.shape[1]
        parts = [parts[i * n : (i + 1) * n] for i in list(range(partitions.shape[0]))]

        arr = np.block(parts)
        return arr
