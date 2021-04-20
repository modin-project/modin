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

from modin.engines.base.io import CSVDispatcher
from modin.engines.ray.cudf_on_ray.frame.partition_manager import GPU_MANAGERS
from typing import Tuple


class cuDFCSVDispatcher(CSVDispatcher):
    @classmethod
    def build_partition(cls, partition_ids, row_lengths, column_widths):
        def create_partition(i, j):
            return cls.frame_partition_cls(
                GPU_MANAGERS[i],
                partition_ids[i][j],
                length=row_lengths[i],
                width=column_widths[j],
            )

        return np.array(
            [
                [create_partition(i, j) for j in range(len(partition_ids[i]))]
                for i in range(len(partition_ids))
            ]
        )

    @classmethod
    def _launch_tasks(cls, splits: list, **partition_kwargs) -> Tuple[list, list, list]:
        """
        Launch tasks to read partitions.
        ----------
        splits: list
            list of tuples with partitions data, which defines
            parser task (start/end read bytes and etc.)
        partition_kwargs:
            kwargs that should be passed to the parser function.

        Returns
        -------
        partition_ids: list
                array with references to the partitions data.
        index_ids: list
                array with references to the partitions index objects.
        dtypes_ids: list
                array with references to the partitions dtypes objects.
        """
        partition_ids = []
        index_ids = []
        dtypes_ids = []
        gpu_manager = 0
        for start, end in splits:
            partition_kwargs.update({"start": start, "end": end, "gpu": gpu_manager})
            partition_id = cls.deploy(
                cls.parse, partition_kwargs.get("num_splits") + 2, partition_kwargs
            )
            partition_ids.append(partition_id[:-2])
            index_ids.append(partition_id[-2])
            dtypes_ids.append(partition_id[-1])
            gpu_manager += 1

        return partition_ids, index_ids, dtypes_ids
