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

import warnings

from modin.engines.base.io.file_dispatcher import FileDispatcher
from modin.config import NPartitions


class PickleExperimentalDispatcher(FileDispatcher):
    @classmethod
    def _read(cls, filepath_or_buffer, **kwargs):
        """
        In experimental mode, we can pass a list of files as an input parameter.

        Note: the number of partitions is equal to the number of input files.
        """
        if not isinstance(filepath_or_buffer, (str, list)):
            warnings.warn("Defaulting to Modin core implementation")
            return cls.single_worker_read(
                filepath_or_buffer,
                **kwargs,
            )

        if isinstance(filepath_or_buffer, list) and not all(
            map(lambda filepath: isinstance(filepath, str), filepath_or_buffer)
        ):
            raise TypeError(
                f"Only support list[str], passed value: {filepath_or_buffer}"
            )

        if len(filepath_or_buffer) == 0:
            raise ValueError(
                "filepath_or_buffer parameter of read_pickle is empty list"
            )

        partition_ids = []
        lengths_ids = []
        widths_ids = []

        if len(filepath_or_buffer) != NPartitions.get():
            # do we need to do a repartitioning?
            warnings.warn("can be inefficient partitioning")

        for file_name in filepath_or_buffer:
            partition_id = cls.deploy(
                cls.parse,
                3,
                dict(
                    file_name,
                    **kwargs,
                ),
            )
            partition_ids.append(partition_id[:-2])
            lengths_ids.append(partition_id[-2])
            widths_ids.append(partition_id[-1])

        lengths = cls.materialize(lengths_ids)
        widths = cls.materialize(widths_ids)

        # while num_splits is 1, need only one value
        partition_ids = cls.build_partition(partition_ids, lengths, [widths[0]])

        new_index = cls.frame_cls._frame_mgr_cls.get_indices(
            0, partition_ids, lambda df: df.axes[0]
        )
        new_columns = cls.frame_cls._frame_mgr_cls.get_indices(
            1, partition_ids, lambda df: df.axes[1]
        )

        return cls.query_compiler_cls(
            cls.frame_cls(partition_ids, new_index, new_columns)
        )
