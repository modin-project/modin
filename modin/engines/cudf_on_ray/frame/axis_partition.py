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

import cudf
import ray

from .partition import cuDFOnRayFramePartition


class cuDFOnRayFrameAxisPartition(object):

    def __init__(self, partitions):
        self.partitions = [obj for obj in partitions]

    partition_type = cuDFOnRayFramePartition
    instance_type = cudf.DataFrame

class cuDFOnRayFrameColumnPartition(cuDFOnRayFrameAxisPartition):
    axis = 0
    def reduce(self, func):
        keys = [partition.get_key() for partition in self.partitions]
        gpu_managers = [partition.get_gpu_manager() for partition in self.partitions]
        head_gpu_manager = gpu_managers[0]
        cudf_dataframe_object_ids = [
            gpu_manager.get.remote(key)
            for gpu_manager, key in zip(gpu_managers, keys)
        ]
        key = head_gpu_manager.reduce.remote(cudf_dataframe_object_ids, axis=self.axis, func=func)
        key = ray.get(key)
        result = cuDFOnRayFramePartition(gpu_manager=head_gpu_manager, key=key)
        return result

class cuDFOnRayFrameRowPartition(cuDFOnRayFrameAxisPartition):
    axis = 1
    # Since we are using row partitions, we can bypass the ray plasma store during axis reduction
    # functions.
    def reduce(self, func):
        keys = [partition.get_key() for partition in self.partitions]
        gpu = self.partitions[0].get_gpu_manager()
        key = gpu.reduce_key_list.remote(keys, func)
        key = ray.get(key)
        return cuDFOnRayFramePartition(gpu_manager=gpu, key=key)
