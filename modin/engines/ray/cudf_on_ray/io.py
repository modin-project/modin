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

from modin.engines.base.io import BaseIO
from modin.engines.base.io import CSVDispatcher
from modin.backends.cudf.query_compiler import cuDFQueryCompiler
from modin.engines.ray.cudf_on_ray.frame.data import cuDFOnRayFrame
from modin.engines.ray.cudf_on_ray.frame.partition_manager import cuDFOnRayFrameManager
from modin.engines.ray.cudf_on_ray.frame.partition import cuDFOnRayFramePartition


from modin.engines.ray.task_wrapper import RayTask
from modin.backends.cudf.parser import cuDFCSVParser

class cuDFOnRayIO(BaseIO):

    frame_cls = cuDFOnRayFrame
    query_compiler_cls = cuDFQueryCompiler

    build_args = dict(
        frame_partition_cls=cuDFOnRayFramePartition,
        query_compiler_cls=cuDFQueryCompiler,
        frame_cls=cuDFOnRayFrame,
        frame_partition_mgr_cls=cuDFOnRayFrameManager,
    )

    read_csv = type("", (RayTask, cuDFCSVParser, CSVDispatcher), build_args).read
