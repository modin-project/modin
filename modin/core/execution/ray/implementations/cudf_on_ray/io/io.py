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

"""Module holds implementation of ``BaseIO`` using cuDF."""

from modin.core.io import BaseIO
from modin.core.storage_formats.cudf.query_compiler import cuDFQueryCompiler
from modin.core.storage_formats.cudf.parser import cuDFCSVParser
from modin.core.execution.ray.common import RayTask
from ..dataframe import cuDFOnRayDataframe
from ..partitioning import (
    cuDFOnRayDataframePartition,
    cuDFOnRayDataframePartitionManager,
)
from .text import cuDFCSVDispatcher


class cuDFOnRayIO(BaseIO):
    """The class implements ``BaseIO`` class using cuDF-entities."""

    frame_cls = cuDFOnRayDataframe
    query_compiler_cls = cuDFQueryCompiler

    build_args = dict(
        frame_partition_cls=cuDFOnRayDataframePartition,
        query_compiler_cls=cuDFQueryCompiler,
        frame_cls=cuDFOnRayDataframe,
        frame_partition_mgr_cls=cuDFOnRayDataframePartitionManager,
    )

    read_csv = type("", (RayTask, cuDFCSVParser, cuDFCSVDispatcher), build_args).read
