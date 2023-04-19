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

"""
Module houses experimental IO classes and parser functions needed for these classes.

Any function or class can be considered experimental API if it is not strictly replicating existent
Query Compiler API, even if it is only extending the API.
"""

from modin.core.storage_formats.pandas.parsers import (
    PandasCSVGlobParser,
    ExperimentalPandasPickleParser,
    ExperimentalCustomTextParser,
)
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.core.execution.ray.implementations.pandas_on_ray.io import PandasOnRayIO
from modin.experimental.core.io import (
    ExperimentalCSVGlobDispatcher,
    ExperimentalSQLDispatcher,
    ExperimentalPickleDispatcher,
    ExperimentalCustomTextDispatcher,
)
from modin.core.execution.ray.implementations.pandas_on_ray.dataframe import (
    PandasOnRayDataframe,
)
from modin.core.execution.ray.implementations.pandas_on_ray.partitioning import (
    PandasOnRayDataframePartition,
)
from modin.core.execution.ray.common import RayWrapper


class ExperimentalPandasOnRayIO(PandasOnRayIO):
    """
    Class for handling experimental IO functionality with pandas storage format and Ray engine.

    ``ExperimentalPandasOnRayIO`` inherits some util functions and unmodified IO functions
    from ``PandasOnRayIO`` class.
    """

    build_args = dict(
        frame_partition_cls=PandasOnRayDataframePartition,
        query_compiler_cls=PandasQueryCompiler,
        frame_cls=PandasOnRayDataframe,
        base_io=PandasOnRayIO,
    )

    def __make_read(*classes, build_args=build_args):
        # used to reduce code duplication
        return type("", (RayWrapper, *classes), build_args).read

    def __make_write(*classes, build_args=build_args):
        # used to reduce code duplication
        return type("", (RayWrapper, *classes), build_args).write

    read_csv_glob = __make_read(PandasCSVGlobParser, ExperimentalCSVGlobDispatcher)
    read_pickle_distributed = __make_read(
        ExperimentalPandasPickleParser, ExperimentalPickleDispatcher
    )
    to_pickle_distributed = __make_write(ExperimentalPickleDispatcher)
    read_custom_text = __make_read(
        ExperimentalCustomTextParser, ExperimentalCustomTextDispatcher
    )
    read_sql = __make_read(ExperimentalSQLDispatcher)

    del __make_read  # to not pollute class namespace
    del __make_write  # to not pollute class namespace
