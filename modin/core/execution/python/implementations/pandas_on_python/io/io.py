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

"""Module for housing IO classes with pandas storage format and Python engine."""

from modin.core.execution.python.implementations.pandas_on_python.dataframe.dataframe import (
    PandasOnPythonDataframe,
)
from modin.core.io import BaseIO
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler


class PandasOnPythonIO(BaseIO):
    """
    Class for storing IO functions operating on pandas storage format and Python engine.

    Inherits default function implementations from ``BaseIO`` parent class.
    """

    frame_cls = PandasOnPythonDataframe
    query_compiler_cls = PandasQueryCompiler
