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

import pytest

from modin.config import Backend, Engine, Execution, StorageFormat
from modin.core.execution.dispatching.factories import factories
from modin.core.execution.dispatching.factories.factories import BaseFactory, NativeIO
from modin.core.storage_formats.pandas.native_query_compiler import NativeQueryCompiler
from modin.pandas.api.extensions.extensions import _NON_EXTENDABLE_ATTRIBUTES


class Test1QueryCompiler(NativeQueryCompiler):
    storage_format = property(lambda self: "Test1_Storage_Format")
    engine = property(lambda self: "Test1_Engine")


class Test1IO(NativeIO):
    query_compiler_cls = Test1QueryCompiler


class Test1Factory(BaseFactory):

    @classmethod
    def prepare(cls):
        cls.io_cls = Test1IO


@pytest.fixture
def Backend1():
    factories.Test1_Storage_FormatOnTest1_EngineFactory = Test1Factory
    if "Backend1" not in Backend.choices:
        StorageFormat.add_option("Test1_storage_format")
        Engine.add_option("Test1_engine")
        Backend.register_backend(
            "Backend1",
            Execution(storage_format="Test1_Storage_Format", engine="Test1_Engine"),
        )
    return "Backend1"


@pytest.fixture(
    # sort the set of non-extendable attributes to make the sequence of test
    # cases deterministic for pytest-xdist.
    params=sorted(_NON_EXTENDABLE_ATTRIBUTES),
)
def non_extendable_attribute_name(request) -> str:
    return request.param
