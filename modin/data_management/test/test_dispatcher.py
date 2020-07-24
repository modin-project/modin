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

from modin import execution_engine, partition_format, set_backends

from modin.data_management.dispatcher import EngineDispatcher, FactoryNotFoundError
from modin.data_management import factories


class PandasOnTestFactory(factories.BaseFactory):
    """
    Stub factory to ensure we can switch execution engine to 'Test'
    """

    @classmethod
    def prepare(cls):
        """
        Fills in .io_cls class attribute lazily
        """
        cls.io_cls = "Foo"


class TestOnPythonFactory(factories.BaseFactory):
    """
    Stub factory to ensure we can switch partition format to 'Test'
    """

    @classmethod
    def prepare(cls):
        """
        Fills in .io_cls class attribute lazily
        """
        cls.io_cls = "Bar"


class FooOnBarFactory(factories.BaseFactory):
    """
    Stub factory to ensure we can switch engine and partition to 'Foo' and 'Bar'
    """

    @classmethod
    def prepare(cls):
        """
        Fills in .io_cls class attribute lazily
        """
        cls.io_cls = "Zug-zug"


# inject the stubs
factories.PandasOnTestFactory = PandasOnTestFactory
factories.TestOnPythonFactory = TestOnPythonFactory
factories.FooOnBarFactory = FooOnBarFactory


def test_default_engine():
    assert issubclass(EngineDispatcher.get_engine(), factories.BaseFactory)
    assert EngineDispatcher.get_engine().io_cls


def test_engine_switch():
    execution_engine.put("Test")
    assert EngineDispatcher.get_engine() == PandasOnTestFactory
    assert EngineDispatcher.get_engine().io_cls == "Foo"
    execution_engine.put("Python")  # revert engine to default

    partition_format.put("Test")
    assert EngineDispatcher.get_engine() == TestOnPythonFactory
    assert EngineDispatcher.get_engine().io_cls == "Bar"
    partition_format.put("Pandas")  # revert engine to default


def test_engine_wrong_factory():
    with pytest.raises(FactoryNotFoundError):
        execution_engine.put("BadEngine")
    execution_engine.put("Python")  # revert engine to default


def test_set_backends():
    set_backends("Bar", "Foo")
    assert EngineDispatcher.get_engine() == FooOnBarFactory
