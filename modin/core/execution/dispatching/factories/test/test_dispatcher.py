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

from modin.config import Engine, StorageFormat
from modin import set_execution

from modin.core.execution.dispatching.factories.dispatcher import (
    FactoryDispatcher,
    FactoryNotFoundError,
)
from modin.core.execution.dispatching.factories import factories

import modin.pandas as pd


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

# register them as known "no init" engines for modin.pandas
pd._NOINIT_ENGINES |= {"Test", "Bar"}


def test_default_factory():
    assert issubclass(FactoryDispatcher.get_factory(), factories.BaseFactory)
    assert FactoryDispatcher.get_factory().io_cls


def test_factory_switch():
    Engine.put("Test")
    assert FactoryDispatcher.get_factory() == PandasOnTestFactory
    assert FactoryDispatcher.get_factory().io_cls == "Foo"
    Engine.put("Python")  # revert engine to default

    StorageFormat.put("Test")
    assert FactoryDispatcher.get_factory() == TestOnPythonFactory
    assert FactoryDispatcher.get_factory().io_cls == "Bar"
    StorageFormat.put("Pandas")  # revert engine to default


def test_engine_wrong_factory():
    with pytest.raises(FactoryNotFoundError):
        Engine.put("BadEngine")
    Engine.put("Python")  # revert engine to default


def test_set_execution():
    set_execution("Bar", "Foo")
    assert FactoryDispatcher.get_factory() == FooOnBarFactory
