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

from contextlib import contextmanager

import pytest

import modin.pandas as pd
from modin import set_execution
from modin.config import Engine, Parameter, StorageFormat
from modin.core.execution.dispatching.factories import factories
from modin.core.execution.dispatching.factories.dispatcher import (
    FactoryDispatcher,
    FactoryNotFoundError,
)
from modin.core.execution.python.implementations.pandas_on_python.io import (
    PandasOnPythonIO,
)
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler


@contextmanager
def _switch_execution(engine: str, storage_format: str):
    old_engine, old_storage = set_execution(engine, storage_format)
    try:
        yield
    finally:
        set_execution(old_engine, old_storage)


@contextmanager
def _switch_value(config: Parameter, value: str):
    old_value = config.get()
    try:
        yield config.put(value)
    finally:
        config.put(old_value)


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
Engine.NOINIT_ENGINES |= {"Test", "Bar"}


def test_default_factory():
    assert issubclass(FactoryDispatcher.get_factory(), factories.BaseFactory)
    assert FactoryDispatcher.get_factory().io_cls


def test_factory_switch():
    with _switch_execution("Python", "Pandas"):
        with _switch_value(Engine, "Test"):
            assert FactoryDispatcher.get_factory() == PandasOnTestFactory
            assert FactoryDispatcher.get_factory().io_cls == "Foo"

        with _switch_value(StorageFormat, "Test"):
            assert FactoryDispatcher.get_factory() == TestOnPythonFactory
            assert FactoryDispatcher.get_factory().io_cls == "Bar"


def test_engine_wrong_factory():
    with pytest.raises(FactoryNotFoundError):
        with _switch_value(Engine, "Dask"):
            with _switch_value(StorageFormat, "Pyarrow"):
                pass


def test_set_execution():
    with _switch_execution("Bar", "Foo"):
        assert FactoryDispatcher.get_factory() == FooOnBarFactory


def test_add_option():
    class DifferentlyNamedFactory(factories.BaseFactory):
        @classmethod
        def prepare(cls):
            cls.io_cls = PandasOnPythonIO

    factories.StorageOnExecFactory = DifferentlyNamedFactory
    StorageFormat.add_option("sToragE")
    Engine.add_option("Exec")

    with _switch_execution("Exec", "Storage"):
        df = pd.DataFrame([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
        assert isinstance(df._query_compiler, PandasQueryCompiler)
