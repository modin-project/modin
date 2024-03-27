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

import collections
import logging

import pytest

import modin.logging
from modin.config import LogMode


class _FakeLogger:
    _loggers = {}

    def __init__(self, namespace):
        self.messages = collections.defaultdict(list)
        self.namespace = namespace

    def log(self, log_level, message, *args, **kw):
        self.messages[log_level].append(message.format(*args, **kw))

    def exception(self, message, *args, **kw):
        self.messages["exception"].append(message.format(*args, **kw))

    @classmethod
    def make(cls, namespace):
        return cls._loggers.setdefault(namespace, cls(namespace))

    @classmethod
    def get(cls, namespace="modin.logger.default"):
        return cls._loggers[namespace].messages

    @classmethod
    def clear(cls):
        cls._loggers = {}


def _get_logger(namespace="modin.logger.default"):
    return _FakeLogger.make(namespace)


def mock_get_logger(ctx):
    ctx.setattr(logging, "getLogger", _get_logger)


@pytest.fixture
def get_log_messages():
    old = LogMode.get()
    LogMode.enable()
    modin.logging.get_logger()  # initialize the logging pior to mocking getLogger()

    yield _FakeLogger.get

    _FakeLogger.clear()
    LogMode.put(old)


def test_function_decorator(monkeypatch, get_log_messages):
    @modin.logging.enable_logging
    def func(do_raise):
        if do_raise:
            raise ValueError()

    with monkeypatch.context() as ctx:
        # NOTE: we cannot patch in the fixture as mockin logger.getLogger()
        # without monkeypatch.context() breaks pytest
        mock_get_logger(ctx)

        func(do_raise=False)
        with pytest.raises(ValueError):
            func(do_raise=True)

    assert "func" in get_log_messages()[logging.INFO][0]
    assert "START" in get_log_messages()[logging.INFO][0]
    assert get_log_messages("modin.logger.errors")["exception"] == [
        "STOP::PANDAS-API::func"
    ]


def test_function_decorator_on_outer_function_6237(monkeypatch, get_log_messages):
    @modin.logging.enable_logging
    def inner_func():
        raise ValueError()

    @modin.logging.enable_logging
    def outer_func():
        inner_func()

    with monkeypatch.context() as ctx:
        # NOTE: we cannot patch in the fixture as mockin logger.getLogger()
        # without monkeypatch.context() breaks pytest
        mock_get_logger(ctx)

        with pytest.raises(ValueError):
            outer_func()

    assert get_log_messages("modin.logger.errors")["exception"] == [
        "STOP::PANDAS-API::inner_func"
    ]


def test_class_decorator(monkeypatch, get_log_messages):
    @modin.logging.enable_logging("CUSTOM")
    class Foo:
        def method1(self):
            pass

        @classmethod
        def method2(cls):
            pass

        @staticmethod
        def method3():
            pass

    class Bar(Foo):
        def method4(self):
            pass

    with monkeypatch.context() as ctx:
        mock_get_logger(ctx)
        Foo().method1()
        Foo.method2()
        Foo.method3()

        Bar().method1()
        Bar().method4()

    assert get_log_messages()[logging.INFO] == [
        "START::CUSTOM::Foo.method1",
        "STOP::CUSTOM::Foo.method1",
        "START::CUSTOM::Foo.method2",
        "STOP::CUSTOM::Foo.method2",
        "START::CUSTOM::Foo.method3",
        "STOP::CUSTOM::Foo.method3",
        "START::CUSTOM::Foo.method1",
        "STOP::CUSTOM::Foo.method1",
    ]


def test_class_inheritance(monkeypatch, get_log_messages):
    class Foo(modin.logging.ClassLogger, modin_layer="CUSTOM"):
        def method1(self):
            pass

    class Bar(Foo):
        def method2(self):
            pass

    with monkeypatch.context() as ctx:
        mock_get_logger(ctx)
        Foo().method1()
        Bar().method1()
        Bar().method2()

    assert get_log_messages()[logging.INFO] == [
        "START::CUSTOM::Foo.method1",
        "STOP::CUSTOM::Foo.method1",
        "START::CUSTOM::Foo.method1",
        "STOP::CUSTOM::Foo.method1",
        "START::CUSTOM::Bar.method2",
        "STOP::CUSTOM::Bar.method2",
    ]
