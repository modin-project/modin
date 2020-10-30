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

import modin

from modin.backends import PandasQueryCompiler, BaseQueryCompiler
from modin.engines.python.pandas_on_python.io import PandasOnPythonIO
from modin.data_management.factories import factories
from modin.utils import get_current_backend

import pytest

BASE_BACKEND_NAME = "BaseOnPython"


class TestQC(BaseQueryCompiler):
    def __init__(self, modin_frame):
        self._modin_frame = modin_frame

    @classmethod
    def from_pandas(cls, df, data_cls):
        return cls(data_cls.from_pandas(df))

    @classmethod
    def from_arrow(cls, at, data_cls):
        return cls(data_cls.from_arrow(at))

    def free(self):
        pass

    to_pandas = PandasQueryCompiler.to_pandas
    default_to_pandas = PandasQueryCompiler.default_to_pandas


class BaseOnPythonIO(PandasOnPythonIO):
    query_compiler_cls = TestQC


class BaseOnPythonFactory(factories.BaseFactory):
    @classmethod
    def prepare(cls):
        cls.io_cls = BaseOnPythonIO


def set_base_backend(name=BASE_BACKEND_NAME):
    setattr(factories, f"{name}Factory", BaseOnPythonFactory)
    modin.set_backends(engine="python", partition=name.split("On")[0])


def pytest_addoption(parser):
    parser.addoption("--backend", action="store", default=None)


def pytest_configure(config):
    backend = config.option.backend

    if backend is None:
        return

    if backend == BASE_BACKEND_NAME:
        set_base_backend(BASE_BACKEND_NAME)
    else:
        partition, engine = backend.split("On")
        modin.set_backends(engine=engine, partition=partition)


def pytest_runtest_call(item):
    custom_markers = ["xfail", "skip"]

    # dynamicly adding custom markers to tests
    for custom_marker in custom_markers:
        for marker in item.iter_markers(name=f"{custom_marker}_backends"):
            backends = marker.args[0]
            if not isinstance(backends, list):
                backends = [backends]

            current_backend = get_current_backend()
            reason = marker.kwargs.pop("reason", "")

            item.add_marker(
                getattr(pytest.mark, custom_marker)(
                    condition=current_backend in backends,
                    reason=f"Backend {current_backend} does not pass this test. {reason}",
                    **marker.kwargs,
                )
            )
