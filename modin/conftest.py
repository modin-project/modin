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

import os
import sys
import pytest

import modin
import modin.config
from modin.config import IsExperimental

from modin.backends import PandasQueryCompiler, BaseQueryCompiler
from modin.engines.python.pandas_on_python.io import PandasOnPythonIO
from modin.data_management.factories import factories
from modin.utils import get_current_backend


def pytest_addoption(parser):
    parser.addoption(
        "--simulate-cloud",
        action="store",
        default="off",
        help="simulate cloud for testing: off|normal|experimental",
    )
    parser.addoption(
        "--backend",
        action="store",
        default=None,
        help="specifies backend to run tests on",
    )


class Patcher:
    def __init__(self, conn, *pairs):
        self.pairs = pairs
        self.originals = None
        self.conn = conn

    def __wrap(self, func):
        def wrapper(*a, **kw):
            return func(
                *(tuple(self.conn.obtain(x) for x in a)),
                **({k: self.conn.obtain(v) for k, v in kw.items()}),
            )

        return func, wrapper

    def __enter__(self):
        self.originals = []
        for module, attrname in self.pairs:
            orig, wrapped = self.__wrap(getattr(module, attrname))
            self.originals.append((module, attrname, orig))
            setattr(module, attrname, wrapped)
        return self

    def __exit__(self, *a, **kw):
        for module, attrname, orig in self.originals:
            setattr(module, attrname, orig)


def set_experimental_env(mode):
    from modin.config import IsExperimental

    IsExperimental.put(mode == "experimental")


@pytest.fixture(scope="session", autouse=True)
def simulate_cloud(request):
    mode = request.config.getoption("--simulate-cloud").lower()
    if mode == "off":
        yield
        return

    if mode not in ("normal", "experimental"):
        raise ValueError(f"Unsupported --simulate-cloud mode: {mode}")
    assert IsExperimental.get(), "Simulated cloud must be started in experimental mode"

    from modin.experimental.cloud import create_cluster, get_connection
    import pandas._testing
    import pandas._libs.testing as cyx_testing

    with create_cluster("local", cluster_type="local"):
        get_connection().teleport(set_experimental_env)(mode)
        with Patcher(
            get_connection(),
            (pandas._testing, "assert_class_equal"),
            (cyx_testing, "assert_almost_equal"),
        ):
            yield


@pytest.fixture(scope="session", autouse=True)
def enforce_config():
    """
    A fixture that ensures that all checks for MODIN_* variables
    are done using modin.config to prevent leakage
    """
    orig_env = os.environ
    modin_start = os.path.dirname(modin.__file__)
    modin_exclude = [os.path.dirname(modin.config.__file__)]

    class PatchedEnv:
        @staticmethod
        def __check_var(name):
            if name.upper().startswith("MODIN_"):
                frame = sys._getframe()
                try:
                    # get the path to module where caller of caller is defined;
                    # caller of this function is inside PatchedEnv, and we're
                    # interested in whomever called a method on PatchedEnv
                    caller_file = frame.f_back.f_back.f_code.co_filename
                finally:
                    del frame
                pkg_name = os.path.dirname(caller_file)
                if pkg_name.startswith(modin_start):
                    assert any(
                        pkg_name.startswith(excl) for excl in modin_exclude
                    ), "Do not access MODIN_ environment variable bypassing modin.config"

        def __getitem__(self, name):
            self.__check_var(name)
            return orig_env[name]

        def __setitem__(self, name, value):
            self.__check_var(name)
            orig_env[name] = value

        def __delitem__(self, name):
            self.__check_var(name)
            del orig_env[name]

        def pop(self, name):
            self.__check_var(name)
            return orig_env.pop(name)

        def get(self, name, defvalue=None):
            self.__check_var(name)
            return orig_env.get(name, defvalue)

        def __contains__(self, name):
            self.__check_var(name)
            return name in orig_env

        def __getattr__(self, name):
            return getattr(orig_env, name)

        def __iter__(self):
            return iter(orig_env)

    os.environ = PatchedEnv()
    yield
    os.environ = orig_env


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
