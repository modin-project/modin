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
import os


def pytest_addoption(parser):
    parser.addoption(
        "--simulate-cloud",
        action="store",
        default="off",
        help="simulate cloud for testing: off|normal|experimental",
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
    import os

    os.environ["MODIN_EXPERIMENTAL"] = "True" if mode == "experimental" else "False"


@pytest.fixture(scope="session", autouse=True)
def simulate_cloud(request):
    mode = request.config.getoption("--simulate-cloud").lower()
    if mode == "off":
        yield
        return

    if mode not in ("normal", "experimental"):
        raise ValueError(f"Unsupported --simulate-cloud mode: {mode}")
    assert (
        os.environ.get("MODIN_EXPERIMENTAL", "").title() == "True"
    ), "Simulated cloud must be started in experimental mode"

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
