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
import pytest
import modin.config as cfg
from modin.config.envvars import EnvironmentVariable, _check_vars, ExactStr

from packaging import version


@pytest.fixture
def make_unknown_env():
    varname = "MODIN_UNKNOWN"
    os.environ[varname] = "foo"
    yield varname
    del os.environ[varname]


@pytest.fixture(params=[str, ExactStr])
def make_custom_envvar(request):
    class CustomVar(EnvironmentVariable, type=request.param):
        """custom var"""

        default = 10
        varname = "MODIN_CUSTOM"
        choices = (1, 5, 10)

    return CustomVar


@pytest.fixture
def set_custom_envvar(make_custom_envvar):
    os.environ[make_custom_envvar.varname] = "  custom  "
    yield "Custom" if make_custom_envvar.type is str else "  custom  "
    del os.environ[make_custom_envvar.varname]


def test_unknown(make_unknown_env):
    with pytest.warns(UserWarning, match=f"Found unknown .*{make_unknown_env}.*"):
        _check_vars()


def test_custom_default(make_custom_envvar):
    assert make_custom_envvar.get() == 10


def test_custom_set(make_custom_envvar, set_custom_envvar):
    assert make_custom_envvar.get() == set_custom_envvar


def test_custom_help(make_custom_envvar):
    assert "MODIN_CUSTOM" in make_custom_envvar.get_help()
    assert "custom var" in make_custom_envvar.get_help()


def test_hdk_envvar():
    try:
        import pyhdk

        defaults = cfg.HdkLaunchParameters.get()
        assert defaults["enable_union"] == 1
        if version.parse(pyhdk.__version__) >= version.parse("0.6.1"):
            assert defaults["log_dir"] == "pyhdk_log"
        del cfg.HdkLaunchParameters._value
    except ImportError:
        # This test is intended to check pyhdk internals. If pyhdk is not available, skip the version check test.
        pass

    os.environ[
        cfg.OmnisciLaunchParameters.varname
    ] = "enable_union=2,enable_thrift_logs=3"
    del cfg.OmnisciLaunchParameters._value
    params = cfg.OmnisciLaunchParameters.get()
    assert params["enable_union"] == 2
    assert params["enable_thrift_logs"] == 3

    params = cfg.HdkLaunchParameters.get()
    assert params["enable_union"] == 2
    assert params["enable_thrift_logs"] == 3

    os.environ[cfg.HdkLaunchParameters.varname] = "unsupported=X"
    params = cfg.HdkLaunchParameters.get()
    assert params["unsupported"] == "X"
    try:
        import pyhdk

        pyhdk.buildConfig(**cfg.HdkLaunchParameters.get())
    except RuntimeError as e:
        assert str(e) == "unrecognised option '--unsupported'"
    except ImportError:
        # This test is intended to check pyhdk internals. If pyhdk is not available, skip the version check test.
        pass

    os.environ[
        cfg.HdkLaunchParameters.varname
    ] = "enable_union=4,enable_thrift_logs=5,enable_lazy_dict_materialization=6"
    del cfg.HdkLaunchParameters._value
    params = cfg.HdkLaunchParameters.get()
    assert params["enable_union"] == 4
    assert params["enable_thrift_logs"] == 5
    assert params["enable_lazy_dict_materialization"] == 6

    params = cfg.OmnisciLaunchParameters.get()
    assert params["enable_union"] == 2
    assert params["enable_thrift_logs"] == 3

    del os.environ[cfg.OmnisciLaunchParameters.varname]
    del cfg.OmnisciLaunchParameters._value
    params = cfg.OmnisciLaunchParameters.get()
    assert params["enable_union"] == 4
    assert params["enable_thrift_logs"] == 5
