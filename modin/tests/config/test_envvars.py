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

import pandas
import pytest

import modin.config as cfg
import modin.pandas as pd
from modin.config.envvars import _check_vars
from modin.config.pubsub import _UNSET, ExactStr
from modin.pandas.base import BasePandasDataset


def reset_vars(*vars: tuple[cfg.Parameter]):
    """
    Reset value for the passed parameters.

    Parameters
    ----------
    *vars : tuple[Parameter]
    """
    for var in vars:
        var._value = _UNSET
        _ = os.environ.pop(var.varname, None)


@pytest.fixture
def make_unknown_env():
    varname = "MODIN_UNKNOWN"
    os.environ[varname] = "foo"
    yield varname
    del os.environ[varname]


@pytest.fixture(params=[str, ExactStr])
def make_custom_envvar(request):
    class CustomVar(cfg.EnvironmentVariable, type=request.param):
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


class TestDocModule:
    """
    Test using a module to replace default docstrings.
    """

    def test_overrides(self):
        cfg.DocModule.put("modin.tests.config.docs_module")

        # Test for override
        assert BasePandasDataset.__doc__ == (
            "This is a test of the documentation module for BasePandasDataSet."
        )
        assert BasePandasDataset.apply.__doc__ == (
            "This is a test of the documentation module for BasePandasDataSet.apply."
        )
        # Test scenario 2 from https://github.com/modin-project/modin/issues/7113:
        # We can correctly override the docstring for BasePandasDataset.astype,
        # which is the same method as Series.astype.
        assert pd.Series.astype is BasePandasDataset.astype
        assert BasePandasDataset.astype.__doc__ == (
            "This is a test of the documentation module for BasePandasDataSet.astype."
        )
        assert (
            pd.DataFrame.apply.__doc__
            == "This is a test of the documentation module for DataFrame."
        )
        # Test for pandas doc when method is not defined on the plugin module
        assert pandas.DataFrame.isna.__doc__ in pd.DataFrame.isna.__doc__
        assert pandas.DataFrame.isnull.__doc__ in pd.DataFrame.isnull.__doc__
        assert BasePandasDataset.astype.__doc__ in pd.DataFrame.astype.__doc__
        # Test for override
        assert (
            pd.Series.isna.__doc__
            == "This is a test of the documentation module for Series."
        )
        # Test for pandas doc when method is not defined on the plugin module
        assert pandas.Series.isnull.__doc__ in pd.Series.isnull.__doc__
        assert pandas.Series.apply.__doc__ in pd.Series.apply.__doc__
        # Test for override
        assert pd.read_csv.__doc__ == "Test override for functions on the module."
        # Test for pandas doc when function is not defined on module.
        assert pandas.read_table.__doc__ in pd.read_table.__doc__

    def test_not_redefining_classes_modin_issue_7138(self):
        original_dataframe_class = pd.DataFrame

        cfg.DocModule.put("modin.tests.config.docs_module")

        # Test for override
        assert (
            pd.DataFrame.apply.__doc__
            == "This is a test of the documentation module for DataFrame."
        )

        assert pd.DataFrame is original_dataframe_class

    def test_base_docstring_override_with_no_dataframe_or_series_class_issue_7113(
        self,
    ):
        # This test case tests scenario 1 from issue 7113.
        sys.path.append(f"{os.path.dirname(__file__)}")
        cfg.DocModule.put("docs_module_with_just_base")
        assert BasePandasDataset.astype.__doc__ == (
            "This is a test of the documentation module for BasePandasDataSet.astype."
        )


@pytest.mark.skipif(cfg.Engine.get() != "Ray", reason="Ray specific test")
def test_ray_cluster_resources():
    import ray

    cfg.RayInitCustomResources.put({"special_hardware": 1.0})
    # create a dummy df to initialize Ray engine
    _ = pd.DataFrame([1, 2, 3])
    assert ray.cluster_resources()["special_hardware"] == 1.0


@pytest.mark.parametrize(
    "modify_config",
    [{cfg.RangePartitioning: False, cfg.LazyExecution: "Auto"}],
    indirect=True,
)
def test_context_manager_update_config(modify_config):
    # simple case, 1 parameter
    assert cfg.RangePartitioning.get() is False
    with cfg.context(RangePartitioning=True):
        assert cfg.RangePartitioning.get() is True
    assert cfg.RangePartitioning.get() is False

    # nested case, 1 parameter
    assert cfg.RangePartitioning.get() is False
    with cfg.context(RangePartitioning=True):
        assert cfg.RangePartitioning.get() is True
        with cfg.context(RangePartitioning=False):
            assert cfg.RangePartitioning.get() is False
            with cfg.context(RangePartitioning=False):
                assert cfg.RangePartitioning.get() is False
            assert cfg.RangePartitioning.get() is False
        assert cfg.RangePartitioning.get() is True
    assert cfg.RangePartitioning.get() is False

    # simple case, 2 parameters
    assert cfg.RangePartitioning.get() is False
    assert cfg.LazyExecution.get() == "Auto"
    with cfg.context(RangePartitioning=True, LazyExecution="Off"):
        assert cfg.RangePartitioning.get() is True
        assert cfg.LazyExecution.get() == "Off"
    assert cfg.RangePartitioning.get() is False
    assert cfg.LazyExecution.get() == "Auto"

    # nested case, 2 parameters
    assert cfg.RangePartitioning.get() is False
    assert cfg.LazyExecution.get() == "Auto"
    with cfg.context(RangePartitioning=True, LazyExecution="Off"):
        assert cfg.RangePartitioning.get() is True
        assert cfg.LazyExecution.get() == "Off"
        with cfg.context(RangePartitioning=False):
            assert cfg.RangePartitioning.get() is False
            assert cfg.LazyExecution.get() == "Off"
            with cfg.context(LazyExecution="On"):
                assert cfg.RangePartitioning.get() is False
                assert cfg.LazyExecution.get() == "On"
                with cfg.context(RangePartitioning=True, LazyExecution="Off"):
                    assert cfg.RangePartitioning.get() is True
                    assert cfg.LazyExecution.get() == "Off"
                assert cfg.RangePartitioning.get() is False
                assert cfg.LazyExecution.get() == "On"
            assert cfg.RangePartitioning.get() is False
            assert cfg.LazyExecution.get() == "Off"
        assert cfg.RangePartitioning.get() is True
        assert cfg.LazyExecution.get() == "Off"
    assert cfg.RangePartitioning.get() is False
    assert cfg.LazyExecution.get() == "Auto"


@pytest.mark.parametrize(
    "config_name",
    [
        "NPartitions",
        "CpuCount",
        "LogMemoryInterval",
        "LogFileSize",
        "MinRowPartitionSize",
        "MinColumnPartitionSize",
    ],
)
def test_wrong_values(config_name):
    config: cfg.EnvironmentVariable = getattr(cfg, config_name)
    new_value = -1
    with pytest.raises(ValueError):
        with cfg.context(**{config_name: new_value}):
            _ = config.get()
