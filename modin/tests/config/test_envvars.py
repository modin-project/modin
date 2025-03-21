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

import itertools
import os
import re
import sys
import unittest.mock as mock
from unittest.mock import Mock, patch

import pandas
import pytest
from pytest import param

import modin.config as cfg
import modin.pandas as pd
from modin.config.envvars import _check_vars
from modin.config.pubsub import _UNSET, ExactStr, ValueSource
from modin.pandas.base import BasePandasDataset
from modin.tests.pandas.utils import switch_execution

################# WARNING #####################################################
# Test cases in this file affect global state, e.g. by setting environment
# variables. The test cases may produce unexpected results when repeated on run
# out of the order they are defined in. Be careful when running the test
# locally or when adding new test cases. In particular, note:
#   - test_ray_cluster_resources() causes us to permanently attach the
#     `_initialize_engine` subscriber to Engine: https://github.com/modin-project/modin/blob/6252ebde19935bd1f6a6850209bf8a1f5e5ecfb7/modin/core/execution/dispatching/factories/dispatcher.py#L115
#     Changing to any engine after that test runs will cause Modin to try to
#     initialize the engine.
#   - In CI, we only run these tests with Ray execution, in the
#     `test-internal` job.
#   - test_wrong_values() permanently messes up some config variables. For more
#     details see https://github.com/modin-project/modin/issues/7454
################# WARNING ######################

UNIDIST_SKIP_REASON = (
    "Switching to unidist causes an error since we have to execute unidist "
    + "tests differently, with `mpiexec` instead of just `pytest`"
)


@pytest.fixture
def clear_backend_execution_and_storage_format(monkeypatch):
    """
    Reset environment variables and config classes for backend, execution, and storage format.

    Parameters
    ----------
    *vars : tuple[Parameter]
    """
    for variable in (cfg.Backend, cfg.StorageFormat, cfg.Engine):
        monkeypatch.setattr(variable, "_value", _UNSET)
        monkeypatch.setattr(variable, "_value_source", ValueSource.DEFAULT)
        monkeypatch.delitem(os.environ, variable.varname, raising=False)


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


@pytest.fixture(scope="session")
def add_pandas_duplicate_on_ray_execution():
    """
    Add an execution mode with the storage format Test_Pandasduplicate and engine Ray.

    This mode's execution is equivalent to PandasOnRay execution.
    """
    cfg.StorageFormat.add_option("Test_Pandasduplicate")
    from modin.core.execution.dispatching.factories import factories

    factories.Test_PandasduplicateOnRayFactory = factories.PandasOnRayFactory
    cfg.Backend.register_backend(
        "Test_Backend_1",
        cfg.Execution(
            storage_format="Test_Pandasduplicate",
            engine="Ray",
        ),
    )


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
        # which is the same method (modulo some wrapping that we add to handle
        # extensions) as Series.astype.
        assert (
            pd.Series.astype.__wrapped__.__wrapped__
            is BasePandasDataset.astype.__wrapped__
        )
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


class TestBackend:

    @pytest.mark.parametrize(
        "engine, storage_format, expected_backend",
        [
            ("Python", "Pandas", "Python_Test"),
            ("Ray", "Pandas", "Ray"),
            param(
                "Unidist",
                "Pandas",
                "Unidist",
                marks=pytest.mark.skip(reason=UNIDIST_SKIP_REASON),
            ),
            ("Dask", "Pandas", "Dask"),
            ("Native", "Native", "Pandas"),
        ],
    )
    def test_setting_execution_changes_backend(
        self, engine, storage_format, expected_backend
    ):
        previous_backend = cfg.Backend.get()
        with switch_execution(engine, storage_format):
            assert cfg.Backend.get() == expected_backend
        assert cfg.Backend.get() == previous_backend

    def test_subscribing_to_backend_triggers_callback(self):
        backend_subscriber = Mock()
        cfg.Backend.subscribe(backend_subscriber)
        backend_subscriber.assert_called_once_with(cfg.Backend)

    def test_setting_backend_triggers_all_callbacks(self):
        # Start with a known backend (rather than the one that we start the
        # test with).
        with cfg.context(Backend="Pandas"):
            backend_subscriber = Mock()
            cfg.Backend.subscribe(backend_subscriber)
            backend_subscriber.reset_mock()

            storage_format_subscriber = Mock()
            cfg.StorageFormat.subscribe(storage_format_subscriber)
            storage_format_subscriber.reset_mock()

            engine_subscriber = Mock()
            cfg.Engine.subscribe(engine_subscriber)
            engine_subscriber.reset_mock()

            with cfg.context(Backend="Python_Test"):
                backend_subscriber.assert_called_once_with(cfg.Backend)
                storage_format_subscriber.assert_called_once_with(cfg.StorageFormat)
                engine_subscriber.assert_called_once_with(cfg.Engine)

    @pytest.mark.parametrize(
        "backend, expected_engine, expected_storage_format",
        [
            ("Python_test", "Python", "Pandas"),
            ("PYTHON_test", "Python", "Pandas"),
            ("python_TEST", "Python", "Pandas"),
            ("Ray", "Ray", "Pandas"),
            param(
                "Unidist",
                "Unidist",
                "Pandas",
                marks=pytest.mark.skip(reason=UNIDIST_SKIP_REASON),
            ),
            ("Dask", "Dask", "Pandas"),
            ("Pandas", "Native", "Native"),
        ],
    )
    def test_setting_backend_changes_execution(
        self, backend, expected_engine, expected_storage_format
    ):
        previous_engine = cfg.Engine.get()
        previous_storage_format = cfg.StorageFormat.get()
        with cfg.context(Backend=backend):
            assert cfg.Engine.get() == expected_engine
            assert cfg.StorageFormat.get() == expected_storage_format
        assert cfg.Engine.get() == previous_engine
        assert cfg.StorageFormat.get() == previous_storage_format

    def test_setting_engine_alone_changes_backend(self):
        # Start with a known backend (rather than the one that we start the
        # test with).
        with switch_execution(storage_format="Pandas", engine="Ray"):
            current_backend = cfg.Backend.get()
            assert current_backend == "Ray"
            with cfg.context(Engine="Python"):
                assert cfg.Backend.get() == "Python_Test"
            assert cfg.Backend.get() == current_backend

    def test_setting_engine_triggers_callbacks(self):
        # Start with a known backend (rather than the one that we start the
        # test with).
        with switch_execution(storage_format="Pandas", engine="Ray"):
            engine_subscriber = Mock()
            cfg.Engine.subscribe(engine_subscriber)
            engine_subscriber.reset_mock()

            backend_subscriber = Mock()
            cfg.Backend.subscribe(backend_subscriber)
            backend_subscriber.reset_mock()

            storage_format_subscriber = Mock()
            cfg.StorageFormat.subscribe(storage_format_subscriber)
            storage_format_subscriber.reset_mock()

            with cfg.context(Engine="Dask"):
                engine_subscriber.assert_called_once_with(cfg.Engine)
                backend_subscriber.assert_called_once_with(cfg.Backend)
                # StorageFormat stayed the same, so we don't call its callback.
                storage_format_subscriber.assert_not_called()

    def test_setting_storage_format_triggers_callbacks(self):
        # There's only one built-in storage format, pandas, so we add a new one
        # here.
        cfg.StorageFormat.add_option("Pandasduplicate")
        from modin.core.execution.dispatching.factories import factories

        factories.PandasduplicateOnRayFactory = factories.PandasOnRayFactory
        cfg.Backend.register_backend(
            "NewBackend",
            cfg.Execution(
                storage_format="Pandasduplicate",
                engine="Ray",
            ),
        )

        with switch_execution(storage_format="Pandas", engine="Ray"):
            engine_subscriber = Mock()
            cfg.Engine.subscribe(engine_subscriber)
            engine_subscriber.reset_mock()
            backend_subscriber = Mock()
            cfg.Backend.subscribe(backend_subscriber)
            backend_subscriber.reset_mock()
            storage_format_subscriber = Mock()
            cfg.StorageFormat.subscribe(storage_format_subscriber)
            storage_format_subscriber.reset_mock()
            with cfg.context(StorageFormat="PANDASDUPLICATE"):
                storage_format_subscriber.assert_called_once_with(cfg.StorageFormat)
                backend_subscriber.assert_called_once_with(cfg.Backend)
                # Engine stayed the same, so we don't call its callback.
                engine_subscriber.assert_not_called()

    @pytest.mark.parametrize("name", ["Python_Test", "python_Test"])
    def test_register_existing_backend(self, name):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Backend 'Python_Test' is already registered with the execution "
                + "Execution(storage_format='Pandas', engine='Python')"
            ),
        ):
            cfg.Backend.register_backend(
                name,
                cfg.Execution(
                    storage_format="Pandas",
                    engine="Python",
                ),
            )

    def test_register_existing_execution(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Execution(storage_format='Pandas', engine='Python') is already registered with the backend Python_Test."
            ),
        ):
            cfg.Backend.register_backend(
                "NewBackend2",
                cfg.Execution(
                    storage_format="Pandas",
                    engine="Python",
                ),
            )

    def test_set_invalid_backend(self):
        with pytest.raises(ValueError, match=re.escape("Unknown backend 'Unknown'")):
            cfg.Backend.put("Unknown")

    def test_switch_to_unregistered_backend_with_switch_execution(self):
        cfg.StorageFormat.add_option("Pandas2")
        from modin.core.execution.dispatching.factories import factories

        factories.Pandas2OnRayFactory = factories.PandasOnRayFactory
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Execution(storage_format='Pandas2', engine='Ray') "
                + "has no known backend. Please register a backend for it with "
                + "Backend.register_backend()"
            ),
        ), switch_execution(engine="Ray", storage_format="Pandas2"):
            pass

    def test_switch_to_unregistered_backend_with_switch_storage_format(self):
        cfg.StorageFormat.add_option("Pandas3")
        from modin.core.execution.dispatching.factories import factories

        factories.Pandas2OnRayFactory = factories.PandasOnPythonFactory
        with cfg.context(StorageFormat="Pandas", Engine="Python"):
            with pytest.raises(
                ValueError,
                match=re.escape(
                    "Execution(storage_format='Pandas3', engine='Python') "
                    + "has no known backend. Please register a backend for it with "
                    + "Backend.register_backend()"
                ),
            ):
                cfg.StorageFormat.put("Pandas3")

    def test_switch_to_unregistered_backend_with_switch_engine(self):
        cfg.Engine.add_option("Python2")
        from modin.core.execution.dispatching.factories import factories

        factories.PandasOnPython2Factory = factories.PandasOnPythonFactory
        with cfg.context(StorageFormat="Pandas", Engine="Python"):
            with pytest.raises(
                ValueError,
                match=re.escape(
                    "Execution(storage_format='Pandas', engine='Python2') "
                    + "has no known backend. Please register a backend for it with "
                    + "Backend.register_backend()"
                ),
            ):
                cfg.Engine.put("Python2")

    # The default engine and storage format, and hence the default backend,
    # will depend on which engines are available in the current environment.
    # For simplicity, patch the defaults.
    @patch(
        target="modin.config.StorageFormat._get_default",
    )
    @patch(
        target="modin.config.Engine._get_default",
    )
    def test_backend_default(
        self,
        mocked_get_default,
        mocked_get_default2,
    ):
        mocked_get_default.return_value = "Native"
        mocked_get_default2.return_value = "Native"
        assert cfg.Backend._get_default() == "Pandas"

    def test_add_backend_option(self):
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Cannot add an option to Backend directly. Use Backend.register_backend instead."
            ),
        ):
            cfg.Backend.add_option("NewBackend")

    @pytest.mark.parametrize(
        "order_to_get_in",
        itertools.permutations(
            [
                cfg.Backend,
                cfg.Engine,
                cfg.StorageFormat,
            ]
        ),
        ids=lambda permutation: "_".join(x.__name__ for x in permutation),
    )
    @pytest.mark.parametrize(
        "storage_environment_variable, engine_environment_variable, variable_to_expected_value",
        [
            (
                "Native",
                "Native",
                {
                    cfg.Backend: "Pandas",
                    cfg.Engine: "Native",
                    cfg.StorageFormat: "Native",
                },
            ),
            (
                "NATIVE",
                "NATIVE",
                {
                    cfg.Backend: "Pandas",
                    cfg.Engine: "Native",
                    cfg.StorageFormat: "Native",
                },
            ),
            (
                "Pandas",
                "Dask",
                {
                    cfg.Backend: "Dask",
                    cfg.Engine: "Dask",
                    cfg.StorageFormat: "Pandas",
                },
            ),
        ],
    )
    def test_storage_format_and_engine_come_from_environment(
        self,
        monkeypatch,
        clear_backend_execution_and_storage_format,
        order_to_get_in,
        storage_environment_variable,
        engine_environment_variable,
        variable_to_expected_value,
    ):
        with mock.patch.dict(
            os.environ,
            {
                cfg.StorageFormat.varname: storage_environment_variable,
                cfg.Engine.varname: engine_environment_variable,
            },
        ):
            for variable in order_to_get_in:
                expected_value = variable_to_expected_value[variable]
                assert (
                    variable.get() == expected_value
                ), f"{variable.__name__} was {variable.get()} instead of {expected_value}"

    @pytest.mark.parametrize(
        "order_to_get_in",
        itertools.permutations(
            [
                cfg.Backend,
                cfg.Engine,
                cfg.StorageFormat,
            ]
        ),
        ids=lambda permutation: "_".join(x.__name__ for x in permutation),
    )
    @pytest.mark.parametrize(
        "engine_environment_variable, variable_to_expected_value",
        [
            (
                "Dask",
                {cfg.Backend: "Dask", cfg.StorageFormat: "Pandas", cfg.Engine: "Dask"},
            ),
            (
                "DASK",
                {cfg.Backend: "Dask", cfg.StorageFormat: "Pandas", cfg.Engine: "Dask"},
            ),
            (
                "python",
                {
                    cfg.Backend: "Python_Test",
                    cfg.StorageFormat: "Pandas",
                    cfg.Engine: "Python",
                },
            ),
            (
                "ray",
                {cfg.Backend: "Ray", cfg.StorageFormat: "Pandas", cfg.Engine: "Ray"},
            ),
            # note that we can't test Native here because it's not valid to use
            # "Native" engine with the default storage format of "Pandas."
        ],
    )
    def test_only_engine_comes_from_environment(
        self,
        clear_backend_execution_and_storage_format,
        order_to_get_in,
        engine_environment_variable,
        variable_to_expected_value,
    ):
        with mock.patch.dict(
            os.environ,
            {cfg.Engine.varname: engine_environment_variable},
        ):
            for var in order_to_get_in:
                expected_value = variable_to_expected_value[var]
                assert (
                    var.get() == expected_value
                ), f"{var.__name__} was {var.get()} instead of {expected_value}"

    @pytest.mark.parametrize(
        "order_to_get_in",
        itertools.permutations(
            [
                cfg.Backend,
                cfg.Engine,
                cfg.StorageFormat,
            ]
        ),
        ids=lambda permutation: "_".join(x.__name__ for x in permutation),
    )
    def test_only_storage_format_comes_from_environment(
        self,
        clear_backend_execution_and_storage_format,
        order_to_get_in,
        add_pandas_duplicate_on_ray_execution,
    ):
        # To test switching StorageFormat alone, we have to add a new backend
        # that works with the default "Pandas" execution.
        with mock.patch.dict(
            os.environ,
            {
                cfg.StorageFormat.varname: "Test_Pandasduplicate",
            },
        ):
            cfg.Engine.put("Ray")
            for variable in order_to_get_in:
                expected_value = {
                    cfg.Backend: "Test_Backend_1",
                    cfg.Engine: "Ray",
                    cfg.StorageFormat: "Test_Pandasduplicate",
                }[variable]
                assert (
                    variable.get() == expected_value
                ), f"{variable.__name__} was {variable.get()} instead of {expected_value}"

    @pytest.mark.parametrize(
        "order_to_get_in",
        itertools.permutations(
            [
                cfg.Backend,
                cfg.Engine,
                cfg.StorageFormat,
            ]
        ),
        ids=lambda permutation: "_".join(x.__name__ for x in permutation),
    )
    @pytest.mark.parametrize(
        "backend_environment_variable, variable_to_expected_value",
        [
            (
                "Pandas",
                {
                    cfg.Backend: "Pandas",
                    cfg.Engine: "Native",
                    cfg.StorageFormat: "Native",
                },
            ),
            (
                "Ray",
                {cfg.Backend: "Ray", cfg.Engine: "Ray", cfg.StorageFormat: "Pandas"},
            ),
            (
                "Dask",
                {cfg.Backend: "Dask", cfg.Engine: "Dask", cfg.StorageFormat: "Pandas"},
            ),
            (
                "python_test",
                {
                    cfg.Backend: "Python_Test",
                    cfg.Engine: "Python",
                    cfg.StorageFormat: "Pandas",
                },
            ),
        ],
    )
    def test_backend_comes_from_environment(
        self,
        monkeypatch,
        clear_backend_execution_and_storage_format,
        order_to_get_in,
        backend_environment_variable,
        variable_to_expected_value,
    ):
        with mock.patch.dict(
            os.environ,
            {
                cfg.Backend.varname: backend_environment_variable,
            },
        ):
            for variable in order_to_get_in:
                expected_value = variable_to_expected_value[variable]
                assert (
                    variable.get() == expected_value
                ), f"{variable.__name__} was {variable.get()} instead of {expected_value}"

    @pytest.mark.parametrize(
        "order_to_get_in",
        itertools.permutations(
            [cfg.Backend, cfg.Engine, cfg.StorageFormat],
        ),
        ids=lambda permutation: "_".join(x.__name__ for x in permutation),
    )
    def test_environment_not_set_and_pick_up_default_engine(
        self, clear_backend_execution_and_storage_format, order_to_get_in
    ):
        for variable in order_to_get_in:
            assert variable.get() == variable._get_default()

    @pytest.mark.parametrize(
        "execution_variable, value",
        [(cfg.Engine, "Python"), (cfg.StorageFormat, "Pandas")],
    )
    @pytest.mark.parametrize(
        "variable_to_get",
        [cfg.Backend, cfg.Engine, cfg.StorageFormat],
    )
    def test_conflicting_execution_and_backend_in_environment(
        self,
        monkeypatch,
        clear_backend_execution_and_storage_format,
        execution_variable,
        value,
        variable_to_get,
    ):
        monkeypatch.setitem(os.environ, cfg.Backend.varname, "Ray")
        monkeypatch.setitem(os.environ, execution_variable.varname, value)
        with pytest.raises(
            ValueError,
            match=re.escape("Can't specify both execution and backend in environment"),
        ):
            variable_to_get.get()

    def test_get_execution_for_unknown_backend(self):
        backend_choice_string = ", ".join(
            f"'{choice}'" for choice in cfg.Backend.choices
        )
        with pytest.raises(
            ValueError,
            match=re.escape(
                f"Unknown backend 'Unknown'. Available backends are: {backend_choice_string}"
            ),
        ):
            cfg.Backend.get_execution_for_backend("Unknown")


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
