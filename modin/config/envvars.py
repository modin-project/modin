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

"""Module houses Modin configs originated from environment variables."""

import os
import secrets
import sys
import warnings
from collections import namedtuple
from textwrap import dedent
from typing import Any, NoReturn, Optional

from packaging import version
from pandas.util._decorators import doc  # type: ignore[attr-defined]

from modin import set_execution
from modin.config.pubsub import (
    _TYPE_PARAMS,
    _UNSET,
    DeprecationDescriptor,
    ExactStr,
    Parameter,
    ValueSource,
)


class EnvironmentVariable(Parameter, type=str, abstract=True):
    """Base class for environment variables-based configuration."""

    varname: Optional[str] = None

    @classmethod
    def _get_value_from_config(cls) -> Any:
        """
        Read the value from environment variable.

        Returns
        -------
        Any
            Config raw value if it's set, otherwise `_UNSET`.
        """
        if cls.varname is None:
            raise TypeError("varname should not be None")
        if cls.varname not in os.environ:
            return _UNSET
        raw = os.environ[cls.varname]
        if not _TYPE_PARAMS[cls.type].verify(raw):
            # TODO: use and test a better error message, like "Invalid value
            # for {cls.varname}: {raw}"
            raise ValueError(f"Unsupported raw value: {raw}")
        return _TYPE_PARAMS[cls.type].decode(raw)

    @classmethod
    def get_help(cls) -> str:
        """
        Generate user-presentable help for the config.

        Returns
        -------
        str
        """
        help = f"{cls.varname}: {dedent(cls.__doc__ or 'Unknown').strip()}\n\tProvide {_TYPE_PARAMS[cls.type].help}"
        if cls.choices:
            help += f" (valid examples are: {', '.join(str(c) for c in cls.choices)})"
        return help


class EnvWithSibilings(
    EnvironmentVariable,
    # 'type' is a mandatory parameter for '__init_subclasses__', so we have to pass something here,
    # this doesn't force child classes to have 'str' type though, they actually can be any type
    type=str,
):
    """Ensure values synchronization between sibling parameters."""

    _update_sibling = True

    @classmethod
    def _sibling(cls) -> type["EnvWithSibilings"]:
        """Return a sibling parameter."""
        raise NotImplementedError()

    @classmethod
    def get(cls) -> Any:
        """
        Get parameter's value and ensure that it's equal to the sibling's value.

        Returns
        -------
        Any
        """
        sibling = cls._sibling()

        if sibling._value is _UNSET and cls._value is _UNSET:
            super().get()
            with warnings.catch_warnings():
                # filter warnings that can potentially come from the potentially deprecated sibling
                warnings.filterwarnings("ignore", category=FutureWarning)
                super(EnvWithSibilings, sibling).get()

            if (
                cls._value_source
                == sibling._value_source
                == ValueSource.GOT_FROM_CFG_SOURCE
            ):
                raise ValueError(
                    f"Configuration is ambiguous. You cannot set '{cls.varname}' and '{sibling.varname}' at the same time."
                )

            # further we assume that there are only two valid sources for the variables: 'GOT_FROM_CFG' and 'DEFAULT',
            # as otherwise we wouldn't ended-up in this branch at all, because all other ways of setting a value
            # changes the '._value' attribute from '_UNSET' to something meaningful
            from modin.error_message import ErrorMessage

            if cls._value_source == ValueSource.GOT_FROM_CFG_SOURCE:
                ErrorMessage.catch_bugs_and_request_email(
                    failure_condition=sibling._value_source != ValueSource.DEFAULT
                )
                sibling._value = cls._value
                sibling._value_source = ValueSource.GOT_FROM_CFG_SOURCE
            elif sibling._value_source == ValueSource.GOT_FROM_CFG_SOURCE:
                ErrorMessage.catch_bugs_and_request_email(
                    failure_condition=cls._value_source != ValueSource.DEFAULT
                )
                cls._value = sibling._value
                cls._value_source = ValueSource.GOT_FROM_CFG_SOURCE
            else:
                ErrorMessage.catch_bugs_and_request_email(
                    failure_condition=cls._value_source != ValueSource.DEFAULT
                    or sibling._value_source != ValueSource.DEFAULT
                )
                # propagating 'cls' default value to the sibling
                sibling._value = cls._value
        return super().get()

    @classmethod
    def put(cls, value: Any) -> None:
        """
        Set a new value to this parameter as well as to its sibling.

        Parameters
        ----------
        value : Any
        """
        super().put(value)
        # avoid getting into an infinite recursion
        if cls._update_sibling:
            cls._update_sibling = False
            try:
                with warnings.catch_warnings():
                    # filter potential future warnings of the sibling
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    cls._sibling().put(value)
            finally:
                cls._update_sibling = True


class EnvironmentVariableDisallowingExecutionAndBackendBothSet(
    EnvironmentVariable,
    type=EnvironmentVariable.type,
    abstract=True,
):
    """Subclass to disallow getting this variable from the environment when both execution and backend are set in the environment."""

    @classmethod
    @doc(EnvironmentVariable._get_value_from_config.__doc__)
    def _get_value_from_config(cls) -> str:
        if Backend.varname in os.environ and (
            Engine.varname in os.environ or StorageFormat.varname in os.environ
        ):
            # Handling this case is tricky, in part because the combination of
            # Backend and Engine/StorageFormat may be invalid. For now just
            # disallow it.
            raise ValueError("Can't specify both execution and backend in environment")
        return super()._get_value_from_config()


class IsDebug(EnvironmentVariable, type=bool):
    """Force Modin engine to be "Python" unless specified by $MODIN_ENGINE."""

    varname = "MODIN_DEBUG"


class Engine(
    EnvironmentVariableDisallowingExecutionAndBackendBothSet,
    type=str,
):
    """Distribution engine to run queries by."""

    varname = "MODIN_ENGINE"
    choices = ("Ray", "Dask", "Python", "Unidist", "Native")

    NOINIT_ENGINES = {
        "Python",
        "Native",
    }  # engines that don't require initialization, useful for unit tests

    has_custom_engine = False

    @classmethod
    def _get_default(cls) -> str:
        """
        Get default value of the config.

        Returns
        -------
        str
        """
        from modin.utils import MIN_DASK_VERSION, MIN_RAY_VERSION, MIN_UNIDIST_VERSION

        # If there's a custom engine, we don't need to check for any engine
        # dependencies. Return the default "Python" engine.
        if IsDebug.get() or cls.has_custom_engine:
            return "Python"
        try:
            import ray

        except ImportError:
            pass
        else:
            if version.parse(ray.__version__) < MIN_RAY_VERSION:
                raise ImportError(
                    'Please `pip install "modin[ray]"` to install compatible Ray '
                    + "version "
                    + f"(>={MIN_RAY_VERSION})."
                )
            return "Ray"
        try:
            import dask
            import distributed

        except ImportError:
            pass
        else:
            if (
                version.parse(dask.__version__) < MIN_DASK_VERSION
                or version.parse(distributed.__version__) < MIN_DASK_VERSION
            ):
                raise ImportError(
                    f'Please `pip install "modin[dask]"` to install compatible Dask version (>={MIN_DASK_VERSION}).'
                )
            return "Dask"
        try:
            import unidist

        except ImportError:
            pass
        else:
            if version.parse(unidist.__version__) < MIN_UNIDIST_VERSION:
                raise ImportError(
                    'Please `pip install "unidist[mpi]"` to install compatible unidist on MPI '
                    + "version "
                    + f"(>={MIN_UNIDIST_VERSION})."
                )
            return "Unidist"
        raise ImportError(
            "Please refer to installation documentation page to install an engine"
        )

    @classmethod
    @doc(Parameter.add_option.__doc__)
    def add_option(cls, choice: Any) -> Any:
        choice = super().add_option(choice)
        cls.NOINIT_ENGINES.add(choice)
        cls.has_custom_engine = True
        return choice

    @classmethod
    def put(cls, value: str) -> None:
        """
        Set the engine value.

        Parameters
        ----------
        value : str
            Engine value to set.
        """
        value = cls.normalize(value)
        # Backend.put() will set Engine.
        Backend.put(
            Backend.get_backend_for_execution(
                Execution(engine=value, storage_format=StorageFormat.get())
            )
        )

    @classmethod
    def get(cls) -> str:
        """
        Get the engine value.

        Returns
        -------
        str
            Engine value.
        """
        # We have to override get() because Engine may need to get its value
        # from the OS's environment variables for Backend or Engine.

        cls._warn_if_deprecated()

        # First, check if we've already set the engine value.
        if cls._value is not _UNSET:
            return cls._value

        engine_config_value = cls._get_value_from_config()
        backend_config_value = Backend._get_value_from_config()

        # If Engine is in the OS's configuration, use the configured Engine value.
        # Otherwise, use the Backend config value if that exists. If it doesn't,
        # fall back to the default Engine value.
        cls._value = (
            engine_config_value
            if engine_config_value is not _UNSET
            else (
                Backend.get_execution_for_backend(backend_config_value).engine
                if backend_config_value is not _UNSET
                else cls._get_default()
            )
        )

        return cls._value


class StorageFormat(EnvironmentVariableDisallowingExecutionAndBackendBothSet, type=str):
    """Engine to run on a single node of distribution."""

    @classmethod
    def put(cls, value: str) -> None:
        """
        Set the storage format value.

        Parameters
        ----------
        value : str
            Storage format value to set.
        """
        value = cls.normalize(value)
        # Backend.put() will set StorageFormat.
        Backend.put(
            Backend.get_backend_for_execution(
                Execution(engine=Engine.get(), storage_format=value)
            )
        )

    @classmethod
    def get(cls) -> str:
        """
        Get the storage format value.

        Returns
        -------
        str
            Storage format value.
        """
        # We have to override get() because StorageFormat may need to get its
        # value from the OS's environment variables for Backend or StorageFormat.

        cls._warn_if_deprecated()

        # First, check if we've already set the engine value.
        if cls._value is not _UNSET:
            return cls._value

        storage_format_config_value = cls._get_value_from_config()
        backend_config_value = Backend._get_value_from_config()

        # If StorageFormat is in the OS's configuration, use the configured
        # StorageFormat value. Otherwise, use the Backend config value if that
        # exists. If it doesn't, fall back to the default StorageFormat value.
        cls._value = (
            storage_format_config_value
            if storage_format_config_value is not _UNSET
            else (
                Backend.get_execution_for_backend(backend_config_value).storage_format
                if backend_config_value is not _UNSET
                else cls._get_default()
            )
        )

        return cls._value

    varname = "MODIN_STORAGE_FORMAT"
    default = "Pandas"
    choices = ("Pandas", "Native")


Execution = namedtuple("Execution", ["storage_format", "engine"])


class Backend(EnvironmentVariableDisallowingExecutionAndBackendBothSet, type=str):
    """
    An alias for execution, i.e. the combination of StorageFormat and Engine.

    Setting backend may change StorageFormat and/or Engine to the corresponding
    respective values, and setting Engine or StorageFormat may change Backend.

    Modin's built-in backends include:
        - "Ray" <-> (StorageFormat="Pandas", Engine="Ray")
        - "Dask" <-> (StorageFormat="Pandas", Engine="Dask")
        - "Python_Test" <-> (StorageFormat="Pandas", Engine="Python")
            - This execution mode is meant for testing only.
        - "Unidist" <-> (StorageFormat="Pandas", Engine="Unidist")
        - "Pandas" <-> (StorageFormat="Native", Engine="Native")
    """

    _BACKEND_TO_EXECUTION: dict[str, Execution] = {}
    _EXECUTION_TO_BACKEND: dict[Execution, str] = {}
    varname: str = "MODIN_BACKEND"
    choices: tuple[str, ...] = ("Ray", "Dask", "Python_Test", "Unidist", "Pandas")

    @classmethod
    def put(cls, value: str) -> None:
        """
        Set the backend value.

        Parameters
        ----------
        value : str
            Backend value to set.
        """
        execution = cls.get_execution_for_backend(value)
        set_execution(execution.engine, execution.storage_format)

    @classmethod
    def _get_default(cls) -> str:
        """
        Get the default backend value.

        Returns
        -------
        str
            Default backend value.
        """
        return cls._EXECUTION_TO_BACKEND[
            Execution(StorageFormat._get_default(), Engine._get_default())
        ]

    @classmethod
    def register_backend(cls: type["Backend"], name: str, execution: Execution) -> None:
        """
        Register a new backend.

        Parameters
        ----------
        name : str
            Backend name.
        execution : Execution
            Execution that corresponds to the backend.
        """
        name = cls.normalize(name)
        super().add_option(name)
        if name in cls._BACKEND_TO_EXECUTION:
            raise ValueError(
                f"Backend '{name}' is already registered with the execution {cls._BACKEND_TO_EXECUTION[name]}."
            )
        if execution in cls._EXECUTION_TO_BACKEND:
            raise ValueError(
                f"{execution} is already registered with the backend {cls._EXECUTION_TO_BACKEND[execution]}."
            )
        cls._BACKEND_TO_EXECUTION[name] = execution
        cls._EXECUTION_TO_BACKEND[execution] = name

    @classmethod
    def add_option(cls, choice: str) -> NoReturn:
        """
        Raise an exception for trying to add an option to Backend directly.

        Parameters
        ----------
        choice : str
            Choice to add. Unused.

        Raises
        ------
        ValueError
            Always.
        """
        raise ValueError(
            "Cannot add an option to Backend directly. Use Backend.register_backend instead."
        )

    @classmethod
    def set_active_backends(cls, new_choices: tuple) -> None:
        """
        Set the active backends available for manual and automatic switching.

        Other backends may have been registered, and those backends remain registered, but the
        set of engines that can be used is dynamically modified.

        Parameters
        ----------
        new_choices : tuple
            Choices to add.

        Raises
        ------
        ValueError
            Raises a ValueError when the set of new_choices are not already registered
        """
        if not all(i in cls._BACKEND_TO_EXECUTION for i in new_choices):
            raise ValueError(
                f"Active backend choices {new_choices} are not all registered."
            )
        cls.choices = new_choices

    @classmethod
    def get_active_backends(cls) -> tuple[str, ...]:
        """
        Get the active backends available for manual and automatic switching.

        Returns
        -------
        tuple[str, ...]
            returns the active set of backends for switching
        """
        return cls.choices

    @classmethod
    def get_backend_for_execution(cls, execution: Execution) -> str:
        """
        Get the backend for the execution.

        Parameters
        ----------
        execution : Execution
            Execution to get the backend for.

        Returns
        -------
        str
            Backend for the execution.
        """
        if execution not in cls._EXECUTION_TO_BACKEND:
            raise ValueError(
                f"{execution} has no known backend. Please register a "
                + "backend for it with Backend.register_backend()"
            )
        return cls._EXECUTION_TO_BACKEND[execution]

    @classmethod
    def get_execution_for_backend(cls, backend: str) -> Execution:
        """
        Get the execution for the given backend.

        Parameters
        ----------
        backend : str
            Backend to get the execution for.

        Returns
        -------
        execution : Execution
            The execution for the given backend
        """
        if not isinstance(backend, str):
            raise TypeError(
                "Backend value should be a string, but instead it is "
                + f"{repr(backend)} of type {type(backend)}."
            )
        normalized_value = cls.normalize(backend)
        if normalized_value not in cls.choices:
            backend_choice_string = ", ".join(f"'{choice}'" for choice in cls.choices)
            raise ValueError(
                f"Unknown backend '{backend}'. Available backends are: "
                + backend_choice_string
            )
        if normalized_value not in cls._BACKEND_TO_EXECUTION:
            raise ValueError(
                f"Backend '{backend}' has no known execution. Please "
                + "register an execution for it with Backend.register_backend()."
            )
        return cls._BACKEND_TO_EXECUTION[normalized_value]

    @classmethod
    def get(cls) -> str:
        """
        Get the backend.

        Returns
        -------
        str
            Backend.
        """
        # We have to override get() because Backend may need to get its value
        # from the OS's environment variables for Backend or Engine.

        cls._warn_if_deprecated()

        # First, check if we've already set the Backend value.
        if cls._value is not _UNSET:
            return cls._value

        backend_config_value = Backend._get_value_from_config()

        # If Backend is in the OS's configuration, use the configured Backend
        # value. Otherwise, we need to figure out the Backend value based on
        # the Engine and StorageFormat values.
        cls._value = (
            backend_config_value
            if backend_config_value is not _UNSET
            else cls.get_backend_for_execution(
                Execution(storage_format=StorageFormat.get(), engine=Engine.get())
            )
        )

        return cls._value


Backend.register_backend("Ray", Execution("Pandas", "Ray"))
Backend.register_backend("Dask", Execution("Pandas", "Dask"))
Backend.register_backend("Python_Test", Execution("Pandas", "Python"))
Backend.register_backend("Unidist", Execution("Pandas", "Unidist"))
Backend.register_backend("Pandas", Execution("Native", "Native"))


class AutoSwitchBackend(EnvironmentVariable, type=bool):
    """
    Whether automatic backend switching is allowed.

    When this flag is set, a Modin backend can attempt to automatically choose an appropriate backend
    for different operations based on features of the input data. When disabled, backends should
    avoid implicit backend switching outside of explicit operations like `to_pandas` and `to_ray`.
    """

    varname = "MODIN_AUTO_SWITCH_BACKENDS"
    default = True

    @classmethod
    def enable(cls) -> None:
        """Enable automatic backend switching."""
        cls.put(True)

    @classmethod
    def disable(cls) -> None:
        """Disable automatic backend switching."""
        cls.put(False)


class ShowBackendSwitchProgress(EnvironmentVariable, type=bool):
    """
    Whether to show progress when switching between backends.

    When enabled, progress messages are displayed during backend switches to inform users
    about data transfer operations. When disabled, backend switches occur silently.
    """

    varname = "MODIN_BACKEND_SWITCH_PROGRESS"
    default = True

    @classmethod
    def enable(cls) -> None:
        """Enable backend switch progress display."""
        cls.put(True)

    @classmethod
    def disable(cls) -> None:
        """Disable backend switch progress display."""
        cls.put(False)


class IsExperimental(EnvironmentVariable, type=bool):
    """Whether to Turn on experimental features."""

    varname = "MODIN_EXPERIMENTAL"


class IsRayCluster(EnvironmentVariable, type=bool):
    """Whether Modin is running on pre-initialized Ray cluster."""

    varname = "MODIN_RAY_CLUSTER"


class RayRedisAddress(EnvironmentVariable, type=ExactStr):
    """Redis address to connect to when running in Ray cluster."""

    varname = "MODIN_REDIS_ADDRESS"


class RayRedisPassword(EnvironmentVariable, type=ExactStr):
    """What password to use for connecting to Redis."""

    varname = "MODIN_REDIS_PASSWORD"
    default = secrets.token_hex(32)


class RayInitCustomResources(EnvironmentVariable, type=dict):
    """
    Ray node's custom resources to initialize with.

    Visit Ray documentation for more details:
    https://docs.ray.io/en/latest/ray-core/scheduling/resources.html#custom-resources

    Notes
    -----
    Relying on Modin to initialize Ray, you should set this config
    for the proper initialization with custom resources.
    """

    varname = "MODIN_RAY_INIT_CUSTOM_RESOURCES"
    default = None


class RayTaskCustomResources(EnvironmentVariable, type=dict):
    """
    Ray node's custom resources to request them in tasks or actors.

    Visit Ray documentation for more details:
    https://docs.ray.io/en/latest/ray-core/scheduling/resources.html#custom-resources

    Notes
    -----
    You can use this config to limit the parallelism for the entire workflow
    by setting the config at the very beginning.
    >>> import modin.config as cfg
    >>> cfg.RayTaskCustomResources.put({"special_hardware": 0.001})
    This way each single remote task or actor will require 0.001 of "special_hardware" to run.
    You can also use this config to limit the parallelism for a certain operation
    by setting the config with context.
    >>> with context(RayTaskCustomResources={"special_hardware": 0.001}):
    ...     df.<op>
    This way each single remote task or actor will require 0.001 of "special_hardware" to run
    within the context only.
    """

    varname = "MODIN_RAY_TASK_CUSTOM_RESOURCES"
    default = None


class CpuCount(EnvironmentVariable, type=int):
    """How many CPU cores to use during initialization of the Modin engine."""

    varname = "MODIN_CPUS"

    @classmethod
    def _put(cls, value: int) -> None:
        """
        Put specific value if CpuCount wasn't set by a user yet.

        Parameters
        ----------
        value : int
            Config value to set.

        Notes
        -----
        This method is used to set CpuCount from cluster resources internally
        and should not be called by a user.
        """
        if cls.get_value_source() == ValueSource.DEFAULT:
            cls.put(value)

    @classmethod
    def _get_default(cls) -> int:
        """
        Get default value of the config.

        Returns
        -------
        int
        """
        import multiprocessing

        return multiprocessing.cpu_count()

    @classmethod
    def get(cls) -> int:
        """
        Get ``CpuCount`` with extra checks.

        Returns
        -------
        int
        """
        cpu_count = super().get()
        if cpu_count <= 0:
            raise ValueError(f"`CpuCount` should be > 0; current value: {cpu_count}")
        return cpu_count


class GpuCount(EnvironmentVariable, type=int):
    """How may GPU devices to utilize across the whole distribution."""

    varname = "MODIN_GPUS"


class Memory(EnvironmentVariable, type=int):
    """
    How much memory (in bytes) give to an execution engine.

    Notes
    -----
    * In Ray case: the amount of memory to start the Plasma object store with.
    * In Dask case: the amount of memory that is given to each worker depending on CPUs used.
    """

    varname = "MODIN_MEMORY"


class NPartitions(EnvironmentVariable, type=int):
    """How many partitions to use for a Modin DataFrame (along each axis)."""

    varname = "MODIN_NPARTITIONS"

    @classmethod
    def _put(cls, value: int) -> None:
        """
        Put specific value if NPartitions wasn't set by a user yet.

        Parameters
        ----------
        value : int
            Config value to set.

        Notes
        -----
        This method is used to set NPartitions from cluster resources internally
        and should not be called by a user.
        """
        if cls.get_value_source() == ValueSource.DEFAULT:
            cls.put(value)

    @classmethod
    def _get_default(cls) -> int:
        """
        Get default value of the config.

        Returns
        -------
        int
        """
        return CpuCount.get()

    @classmethod
    def get(cls) -> int:
        """
        Get ``NPartitions`` with extra checks.

        Returns
        -------
        int
        """
        nparts = super().get()
        if nparts <= 0:
            raise ValueError(f"`NPartitions` should be > 0; current value: {nparts}")
        return nparts


class TestDatasetSize(EnvironmentVariable, type=str):
    """Dataset size for running some tests."""

    varname = "MODIN_TEST_DATASET_SIZE"
    choices = ("Small", "Normal", "Big")


class TrackFileLeaks(EnvironmentVariable, type=bool):
    """Whether to track for open file handles leakage during testing."""

    varname = "MODIN_TEST_TRACK_FILE_LEAKS"
    # Turn off tracking on Windows by default because
    # psutil's open_files() can be extremely slow on Windows (up to adding a few hours).
    # see https://github.com/giampaolo/psutil/pull/597
    default = sys.platform != "win32"


class AsvImplementation(EnvironmentVariable, type=ExactStr):
    """Allows to select a library that we will use for testing performance."""

    varname = "MODIN_ASV_USE_IMPL"
    choices = ("modin", "pandas")

    default = "modin"


class AsvDataSizeConfig(EnvironmentVariable, type=ExactStr):
    """Allows to override default size of data (shapes)."""

    varname = "MODIN_ASV_DATASIZE_CONFIG"
    default = None


class ProgressBar(EnvironmentVariable, type=bool):
    """Whether or not to show the progress bar."""

    varname = "MODIN_PROGRESS_BAR"
    default = False

    @classmethod
    def enable(cls) -> None:
        """Enable ``ProgressBar`` feature."""
        cls.put(True)

    @classmethod
    def disable(cls) -> None:
        """Disable ``ProgressBar`` feature."""
        cls.put(False)

    @classmethod
    def put(cls, value: bool) -> None:
        """
        Set ``ProgressBar`` value only if synchronous benchmarking is disabled.

        Parameters
        ----------
        value : bool
            Config value to set.
        """
        if value and BenchmarkMode.get():
            raise ValueError("ProgressBar isn't compatible with BenchmarkMode")
        super().put(value)


class BenchmarkMode(EnvironmentVariable, type=bool):
    """Whether or not to perform computations synchronously."""

    varname = "MODIN_BENCHMARK_MODE"
    default = False

    @classmethod
    def put(cls, value: bool) -> None:
        """
        Set ``BenchmarkMode`` value only if progress bar feature is disabled.

        Parameters
        ----------
        value : bool
            Config value to set.
        """
        if value and ProgressBar.get():
            raise ValueError("BenchmarkMode isn't compatible with ProgressBar")
        super().put(value)


class LogMode(EnvironmentVariable, type=ExactStr):
    """Set ``LogMode`` value if users want to opt-in."""

    varname = "MODIN_LOG_MODE"
    choices = ("enable", "disable")
    default = "disable"

    @classmethod
    def enable(cls) -> None:
        """Enable all logging levels."""
        cls.put("enable")

    @classmethod
    def disable(cls) -> None:
        """Disable logging feature."""
        cls.put("disable")


class LogMemoryInterval(EnvironmentVariable, type=int):
    """Interval (in seconds) to profile memory utilization for logging."""

    varname = "MODIN_LOG_MEMORY_INTERVAL"
    default = 5

    @classmethod
    def put(cls, value: int) -> None:
        """
        Set ``LogMemoryInterval`` with extra checks.

        Parameters
        ----------
        value : int
            Config value to set.
        """
        if value <= 0:
            raise ValueError(f"Log memory Interval should be > 0, passed value {value}")
        super().put(value)

    @classmethod
    def get(cls) -> int:
        """
        Get ``LogMemoryInterval`` with extra checks.

        Returns
        -------
        int
        """
        log_memory_interval = super().get()
        if log_memory_interval <= 0:
            raise ValueError(
                f"`LogMemoryInterval` should be > 0; current value: {log_memory_interval}"
            )
        return log_memory_interval


class LogFileSize(EnvironmentVariable, type=int):
    """Max size of logs (in MBs) to store per Modin job."""

    varname = "MODIN_LOG_FILE_SIZE"
    default = 10

    @classmethod
    def put(cls, value: int) -> None:
        """
        Set ``LogFileSize`` with extra checks.

        Parameters
        ----------
        value : int
            Config value to set.
        """
        if value <= 0:
            raise ValueError(f"Log file size should be > 0 MB, passed value {value}")
        super().put(value)

    @classmethod
    def get(cls) -> int:
        """
        Get ``LogFileSize`` with extra checks.

        Returns
        -------
        int
        """
        log_file_size = super().get()
        if log_file_size <= 0:
            raise ValueError(
                f"`LogFileSize` should be > 0; current value: {log_file_size}"
            )
        return log_file_size


class MetricsMode(EnvironmentVariable, type=ExactStr):
    """
    Set ``MetricsMode`` value to disable/enable metrics collection.

    Metric handlers are registered through `add_metric_handler` and can
    be used to record graphite-style timings or values. It is the
    responsibility of the handler to define how those emitted metrics
    are handled.
    """

    varname = "MODIN_METRICS_MODE"
    choices = ("enable", "disable")
    default = "enable"

    @classmethod
    def enable(cls) -> None:
        """Enable all metric collection."""
        cls.put("enable")

    @classmethod
    def disable(cls) -> None:
        """Disable all metric collection."""
        cls.put("disable")


class PersistentPickle(EnvironmentVariable, type=bool):
    """Whether serialization should be persistent."""

    varname = "MODIN_PERSISTENT_PICKLE"
    # When set to off, it allows faster serialization which is only
    # valid in current run (i.e. useless for saving to disk).
    # When set to on, Modin objects could be saved to disk and loaded
    # but serialization/deserialization could take more time.
    default = False


class MinPartitionSize(EnvironmentVariable, type=int):
    """
    Minimum number of rows/columns in a single pandas partition split.

    Once a partition for a pandas dataframe has more than this many elements,
    Modin adds another partition.
    """

    varname = "MODIN_MIN_PARTITION_SIZE"
    default = 32

    @classmethod
    def put(cls, value: int) -> None:
        """
        Set ``MinPartitionSize`` with extra checks.

        Parameters
        ----------
        value : int
            Config value to set.
        """
        if value <= 0:
            raise ValueError(f"Min partition size should be > 0, passed value {value}")
        super().put(value)

    @classmethod
    def get(cls) -> int:
        """
        Get ``MinPartitionSize`` with extra checks.

        Returns
        -------
        int
        """
        from modin.error_message import ErrorMessage

        ErrorMessage.single_warning(
            "`MinPartitionSize` is deprecated and will be removed in a future version. "
            + "This config has no longer effect, "
            + "use `MinRowPartitionSize` and `MinColumnPartitionSize` instead.",
            FutureWarning,
        )
        min_partition_size = super().get()
        if min_partition_size <= 0:
            raise ValueError(
                f"`MinPartitionSize` should be > 0; current value: {min_partition_size}"
            )
        return min_partition_size


class MinRowPartitionSize(EnvironmentVariable, type=int):
    """
    Minimum number of rows in a single pandas partition split.

    Once a partition for a pandas dataframe has more than this many elements,
    Modin adds another partition.
    """

    varname = "MODIN_MIN_ROW_PARTITION_SIZE"
    default = 32

    @classmethod
    def put(cls, value: int) -> None:
        """
        Set ``MinRowPartitionSize`` with extra checks.

        Parameters
        ----------
        value : int
            Config value to set.
        """
        if value <= 0:
            raise ValueError(
                f"Min row partition size should be > 0, passed value {value}"
            )
        super().put(value)

    @classmethod
    def get(cls) -> int:
        """
        Get ``MinRowPartitionSize`` with extra checks.

        Returns
        -------
        int
        """
        min_row_partition_size = super().get()
        if min_row_partition_size <= 0:
            raise ValueError(
                f"`MinRowPartitionSize` should be > 0; current value: {min_row_partition_size}"
            )
        return min_row_partition_size


class MinColumnPartitionSize(EnvironmentVariable, type=int):
    """
    Minimum number of columns in a single pandas partition split.

    Once a partition for a pandas dataframe has more than this many elements,
    Modin adds another partition.
    """

    varname = "MODIN_MIN_COLUMN_PARTITION_SIZE"
    default = 32

    @classmethod
    def put(cls, value: int) -> None:
        """
        Set ``MinColumnPartitionSize`` with extra checks.

        Parameters
        ----------
        value : int
            Config value to set.
        """
        if value <= 0:
            raise ValueError(
                f"Min column partition size should be > 0, passed value {value}"
            )
        super().put(value)

    @classmethod
    def get(cls) -> int:
        """
        Get ``MinColumnPartitionSize`` with extra checks.

        Returns
        -------
        int
        """
        min_column_partition_size = super().get()
        if min_column_partition_size <= 0:
            raise ValueError(
                f"`MinColumnPartitionSize` should be > 0; current value: {min_column_partition_size}"
            )
        return min_column_partition_size


class TestReadFromSqlServer(EnvironmentVariable, type=bool):
    """Set to true to test reading from SQL server."""

    varname = "MODIN_TEST_READ_FROM_SQL_SERVER"
    default = False


class TestReadFromPostgres(EnvironmentVariable, type=bool):
    """Set to true to test reading from Postgres."""

    varname = "MODIN_TEST_READ_FROM_POSTGRES"
    default = False


class GithubCI(EnvironmentVariable, type=bool):
    """Set to true when running Modin in GitHub CI."""

    varname = "MODIN_GITHUB_CI"
    default = False


class ModinNumpy(EnvironmentVariable, type=bool):
    """Set to true to use Modin's implementation of NumPy API."""

    varname = "MODIN_NUMPY"
    default = False


class RangePartitioning(EnvironmentVariable, type=bool):
    """
    Set to true to use Modin's range-partitioning implementation where possible.

    Please refer to documentation for cases where enabling this options would be beneficial:
    https://modin.readthedocs.io/en/stable/flow/modin/experimental/range_partitioning_groupby.html
    """

    varname = "MODIN_RANGE_PARTITIONING"
    default = False


class CIAWSSecretAccessKey(EnvironmentVariable, type=str):
    """Set to AWS_SECRET_ACCESS_KEY when running mock S3 tests for Modin in GitHub CI."""

    varname = "AWS_SECRET_ACCESS_KEY"
    default = "foobar_secret"


class CIAWSAccessKeyID(EnvironmentVariable, type=str):
    """Set to AWS_ACCESS_KEY_ID when running mock S3 tests for Modin in GitHub CI."""

    varname = "AWS_ACCESS_KEY_ID"
    default = "foobar_key"


class AsyncReadMode(EnvironmentVariable, type=bool):
    """
    It does not wait for the end of reading information from the source.

    It basically means, that the reading function only launches tasks for the dataframe
    to be read/created, but not ensures that the construction is finalized by the time
    the reading function returns a dataframe.

    This option was brought to improve performance of reading/construction
    of Modin DataFrames, however it may also:

    1. Increase the peak memory consumption. Since the garbage collection of the
    temporary objects created during the reading is now also lazy and will only
    be performed when the reading/construction is actually finished.

    2. Can break situations when the source is manually deleted after the reading
    function returns a result, for example, when reading inside of a context-block
    that deletes the file on ``__exit__()``.
    """

    varname = "MODIN_ASYNC_READ_MODE"
    default = False


class ReadSqlEngine(EnvironmentVariable, type=str):
    """Engine to run `read_sql`."""

    varname = "MODIN_READ_SQL_ENGINE"
    default = "Pandas"
    choices = ("Pandas", "Connectorx")


class LazyExecution(EnvironmentVariable, type=str):
    """
    Lazy execution mode.

    Supported values:
        `Auto` - the execution mode is chosen by the engine for each operation (default value).
        `On`   - the lazy execution is performed wherever it's possible.
        `Off`  - the lazy execution is disabled.
    """

    varname = "MODIN_LAZY_EXECUTION"
    choices = ("Auto", "On", "Off")
    default = "Auto"


class DocModule(EnvironmentVariable, type=ExactStr):
    """
    The module to use that will be used for docstrings.

    The value set here must be a valid, importable module. It should have
    a `DataFrame`, `Series`, and/or several APIs directly (e.g. `read_csv`).
    """

    varname = "MODIN_DOC_MODULE"
    default = "pandas"


class DaskThreadsPerWorker(EnvironmentVariable, type=int):
    """Number of threads per Dask worker."""

    varname = "MODIN_DASK_THREADS_PER_WORKER"
    default = 1


class NativePandasMaxRows(EnvironmentVariable, type=int):
    """Maximum number of rows which can be processed using local, native, pandas."""

    varname = "MODIN_NATIVE_MAX_ROWS"
    default = 10_000_000


class NativePandasTransferThreshold(EnvironmentVariable, type=int):
    """
    Targeted max number of dataframe rows which should be transferred between engines.

    This is often the same value as MODIN_NATIVE_MAX_ROWS but it can be independently
    set to change how transfer costs are considered.
    """

    varname = "MODIN_NATIVE_MAX_XFER_ROWS"
    default = 10_000_000


class DynamicPartitioning(EnvironmentVariable, type=bool):
    """
    Set to true to use Modin's dynamic-partitioning implementation where possible.

    Please refer to documentation for cases where enabling this options would be beneficial:
    https://modin.readthedocs.io/en/stable/usage_guide/optimization_notes/index.html#dynamic-partitioning-in-modin
    """

    varname = "MODIN_DYNAMIC_PARTITIONING"
    default = False


def _check_vars() -> None:
    """
    Check validity of environment variables.

    Look out for any environment variables that start with "MODIN_" prefix
    that are unknown - they might be a typo, so warn a user.
    """
    valid_names = {
        obj.varname
        for obj in globals().values()
        if isinstance(obj, type)
        and issubclass(obj, EnvironmentVariable)
        and not obj.is_abstract
    }
    found_names = {name for name in os.environ if name.startswith("MODIN_")}
    unknown = found_names - valid_names
    deprecated: dict[str, DeprecationDescriptor] = {
        obj.varname: obj._deprecation_descriptor
        for obj in globals().values()
        if isinstance(obj, type)
        and issubclass(obj, EnvironmentVariable)
        and not obj.is_abstract
        and obj.varname is not None
        and obj._deprecation_descriptor is not None
    }
    found_deprecated = found_names & deprecated.keys()
    if unknown:
        warnings.warn(
            f"Found unknown environment variable{'s' if len(unknown) > 1 else ''},"
            + f" please check {'their' if len(unknown) > 1 else 'its'} spelling: "
            + ", ".join(sorted(unknown))
        )
    for depr_var in found_deprecated:
        warnings.warn(
            deprecated[depr_var].deprecation_message(use_envvar_names=True),
            FutureWarning,
        )


_check_vars()
