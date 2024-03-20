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

import importlib
import os
import secrets
import sys
import warnings
from textwrap import dedent
from typing import Any, Optional

from packaging import version
from pandas.util._decorators import doc  # type: ignore[attr-defined]

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
    def _get_raw_from_config(cls) -> str:
        """
        Read the value from environment variable.

        Returns
        -------
        str
            Config raw value.

        Raises
        ------
        TypeError
            If `varname` is None.
        KeyError
            If value is absent.
        """
        if cls.varname is None:
            raise TypeError("varname should not be None")
        return os.environ[cls.varname]

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


class IsDebug(EnvironmentVariable, type=bool):
    """Force Modin engine to be "Python" unless specified by $MODIN_ENGINE."""

    varname = "MODIN_DEBUG"


class Engine(EnvironmentVariable, type=str):
    """Distribution engine to run queries by."""

    varname = "MODIN_ENGINE"
    choices = ("Ray", "Dask", "Python", "Native", "Unidist")

    NOINIT_ENGINES = {
        "Python",
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
            # We import ``DbWorker`` from this module since correct import of ``DbWorker`` itself
            # from HDK is located in it with all the necessary options for dlopen.
            from modin.experimental.core.execution.native.implementations.hdk_on_native.db_worker import (  # noqa
                DbWorker,
            )
        except ImportError:
            pass
        else:
            return "Native"
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


class StorageFormat(EnvironmentVariable, type=str):
    """Engine to run on a single node of distribution."""

    varname = "MODIN_STORAGE_FORMAT"
    default = "Pandas"
    choices = ("Pandas", "Hdk", "Cudf")


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


class CpuCount(EnvironmentVariable, type=int):
    """How many CPU cores to use during initialization of the Modin engine."""

    varname = "MODIN_CPUS"

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
        if StorageFormat.get() == "Cudf":
            return GpuCount.get()
        else:
            return CpuCount.get()


class HdkFragmentSize(EnvironmentVariable, type=int):
    """How big a fragment in HDK should be when creating a table (in rows)."""

    varname = "MODIN_HDK_FRAGMENT_SIZE"


class DoUseCalcite(EnvironmentVariable, type=bool):
    """Whether to use Calcite for HDK queries execution."""

    varname = "MODIN_USE_CALCITE"
    default = True


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
    choices = ("enable", "disable", "enable_api_only")
    default = "disable"

    @classmethod
    def enable(cls) -> None:
        """Enable all logging levels."""
        cls.put("enable")

    @classmethod
    def disable(cls) -> None:
        """Disable logging feature."""
        cls.put("disable")

    @classmethod
    def enable_api_only(cls) -> None:
        """Enable API level logging."""
        warnings.warn(
            "enable_api_only value for LogMode would be deprecated in Modin 0.30.0 use enable instead. For more details https://github.com/modin-project/modin/issues/7102"
        )
        cls.put("enable_api_only")


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
        assert log_memory_interval > 0, "`LogMemoryInterval` should be > 0"
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
        assert log_file_size > 0, "`LogFileSize` should be > 0"
        return log_file_size


class PersistentPickle(EnvironmentVariable, type=bool):
    """Whether serialization should be persistent."""

    varname = "MODIN_PERSISTENT_PICKLE"
    # When set to off, it allows faster serialization which is only
    # valid in current run (i.e. useless for saving to disk).
    # When set to on, Modin objects could be saved to disk and loaded
    # but serialization/deserialization could take more time.
    default = False


class HdkLaunchParameters(EnvironmentVariable, type=dict):
    """
    Additional command line options for the HDK engine.

    Please visit OmniSci documentation for the description of available parameters:
    https://docs.omnisci.com/installation-and-configuration/config-parameters#configuration-parameters-for-omniscidb
    """

    varname = "MODIN_HDK_LAUNCH_PARAMETERS"

    @classmethod
    def get(cls) -> dict:
        """
        Get the resulted command-line options.

        Decode and merge specified command-line options with the default one.

        Returns
        -------
        dict
            Decoded and verified config value.
        """
        custom_parameters = super().get()
        result = cls._get_default().copy()
        result.update(
            {key.replace("-", "_"): value for key, value in custom_parameters.items()}
        )
        return result

    @classmethod
    def _get_default(cls) -> Any:
        """
        Get default value of the config. Checks the pyhdk version and omits variables unsupported in prior versions.

        Returns
        -------
        dict
            Config keys and corresponding values.
        """
        if (default := getattr(cls, "default", None)) is None:
            cls.default = default = {
                "enable_union": 1,
                "enable_columnar_output": 1,
                "enable_lazy_fetch": 0,
                "null_div_by_zero": 1,
                "enable_watchdog": 0,
                "enable_thrift_logs": 0,
                "enable_multifrag_execution_result": 1,
                "cpu_only": 1,
            }

            try:
                import pyhdk

                if version.parse(pyhdk.__version__) >= version.parse("0.6.1"):
                    default["enable_lazy_dict_materialization"] = 0
                    default["log_dir"] = "pyhdk_log"
            except ImportError:
                # if pyhdk is not available, do not show any additional options
                pass
        return default


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
        min_partition_size = super().get()
        assert min_partition_size > 0, "`min_partition_size` should be > 0"
        return min_partition_size


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


class ModinNumpy(EnvWithSibilings, type=bool):
    """Set to true to use Modin's implementation of NumPy API."""

    varname = "MODIN_NUMPY"
    default = False

    @classmethod
    def _sibling(cls) -> type[EnvWithSibilings]:
        """Get a parameter sibling."""
        return ExperimentalNumPyAPI


class ExperimentalNumPyAPI(EnvWithSibilings, type=bool):
    """
    Set to true to use Modin's implementation of NumPy API.

    This parameter is deprecated. Use ``ModinNumpy`` instead.
    """

    varname = "MODIN_EXPERIMENTAL_NUMPY_API"
    default = False

    @classmethod
    def _sibling(cls) -> type[EnvWithSibilings]:
        """Get a parameter sibling."""
        return ModinNumpy


# Let the parameter's handling logic know that this variable is deprecated and that
# we should raise respective warnings
ExperimentalNumPyAPI._deprecation_descriptor = DeprecationDescriptor(
    ExperimentalNumPyAPI, ModinNumpy
)


class RangePartitioningGroupby(EnvWithSibilings, type=bool):
    """
    Set to true to use Modin's range-partitioning group by implementation.

    Experimental groupby is implemented using a range-partitioning technique,
    note that it may not always work better than the original Modin's TreeReduce
    and FullAxis implementations. For more information visit the according section
    of Modin's documentation: TODO: add a link to the section once it's written.
    """

    varname = "MODIN_RANGE_PARTITIONING_GROUPBY"
    default = False

    @classmethod
    def _sibling(cls) -> type[EnvWithSibilings]:
        """Get a parameter sibling."""
        return ExperimentalGroupbyImpl


class ExperimentalGroupbyImpl(EnvWithSibilings, type=bool):
    """
    Set to true to use Modin's range-partitioning group by implementation.

    This parameter is deprecated. Use ``RangePartitioningGroupby`` instead.
    """

    varname = "MODIN_EXPERIMENTAL_GROUPBY"
    default = False

    @classmethod
    def _sibling(cls) -> type[EnvWithSibilings]:
        """Get a parameter sibling."""
        return RangePartitioningGroupby


# Let the parameter's handling logic know that this variable is deprecated and that
# we should raise respective warnings
ExperimentalGroupbyImpl._deprecation_descriptor = DeprecationDescriptor(
    ExperimentalGroupbyImpl, RangePartitioningGroupby
)


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

    @classmethod
    def put(cls, value: str) -> None:
        """
        Assign a value to the DocModule config.

        Parameters
        ----------
        value : str
            Config value to set.
        """
        super().put(value)
        # Reload everything to apply the documentation. This is required since the
        # docs might already have been created and the implementation will assume
        # that the new docs are applied when the config is set. This set of operations
        # does this.
        import modin.pandas as pd

        importlib.reload(pd.accessor)
        importlib.reload(pd.base)
        importlib.reload(pd.dataframe)
        importlib.reload(pd.general)
        importlib.reload(pd.groupby)
        importlib.reload(pd.io)
        importlib.reload(pd.iterator)
        importlib.reload(pd.series)
        importlib.reload(pd.series_utils)
        importlib.reload(pd.utils)
        importlib.reload(pd.window)
        importlib.reload(pd)


class DaskThreadsPerWorker(EnvironmentVariable, type=int):
    """Number of threads per Dask worker."""

    varname = "MODIN_DASK_THREADS_PER_WORKER"
    default = 1


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
