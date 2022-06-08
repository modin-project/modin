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
import sys
from textwrap import dedent
import warnings
from packaging import version
import secrets

from .pubsub import Parameter, _TYPE_PARAMS, ExactStr, ValueSource


class EnvironmentVariable(Parameter, type=str, abstract=True):
    """Base class for environment variables-based configuration."""

    varname: str = None

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
        KeyError
            If value is absent.
        """
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


class IsDebug(EnvironmentVariable, type=bool):
    """Force Modin engine to be "Python" unless specified by $MODIN_ENGINE."""

    varname = "MODIN_DEBUG"


class Engine(EnvironmentVariable, type=str):
    """Distribution engine to run queries by."""

    varname = "MODIN_ENGINE"
    choices = ("Ray", "Dask", "Python", "Native")

    @classmethod
    def _get_default(cls):
        """
        Get default value of the config.

        Returns
        -------
        str
        """
        from modin.utils import MIN_RAY_VERSION, MIN_DASK_VERSION

        if IsDebug.get():
            return "Python"
        try:
            import ray

        except ImportError:
            pass
        else:
            if version.parse(ray.__version__) < MIN_RAY_VERSION:
                raise ImportError(
                    f"Please `pip install modin[ray]` to install compatible Ray version (>={MIN_RAY_VERSION})."
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
                    f"Please `pip install modin[dask]` to install compatible Dask version (>={MIN_DASK_VERSION})."
                )
            return "Dask"
        try:
            # We import ``PyDbEngine`` from this module since correct import of ``PyDbEngine`` itself
            # from Omnisci is located in it with all the necessary options for dlopen.
            from modin.experimental.core.execution.native.implementations.omnisci_on_native.utils import (  # noqa
                PyDbEngine,
            )
        except ImportError:
            pass
        else:
            return "Native"
        raise ImportError(
            "Please refer to installation documentation page to install an engine"
        )


class StorageFormat(EnvironmentVariable, type=str):
    """Engine to run on a single node of distribution."""

    varname = "MODIN_STORAGE_FORMAT"
    default = "Pandas"
    choices = ("Pandas", "OmniSci", "Pyarrow", "Cudf")


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
    def _get_default(cls):
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
    def _put(cls, value):
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
    def _get_default(cls):
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


class SocksProxy(EnvironmentVariable, type=ExactStr):
    """SOCKS proxy address if it is needed for SSH to work."""

    varname = "MODIN_SOCKS_PROXY"


class DoLogRpyc(EnvironmentVariable, type=bool):
    """Whether to gather RPyC logs (applicable for remote context)."""

    varname = "MODIN_LOG_RPYC"


class DoTraceRpyc(EnvironmentVariable, type=bool):
    """Whether to trace RPyC calls (applicable for remote context)."""

    varname = "MODIN_TRACE_RPYC"


class OmnisciFragmentSize(EnvironmentVariable, type=int):
    """How big a fragment in OmniSci should be when creating a table (in rows)."""

    varname = "MODIN_OMNISCI_FRAGMENT_SIZE"


class DoUseCalcite(EnvironmentVariable, type=bool):
    """Whether to use Calcite for OmniSci queries execution."""

    varname = "MODIN_USE_CALCITE"
    default = True


class TestDatasetSize(EnvironmentVariable, type=str):
    """Dataset size for running some tests."""

    varname = "MODIN_TEST_DATASET_SIZE"
    choices = ("Small", "Normal", "Big")


class TestRayClient(EnvironmentVariable, type=bool):
    """Set to true to start and connect Ray client before a testing session starts."""

    varname = "MODIN_TEST_RAY_CLIENT"
    default = False


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
    def enable(cls):
        """Enable ``ProgressBar`` feature."""
        cls.put(True)

    @classmethod
    def disable(cls):
        """Disable ``ProgressBar`` feature."""
        cls.put(False)

    @classmethod
    def put(cls, value):
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
    def put(cls, value):
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
    def enable(cls):
        """Enable all logging levels."""
        cls.put("enable")

    @classmethod
    def disable(cls):
        """Disable logging feature."""
        cls.put("disable")

    @classmethod
    def enable_api_only(cls):
        """Enable API level logging."""
        cls.put("enable_api_only")


class LogMemoryInterval(EnvironmentVariable, type=int):
    """Interval (in seconds) to profile memory utilization for logging."""

    varname = "MODIN_LOG_MEMORY_INTERVAL"
    default = 5

    @classmethod
    def put(cls, value):
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
    def get(cls):
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
    def put(cls, value):
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
    def get(cls):
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


class OmnisciLaunchParameters(EnvironmentVariable, type=dict):
    """
    Additional command line options for the OmniSci engine.

    Please visit OmniSci documentation for the description of available parameters:
    https://docs.omnisci.com/installation-and-configuration/config-parameters#configuration-parameters-for-omniscidb
    """

    varname = "MODIN_OMNISCI_LAUNCH_PARAMETERS"
    default = {
        "enable_union": 1,
        "enable_columnar_output": 1,
        "enable_lazy_fetch": 0,
        "null_div_by_zero": 1,
        "enable_watchdog": 0,
        "enable_thrift_logs": 0,
    }

    @classmethod
    def get(self):
        """
        Get the resulted command-line options.

        Decode and merge specified command-line options with the default one.

        Returns
        -------
        dict
            Decoded and verified config value.
        """
        custom_parameters = super().get()
        result = self.default.copy()
        result.update(
            {key.replace("-", "_"): value for key, value in custom_parameters.items()}
        )
        return result


class MinPartitionSize(EnvironmentVariable, type=int):
    """
    Minimum number of rows/columns in a single pandas partition split.

    Once a partition for a pandas dataframe has more than this many elements,
    Modin adds another partition.
    """

    varname = "MODIN_MIN_PARTITION_SIZE"
    default = 32

    @classmethod
    def put(cls, value):
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
    def get(cls):
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


class ReadSqlEngine(EnvironmentVariable, type=str):
    """Engine to run `read_sql`."""

    varname = "MODIN_READ_SQL_ENGINE"
    default = "Pandas"
    choices = ("Pandas", "Connectorx")


def _check_vars():
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
    if unknown:
        warnings.warn(
            f"Found unknown environment variable{'s' if len(unknown) > 1 else ''},"
            + f" please check {'their' if len(unknown) > 1 else 'its'} spelling: "
            + ", ".join(sorted(unknown))
        )


_check_vars()
