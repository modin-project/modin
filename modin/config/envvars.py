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
from textwrap import dedent
import warnings
from packaging import version

from .pubsub import Parameter, _TYPE_PARAMS, ExactStr


class EnvironmentVariable(Parameter, type=str, abstract=True):
    """
    Base class for environment variables-based configuration
    """

    varname: str = None

    @classmethod
    def _get_raw_from_config(cls) -> str:
        return os.environ[cls.varname]

    @classmethod
    def get_help(cls) -> str:
        help = f"{cls.varname}: {dedent(cls.__doc__ or 'Unknown').strip()}\n\tProvide {_TYPE_PARAMS[cls.type].help}"
        if cls.choices:
            help += f" (valid examples are: {', '.join(str(c) for c in cls.choices)})"
        return help


class IsDebug(EnvironmentVariable, type=bool):
    """
    Forces Modin engine to be "Python" unless specified by $MODIN_ENGINE
    """

    varname = "MODIN_DEBUG"


class Engine(EnvironmentVariable, type=str):
    """
    Distribution engine to run queries by
    """

    varname = "MODIN_ENGINE"
    choices = ("Ray", "Dask", "Python")

    @classmethod
    def _get_default(cls):
        if IsDebug.get():
            return "Python"
        try:
            import ray

        except ImportError:
            pass
        else:
            if version.parse(ray.__version__) < version.parse("1.0.0"):
                raise ImportError(
                    "Please `pip install modin[ray]` to install compatible Ray version."
                )
            return "Ray"
        try:
            import dask
            import distributed

        except ImportError:
            raise ImportError(
                "Please `pip install modin[ray]` or `modin[dask]` to install an engine"
            )
        if version.parse(dask.__version__) < version.parse("2.1.0") or version.parse(
            distributed.__version__
        ) < version.parse("2.3.2"):
            raise ImportError(
                "Please `pip install modin[dask]` to install compatible Dask version."
            )
        return "Dask"


class Backend(EnvironmentVariable, type=str):
    """
    Engine running on a single node of distribution
    """

    varname = "MODIN_BACKEND"
    default = "Pandas"
    choices = ("Pandas", "OmniSci", "Pyarrow")


class IsExperimental(EnvironmentVariable, type=bool):
    """
    Turns on experimental features
    """

    varname = "MODIN_EXPERIMENTAL"


class IsRayCluster(EnvironmentVariable, type=bool):
    """
    True if Modin is running on pre-initialized Ray cluster
    """

    varname = "MODIN_RAY_CLUSTER"


class RayRedisAddress(EnvironmentVariable, type=ExactStr):
    """
    What Redis address to connect to when running in Ray cluster
    """

    varname = "MODIN_REDIS_ADDRESS"


class CpuCount(EnvironmentVariable, type=int):
    """
    How many CPU cores to use when initialization of the Modin engine.
    """

    varname = "MODIN_CPUS"

    @classmethod
    def _get_default(cls):
        import multiprocessing

        return multiprocessing.cpu_count()


class Memory(EnvironmentVariable, type=int):
    """
    How much memory give to each Ray worker (in bytes)
    """

    varname = "MODIN_MEMORY"


class NPartitions(EnvironmentVariable, type=int):
    """
    How many partitions to use for a Modin DataFrame (along each axis)
    """

    varname = "MODIN_NPARTITIONS"
    # This flag is used to detect whether NPartitions is default value or not
    _is_default = False

    @classmethod
    def put(cls, value):
        super().put(value)
        cls._is_default = False

    @classmethod
    def put_if_default(cls, value):
        """
        Put specific value if NPartitions wasn't set by a user yet

        Notes
        -----
        This method is used to set NPartitions from cluster resources internally
        and should not be called by a user.
        """
        if cls._is_default:
            cls.put(value)

    @classmethod
    def _get_default(cls):
        cls._is_default = True
        return CpuCount.get()


class RayPlasmaDir(EnvironmentVariable, type=ExactStr):
    """
    Path to Plasma storage for Ray
    """

    varname = "MODIN_ON_RAY_PLASMA_DIR"


class IsOutOfCore(EnvironmentVariable, type=bool):
    """
    Changes primary location of the DataFrame to disk, allowing one to exceed total system memory
    """

    varname = "MODIN_OUT_OF_CORE"


class SocksProxy(EnvironmentVariable, type=ExactStr):
    """
    SOCKS proxy address if it is needed for SSH to work
    """

    varname = "MODIN_SOCKS_PROXY"


class DoLogRpyc(EnvironmentVariable, type=bool):
    """
    Whether to gather RPyC logs (applicable for remote context)
    """

    varname = "MODIN_LOG_RPYC"


class DoTraceRpyc(EnvironmentVariable, type=bool):
    """
    Whether to trace RPyC calls (applicable for remote context)
    """

    varname = "MODIN_TRACE_RPYC"


class OmnisciFragmentSize(EnvironmentVariable, type=int):
    """
    How big a fragment in OmniSci should be when creating a table (in rows)
    """

    varname = "MODIN_OMNISCI_FRAGMENT_SIZE"


class DoUseCalcite(EnvironmentVariable, type=bool):
    """
    Whether to use Calcite for OmniSci queries execution
    """

    varname = "MODIN_USE_CALCITE"
    default = True


class TestDatasetSize(EnvironmentVariable, type=str):
    """
    Dataset size for running some tests
    """

    varname = "MODIN_TEST_DATASET_SIZE"
    choices = ("Small", "Normal", "Big")


class TrackFileLeaks(EnvironmentVariable, type=bool):
    """
    Whether to track for open file handles leakage during testing
    """

    varname = "MODIN_TEST_TRACK_FILE_LEAKS"
    # Turn off tracking on Windows by default because
    # psutil's open_files() can be extremely slow on Windows (up to adding a few hours).
    # see https://github.com/giampaolo/psutil/pull/597
    default = sys.platform != "win32"


class AsvImplementation(EnvironmentVariable, type=ExactStr):
    """
    Allows to select a library that we will use for testing performance.
    """

    varname = "MODIN_ASV_USE_IMPL"
    choices = ("modin", "pandas")

    default = "modin"


class ProgressBar(EnvironmentVariable, type=bool):
    """
    Whether or not to show the progress bar
    """

    varname = "MODIN_PROGRESS_BAR"
    default = False

    @classmethod
    def enable(cls):
        cls.put(True)

    @classmethod
    def disable(cls):
        cls.put(False)

    @classmethod
    def put(cls, value):
        if value and BenchmarkMode.get():
            raise ValueError("ProgressBar isn't compatible with BenchmarkMode")
        super().put(value)


class BenchmarkMode(EnvironmentVariable, type=bool):
    """
    Whether or not to perform computations syncronous.
    """

    varname = "MODIN_BENCHMARK_MODE"
    default = False

    @classmethod
    def put(cls, value):
        if value and ProgressBar.get():
            raise ValueError("BenchmarkMode isn't compatible with ProgressBar")
        super().put(value)


class PersistentPickle(EnvironmentVariable, type=bool):
    """
    When set to off, it allows faster serialization which is only
    valid in current run (i.e. useless for saving to disk).
    When set to on, Modin objects could be saved to disk and loaded
    but serialization/deserialization could take more time.
    """

    varname = "MODIN_PERSISTENT_PICKLE"
    default = False


def _check_vars():
    """
    Look out for any environment variables that start with "MODIN_" prefix
    that are unknown - they might be a typo, so warn a user
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
            f" please check {'their' if len(unknown) > 1 else 'its'} spelling: "
            + ", ".join(sorted(unknown))
        )


_check_vars()
