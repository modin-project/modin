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
from textwrap import dedent
import warnings

from .pubsub import Publisher, _TYPE_HELP


class EnvironmentVariable(Publisher, type=str):
    """
    Base class for environment variables-based configuration
    """

    def __init_subclass__(cls, varname: str, **kw):
        cls.varname = varname
        if not kw.get("type"):
            kw["type"] = EnvironmentVariable.type
        super().__init_subclass__(**kw)

    @classmethod
    def _get_raw_from_config(cls) -> str:
        return os.environ[cls.varname]

    @classmethod
    def _get_help(cls) -> str:
        help = f"{cls.varname}: {dedent(cls.__doc__ or 'Unknown').strip()}\n\tProvide a {_TYPE_HELP[cls.type]}"
        if cls.choices:
            help += f" (valid examples are: {', '.join(cls.choices)})"
        return help


class Engine(EnvironmentVariable, varname="MODIN_ENGINE"):
    """
    Distribution engine to run queries by
    """

    choices = ("Ray", "Dask", "Python")


class Backend(EnvironmentVariable, varname="MODIN_BACKEND"):
    """
    Engine running on a single node of distribution
    """

    choices = ("Pandas", "OmniSci", "Pyarrow")


class IsDebug(EnvironmentVariable, varname="MODIN_DEBUG", type=bool):
    """
    Forces Modin engine to be "Python" unless specified by $MODIN_ENGINE
    """


class IsExperimental(EnvironmentVariable, varname="MODIN_EXPERIMENTAL", type=bool):
    """
    Turns on experimental features
    """


class IsRayCluster(EnvironmentVariable, varname="MODIN_RAY_CLUSTER", type=bool):
    """
    True if Modin is running on pre-initialized Ray cluster
    """


class RayRedisAddress(EnvironmentVariable, varname="MODIN_REDIS_ADDRESS"):
    """
    What Redis address to connect to when running in Ray cluster
    """


class CpuCount(EnvironmentVariable, varname="MODIN_CPUS", type=int):
    """
    How may CPU cores to utilize across the whole distribution
    """


class Memory(EnvironmentVariable, varname="MODIN_MEMORY", type=int):
    """
    How much memory give to each Ray worker (in bytes)
    """


class RayPlasmaDir(EnvironmentVariable, varname="MODIN_ON_RAY_PLASMA_DIR"):
    """
    Path to Plasma storage for Ray
    """


class IsOutOfCore(EnvironmentVariable, varname="MODIN_OUT_OF_CORE", type=bool):
    pass


class SocksProxy(EnvironmentVariable, varname="MODIN_SOCKS_PROXY"):
    """
    SOCKS proxy address if it is needed for SSH to work
    """


class DoLogRpyc(EnvironmentVariable, varname="MODIN_LOG_RPYC", type=bool):
    """
    Whether to gather RPyC logs (applicable for remote context)
    """


class DoTraceRpyc(EnvironmentVariable, varname="MODIN_TRACE_RPYC", type=bool):
    """
    Whether to trace RPyC calls (applicable for remote context)
    """


class OmnisciFragmentSize(
    EnvironmentVariable, varname="MODIN_OMNISCI_FRAGMENT_SIZE", type=int
):
    """
    How big a fragment in OmniSci should be when creating a table (in rows)
    """


class DoUseCalcite(EnvironmentVariable, varname="MODIN_USE_CALCITE", type=bool):
    """
    Whether to use Calcite for OmniSci queries execution
    """


class TestDatasetSize(EnvironmentVariable, varname="MODIN_TEST_DATASET_SIZE"):
    """
    Dataset size for running some tests
    """

    choices = ("small", "normal", "big")

def _check_vars():
    valid_names = {obj.varname for obj in globals().values() if obj is not EnvironmentVariable and isinstance(obj, type) and issubclass(obj, EnvironmentVariable)}
    found_names = {name for name in os.environ.keys() if name.startswith('MODIN_')}
    unknown = found_names - valid_names
    if unknown:
        warnings.warn(f"Found unknown environment variables, please check their spelling: {', '.join(sorted(unknown))}")

_check_vars()
