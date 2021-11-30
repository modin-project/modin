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

"""The module holds utility and initialization routines for Modin on Ray."""

import os
import sys
import psutil
import warnings
import asyncio

import ray

from modin.config import (
    StorageFormat,
    IsRayCluster,
    RayRedisAddress,
    RayRedisPassword,
    CpuCount,
    GpuCount,
    Memory,
    NPartitions,
)


def _move_stdlib_ahead_of_site_packages(*args):
    """
    Ensure packages from stdlib have higher import priority than from site-packages.

    Parameters
    ----------
    *args : tuple
        Ignored, added for compatibility with Ray.

    Notes
    -----
    This function is expected to be run on all workers including the driver.
    This is a hack solution to fix GH-#647, GH-#746.
    """
    site_packages_path = None
    site_packages_path_index = -1
    for i, path in enumerate(sys.path):
        if sys.exec_prefix in path and path.endswith("site-packages"):
            site_packages_path = path
            site_packages_path_index = i
            # break on first found
            break

    if site_packages_path is not None:
        # stdlib packages layout as follows:
        # - python3.x
        #   - typing.py
        #   - site-packages/
        #     - pandas
        # So extracting the dirname of the site_packages can point us
        # to the directory containing standard libraries.
        sys.path.insert(site_packages_path_index, os.path.dirname(site_packages_path))


def _import_pandas(*args):
    """
    Import pandas to make sure all its machinery is ready.

    This prevents a race condition between two threads deserializing functions
    and trying to import pandas at the same time.

    Parameters
    ----------
    *args : tuple
        Ignored, added for compatibility with Ray.

    Notes
    -----
    This function is expected to be run on all workers before any
    serialization or deserialization starts.
    """
    import pandas  # noqa F401


def initialize_ray(
    override_is_cluster=False,
    override_redis_address: str = None,
    override_redis_password: str = None,
):
    """
    Initialize Ray based on parameters, ``modin.config`` variables and internal defaults.

    Parameters
    ----------
    override_is_cluster : bool, default: False
        Whether to override the detection of Modin being run in a cluster
        and always assume this runs on cluster head node.
        This also overrides Ray worker detection and always runs the initialization
        function (runs from main thread only by default).
        If not specified, ``modin.config.IsRayCluster`` variable is used.
    override_redis_address : str, optional
        What Redis address to connect to when running in Ray cluster.
        If not specified, ``modin.config.RayRedisAddress`` is used.
    override_redis_password : str, optional
        What password to use when connecting to Redis.
        If not specified, ``modin.config.RayRedisPassword`` is used.
    """
    if not ray.is_initialized() or override_is_cluster:
        cluster = override_is_cluster or IsRayCluster.get()
        redis_address = override_redis_address or RayRedisAddress.get()
        redis_password = override_redis_password or RayRedisPassword.get()

        if cluster:
            # We only start ray in a cluster setting for the head node.
            ray.init(
                address=redis_address or "auto",
                include_dashboard=False,
                ignore_reinit_error=True,
                _redis_password=redis_password,
            )
        else:
            from modin.error_message import ErrorMessage

            # This string is intentionally formatted this way. We want it indented in
            # the warning message.
            ErrorMessage.not_initialized(
                "Ray",
                """
    import ray
    ray.init()
""",
            )
            object_store_memory = Memory.get()
            # In case anything failed above, we can still improve the memory for Modin.
            if object_store_memory is None:
                virtual_memory = psutil.virtual_memory().total
                if sys.platform.startswith("linux"):
                    shm_fd = os.open("/dev/shm", os.O_RDONLY)
                    try:
                        shm_stats = os.fstatvfs(shm_fd)
                        system_memory = shm_stats.f_bsize * shm_stats.f_bavail
                        if system_memory / (virtual_memory / 2) < 0.99:
                            warnings.warn(
                                f"The size of /dev/shm is too small ({system_memory} bytes). The required size "
                                f"at least half of RAM ({virtual_memory // 2} bytes). Please, delete files in /dev/shm or "
                                "increase size of /dev/shm with --shm-size in Docker. Also, you can set "
                                "the required memory size for each Ray worker in bytes to MODIN_MEMORY environment variable."
                            )
                    finally:
                        os.close(shm_fd)
                else:
                    system_memory = virtual_memory
                object_store_memory = int(0.6 * system_memory // 1e9 * 1e9)
                # If the memory pool is smaller than 2GB, just use the default in ray.
                if object_store_memory == 0:
                    object_store_memory = None
            else:
                object_store_memory = int(object_store_memory)

            ray_init_kwargs = {
                "num_cpus": CpuCount.get(),
                "num_gpus": GpuCount.get(),
                "include_dashboard": False,
                "ignore_reinit_error": True,
                "object_store_memory": object_store_memory,
                "address": redis_address,
                "_redis_password": redis_password,
                "_memory": object_store_memory,
            }
            ray.init(**ray_init_kwargs)

        if StorageFormat.get() == "Cudf":
            from modin.core.execution.ray.implementations.cudf_on_ray.frame.gpu_manager import (
                GPUManager,
            )
            from modin.core.execution.ray.implementations.cudf_on_ray.frame.partition_manager import (
                GPU_MANAGERS,
            )

            # Check that GPU_MANAGERS is empty because _update_engine can be called multiple times
            if not GPU_MANAGERS:
                for i in range(GpuCount.get()):
                    GPU_MANAGERS.append(GPUManager.remote(i))
    _move_stdlib_ahead_of_site_packages()
    ray.worker.global_worker.run_function_on_all_workers(
        _move_stdlib_ahead_of_site_packages
    )
    ray.worker.global_worker.run_function_on_all_workers(_import_pandas)
    num_cpus = int(ray.cluster_resources()["CPU"])
    num_gpus = int(ray.cluster_resources().get("GPU", 0))
    if StorageFormat.get() == "Cudf":
        NPartitions._put(num_gpus)
    else:
        NPartitions._put(num_cpus)


@ray.remote
class SignalActor:  # pragma: no cover
    """
    Help synchronize across tasks and actors on cluster.

    For details see: https://docs.ray.io/en/latest/advanced.html?highlight=signalactor#multi-node-synchronization-using-an-actor

    Parameters
    ----------
    event_count : int
        Number of events required for synchronization.
    """

    def __init__(self, event_count: int):
        self.events = [asyncio.Event() for _ in range(event_count)]

    def send(self, event_idx: int):
        """
        Indicate that event with `event_idx` has occured.

        Parameters
        ----------
        event_idx : int
        """
        self.events[event_idx].set()

    async def wait(self, event_idx: int):
        """
        Wait until event with `event_idx` has occured.

        Parameters
        ----------
        event_idx : int
        """
        await self.events[event_idx].wait()

    def is_set(self, event_idx: int) -> bool:
        """
        Check that event with `event_idx` had occured or not.

        Parameters
        ----------
        event_idx : int

        Returns
        -------
        bool
        """
        return self.events[event_idx].is_set()
