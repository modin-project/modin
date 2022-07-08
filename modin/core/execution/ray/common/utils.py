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
from packaging import version
import warnings

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
    ValueSource,
)
from modin.error_message import ErrorMessage

ObjectIDType = ray.ObjectRef
if version.parse(ray.__version__) >= version.parse("1.2.0"):
    from ray.util.client.common import ClientObjectRef

    ObjectIDType = (ray.ObjectRef, ClientObjectRef)


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
    extra_init_kw = {"runtime_env": {"env_vars": {"__MODIN_AUTOIMPORT_PANDAS__": "1"}}}
    if not ray.is_initialized() or override_is_cluster:
        cluster = override_is_cluster or IsRayCluster.get()
        redis_address = override_redis_address or RayRedisAddress.get()
        redis_password = (
            (
                ray.ray_constants.REDIS_DEFAULT_PASSWORD
                if cluster
                else RayRedisPassword.get()
            )
            if override_redis_password is None
            and RayRedisPassword.get_value_source() == ValueSource.DEFAULT
            else override_redis_password or RayRedisPassword.get()
        )

        if cluster:
            # We only start ray in a cluster setting for the head node.
            ray.init(
                address=redis_address or "auto",
                include_dashboard=False,
                ignore_reinit_error=True,
                _redis_password=redis_password,
                **extra_init_kw,
            )
        else:
            # This string is intentionally formatted this way. We want it indented in
            # the warning message.
            ErrorMessage.not_initialized(
                "Ray",
                f"""
    import ray
    ray.init({', '.join([f'{k}={v}' for k,v in extra_init_kw.items()])})
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
                                + f"at least half of RAM ({virtual_memory // 2} bytes). Please, delete files in /dev/shm or "
                                + "increase size of /dev/shm with --shm-size in Docker. Also, you can set "
                                + "the required memory size for each Ray worker in bytes to MODIN_MEMORY environment variable."
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

            mac_size_limit = getattr(
                ray.ray_constants, "MAC_DEGRADED_PERF_MMAP_SIZE_LIMIT", None
            )
            if (
                sys.platform == "darwin"
                and mac_size_limit is not None
                and object_store_memory > mac_size_limit
            ):
                warnings.warn(
                    "On Macs, Ray's performance is known to degrade with "
                    + "object store size greater than "
                    + f"{mac_size_limit / 2 ** 30:.4} GiB. Ray by default does "
                    + "not allow setting an object store size greater than "
                    + "that. Modin is overriding that default limit because "
                    + "it would rather have a larger, slower object store "
                    + "than spill to disk more often. To override Modin's "
                    + "behavior, you can initialize Ray yourself."
                )
                os.environ["RAY_ENABLE_MAC_LARGE_OBJECT_STORE"] = "1"

            ray_init_kwargs = {
                "num_cpus": CpuCount.get(),
                "num_gpus": GpuCount.get(),
                "include_dashboard": False,
                "ignore_reinit_error": True,
                "object_store_memory": object_store_memory,
                "_redis_password": redis_password,
                "_memory": object_store_memory,
                **extra_init_kw,
            }
            ray.init(**ray_init_kwargs)

        if StorageFormat.get() == "Cudf":
            from modin.core.execution.ray.implementations.cudf_on_ray.partitioning import (
                GPUManager,
                GPU_MANAGERS,
            )

            # Check that GPU_MANAGERS is empty because _update_engine can be called multiple times
            if not GPU_MANAGERS:
                for i in range(GpuCount.get()):
                    GPU_MANAGERS.append(GPUManager.remote(i))

    # Now ray is initialized, check runtime env config - especially useful if we join
    # an externally pre-configured cluster
    env_vars = ray.get_runtime_context().runtime_env.get("env_vars", {})
    for varname, varvalue in extra_init_kw["runtime_env"]["env_vars"].items():
        if str(env_vars.get(varname, "")) != str(varvalue):
            ErrorMessage.single_warning(
                "When using a pre-initialized Ray cluster, please ensure that the runtime env "
                + f"sets environment variable {varname} to {varvalue}"
            )

    num_cpus = int(ray.cluster_resources()["CPU"])
    num_gpus = int(ray.cluster_resources().get("GPU", 0))
    if StorageFormat.get() == "Cudf":
        NPartitions._put(num_gpus)
    else:
        NPartitions._put(num_cpus)


def deserialize(obj):
    """
    Deserialize a Ray object.

    Parameters
    ----------
    obj : ObjectIDType, iterable of ObjectIDType, or mapping of keys to ObjectIDTypes
        Object(s) to deserialize.

    Returns
    -------
    obj
        The deserialized object.
    """
    if isinstance(obj, ObjectIDType):
        return ray.get(obj)
    elif isinstance(obj, (tuple, list)) and any(
        isinstance(o, ObjectIDType) for o in obj
    ):
        return ray.get(list(obj))
    elif isinstance(obj, dict) and any(
        isinstance(val, ObjectIDType) for val in obj.values()
    ):
        return dict(zip(obj.keys(), ray.get(list(obj.values()))))
    else:
        return obj
