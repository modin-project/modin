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
from typing import Optional
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
from .engine_wrapper import RayWrapper

_OBJECT_STORE_TO_SYSTEM_MEMORY_RATIO = 0.6
# This constant should be in sync with the limit in ray, which is private,
# not exposed to users, and not documented:
# https://github.com/ray-project/ray/blob/4692e8d8023e789120d3f22b41ffb136b50f70ea/python/ray/_private/ray_constants.py#L57-L62
_MAC_OBJECT_STORE_LIMIT_BYTES = 2 * 2**30

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
            object_store_memory = _get_object_store_memory()
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


def _get_object_store_memory() -> Optional[int]:
    """
    Get the object store memory we should start Ray with, in bytes.

    - If the ``Memory`` config variable is set, return that.
    - On Linux, take system memory from /dev/shm. On other systems use total
      virtual memory.
    - On Mac, never return more than Ray-specified upper limit.

    Returns
    -------
    Optional[int]
        The object store memory size in bytes, or None if we should use the Ray
        default.
    """
    object_store_memory = Memory.get()
    if object_store_memory is not None:
        return object_store_memory
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
                    + "increase size of /dev/shm with --shm-size in Docker. Also, you can can override the memory "
                    + "size for each Ray worker (in bytes) to the MODIN_MEMORY environment variable."
                )
        finally:
            os.close(shm_fd)
    else:
        system_memory = virtual_memory
    bytes_per_gb = 1e9
    object_store_memory = int(
        _OBJECT_STORE_TO_SYSTEM_MEMORY_RATIO
        * system_memory
        // bytes_per_gb
        * bytes_per_gb
    )
    if object_store_memory == 0:
        return None
    # Newer versions of ray don't allow us to initialize ray with object store
    # size larger than that _MAC_OBJECT_STORE_LIMIT_BYTES. It seems that
    # object store > the limit is too slow even on ray 1.0.0. However, limiting
    # the object store to _MAC_OBJECT_STORE_LIMIT_BYTES only seems to start
    # helping at ray version 1.3.0. So if ray version is at least 1.3.0, cap
    # the object store at _MAC_OBJECT_STORE_LIMIT_BYTES.
    # For background on the ray bug see:
    # - https://github.com/ray-project/ray/issues/20388
    # - https://github.com/modin-project/modin/issues/4872
    if sys.platform == "darwin" and version.parse(ray.__version__) >= version.parse(
        "1.3.0"
    ):
        object_store_memory = min(object_store_memory, _MAC_OBJECT_STORE_LIMIT_BYTES)
    return object_store_memory


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
        return RayWrapper.materialize(obj)
    elif isinstance(obj, (tuple, list)):
        # Ray will error if any elements are not ObjectIDType, but we still want ray to
        # perform batch deserialization for us -- thus, we must submit only the list elements
        # that are ObjectIDType, deserialize them, and restore them to their correct list index
        oid_indices, oids = [], []
        for i, ray_id in enumerate(obj):
            if isinstance(ray_id, ObjectIDType):
                oid_indices.append(i)
                oids.append(ray_id)
        ray_result = RayWrapper.materialize(oids)
        new_lst = list(obj[:])
        for i, deser_item in zip(oid_indices, ray_result):
            new_lst[i] = deser_item
        # Check that all objects have been deserialized
        assert not any([isinstance(o, ObjectIDType) for o in new_lst])
        return new_lst
    elif isinstance(obj, dict) and any(
        isinstance(val, ObjectIDType) for val in obj.values()
    ):
        return dict(zip(obj.keys(), RayWrapper.materialize(list(obj.values()))))
    else:
        return obj


def wait(obj_ids):
    """
    Wrap ``ray.wait`` to handle duplicate object references.

    ``ray.wait`` assumes a list of unique object references: see
    https://github.com/modin-project/modin/issues/5045

    Parameters
    ----------
    obj_ids : List[ObjectIDType]
        The object IDs to wait on.

    Returns
    -------
    Tuple[List[ObjectIDType], List[ObjectIDType]]
        A list of object IDs that are ready, and a list of object IDs remaining (this
        is the same as for ``ray.wait``). Unlike ``ray.wait``, the order of these IDs is not
        guaranteed.
    """
    unique_ids = list(set(obj_ids))
    return ray.wait(unique_ids, num_returns=len(unique_ids))
