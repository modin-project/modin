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

import builtins
import os
import sys

from modin.config import (
    IsRayCluster,
    RayRedisAddress,
    CpuCount,
    Memory,
    RayPlasmaDir,
    IsOutOfCore,
    NPartitions,
)


def handle_ray_task_error(e):
    for s in e.traceback_str.split("\n")[::-1]:
        if "Error" in s or "Exception" in s:
            try:
                raise getattr(builtins, s.split(":")[0])("".join(s.split(":")[1:]))
            except AttributeError as att_err:
                if "module" in str(att_err) and builtins.__name__ in str(att_err):
                    pass
                else:
                    raise att_err
    raise e


# Register a fix import function to run on all_workers including the driver.
# This is a hack solution to fix #647, #746
def _move_stdlib_ahead_of_site_packages(*args):
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


# Register a fix to import pandas on all workers before running tasks.
# This prevents a race condition between two threads deserializing functions
# and trying to import pandas at the same time.
def _import_pandas(*args):
    import pandas  # noqa F401


def initialize_ray(
    override_is_cluster=False,
    override_redis_address: str = None,
    override_redis_password: str = None,
):
    """
    Initializes ray based on parameters, environment variables and internal defaults.

    Parameters
    ----------
    override_is_cluster: bool, optional
        Whether to override the detection of Moding being run in a cluster
        and always assume this runs on cluster head node.
        This also overrides Ray worker detection and always runs the function,
        not only from main thread.
        If not specified, $MODIN_RAY_CLUSTER env variable is used.
    override_redis_address: str, optional
        What Redis address to connect to when running in Ray cluster.
        If not specified, $MODIN_REDIS_ADDRESS is used.
    override_redis_password: str, optional
        What password to use when connecting to Redis.
        If not specified, a new random one is generated.
    """
    import ray

    if not ray.is_initialized() or override_is_cluster:
        import secrets

        cluster = override_is_cluster or IsRayCluster.get()
        redis_address = override_redis_address or RayRedisAddress.get()
        redis_password = override_redis_password or secrets.token_hex(32)

        if cluster:
            # We only start ray in a cluster setting for the head node.
            ray.init(
                address=redis_address or "auto",
                include_dashboard=False,
                ignore_reinit_error=True,
                _redis_password=redis_password,
                logging_level=100,
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
            plasma_directory = RayPlasmaDir.get()
            if IsOutOfCore.get():
                if plasma_directory is None:
                    from tempfile import gettempdir

                    plasma_directory = gettempdir()
                # We may have already set the memory from the environment variable, we don't
                # want to overwrite that value if we have.
                if object_store_memory is None:
                    # Round down to the nearest Gigabyte.
                    mem_bytes = ray.utils.get_system_memory() // 10 ** 9 * 10 ** 9
                    # Default to 8x memory for out of core
                    object_store_memory = 8 * mem_bytes
            # In case anything failed above, we can still improve the memory for Modin.
            if object_store_memory is None:
                # Round down to the nearest Gigabyte.
                object_store_memory = int(
                    0.6 * ray.utils.get_system_memory() // 10 ** 9 * 10 ** 9
                )
                # If the memory pool is smaller than 2GB, just use the default in ray.
                if object_store_memory == 0:
                    object_store_memory = None
            else:
                object_store_memory = int(object_store_memory)
            ray.init(
                num_cpus=CpuCount.get(),
                include_dashboard=False,
                ignore_reinit_error=True,
                _plasma_directory=plasma_directory,
                object_store_memory=object_store_memory,
                address=redis_address,
                _redis_password=redis_password,
                logging_level=100,
                _memory=object_store_memory,
                _lru_evict=True,
            )
        _move_stdlib_ahead_of_site_packages()
        ray.worker.global_worker.run_function_on_all_workers(
            _move_stdlib_ahead_of_site_packages
        )

        ray.worker.global_worker.run_function_on_all_workers(_import_pandas)

    num_cpus = int(ray.cluster_resources()["CPU"])
    NPartitions.put_if_default(num_cpus)
