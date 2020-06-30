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

import threading
import os
import sys
import multiprocessing


def from_non_pandas(df, index, columns, dtype):
    from modin.data_management.dispatcher import EngineDispatcher

    new_qc = EngineDispatcher.from_non_pandas(df, index, columns, dtype)
    if new_qc is not None:
        from .dataframe import DataFrame

        return DataFrame(query_compiler=new_qc)
    return new_qc


def from_pandas(df):
    """Converts a pandas DataFrame to a Modin DataFrame.
    Args:
        df (pandas.DataFrame): The pandas DataFrame to convert.

    Returns:
        A new Modin DataFrame object.
    """
    from modin.data_management.dispatcher import EngineDispatcher
    from .dataframe import DataFrame

    return DataFrame(query_compiler=EngineDispatcher.from_pandas(df))


def to_pandas(modin_obj):
    """Converts a Modin DataFrame/Series to a pandas DataFrame/Series.

    Args:
        obj {modin.DataFrame, modin.Series}: The Modin DataFrame/Series to convert.

    Returns:
        A new pandas DataFrame or Series.
    """
    return modin_obj._to_pandas()


def _inherit_docstrings(parent, excluded=[]):
    """Creates a decorator which overwrites a decorated class' __doc__
    attribute with parent's __doc__ attribute. Also overwrites __doc__ of
    methods and properties defined in the class with the __doc__ of matching
    methods and properties in parent.

    Args:
        parent (object): Class from which the decorated class inherits __doc__.
        excluded (list): List of parent objects from which the class does not
            inherit docstrings.

    Returns:
        function: decorator which replaces the decorated class' documentation
            parent's documentation.
    """

    def decorator(cls):
        if parent not in excluded:
            cls.__doc__ = parent.__doc__
        for attr, obj in cls.__dict__.items():
            parent_obj = getattr(parent, attr, None)
            if parent_obj in excluded or (
                not callable(parent_obj) and not isinstance(parent_obj, property)
            ):
                continue
            if callable(obj):
                obj.__doc__ = parent_obj.__doc__
            elif isinstance(obj, property) and obj.fget is not None:
                p = property(obj.fget, obj.fset, obj.fdel, parent_obj.__doc__)
                setattr(cls, attr, p)
        return cls

    return decorator


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


def initialize_ray(cluster=None, redis_address=None, redis_password=None):
    import ray

    """Initializes ray based on environment variables and internal defaults."""
    if threading.current_thread().name == "MainThread":
        import secrets

        plasma_directory = None
        num_cpus = os.environ.get("MODIN_CPUS", None) or multiprocessing.cpu_count()
        cluster = os.environ.get("MODIN_RAY_CLUSTER", None) or cluster
        redis_address = os.environ.get("MODIN_REDIS_ADDRESS", None) or redis_address
        redis_password = redis_password or secrets.token_hex(16)

        if cluster == "True" and redis_address is not None:
            # We only start ray in a cluster setting for the head node.
            ray.init(
                num_cpus=int(num_cpus),
                include_webui=False,
                ignore_reinit_error=True,
                address=redis_address,
                redis_password=redis_password,
                logging_level=100,
            )
        elif cluster is None:
            object_store_memory = os.environ.get("MODIN_MEMORY", None)
            if os.environ.get("MODIN_OUT_OF_CORE", "False").title() == "True":
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
                num_cpus=int(num_cpus),
                include_webui=False,
                ignore_reinit_error=True,
                plasma_directory=plasma_directory,
                object_store_memory=object_store_memory,
                address=redis_address,
                redis_password=redis_password,
                logging_level=100,
                memory=object_store_memory,
                lru_evict=True,
            )

        _move_stdlib_ahead_of_site_packages()
        ray.worker.global_worker.run_function_on_all_workers(
            _move_stdlib_ahead_of_site_packages
        )

        ray.worker.global_worker.run_function_on_all_workers(_import_pandas)
