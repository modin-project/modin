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

"""Module houses utility function to initialize Dask environment."""

from modin.config import CpuCount, Memory, NPartitions
from modin.error_message import ErrorMessage
from modin._compat import PandasCompatVersion


def initialize_dask():
    """Initialize Dask environment."""
    if PandasCompatVersion.CURRENT == PandasCompatVersion.PY36:
        try:
            import pickle5  # noqa: F401
        except ImportError:
            raise RuntimeError("Dask usage by Modin requires pickle5 on older Python")
    from distributed.client import default_client

    try:
        client = default_client()
    except ValueError:
        from distributed import Client

        # The indentation here is intentional, we want the code to be indented.
        ErrorMessage.not_initialized(
            "Dask",
            """
    from distributed import Client

    client = Client()
""",
        )
        num_cpus = CpuCount.get()
        memory_limit = Memory.get()
        worker_memory_limit = memory_limit // num_cpus if memory_limit else "auto"
        client = Client(n_workers=num_cpus, memory_limit=worker_memory_limit)

    num_cpus = len(client.ncores())
    NPartitions._put(num_cpus)
