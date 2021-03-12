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

from modin.config import CpuCount, NPartitions
from modin.error_message import ErrorMessage


def initialize_dask():
    from distributed.client import get_client

    try:
        client = get_client()
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
        client = Client(n_workers=CpuCount.get())

    num_cpus = len(client.ncores())
    NPartitions.put_if_default(num_cpus)
