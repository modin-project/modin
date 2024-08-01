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

import os

from modin.config import (
    CIAWSAccessKeyID,
    CIAWSSecretAccessKey,
    CpuCount,
    DaskThreadsPerWorker,
    GithubCI,
    Memory,
    NPartitions,
)
from modin.core.execution.utils import set_env


def initialize_dask():
    """Initialize Dask environment."""
    from distributed.client import default_client

    try:
        client = default_client()

        def _disable_warnings():
            import warnings

            warnings.simplefilter("ignore", category=FutureWarning)

        client.run(_disable_warnings)

    except ValueError:
        from distributed import Client

        num_cpus = CpuCount.get()
        threads_per_worker = DaskThreadsPerWorker.get()
        memory_limit = Memory.get()
        worker_memory_limit = memory_limit // num_cpus if memory_limit else "auto"

        # when the client is initialized, environment variables are inherited
        with set_env(PYTHONWARNINGS="ignore::FutureWarning"):
            client = Client(
                n_workers=num_cpus,
                threads_per_worker=threads_per_worker,
                memory_limit=worker_memory_limit,
            )

        if GithubCI.get():
            # set these keys to run tests that write to the mock s3 service. this seems
            # to be the way to pass environment variables to the workers:
            # https://jacobtomlinson.dev/posts/2021/bio-for-2021/
            access_key = CIAWSAccessKeyID.get()
            aws_secret = CIAWSSecretAccessKey.get()
            client.run(
                lambda: os.environ.update(
                    {
                        "AWS_ACCESS_KEY_ID": access_key,
                        "AWS_SECRET_ACCESS_KEY": aws_secret,
                    }
                )
            )

    num_cpus = len(client.ncores())
    NPartitions._put(num_cpus)
    CpuCount._put(num_cpus)
