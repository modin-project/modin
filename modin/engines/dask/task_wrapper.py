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

"""Module houses class responsible for execution of remote operations."""

from distributed.client import default_client


class DaskTask:
    """The class responsible for execution of remote operations."""

    @classmethod
    def deploy(cls, func, num_returns, kwargs):
        """
        Deploy a function in a worker process.

        Parameters
        ----------
        func : callable
            Function to be deployed in a worker process.
        num_returns : int
            The number of returned objects.
        kwargs : dict
            Additional keyword arguments to be passed in ``func``.

        Returns
        -------
        list
            The result of ``func`` splitted into parts in accordance with ``num_returns``.
        """
        client = default_client()
        remote_task_future = client.submit(func, **kwargs)
        return [
            client.submit(lambda l, i: l[i], remote_task_future, i)
            for i in range(num_returns)
        ]

    @classmethod
    def materialize(cls, future):
        """
        Materialize data matching `future` object.

        Parameters
        ----------
        future : distributed.Future or list
            Future object of list of future objects whereby data needs to be materialized.

        Returns
        -------
        Any
            An object(s) from the distributed memory.
        """
        client = default_client()
        return client.gather(future)
