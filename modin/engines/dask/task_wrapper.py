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

from modin import execution_engine, Publisher


class DaskTask:
    get_global_client = None

    @classmethod
    def _update(cls, publisher: Publisher):
        from distributed.client import _get_global_client

        cls.get_global_client = _get_global_client

    @classmethod
    def deploy(cls, func, num_return_vals, kwargs):
        client = cls.get_global_client()
        remote_task_future = client.submit(func, **kwargs)
        return [
            client.submit(lambda l, i: l[i], remote_task_future, i)
            for i in range(num_return_vals)
        ]

    @classmethod
    def materialize(cls, future):
        client = cls.get_global_client()
        return client.gather(future)


execution_engine.once("Dask", DaskTask._update)
