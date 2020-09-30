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

import ray


@ray.remote
def deploy_ray_func(func, args):  # pragma: no cover
    return func(**args)


class RayTask:
    @classmethod
    def deploy(cls, func, num_returns, kwargs):
        return deploy_ray_func._remote(args=(func, kwargs), num_returns=num_returns)

    @classmethod
    def materialize(cls, obj_id):
        return ray.get(obj_id)
