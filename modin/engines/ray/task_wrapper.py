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


class RayTask:
    ray = None
    deploy_ray_func = None

    @classmethod
    def _update(cls, publisher: Publisher):
        import ray

        @ray.remote
        def deploy_ray_func(func, args):  # pragma: no cover
            return func(**args)

        cls.ray = ray
        cls.deploy_ray_func = staticmethod(deploy_ray_func)

    @classmethod
    def deploy(cls, func, num_return_vals, kwargs):
        return cls.deploy_ray_func._remote(
            args=(func, kwargs), num_return_vals=num_return_vals
        )

    @classmethod
    def materialize(cls, obj_id):
        return cls.ray.get(obj_id)


execution_engine.once("Ray", RayTask._update)
