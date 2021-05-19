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

"""
The module with helper mixin for executing functions remotely.

To be used as a piece of building a Ray-based engine.
"""

import ray


@ray.remote
def deploy_ray_func(func, args):  # pragma: no cover
    """
    Wrap `func` to ease calling it remotely.

    Parameters
    ----------
    func : callable
        A local function that we want to call remotely.
    args : dict
        Keyword arguments to pass to `func` when calling remotely.

    Returns
    -------
    ray.ObjectRef or list
        Ray identifier of the result being put to Plasma store.
    """
    return func(**args)


class RayTask:
    """Mixin that provides means of running functions remotely and getting local results."""

    @classmethod
    def deploy(cls, func, num_returns, kwargs):
        """
        Run local `func` remotely.

        Parameters
        ----------
        func : callable
            A function to call.
        num_returns : int
            Amount of return values expected from `func`.
        kwargs : dict
            Keyword arguments to pass to remote instance of `func`.

        Returns
        -------
        ray.ObjectRef or list
            Ray identifier of the result being put to Plasma store.
        """
        return deploy_ray_func._remote(args=(func, kwargs), num_returns=num_returns)

    @classmethod
    def materialize(cls, obj_id):
        """
        Get the value of object from the Plasma store.

        Parameters
        ----------
        obj_id : ray.ObjectID
            Ray object identifier to get the value by.

        Returns
        -------
        object
            Whatever was identified by `obj_id`.
        """
        return ray.get(obj_id)
