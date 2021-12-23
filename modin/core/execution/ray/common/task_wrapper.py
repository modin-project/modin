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

import asyncio

import ray


@ray.remote
def deploy_ray_func(func, *args, **kwargs):  # pragma: no cover
    """
    Wrap `func` to ease calling it remotely.

    Parameters
    ----------
    func : callable
        A local function that we want to call remotely.
    *args : list
        Positional arguments to pass to `func` when calling remotely.
    **kwargs : dict
        Keyword arguments to be passed in ``func``.

    Returns
    -------
    ray.ObjectRef or list
        Ray identifier of the result being put to Plasma store.
    """
    return func(*args, **kwargs)


class RayWrapper:
    """Mixin that provides means of running functions remotely and getting local results."""

    @classmethod
    def deploy(cls, func, num_returns, *args, **kwargs):
        """
        Run local `func` remotely.

        Parameters
        ----------
        func : callable
            A function to call.
        num_returns : int
            Amount of return values expected from ``func``.
        *args : list
            Additional positional arguments to be passed in ``func``.
        **kwargs : dict
            Additional keyword arguments to be passed in ``func``.

        Returns
        -------
        ray.ObjectRef or list
            Ray identifier of the result being put to Plasma store.
        """
        # return ray.remote(func, num_returns=num_returns).remote(*args, **kwargs)
        return deploy_ray_func.options(num_returns=num_returns).remote(
            func, *args, kwargs
        )

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

    @classmethod
    def put(cls, data, **kwargs):
        """
        Store an object in the object store.

        The object may not be evicted while a reference to the returned ID exists.

        Parameters
        ----------
        data : object
            The Python object to be stored.
        **kwargs : dict
            Additional keyword arguments to be passed in ``ray.put``.

        Returns
        -------
        The object ref assigned to this value.
        """
        return ray.put(data, **kwargs)

    @classmethod
    def create_actor(cls, _class, *args, **kwargs):
        """
        TODO: add doc.
        """
        return ActorWrapper(_class, *args, **kwargs)


class ActorWrapper:
    def __init__(self, _cls, *args, **kwargs):
        self._cls = _cls
        self._remote_cls = ray.remote(_cls).remote(*args, **kwargs)

    def __getattribute__(self, name):
        if not name.startswith("_") and hasattr(self._cls, name):

            def wrapper(*args, **kwargs):
                print(name)
                return self._remote_cls.__getattribute__(name).remote(*args, **kwargs)

            return wrapper
        return super().__getattribute__(name)


class SignalActor:  # pragma: no cover
    """
    Help synchronize across tasks and actors on cluster.

    For details see: https://docs.ray.io/en/latest/advanced.html?highlight=signalactor#multi-node-synchronization-using-an-actor

    Parameters
    ----------
    event_count : int
        Number of events required for synchronization.
    """

    def __init__(self, event_count: int):
        self.events = [asyncio.Event() for _ in range(event_count)]

    def send(self, event_idx: int):
        """
        Indicate that event with `event_idx` has occured.

        Parameters
        ----------
        event_idx : int
        """
        self.events[event_idx].set()

    async def wait(self, event_idx: int):
        """
        Wait until event with `event_idx` has occured.

        Parameters
        ----------
        event_idx : int
        """
        await self.events[event_idx].wait()

    def is_set(self, event_idx: int) -> bool:
        """
        Check that event with `event_idx` had occured or not.

        Parameters
        ----------
        event_idx : int

        Returns
        -------
        bool
        """
        return self.events[event_idx].is_set()
