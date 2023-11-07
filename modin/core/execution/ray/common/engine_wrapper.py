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
import os
from types import FunctionType

import ray
from ray.util.client.common import ClientObjectRef

from modin.core.execution.ray.common.deferred_execution import (
    DeferredExecutionException,
)
from modin.error_message import ErrorMessage


@ray.remote
def _deploy_ray_func(func, *args, **kwargs):  # pragma: no cover
    """
    Wrap `func` to ease calling it remotely.

    Parameters
    ----------
    func : callable
        A local function that we want to call remotely.
    *args : iterable
        Positional arguments to pass to `func` when calling remotely.
    **kwargs : dict
        Keyword arguments to pass to `func` when calling remotely.

    Returns
    -------
    ray.ObjectRef or list
        Ray identifier of the result being put to Plasma store.
    """
    return func(*args, **kwargs)


class RayWrapper:
    """Mixin that provides means of running functions remotely and getting local results."""

    _func_cache = {}

    @classmethod
    def deploy(cls, func, f_args=None, f_kwargs=None, num_returns=1):
        """
        Run local `func` remotely.

        Parameters
        ----------
        func : callable or ray.ObjectID
            The function to perform.
        f_args : list or tuple, optional
            Positional arguments to pass to ``func``.
        f_kwargs : dict, optional
            Keyword arguments to pass to ``func``.
        num_returns : int, default: 1
            Amount of return values expected from `func`.

        Returns
        -------
        ray.ObjectRef or list
            Ray identifier of the result being put to Plasma store.
        """
        args = [] if f_args is None else f_args
        kwargs = {} if f_kwargs is None else f_kwargs
        return _deploy_ray_func.options(num_returns=num_returns).remote(
            func, *args, **kwargs
        )

    @classmethod
    def is_future(cls, item):
        """
        Check if the item is a Future.

        Parameters
        ----------
        item : ray.ObjectID or object
            Future or object to check.

        Returns
        -------
        boolean
            If the value is a future.
        """
        ObjectIDType = (ray.ObjectRef, ClientObjectRef)
        return isinstance(item, ObjectIDType)

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
        if isinstance(obj_id, (list, tuple)):
            return [cls.materialize(o) for o in obj_id]
        obj = ray.get(obj_id)
        if isinstance(obj, DeferredExecutionException):
            raise obj.args[1] from obj
        return obj

    @classmethod
    def put(cls, data, **kwargs):
        """
        Store an object in the object store.

        Parameters
        ----------
        data : object
            The Python object to be stored.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        ray.ObjectID
            Ray object identifier to get the value by.
        """
        if isinstance(data, FunctionType):
            qname = data.__qualname__
            if "<locals>" not in qname and "<lambda>" not in qname:
                ref = cls._func_cache.get(data, None)
                if ref is None:
                    if len(cls._func_cache) < 1024:
                        ref = ray.put(data)
                        cls._func_cache[data] = ref
                    else:
                        msg = "To many functions in the RayWrapper cache!"
                        assert "MODIN_GITHUB_CI" not in os.environ, msg
                        ErrorMessage.warn(msg)
                return ref
        return ray.put(data, **kwargs)

    @classmethod
    def wait(cls, obj_ids, num_returns=None):
        """
        Wait on the objects without materializing them (blocking operation).

        ``ray.wait`` assumes a list of unique object references: see
        https://github.com/modin-project/modin/issues/5045

        Parameters
        ----------
        obj_ids : list, scalar
        num_returns : int, optional
        """
        if not isinstance(obj_ids, list):
            obj_ids = [obj_ids]
        unique_ids = list(set(obj_ids))
        if num_returns is None:
            num_returns = len(unique_ids)
        ray.wait(unique_ids, num_returns=num_returns)


@ray.remote
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
