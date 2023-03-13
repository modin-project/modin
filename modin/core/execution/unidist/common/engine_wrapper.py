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

To be used as a piece of building a unidist-based engine.
"""

import asyncio
import unidist


@unidist.remote
def _deploy_unidist_func(func, *args, **kwargs):  # pragma: no cover
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
    unidist.ObjectRef or list[unidist.ObjectRef]
        Unidist identifier of the result being put to object store.
    """
    return func(*args, **kwargs)


class UnidistWrapper:
    """Mixin that provides means of running functions remotely and getting local results."""

    @classmethod
    def deploy(cls, func, f_args=None, f_kwargs=None, num_returns=1):
        """
        Run local `func` remotely.

        Parameters
        ----------
        func : callable or unidist.ObjectRef
            The function to perform.
        f_args : list or tuple, optional
            Positional arguments to pass to ``func``.
        f_kwargs : dict, optional
            Keyword arguments to pass to ``func``.
        num_returns : int, default: 1
            Amount of return values expected from `func`.

        Returns
        -------
        unidist.ObjectRef or list
            Unidist identifier of the result being put to object store.
        """
        args = [] if f_args is None else f_args
        kwargs = {} if f_kwargs is None else f_kwargs
        return _deploy_unidist_func.options(num_returns=num_returns).remote(
            func, *args, **kwargs
        )

    @classmethod
    def materialize(cls, obj_id):
        """
        Get the value of object from the object store.

        Parameters
        ----------
        obj_id : unidist.ObjectRef
            Unidist object identifier to get the value by.

        Returns
        -------
        object
            Whatever was identified by `obj_id`.
        """
        return unidist.get(obj_id)

    @classmethod
    def put(cls, data, **kwargs):
        """
        Put data into the object store.

        Parameters
        ----------
        data : object
            Data to be put.
        **kwargs : dict
            Additional keyword arguments (mostly for compatibility).

        Returns
        -------
        unidist.ObjectRef
            A reference to `data`.
        """
        return unidist.put(data)


@unidist.remote
class SignalActor:  # pragma: no cover
    """
    Help synchronize across tasks and actors on cluster.

    Parameters
    ----------
    event_count : int
        Number of events required for synchronization.

    Notes
    -----
    For details see: https://docs.ray.io/en/latest/advanced.html?highlight=signalactor#multi-node-synchronization-using-an-actor.
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
