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

import pandas
import unidist


@unidist.remote
def _deploy_unidist_func(
    func, *args, return_pandas_df=None, **kwargs
):  # pragma: no cover
    """
    Wrap `func` to ease calling it remotely.

    Parameters
    ----------
    func : callable
        A local function that we want to call remotely.
    *args : iterable
        Positional arguments to pass to `func` when calling remotely.
    return_pandas_df : bool, optional
        Whether to convert the result of `func` to a pandas DataFrame or not.
    **kwargs : dict
        Keyword arguments to pass to `func` when calling remotely.

    Returns
    -------
    unidist.ObjectRef or list[unidist.ObjectRef]
        Unidist identifier of the result being put to object store.
    """
    result = func(*args, **kwargs)
    if return_pandas_df and not isinstance(result, pandas.DataFrame):
        result = pandas.DataFrame(result)
    return result


class UnidistWrapper:
    """Mixin that provides means of running functions remotely and getting local results."""

    @classmethod
    def deploy(
        cls, func, f_args=None, f_kwargs=None, return_pandas_df=None, num_returns=1
    ):
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
        return_pandas_df : bool, optional
            Whether to convert the result of `func` to a pandas DataFrame or not.
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
            func, *args, return_pandas_df=return_pandas_df, **kwargs
        )

    @classmethod
    def is_future(cls, item):
        """
        Check if the item is a Future.

        Parameters
        ----------
        item : unidist.ObjectRef or object
            Future or object to check.

        Returns
        -------
        boolean
            If the value is a future.
        """
        return unidist.is_object_ref(item)

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

    @classmethod
    def wait(cls, obj_ids, num_returns=None):
        """
        Wait on the objects without materializing them (blocking operation).

        ``unidist.wait`` assumes a list of unique object references: see
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
        unidist.wait(unique_ids, num_returns=num_returns)


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
