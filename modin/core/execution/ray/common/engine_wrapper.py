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
from typing import Sequence

import pandas
import ray

from modin.config import RayTaskCustomResources
from modin.error_message import ErrorMessage


@ray.remote
def _deploy_ray_func(func, *args, return_pandas_df=None, **kwargs):  # pragma: no cover
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
    ray.ObjectRef or list
        Ray identifier of the result being put to Plasma store.
    """
    result = func(*args, **kwargs)
    if return_pandas_df and not isinstance(result, pandas.DataFrame):
        result = pandas.DataFrame(result)
    return result


class RayWrapper:
    """Mixin that provides means of running functions remotely and getting local results."""

    _func_cache = {}

    @classmethod
    def deploy(
        cls, func, f_args=None, f_kwargs=None, return_pandas_df=None, num_returns=1
    ):
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
        return_pandas_df : bool, optional
            Whether to convert the result of `func` to a pandas DataFrame or not.
        num_returns : int, default: 1
            Amount of return values expected from `func`.

        Returns
        -------
        ray.ObjectRef or list
            Ray identifier of the result being put to Plasma store.
        """
        args = [] if f_args is None else f_args
        kwargs = {} if f_kwargs is None else f_kwargs
        return _deploy_ray_func.options(
            num_returns=num_returns, resources=RayTaskCustomResources.get()
        ).remote(func, *args, return_pandas_df=return_pandas_df, **kwargs)

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
        return isinstance(item, ObjectRefTypes)

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
        if isinstance(obj_id, MaterializationHook):
            obj = obj_id.pre_materialize()
            return (
                obj_id.post_materialize(ray.get(obj))
                if isinstance(obj, ray.ObjectRef)
                else obj
            )

        if not isinstance(obj_id, Sequence):
            return ray.get(obj_id) if isinstance(obj_id, ray.ObjectRef) else obj_id

        if all(isinstance(obj, ray.ObjectRef) for obj in obj_id):
            return ray.get(obj_id)

        ids = {}
        result = []
        for obj in obj_id:
            if not isinstance(obj, ObjectRefTypes):
                result.append(obj)
                continue
            if isinstance(obj, MaterializationHook):
                oid = obj.pre_materialize()
                if isinstance(oid, ray.ObjectRef):
                    hook = obj
                    obj = oid
                else:
                    result.append(oid)
                    continue
            else:
                hook = None

            idx = ids.get(obj, None)
            if idx is None:
                ids[obj] = idx = len(ids)
            if hook is None:
                result.append(obj)
            else:
                hook._materialized_idx = idx
                result.append(hook)

        if len(ids) == 0:
            return result

        materialized = ray.get(list(ids.keys()))
        for i in range(len(result)):
            if isinstance((obj := result[i]), ObjectRefTypes):
                if isinstance(obj, MaterializationHook):
                    result[i] = obj.post_materialize(
                        materialized[obj._materialized_idx]
                    )
                else:
                    result[i] = materialized[ids[obj]]
        return result

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
        if not isinstance(obj_ids, Sequence):
            obj_ids = list(obj_ids)

        ids = set()
        for obj in obj_ids:
            if isinstance(obj, MaterializationHook):
                obj = obj.pre_materialize()
            if isinstance(obj, ray.ObjectRef):
                ids.add(obj)

        if num_ids := len(ids):
            ray.wait(list(ids), num_returns=num_returns or num_ids)


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


class MaterializationHook:
    """The Hook is called during the materialization and allows performing pre/post computations."""

    def pre_materialize(self):
        """
        Get an object reference to be materialized or a pre-computed value.

        Returns
        -------
        ray.ObjectRef or object
        """
        raise NotImplementedError()

    def post_materialize(self, materialized):
        """
        Perform computations on the materialized object.

        Parameters
        ----------
        materialized : object
            The materialized object to be post-computed.

        Returns
        -------
        object
            The post-computed object.
        """
        raise NotImplementedError()

    def __reduce__(self):
        """
        Replace this hook with the materialized object on serialization.

        Returns
        -------
        tuple
        """
        data = RayWrapper.materialize(self)
        if not isinstance(data, int):
            raise NotImplementedError("Only integers are currently supported")
        return int, (data,)


ObjectRefTypes = (ray.ObjectRef, MaterializationHook)
