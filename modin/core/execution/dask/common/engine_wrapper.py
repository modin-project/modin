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

from collections import UserDict

import pandas
from dask.distributed import wait
from distributed import Future
from distributed.client import default_client


def _deploy_dask_func(func, *args, return_pandas_df=None, **kwargs):  # pragma: no cover
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
    distributed.Future or list
        Dask identifier of the result being put into distributed memory.
    """
    result = func(*args, **kwargs)
    if return_pandas_df and not isinstance(result, pandas.DataFrame):
        result = pandas.DataFrame(result)
    return result


class DaskWrapper:
    """The class responsible for execution of remote operations."""

    @classmethod
    def deploy(
        cls,
        func,
        f_args=None,
        f_kwargs=None,
        return_pandas_df=None,
        num_returns=1,
        pure=True,
    ):
        """
        Deploy a function in a worker process.

        Parameters
        ----------
        func : callable or distributed.Future
            Function to be deployed in a worker process.
        f_args : list or tuple, optional
            Positional arguments to pass to ``func``.
        f_kwargs : dict, optional
            Keyword arguments to pass to ``func``.
        return_pandas_df : bool, optional
            Whether to convert the result of `func` to a pandas DataFrame or not.
        num_returns : int, default: 1
            The number of returned objects.
        pure : bool, default: True
            Whether or not `func` is pure. See `Client.submit` for details.

        Returns
        -------
        list
            The result of ``func`` split into parts in accordance with ``num_returns``.
        """
        client = default_client()
        args = [] if f_args is None else f_args
        kwargs = {} if f_kwargs is None else f_kwargs
        if callable(func):
            remote_task_future = client.submit(func, *args, pure=pure, **kwargs)
        else:
            # for the case where type(func) is distributed.Future
            remote_task_future = client.submit(
                _deploy_dask_func,
                func,
                *args,
                pure=pure,
                return_pandas_df=return_pandas_df,
                **kwargs,
            )
        if num_returns != 1:
            return [
                client.submit(lambda tup, i: tup[i], remote_task_future, i)
                for i in range(num_returns)
            ]
        return remote_task_future

    @classmethod
    def is_future(cls, item):
        """
        Check if the item is a Future.

        Parameters
        ----------
        item : distributed.Future or object
            Future or object to check.

        Returns
        -------
        boolean
            If the value is a future.
        """
        return isinstance(item, Future)

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

    @classmethod
    def put(cls, data, **kwargs):
        """
        Put data into distributed memory.

        Parameters
        ----------
        data : list, dict, or object
            Data to scatter out to workers. Output type matches input type.
        **kwargs : dict
            Additional keyword arguments to be passed in `Client.scatter`.

        Returns
        -------
        List, dict, iterator, or queue of futures matching the type of input.
        """
        if isinstance(data, dict):
            # there is a bug that looks similar to https://github.com/dask/distributed/issues/3965;
            # to avoid this we could change behaviour for serialization:
            # <Future: finished, type: collections.UserDict, key: UserDict-b8a15c164319c1d32fd28481125de455>
            # vs
            # {'sep': <Future: finished, type: pandas._libs.lib._NoDefault, key: sep>, \
            #  'delimiter': <Future: finished, type: NoneType, key: delimiter> ...
            data = UserDict(data)
        client = default_client()
        return client.scatter(data, **kwargs)

    @classmethod
    def wait(cls, obj_ids, num_returns=None):
        """
        Wait on the objects without materializing them (blocking operation).

        Parameters
        ----------
        obj_ids : list, scalar
        num_returns : int, optional
        """
        if not isinstance(obj_ids, list):
            obj_ids = [obj_ids]
        if num_returns is None:
            num_returns = len(obj_ids)
        if num_returns == len(obj_ids):
            wait(obj_ids, return_when="ALL_COMPLETED")
        else:
            # Dask doesn't natively support `num_returns` as int.
            # `wait` function doesn't always return only one finished future,
            # so a simple loop is not enough here
            done, not_done = wait(obj_ids, return_when="FIRST_COMPLETED")
            while len(done) < num_returns and (i := 0 < num_returns):
                extra_done, not_done = wait(not_done, return_when="FIRST_COMPLETED")
                done.update(extra_done)
                i += 1
