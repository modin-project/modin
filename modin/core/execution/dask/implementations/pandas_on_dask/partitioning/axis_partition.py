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

"""Module houses classes responsible for storing an axis partition and applying a function to it."""

from modin.core.dataframe.pandas.partitioning.axis_partition import (
    PandasDataframeAxisPartition,
)
from .partition import PandasOnDaskDataframePartition

from distributed.client import default_client
from distributed import Future
from distributed.utils import get_ip
import pandas


class PandasOnDaskDataframeAxisPartition(PandasDataframeAxisPartition):
    """
    The class implements the interface in ``PandasDataframeAxisPartition``.

    Parameters
    ----------
    list_of_blocks : list
        List of ``PandasOnDaskDataframePartition`` objects.
    get_ip : bool, default: False
        Whether to get node IP addresses of conforming partitions or not.
    """

    def __init__(self, list_of_blocks, get_ip=False):
        for obj in list_of_blocks:
            obj.drain_call_queue()
        # Unwrap from PandasDataframePartition object for ease of use
        self.list_of_blocks = [obj.future for obj in list_of_blocks]
        if get_ip:
            self.list_of_ips = [obj._ip_cache for obj in list_of_blocks]

    partition_type = PandasOnDaskDataframePartition
    instance_type = Future

    @classmethod
    def deploy_axis_func(
        cls, axis, func, num_splits, kwargs, maintain_partitioning, *partitions
    ):
        """
        Deploy a function along a full axis.

        Parameters
        ----------
        axis : {0, 1}
            The axis to perform the function along.
        func : callable
            The function to perform.
        num_splits : int
            The number of splits to return (see `split_result_of_axis_func_pandas`).
        kwargs : dict
            Additional keywords arguments to be passed in `func`.
        maintain_partitioning : bool
            If True, keep the old partitioning if possible.
            If False, create a new partition layout.
        *partitions : iterable
            All partitions that make up the full axis (row or column).

        Returns
        -------
        list
            A list of distributed.Future.
        """
        client = default_client()
        axis_result = client.submit(
            deploy_dask_func,
            PandasDataframeAxisPartition.deploy_axis_func,
            axis,
            func,
            num_splits,
            kwargs,
            maintain_partitioning,
            *partitions,
            pure=False,
        )

        lengths = kwargs.get("_lengths", None)
        result_num_splits = len(lengths) if lengths else num_splits

        # We have to do this to split it back up. It is already split, but we need to
        # get futures for each.
        return [
            client.submit(lambda l: l[i], axis_result, pure=False)
            for i in range(result_num_splits * 4)
        ]

    @classmethod
    def deploy_func_between_two_axis_partitions(
        cls, axis, func, num_splits, len_of_left, other_shape, kwargs, *partitions
    ):
        """
        Deploy a function along a full axis between two data sets.

        Parameters
        ----------
        axis : {0, 1}
            The axis to perform the function along.
        func : callable
            The function to perform.
        num_splits : int
            The number of splits to return (see `split_result_of_axis_func_pandas`).
        len_of_left : int
            The number of values in `partitions` that belong to the left data set.
        other_shape : np.ndarray
            The shape of right frame in terms of partitions, i.e.
            (other_shape[i-1], other_shape[i]) will indicate slice to restore i-1 axis partition.
        kwargs : dict
            Additional keywords arguments to be passed in `func`.
        *partitions : iterable
            All partitions that make up the full axis (row or column) for both data sets.

        Returns
        -------
        list
            A list of distributed.Future.
        """
        client = default_client()
        axis_result = client.submit(
            deploy_dask_func,
            PandasDataframeAxisPartition.deploy_func_between_two_axis_partitions,
            axis,
            func,
            num_splits,
            len_of_left,
            other_shape,
            kwargs,
            *partitions,
            pure=False,
        )
        # We have to do this to split it back up. It is already split, but we need to
        # get futures for each.
        return [
            client.submit(lambda l: l[i], axis_result, pure=False)
            for i in range(num_splits * 4)
        ]

    def _wrap_partitions(self, partitions):
        """
        Wrap partitions passed as a list of distributed.Future with ``PandasOnDaskDataframePartition`` class.

        Parameters
        ----------
        partitions : list
            List of distributed.Future.

        Returns
        -------
        list
            List of ``PandasOnDaskDataframePartition`` objects.
        """
        return [
            self.partition_type(future, length, width, ip)
            for (future, length, width, ip) in zip(*[iter(partitions)] * 4)
        ]


class PandasOnDaskDataframeColumnPartition(PandasOnDaskDataframeAxisPartition):
    """
    The column partition implementation.

    All of the implementation for this class is in the parent class,
    and this class defines the axis to perform the computation over.

    Parameters
    ----------
    list_of_blocks : list
        List of ``PandasOnDaskDataframePartition`` objects.
    get_ip : bool, default: False
        Whether to get node IP addresses to conforming partitions or not.
    """

    axis = 0


class PandasOnDaskDataframeRowPartition(PandasOnDaskDataframeAxisPartition):
    """
    The row partition implementation.

    All of the implementation for this class is in the parent class,
    and this class defines the axis to perform the computation over.

    Parameters
    ----------
    list_of_blocks : list
        List of ``PandasOnDaskDataframePartition`` objects.
    get_ip : bool, default: False
        Whether to get node IP addresses to conforming partitions or not.
    """

    axis = 1


def deploy_dask_func(func, *args):
    """
    Execute a function on an axis partition in a worker process.

    Parameters
    ----------
    func : callable
        Function to be executed on an axis partition.
    *args : iterable
        Additional arguments that need to passed in ``func``.

    Returns
    -------
    list
        The result of the function ``func`` and metadata for it.
    """
    result = func(*args)
    ip = get_ip()
    if isinstance(result, pandas.DataFrame):
        return result, len(result), len(result.columns), ip
    elif all(isinstance(r, pandas.DataFrame) for r in result):
        return [i for r in result for i in [r, len(r), len(r.columns), ip]]
    else:
        return [i for r in result for i in [r, None, None, ip]]
