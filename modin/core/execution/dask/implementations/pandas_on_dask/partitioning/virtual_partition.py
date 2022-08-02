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

"""Module houses classes responsible for storing a virtual partition and applying a function to it."""

from distributed import Future
from distributed.utils import get_ip
from dask.distributed import wait as dask_wait

import pandas

from modin.core.dataframe.pandas.partitioning.axis_partition import (
    PandasDataframeAxisPartition,
)
from .partition import PandasOnDaskDataframePartition
from modin.core.execution.dask.common.engine_wrapper import DaskWrapper


class PandasOnDaskDataframeVirtualPartition(PandasDataframeAxisPartition):

    block_partition_type = PandasOnDaskDataframePartition
    instance_type = Future
    wait = dask_wait
    axis = None

    def wait(self):
        """Wait completing computations on the object wrapped by the partition."""
        self.drain_call_queue()
        wait(self.list_of_blocks)

    @classmethod
    def deploy_axis_func(
        cls,
        axis,
        func,
        num_splits,
        maintain_partitioning,
        *partitions,
        **kwargs,
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
        maintain_partitioning : bool
            If True, keep the old partitioning if possible.
            If False, create a new partition layout.
        *partitions : iterable
            All partitions that make up the full axis (row or column).
        **kwargs : dict
            Additional keywords arguments to be passed in `func`.

        Returns
        -------
        list
            A list of distributed.Future.
        """
        lengths = kwargs.get("_lengths", None)
        result_num_splits = len(lengths) if lengths else num_splits
        return DaskWrapper.deploy(
            deploy_dask_func,
            PandasDataframeAxisPartition.deploy_axis_func,
            axis,
            func,
            num_splits,
            maintain_partitioning,
            *partitions,
            num_returns=result_num_splits * 4,
            pure=False,
            **kwargs,
        )

    @classmethod
    def deploy_func_between_two_axis_partitions(
        cls, axis, func, num_splits, len_of_left, other_shape, *partitions, **kwargs
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
        *partitions : iterable
            All partitions that make up the full axis (row or column) for both data sets.
        **kwargs : dict
            Additional keywords arguments to be passed in `func`.

        Returns
        -------
        list
            A list of distributed.Future.
        """
        return DaskWrapper.deploy(
            deploy_dask_func,
            PandasDataframeAxisPartition.deploy_func_between_two_axis_partitions,
            axis,
            func,
            num_splits,
            len_of_left,
            other_shape,
            *partitions,
            num_returns=num_splits * 4,
            pure=False,
            **kwargs,
        )


class PandasOnDaskDataframeColumnPartition(PandasOnDaskDataframeVirtualPartition):
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
    full_axis : bool, default: True
        Whether this partition spans an entire axis of the dataframe.
    call_queue : list, optional
        A list of tuples (callable, args, kwargs) that contains deferred calls.
    """

    axis = 0


class PandasOnDaskDataframeRowPartition(PandasOnDaskDataframeVirtualPartition):
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
    full_axis : bool, default: True
        Whether this partition spans an entire axis of the dataframe.
    call_queue : list, optional
        A list of tuples (callable, args, kwargs) that contains deferred calls.
    """

    axis = 1


def deploy_dask_func(func, *args, **kwargs):
    """
    Execute a function on an axis partition in a worker process.

    Parameters
    ----------
    func : callable
        Function to be executed on an axis partition.
    *args : iterable
        Additional arguments that need to passed in ``func``.
    **kwargs : dict
        Additional keyword arguments to be passed in `func`.

    Returns
    -------
    list
        The result of the function ``func`` and metadata for it.
    """
    result = func(*args, **kwargs)
    ip = get_ip()
    if isinstance(result, pandas.DataFrame):
        return result, len(result), len(result.columns), ip
    elif all(isinstance(r, pandas.DataFrame) for r in result):
        return [i for r in result for i in [r, len(r), len(r.columns), ip]]
    else:
        return [i for r in result for i in [r, None, None, ip]]
