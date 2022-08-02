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

import pandas
import ray
from ray.util import get_node_ip_address

from modin.core.dataframe.pandas.partitioning.axis_partition import (
    PandasDataframeAxisPartition,
)
from modin.core.execution.ray.common.utils import deserialize
from .partition import PandasOnRayDataframePartition


class PandasOnRayDataframeVirtualPartition(PandasDataframeAxisPartition):
    """
    The class implements the interface in ``PandasDataframeAxisPartition``.

    Parameters
    ----------
    list_of_blocks : Union[list, PandasOnRayDataframePartition]
        List of ``PandasOnRayDataframePartition`` and
        ``PandasOnRayDataframeVirtualPartition`` objects, or a single
        ``PandasOnRayDataframePartition``.
    get_ip : bool, default: False
        Whether to get node IP addresses to conforming partitions or not.
    full_axis : bool, default: True
        Whether or not the virtual partition encompasses the whole axis.
    call_queue : list, optional
        A list of tuples (callable, args, kwargs) that contains deferred calls.
    """

    block_partition_type = PandasOnRayDataframePartition
    instance_type = ray.ObjectRef
    wait = ray.wait
    axis = None

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
            The number of splits to return (see ``split_result_of_axis_func_pandas``).
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
            A list of ``ray.ObjectRef``-s.
        """
        lengths = kwargs.get("_lengths", None)
        max_retries = kwargs.pop("max_retries", None)
        return deploy_ray_func.options(
            num_returns=(num_splits if lengths is None else len(lengths)) * 4,
            **({"max_retries": max_retries} if max_retries is not None else {}),
        ).remote(
            PandasDataframeAxisPartition.deploy_axis_func,
            axis,
            func,
            num_splits,
            maintain_partitioning,
            *partitions,
            **kwargs,
        )

    @classmethod
    def deploy_func_between_two_axis_partitions(
        cls,
        axis,
        func,
        num_splits,
        len_of_left,
        other_shape,
        *partitions,
        **kwargs,
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
            The number of splits to return (see ``split_result_of_axis_func_pandas``).
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
            A list of ``ray.ObjectRef``-s.
        """
        return deploy_ray_func.options(num_returns=num_splits * 4).remote(
            PandasDataframeAxisPartition.deploy_func_between_two_axis_partitions,
            axis,
            func,
            num_splits,
            len_of_left,
            other_shape,
            *partitions,
            **kwargs,
        )

    def wait(self):
        """Wait completing computations on the object wrapped by the partition."""
        self.drain_call_queue()
        futures = self.list_of_blocks
        ray.wait(futures, num_returns=len(futures))


class PandasOnRayDataframeColumnPartition(PandasOnRayDataframeVirtualPartition):
    """
    The column partition implementation.

    All of the implementation for this class is in the parent class,
    and this class defines the axis to perform the computation over.

    Parameters
    ----------
    list_of_blocks : list
        List of ``PandasOnRayDataframePartition`` objects.
    get_ip : bool, default: False
        Whether to get node IP addresses to conforming partitions or not.
    full_axis : bool, default: True
        Whether this partition spans an entire axis of the dataframe.
    call_queue : list, default: None
        Call queue that needs to be executed on the partition.
    """

    axis = 0


class PandasOnRayDataframeRowPartition(PandasOnRayDataframeVirtualPartition):
    """
    The row partition implementation.

    All of the implementation for this class is in the parent class,
    and this class defines the axis to perform the computation over.

    Parameters
    ----------
    list_of_blocks : list
        List of ``PandasOnRayDataframePartition`` objects.
    get_ip : bool, default: False
        Whether to get node IP addresses to conforming partitions or not.
    full_axis : bool, default: True
        Whether this partition spans an entire axis of the dataframe.
    call_queue : list, default: None
        Call queue that needs to be executed on the partition.
    """

    axis = 1


@ray.remote
def deploy_ray_func(func, *args, **kwargs):  # pragma: no cover
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
    list : Union[tuple, list]
        The result of the function ``func`` and metadata for it.

    Notes
    -----
    Ray functions are not detected by codecov (thus pragma: no cover).
    """
    if "args" in kwargs:
        kwargs["args"] = deserialize(kwargs["args"])
    result = func(*args, **kwargs)
    ip = get_node_ip_address()
    if isinstance(result, pandas.DataFrame):
        return result, len(result), len(result.columns), ip
    elif all(isinstance(r, pandas.DataFrame) for r in result):
        return [i for r in result for i in [r, len(r), len(r.columns), ip]]
    else:
        return [i for r in result for i in [r, None, None, ip]]
