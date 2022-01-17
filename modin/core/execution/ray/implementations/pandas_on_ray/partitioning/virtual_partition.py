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

import pandas
from typing import List

from modin.core.dataframe.pandas.partitioning.axis_partition import (
    PandasDataframeAxisPartition,
)
from .partition import PandasOnRayDataframePartition

import ray
from ray.util import get_node_ip_address


class PandasOnRayDataframeVirtualPartition(PandasDataframeAxisPartition):
    """
    The class implements the interface in ``PandasDataframeAxisPartition``.

    Parameters
    ----------
    list_of_blocks : list
        List of ``PandasOnRayDataframePartition`` objects.
    get_ip : bool, default: False
        Whether to get node IP addresses to conforming partitions or not.
    """

    partition_type = PandasOnRayDataframePartition
    instance_type = ray.ObjectRef
    axis = None

    def __init__(self, list_of_blocks, get_ip=False, full_axis=True, call_queue=None):
        if isinstance(list_of_blocks, PandasOnRayDataframePartition):
            list_of_blocks = [list_of_blocks]
        if any(
            isinstance(o, PandasOnRayDataframeVirtualPartition) for o in list_of_blocks
        ):
            # Check that all axis are the same in `list_of_blocks`
            assert (
                len(
                    set(
                        o.axis
                        for o in list_of_blocks
                        if isinstance(o, PandasOnRayDataframeVirtualPartition)
                    )
                )
                == 1
            )
            # When the axis of all virtual partitions matches this axis, extend and combine the lists of
            # physical partitions
            if (
                next(
                    o
                    for o in list_of_blocks
                    if isinstance(o, PandasOnRayDataframeVirtualPartition)
                ).axis
                == self.axis
            ):
                new_list_of_blocks = []
                for o in list_of_blocks:
                    new_list_of_blocks.extend(
                        o.list_of_partitions_to_combine
                    ) if isinstance(
                        o, PandasOnRayDataframeVirtualPartition
                    ) else new_list_of_blocks.append(
                        o
                    )
                self.list_of_partitions_to_combine = new_list_of_blocks
            # Materialize partitions if the axis of this virtual does not match the virtual partitions
            else:
                self.list_of_partitions_to_combine = [
                    obj.force_materialization().list_of_partitions_to_combine[0]
                    if isinstance(obj, PandasOnRayDataframeVirtualPartition)
                    else obj
                    for obj in list_of_blocks
                ]
        else:
            self.list_of_partitions_to_combine = list_of_blocks
        self.full_axis = full_axis
        self.call_queue = call_queue or []

    @property
    def list_of_blocks(self):
        # Defer draining call queue until we get the partitions
        # TODO Look into draining call queue at the same time as the task
        for partition in self.list_of_partitions_to_combine:
            partition.drain_call_queue()
        return [o.oid for o in self.list_of_partitions_to_combine]

    @property
    def list_of_ips(self):
        return [obj._ip_cache for obj in self.list_of_partitions_to_combine]

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
            The number of splits to return (see ``split_result_of_axis_func_pandas``).
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
            A list of ``pandas.DataFrame``-s.
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
            kwargs,
            maintain_partitioning,
            *partitions,
        )

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
            The number of splits to return (see ``split_result_of_axis_func_pandas``).
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
            A list of ``pandas.DataFrame``-s.
        """
        return deploy_ray_func.options(num_returns=num_splits * 4).remote(
            PandasDataframeAxisPartition.deploy_func_between_two_axis_partitions,
            axis,
            func,
            num_splits,
            len_of_left,
            other_shape,
            kwargs,
            *partitions,
        )

    def _wrap_partitions(self, partitions):
        """
        Wrap partitions passed as a list of ``ray.ObjectRef`` with ``PandasOnRayDataframePartition`` class.

        Parameters
        ----------
        partitions : list
            List of ``ray.ObjectRef``.

        Returns
        -------
        list
            List of ``PandasOnRayDataframePartition`` objects.
        """
        return [
            self.partition_type(object_id, length, width, ip)
            for (object_id, length, width, ip) in zip(*[iter(partitions)] * 4)
        ]

    def apply(
        self,
        func,
        num_splits=None,
        other_axis_partition=None,
        maintain_partitioning=True,
        **kwargs,
    ):
        if not self.full_axis:
            num_splits = 1
        if len(self.call_queue) > 0:
            self.drain_call_queue()
        result = super(PandasOnRayDataframeVirtualPartition, self).apply(
            func, num_splits, other_axis_partition, maintain_partitioning, **kwargs
        )
        if not self.full_axis:
            # must unpack subset of the axis to ensure correct dimensions on partitions object
            return result[0]
        else:
            return result

    def add_partitions(self, axis: int, parts: List[PandasOnRayDataframePartition]):
        if axis == self.axis:
            return type(self)(self.list_of_partitions_to_combine + parts)
        else:
            raise NotImplementedError("I'm doing the simple case first :)")

    def force_materialization(self, get_ip=False):
        materialized = super(
            PandasOnRayDataframeVirtualPartition, self
        ).force_materialization(get_ip)
        self.list_of_partitions_to_combine = materialized.list_of_partitions_to_combine
        return materialized

    def mask(self, row_indices, col_indices):
        return (
            self.force_materialization()
            .list_of_partitions_to_combine[0]
            .mask(row_indices, col_indices)
        )

    def to_pandas(self):
        return self.force_materialization().list_of_partitions_to_combine[0].to_pandas()

    _length_cache = None

    def length(self):
        if self._length_cache is None:
            if self.axis == 0:
                self._length_cache = sum(
                    o.length() for o in self.list_of_partitions_to_combine
                )
            else:
                self._length_cache = self.list_of_partitions_to_combine[0].length()
        return self._length_cache

    _width_cache = None

    def width(self):
        if self._width_cache is None:
            if self.axis == 1:
                self._width_cache = sum(
                    o.width() for o in self.list_of_partitions_to_combine
                )
            else:
                self._width_cache = self.list_of_partitions_to_combine[0].width()
        return self._width_cache

    def drain_call_queue(self):
        def drain(df):
            for func, args, kwargs in self.call_queue:
                df = func(df, *args, **kwargs)
            return df

        drained = super(PandasOnRayDataframeVirtualPartition, self).apply(drain)
        self.list_of_partitions_to_combine = drained
        self.call_queue = []

    def add_to_apply_calls(self, func, *args, **kwargs):
        return type(self)(
            self.list_of_partitions_to_combine,
            full_axis=self.full_axis,
            call_queue=self.call_queue + [(func, args, kwargs)],
        )


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
    """

    axis = 1


@ray.remote
def deploy_ray_func(func, *args):  # pragma: no cover
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

    Notes
    -----
    Ray functions are not detected by codecov (thus pragma: no cover).
    """
    result = func(*args)
    ip = get_node_ip_address()
    if isinstance(result, pandas.DataFrame):
        return result, len(result), len(result.columns), ip
    elif all(isinstance(r, pandas.DataFrame) for r in result):
        return [i for r in result for i in [r, len(r), len(r.columns), ip]]
    else:
        return [i for r in result for i in [r, None, None, ip]]
