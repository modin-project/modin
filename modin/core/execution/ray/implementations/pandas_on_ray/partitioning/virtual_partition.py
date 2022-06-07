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

    partition_type = PandasOnRayDataframePartition
    instance_type = ray.ObjectRef
    axis = None

    def __init__(self, list_of_blocks, get_ip=False, full_axis=True, call_queue=None):
        if isinstance(list_of_blocks, PandasOnRayDataframePartition):
            list_of_blocks = [list_of_blocks]
        self.full_axis = full_axis
        self.call_queue = call_queue or []
        # In the simple case, none of the partitions that will compose this
        # partition are themselves virtual partition. The partitions that will
        # be combined are just the partitions as given to the constructor.
        if not any(
            isinstance(o, PandasOnRayDataframeVirtualPartition) for o in list_of_blocks
        ):
            self.list_of_partitions_to_combine = list_of_blocks
            return
        # Check that all axis are the same in `list_of_blocks`
        # We should never have mismatching axis in the current implementation. We add this
        # defensive assertion to ensure that undefined behavior does not happen.
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
        # When the axis of all virtual partitions matches this axis,
        # extend and combine the lists of physical partitions.
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

    @property
    def list_of_blocks(self):
        """
        Get the list of physical partition objects that compose this partition.

        Returns
        -------
        List
            A list of ``ray.ObjectRef``.
        """
        # Defer draining call queue until we get the partitions
        # TODO Look into draining call queue at the same time as the task
        result = [None] * len(self.list_of_partitions_to_combine)
        for idx, partition in enumerate(self.list_of_partitions_to_combine):
            partition.drain_call_queue()
            result[idx] = partition.oid
        return result

    @property
    def list_of_ips(self):
        """
        Get the IPs holding the physical objects composing this partition.

        Returns
        -------
        List
            A list of IPs as ``ray.ObjectRef`` or str.
        """
        # Defer draining call queue until we get the ip address
        result = [None] * len(self.list_of_partitions_to_combine)
        for idx, partition in enumerate(self.list_of_partitions_to_combine):
            partition.drain_call_queue()
            result[idx] = partition._ip_cache
        return result

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
        *args,
        num_splits=None,
        other_axis_partition=None,
        maintain_partitioning=True,
        **kwargs,
    ):
        """
        Apply a function to this axis partition along full axis.

        Parameters
        ----------
        func : callable
            The function to apply.
        *args : iterable
            Additional positional arguments to be passed in `func`.
        num_splits : int, default: None
            The number of times to split the result object.
        other_axis_partition : PandasDataframeAxisPartition, default: None
            Another `PandasDataframeAxisPartition` object to be applied
            to func. This is for operations that are between two data sets.
        maintain_partitioning : bool, default: True
            Whether to keep the partitioning in the same
            orientation as it was previously or not. This is important because we may be
            operating on an individual AxisPartition and not touching the rest.
            In this case, we have to return the partitioning to its previous
            orientation (the lengths will remain the same). This is ignored between
            two axis partitions.
        **kwargs : dict
            Additional keywords arguments to be passed in `func`.

        Returns
        -------
        list
            A list of `PandasOnRayDataframeVirtualPartition` objects.
        """
        if not self.full_axis:
            # If this is not a full axis partition, it already contains a subset of
            # the full axis, so we shouldn't split the result further.
            num_splits = 1
        if len(self.call_queue) > 0:
            self.drain_call_queue()
        kwargs["args"] = args
        result = super(PandasOnRayDataframeVirtualPartition, self).apply(
            func,
            num_splits,
            other_axis_partition,
            maintain_partitioning,
            **kwargs,
        )
        if self.full_axis:
            return result
        else:
            # If this is a full axis partition, just take out the single split in the result.
            return result[0]

    def force_materialization(self, get_ip=False):
        """
        Materialize partitions into a single partition.

        Parameters
        ----------
        get_ip : bool, default: False
            Whether to get node ip address to a single partition or not.

        Returns
        -------
        PandasOnRayDataframeVirtualPartition
            An axis partition containing only a single materialized partition.
        """
        materialized = super(
            PandasOnRayDataframeVirtualPartition, self
        ).force_materialization(get_ip=get_ip)
        self.list_of_partitions_to_combine = materialized.list_of_partitions_to_combine
        return materialized

    def mask(self, row_indices, col_indices):
        """
        Create (synchronously) a mask that extracts the indices provided.

        Parameters
        ----------
        row_indices : list-like, slice or label
            The row labels for the rows to extract.
        col_indices : list-like, slice or label
            The column labels for the columns to extract.

        Returns
        -------
        PandasOnRayDataframeVirtualPartition
            A new ``PandasOnRayDataframeVirtualPartition`` object,
            materialized.
        """
        return (
            self.force_materialization()
            .list_of_partitions_to_combine[0]
            .mask(row_indices, col_indices)
        )

    def to_pandas(self):
        """
        Convert the data in this partition to a ``pandas.DataFrame``.

        Returns
        -------
        pandas DataFrame.
        """
        return self.force_materialization().list_of_partitions_to_combine[0].to_pandas()

    _length_cache = None

    def length(self):
        """
        Get the length of this partition.

        Returns
        -------
        int
            The length of the partition.
        """
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
        """
        Get the width of this partition.

        Returns
        -------
        int
            The width of the partition.
        """
        if self._width_cache is None:
            if self.axis == 1:
                self._width_cache = sum(
                    o.width() for o in self.list_of_partitions_to_combine
                )
            else:
                self._width_cache = self.list_of_partitions_to_combine[0].width()
        return self._width_cache

    def drain_call_queue(self, num_splits=None):
        """
        Execute all operations stored in this partition's call queue.

        Parameters
        ----------
        num_splits : int, default: None
            The number of times to split the result object.
        """

        def drain(df):
            for func, args, kwargs in self.call_queue:
                df = func(df, *args, **kwargs)
            return df

        drained = super(PandasOnRayDataframeVirtualPartition, self).apply(
            drain, num_splits=num_splits
        )
        self.list_of_partitions_to_combine = drained
        self.call_queue = []

    def wait(self):
        """Wait completing computations on the object wrapped by the partition."""
        self.drain_call_queue()
        futures = self.list_of_blocks
        ray.wait(futures, num_returns=len(futures))

    def add_to_apply_calls(self, func, *args, **kwargs):
        """
        Add a function to the call queue.

        Parameters
        ----------
        func : callable or ray.ObjectRef
            Function to be added to the call queue.
        *args : iterable
            Additional positional arguments to be passed in `func`.
        **kwargs : dict
            Additional keyword arguments to be passed in `func`.

        Returns
        -------
        PandasOnRayDataframeVirtualPartition
            A new ``PandasOnRayDataframeVirtualPartition`` object.

        Notes
        -----
        It does not matter if `func` is callable or an ``ray.ObjectRef``. Ray will
        handle it correctly either way. The keyword arguments are sent as a dictionary.
        """
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
