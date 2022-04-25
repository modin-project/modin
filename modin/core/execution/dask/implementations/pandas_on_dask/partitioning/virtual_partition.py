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
from dask.distributed import wait

import pandas

from modin.core.dataframe.pandas.partitioning.axis_partition import (
    PandasDataframeAxisPartition,
)
from .partition import PandasOnDaskDataframePartition
from modin.core.execution.dask.common.engine_wrapper import DaskWrapper


class PandasOnDaskDataframeVirtualPartition(PandasDataframeAxisPartition):
    """
    The class implements the interface in ``PandasDataframeAxisPartition``.

    Parameters
    ----------
    list_of_blocks : list
        List of ``PandasOnDaskDataframePartition`` objects.
    get_ip : bool, default: False
        Whether to get node IP addresses of conforming partitions or not.
    full_axis : bool, default: True
        Whether or not the virtual partition encompasses the whole axis.
    """

    axis = None

    def __init__(self, list_of_blocks, get_ip=False, full_axis=True, call_queue=None):
        self.call_queue = call_queue or []
        self.full_axis = full_axis
        if isinstance(list_of_blocks, PandasOnDaskDataframePartition):
            list_of_blocks = [list_of_blocks]
        if not any(
            isinstance(o, PandasOnDaskDataframeVirtualPartition) for o in list_of_blocks
        ):
            self.list_of_partitions_to_combine = list_of_blocks
            return

        assert (
            len(
                set(
                    o.axis
                    for o in list_of_blocks
                    if isinstance(o, PandasOnDaskDataframeVirtualPartition)
                )
            )
            == 1
        )
        if (
            next(
                o
                for o in list_of_blocks
                if isinstance(o, PandasOnDaskDataframeVirtualPartition)
            ).axis
            == self.axis
        ):
            new_list_of_blocks = []
            for o in list_of_blocks:
                new_list_of_blocks.extend(
                    o.list_of_partitions_to_combine
                ) if isinstance(
                    0, PandasOnDaskDataframeVirtualPartition
                ) else new_list_of_blocks.append(
                    o
                )
            self.list_of_partitions_to_combine = new_list_of_blocks
        else:
            self.list_of_partitions_to_combine = [
                obj.force_materialization().list_of_partitions_to_combine[0]
                if isinstance(obj, PandasOnDaskDataframeVirtualPartition)
                else obj
                for obj in list_of_blocks
            ]

    partition_type = PandasOnDaskDataframePartition
    instance_type = Future

    @property
    def list_of_blocks(self):
        """
        Get the list of physical partition objects that compose this partition.

        Returns
        -------
        List
            A list of ``distributed.Future``.
        """
        result = [None] * len(self.list_of_partitions_to_combine)
        for idx, ptn in enumerate(self.list_of_partitions_to_combine):
            ptn.drain_call_queue()
            result[idx] = ptn._ip_cache
        return result

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
        lengths = kwargs.get("_lengths", None)
        result_num_splits = len(lengths) if lengths else num_splits
        return DaskWrapper.deploy(
            deploy_dask_func,
            PandasDataframeAxisPartition.deploy_axis_func,
            axis,
            func,
            num_splits,
            kwargs,
            maintain_partitioning,
            *partitions,
            num_returns=result_num_splits * 4,
            pure=False,
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
        return DaskWrapper.deploy(
            deploy_dask_func,
            PandasDataframeAxisPartition.deploy_func_between_two_axis_partitions,
            axis,
            func,
            num_splits,
            len_of_left,
            other_shape,
            kwargs,
            *partitions,
            num_returns=num_splits * 4,
            pure=False,
        )

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

    def apply(
        self,
        func,
        num_splits=None,
        other_axis_partition=None,
        maintain_partitioning=True,
        **kwargs
    ):
        """
        Apply a function to this axis partition along full axis.

        Parameters
        ----------
        func : callable
            The function to apply.
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
            A list of `PandasOnDaskDataframeVirtualPartition` objects.
        """
        if not self.full_axis:
            num_splits = 1
        if len(self.call_queue) > 0:
            self.drain_call_queue()
        result = super(PandasOnDaskDataframeVirtualPartition, self).apply(
            func, num_splits, other_axis_partition, maintain_partitioning, **kwargs
        )
        if self.full_axis:
            return result
        else:
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
        PandasOnDaskDataframeVirtualPartition
            An axis partition containing only a single materialized partition.
        """
        materialized = super(
            PandasOnDaskDataframeVirtualPartition, self
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
        PandasOnDaskDataframeVirtualPartition
            A new ``PandasOnDaskDataframeVirtualPartition`` object,
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
            THe width of the partition.
        """
        if self._width_cache is None:
            if self.axis == 1:
                self._width_cache = sum(
                    o.width() for o in self.list_of_partitions_to_combine
                )
            else:
                self._width_cache = self.list_of_partitions_to_combine[0].width()
        return self._width_cache

    def drain_call_queue(self):
        """Execute all operations stored in this partition's call queue."""

        def drain(df):
            for func, args, kwargs in self.call_queue:
                df = func(df, *args, **kwargs)
            return df

        drained = super(PandasOnDaskDataframeVirtualPartition, self).apply(drain)
        self.list_of_partitions_to_combine = drained
        self.call_queue = []

    def wait(self):
        """Wait completing computations on the object wrapped by the partition."""
        self.drain_call_queue()
        wait(self.list_of_blocks)

    def add_to_apply_calls(self, func, *args, **kwargs):
        """
        Add a function to the call queue.

        Parameters
        ----------
        func : callable
            Function to be added to the call queue.
        *args : iterable
            Additional positional arguments to be passed in `func`.
        **kwargs : dict
            Additional keyword arguments to be passed in `func`.

        Returns
        -------
        PandasOnDaskDataframeVirtualPartition
            A new ``PandasOnDaskDataframeVirtualPartition`` object.

        Notes
        -----
        The keyword arguments are sent as a dictionary.
        """
        return type(self)(
            self.list_of_partitions_to_combine,
            full_axis=self.full_axis,
            call_queue=self.call_queue + [(func, args, kwargs)],
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
        Whether or not the virtual partition encompasses the whole axis.
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
        Whether or not the virtual partition encompasses the whole axis.
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
