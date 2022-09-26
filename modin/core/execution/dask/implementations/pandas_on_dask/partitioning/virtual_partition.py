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
from modin.core.execution.dask.common import DaskWrapper
from modin.utils import _inherit_docstrings


class PandasOnDaskDataframeVirtualPartition(PandasDataframeAxisPartition):
    """
    The class implements the interface in ``PandasDataframeAxisPartition``.

    Parameters
    ----------
    list_of_partitions : Union[list, PandasOnDaskDataframePartition]
        List of ``PandasOnDaskDataframePartition`` and
        ``PandasOnDaskDataframeVirtualPartition`` objects, or a single
        ``PandasOnDaskDataframePartition``.
    get_ip : bool, default: False
        Whether to get node IP addresses of conforming partitions or not.
    full_axis : bool, default: True
        Whether or not the virtual partition encompasses the whole axis.
    call_queue : list, optional
        A list of tuples (callable, args, kwargs) that contains deferred calls.
    length : distributed.Future or int, optional
        Length, or reference to length, of wrapped ``pandas.DataFrame``.
    width : distributed.Future or int, optional
        Width, or reference to width, of wrapped ``pandas.DataFrame``.
    """

    axis = None
    partition_type = PandasOnDaskDataframePartition
    instance_type = Future

    def __init__(
        self,
        list_of_partitions,
        get_ip=False,
        full_axis=True,
        call_queue=None,
        length=None,
        width=None,
    ):
        if isinstance(list_of_partitions, PandasOnDaskDataframePartition):
            list_of_partitions = [list_of_partitions]
        self.call_queue = call_queue or []
        self.full_axis = full_axis
        self._length_cache = length
        self._width_cache = width
        # Check that all virtual partition axes are the same in `list_of_partitions`
        # We should never have mismatching axis in the current implementation. We add this
        # defensive assertion to ensure that undefined behavior does not happen.
        assert (
            len(
                set(
                    obj.axis
                    for obj in list_of_partitions
                    if isinstance(obj, PandasOnDaskDataframeVirtualPartition)
                )
            )
            <= 1
        )
        self._list_of_constituent_partitions = list_of_partitions
        # Defer computing _list_of_block_partitions because we might need to
        # drain call queues for that.
        self._list_of_block_partitions = None

    @property
    def list_of_block_partitions(self) -> list:
        """
        Get the list of block partitions that compose this partition.

        Returns
        -------
        List
            A list of ``PandasOnDaskDataframePartition``.
        """
        if self._list_of_block_partitions is not None:
            return self._list_of_block_partitions
        self._list_of_block_partitions = []
        # Extract block partitions from the block and virtual partitions that
        # constitute this partition.
        for partition in self._list_of_constituent_partitions:
            if isinstance(partition, PandasOnDaskDataframeVirtualPartition):
                if partition.axis == self.axis:
                    # We are building a virtual partition out of another
                    # virtual partition `partition` that contains its own list
                    # of block partitions, partition.list_of_block_partitions.
                    # `partition` may have its own call queue, which has to be
                    # applied to the entire `partition` before we execute any
                    # further operations on its block parittions.
                    partition.drain_call_queue()
                    self._list_of_block_partitions.extend(
                        partition.list_of_block_partitions
                    )
                else:
                    # If this virtual partition is made of virtual partitions
                    # for the other axes, squeeze such partitions into a single
                    # block so that this partition only holds a one-dimensional
                    # list of blocks. We could change this implementation to
                    # hold a 2-d list of blocks, but that would complicate the
                    # code quite a bit.
                    self._list_of_block_partitions.append(
                        partition.force_materialization().list_of_block_partitions[0]
                    )
            else:
                self._list_of_block_partitions.append(partition)
        return self._list_of_block_partitions

    @property
    def list_of_ips(self):
        """
        Get the IPs holding the physical objects composing this partition.

        Returns
        -------
        List
            A list of IPs as ``distributed.Future`` or str.
        """
        # Defer draining call queue until we get the ip address
        result = [None] * len(self.list_of_block_partitions)
        for idx, partition in enumerate(self.list_of_block_partitions):
            partition.drain_call_queue()
            result[idx] = partition._ip_cache
        return result

    @classmethod
    def deploy_axis_func(
        cls,
        axis,
        func,
        f_args,
        f_kwargs,
        num_splits,
        maintain_partitioning,
        *partitions,
        lengths=None,
        manual_partition=False,
    ):
        """
        Deploy a function along a full axis.

        Parameters
        ----------
        axis : {0, 1}
            The axis to perform the function along.
        func : callable
            The function to perform.
        f_args : list or tuple
            Positional arguments to pass to ``func``.
        f_kwargs : dict
            Keyword arguments to pass to ``func``.
        num_splits : int
            The number of splits to return (see `split_result_of_axis_func_pandas`).
        maintain_partitioning : bool
            If True, keep the old partitioning if possible.
            If False, create a new partition layout.
        *partitions : iterable
            All partitions that make up the full axis (row or column).
        lengths : iterable, default: None
            The list of lengths to shuffle the partition into.
        manual_partition : bool, default: False
            If True, partition the result with `lengths`.

        Returns
        -------
        list
            A list of distributed.Future.
        """
        result_num_splits = len(lengths) if lengths else num_splits
        return DaskWrapper.deploy(
            func=deploy_dask_func,
            f_args=(
                PandasDataframeAxisPartition.deploy_axis_func,
                axis,
                func,
                f_args,
                f_kwargs,
                num_splits,
                maintain_partitioning,
                *partitions,
            ),
            f_kwargs={
                "lengths": lengths,
                "manual_partition": manual_partition,
            },
            num_returns=result_num_splits * 4,
            pure=False,
        )

    @classmethod
    def deploy_func_between_two_axis_partitions(
        cls,
        axis,
        func,
        f_args,
        f_kwargs,
        num_splits,
        len_of_left,
        other_shape,
        *partitions,
    ):
        """
        Deploy a function along a full axis between two data sets.

        Parameters
        ----------
        axis : {0, 1}
            The axis to perform the function along.
        func : callable
            The function to perform.
        f_args : list or tuple
            Positional arguments to pass to ``func``.
        f_kwargs : dict
            Keyword arguments to pass to ``func``.
        num_splits : int
            The number of splits to return (see `split_result_of_axis_func_pandas`).
        len_of_left : int
            The number of values in `partitions` that belong to the left data set.
        other_shape : np.ndarray
            The shape of right frame in terms of partitions, i.e.
            (other_shape[i-1], other_shape[i]) will indicate slice to restore i-1 axis partition.
        *partitions : iterable
            All partitions that make up the full axis (row or column) for both data sets.

        Returns
        -------
        list
            A list of distributed.Future.
        """
        return DaskWrapper.deploy(
            func=deploy_dask_func,
            f_args=(
                PandasDataframeAxisPartition.deploy_func_between_two_axis_partitions,
                axis,
                func,
                f_args,
                f_kwargs,
                num_splits,
                len_of_left,
                other_shape,
                *partitions,
            ),
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
            A list of `PandasOnDaskDataframeVirtualPartition` objects.
        """
        if not self.full_axis:
            # If this is not a full axis partition, it already contains a subset of
            # the full axis, so we shouldn't split the result further.
            num_splits = 1
        if len(self.call_queue) > 0:
            self.drain_call_queue()
        result = super(PandasOnDaskDataframeVirtualPartition, self).apply(
            func,
            *args,
            num_splits=num_splits,
            other_axis_partition=other_axis_partition,
            maintain_partitioning=maintain_partitioning,
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
        PandasOnDaskDataframeVirtualPartition
            An axis partition containing only a single materialized partition.
        """
        materialized = super(
            PandasOnDaskDataframeVirtualPartition, self
        ).force_materialization(get_ip=get_ip)
        self._list_of_block_partitions = materialized.list_of_block_partitions
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
            .list_of_block_partitions[0]
            .mask(row_indices, col_indices)
        )

    def to_pandas(self):
        """
        Convert the data in this partition to a ``pandas.DataFrame``.

        Returns
        -------
        pandas DataFrame.
        """
        return self.force_materialization().list_of_block_partitions[0].to_pandas()

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
                    obj.length() for obj in self.list_of_block_partitions
                )
            else:
                self._length_cache = self.list_of_block_partitions[0].length()
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
                    obj.width() for obj in self.list_of_block_partitions
                )
            else:
                self._width_cache = self.list_of_block_partitions[0].width()
        return self._width_cache

    def drain_call_queue(self, num_splits=None):
        """
        Execute all operations stored in this partition's call queue.

        Parameters
        ----------
        num_splits : int, default: None
            The number of times to split the result object.
        """
        # TODO: Need to check if `drain_call_queue` speeds up if helper
        # `drain` function is serialized only once.
        drained = super(PandasOnDaskDataframeVirtualPartition, self).apply(
            PandasDataframeAxisPartition.drain,
            num_splits=num_splits,
            call_queue=self.call_queue,
        )
        self._list_of_block_partitions = drained
        self.call_queue = []

    def wait(self):
        """Wait completing computations on the object wrapped by the partition."""
        self.drain_call_queue()
        wait(self.list_of_blocks)

    def add_to_apply_calls(self, func, *args, length=None, width=None, **kwargs):
        """
        Add a function to the call queue.

        Parameters
        ----------
        func : callable
            Function to be added to the call queue.
        *args : iterable
            Additional positional arguments to be passed in `func`.
        length : distributed.Future or int, optional
            Length, or reference to length, of wrapped ``pandas.DataFrame``.
        width : distributed.Future or int, optional
            Width, or reference to width, of wrapped ``pandas.DataFrame``.
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
            self.list_of_block_partitions,
            full_axis=self.full_axis,
            call_queue=self.call_queue + [[func, args, kwargs]],
            length=length,
            width=width,
        )


@_inherit_docstrings(PandasOnDaskDataframeVirtualPartition.__init__)
class PandasOnDaskDataframeColumnPartition(PandasOnDaskDataframeVirtualPartition):
    axis = 0


@_inherit_docstrings(PandasOnDaskDataframeVirtualPartition.__init__)
class PandasOnDaskDataframeRowPartition(PandasOnDaskDataframeVirtualPartition):
    axis = 1


def deploy_dask_func(deployer, axis, f_to_deploy, f_args, f_kwargs, *args, **kwargs):
    """
    Execute a function on an axis partition in a worker process.

    This is ALWAYS called on either ``PandasDataframeAxisPartition.deploy_axis_func``
    or ``PandasDataframeAxisPartition.deploy_func_between_two_axis_partitions``, which both
    serve to deploy another dataframe function on a Dask worker process.

    Parameters
    ----------
    deployer : callable
        A `PandasDataFrameAxisPartition.deploy_*` method that will call `deploy_f`.
    axis : {0, 1}
        The axis to perform the function along.
    f_to_deploy : callable or RayObjectID
        The function to deploy.
    f_args : list or tuple
        Positional arguments to pass to ``f_to_deploy``.
    f_kwargs : dict
        Keyword arguments to pass to ``f_to_deploy``.
    *args : list
        Positional arguments to pass to ``func``.
    **kwargs : dict
        Keyword arguments to pass to ``func``.

    Returns
    -------
    list
        The result of the function ``func`` and metadata for it.
    """
    result = deployer(axis, f_to_deploy, f_args, f_kwargs, *args, **kwargs)
    ip = get_ip()
    if isinstance(result, pandas.DataFrame):
        return result, len(result), len(result.columns), ip
    elif all(isinstance(r, pandas.DataFrame) for r in result):
        return [i for r in result for i in [r, len(r), len(r.columns), ip]]
    else:
        return [i for r in result for i in [r, None, None, ip]]
