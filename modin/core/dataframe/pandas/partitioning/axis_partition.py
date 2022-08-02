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

"""The module defines base interface for an axis partition of a Modin DataFrame."""

import pandas
import numpy as np
from modin.core.storage_formats.pandas.utils import split_result_of_axis_func_pandas
from modin.core.dataframe.base.partitioning.axis_partition import (
    BaseDataframeAxisPartition,
)


class PandasDataframeAxisPartition(BaseDataframeAxisPartition):

    block_partition_type = None
    wait = None

    """
    An abstract class is created to simplify and consolidate the code for axis partition that run pandas.

    Because much of the code is similar, this allows us to reuse this code.


    Parameters
    ----------
    list_of_blocks : Union[list, block_partition_type]
        List of ``block_partition_type`` and
        ``PandasDataframeAxisPartition`` objects, or a single
        ``block_partition_type``.
    get_ip : bool, default: False
        Whether to get node IP addresses of conforming partitions or not.
    full_axis : bool, default: True
        Whether or not the virtual partition encompasses the whole axis.
    call_queue : list, optional
        A list of tuples (callable, args, kwargs) that contains deferred calls.
    """

    def __init__(self, list_of_blocks, get_ip=False, full_axis=True, call_queue=None):
        if isinstance(list_of_blocks, self.block_partition_type):
            list_of_blocks = [list_of_blocks]
        self.call_queue = call_queue or []
        self.full_axis = full_axis
        # In the simple case, none of the partitions that will compose this
        # partition are themselves virtual partition. The partitions that will
        # be combined are just the partitions as given to the constructor.
        if not any(
            isinstance(obj, PandasDataframeAxisPartition) for obj in list_of_blocks
        ):
            self.list_of_partitions_to_combine = list_of_blocks
            return
        # Check that all axis are the same in `list_of_blocks`
        # We should never have mismatching axis in the current implementation. We add this
        # defensive assertion to ensure that undefined behavior does not happen.
        assert (
            len(
                set(
                    obj.axis
                    for obj in list_of_blocks
                    if isinstance(obj, PandasDataframeAxisPartition)
                )
            )
            == 1
        )
        # When the axis of all virtual partitions matches this axis,
        # extend and combine the lists of physical partitions.
        if (
            next(
                obj
                for obj in list_of_blocks
                if isinstance(obj, PandasDataframeAxisPartition)
            ).axis
            == self.axis
        ):
            new_list_of_blocks = []
            for obj in list_of_blocks:
                new_list_of_blocks.extend(
                    obj.list_of_partitions_to_combine
                ) if isinstance(
                    obj, PandasDataframeAxisPartition
                ) else new_list_of_blocks.append(
                    obj
                )
            self.list_of_partitions_to_combine = new_list_of_blocks
        # Materialize partitions if the axis of this virtual does not match the virtual partitions
        else:
            self.list_of_partitions_to_combine = [
                obj.force_materialization().list_of_partitions_to_combine[0]
                if isinstance(obj, PandasDataframeAxisPartition)
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
            A list of ``distributed.Future``.
        """
        # Defer draining call queue until we get the partitions
        # TODO Look into draining call queue at the same time as the task
        result = [None] * len(self.list_of_partitions_to_combine)
        for idx, partition in enumerate(self.list_of_partitions_to_combine):
            partition.drain_call_queue()
            result[idx] = partition._data
        return result

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
        result = [None] * len(self.list_of_partitions_to_combine)
        for idx, partition in enumerate(self.list_of_partitions_to_combine):
            partition.drain_call_queue()
            result[idx] = partition._ip_cache
        return result

    def _wrap_partitions(self, partitions):
        """
        Wrap partitions passed as a list of distributed.Future with ``block_partition_type`` class.

        Parameters
        ----------
        partitions : list
            List of distributed.Future.

        Returns
        -------
        list
            List of ``block_partition_type`` objects.
        """
        return [
            self.block_partition_type(future, length, width, ip)
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
            A list of `PandasDataframePartition` objects.
        """
        if not self.full_axis:
            # If this is not a full axis partition, it already contains a subset of
            # the full axis, so we shouldn't split the result further.
            num_splits = 1
        elif num_splits is None:
            num_splits = len(self.list_of_blocks)
        if len(self.call_queue) > 0:
            self.drain_call_queue()
        kwargs["args"] = args

        if other_axis_partition is not None:
            if not isinstance(other_axis_partition, list):
                other_axis_partition = [other_axis_partition]

            # (other_shape[i-1], other_shape[i]) will indicate slice
            # to restore i-1 axis partition
            other_shape = np.cumsum(
                [0] + [len(o.list_of_blocks) for o in other_axis_partition]
            )

            result_partitions = self._wrap_partitions(
                self.deploy_func_between_two_axis_partitions(
                    self.axis,
                    func,
                    num_splits,
                    len(self.list_of_blocks),
                    other_shape,
                    *tuple(
                        self.list_of_blocks
                        + [
                            part
                            for axis_partition in other_axis_partition
                            for part in axis_partition.list_of_blocks
                        ]
                    ),
                    **kwargs,
                )
            )
        args = [self.axis, func, num_splits, maintain_partitioning]
        args.extend(self.list_of_blocks)
        result_partitions = self._wrap_partitions(
            self.deploy_axis_func(*args, **kwargs)
        )
        if self.full_axis:
            return result_partitions
        else:
            # If this is a full axis partition, just take out the single split in the result.
            return result_partitions[0]

    def force_materialization(self, get_ip=False):
        """
        Materialize partitions into a single partition.

        Parameters
        ----------
        get_ip : bool, default: False
            Whether to get node ip address to a single partition or not.

        Returns
        -------
        PandasDataframeAxisPartition
            An axis partition containing only a single materialized partition.
        """
        materialized = super().force_materialization(get_ip=get_ip)
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
        PandasDataframeAxisPartition
            A new ``PandasDataframeAxisPartition`` object,
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
                    obj.length() for obj in self.list_of_partitions_to_combine
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
                    obj.width() for obj in self.list_of_partitions_to_combine
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

        # Copy the original call queue and set it to empty so we don't try to
        # drain it again when we apply().
        call_queue = self.call_queue
        self.call_queue = []

        def drain(df):
            for func, args, kwargs in call_queue:
                df = func(df, *args, **kwargs)
            return df

        drained = self.apply(drain, num_splits=num_splits)
        if not self.full_axis:
            drained = [drained]
        self.list_of_partitions_to_combine = drained

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
        PandasDataframeAxisPartition
            A new ``PandasDataframeAxisPartition`` object.

        Notes
        -----
        The keyword arguments are sent as a dictionary.
        """
        return type(self)(
            self.list_of_partitions_to_combine,
            full_axis=self.full_axis,
            call_queue=self.call_queue + [(func, args, kwargs)],
        )

    @classmethod
    def deploy_axis_func(
        cls, axis, func, num_splits, maintain_partitioning, *partitions, **kwargs
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
            A list of pandas DataFrames.
        """
        # Pop these off first because they aren't expected by the function.
        manual_partition = kwargs.pop("manual_partition", False)
        lengths = kwargs.pop("_lengths", None)

        dataframe = pandas.concat(list(partitions), axis=axis, copy=False)
        # To not mix the args for deploy_axis_func and args for func, we fold
        # args into kwargs. This is a bit of a hack, but it works.
        result = func(dataframe, *kwargs.pop("args", ()), **kwargs)

        if manual_partition:
            # The split function is expecting a list
            lengths = list(lengths)
        # We set lengths to None so we don't use the old lengths for the resulting partition
        # layout. This is done if the number of splits is changing or we are told not to
        # keep the old partitioning.
        elif num_splits != len(partitions) or not maintain_partitioning:
            lengths = None
        else:
            if axis == 0:
                lengths = [len(part) for part in partitions]
                if sum(lengths) != len(result):
                    lengths = None
            else:
                lengths = [len(part.columns) for part in partitions]
                if sum(lengths) != len(result.columns):
                    lengths = None
        return split_result_of_axis_func_pandas(axis, num_splits, result, lengths)

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
            A list of pandas DataFrames.
        """
        lt_frame = pandas.concat(partitions[:len_of_left], axis=axis, copy=False)

        rt_parts = partitions[len_of_left:]

        # reshaping flattened `rt_parts` array into a frame with shape `other_shape`
        combined_axis = [
            pandas.concat(
                rt_parts[other_shape[i - 1] : other_shape[i]],
                axis=axis,
                copy=False,
            )
            for i in range(1, len(other_shape))
        ]
        rt_frame = pandas.concat(combined_axis, axis=axis ^ 1, copy=False)
        # To not mix the args for deploy_func_between_two_axis_partitions and args
        # for func, we fold args into kwargs. This is a bit of a hack, but it works.
        result = func(lt_frame, rt_frame, *kwargs.pop("args", ()), **kwargs)
        return split_result_of_axis_func_pandas(axis, num_splits, result)
