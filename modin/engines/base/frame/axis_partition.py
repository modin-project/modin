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

from abc import ABC
import pandas
import numpy as np
from modin.data_management.utils import split_result_of_axis_func_pandas


class BaseFrameAxisPartition(ABC):  # pragma: no cover
    """An abstract class that represents the Parent class for any `ColumnPartition` or `RowPartition` class.

    This class is intended to simplify the way that operations are performed.

    Note 0: The procedures that use this class and its methods assume that
        they have some global knowledge about the entire axis. This may
        require the implementation to use concatenation or append on the
        list of block partitions in this object.

    Note 1: The `BaseFrameManager` object that controls these objects
        (through the API exposed here) has an invariant that requires that
        this object is never returned from a function. It assumes that
        there will always be `BaseFramePartition` object stored and structures
        itself accordingly.

    The abstract methods that need implemented are `apply` and `shuffle`.
    The children classes must also implement `instance_type` and `partition_type`
    (see below).
    """

    def apply(
        self,
        func,
        num_splits=None,
        other_axis_partition=None,
        maintain_partitioning=True,
        **kwargs,
    ):
        """Apply a function to a full axis.

        Note: The procedures that invoke this method assume full axis
            knowledge. Implement this method accordingly.

        Important: You must return a list of `BaseFramePartition` objects from
            this method. See Note 1 for this class above for more information.

        Args:
            func: The function to apply. This will be preprocessed according to
                the corresponding `RemotePartitions` object.
            num_splits: The number of objects to return, the number of splits
                for the resulting object. It is up to this method to choose the
                splitting at this time.
            other_axis_partition: Another `BaseFrameAxisPartition` object to be applied
                to func. This is for operations that are between datasets.
            maintain_partitioning: Whether or not to keep the partitioning in the same
                orientation as it was previously. This is important because we may be
                operating on an individual AxisPartition and not touching the rest.
                In this case, we have to return the partitioning to its previous
                orientation (the lengths will remain the same). This is ignored between
                two axis partitions.

        Returns
        -------
            A list of `BaseFramePartition` objects.
        """
        pass

    def shuffle(self, func, lengths, **kwargs):
        """Shuffle the order of the data in this axis based on the `lengths`.

        Args:
            func: The function to apply before splitting.
            lengths: The list of partition lengths to split the result into.

        Returns
        -------
            A list of RemotePartition objects split by `lengths`.
        """
        pass

    # Child classes must have these in order to correctly subclass.
    instance_type = None
    partition_type = None

    def _wrap_partitions(self, partitions):
        return [self.partition_type(obj) for obj in partitions]

    def force_materialization(self, get_ip=False):
        """
        Materialize axis partitions into a single partition.

        Parameters
        ----------
        get_ip : boolean, default False
            Whether to get node ip address to a single partition or not.

        Returns
        -------
        BaseFrameAxisPartition
            An axis partition containing only a single materialized partition.
        """
        materialized = self.apply(
            lambda x: x, num_splits=1, maintain_partitioning=False
        )
        return type(self)(materialized, get_ip=get_ip)

    def unwrap(self, squeeze=False, get_ip=False):
        """
        Unwrap partitions from axis partition.

        Parameters
        ----------
        squeeze : boolean, default False
            The flag used to unwrap only one partition.
        get_ip : boolean, default False
            Whether to get node ip address to each partition or not.

        Returns
        -------
        list
            A list of partitions from axis partition.

        Notes
        -----
        If `get_ip=True`, a list of tuples of Ray.ObjectRef/Dask.Future to node ip addresses and
        unwrapped partitions, respectively, is returned if Ray/Dask is used as an engine
        (i.e. [(Ray.ObjectRef/Dask.Future, Ray.ObjectRef/Dask.Future), ...]).
        """
        if squeeze and len(self.list_of_blocks) == 1:
            if get_ip:
                return self.list_of_ips[0], self.list_of_blocks[0]
            else:
                return self.list_of_blocks[0]
        else:
            if get_ip:
                return list(zip(self.list_of_ips, self.list_of_blocks))
            else:
                return self.list_of_blocks


class PandasFrameAxisPartition(BaseFrameAxisPartition):
    """An abstract class is created to simplify and consolidate the code for AxisPartitions that run pandas.

    Because much of the code is similar, this allows us to reuse this code.

    Subclasses must implement `list_of_blocks` which unwraps the `RemotePartition`
    objects and creates something interpretable as a pandas DataFrame.

    See `modin.engines.ray.pandas_on_ray.axis_partition.PandasOnRayFrameAxisPartition`
    for an example on how to override/use this class when the implementation needs
    to be augmented.
    """

    def apply(
        self,
        func,
        num_splits=None,
        other_axis_partition=None,
        maintain_partitioning=True,
        **kwargs,
    ):
        """Apply func to the object in the plasma store.

        See notes in Parent class about this method.

        Parameters
        ----------
        func: callable
            The function to apply.
        num_splits: int
            The number of times to split the result object.
        other_axis_partition: PandasOnRayFrameAxisPartition object
            Another `PandasOnRayFrameAxisPartition` object to apply to func with this one.
        maintain_partitioning: boolean
            Whether or not to keep the partitioning in the same
            orientation as it was previously. This is important because we may be
            operating on an individual AxisPartition and not touching the rest.
            In this case, we have to return the partitioning to its previous
            orientation (the lengths will remain the same). This is ignored between
            two axis partitions.

        Returns
        -------
            A list of `RayRemotePartition` objects.
        """
        if num_splits is None:
            num_splits = len(self.list_of_blocks)

        if other_axis_partition is not None:
            if not isinstance(other_axis_partition, list):
                other_axis_partition = [other_axis_partition]

            # (other_shape[i-1], other_shape[i]) will indicate slice
            # to restore i-1 axis partition
            other_shape = np.cumsum(
                [0] + [len(o.list_of_blocks) for o in other_axis_partition]
            )

            return self._wrap_partitions(
                self.deploy_func_between_two_axis_partitions(
                    self.axis,
                    func,
                    num_splits,
                    len(self.list_of_blocks),
                    other_shape,
                    kwargs,
                    *tuple(
                        self.list_of_blocks
                        + [
                            part
                            for axis_partition in other_axis_partition
                            for part in axis_partition.list_of_blocks
                        ]
                    ),
                )
            )
        args = [self.axis, func, num_splits, kwargs, maintain_partitioning]
        args.extend(self.list_of_blocks)
        return self._wrap_partitions(self.deploy_axis_func(*args))

    def shuffle(self, func, lengths, **kwargs):
        """Shuffle the order of the data in this axis based on the `lengths`.

        Extends `BaseFrameAxisPartition.shuffle`.

        Args:
            func: The function to apply before splitting.
            lengths: The list of partition lengths to split the result into.

        Returns
        -------
            A list of RemotePartition objects split by `lengths`.
        """
        num_splits = len(lengths)
        # We add these to kwargs and will pop them off before performing the operation.
        kwargs["manual_partition"] = True
        kwargs["_lengths"] = lengths
        args = [self.axis, func, num_splits, kwargs, False]
        args.extend(self.list_of_blocks)
        return self._wrap_partitions(self.deploy_axis_func(*args))

    @classmethod
    def deploy_axis_func(
        cls, axis, func, num_splits, kwargs, maintain_partitioning, *partitions
    ):
        """Deploy a function along a full axis in Ray.

        Args:
            axis: The axis to perform the function along.
            func: The function to perform.
            num_splits: The number of splits to return
                (see `split_result_of_axis_func_pandas`)
            kwargs: A dictionary of keyword arguments.
            maintain_partitioning: If True, keep the old partitioning if possible.
                If False, create a new partition layout.
            partitions: All partitions that make up the full axis (row or column)

        Returns
        -------
            A list of Pandas DataFrames.
        """
        # Pop these off first because they aren't expected by the function.
        manual_partition = kwargs.pop("manual_partition", False)
        lengths = kwargs.pop("_lengths", None)

        dataframe = pandas.concat(list(partitions), axis=axis, copy=False)
        result = func(dataframe, **kwargs)

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
        cls, axis, func, num_splits, len_of_left, other_shape, kwargs, *partitions
    ):
        """Deploy a function along a full axis between two data sets in Ray.

        Parameters
        ----------
            axis: The axis to perform the function along.
            func: The function to perform.
            num_splits: The number of splits to return
                (see `split_result_of_axis_func_pandas`).
            len_of_left: The number of values in `partitions` that belong to the
                left data set.
            other_shape: The shape of right frame in terms of partitions
            kwargs: A dictionary of keyword arguments.
            partitions: All partitions that make up the full axis (row or column)
                for both data sets.

        Returns
        -------
            A list of Pandas DataFrames.
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

        result = func(lt_frame, rt_frame, **kwargs)
        return split_result_of_axis_func_pandas(axis, num_splits, result)
