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

"""Base class of an axis partition for a Modin Dataframe."""

from abc import ABC


class BaseDataframeAxisPartition(ABC):  # pragma: no cover
    """
    An abstract class that represents the parent class for any axis partition class.

    This class is intended to simplify the way that operations are performed.
    """

    def apply(
        self,
        func,
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
            The function to apply. This will be preprocessed according to
            the corresponding `BaseDataframePartition` objects.
        num_splits : int, default: None
            The number of times to split the result object.
        other_axis_partition : BaseDataframeAxisPartition, default: None
            Another `BaseDataframeAxisPartition` object to be applied
            to func. This is for operations that are between two data sets.
        maintain_partitioning : bool, default: True
            Whether to keep the partitioning in the same
            orientation as it was previously or not. This is important because we may be
            operating on an individual axis partition and not touching the rest.
            In this case, we have to return the partitioning to its previous
            orientation (the lengths will remain the same). This is ignored between
            two axis partitions.
        **kwargs : dict
            Additional keywords arguments to be passed in `func`.

        Returns
        -------
        list
            A list of `BaseDataframePartition` objects.

        Notes
        -----
        The procedures that invoke this method assume full axis
        knowledge. Implement this method accordingly.

        You must return a list of `BaseDataframePartition` objects from this method.
        """
        pass

    def shuffle(self, func, lengths, **kwargs):
        """
        Shuffle the order of the data in this axis partition based on the `lengths`.

        Parameters
        ----------
        func : callable
            The function to apply before splitting.
        lengths : list
            The list of partition lengths to split the result into.
        **kwargs : dict
            Additional keywords arguments to be passed in `func`.

        Returns
        -------
        list
            A list of `BaseDataframePartition` objects split by `lengths`.
        """
        pass

    # Child classes must have these in order to correctly subclass.
    instance_type = None
    partition_type = None

    def _wrap_partitions(self, partitions):
        """
        Wrap remote partition objects with `BaseDataframePartition` class.

        Parameters
        ----------
        partitions : list
            List of remotes partition objects to be wrapped with `BaseDataframePartition` class.

        Returns
        -------
        list
            List of wrapped remote partition objects.
        """
        return [self.partition_type(obj) for obj in partitions]

    def force_materialization(self, get_ip=False):
        """
        Materialize axis partitions into a single partition.

        Parameters
        ----------
        get_ip : bool, default: False
            Whether to get node ip address to a single partition or not.

        Returns
        -------
        BaseDataframeAxisPartition
            An axis partition containing only a single materialized partition.
        """
        materialized = self.apply(
            lambda x: x, num_splits=1, maintain_partitioning=False
        )
        return type(self)(materialized, get_ip=get_ip)

    def unwrap(self, squeeze=False, get_ip=False):
        """
        Unwrap partitions from this axis partition.

        Parameters
        ----------
        squeeze : bool, default: False
            Flag used to unwrap only one partition.
        get_ip : bool, default: False
            Whether to get node ip address to each partition or not.

        Returns
        -------
        list
            List of partitions from this axis partition.

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
