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

from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Optional, Tuple, Type, Union

from modin.logging import ClassLogger
from modin.logging.config import LogLevel


class BaseDataframeAxisPartition(
    ABC, ClassLogger, modin_layer="VIRTUAL-PARTITION", log_level=LogLevel.DEBUG
):  # pragma: no cover
    """
    An abstract class that represents the parent class for any axis partition class.

    This class is intended to simplify the way that operations are performed.

    Attributes
    ----------
    _PARTITIONS_METADATA_LEN : int
        The number of metadata values that the object of `partition_type` consumes.
    """

    @property
    @abstractmethod
    def list_of_blocks(self) -> list:
        """Get the list of physical partition objects that compose this partition."""
        pass

    def apply(
        self,
        func: Callable,
        *args: Iterable,
        num_splits: Optional[int] = None,
        other_axis_partition: Optional["BaseDataframeAxisPartition"] = None,
        maintain_partitioning: bool = True,
        lengths: Optional[Iterable] = None,
        manual_partition: bool = False,
        **kwargs: dict,
    ) -> Any:
        """
        Apply a function to this axis partition along full axis.

        Parameters
        ----------
        func : callable
            The function to apply. This will be preprocessed according to
            the corresponding `BaseDataframePartition` objects.
        *args : iterable
            Positional arguments to pass to `func`.
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
        lengths : iterable, default: None
            The list of lengths to shuffle the partition into.
        manual_partition : bool, default: False
            If True, partition the result with `lengths`.
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

    # Child classes must have these in order to correctly subclass.
    partition_type: Type
    _PARTITIONS_METADATA_LEN = 0

    def _wrap_partitions(
        self, partitions: list, extract_metadata: Optional[bool] = None
    ) -> list:
        """
        Wrap remote partition objects with `BaseDataframePartition` class.

        Parameters
        ----------
        partitions : list
            List of remotes partition objects to be wrapped with `BaseDataframePartition` class.
        extract_metadata : bool, optional
            Whether the partitions list contains information about partition's metadata.
            If `None` was passed will take the argument's value from the value of `cls._PARTITIONS_METADATA_LEN`.

        Returns
        -------
        list
            List of wrapped remote partition objects.
        """
        assert self.partition_type is not None

        if extract_metadata is None:
            # If `_PARTITIONS_METADATA_LEN == 0` then the execution doesn't support metadata
            # and thus we should never try extracting it, otherwise assuming that the common
            # approach of always passing the metadata is used.
            extract_metadata = bool(self._PARTITIONS_METADATA_LEN)

        if extract_metadata:
            # Here we recieve a 1D array of futures describing partitions and their metadata as:
            # [object_id{partition_idx}, metadata{partition_idx}_{metadata_idx}, ...]
            # Here's an example of such array:
            # [
            #  object_id1, metadata1_1, metadata1_2, ..., metadata1_PARTITIONS_METADATA_LEN,
            #  object_id2, metadata2_1, ..., metadata2_PARTITIONS_METADATA_LEN,
            #  ...
            #  object_idN, metadataN_1, ..., metadataN_PARTITIONS_METADATA_LEN,
            # ]
            return [
                self.partition_type(*init_args)
                for init_args in zip(
                    # `partition_type` consumes `(object_id, *metadata)`, thus adding `+1`
                    *[iter(partitions)]
                    * (1 + self._PARTITIONS_METADATA_LEN)
                )
            ]
        else:
            return [self.partition_type(object_id) for object_id in partitions]

    def force_materialization(
        self, get_ip: bool = False
    ) -> "BaseDataframeAxisPartition":
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
        return type(self)(materialized, get_ip=get_ip)  # type: ignore[call-arg]

    def unwrap(
        self, squeeze: bool = False, get_ip: bool = False
    ) -> Union[list, Tuple[list, list]]:
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
        If `get_ip=True`, a tuple of lists of Ray.ObjectRef/Dask.Future to node ip addresses and
        unwrapped partitions, respectively, is returned if Ray/Dask is used as an engine
        (i.e. [(Ray.ObjectRef/Dask.Future, Ray.ObjectRef/Dask.Future), ...]).
        """
        if squeeze and len(self.list_of_blocks) == 1:
            if get_ip:
                # TODO(https://github.com/modin-project/modin/issues/5176): Stop ignoring the list_of_ips
                # check once we know that we're not calling list_of_ips on python axis partitions
                return self.list_of_ips[0], self.list_of_blocks[0]  # type: ignore[attr-defined]
            else:
                return self.list_of_blocks[0]
        else:
            if get_ip:
                return list(zip(self.list_of_ips, self.list_of_blocks))  # type: ignore[attr-defined]
            else:
                return self.list_of_blocks
