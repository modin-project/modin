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
    """
    An abstract class is created to simplify and consolidate the code for axis partition that run pandas.

    Because much of the code is similar, this allows us to reuse this code.
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
            A list of `PandasDataframePartition` objects.
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
        return self._wrap_partitions(self.deploy_axis_func(*args, **kwargs))

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
            A list of `PandasDataframePartition` objects split by `lengths`.
        """
        num_splits = len(lengths)
        # We add these to kwargs and will pop them off before performing the operation.
        kwargs["manual_partition"] = True
        kwargs["_lengths"] = lengths
        args = [self.axis, func, num_splits, False]
        args.extend(self.list_of_blocks)
        return self._wrap_partitions(self.deploy_axis_func(*args, **kwargs))

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
