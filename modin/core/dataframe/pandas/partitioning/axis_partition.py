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

import warnings

import numpy as np
import pandas

from modin.config import MinColumnPartitionSize, MinRowPartitionSize
from modin.core.dataframe.base.partitioning.axis_partition import (
    BaseDataframeAxisPartition,
)
from modin.core.storage_formats.pandas.utils import (
    generate_result_of_axis_func_pandas,
    split_result_of_axis_func_pandas,
)

from .partition import PandasDataframePartition


class PandasDataframeAxisPartition(BaseDataframeAxisPartition):
    """
    An abstract class is created to simplify and consolidate the code for axis partition that run pandas.

    Because much of the code is similar, this allows us to reuse this code.

    Parameters
    ----------
    list_of_partitions : Union[list, PandasDataframePartition]
        List of ``PandasDataframePartition`` and
        ``PandasDataframeAxisPartition`` objects, or a single
        ``PandasDataframePartition``.
    get_ip : bool, default: False
        Whether to get node IP addresses to conforming partitions or not.
    full_axis : bool, default: True
        Whether or not the axis partition encompasses the whole axis.
    call_queue : list, optional
        A list of tuples (callable, args, kwargs) that contains deferred calls.
    length : the future's type or int, optional
        Length, or reference to length, of wrapped ``pandas.DataFrame``.
    width : the future's type or int, optional
        Width, or reference to width, of wrapped ``pandas.DataFrame``.
    """

    def __init__(
        self,
        list_of_partitions,
        get_ip=False,
        full_axis=True,
        call_queue=None,
        length=None,
        width=None,
    ):
        if isinstance(list_of_partitions, PandasDataframePartition):
            list_of_partitions = [list_of_partitions]
        self.full_axis = full_axis
        self.call_queue = call_queue or []
        self._length_cache = length
        self._width_cache = width
        # Check that all axis partition axes are the same in `list_of_partitions`
        # We should never have mismatching axis in the current implementation. We add this
        # defensive assertion to ensure that undefined behavior does not happen.
        assert (
            len(
                set(
                    obj.axis
                    for obj in list_of_partitions
                    if isinstance(obj, PandasDataframeAxisPartition)
                )
            )
            <= 1
        )
        self._list_of_constituent_partitions = list_of_partitions
        # Defer computing _list_of_block_partitions because we might need to
        # drain call queues for that.
        self._list_of_block_partitions = None

    @property
    def list_of_blocks(self):
        """
        Get the list of physical partition objects that compose this partition.

        Returns
        -------
        list
            A list of physical partition objects (``ray.ObjectRef``, ``distributed.Future`` e.g.).
        """
        # Defer draining call queue (which is hidden in `partition.list_of_blocks` call) until we get the partitions.
        # TODO Look into draining call queue at the same time as the task
        return [
            partition.list_of_blocks[0] for partition in self.list_of_block_partitions
        ]

    @property
    def list_of_block_partitions(self) -> list:
        """
        Get the list of block partitions that compose this partition.

        Returns
        -------
        List
            A list of ``PandasDataframePartition``.
        """
        if self._list_of_block_partitions is not None:
            return self._list_of_block_partitions
        self._list_of_block_partitions = []
        # Extract block partitions from the block and axis partitions that
        # constitute this partition.
        for partition in self._list_of_constituent_partitions:
            if isinstance(partition, PandasDataframeAxisPartition):
                if partition.axis == self.axis:
                    # We are building an axis partition out of another
                    # axis partition `partition` that contains its own list
                    # of block partitions, partition.list_of_block_partitions.
                    # `partition` may have its own call queue, which has to be
                    # applied to the entire `partition` before we execute any
                    # further operations on its block parittions.
                    partition.drain_call_queue()
                    self._list_of_block_partitions.extend(
                        partition.list_of_block_partitions
                    )
                else:
                    # If this axis partition is made of axis partitions
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

    @classmethod
    def _get_drain_func(cls):  # noqa: GL08
        return PandasDataframeAxisPartition.drain

    def drain_call_queue(self, num_splits=None):
        """
        Execute all operations stored in this partition's call queue.

        Parameters
        ----------
        num_splits : int, default: None
            The number of times to split the result object.
        """
        if len(self.call_queue) == 0:
            # this implicitly calls `drain_call_queue` for block partitions,
            # which might have deferred call queues
            _ = self.list_of_blocks
            return
        call_queue = self.call_queue
        try:
            # Clearing the queue before calling `.apply()` so it won't try to drain it repeatedly
            self.call_queue = []
            drained = self.apply(
                self._get_drain_func(), num_splits=num_splits, call_queue=call_queue
            )
        except Exception:
            # Restoring the call queue in case of an exception as it most likely wasn't drained
            self.call_queue = call_queue
            raise
        if not isinstance(drained, list):
            drained = [drained]
        self._list_of_block_partitions = drained

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
        self._list_of_block_partitions = materialized.list_of_block_partitions
        return materialized

    def apply(
        self,
        func,
        *args,
        num_splits=None,
        other_axis_partition=None,
        maintain_partitioning=True,
        lengths=None,
        manual_partition=False,
        **kwargs,
    ):
        """
        Apply a function to this axis partition along full axis.

        Parameters
        ----------
        func : callable
            The function to apply.
        *args : iterable
            Positional arguments to pass to `func`.
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
        lengths : iterable, default: None
            The list of lengths to shuffle the object.
        manual_partition : bool, default: False
            If True, partition the result with `lengths`.
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
        if len(self.call_queue) > 0:
            self.drain_call_queue()

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
                    args,
                    kwargs,
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
                    min_block_size=(
                        MinRowPartitionSize.get()
                        if self.axis == 0
                        else MinColumnPartitionSize.get()
                    ),
                )
            )
        result = self._wrap_partitions(
            self.deploy_axis_func(
                self.axis,
                func,
                args,
                kwargs,
                num_splits,
                maintain_partitioning,
                *self.list_of_blocks,
                min_block_size=(
                    MinRowPartitionSize.get()
                    if self.axis == 0
                    else MinColumnPartitionSize.get()
                ),
                lengths=lengths,
                manual_partition=manual_partition,
            )
        )
        if self.full_axis:
            return result
        else:
            # If this is not a full axis partition, just take out the single split in the result.
            return result[0]

    def split(
        self, split_func, num_splits, f_args=None, f_kwargs=None, extract_metadata=False
    ):
        """
        Split axis partition into multiple partitions using the `split_func`.

        Parameters
        ----------
        split_func : callable(pandas.DataFrame) -> list[pandas.DataFrame]
            A function that takes partition's content and split it into multiple chunks.
        num_splits : int
            The number of splits the `split_func` return.
        f_args : iterable, optional
            Positional arguments to pass to the `split_func`.
        f_kwargs : dict, optional
            Keyword arguments to pass to the `split_func`.
        extract_metadata : bool, default: False
            Whether to return metadata (length, width, ip) of the result. Passing `False` may relax
            the load on object storage as the remote function would return X times fewer futures
            (where X is the number of metadata values). Passing `False` makes sense for temporary
            results where you know for sure that the metadata will never be requested.

        Returns
        -------
        list
            List of wrapped remote partition objects.
        """
        f_args = tuple() if f_args is None else f_args
        f_kwargs = {} if f_kwargs is None else f_kwargs
        return self._wrap_partitions(
            self.deploy_splitting_func(
                self.axis,
                split_func,
                f_args,
                f_kwargs,
                num_splits,
                *self.list_of_blocks,
                extract_metadata=extract_metadata,
            ),
            extract_metadata=extract_metadata,
        )

    @classmethod
    def deploy_splitting_func(
        cls,
        axis,
        split_func,
        f_args,
        f_kwargs,
        num_splits,
        *partitions,
        extract_metadata=False,
    ):
        """
        Deploy a splitting function along a full axis.

        Parameters
        ----------
        axis : {0, 1}
            The axis to perform the function along.
        split_func : callable(pandas.DataFrame) -> list[pandas.DataFrame]
            The function to perform.
        f_args : list or tuple
            Positional arguments to pass to `split_func`.
        f_kwargs : dict
            Keyword arguments to pass to `split_func`.
        num_splits : int
            The number of splits the `split_func` return.
        *partitions : iterable
            All partitions that make up the full axis (row or column).
        extract_metadata : bool, default: False
            Whether to return metadata (length, width, ip) of the result. Note that `True` value
            is not supported in `PandasDataframeAxisPartition` class.

        Returns
        -------
        list
            A list of pandas DataFrames.
        """
        dataframe = pandas.concat(list(partitions), axis=axis, copy=False)
        # to reduce peak memory consumption
        del partitions
        return split_func(dataframe, *f_args, **f_kwargs)

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
        min_block_size,
        lengths=None,
        manual_partition=False,
        return_generator=False,
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
        min_block_size : int
            Minimum number of rows/columns in a single split.
        lengths : list, optional
            The list of lengths to shuffle the object.
        manual_partition : bool, default: False
            If True, partition the result with `lengths`.
        return_generator : bool, default: False
            Return a generator from the function, set to `True` for Ray backend
            as Ray remote functions can return Generators.

        Returns
        -------
        list | Generator
            A list or generator of pandas DataFrames.
        """
        len_partitions = len(partitions)
        lengths_partitions = [len(part) for part in partitions]
        widths_partitions = [len(part.columns) for part in partitions]

        dataframe = pandas.concat(list(partitions), axis=axis, copy=False)

        # to reduce peak memory consumption
        del partitions

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            try:
                result = func(dataframe, *f_args, **f_kwargs)
            except ValueError as err:
                if "assignment destination is read-only" in str(err):
                    result = func(dataframe.copy(), *f_args, **f_kwargs)
                else:
                    raise err

        # to reduce peak memory consumption
        del dataframe

        if num_splits == 1:
            # If we're not going to split the result, we don't need to specify
            # split lengths.
            lengths = None
        elif manual_partition:
            # The split function is expecting a list
            lengths = list(lengths)
        # We set lengths to None so we don't use the old lengths for the resulting partition
        # layout. This is done if the number of splits is changing or we are told not to
        # keep the old partitioning.
        elif num_splits != len_partitions or not maintain_partitioning:
            lengths = None
        else:
            if axis == 0:
                lengths = lengths_partitions
                if sum(lengths) != len(result):
                    lengths = None
            else:
                lengths = widths_partitions
                if sum(lengths) != len(result.columns):
                    lengths = None
        if return_generator:
            return generate_result_of_axis_func_pandas(
                axis,
                num_splits,
                result,
                min_block_size,
                lengths,
            )
        else:
            return split_result_of_axis_func_pandas(
                axis, num_splits, result, min_block_size, lengths
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
        min_block_size,
        return_generator=False,
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
        min_block_size : int
            Minimum number of rows/columns in a single split.
        return_generator : bool, default: False
            Return a generator from the function, set to `True` for Ray backend
            as Ray remote functions can return Generators.

        Returns
        -------
        list | Generator
            A list or generator of pandas DataFrames.
        """
        lt_frame = pandas.concat(partitions[:len_of_left], axis=axis, copy=False)

        rt_parts = partitions[len_of_left:]

        # to reduce peak memory consumption
        del partitions

        # reshaping flattened `rt_parts` array into a frame with shape `other_shape`
        combined_axis = [
            pandas.concat(
                rt_parts[other_shape[i - 1] : other_shape[i]],
                axis=axis,
                copy=False,
            )
            for i in range(1, len(other_shape))
        ]

        # to reduce peak memory consumption
        del rt_parts

        rt_frame = pandas.concat(combined_axis, axis=axis ^ 1, copy=False)

        # to reduce peak memory consumption
        del combined_axis

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            result = func(lt_frame, rt_frame, *f_args, **f_kwargs)

        # to reduce peak memory consumption
        del lt_frame, rt_frame

        if return_generator:
            return generate_result_of_axis_func_pandas(
                axis,
                num_splits,
                result,
                min_block_size,
            )
        else:
            return split_result_of_axis_func_pandas(
                axis,
                num_splits,
                result,
                min_block_size,
            )

    @classmethod
    def drain(cls, df: pandas.DataFrame, call_queue: list):
        """
        Execute all operations stored in the call queue on the pandas object (helper function).

        Parameters
        ----------
        df : pandas.DataFrame
        call_queue : list
            Call queue that needs to be executed on pandas DataFrame.

        Returns
        -------
        pandas.DataFrame
        """
        for func, args, kwargs in call_queue:
            df = func(df, *args, **kwargs)
        return df

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
            A new ``PandasDataframeAxisPartition`` object, materialized.
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

    def to_numpy(self):
        """
        Convert the data in this partition to a ``numpy.array``.

        Returns
        -------
        NumPy array.
        """
        return self.force_materialization().list_of_block_partitions[0].to_numpy()

    _length_cache = None

    def length(self, materialize=True):
        """
        Get the length of this partition.

        Parameters
        ----------
        materialize : bool, default: True
            Whether to forcibly materialize the result into an integer. If ``False``
            was specified, may return a future of the result if it hasn't been
            materialized yet.

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
                self._length_cache = self.list_of_block_partitions[0].length(
                    materialize
                )
        return self._length_cache

    _width_cache = None

    def width(self, materialize=True):
        """
        Get the width of this partition.

        Parameters
        ----------
        materialize : bool, default: True
            Whether to forcibly materialize the result into an integer. If ``False``
            was specified, may return a future of the result if it hasn't been
            materialized yet.

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
                self._width_cache = self.list_of_block_partitions[0].width(materialize)
        return self._width_cache

    def wait(self):
        """Wait completing computations on the object wrapped by the partition."""
        pass

    def add_to_apply_calls(self, func, *args, length=None, width=None, **kwargs):
        """
        Add a function to the call queue.

        Parameters
        ----------
        func : callable or a future type
            Function to be added to the call queue.
        *args : iterable
            Additional positional arguments to be passed in `func`.
        length : A future type or int, optional
            Length, or reference to it, of wrapped ``pandas.DataFrame``.
        width : A future type or int, optional
            Width, or reference to it, of wrapped ``pandas.DataFrame``.
        **kwargs : dict
            Additional keyword arguments to be passed in `func`.

        Returns
        -------
        PandasDataframeAxisPartition
            A new ``PandasDataframeAxisPartition`` object.
        """
        return type(self)(
            self.list_of_block_partitions,
            full_axis=self.full_axis,
            call_queue=self.call_queue + [[func, args, kwargs]],
            length=length,
            width=width,
        )
