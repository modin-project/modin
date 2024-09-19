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

"""
Module holding base PartitionManager class - the thing that tracks partitions across the distribution.

The manager also allows manipulating the data - running functions at each partition, shuffle over the distribution, etc.
"""

import os
import warnings
from abc import ABC
from functools import wraps
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas
from pandas._libs.lib import no_default

from modin.config import (
    BenchmarkMode,
    CpuCount,
    DynamicPartitioning,
    Engine,
    MinColumnPartitionSize,
    MinRowPartitionSize,
    NPartitions,
    PersistentPickle,
    ProgressBar,
)
from modin.core.dataframe.pandas.utils import create_pandas_df_from_partitions
from modin.core.storage_formats.pandas.utils import compute_chunksize
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from modin.logging.config import LogLevel
from modin.pandas.utils import get_pandas_backend

if TYPE_CHECKING:
    from modin.core.dataframe.pandas.dataframe.utils import ShuffleFunctions


def wait_computations_if_benchmark_mode(func):
    """
    Make sure a `func` finished its computations in benchmark mode.

    Parameters
    ----------
    func : callable
        A function that should be performed in syncronous mode.

    Returns
    -------
    callable
        Wrapped function that executes eagerly (if benchmark mode) or original `func`.

    Notes
    -----
    `func` should return NumPy array with partitions.
    """

    @wraps(func)
    def wait(cls, *args, **kwargs):
        """Wait for computation results."""
        result = func(cls, *args, **kwargs)
        if BenchmarkMode.get():
            if isinstance(result, tuple):
                partitions = result[0]
            else:
                partitions = result
            # When partitions have a deferred call queue, calling
            # partition.wait() on each partition serially will serially kick
            # off each deferred computation and wait for each partition to
            # finish before kicking off the next one. Instead, we want to
            # serially kick off all the deferred computations so that they can
            # all run asynchronously, then wait on all the results.
            cls.finalize(partitions)
            # The partition manager invokes the relevant .wait() method under
            # the hood, which should wait in parallel for all computations to finish
            cls.wait_partitions(partitions.flatten())
        return result

    return wait


class PandasDataframePartitionManager(
    ClassLogger, ABC, modin_layer="PARTITION-MANAGER", log_level=LogLevel.DEBUG
):
    """
    Base class for managing the dataframe data layout and operators across the distribution of partitions.

    Partition class is the class to use for storing each partition.
    Each partition must extend the `PandasDataframePartition` class.
    """

    _partition_class = None
    # Column partitions class is the class to use to create the column partitions.
    _column_partitions_class = None
    # Row partitions class is the class to use to create the row partitions.
    _row_partition_class = None
    _execution_wrapper = None

    @classmethod
    def materialize_futures(cls, input_list):
        """
        Materialize all futures in the input list.

        Parameters
        ----------
        input_list : list
            The list that has to be manipulated.

        Returns
        -------
        list
           A new list with materialized objects.
        """
        # Do nothing if input_list is None or [].
        if input_list is None:
            return None
        filtered_list = []
        filtered_idx = []
        for idx, item in enumerate(input_list):
            if cls._execution_wrapper.is_future(item):
                filtered_idx.append(idx)
                filtered_list.append(item)
        filtered_list = cls._execution_wrapper.materialize(filtered_list)
        result = input_list.copy()
        for idx, item in zip(filtered_idx, filtered_list):
            result[idx] = item
        return result

    @classmethod
    def preprocess_func(cls, map_func):
        """
        Preprocess a function to be applied to `PandasDataframePartition` objects.

        Parameters
        ----------
        map_func : callable
            The function to be preprocessed.

        Returns
        -------
        callable
            The preprocessed version of the `map_func` provided.

        Notes
        -----
        Preprocessing does not require any specific format, only that the
        `PandasDataframePartition.apply` method will recognize it (for the subclass
        being used).

        If your `PandasDataframePartition` objects assume that a function provided
        is serialized or wrapped or in some other format, this is the place
        to add that logic. It is possible that this can also just return
        `map_func` if the `apply` method of the `PandasDataframePartition` object
        you are using does not require any modification to a given function.
        """
        if cls._execution_wrapper.is_future(map_func):
            return map_func  # Has already been preprocessed

        old_value = PersistentPickle.get()
        # When performing a function with Modin objects, it is more profitable to
        # do the conversion to pandas once on the main process than several times
        # on worker processes. Details: https://github.com/modin-project/modin/pull/6673/files#r1391086755
        # For Dask, otherwise there may be an error: `coroutine 'Client._gather' was never awaited`
        need_update = not PersistentPickle.get() and Engine.get() != "Dask"
        if need_update:
            PersistentPickle.put(True)
        try:
            result = cls._partition_class.preprocess_func(map_func)
        finally:
            if need_update:
                PersistentPickle.put(old_value)
        return result

    # END Abstract Methods

    @classmethod
    def create_partition_from_metadata(
        cls, dtypes: Optional[pandas.Series] = None, **metadata
    ):
        """
        Create NumPy array of partitions that holds an empty dataframe with given metadata.

        Parameters
        ----------
        dtypes : pandas.Series, optional
            Column dtypes.
            Upon creating a pandas DataFrame from `metadata` we call `astype` since
            pandas doesn't allow to pass a list of dtypes directly in the constructor.
        **metadata : dict
            Metadata that has to be wrapped in a partition.

        Returns
        -------
        np.ndarray
            A NumPy 2D array of a single partition which contains the data.
        """
        metadata_dataframe = pandas.DataFrame(**metadata)
        if dtypes is not None:
            metadata_dataframe = metadata_dataframe.astype(dtypes)
        return np.array([[cls._partition_class.put(metadata_dataframe)]])

    @classmethod
    def column_partitions(cls, partitions, full_axis=True):
        """
        Get the list of `BaseDataframeAxisPartition` objects representing column-wise partitions.

        Parameters
        ----------
        partitions : list-like
            List of (smaller) partitions to be combined to column-wise partitions.
        full_axis : bool, default: True
            Whether or not this partition contains the entire column axis.

        Returns
        -------
        list
            A list of `BaseDataframeAxisPartition` objects.

        Notes
        -----
        Each value in this list will be an `BaseDataframeAxisPartition` object.
        `BaseDataframeAxisPartition` is located in `axis_partition.py`.
        """
        if not isinstance(partitions, list):
            partitions = [partitions]
        return [
            cls._column_partitions_class(col, full_axis=full_axis)
            for frame in partitions
            for col in frame.T
        ]

    @classmethod
    def row_partitions(cls, partitions):
        """
        List of `BaseDataframeAxisPartition` objects representing row-wise partitions.

        Parameters
        ----------
        partitions : list-like
            List of (smaller) partitions to be combined to row-wise partitions.

        Returns
        -------
        list
            A list of `BaseDataframeAxisPartition` objects.

        Notes
        -----
        Each value in this list will an `BaseDataframeAxisPartition` object.
        `BaseDataframeAxisPartition` is located in `axis_partition.py`.
        """
        if not isinstance(partitions, list):
            partitions = [partitions]
        return [cls._row_partition_class(row) for frame in partitions for row in frame]

    @classmethod
    def axis_partition(cls, partitions, axis, full_axis: bool = True):
        """
        Logically partition along given axis (columns or rows).

        Parameters
        ----------
        partitions : list-like
            List of partitions to be combined.
        axis : {0, 1}
            0 for column partitions, 1 for row partitions.
        full_axis : bool, default: True
            Whether or not this partition contains the entire column axis.

        Returns
        -------
        list
            A list of `BaseDataframeAxisPartition` objects.
        """
        make_column_partitions = axis == 0
        if not full_axis and not make_column_partitions:
            raise NotImplementedError(
                (
                    "Row partitions must contain the entire axis. We don't "
                    + "support virtual partitioning for row partitions yet."
                )
            )
        return (
            cls.column_partitions(partitions)
            if make_column_partitions
            else cls.row_partitions(partitions)
        )

    @classmethod
    def groupby_reduce(
        cls, axis, partitions, by, map_func, reduce_func, apply_indices=None
    ):
        """
        Groupby data using the `map_func` provided along the `axis` over the `partitions` then reduce using `reduce_func`.

        Parameters
        ----------
        axis : {0, 1}
            Axis to groupby over.
        partitions : NumPy 2D array
            Partitions of the ModinFrame to groupby.
        by : NumPy 2D array
            Partitions of 'by' to broadcast.
        map_func : callable
            Map function.
        reduce_func : callable,
            Reduce function.
        apply_indices : list of ints, default: None
            Indices of `axis ^ 1` to apply function over.

        Returns
        -------
        NumPy array
            Partitions with applied groupby.
        """
        if apply_indices is not None:
            partitions = (
                partitions[apply_indices] if axis else partitions[:, apply_indices]
            )

        if by is not None:
            # need to make sure that the partitioning of the following objects
            # coincides in the required axis, because `partition_manager.broadcast_apply`
            # doesn't call `_copartition` unlike `modin_frame.broadcast_apply`
            assert partitions.shape[axis] == by.shape[axis], (
                f"the number of partitions along {axis=} is not equal: "
                + f"{partitions.shape[axis]} != {by.shape[axis]}"
            )
            mapped_partitions = cls.broadcast_apply(
                axis, map_func, left=partitions, right=by
            )
        else:
            mapped_partitions = cls.map_partitions(partitions, map_func)

        # Assuming, that the output will not be larger than the input,
        # keep the current number of partitions.
        num_splits = min(len(partitions), NPartitions.get())
        return cls.map_axis_partitions(
            axis,
            mapped_partitions,
            reduce_func,
            enumerate_partitions=True,
            num_splits=num_splits,
        )

    @classmethod
    @wait_computations_if_benchmark_mode
    def broadcast_apply_select_indices(
        cls,
        axis,
        apply_func,
        left,
        right,
        left_indices,
        right_indices,
        keep_remaining=False,
    ):
        """
        Broadcast the `right` partitions to `left` and apply `apply_func` to selected indices.

        Parameters
        ----------
        axis : {0, 1}
            Axis to apply and broadcast over.
        apply_func : callable
            Function to apply.
        left : NumPy 2D array
            Left partitions.
        right : NumPy 2D array
            Right partitions.
        left_indices : list-like
            Indices to apply function to.
        right_indices : dictionary of indices of right partitions
            Indices that you want to bring at specified left partition, for example
            dict {key: {key1: [0, 1], key2: [5]}} means that in left[key] you want to
            broadcast [right[key1], right[key2]] partitions and internal indices
            for `right` must be [[0, 1], [5]].
        keep_remaining : bool, default: False
            Whether or not to keep the other partitions.
            Some operations may want to drop the remaining partitions and
            keep only the results.

        Returns
        -------
        NumPy array
            An array of partition objects.

        Notes
        -----
        Your internal function must take these kwargs:
        [`internal_indices`, `other`, `internal_other_indices`] to work correctly!
        """
        if not axis:
            partitions_for_apply = left.T
            right = right.T
        else:
            partitions_for_apply = left

        [obj.drain_call_queue() for row in right for obj in row]

        def get_partitions(index):
            """Grab required partitions and indices from `right` and `right_indices`."""
            must_grab = right_indices[index]
            partitions_list = np.array([right[i] for i in must_grab.keys()])
            indices_list = list(must_grab.values())
            return {"other": partitions_list, "internal_other_indices": indices_list}

        new_partitions = np.array(
            [
                (
                    partitions_for_apply[i]
                    if i not in left_indices
                    else cls._apply_func_to_list_of_partitions_broadcast(
                        apply_func,
                        partitions_for_apply[i],
                        internal_indices=left_indices[i],
                        **get_partitions(i),
                    )
                )
                for i in range(len(partitions_for_apply))
                if i in left_indices or keep_remaining
            ]
        )
        if not axis:
            new_partitions = new_partitions.T
        return new_partitions

    @classmethod
    @wait_computations_if_benchmark_mode
    def base_broadcast_apply(cls, axis, apply_func, left, right):
        """
        Broadcast the `right` partitions to `left` and apply `apply_func` function.

        Parameters
        ----------
        axis : {0, 1}
            Axis to apply and broadcast over.
        apply_func : callable
            Function to apply.
        left : np.ndarray
            NumPy array of left partitions.
        right : np.ndarray
            NumPy array of right partitions.

        Returns
        -------
        np.ndarray
            NumPy array of result partition objects.

        Notes
        -----
        This will often be overridden by implementations. It materializes the
        entire partitions of the right and applies them to the left through `apply`.
        """

        def map_func(df, *others):
            other = (
                pandas.concat(others, axis=axis ^ 1) if len(others) > 1 else others[0]
            )
            # to reduce peak memory consumption
            del others
            return apply_func(df, other)

        map_func = cls.preprocess_func(map_func)
        rt_axis_parts = cls.axis_partition(right, axis ^ 1)
        return np.array(
            [
                [
                    part.apply(
                        map_func,
                        *(
                            rt_axis_parts[col_idx].list_of_blocks
                            if axis
                            else rt_axis_parts[row_idx].list_of_blocks
                        ),
                    )
                    for col_idx, part in enumerate(left[row_idx])
                ]
                for row_idx in range(len(left))
            ]
        )

    @classmethod
    @wait_computations_if_benchmark_mode
    def broadcast_axis_partitions(
        cls,
        axis,
        apply_func,
        left,
        right,
        keep_partitioning=False,
        num_splits=None,
        apply_indices=None,
        broadcast_all=True,
        enumerate_partitions=False,
        lengths=None,
        apply_func_args=None,
        **kwargs,
    ):
        """
        Broadcast the `right` partitions to `left` and apply `apply_func` along full `axis`.

        Parameters
        ----------
        axis : {0, 1}
            Axis to apply and broadcast over.
        apply_func : callable
            Function to apply.
        left : NumPy 2D array
            Left partitions.
        right : NumPy 2D array
            Right partitions.
        keep_partitioning : boolean, default: False
            The flag to keep partition boundaries for Modin Frame if possible.
            Setting it to True disables shuffling data from one partition to another in case the resulting
            number of splits is equal to the initial number of splits.
        num_splits : int, optional
            The number of partitions to split the result into across the `axis`. If None, then the number
            of splits will be infered automatically. If `num_splits` is None and `keep_partitioning=True`
            then the number of splits is preserved.
        apply_indices : list of ints, default: None
            Indices of `axis ^ 1` to apply function over.
        broadcast_all : bool, default: True
            Whether or not to pass all right axis partitions to each of the left axis partitions.
        enumerate_partitions : bool, default: False
            Whether or not to pass partition index into `apply_func`.
            Note that `apply_func` must be able to accept `partition_idx` kwarg.
        lengths : list of ints, default: None
            The list of lengths to shuffle the object. Note:
                1. Passing `lengths` omits the `num_splits` parameter as the number of splits
                will now be inferred from the number of integers present in `lengths`.
                2. When passing lengths you must explicitly specify `keep_partitioning=False`.
        apply_func_args : list-like, optional
            Positional arguments to pass to the `func`.
        **kwargs : dict
            Additional options that could be used by different engines.

        Returns
        -------
        NumPy array
            An array of partition objects.
        """
        ErrorMessage.catch_bugs_and_request_email(
            failure_condition=keep_partitioning and lengths is not None,
            extra_log=f"`keep_partitioning` must be set to `False` when passing `lengths`. Got: {keep_partitioning=} | {lengths=}",
        )

        # Since we are already splitting the DataFrame back up after an
        # operation, we will just use this time to compute the number of
        # partitions as best we can right now.
        if keep_partitioning and num_splits is None:
            num_splits = len(left) if axis == 0 else len(left.T)
        elif lengths:
            num_splits = len(lengths)
        elif num_splits is None:
            num_splits = NPartitions.get()
        else:
            ErrorMessage.catch_bugs_and_request_email(
                failure_condition=not isinstance(num_splits, int),
                extra_log=f"Expected `num_splits` to be an integer, got: {type(num_splits)} | {num_splits=}",
            )
        preprocessed_map_func = cls.preprocess_func(apply_func)
        left_partitions = cls.axis_partition(left, axis)
        right_partitions = None if right is None else cls.axis_partition(right, axis)
        # For mapping across the entire axis, we don't maintain partitioning because we
        # may want to line to partitioning up with another BlockPartitions object. Since
        # we don't need to maintain the partitioning, this gives us the opportunity to
        # load-balance the data as well.
        kw = {
            "num_splits": num_splits,
            "maintain_partitioning": keep_partitioning,
        }
        if lengths:
            kw["lengths"] = lengths
            kw["manual_partition"] = True

        if apply_indices is None:
            apply_indices = np.arange(len(left_partitions))

        result_blocks = np.array(
            [
                left_partitions[i].apply(
                    preprocessed_map_func,
                    *(apply_func_args if apply_func_args else []),
                    other_axis_partition=(
                        right_partitions if broadcast_all else right_partitions[i]
                    ),
                    **kw,
                    **({"partition_idx": idx} if enumerate_partitions else {}),
                    **kwargs,
                )
                for idx, i in enumerate(apply_indices)
            ]
        )
        # If we are mapping over columns, they are returned to use the same as
        # rows, so we need to transpose the returned 2D NumPy array to return
        # the structure to the correct order.
        return result_blocks.T if not axis else result_blocks

    @classmethod
    @wait_computations_if_benchmark_mode
    def base_map_partitions(
        cls,
        partitions,
        map_func,
        func_args=None,
        func_kwargs=None,
    ):
        """
        Apply `map_func` to every partition in `partitions`.

        Parameters
        ----------
        partitions : NumPy 2D array
            Partitions housing the data of Modin Frame.
        map_func : callable
            Function to apply.
        func_args : iterable, optional
            Positional arguments for the 'map_func'.
        func_kwargs : dict, optional
            Keyword arguments for the 'map_func'.

        Returns
        -------
        NumPy array
            An array of partitions
        """
        preprocessed_map_func = cls.preprocess_func(map_func)
        return np.array(
            [
                [
                    part.apply(
                        preprocessed_map_func,
                        *func_args if func_args is not None else (),
                        **func_kwargs if func_kwargs is not None else {},
                    )
                    for part in row_of_parts
                ]
                for row_of_parts in partitions
            ]
        )

    @classmethod
    @wait_computations_if_benchmark_mode
    def broadcast_apply(
        cls,
        axis,
        apply_func,
        left,
        right,
    ):
        """
        Broadcast the `right` partitions to `left` and apply `apply_func` function using different approaches to achieve the best performance.

        Parameters
        ----------
        axis : {0, 1}
            Axis to apply and broadcast over.
        apply_func : callable
            Function to apply.
        left : np.ndarray
            NumPy array of left partitions.
        right : np.ndarray
            NumPy array of right partitions.

        Returns
        -------
        np.ndarray
            NumPy array of result partition objects.
        """
        if not DynamicPartitioning.get():
            # block-wise broadcast
            new_partitions = cls.base_broadcast_apply(
                axis,
                apply_func,
                left,
                right,
            )
        else:
            # The dynamic partitioning behavior of `broadcast_apply` differs from that of `map_partitions`,
            # since the columnar approach for `broadcast_apply` results in slowdown.
            # axis-wise broadcast
            new_partitions = cls.broadcast_axis_partitions(
                axis=axis ^ 1,
                left=left,
                right=right,
                apply_func=apply_func,
                broadcast_all=False,
                keep_partitioning=True,
            )
        return new_partitions

    @classmethod
    @wait_computations_if_benchmark_mode
    def map_partitions(
        cls,
        partitions,
        map_func,
        func_args=None,
        func_kwargs=None,
    ):
        """
        Apply `map_func` to `partitions` using different approaches to achieve the best performance.

        Parameters
        ----------
        partitions : NumPy 2D array
            Partitions housing the data of Modin Frame.
        map_func : callable
            Function to apply.
        func_args : iterable, optional
            Positional arguments for the 'map_func'.
        func_kwargs : dict, optional
            Keyword arguments for the 'map_func'.

        Returns
        -------
        NumPy array
            An array of partitions
        """
        if not DynamicPartitioning.get():
            # block-wise map
            new_partitions = cls.base_map_partitions(
                partitions, map_func, func_args, func_kwargs
            )
        else:
            # axis-wise map
            # we choose an axis for a combination of partitions
            # whose size is closer to the number of CPUs
            if abs(partitions.shape[0] - CpuCount.get()) < abs(
                partitions.shape[1] - CpuCount.get()
            ):
                axis = 1
            else:
                axis = 0

            column_splits = CpuCount.get() // partitions.shape[1]

            if axis == 0 and column_splits > 1:
                # splitting by parts of columnar partitions
                new_partitions = cls.map_partitions_joined_by_column(
                    partitions, column_splits, map_func, func_args, func_kwargs
                )
            else:
                # splitting by full axis partitions
                new_partitions = cls.map_axis_partitions(
                    axis,
                    partitions,
                    lambda df: map_func(
                        df,
                        *(func_args if func_args is not None else ()),
                        **(func_kwargs if func_kwargs is not None else {}),
                    ),
                    keep_partitioning=True,
                )
        return new_partitions

    @classmethod
    @wait_computations_if_benchmark_mode
    def lazy_map_partitions(
        cls,
        partitions,
        map_func,
        func_args=None,
        func_kwargs=None,
        enumerate_partitions=False,
    ):
        """
        Apply `map_func` to every partition in `partitions` *lazily*.

        Parameters
        ----------
        partitions : NumPy 2D array
            Partitions of Modin Frame.
        map_func : callable
            Function to apply.
        func_args : iterable, optional
            Positional arguments for the 'map_func'.
        func_kwargs : dict, optional
            Keyword arguments for the 'map_func'.
        enumerate_partitions : bool, default: False

        Returns
        -------
        NumPy array
            An array of partitions
        """
        preprocessed_map_func = cls.preprocess_func(map_func)
        return np.array(
            [
                [
                    part.add_to_apply_calls(
                        preprocessed_map_func,
                        *(tuple() if func_args is None else func_args),
                        **func_kwargs if func_kwargs is not None else {},
                        **({"partition_idx": i} if enumerate_partitions else {}),
                    )
                    for part in row
                ]
                for i, row in enumerate(partitions)
            ]
        )

    @classmethod
    def map_axis_partitions(
        cls,
        axis,
        partitions,
        map_func,
        keep_partitioning=False,
        num_splits=None,
        lengths=None,
        enumerate_partitions=False,
        **kwargs,
    ):
        """
        Apply `map_func` to every partition in `partitions` along given `axis`.

        Parameters
        ----------
        axis : {0, 1}
            Axis to perform the map across (0 - index, 1 - columns).
        partitions : NumPy 2D array
            Partitions of Modin Frame.
        map_func : callable
            Function to apply.
        keep_partitioning : boolean, default: False
            The flag to keep partition boundaries for Modin Frame if possible.
            Setting it to True disables shuffling data from one partition to another in case the resulting
            number of splits is equal to the initial number of splits.
        num_splits : int, optional
            The number of partitions to split the result into across the `axis`. If None, then the number
            of splits will be infered automatically. If `num_splits` is None and `keep_partitioning=True`
            then the number of splits is preserved.
        lengths : list of ints, default: None
            The list of lengths to shuffle the object. Note:
                1. Passing `lengths` omits the `num_splits` parameter as the number of splits
                will now be inferred from the number of integers present in `lengths`.
                2. When passing lengths you must explicitly specify `keep_partitioning=False`.
        enumerate_partitions : bool, default: False
            Whether or not to pass partition index into `map_func`.
            Note that `map_func` must be able to accept `partition_idx` kwarg.
        **kwargs : dict
            Additional options that could be used by different engines.

        Returns
        -------
        NumPy array
            An array of new partitions for Modin Frame.

        Notes
        -----
        This method should be used in the case when `map_func` relies on
        some global information about the axis.
        """
        return cls.broadcast_axis_partitions(
            axis=axis,
            left=partitions,
            apply_func=map_func,
            keep_partitioning=keep_partitioning,
            num_splits=num_splits,
            right=None,
            lengths=lengths,
            enumerate_partitions=enumerate_partitions,
            **kwargs,
        )

    @classmethod
    def map_partitions_joined_by_column(
        cls,
        partitions,
        column_splits,
        map_func,
        map_func_args=None,
        map_func_kwargs=None,
    ):
        """
        Combine several blocks by column into one virtual partition and apply "map_func" to them.

        Parameters
        ----------
        partitions : NumPy 2D array
            Partitions of Modin Frame.
        column_splits : int
            The number of splits by column.
        map_func : callable
            Function to apply.
        map_func_args : iterable, optional
            Positional arguments for the 'map_func'.
        map_func_kwargs : dict, optional
            Keyword arguments for the 'map_func'.

        Returns
        -------
        NumPy array
            An array of new partitions for Modin Frame.
        """
        if column_splits < 1:
            raise ValueError(
                "The value of columns_splits must be greater than or equal to 1."
            )
        # step cannot be less than 1
        step = max(partitions.shape[0] // column_splits, 1)
        preprocessed_map_func = cls.preprocess_func(map_func)
        result = np.empty(partitions.shape, dtype=object)
        for i in range(
            0,
            partitions.shape[0],
            step,
        ):
            partitions_subset = partitions[i : i + step]
            # This is necessary when ``partitions.shape[0]`` is not divisible
            # by `column_splits` without a remainder.
            actual_step = len(partitions_subset)
            kw = {
                "num_splits": actual_step,
            }
            joined_column_partitions = cls.column_partitions(partitions_subset)
            for j in range(partitions.shape[1]):
                result[i : i + actual_step, j] = joined_column_partitions[j].apply(
                    preprocessed_map_func,
                    *map_func_args if map_func_args is not None else (),
                    **kw,
                    **map_func_kwargs if map_func_kwargs is not None else {},
                )

        return result

    @classmethod
    def concat(cls, axis, left_parts, right_parts):
        """
        Concatenate the blocks of partitions with another set of blocks.

        Parameters
        ----------
        axis : int
            The axis to concatenate to.
        left_parts : np.ndarray
            NumPy array of partitions to concatenate with.
        right_parts : np.ndarray or list
            NumPy array of partitions to be concatenated.

        Returns
        -------
        np.ndarray
            A new NumPy array with concatenated partitions.
        list[int] or None
            Row lengths if possible to compute it.

        Notes
        -----
        Assumes that the blocks are already the same shape on the
        dimension being concatenated. A ValueError will be thrown if this
        condition is not met.
        """
        # TODO: Possible change is `isinstance(right_parts, list)`
        if type(right_parts) is list:
            # `np.array` with partitions of empty ModinFrame has a shape (0,)
            # but `np.concatenate` can concatenate arrays only if its shapes at
            # specified axis are equals, so filtering empty frames to avoid concat error
            right_parts = [o for o in right_parts if o.size != 0]
            to_concat = (
                [left_parts] + right_parts if left_parts.size != 0 else right_parts
            )
            result = (
                np.concatenate(to_concat, axis=axis) if len(to_concat) else left_parts
            )
        else:
            result = np.append(left_parts, right_parts, axis=axis)
        if axis == 0:
            return cls.rebalance_partitions(result)
        else:
            return result, None

    @classmethod
    def to_pandas(cls, partitions):
        """
        Convert NumPy array of PandasDataframePartition to pandas DataFrame.

        Parameters
        ----------
        partitions : np.ndarray
            NumPy array of PandasDataframePartition.

        Returns
        -------
        pandas.DataFrame
            A pandas DataFrame
        """
        return create_pandas_df_from_partitions(
            cls.get_objects_from_partitions(partitions.flatten()), partitions.shape
        )

    @classmethod
    def to_numpy(cls, partitions, **kwargs):
        """
        Convert NumPy array of PandasDataframePartition to NumPy array of data stored within `partitions`.

        Parameters
        ----------
        partitions : np.ndarray
            NumPy array of PandasDataframePartition.
        **kwargs : dict
            Keyword arguments for PandasDataframePartition.to_numpy function.

        Returns
        -------
        np.ndarray
            A NumPy array.
        """
        return np.block(
            [[block.to_numpy(**kwargs) for block in row] for row in partitions]
        )

    @classmethod
    def split_pandas_df_into_partitions(
        cls, df, row_chunksize, col_chunksize, update_bar
    ):
        """
        Split given pandas DataFrame according to the row/column chunk sizes into distributed partitions.

        Parameters
        ----------
        df : pandas.DataFrame
        row_chunksize : int
        col_chunksize : int
        update_bar : callable(x) -> x
            Function that updates a progress bar.

        Returns
        -------
        2D np.ndarray[PandasDataframePartition]
        """
        put_func = cls._partition_class.put
        # even a full-axis slice can cost something (https://github.com/pandas-dev/pandas/issues/55202)
        # so we try not to do it if unnecessary.
        if col_chunksize >= len(df.columns):
            col_parts = [df]
        else:
            col_parts = [
                df.iloc[:, i : i + col_chunksize]
                for i in range(0, len(df.columns), col_chunksize)
            ]
        parts = [
            [
                update_bar(
                    put_func(col_part.iloc[i : i + row_chunksize]),
                )
                for col_part in col_parts
            ]
            for i in range(0, len(df), row_chunksize)
        ]
        return np.array(parts)

    @classmethod
    @wait_computations_if_benchmark_mode
    def from_pandas(cls, df, return_dims=False):
        """
        Return the partitions from pandas.DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            A pandas.DataFrame.
        return_dims : bool, default: False
            If it's True, return as (np.ndarray, row_lengths, col_widths),
            else np.ndarray.

        Returns
        -------
        (np.ndarray, backend) or (np.ndarray, backend, row_lengths, col_widths)
            A NumPy array with partitions (with dimensions or not).
        """
        num_splits = NPartitions.get()
        min_row_block_size = MinRowPartitionSize.get()
        min_column_block_size = MinColumnPartitionSize.get()
        row_chunksize = compute_chunksize(df.shape[0], num_splits, min_row_block_size)
        col_chunksize = compute_chunksize(
            df.shape[1], num_splits, min_column_block_size
        )

        bar_format = (
            "{l_bar}{bar}{r_bar}"
            if os.environ.get("DEBUG_PROGRESS_BAR", "False") == "True"
            else "{desc}: {percentage:3.0f}%{bar} Elapsed time: {elapsed}, estimated remaining time: {remaining}"
        )
        if ProgressBar.get():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    from tqdm.autonotebook import tqdm as tqdm_notebook
                except ImportError:
                    raise ImportError("Please pip install tqdm to use the progress bar")

            rows = max(1, round(len(df) / row_chunksize))
            cols = max(1, round(len(df.columns) / col_chunksize))
            update_count = rows * cols
            pbar = tqdm_notebook(
                total=round(update_count),
                desc="Distributing Dataframe",
                bar_format=bar_format,
            )
        else:
            pbar = None

        def update_bar(f):
            if ProgressBar.get():
                pbar.update(1)
            return f

        parts = cls.split_pandas_df_into_partitions(
            df, row_chunksize, col_chunksize, update_bar
        )
        backend = get_pandas_backend(df.dtypes)
        if ProgressBar.get():
            pbar.close()
        if not return_dims:
            return parts, backend
        else:
            row_lengths = [
                (
                    row_chunksize
                    if i + row_chunksize < len(df)
                    else len(df) % row_chunksize or row_chunksize
                )
                for i in range(0, len(df), row_chunksize)
            ]
            col_widths = [
                (
                    col_chunksize
                    if i + col_chunksize < len(df.columns)
                    else len(df.columns) % col_chunksize or col_chunksize
                )
                for i in range(0, len(df.columns), col_chunksize)
            ]
            return parts, backend, row_lengths, col_widths

    @classmethod
    def from_arrow(cls, at, return_dims=False):
        """
        Return the partitions from Apache Arrow (PyArrow).

        Parameters
        ----------
        at : pyarrow.table
            Arrow Table.
        return_dims : bool, default: False
            If it's True, return as (np.ndarray, row_lengths, col_widths),
            else np.ndarray.

        Returns
        -------
        (np.ndarray, backend) or (np.ndarray, backend, row_lengths, col_widths)
            A NumPy array with partitions (with dimensions or not).
        """
        return cls.from_pandas(at.to_pandas(), return_dims=return_dims)

    @classmethod
    def get_objects_from_partitions(cls, partitions):
        """
        Get the objects wrapped by `partitions` (in parallel if supported).

        Parameters
        ----------
        partitions : np.ndarray
            NumPy array with ``PandasDataframePartition``-s.

        Returns
        -------
        list
            The objects wrapped by `partitions`.
        """
        if hasattr(cls, "_execution_wrapper"):
            # more efficient parallel implementation
            for idx, part in enumerate(partitions):
                if hasattr(part, "force_materialization"):
                    partitions[idx] = part.force_materialization()
            assert all(
                [len(partition.list_of_blocks) == 1 for partition in partitions]
            ), "Implementation assumes that each partition contains a single block."
            return cls._execution_wrapper.materialize(
                [partition.list_of_blocks[0] for partition in partitions]
            )
        return [partition.get() for partition in partitions]

    @classmethod
    def wait_partitions(cls, partitions):
        """
        Wait on the objects wrapped by `partitions`, without materializing them.

        This method will block until all computations in the list have completed.

        Parameters
        ----------
        partitions : np.ndarray
            NumPy array with ``PandasDataframePartition``-s.

        Notes
        -----
        This method should be implemented in a more efficient way for engines that supports
        waiting on objects in parallel.
        """
        for partition in partitions:
            partition.wait()

    @classmethod
    def get_indices(cls, axis, partitions, index_func=None):
        """
        Get the internal indices stored in the partitions.

        Parameters
        ----------
        axis : {0, 1}
            Axis to extract the labels over.
        partitions : np.ndarray
            NumPy array with PandasDataframePartition's.
        index_func : callable, default: None
            The function to be used to extract the indices.

        Returns
        -------
        pandas.Index
            A pandas Index object.
        list of pandas.Index
            The list of internal indices for each partition.

        Notes
        -----
        These are the global indices of the object. This is mostly useful
        when you have deleted rows/columns internally, but do not know
        which ones were deleted.
        """
        if index_func is None:
            index_func = lambda df: df.axes[axis]  # noqa: E731
        ErrorMessage.catch_bugs_and_request_email(not callable(index_func))
        func = cls.preprocess_func(index_func)
        target = partitions.T if axis == 0 else partitions
        if len(target):
            new_idx = [idx.apply(func) for idx in target[0]]
            new_idx = cls.get_objects_from_partitions(new_idx)
        else:
            new_idx = [pandas.Index([])]

        # filter empty indexes in case there are multiple partitions
        total_idx = list(filter(len, new_idx))
        if len(total_idx) > 0:
            # TODO FIX INFORMATION LEAK!!!!1!!1!!
            total_idx = total_idx[0].append(total_idx[1:])
        else:
            # Meaning that all partitions returned a zero-length index,
            # in this case, we return an index of any partition to preserve
            # the index's metadata
            total_idx = new_idx[0]
        return total_idx, new_idx

    @classmethod
    def _apply_func_to_list_of_partitions_broadcast(
        cls, func, partitions, other, **kwargs
    ):
        """
        Apply a function to a list of remote partitions.

        `other` partitions will be broadcasted to `partitions`
        and `func` will be applied.

        Parameters
        ----------
        func : callable
            The func to apply.
        partitions : np.ndarray
            The partitions to which the `func` will apply.
        other : np.ndarray
            The partitions to be broadcasted to `partitions`.
        **kwargs : dict
            Keyword arguments for PandasDataframePartition.apply function.

        Returns
        -------
        list
            A list of PandasDataframePartition objects.
        """
        preprocessed_func = cls.preprocess_func(func)
        return [
            obj.apply(preprocessed_func, other=[o.get() for o in broadcasted], **kwargs)
            for obj, broadcasted in zip(partitions, other.T)
        ]

    @classmethod
    def _apply_func_to_list_of_partitions(cls, func, partitions, **kwargs):
        """
        Apply a function to a list of remote partitions.

        Parameters
        ----------
        func : callable
            The func to apply.
        partitions : np.ndarray
            The partitions to which the `func` will apply.
        **kwargs : dict
            Keyword arguments for PandasDataframePartition.apply function.

        Returns
        -------
        list
            A list of PandasDataframePartition objects.

        Notes
        -----
        This preprocesses the `func` first before applying it to the partitions.
        """
        preprocessed_func = cls.preprocess_func(func)
        return [obj.apply(preprocessed_func, **kwargs) for obj in partitions]

    @classmethod
    def combine(cls, partitions, new_index=None, new_columns=None):
        """
        Convert a NumPy 2D array of partitions to a NumPy 2D array of a single partition.

        Parameters
        ----------
        partitions : np.ndarray
            The partitions which have to be converted to a single partition.
        new_index : pandas.Index, optional
            Index for propagation into internal partitions.
            Optimization allowing to do this in one remote kernel.
        new_columns : pandas.Index, optional
            Columns for propagation into internal partitions.
            Optimization allowing to do this in one remote kernel.

        Returns
        -------
        np.ndarray
            A NumPy 2D array of a single partition.
        """
        if partitions.size <= 1 and new_index is None and new_columns is None:
            return partitions

        def to_pandas_remote(df, partition_shape, *dfs):
            """Copy of ``cls.to_pandas()`` method adapted for a remote function."""
            return create_pandas_df_from_partitions(
                (df,) + dfs,
                partition_shape,
                called_from_remote=True,
                new_index=new_index,
                new_columns=new_columns,
            )

        preprocessed_func = cls.preprocess_func(to_pandas_remote)
        partition_shape = partitions.shape
        partitions_flattened = partitions.flatten()
        for idx, part in enumerate(partitions_flattened):
            if hasattr(part, "force_materialization"):
                partitions_flattened[idx] = part.force_materialization()
        partition_refs = [
            partition.list_of_blocks[0] for partition in partitions_flattened[1:]
        ]
        combined_partition = partitions.flat[0].apply(
            preprocessed_func, partition_shape, *partition_refs
        )
        return np.array([combined_partition]).reshape(1, -1)

    @classmethod
    @wait_computations_if_benchmark_mode
    def apply_func_to_select_indices(
        cls, axis, partitions, func, indices, keep_remaining=False
    ):
        """
        Apply a function to select indices.

        Parameters
        ----------
        axis : {0, 1}
            Axis to apply the `func` over.
        partitions : np.ndarray
            The partitions to which the `func` will apply.
        func : callable
            The function to apply to these indices of partitions.
        indices : dict
            The indices to apply the function to.
        keep_remaining : bool, default: False
            Whether or not to keep the other partitions. Some operations
            may want to drop the remaining partitions and keep
            only the results.

        Returns
        -------
        np.ndarray
            A NumPy array with partitions.

        Notes
        -----
        Your internal function must take a kwarg `internal_indices` for
        this to work correctly. This prevents information leakage of the
        internal index to the external representation.
        """
        if partitions.size == 0:
            return np.array([[]])
        # Handling dictionaries has to be done differently, but we still want
        # to figure out the partitions that need to be applied to, so we will
        # store the dictionary in a separate variable and assign `indices` to
        # the keys to handle it the same as we normally would.
        if isinstance(func, dict):
            dict_func = func
        else:
            dict_func = None
        if not axis:
            partitions_for_apply = partitions.T
        else:
            partitions_for_apply = partitions
        # We may have a command to perform different functions on different
        # columns at the same time. We attempt to handle this as efficiently as
        # possible here. Functions that use this in the dictionary format must
        # accept a keyword argument `func_dict`.
        if dict_func is not None:
            if not keep_remaining:
                result = np.array(
                    [
                        cls._apply_func_to_list_of_partitions(
                            func,
                            partitions_for_apply[o_idx],
                            func_dict={
                                i_idx: dict_func[i_idx]
                                for i_idx in list_to_apply
                                if i_idx >= 0
                            },
                        )
                        for o_idx, list_to_apply in indices.items()
                    ]
                )
            else:
                result = np.array(
                    [
                        (
                            partitions_for_apply[i]
                            if i not in indices
                            else cls._apply_func_to_list_of_partitions(
                                func,
                                partitions_for_apply[i],
                                func_dict={
                                    idx: dict_func[idx]
                                    for idx in indices[i]
                                    if idx >= 0
                                },
                            )
                        )
                        for i in range(len(partitions_for_apply))
                    ]
                )
        else:
            if not keep_remaining:
                # We are passing internal indices in here. In order for func to
                # actually be able to use this information, it must be able to take in
                # the internal indices. This might mean an iloc in the case of Pandas
                # or some other way to index into the internal representation.
                result = np.array(
                    [
                        cls._apply_func_to_list_of_partitions(
                            func,
                            partitions_for_apply[idx],
                            internal_indices=list_to_apply,
                        )
                        for idx, list_to_apply in indices.items()
                    ]
                )
            else:
                # The difference here is that we modify a subset and return the
                # remaining (non-updated) blocks in their original position.
                result = np.array(
                    [
                        (
                            partitions_for_apply[i]
                            if i not in indices
                            else cls._apply_func_to_list_of_partitions(
                                func,
                                partitions_for_apply[i],
                                internal_indices=indices[i],
                            )
                        )
                        for i in range(len(partitions_for_apply))
                    ]
                )
        return result.T if not axis else result

    @classmethod
    @wait_computations_if_benchmark_mode
    def apply_func_to_select_indices_along_full_axis(
        cls, axis, partitions, func, indices, keep_remaining=False
    ):
        """
        Apply a function to a select subset of full columns/rows.

        Parameters
        ----------
        axis : {0, 1}
            The axis to apply the function over.
        partitions : np.ndarray
            The partitions to which the `func` will apply.
        func : callable
            The function to apply.
        indices : list-like
            The global indices to apply the func to.
        keep_remaining : bool, default: False
            Whether or not to keep the other partitions.
            Some operations may want to drop the remaining partitions and
            keep only the results.

        Returns
        -------
        np.ndarray
            A NumPy array with partitions.

        Notes
        -----
        This should be used when you need to apply a function that relies
        on some global information for the entire column/row, but only need
        to apply a function to a subset.
        For your func to operate directly on the indices provided,
        it must use `internal_indices` as a keyword argument.
        """
        if partitions.size == 0:
            return np.array([[]])
        # Handling dictionaries has to be done differently, but we still want
        # to figure out the partitions that need to be applied to, so we will
        # store the dictionary in a separate variable and assign `indices` to
        # the keys to handle it the same as we normally would.
        if isinstance(func, dict):
            dict_func = func
        else:
            dict_func = None
        preprocessed_func = cls.preprocess_func(func)
        # Since we might be keeping the remaining blocks that are not modified,
        # we have to also keep the block_partitions object in the correct
        # direction (transpose for columns).
        if not keep_remaining:
            selected_partitions = partitions.T if not axis else partitions
            selected_partitions = np.array([selected_partitions[i] for i in indices])
            selected_partitions = (
                selected_partitions.T if not axis else selected_partitions
            )
        else:
            selected_partitions = partitions
        if not axis:
            partitions_for_apply = cls.column_partitions(selected_partitions)
            partitions_for_remaining = partitions.T
        else:
            partitions_for_apply = cls.row_partitions(selected_partitions)
            partitions_for_remaining = partitions
        # We may have a command to perform different functions on different
        # columns at the same time. We attempt to handle this as efficiently as
        # possible here. Functions that use this in the dictionary format must
        # accept a keyword argument `func_dict`.
        if dict_func is not None:
            if not keep_remaining:
                result = np.array(
                    [
                        part.apply(
                            preprocessed_func,
                            func_dict={idx: dict_func[idx] for idx in indices[i]},
                        )
                        for i, part in zip(indices, partitions_for_apply)
                    ]
                )
            else:
                result = np.array(
                    [
                        (
                            partitions_for_remaining[i]
                            if i not in indices
                            else cls._apply_func_to_list_of_partitions(
                                preprocessed_func,
                                partitions_for_apply[i],
                                func_dict={idx: dict_func[idx] for idx in indices[i]},
                            )
                        )
                        for i in range(len(partitions_for_apply))
                    ]
                )
        else:
            if not keep_remaining:
                # See notes in `apply_func_to_select_indices`
                result = np.array(
                    [
                        part.apply(preprocessed_func, internal_indices=indices[i])
                        for i, part in zip(indices, partitions_for_apply)
                    ]
                )
            else:
                # See notes in `apply_func_to_select_indices`
                result = np.array(
                    [
                        (
                            partitions_for_remaining[i]
                            if i not in indices
                            else partitions_for_apply[i].apply(
                                preprocessed_func, internal_indices=indices[i]
                            )
                        )
                        for i in range(len(partitions_for_remaining))
                    ]
                )
        return result.T if not axis else result

    @classmethod
    @wait_computations_if_benchmark_mode
    def apply_func_to_indices_both_axis(
        cls,
        partitions,
        func,
        row_partitions_list,
        col_partitions_list,
        item_to_distribute=no_default,
        row_lengths=None,
        col_widths=None,
    ):
        """
        Apply a function along both axes.

        Parameters
        ----------
        partitions : np.ndarray
            The partitions to which the `func` will apply.
        func : callable
            The function to apply.
        row_partitions_list : iterable of tuples
            Iterable of tuples, containing 2 values:
                1. Integer row partition index.
                2. Internal row indexer of this partition.
        col_partitions_list : iterable of tuples
            Iterable of tuples, containing 2 values:
                1. Integer column partition index.
                2. Internal column indexer of this partition.
        item_to_distribute : np.ndarray or scalar, default: no_default
            The item to split up so it can be applied over both axes.
        row_lengths : list of ints, optional
            Lengths of partitions for every row. If not specified this information
            is extracted from partitions itself.
        col_widths : list of ints, optional
            Widths of partitions for every column. If not specified this information
            is extracted from partitions itself.

        Returns
        -------
        np.ndarray
            A NumPy array with partitions.

        Notes
        -----
        For your func to operate directly on the indices provided,
        it must use `row_internal_indices`, `col_internal_indices` as keyword
        arguments.
        """
        partition_copy = partitions.copy()
        row_position_counter = 0

        if row_lengths is None:
            row_lengths = [None] * len(row_partitions_list)
        if col_widths is None:
            col_widths = [None] * len(col_partitions_list)

        def compute_part_size(indexer, remote_part, part_idx, axis):
            """Compute indexer length along the specified axis for the passed partition."""
            if isinstance(indexer, slice):
                shapes_container = row_lengths if axis == 0 else col_widths
                part_size = shapes_container[part_idx]
                if part_size is None:
                    part_size = (
                        remote_part.length() if axis == 0 else remote_part.width()
                    )
                    shapes_container[part_idx] = part_size
                indexer = range(*indexer.indices(part_size))
            return len(indexer)

        for row_idx, row_values in enumerate(row_partitions_list):
            row_blk_idx, row_internal_idx = row_values
            col_position_counter = 0
            row_offset = 0
            for col_idx, col_values in enumerate(col_partitions_list):
                col_blk_idx, col_internal_idx = col_values
                remote_part = partition_copy[row_blk_idx, col_blk_idx]

                row_offset = compute_part_size(
                    row_internal_idx, remote_part, row_idx, axis=0
                )
                col_offset = compute_part_size(
                    col_internal_idx, remote_part, col_idx, axis=1
                )

                if item_to_distribute is not no_default:
                    if isinstance(item_to_distribute, np.ndarray):
                        item = item_to_distribute[
                            row_position_counter : row_position_counter + row_offset,
                            col_position_counter : col_position_counter + col_offset,
                        ]
                    else:
                        item = item_to_distribute
                    item = {"item": item}
                else:
                    item = {}
                block_result = remote_part.add_to_apply_calls(
                    func,
                    row_internal_indices=row_internal_idx,
                    col_internal_indices=col_internal_idx,
                    **item,
                )
                partition_copy[row_blk_idx, col_blk_idx] = block_result
                col_position_counter += col_offset
            row_position_counter += row_offset
        return partition_copy

    @classmethod
    @wait_computations_if_benchmark_mode
    def n_ary_operation(cls, left, func, right: list):
        r"""
        Apply an n-ary operation to multiple ``PandasDataframe`` objects.

        This method assumes that all the partitions of the dataframes in left
        and right have the same dimensions. For each position i, j in each
        dataframe's partitions, the result has a partition at (i, j) whose data
        is func(left_partitions[i,j], \*each_right_partitions[i,j]).

        Parameters
        ----------
        left : np.ndarray
            The partitions of left ``PandasDataframe``.
        func : callable
            The function to apply.
        right : list of np.ndarray
            The list of partitions of other ``PandasDataframe``.

        Returns
        -------
        np.ndarray
            A NumPy array with new partitions.
        """
        func = cls.preprocess_func(func)

        def get_right_block(right_partitions, row_idx, col_idx):
            partition = right_partitions[row_idx][col_idx]
            blocks = partition.list_of_blocks
            """
            NOTE:
            Currently we do one remote call per right virtual partition to
            materialize the partitions' blocks, then another remote call to do
            the n_ary operation. we could get better performance if we
            assembled the other partition within the remote `apply` call, by
            passing the partition in as `other_axis_partition`. However,
            passing `other_axis_partition` requires some extra care that would
            complicate the code quite a bit:
            - block partitions don't know how to deal with `other_axis_partition`
            - the right axis partition's axis could be different from the axis
              of the corresponding left partition
            - there can be multiple other_axis_partition because this is an n-ary
              operation and n can be > 2.
            So for now just do the materialization in a separate remote step.
            """
            if len(blocks) > 1:
                partition.force_materialization()
            assert len(partition.list_of_blocks) == 1
            return partition.list_of_blocks[0]

        return np.array(
            [
                [
                    part.apply(
                        func,
                        *(
                            get_right_block(right_partitions, row_idx, col_idx)
                            for right_partitions in right
                        ),
                    )
                    for col_idx, part in enumerate(left[row_idx])
                ]
                for row_idx in range(len(left))
            ]
        )

    @classmethod
    def finalize(cls, partitions):
        """
        Perform all deferred calls on partitions.

        Parameters
        ----------
        partitions : np.ndarray
            Partitions of Modin Dataframe on which all deferred calls should be performed.
        """
        [part.drain_call_queue() for row in partitions for part in row]

    @classmethod
    def rebalance_partitions(cls, partitions):
        """
        Rebalance a 2-d array of partitions if we are using ``PandasOnRay`` or ``PandasOnDask`` executions.

        For all other executions, the partitions are returned unchanged.

        Rebalance the partitions by building a new array
        of partitions out of the original ones so that:

        - If all partitions have a length, each new partition has roughly the same number of rows.
        - Otherwise, each new partition spans roughly the same number of old partitions.

        Parameters
        ----------
        partitions : np.ndarray
            The 2-d array of partitions to rebalance.

        Returns
        -------
        np.ndarray
            A NumPy array with the same; or new, rebalanced, partitions, depending on the execution
            engine and storage format.
        list[int] or None
            Row lengths if possible to compute it.
        """
        # We rebalance when the ratio of the number of existing partitions to
        # the ideal number of partitions is larger than this threshold. The
        # threshold is a heuristic that may need to be tuned for performance.
        max_excess_of_num_partitions = 1.5
        num_existing_partitions = partitions.shape[0]
        ideal_num_new_partitions = NPartitions.get()
        if (
            num_existing_partitions
            <= ideal_num_new_partitions * max_excess_of_num_partitions
        ):
            return partitions, None
        # If any partition has an unknown length, give each axis partition
        # roughly the same number of row partitions. We use `_length_cache` here
        # to avoid materializing any unmaterialized lengths.
        if any(
            partition._length_cache is None for row in partitions for partition in row
        ):
            # We need each partition to go into an axis partition, but the
            # number of axis partitions may not evenly divide the number of
            # partitions.
            chunk_size = compute_chunksize(
                num_existing_partitions, ideal_num_new_partitions, min_block_size=1
            )
            new_partitions = np.array(
                [
                    cls.column_partitions(
                        partitions[i : i + chunk_size],
                        full_axis=False,
                    )
                    for i in range(
                        0,
                        num_existing_partitions,
                        chunk_size,
                    )
                ]
            )
            return new_partitions, None

        # If we know the number of rows in every partition, then we should try
        # instead to give each new partition roughly the same number of rows.
        new_partitions = []
        # `start` is the index of the first existing partition that we want to
        # put into the current new partition.
        start = 0
        total_rows = sum(part.length() for part in partitions[:, 0])
        ideal_partition_size = compute_chunksize(
            total_rows, ideal_num_new_partitions, min_block_size=1
        )
        for _ in range(ideal_num_new_partitions):
            # We might pick up old partitions too quickly and exhaust all of them.
            if start >= len(partitions):
                break
            # `stop` is the index of the last existing partition so far that we
            # want to put into the current new partition.
            stop = start
            partition_size = partitions[start][0].length()
            # Add existing partitions into the current new partition until the
            # number of rows in the new partition hits `ideal_partition_size`.
            while stop < len(partitions) and partition_size < ideal_partition_size:
                stop += 1
                if stop < len(partitions):
                    partition_size += partitions[stop][0].length()
            # If the new partition is larger than we want, split the last
            # current partition that it contains into two partitions, where
            # the first partition has just enough rows to make the current
            # new partition have length `ideal_partition_size`, and the second
            # partition has the remainder.
            if partition_size > ideal_partition_size * max_excess_of_num_partitions:
                prev_length = sum(row[0].length() for row in partitions[start:stop])
                new_last_partition_size = ideal_partition_size - prev_length
                partitions = np.insert(
                    partitions,
                    stop + 1,
                    [
                        obj.mask(slice(new_last_partition_size, None), slice(None))
                        for obj in partitions[stop]
                    ],
                    0,
                )
                # TODO: explicit `_length_cache` computing may be avoided after #4903 is merged
                for obj in partitions[stop + 1]:
                    obj._length_cache = partition_size - (
                        prev_length + new_last_partition_size
                    )

                partitions[stop, :] = [
                    obj.mask(slice(None, new_last_partition_size), slice(None))
                    for obj in partitions[stop]
                ]
                # TODO: explicit `_length_cache` computing may be avoided after #4903 is merged
                for obj in partitions[stop]:
                    obj._length_cache = new_last_partition_size

            # The new virtual partitions are not `full_axis`, even if they
            # happen to span all rows in the dataframe, because they are
            # meant to be the final partitions of the dataframe. They've
            # already been split up correctly along axis 0, but using the
            # default full_axis=True would cause partition.apply() to split
            # its result along axis 0.
            new_partitions.append(
                cls.column_partitions(partitions[start : stop + 1], full_axis=False)
            )
            start = stop + 1
        new_partitions = np.array(new_partitions)
        lengths = [part.length() for part in new_partitions[:, 0]]
        return new_partitions, lengths

    @classmethod
    @wait_computations_if_benchmark_mode
    def shuffle_partitions(
        cls,
        partitions,
        index,
        shuffle_functions: "ShuffleFunctions",
        final_shuffle_func,
        right_partitions=None,
    ):
        """
        Return shuffled partitions.

        Parameters
        ----------
        partitions : np.ndarray
            The 2-d array of partitions to shuffle.
        index : int or list of ints
            The index(es) of the column partitions corresponding to the partitions that contain the column to sample.
        shuffle_functions : ShuffleFunctions
            An object implementing the functions that we will be using to perform this shuffle.
        final_shuffle_func : Callable(pandas.DataFrame) -> pandas.DataFrame
            Function that shuffles the data within each new partition.
        right_partitions : np.ndarray, optional
            Partitions to broadcast to `self` partitions. If specified, the method builds range-partitioning
            for `right_partitions` basing on bins calculated for `partitions`, then performs broadcasting.

        Returns
        -------
        np.ndarray
            A list of row-partitions that have been shuffled.
        """
        # Mask the partition that contains the column that will be sampled.
        masked_partitions = partitions[:, index]
        # Sample each partition
        sample_func = cls.preprocess_func(shuffle_functions.sample_fn)
        if masked_partitions.ndim == 1:
            samples = [partition.apply(sample_func) for partition in masked_partitions]
        else:
            samples = [
                cls._row_partition_class(row_part, full_axis=False).apply(sample_func)
                for row_part in masked_partitions
            ]
        # Get each sample to pass in to the pivot function
        samples = cls.get_objects_from_partitions(samples)
        num_bins = shuffle_functions.pivot_fn(samples)
        # Convert our list of block partitions to row partitions. We need to create full-axis
        # row partitions since we need to send the whole partition to the split step as otherwise
        # we wouldn't know how to split the block partitions that don't contain the shuffling key.
        row_partitions = cls.row_partitions(partitions)
        if num_bins > 1:
            # Gather together all of the sub-partitions
            split_row_partitions = np.array(
                [
                    partition.split(
                        shuffle_functions.split_fn,
                        num_splits=num_bins,
                        # The partition's metadata will never be accessed for the split partitions,
                        # thus no need to compute it.
                        extract_metadata=False,
                    )
                    for partition in row_partitions
                ]
            ).T

            if right_partitions is None:
                # We need to convert every partition that came from the splits into a column partition.
                return np.array(
                    [
                        [
                            cls._column_partitions_class(
                                row_partition, full_axis=False
                            ).apply(final_shuffle_func)
                        ]
                        for row_partition in split_row_partitions
                    ]
                )

            right_row_parts = cls.row_partitions(right_partitions)
            right_split_row_partitions = np.array(
                [
                    partition.split(
                        shuffle_functions.split_fn,
                        num_splits=num_bins,
                        extract_metadata=False,
                    )
                    for partition in right_row_parts
                ]
            ).T
            return np.array(
                [
                    cls._column_partitions_class(row_partition, full_axis=False).apply(
                        final_shuffle_func,
                        other_axis_partition=cls._column_partitions_class(
                            right_row_partitions
                        ),
                    )
                    for right_row_partitions, row_partition in zip(
                        right_split_row_partitions, split_row_partitions
                    )
                ]
            )

        else:
            # If there are not pivots we can simply apply the function row-wise
            if right_partitions is None:
                return np.array(
                    [row_part.apply(final_shuffle_func) for row_part in row_partitions]
                )
            right_row_parts = cls.row_partitions(right_partitions)
            return np.array(
                [
                    row_part.apply(
                        final_shuffle_func, other_axis_partition=right_row_part
                    )
                    for right_row_part, row_part in zip(right_row_parts, row_partitions)
                ]
            )
