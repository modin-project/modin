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

from abc import ABC
from functools import wraps
import numpy as np
import pandas
from pandas._libs.lib import no_default
import warnings

from modin.error_message import ErrorMessage
from modin.core.storage_formats.pandas.utils import compute_chunksize
from modin.core.dataframe.pandas.utils import concatenate
from modin.config import NPartitions, ProgressBar, BenchmarkMode

import os


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
    if BenchmarkMode.get():

        @wraps(func)
        def wait(*args, **kwargs):
            """Wait for computation results."""
            result = func(*args, **kwargs)
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
            [part.drain_call_queue() for part in partitions.flatten()]
            # need to go through all the values of the map iterator
            # since `wait` does not return anything, we need to explicitly add
            # the return `True` value from the lambda
            # TODO(https://github.com/modin-project/modin/issues/4491): Wait
            # for all the partitions in parallel.
            all(map(lambda partition: partition.wait() or True, partitions.flatten()))
            return result

        return wait
    return func


class PandasDataframePartitionManager(ABC):
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
        return cls._partition_class.preprocess_func(map_func)

    # END Abstract Methods

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
            mapped_partitions = cls.broadcast_apply(
                axis, map_func, left=partitions, right=by, other_name="other"
            )
        else:
            mapped_partitions = cls.map_partitions(partitions, map_func)
        return cls.map_axis_partitions(
            axis, mapped_partitions, reduce_func, enumerate_partitions=True
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
                partitions_for_apply[i]
                if i not in left_indices
                else cls._apply_func_to_list_of_partitions_broadcast(
                    apply_func,
                    partitions_for_apply[i],
                    internal_indices=left_indices[i],
                    **get_partitions(i),
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
    def broadcast_apply(cls, axis, apply_func, left, right, other_name="r"):
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
        other_name : str, default: "r"
            Name of key-value argument for `apply_func` that
            is used to pass `right` to `apply_func`.

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
            other = pandas.concat(others, axis=axis ^ 1)
            return apply_func(df, **{other_name: other})

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
        apply_indices=None,
        enumerate_partitions=False,
        lengths=None,
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
            The flag to keep partition boundaries for Modin Frame.
            Setting it to True disables shuffling data from one partition to another.
        apply_indices : list of ints, default: None
            Indices of `axis ^ 1` to apply function over.
        enumerate_partitions : bool, default: False
            Whether or not to pass partition index into `apply_func`.
            Note that `apply_func` must be able to accept `partition_idx` kwarg.
        lengths : list of ints, default: None
            The list of lengths to shuffle the object.
        **kwargs : dict
            Additional options that could be used by different engines.

        Returns
        -------
        NumPy array
            An array of partition objects.
        """
        # Since we are already splitting the DataFrame back up after an
        # operation, we will just use this time to compute the number of
        # partitions as best we can right now.
        if keep_partitioning:
            num_splits = len(left) if axis == 0 else len(left.T)
        elif lengths:
            num_splits = len(lengths)
        else:
            num_splits = NPartitions.get()
        preprocessed_map_func = cls.preprocess_func(apply_func)
        left_partitions = cls.axis_partition(left, axis)
        right_partitions = None if right is None else cls.axis_partition(right, axis)
        # For mapping across the entire axis, we don't maintain partitioning because we
        # may want to line to partitioning up with another BlockPartitions object. Since
        # we don't need to maintain the partitioning, this gives us the opportunity to
        # load-balance the data as well.
        kw = {
            "num_splits": num_splits,
            "other_axis_partition": right_partitions,
        }
        if lengths:
            kw["_lengths"] = lengths
            kw["manual_partition"] = True

        if apply_indices is None:
            apply_indices = np.arange(len(left_partitions))

        result_blocks = np.array(
            [
                left_partitions[i].apply(
                    preprocessed_map_func,
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
    def map_partitions(cls, partitions, map_func):
        """
        Apply `map_func` to every partition in `partitions`.

        Parameters
        ----------
        partitions : NumPy 2D array
            Partitions housing the data of Modin Frame.
        map_func : callable
            Function to apply.

        Returns
        -------
        NumPy array
            An array of partitions
        """
        preprocessed_map_func = cls.preprocess_func(map_func)
        return np.array(
            [
                [part.apply(preprocessed_map_func) for part in row_of_parts]
                for row_of_parts in partitions
            ]
        )

    @classmethod
    @wait_computations_if_benchmark_mode
    def lazy_map_partitions(cls, partitions, map_func):
        """
        Apply `map_func` to every partition in `partitions` *lazily*.

        Parameters
        ----------
        partitions : NumPy 2D array
            Partitions of Modin Frame.
        map_func : callable
            Function to apply.

        Returns
        -------
        NumPy array
            An array of partitions
        """
        preprocessed_map_func = cls.preprocess_func(map_func)
        return np.array(
            [
                [part.add_to_apply_calls(preprocessed_map_func) for part in row]
                for row in partitions
            ]
        )

    @classmethod
    def map_axis_partitions(
        cls,
        axis,
        partitions,
        map_func,
        keep_partitioning=False,
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
        keep_partitioning : bool, default: False
            Whether to keep partitioning for Modin Frame.
            Setting it to True stops data shuffling between partitions.
        lengths : list of ints, default: None
            List of lengths to shuffle the object.
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
            right=None,
            lengths=lengths,
            enumerate_partitions=enumerate_partitions,
            **kwargs,
        )

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
            return (
                np.concatenate(to_concat, axis=axis) if len(to_concat) else left_parts
            )
        else:
            return np.append(left_parts, right_parts, axis=axis)

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
        retrieved_objects = [[obj.to_pandas() for obj in part] for part in partitions]
        if all(
            isinstance(part, pandas.Series) for row in retrieved_objects for part in row
        ):
            axis = 0
        elif all(
            isinstance(part, pandas.DataFrame)
            for row in retrieved_objects
            for part in row
        ):
            axis = 1
        else:
            ErrorMessage.catch_bugs_and_request_email(True)
        df_rows = [
            pandas.concat([part for part in row], axis=axis)
            for row in retrieved_objects
            if not all(part.empty for part in row)
        ]
        if len(df_rows) == 0:
            return pandas.DataFrame()
        else:
            return concatenate(df_rows)

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
        np.ndarray or (np.ndarray, row_lengths, col_widths)
            A NumPy array with partitions (with dimensions or not).
        """

        def update_bar(pbar, f):
            if ProgressBar.get():
                pbar.update(1)
            return f

        num_splits = NPartitions.get()
        put_func = cls._partition_class.put
        row_chunksize = compute_chunksize(df.shape[0], num_splits)
        col_chunksize = compute_chunksize(df.shape[1], num_splits)

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
        parts = [
            [
                update_bar(
                    pbar,
                    put_func(
                        df.iloc[i : i + row_chunksize, j : j + col_chunksize].copy()
                    ),
                )
                for j in range(0, len(df.columns), col_chunksize)
            ]
            for i in range(0, len(df), row_chunksize)
        ]
        if ProgressBar.get():
            pbar.close()
        if not return_dims:
            return np.array(parts)
        else:
            row_lengths = [
                row_chunksize
                if i + row_chunksize < len(df)
                else len(df) % row_chunksize or row_chunksize
                for i in range(0, len(df), row_chunksize)
            ]
            col_widths = [
                col_chunksize
                if i + col_chunksize < len(df.columns)
                else len(df.columns) % col_chunksize or col_chunksize
                for i in range(0, len(df.columns), col_chunksize)
            ]
            return np.array(parts), row_lengths, col_widths

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
        np.ndarray or (np.ndarray, row_lengths, col_widths)
            A NumPy array with partitions (with dimensions or not).
        """
        return cls.from_pandas(at.to_pandas(), return_dims=return_dims)

    @classmethod
    def get_objects_from_partitions(cls, partitions):
        """
        Get the objects wrapped by `partitions`.

        Parameters
        ----------
        partitions : np.ndarray
            NumPy array with ``PandasDataframePartition``-s.

        Returns
        -------
        list
            The objects wrapped by `partitions`.

        Notes
        -----
        This method should be implemented in a more efficient way for engines that support
        getting objects in parallel.
        """
        return [partition.get() for partition in partitions]

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

        Notes
        -----
        These are the global indices of the object. This is mostly useful
        when you have deleted rows/columns internally, but do not know
        which ones were deleted.
        """
        ErrorMessage.catch_bugs_and_request_email(not callable(index_func))
        func = cls.preprocess_func(index_func)
        if axis == 0:
            new_idx = (
                [idx.apply(func) for idx in partitions.T[0]]
                if len(partitions.T)
                else []
            )
        else:
            new_idx = (
                [idx.apply(func) for idx in partitions[0]] if len(partitions) else []
            )
        new_idx = cls.get_objects_from_partitions(new_idx)
        # TODO FIX INFORMATION LEAK!!!!1!!1!!
        return new_idx[0].append(new_idx[1:]) if len(new_idx) else new_idx

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
                        partitions_for_apply[i]
                        if i not in indices
                        else cls._apply_func_to_list_of_partitions(
                            func,
                            partitions_for_apply[i],
                            func_dict={
                                idx: dict_func[idx] for idx in indices[i] if idx >= 0
                            },
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
                        partitions_for_apply[i]
                        if i not in indices
                        else cls._apply_func_to_list_of_partitions(
                            func, partitions_for_apply[i], internal_indices=indices[i]
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
                        partitions_for_remaining[i]
                        if i not in indices
                        else cls._apply_func_to_list_of_partitions(
                            preprocessed_func,
                            partitions_for_apply[i],
                            func_dict={idx: dict_func[idx] for idx in indices[i]},
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
                        partitions_for_remaining[i]
                        if i not in indices
                        else partitions_for_apply[i].apply(
                            preprocessed_func, internal_indices=indices[i]
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
    def binary_operation(cls, axis, left, func, right):
        """
        Apply a function that requires two PandasDataframe objects.

        Parameters
        ----------
        axis : {0, 1}
            The axis to apply the function over (0 - rows, 1 - columns).
        left : np.ndarray
            The partitions of left PandasDataframe.
        func : callable
            The function to apply.
        right : np.ndarray
            The partitions of right PandasDataframe.

        Returns
        -------
        np.ndarray
            A NumPy array with new partitions.
        """
        if axis:
            left_partitions = cls.row_partitions(left)
            right_partitions = cls.row_partitions(right)
        else:
            left_partitions = cls.column_partitions(left)
            right_partitions = cls.column_partitions(right)
        func = cls.preprocess_func(func)
        result = np.array(
            [
                left_partitions[i].apply(
                    func,
                    num_splits=NPartitions.get(),
                    other_axis_partition=right_partitions[i],
                )
                for i in range(len(left_partitions))
            ]
        )
        return result if axis else result.T

    @classmethod
    @wait_computations_if_benchmark_mode
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
        Return the provided array of partitions without rebalancing it.

        Parameters
        ----------
        partitions : np.ndarray
            The 2-d array of partitions to rebalance.

        Returns
        -------
        np.ndarray
            The same 2-d array.
        """
        return partitions
