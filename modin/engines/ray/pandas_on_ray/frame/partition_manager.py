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

"""Module houses class that implements ``RayFrameManager`` using Ray."""

import inspect
import numpy as np
import threading

from modin.config import ProgressBar
from modin.engines.ray.generic.frame.partition_manager import RayFrameManager
from .axis_partition import (
    PandasOnRayFrameColumnPartition,
    PandasOnRayFrameRowPartition,
)
from .partition import PandasOnRayFramePartition
from .modin_aqp import call_progress_bar
from modin.error_message import ErrorMessage
import pandas

import ray


def progress_bar_wrapper(f):
    """
    Wrap computation function inside a progress bar.

    Spawns another thread which displays a progress bar showing
    estimated completion time.

    Parameters
    ----------
    f : callable
        The name of the function to be wrapped.

    Returns
    -------
    A new BaseFrameManager object, the type of object that
    called this.
    """
    from functools import wraps

    @wraps(f)
    def magic(*args, **kwargs):
        result_parts = f(*args, **kwargs)
        if ProgressBar.get():
            current_frame = inspect.currentframe()
            function_name = None
            while function_name != "<module>":
                (
                    filename,
                    line_number,
                    function_name,
                    lines,
                    index,
                ) = inspect.getframeinfo(current_frame)
                current_frame = current_frame.f_back
            t = threading.Thread(
                target=call_progress_bar,
                args=(result_parts, line_number),
            )
            t.start()
            # We need to know whether or not we are in a jupyter notebook
            from IPython import get_ipython

            try:
                ipy_str = str(type(get_ipython()))
                if "zmqshell" not in ipy_str:
                    t.join()
            except Exception:
                pass
        return result_parts

    return magic


@ray.remote
def func(df, apply_func, call_queue_df=None, call_queues_other=None, *others):
    """
    Perform `apply_func` for `df` remotely.

    Parameters
    ----------
    df : ray.ObjectRef
        Dataframe to which `apply_func` will be applied.
        After running function, automaterialization
        ray.ObjectRef->pandas.DataFrame happens.
    apply_func : {callable, ray.ObjectRef}
        The function to apply.
    call_queue_df : list, optional
        The call queue to be executed on `df`.
    call_queues_other : list, optional
        The call queue to be executed on `others`.
    *others : iterable
        List of other parameters.

    Returns
    -------
    The same as returns of `apply_func`.
    """
    if call_queue_df is not None and len(call_queue_df) > 0:
        for call, kwargs in call_queue_df:
            if isinstance(call, ray.ObjectRef):
                call = ray.get(call)
            if isinstance(kwargs, ray.ObjectRef):
                kwargs = ray.get(kwargs)
            df = call(df, **kwargs)
    new_others = np.empty(shape=len(others), dtype=object)
    for i, call_queue_other in enumerate(call_queues_other):
        other = others[i]
        if call_queue_other is not None and len(call_queue_other) > 0:
            for call, kwargs in call_queue_other:
                if isinstance(call, ray.ObjectRef):
                    call = ray.get(call)
                if isinstance(kwargs, ray.ObjectRef):
                    kwargs = ray.get(kwargs)
                other = call(other, **kwargs)
        new_others[i] = other
    return apply_func(df, new_others)


class PandasOnRayFrameManager(RayFrameManager):
    """The class implements the interface in ``RayFrameManager`` using Ray."""

    # This object uses RayRemotePartition objects as the underlying store.
    _partition_class = PandasOnRayFramePartition
    _column_partitions_class = PandasOnRayFrameColumnPartition
    _row_partition_class = PandasOnRayFrameRowPartition

    @classmethod
    def get_indices(cls, axis, partitions, index_func=None):
        """
        Get the internal indices stored in the partitions.

        Parameters
        ----------
        axis : {0, 1}
            Axis to extract the labels over.
        partitions : np.ndarray
            NumPy array with BaseFramePartition's.
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
            # We grab the first column of blocks and extract the indices
            new_idx = (
                [idx.apply(func).oid for idx in partitions.T[0]]
                if len(partitions.T)
                else []
            )
        else:
            new_idx = (
                [idx.apply(func).oid for idx in partitions[0]]
                if len(partitions)
                else []
            )
        new_idx = ray.get(new_idx)
        return new_idx[0].append(new_idx[1:]) if len(new_idx) else new_idx

    @classmethod
    def broadcast_apply(cls, axis, apply_func, left, right, other_name="r"):
        """
        Broadcast the `right` partitions to `left` and apply `apply_func` to selected indices.

        Parameters
        ----------
        axis : {0, 1}
            Axis to apply and broadcast over.
        apply_func : callable
            Function to apply.
        left : np.ndarray
            NumPy 2D array of left partitions.
        right : np.ndarray
            NumPy 2D array of right partitions.
        other_name : str, default: "r"
            Name of key-value argument for `apply_func` that
            is used to pass `right` to `apply_func`.

        Returns
        -------
        np.ndarray
            An array of partition objects.
        """

        def mapper(df, others):
            other = pandas.concat(others, axis=axis ^ 1)
            return apply_func(df, **{other_name: other})

        mapper = ray.put(mapper)
        new_partitions = np.array(
            [
                [
                    PandasOnRayFramePartition(
                        func.remote(
                            part.oid,
                            mapper,
                            part.call_queue,
                            [obj[col_idx].call_queue for obj in right]
                            if axis
                            else [obj.call_queue for obj in right[row_idx]],
                            *(
                                [obj[col_idx].oid for obj in right]
                                if axis
                                else [obj.oid for obj in right[row_idx]]
                            ),
                        )
                    )
                    for col_idx, part in enumerate(left[row_idx])
                ]
                for row_idx in range(len(left))
            ]
        )
        return new_partitions

    @classmethod
    @progress_bar_wrapper
    def map_partitions(cls, partitions, map_func):
        """
        Apply `map_func` to every partition in `partitions`.

        Parameters
        ----------
        partitions : np.ndarray
            A NumPy 2D array of partitions to perform operation on.
        map_func : callable
            Function to apply.

        Returns
        -------
        np.ndarray
            A NumPy array of partitions.
        """
        return super(PandasOnRayFrameManager, cls).map_partitions(partitions, map_func)

    @classmethod
    @progress_bar_wrapper
    def lazy_map_partitions(cls, partitions, map_func):
        """
        Apply `map_func` to every partition in `partitions` *lazily*.

        Parameters
        ----------
        partitions : np.ndarray
            A NumPy 2D array of partitions to perform operation on.
        map_func : callable
            Function to apply.

        Returns
        -------
        np.ndarray
            A NumPy array of partitions.
        """
        return super(PandasOnRayFrameManager, cls).lazy_map_partitions(
            partitions, map_func
        )

    @classmethod
    @progress_bar_wrapper
    def map_axis_partitions(
        cls,
        axis,
        partitions,
        map_func,
        keep_partitioning=False,
        lengths=None,
        enumerate_partitions=False,
    ):
        """
        Apply `map_func` to every partition in `partitions` along given `axis`.

        Parameters
        ----------
        axis : {0, 1}
            Axis to perform the map across (0 - index, 1 - columns).
        partitions : np.ndarray
            A NumPy 2D array of partitions to perform operation on.
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

        Returns
        -------
        np.ndarray
            A NumPy array of new partitions for Modin Frame.

        Notes
        -----
        This method should be used in the case when `map_func` relies on
        some global information about the axis.
        """
        return super(PandasOnRayFrameManager, cls).map_axis_partitions(
            axis, partitions, map_func, keep_partitioning, lengths, enumerate_partitions
        )

    @classmethod
    @progress_bar_wrapper
    def _apply_func_to_list_of_partitions(cls, func, partitions, **kwargs):
        """
        Apply a `func` to a list of remote `partitions`.

        Parameters
        ----------
        func : callable
            The func to apply.
        partitions : np.ndarray
            The partitions to which the `func` will apply.
        **kwargs : dict
            Keyword arguments.

        Returns
        -------
        list
            A list of ``RayFramePartition`` objects.

        Notes
        -----
        The main use for this is to preprocess the func.
        """
        return super(PandasOnRayFrameManager, cls)._apply_func_to_list_of_partitions(
            func, partitions, **kwargs
        )

    @classmethod
    @progress_bar_wrapper
    def apply_func_to_select_indices(
        cls, axis, partitions, func, indices, keep_remaining=False
    ):
        """
        Apply a `func` to select indices of `partitions`.

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
        return super(PandasOnRayFrameManager, cls).apply_func_to_select_indices(
            axis, partitions, func, indices, keep_remaining=keep_remaining
        )

    @classmethod
    @progress_bar_wrapper
    def apply_func_to_select_indices_along_full_axis(
        cls, axis, partitions, func, indices, keep_remaining=False
    ):
        """
        Apply a `func` to a select subset of full columns/rows.

        Parameters
        ----------
        axis : {0, 1}
            The axis to apply the `func` over.
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
        return super(
            PandasOnRayFrameManager, cls
        ).apply_func_to_select_indices_along_full_axis(
            axis, partitions, func, indices, keep_remaining
        )

    @classmethod
    @progress_bar_wrapper
    def apply_func_to_indices_both_axis(
        cls,
        partitions,
        func,
        row_partitions_list,
        col_partitions_list,
        item_to_distribute=None,
    ):
        """
        Apply a function along both axes.

        Parameters
        ----------
        partitions : np.ndarray
            The partitions to which the `func` will apply.
        func : callable
            The function to apply.
        row_partitions_list : list
            List of row partitions.
        col_partitions_list : list
            List of column partitions.
        item_to_distribute : item, optional
            The item to split up so it can be applied over both axes.

        Returns
        -------
        np.ndarray
            A NumPy array with partitions.

        Notes
        -----
        For your func to operate directly on the indices provided,
        it must use `row_internal_indices, col_internal_indices` as keyword
        arguments.
        """
        return super(PandasOnRayFrameManager, cls).apply_func_to_indices_both_axis(
            partitions,
            func,
            row_partitions_list,
            col_partitions_list,
            item_to_distribute,
        )

    @classmethod
    @progress_bar_wrapper
    def binary_operation(cls, axis, left, func, right):
        """
        Apply a function that requires two PandasOnRayFrame objects.

        Parameters
        ----------
        axis : {0, 1}
            The axis to apply the function over (0 - rows, 1 - columns).
        left : np.ndarray
            The partitions of left PandasOnRayFrame.
        func : callable
            The function to apply.
        right : np.ndarray
            The partitions of right PandasOnRayFrame.

        Returns
        -------
        np.ndarray
            A NumPy array with new partitions.
        """
        return super(PandasOnRayFrameManager, cls).binary_operation(
            axis, left, func, right
        )
