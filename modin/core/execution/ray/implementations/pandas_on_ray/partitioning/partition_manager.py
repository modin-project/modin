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

"""Module houses class that implements ``GenericRayDataframePartitionManager`` using Ray."""

import inspect
import threading

import numpy as np
import ray

from modin.config import ProgressBar
from modin.core.execution.ray.common.utils import ObjectIDType
from modin.core.execution.ray.generic.partitioning import (
    GenericRayDataframePartitionManager,
)
from .virtual_partition import (
    PandasOnRayDataframeColumnPartition,
    PandasOnRayDataframeRowPartition,
)
from .partition import PandasOnRayDataframePartition
from modin.core.execution.ray.generic.modin_aqp import call_progress_bar
from pandas._libs.lib import no_default


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
    callable
        Decorated version of `f` which reports progress.
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


class PandasOnRayDataframePartitionManager(GenericRayDataframePartitionManager):
    """The class implements the interface in `PandasDataframePartitionManager`."""

    # This object uses RayRemotePartition objects as the underlying store.
    _partition_class = PandasOnRayDataframePartition
    _column_partitions_class = PandasOnRayDataframeColumnPartition
    _row_partition_class = PandasOnRayDataframeRowPartition

    @classmethod
    def get_objects_from_partitions(cls, partitions):
        """
        Get the objects wrapped by `partitions` in parallel.

        Parameters
        ----------
        partitions : np.ndarray
            NumPy array with ``PandasDataframePartition``-s.

        Returns
        -------
        list
            The objects wrapped by `partitions`.
        """
        return ray.get([partition._data for partition in partitions])

    @classmethod
    def wait_partitions(cls, partitions):
        """
        Wait on the objects wrapped by `partitions` in parallel, without materializing them.

        This method will block until all computations in the list have completed.

        Parameters
        ----------
        partitions : np.ndarray
            NumPy array with ``PandasDataframePartition``-s.
        """
        ray.wait(
            [partition._data for partition in partitions], num_returns=len(partitions)
        )

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
        return super(PandasOnRayDataframePartitionManager, cls).map_partitions(
            partitions, map_func
        )

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
        return super(PandasOnRayDataframePartitionManager, cls).lazy_map_partitions(
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
        **kwargs,
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
            Setting it to True prevents data shuffling between partitions.
        lengths : list of ints, default: None
            List of lengths to shuffle the object.
        enumerate_partitions : bool, default: False
            Whether or not to pass partition index into `map_func`.
            Note that `map_func` must be able to accept `partition_idx` kwarg.
        **kwargs : dict
            Additional options that could be used by different engines.

        Returns
        -------
        np.ndarray
            A NumPy array of new partitions for Modin Frame.

        Notes
        -----
        This method should be used in the case when `map_func` relies on
        some global information about the axis.
        """
        return super(PandasOnRayDataframePartitionManager, cls).map_axis_partitions(
            axis,
            partitions,
            map_func,
            keep_partitioning,
            lengths,
            enumerate_partitions,
            **kwargs,
        )

    @classmethod
    def _update_partition_dimension_caches(cls, partitions: np.ndarray):
        """
        Build and set the length and width caches of each _physical_ partition in the given list of partitions.

        Parameters
        ----------
        partitions : np.ndarray
            The partitions for which to update length caches.
        """
        promises = []
        # If `ray_idxs[i] = j`, that means `promises[i]` represents a computation corresponding
        # to a dimension of `partitions[j]`.
        part_idxs = []
        # If `is_lens[i] = True`, then `promises[i]` is a computation for a length; otherwise
        # it is for a width
        is_lens = []
        for i, part in enumerate(partitions):
            l_cache = part.try_build_length_cache()
            if isinstance(l_cache, ObjectIDType):
                promises.append(l_cache)
                part_idxs.append(i)
                is_lens.append(True)
            w_cache = part.try_build_width_cache()
            if isinstance(w_cache, ObjectIDType):
                promises.append(w_cache)
                part_idxs.append(i)
                is_lens.append(False)
        new_dims = ray.get(promises)
        for i, new_cache in enumerate(new_dims):
            part = partitions[part_idxs[i]]
            if is_lens[i]:
                part.try_set_length_cache(new_dims[i])
            else:
                part.try_set_width_cache(new_dims[i])

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
        This preprocesses the `func` first before applying it to the partitions.
        """
        return super(
            PandasOnRayDataframePartitionManager, cls
        )._apply_func_to_list_of_partitions(func, partitions, **kwargs)

    @classmethod
    @progress_bar_wrapper
    def apply_func_to_select_indices(
        cls, axis, partitions, func, indices, keep_remaining=False
    ):
        """
        Apply a `func` to select `indices` of `partitions`.

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
        return super(
            PandasOnRayDataframePartitionManager, cls
        ).apply_func_to_select_indices(
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
            PandasOnRayDataframePartitionManager, cls
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
        row_partitions_list : list
            List of row partitions.
        col_partitions_list : list
            List of column partitions.
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
        it must use ``row_internal_indices`` and ``col_internal_indices`` as keyword
        arguments.
        """
        return super(
            PandasOnRayDataframePartitionManager, cls
        ).apply_func_to_indices_both_axis(
            partitions,
            func,
            row_partitions_list,
            col_partitions_list,
            item_to_distribute,
            row_lengths,
            col_widths,
        )

    @classmethod
    @progress_bar_wrapper
    def binary_operation(cls, left, func, right):
        """
        Apply a function that requires partitions of two ``PandasOnRayDataframe`` objects.

        Parameters
        ----------
        left : np.ndarray
            The partitions of left ``PandasOnRayDataframe``.
        func : callable
            The function to apply.
        right : np.ndarray
            The partitions of right ``PandasOnRayDataframe``.

        Returns
        -------
        np.ndarray
            A NumPy array with new partitions.
        """
        return super(PandasOnRayDataframePartitionManager, cls).binary_operation(
            left, func, right
        )
