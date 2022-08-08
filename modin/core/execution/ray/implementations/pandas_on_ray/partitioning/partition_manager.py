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

import ray

from modin.config import ProgressBar
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
from pandas.util._decorators import doc


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
    @doc(GenericRayDataframePartitionManager.map_partitions)
    def map_partitions(cls, partitions, map_func):
        return super(PandasOnRayDataframePartitionManager, cls).map_partitions(
            partitions, map_func
        )

    @classmethod
    @progress_bar_wrapper
    @doc(GenericRayDataframePartitionManager.lazy_map_partitions)
    def lazy_map_partitions(cls, partitions, map_func):
        return super(PandasOnRayDataframePartitionManager, cls).lazy_map_partitions(
            partitions, map_func
        )

    @classmethod
    @progress_bar_wrapper
    @doc(GenericRayDataframePartitionManager.map_axis_partitions)
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
    @progress_bar_wrapper
    @doc(GenericRayDataframePartitionManager._apply_func_to_list_of_partitions)
    def _apply_func_to_list_of_partitions(cls, func, partitions, **kwargs):
        return super(
            PandasOnRayDataframePartitionManager, cls
        )._apply_func_to_list_of_partitions(func, partitions, **kwargs)

    @classmethod
    @progress_bar_wrapper
    @doc(GenericRayDataframePartitionManager.apply_func_to_select_indices)
    def apply_func_to_select_indices(
        cls, axis, partitions, func, indices, keep_remaining=False
    ):
        return super(
            PandasOnRayDataframePartitionManager, cls
        ).apply_func_to_select_indices(
            axis, partitions, func, indices, keep_remaining=keep_remaining
        )

    @classmethod
    @progress_bar_wrapper
    @doc(GenericRayDataframePartitionManager.apply_func_to_select_indices)
    def apply_func_to_select_indices_along_full_axis(
        cls, axis, partitions, func, indices, keep_remaining=False
    ):
        return super(
            PandasOnRayDataframePartitionManager, cls
        ).apply_func_to_select_indices_along_full_axis(
            axis, partitions, func, indices, keep_remaining
        )

    @classmethod
    @progress_bar_wrapper
    @doc(GenericRayDataframePartitionManager.apply_func_to_indices_both_axis)
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
    @doc(GenericRayDataframePartitionManager.binary_operation)
    def binary_operation(cls, left, func, right):
        return super(PandasOnRayDataframePartitionManager, cls).binary_operation(
            left, func, right
        )
