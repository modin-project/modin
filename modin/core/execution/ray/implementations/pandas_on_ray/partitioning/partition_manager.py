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

from modin.config import ProgressBar
from modin.core.execution.ray.generic.partitioning import (
    GenericRayDataframePartitionManager,
)
from modin.core.execution.ray.common import RayWrapper
from modin.core.execution.ray.common.utils import wait
from .virtual_partition import (
    PandasOnRayDataframeColumnPartition,
    PandasOnRayDataframeRowPartition,
)
from .partition import PandasOnRayDataframePartition
from modin.core.execution.ray.generic.modin_aqp import call_progress_bar


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

        This function assumes that each partition in `partitions` contains a single block.

        Parameters
        ----------
        partitions : np.ndarray
            NumPy array with ``PandasDataframePartition``-s.

        Returns
        -------
        list
            The objects wrapped by `partitions`.
        """
        assert all(
            [len(partition.list_of_blocks) == 1 for partition in partitions]
        ), "Implementation assumes that each partition contains a signle block."
        return RayWrapper.materialize(
            [partition.list_of_blocks[0] for partition in partitions]
        )

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
        blocks = [
            block for partition in partitions for block in partition.list_of_blocks
        ]
        wait(blocks)


def _make_wrapped_method(name: str):
    """
    Define new attribute that should work with progress bar.

    Parameters
    ----------
    name : str
        Name of `GenericRayDataframePartitionManager` attribute that should be reused.

    Notes
    -----
    - `classmethod` decorator shouldn't be applied twice, so we refer to `__func__` attribute.
    - New attribute is defined for `PandasOnRayDataframePartitionManager`.
    """
    setattr(
        PandasOnRayDataframePartitionManager,
        name,
        classmethod(
            progress_bar_wrapper(
                getattr(GenericRayDataframePartitionManager, name).__func__
            )
        ),
    )


for method in (
    "map_partitions",
    "lazy_map_partitions",
    "map_axis_partitions",
    "_apply_func_to_list_of_partitions",
    "apply_func_to_select_indices",
    "apply_func_to_select_indices_along_full_axis",
    "apply_func_to_indices_both_axis",
    "n_ary_operation",
):
    _make_wrapped_method(method)
