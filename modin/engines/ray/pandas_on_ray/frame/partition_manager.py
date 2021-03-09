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
    """Wraps computation function inside a progress bar. Spawns another thread
        which displays a progress bar showing estimated completion time.

    Args:
        f: The name of the function to be wrapped.

    Returns:
        A new BaseFrameManager object, the type of object that called this.
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
    """This method implements the interface in `BaseFrameManager`."""

    # This object uses RayRemotePartition objects as the underlying store.
    _partition_class = PandasOnRayFramePartition
    _column_partitions_class = PandasOnRayFrameColumnPartition
    _row_partition_class = PandasOnRayFrameRowPartition

    @classmethod
    def get_indices(cls, axis, partitions, index_func=None):
        """
        This gets the internal indices stored in the partitions.

        Parameters
        ----------
            axis : 0 or 1
                This axis to extract the labels (0 - index, 1 - columns).
            partitions : NumPy array
                The array of partitions from which need to extract the labels.
            index_func : callable
                The function to be used to extract the function.

        Returns
        -------
        Index
            A Pandas Index object.

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
        return super(PandasOnRayFrameManager, cls).map_partitions(partitions, map_func)

    @classmethod
    @progress_bar_wrapper
    def lazy_map_partitions(cls, partitions, map_func):
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
        return super(PandasOnRayFrameManager, cls).map_axis_partitions(
            axis, partitions, map_func, keep_partitioning, lengths, enumerate_partitions
        )

    @classmethod
    @progress_bar_wrapper
    def _apply_func_to_list_of_partitions(cls, func, partitions, **kwargs):
        return super(PandasOnRayFrameManager, cls)._apply_func_to_list_of_partitions(
            func, partitions, **kwargs
        )

    @classmethod
    @progress_bar_wrapper
    def apply_func_to_select_indices(
        cls, axis, partitions, func, indices, keep_remaining=False
    ):
        return super(PandasOnRayFrameManager, cls).apply_func_to_select_indices(
            axis, partitions, func, indices, keep_remaining=keep_remaining
        )

    @classmethod
    @progress_bar_wrapper
    def apply_func_to_select_indices_along_full_axis(
        cls, axis, partitions, func, indices, keep_remaining=False
    ):
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
        return super(PandasOnRayFrameManager, cls).binary_operation(
            axis, left, func, right
        )
