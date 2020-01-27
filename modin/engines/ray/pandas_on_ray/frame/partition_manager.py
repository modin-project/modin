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

import numpy as np

from modin.engines.ray.generic.frame.partition_manager import RayFrameManager
from .axis_partition import (
    PandasOnRayFrameColumnPartition,
    PandasOnRayFrameRowPartition,
)
from .partition import PandasOnRayFramePartition
from modin.error_message import ErrorMessage
import pandas

import ray
from tqdm import tqdm_notebook
import threading
import os


@ray.remote
def func(df, apply_func, call_queue_df=None, call_queues_other=None, *others):
    if call_queue_df is not None and len(call_queue_df) > 0:
        for call, kwargs in call_queue_df:
            if isinstance(call, ray.ObjectID):
                call = ray.get(call)
            if isinstance(kwargs, ray.ObjectID):
                kwargs = ray.get(kwargs)
            df = call(df, **kwargs)
    new_others = np.empty(shape=len(others), dtype=object)
    for i, call_queue_other in enumerate(call_queues_other):
        other = others[i]
        if call_queue_other is not None and len(call_queue_other) > 0:
            for call, kwargs in call_queue_other:
                if isinstance(call, ray.ObjectID):
                    call = ray.get(call)
                if isinstance(kwargs, ray.ObjectID):
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
    progress_bar = None
    bar_count = 0

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
    def progress_bar_wrapper(cls, function, name=None, *args, **kwargs):
        """Wraps computation function inside a progress bar. Spawns another thread
            which displays a progress bar showing estimated completion time.

        Args:
            function: The name of the function to be wrapped.

        Returns:
            A new BaseFrameManager object, the type of object that called this.
        """
        result_parts = getattr(super(PandasOnRayFrameManager, cls), function)(
            *args, **kwargs
        )
        futures = [x.oid for row in result_parts for x in row]

        def display_progress_bar(futures):
            for i in tqdm_notebook(range(1, len(futures) + 1), desc=name):
                ray.wait(futures, i)

        if (
            "MODIN_PROGRESS_BAR" in os.environ
            and os.environ["MODIN_PROGRESS_BAR"] == "True"
            and name != None
        ):
            threading.Thread(target=display_progress_bar, args=(futures,)).start()
        return result_parts

    @classmethod
    def map_partitions(cls, partitions, map_func, name=None):
        return cls.progress_bar_wrapper("map_partitions", name, partitions, map_func)

    @classmethod
    def lazy_map_partitions(cls, partitions, map_func, name=None):
        return cls.progress_bar_wrapper(
            "lazy_map_partitions", name, partitions, map_func
        )

    @classmethod
    def map_axis_partitions(
        cls, axis, partitions, map_func, keep_partitioning=False, name=None
    ):
        return cls.progress_bar_wrapper(
            "map_axis_partitions", name, axis, partitions, map_func, keep_partitioning
        )

    @classmethod
    def _apply_func_to_list_of_partitions(cls, func, partitions, **kwargs):
        return cls.progress_bar_wrapper(
            "_apply_func_to_list_of_partitions", None, func, partitions, **kwargs
        )

    @classmethod
    def apply_func_to_select_indices(
        cls, axis, partitions, func, indices, keep_remaining=False, name=None
    ):
        return cls.progress_bar_wrapper(
            "apply_func_to_select_indices",
            name,
            axis,
            partitions,
            func,
            indices,
            keep_remaining,
        )

    @classmethod
    def apply_func_to_select_indices_along_full_axis(
        cls, axis, partitions, func, indices, keep_remaining=False, name=None
    ):
        return cls.progress_bar_wrapper(
            "apply_func_to_select_indices_along_full_axis",
            name,
            axis,
            partitions,
            func,
            indices,
            keep_remaining,
        )

    @classmethod
    def apply_func_to_indices_both_axis(
        cls,
        partitions,
        func,
        row_partitions_list,
        col_partitions_list,
        item_to_distribute=None,
        name=None,
    ):
        return cls.progress_bar_wrapper(
            "apply_func_to_indices_both_axis",
            name,
            partitions,
            func,
            row_partitions_list,
            col_partitions_list,
            item_to_distribute=None,
        )

    @classmethod
    def binary_operation(cls, axis, left, func, right, name=None):
        return cls.progress_bar_wrapper(
            "binary_operation", name, axis, left, func, right
        )
