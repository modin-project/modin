import numpy as np

from modin.engines.ray.generic.frame.partition_manager import RayFrameManager
from .axis_partition import (
    PandasOnRayFrameColumnPartition,
    PandasOnRayFrameRowPartition,
)
from .partition import PandasOnRayFramePartition
from modin.error_message import ErrorMessage
from modin import __execution_engine__

if __execution_engine__ == "Ray":
    import ray

from tqdm import tqdm_notebook, tqdm
from progressbar import ProgressBar
import threading

class PandasOnRayFrameManager(RayFrameManager):
    """This method implements the interface in `BaseFrameManager`."""

    # This object uses RayRemotePartition objects as the underlying store.
    _partition_class = PandasOnRayFramePartition
    _column_partitions_class = PandasOnRayFrameColumnPartition
    _row_partition_class = PandasOnRayFrameRowPartition

    @classmethod
    def get_indices(cls, axis, partitions, index_func=None):
        """This gets the internal indices stored in the partitions.

        Note: These are the global indices of the object. This is mostly useful
            when you have deleted rows/columns internally, but do not know
            which ones were deleted.

        Args:
            axis: This axis to extract the labels. (0 - index, 1 - columns).
            index_func: The function to be used to extract the function.
            old_blocks: An optional previous object that this object was
                created from. This is used to compute the correct offsets.

        Returns:
            A Pandas Index object.
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
    def groupby_reduce(
        cls, axis, partitions, by, map_func, reduce_func
    ):  # pragma: no cover
        @ray.remote
        def func(df, other, map_func, call_queue_df=[], call_queue_other=[]):
            if len(call_queue_df) > 0:
                for call, kwargs in call_queue_df:
                    df = call(df, **kwargs)
            if len(call_queue_other) > 0:
                for call, kwargs in call_queue_other:
                    other = call(other, **kwargs)
            return map_func(df, other)

        map_func = ray.put(map_func)
        by_parts = np.squeeze(by)
        if len(by_parts.shape) == 0:
            by_parts = np.array([by_parts.item()])
        new_partitions = np.array(
            [
                [
                    PandasOnRayFramePartition(
                        func.remote(
                            part.oid,
                            by_parts[col_idx].oid if axis else by_parts[row_idx].oid,
                            map_func,
                            part.call_queue,
                            by_parts[col_idx].call_queue
                            if axis
                            else by_parts[row_idx].call_queue,
                        )
                    )
                    for col_idx, part in enumerate(partitions[row_idx])
                ]
                for row_idx in range(len(partitions))
            ]
        )
        return cls.map_axis_partitions(axis, new_partitions, reduce_func)

    
    @classmethod
    @ray.remote
    # This doesn't work because the progress bar does not output correctly
    def run_progress_bar(futures):
        pbar = ProgressBar()
        for i in pbar(range(1, len(futures)+1)):
            ray.wait(futures, i)
    


    @classmethod
    def progress_bar_wrapper(cls, function, *args, **kwargs):
        """Wraps computation function inside a progress bar. Displays a progress
            bar showing estimated completion time.

        Args:
            function: The name of the function to be wrapped.

        Returns:
            A new BaseFrameManager object, the type of object that called this.
        """

        result_parts = getattr(super(PandasOnRayFrameManager, cls), function)(*args, **kwargs)
        futures = [[x.oid for row in result_parts for x in row]]
        
        def display_progress_bar(futures):
            for i in tqdm_notebook(range(1, len(futures)+1)):
                ray.wait(futures, i)
        x = threading.Thread(target=display_progress_bar, args=futures).start()
        
        '''
        ### This doesn't work ###
        cls.run_progress_bar.remote(futures)
        '''

        return result_parts

    @classmethod
    def map_partitions(cls, partitions, map_func):
        return cls.progress_bar_wrapper("map_partitions", partitions, map_func)

    @classmethod
    def map_axis_partitions(cls, axis, partitions, map_func):
        return cls.progress_bar_wrapper("map_axis_partitions", axis, partitions, map_func)

    """@classmethod
    def concat(cls, axis, left_parts, right_parts):
        return cls.progress_bar_wrapper("concat", axis, left_parts, right_parts)"""
    
    @classmethod
    def _apply_func_to_list_of_partitions(cls, func, partitions, **kwargs):
        return cls.progress_bar_wrapper("_apply_func_to_list_of_partitions", func, partitions, **kwargs)

    @classmethod
    def apply_func_to_select_indices(
        cls, axis, partitions, func, indices, keep_remaining=False
    ):
        return cls.progress_bar_wrapper("apply_func_to_select_indices", axis, partitions, func, indices, keep_remaining)

    @classmethod
    def apply_func_to_select_indices_along_full_axis(
        cls, axis, partitions, func, indices, keep_remaining=False
    ):
        return cls.progress_bar_wrapper("apply_func_to_select_indices_along_full_axis", axis, partitions, func, indices, keep_remaining)

    @classmethod
    def apply_func_to_indices_both_axis(
        cls,
        partitions,
        func,
        row_partitions_list,
        col_partitions_list,
        item_to_distribute=None,
    ):
        return cls.progress_bar_wrapper("apply_func_to_indices_both_axis", partitions, func, row_partitions_list, col_partitions_list, item_to_distribute=None)

    @classmethod
    def binary_operation(cls, axis, left, func, right):
        return cls.progress_bar_wrapper("apply_func_to_select_indices", axis, left, func, right)
