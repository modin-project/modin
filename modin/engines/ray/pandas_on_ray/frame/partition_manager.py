import numpy as np
import ray

from modin.engines.ray.generic.frame.partition_manager import RayFrameManager
from .axis_partition import (
    PandasOnRayFrameColumnPartition,
    PandasOnRayFrameRowPartition,
)
from .partition import PandasOnRayFramePartition


class PandasOnRayFrameManager(RayFrameManager):
    """This method implements the interface in `BaseFrameManager`."""

    # This object uses RayRemotePartition objects as the underlying store.
    _partition_class = PandasOnRayFramePartition
    _column_partitions_class = PandasOnRayFrameColumnPartition
    _row_partition_class = PandasOnRayFrameRowPartition

    def groupby_reduce(self, axis, by, map_func, reduce_func):  # pragma: no cover
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
        by_parts = np.squeeze(by.partitions)
        new_partitions = self.__constructor__(
            np.array(
                [
                    [
                        PandasOnRayFramePartition(
                            func.remote(
                                part.oid,
                                by_parts[i].oid,
                                map_func,
                                part.call_queue,
                                by_parts[i].call_queue,
                            )
                        )
                        for part in self.partitions[i]
                    ]
                    for i in range(len(self.partitions))
                ]
            )
        )
        return new_partitions.map_across_full_axis(0, reduce_func)
