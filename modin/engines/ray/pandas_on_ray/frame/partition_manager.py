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

    def groupby_reduce(self, axis, by, map_func, reduce_func):
        @ray.remote
        def func(df, other):
            return map_func(df, other)

        map_func = ray.put(map_func)
        p = np.squeeze(by.partitions)
        for obj in p:
                obj.drain_call_queue()
        for part in self.partitions:
            for obj in part:
                obj.drain_call_queue()
        new_partitions = self.__constructor__(
            np.array(
                [
                    [
                        PandasOnRayFramePartition(func.remote(part.oid, p[i].oid))
                        for part in self.partitions[i]
                    ]
                    for i in range(len(self.partitions))
                ]
            )
        )
        return new_partitions.map_across_full_axis(0, reduce_func)
