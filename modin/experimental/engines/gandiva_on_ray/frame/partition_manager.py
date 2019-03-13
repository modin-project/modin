from modin.engines.ray.generic.frame.partition_manager import RayFrameManager
from .axis_partition import (
    GandivaOnRayFrameColumnPartition,
    GandivaOnRayFrameRowPartition,
)
from .partition import GandivaOnRayFramePartition


class RayFrameManager(RayFrameManager):
    """This method implements the interface in `BaseFrameManager`."""

    # This object uses RayRemotePartition objects as the underlying store.
    _partition_class = GandivaOnRayFramePartition
    _column_partitions_class = GandivaOnRayFrameColumnPartition
    _row_partition_class = GandivaOnRayFrameRowPartition
