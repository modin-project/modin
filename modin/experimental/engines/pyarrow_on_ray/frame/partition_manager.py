from modin.engines.ray.generic.frame.partition_manager import RayFrameManager
from .axis_partition import (
    PyarrowOnRayFrameColumnPartition,
    PyarrowOnRayFrameRowPartition,
)
from .partition import PyarrowOnRayFramePartition


class PyarrowOnRayFrameManager(RayFrameManager):
    """This method implements the interface in `BaseFrameManager`."""

    # This object uses RayRemotePartition objects as the underlying store.
    _partition_class = PyarrowOnRayFramePartition
    _column_partitions_class = PyarrowOnRayFrameColumnPartition
    _row_partition_class = PyarrowOnRayFrameRowPartition
