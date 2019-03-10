from modin.engines.base.frame.partition_manager import BaseFramePartitionManager
from .axis_partition import DaskFrameFullColumnPartition, DaskFrameFullRowPartition
from .partition import DaskFramePartition


class DaskFramePartitionManager(BaseFramePartitionManager):
    """This class implements the interface in `BaseFramePartitionManager`."""

    # This object uses RayRemotePartition objects as the underlying store.
    _partition_class = DaskFramePartition
    _column_partitions_class = DaskFrameFullColumnPartition
    _row_partition_class = DaskFrameFullRowPartition

    def __init__(self, partitions):
        self.partitions = partitions
