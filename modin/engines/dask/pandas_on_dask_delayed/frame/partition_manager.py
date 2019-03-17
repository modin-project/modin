from modin.engines.base.frame.partition_manager import BaseFrameManager
from .axis_partition import DaskFrameColumnPartition, DaskFrameRowPartition
from .partition import DaskFramePartition


class DaskFrameManager(BaseFrameManager):
    """This class implements the interface in `BaseFrameManager`."""

    # This object uses RayRemotePartition objects as the underlying store.
    _partition_class = DaskFramePartition
    _column_partitions_class = DaskFrameColumnPartition
    _row_partition_class = DaskFrameRowPartition

    def __init__(self, partitions):
        self.partitions = partitions
