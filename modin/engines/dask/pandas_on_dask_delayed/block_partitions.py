from modin.engines.base.block_partitions import BaseBlockPartitions
from .axis_partition import DaskColumnPartition, DaskRowPartition
from .remote_partition import DaskRemotePartition


class DaskBlockPartitions(BaseBlockPartitions):
    """This class implements the interface in `BaseBlockPartitions`."""

    # This object uses RayRemotePartition objects as the underlying store.
    _partition_class = DaskRemotePartition
    _column_partitions_class = DaskColumnPartition
    _row_partition_class = DaskRowPartition

    def __init__(self, partitions):
        self.partitions = partitions
