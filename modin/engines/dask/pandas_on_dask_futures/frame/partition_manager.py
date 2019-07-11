from modin.engines.base.frame.partition_manager import BaseFrameManager
from .axis_partition import PandasOnDaskFrameColumnPartition, PandasOnDaskFrameRowPartition
from .partition import PandasOnDaskFramePartition


class DaskFrameManager(BaseFrameManager):
    """This class implements the interface in `BaseFrameManager`."""

    # This object uses RayRemotePartition objects as the underlying store.
    _partition_class = PandasOnDaskFramePartition
    _column_partitions_class = PandasOnDaskFrameColumnPartition
    _row_partition_class = PandasOnDaskFrameRowPartition

    def __init__(self, partitions):
        self.partitions = partitions
