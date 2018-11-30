from .axis_partition import BaseAxisPartition


class DaskAxisPartition(BaseAxisPartition):
    """Dask implementation for Column and Row partitions"""

    pass


class DaskColumnPartition(DaskAxisPartition):
    """Dask implementation for Column partitions"""

    pass


class DaskRowPartition(DaskAxisPartition):
    """Dask implementation for Row partitions"""

    pass
