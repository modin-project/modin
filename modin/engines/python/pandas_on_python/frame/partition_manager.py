from modin.engines.base.frame.partition_manager import BaseFrameManager
from .axis_partition import (
    PandasOnPythonFrameColumnPartition,
    PandasOnPythonFrameRowPartition,
)
from .partition import PandasOnPythonFramePartition


class PythonFrameManager(BaseFrameManager):
    """This method implements the interface in `BaseFrameManager`."""

    # This object uses RayRemotePartition objects as the underlying store.
    _partition_class = PandasOnPythonFramePartition
    _column_partitions_class = PandasOnPythonFrameColumnPartition
    _row_partition_class = PandasOnPythonFrameRowPartition

    def __init__(self, partitions):
        self.partitions = partitions

    def put(self, obj):
        return obj

    # We override these for performance reasons.
    # Lengths of the blocks
    _lengths_cache = None
    # Widths of the blocks
    _widths_cache = None

    # These are set up as properties so that we only use them when we need
    # them. We also do not want to trigger this computation on object creation.
    @property
    def block_lengths(self):
        """Gets the lengths of the blocks.

        Note: This works with the property structure `_lengths_cache` to avoid
            having to recompute these values each time they are needed.
        """
        if self._lengths_cache is None:
            # The first column will have the correct lengths. We have an
            # invariant that requires that all blocks be the same length in a
            # row of blocks.
            self._lengths_cache = (
                [obj.length() for obj in self._partitions_cache.T[0]]
                if len(self._partitions_cache.T) > 0
                else []
            )
        return self._lengths_cache

    @property
    def block_widths(self):
        """Gets the widths of the blocks.

        Note: This works with the property structure `_widths_cache` to avoid
            having to recompute these values each time they are needed.
        """
        if self._widths_cache is None:
            # The first column will have the correct lengths. We have an
            # invariant that requires that all blocks be the same width in a
            # column of blocks.
            self._widths_cache = (
                [obj.width() for obj in self._partitions_cache[0]]
                if len(self._partitions_cache) > 0
                else []
            )
        return self._widths_cache
