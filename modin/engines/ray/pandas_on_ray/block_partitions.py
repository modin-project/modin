import ray
import numpy as np

from modin.engines.base.block_partitions import BaseBlockPartitions
from .axis_partition import PandasOnRayColumnPartition, PandasOnRayRowPartition
from .remote_partition import PandasOnRayRemotePartition


class RayBlockPartitions(BaseBlockPartitions):
    """This method implements the interface in `BaseBlockPartitions`."""

    # This object uses RayRemotePartition objects as the underlying store.
    _partition_class = PandasOnRayRemotePartition
    _column_partitions_class = PandasOnRayColumnPartition
    _row_partition_class = PandasOnRayRowPartition

    def __init__(self, partitions):
        self.partitions = partitions

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
            self._lengths_cache = np.array(
                ray.get([obj.length().oid for obj in self._partitions_cache.T[0]])
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
            self._widths_cache = np.array(
                ray.get([obj.width().oid for obj in self._partitions_cache[0]])
                if len(self._partitions_cache) > 0
                else []
            )
        return self._widths_cache
