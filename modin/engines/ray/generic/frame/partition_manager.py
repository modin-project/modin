import ray
import numpy as np
from ray.worker import RayTaskError

from modin.engines.base.frame.partition_manager import BaseFrameManager
from modin.engines.ray.utils import handle_ray_task_error
from pandas.api.types import is_integer


class RayFrameManager(BaseFrameManager):
    """This method implements the interface in `BaseFrameManager`."""

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
            if not is_integer(self._partitions_cache[0][0].length()):
                try:
                    # The first column will have the correct lengths. We have an
                    # invariant that requires that all blocks be the same length in a
                    # row of blocks.
                    self._lengths_cache = np.array(
                        ray.get(
                            [obj.length().oid for obj in self._partitions_cache.T[0]]
                        )
                        if len(self._partitions_cache.T) > 0
                        else []
                    )
                except RayTaskError as e:
                    handle_ray_task_error(e)
                except AttributeError:
                    self._lengths_cache = np.array(
                        [
                            obj.length()
                            if is_integer(obj.length())
                            else ray.get(obj.length().oid)
                            for obj in self._partitions_cache.T[0]
                        ]
                    )
            else:
                self._lengths_cache = np.array(
                    [
                        obj.length()
                        if is_integer(obj.length())
                        else ray.get(obj.length().oid)
                        for obj in self._partitions_cache.T[0]
                    ]
                )
        return self._lengths_cache

    @property
    def block_widths(self):
        """Gets the widths of the blocks.

        Note: This works with the property structure `_widths_cache` to avoid
            having to recompute these values each time they are needed.
        """
        if self._widths_cache is None:
            if not is_integer(self._partitions_cache[0][0].width()):
                try:
                    # The first column will have the correct lengths. We have an
                    # invariant that requires that all blocks be the same width in a
                    # column of blocks.
                    self._widths_cache = np.array(
                        ray.get([obj.width().oid for obj in self._partitions_cache[0]])
                        if len(self._partitions_cache) > 0
                        else []
                    )
                except RayTaskError as e:
                    handle_ray_task_error(e)
                except AttributeError:
                    self._widths_cache = np.array(
                        [
                            obj.width()
                            if is_integer(obj.width())
                            else ray.get(obj.width().oid)
                            for obj in self._partitions_cache[0]
                        ]
                    )
            else:
                self._widths_cache = np.array(
                    [
                        obj.width()
                        if is_integer(obj.width())
                        else ray.get(obj.width().oid)
                        for obj in self._partitions_cache[0]
                    ]
                )
        return self._widths_cache

    def to_numpy(self, is_transposed=False):
        """Convert this object into a NumPy Array from the partitions.

        Returns:
            A NumPy Array
        """
        parts = ray.get(
            [
                obj.apply(lambda df: df.to_numpy()).oid
                for row in self.partitions
                for obj in row
            ]
        )
        n = self.partitions.shape[1]
        parts = [
            parts[i * n : (i + 1) * n] for i in list(range(self.partitions.shape[0]))
        ]

        arr = np.block(parts)
        if is_transposed:
            return arr.T
        return arr
