import ray
import numpy as np
from ray.worker import RayTaskError

from modin.engines.base.frame.partition_manager import BaseFrameManager
from modin.engines.ray.utils import handle_ray_task_error


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
            if not isinstance(self._partitions_cache[0][0].length(), int):
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
                            if isinstance(obj.length(), int)
                            else ray.get(obj.length().oid)
                            for obj in self._partitions_cache.T[0]
                        ]
                    )
            else:
                self._lengths_cache = np.array(
                    [
                        obj.length()
                        if isinstance(obj.length(), int)
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
            if not isinstance(self._partitions_cache[0][0].width(), int):
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
                            if isinstance(obj.width(), int)
                            else ray.get(obj.width().oid)
                            for obj in self._partitions_cache[0]
                        ]
                    )
            else:
                self._widths_cache = np.array(
                    [
                        obj.width()
                        if isinstance(obj.width(), int)
                        else ray.get(obj.width().oid)
                        for obj in self._partitions_cache[0]
                    ]
                )
        return self._widths_cache

    def _broadcast_values(self, axis, values):
        """Splits the values to the size of the partitions.

        Args:
            axis: Axis value of the axis the function is being applied on.
            values: List-like of values to broadcast.
            block_idx: Index of the partition to give the values to.

        Returns:
            A list-like of values to be associated with the block_idx.
        """
        axis_lengths = self.block_widths if axis else self.block_lengths
        broadcast_values = []
        for block_idx in range(len(axis_lengths)):
            cumulative_axis = np.insert(np.cumsum(axis_lengths), 0, 0)
            broadcast_values.append(ray.put(values[
                cumulative_axis[block_idx] : cumulative_axis[block_idx] + axis_lengths[block_idx]
            ]))
            print(values[
                cumulative_axis[block_idx] : cumulative_axis[block_idx] + axis_lengths[block_idx]
            ])
        return broadcast_values
