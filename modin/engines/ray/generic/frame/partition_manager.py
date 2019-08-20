import ray
import numpy as np
from ray.worker import RayTaskError

from modin.engines.base.frame.partition_manager import BaseFrameManager
from modin.engines.ray.utils import handle_ray_task_error


class RayFrameManager(BaseFrameManager):
    """This method implements the interface in `BaseFrameManager`."""

    def __init__(self, partitions, block_lengths=None, block_widths=None):
        if isinstance(partitions, list):
            partitions = np.array(partitions)
        self.partitions = partitions
        self._lengths_cache = block_lengths
        self._widths_cache = block_widths

    # We override these for performance reasons.
    # Lengths of the blocks
    _lengths_cache = None
    # Widths of the blocks
    _widths_cache = None

    # These are set up as properties so that we only use them when we need
    # them. We also do not want to trigger this computation on object creation.
    @staticmethod
    def block_lengths(partitions):
        """Gets the lengths of the blocks.

        Note: This works with the property structure `_lengths_cache` to avoid
            having to recompute these values each time they are needed.
        """
        if not isinstance(partitions[0][0].length(), int):
            try:
                # The first column will have the correct lengths. We have an
                # invariant that requires that all blocks be the same length in a
                # row of blocks.
                return np.array(
                    ray.get([obj.length().oid for obj in partitions.T[0]])
                    if len(partitions.T) > 0
                    else []
                )
            except RayTaskError as e:
                handle_ray_task_error(e)
            except AttributeError:
                return np.array(
                    [
                        obj.length()
                        if isinstance(obj.length(), int)
                        else ray.get(obj.length().oid)
                        for obj in partitions.T[0]
                    ]
                )
        else:
            return np.array(
                [
                    obj.length()
                    if isinstance(obj.length(), int)
                    else ray.get(obj.length().oid)
                    for obj in partitions.T[0]
                ]
            )

    @staticmethod
    def block_widths(partitions):
        """Gets the widths of the blocks.

        Note: This works with the property structure `_widths_cache` to avoid
            having to recompute these values each time they are needed.
        """
        if not isinstance(partitions[0][0].width(), int):
            try:
                # The first column will have the correct lengths. We have an
                # invariant that requires that all blocks be the same width in a
                # column of blocks.
                return np.array(
                    ray.get([obj.width().oid for obj in partitions[0]])
                    if len(partitions) > 0
                    else []
                )
            except RayTaskError as e:
                handle_ray_task_error(e)
            except AttributeError:
                return np.array(
                    [
                        obj.width()
                        if isinstance(obj.width(), int)
                        else ray.get(obj.width().oid)
                        for obj in partitions[0]
                    ]
                )
        else:
            return np.array(
                [
                    obj.width()
                    if isinstance(obj.width(), int)
                    else ray.get(obj.width().oid)
                    for obj in partitions[0]
                ]
            )

    @classmethod
    def to_numpy(cls, partitions):
        """Convert this object into a NumPy Array from the partitions.

        Returns:
            A NumPy Array
        """
        parts = ray.get(
            [
                obj.apply(lambda df: df.to_numpy()).oid
                for row in partitions
                for obj in row
            ]
        )
        n = partitions.shape[1]
        parts = [parts[i * n : (i + 1) * n] for i in list(range(partitions.shape[0]))]

        arr = np.block(parts)
        return arr
