import numpy as np

from modin.engines.base.frame.partition_manager import BaseFrameManager
from modin import __execution_engine__

if __execution_engine__ == "Ray":
    import ray


class RayFrameManager(BaseFrameManager):
    """This method implements the interface in `BaseFrameManager`."""

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
