from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas
import ray

from modin.engines.base.frame.axis_partition import PandasFrameAxisPartition
from .partition import PandasOnRayFramePartition


class PandasOnRayFrameAxisPartition(PandasFrameAxisPartition):
    def __init__(self, list_of_blocks):
        # Unwrap from BaseFramePartition object for ease of use
        for obj in list_of_blocks:
            obj.drain_call_queue()
        self.list_of_blocks = [obj.oid for obj in list_of_blocks]

    partition_type = PandasOnRayFramePartition
    instance_type = ray.ObjectID

    @classmethod
    def deploy_axis_func(
        cls, axis, func, num_splits, kwargs, maintain_partitioning, *partitions
    ):
        return deploy_ray_func._remote(
            args=(
                PandasFrameAxisPartition.deploy_axis_func,
                axis,
                func,
                num_splits,
                kwargs,
                maintain_partitioning,
            )
            + tuple(partitions),
            num_return_vals=num_splits * 3,
        )

    @classmethod
    def deploy_func_between_two_axis_partitions(
        cls, axis, func, num_splits, len_of_left, kwargs, *partitions
    ):
        return deploy_ray_func._remote(
            args=(
                PandasFrameAxisPartition.deploy_func_between_two_axis_partitions,
                axis,
                func,
                num_splits,
                len_of_left,
                kwargs,
            )
            + tuple(partitions),
            num_return_vals=num_splits * 3,
        )

    def _wrap_partitions(self, partitions):
        return [
            self.partition_type(
                partitions[i],
                self.partition_type(partitions[i + 1]),
                self.partition_type(partitions[i + 2]),
            )
            for i in range(0, len(partitions), 3)
        ]


class PandasOnRayFrameColumnPartition(PandasOnRayFrameAxisPartition):
    """The column partition implementation for Ray. All of the implementation
        for this class is in the parent class, and this class defines the axis
        to perform the computation over.
    """

    axis = 0


class PandasOnRayFrameRowPartition(PandasOnRayFrameAxisPartition):
    """The row partition implementation for Ray. All of the implementation
        for this class is in the parent class, and this class defines the axis
        to perform the computation over.
    """

    axis = 1


@ray.remote
def deploy_ray_func(func, *args):  # pragma: no cover
    """Run a function on a remote partition.

    Note: Ray functions are not detected by codecov (thus pragma: no cover)

    Args:
        func: The function to run.

    Returns:
        The result of the function `func`.
    """
    result = func(*args)
    if isinstance(result, pandas.DataFrame):
        return result, len(result), len(result.columns)
    elif all(isinstance(r, pandas.DataFrame) for r in result):
        return [i for r in result for i in [r, len(r), len(r.columns)]]
    else:
        return [i for r in result for i in [r, None, None]]
