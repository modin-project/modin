# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas

from modin.engines.base.frame.axis_partition import PandasFrameAxisPartition
from .partition import PandasOnMultiprocessFramePartition


def deserialize(obj):
    import cloudpickle as cp
    if isinstance(obj, bytes):
        return cp.loads(obj)
    return obj


class PandasOnMultiprocessFrameAxisPartition(PandasFrameAxisPartition):
    def __init__(self, list_of_blocks):
        # Unwrap from BaseFramePartition object for ease of use
        for obj in list_of_blocks:
            obj.drain_call_queue()
        self.list_of_blocks = [obj for obj in list_of_blocks]

    partition_type = PandasOnMultiprocessFramePartition
    instance_type = pandas.DataFrame

    @classmethod
    def deploy_axis_func(
        cls, axis, func, num_splits, kwargs, maintain_partitioning, *partitions
    ):
        func = deserialize(func)
        partitions = [part.get() for part in partitions]
        return PandasFrameAxisPartition.deploy_axis_func(axis, func, num_splits, kwargs, maintain_partitioning, *partitions)

    @classmethod
    def deploy_func_between_two_axis_partitions(
        cls, axis, func, num_splits, len_of_left, kwargs, *partitions
    ):
        func = deserialize(func)
        partitions = [part.get() for part in partitions]
        return PandasFrameAxisPartition.deploy_func_between_two_axis_partitions(axis, func, num_splits, len_of_left, kwargs, *partitions)

    def _wrap_partitions(self, partitions):
        if isinstance(partitions, self.instance_type):
            return [self.partition_type(partitions)]
        else:
            return [self.partition_type.put(obj) for obj in partitions]


class PandasOnMultiprocessFrameColumnPartition(PandasOnMultiprocessFrameAxisPartition):
    """The column partition implementation for Multiprocess. All of the implementation
        for this class is in the parent class, and this class defines the axis
        to perform the computation over.
    """

    axis = 0


class PandasOnMultiprocessFrameRowPartition(PandasOnMultiprocessFrameAxisPartition):
    """The row partition implementation for Multiprocess. All of the implementation
        for this class is in the parent class, and this class defines the axis
        to perform the computation over.
    """

    axis = 1
