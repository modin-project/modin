# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas
from distributed.client import _get_global_client

from modin.engines.base.frame.axis_partition import PandasFrameAxisPartition
from .partition import PandasOnDaskFramePartition




class PandasOnMultiprocessFrameAxisPartition(PandasFrameAxisPartition):

    def __init__(self, list_of_blocks):
        # Unwrap from BaseFramePartition object for ease of use
        for obj in list_of_blocks:
            obj.drain_call_queue()
        self.list_of_blocks = [obj for obj in list_of_blocks]

    partition_type = PandasOnDaskFramePartition
    instance_type = pandas.DataFrame

    @classmethod
    def deploy_axis_func(
        cls, axis, func, num_splits, kwargs, maintain_partitioning, *partitions
    ):
        client = _get_global_client()
        axis_result = client.submit(PandasFrameAxisPartition.deploy_axis_func, axis, func, num_splits, kwargs, maintain_partitioning, *partitions)
        # We have to do this to split it back up. It is already split, but we need to
        # get futures for each.
        return cls._wrap_partitions([client.submit(lambda l: l[i], axis_result) for i in range(num_splits)])

    @classmethod
    def deploy_func_between_two_axis_partitions(
        cls, axis, func, num_splits, len_of_left, kwargs, *partitions
    ):
        client = _get_global_client()
        axis_result = client.submit(PandasFrameAxisPartition.deploy_func_between_two_axis_partitions, axis, func, num_splits, len_of_left, kwargs, *partitions)
        # We have to do this to split it back up. It is already split, but we need to
        # get futures for each.
        return cls._wrap_partitions([client.submit(lambda l: l[i], axis_result) for i in range(num_splits)])

    @classmethod
    def _wrap_partitions(cls, partitions):
        if isinstance(partitions, cls.instance_type):
            return [cls.partition_type(partitions)]
        else:
            return [cls.partition_type.put(obj) for obj in partitions]


class PandasOnDaskFrameColumnPartition(PandasOnMultiprocessFrameAxisPartition):
    """The column partition implementation for Multiprocess. All of the implementation
        for this class is in the parent class, and this class defines the axis
        to perform the computation over.
    """

    axis = 0


class PandasOnDaskFrameRowPartition(PandasOnMultiprocessFrameAxisPartition):
    """The row partition implementation for Multiprocess. All of the implementation
        for this class is in the parent class, and this class defines the axis
        to perform the computation over.
    """

    axis = 1
