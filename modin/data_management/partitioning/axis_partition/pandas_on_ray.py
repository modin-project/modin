from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas
import ray

from .axis_partition import BaseAxisPartition
from ...partitioning.remote_partition import PandasOnRayRemotePartition
from .utils import split_result_of_axis_func_pandas


class PandasOnRayAxisPartition(BaseAxisPartition):
    def __init__(self, list_of_blocks):
        # Unwrap from BaseRemotePartition object for ease of use
        self.list_of_blocks = [obj.oid for obj in list_of_blocks]

    def apply(self, func, num_splits=None, other_axis_partition=None, **kwargs):
        """Applies func to the object in the plasma store.

        See notes in Parent class about this method.

        Args:
            func: The function to apply.
            num_splits: The number of times to split the result object.
            other_axis_partition: Another `PandasOnRayAxisPartition` object to apply to
                func with this one.

        Returns:
            A list of `RayRemotePartition` objects.
        """
        if num_splits is None:
            num_splits = len(self.list_of_blocks)

        if other_axis_partition is not None:
            return [
                PandasOnRayRemotePartition(obj)
                for obj in deploy_ray_func_between_two_axis_partitions._submit(
                    args=(self.axis, func, num_splits, len(self.list_of_blocks), kwargs)
                    + tuple(self.list_of_blocks + other_axis_partition.list_of_blocks),
                    num_return_vals=num_splits,
                )
            ]

        args = [self.axis, func, num_splits, kwargs]
        args.extend(self.list_of_blocks)
        return [
            PandasOnRayRemotePartition(obj)
            for obj in deploy_ray_axis_func._submit(args, num_return_vals=num_splits)
        ]

    def shuffle(self, func, num_splits=None, **kwargs):
        """Shuffle the order of the data in this axis based on the `func`.

        Extends `BaseAxisPartition.shuffle`.

        :param func:
        :param num_splits:
        :param kwargs:
        :return:
        """
        if num_splits is None:
            num_splits = len(self.list_of_blocks)

        args = [self.axis, func, num_splits, kwargs]
        args.extend(self.list_of_blocks)
        return [
            PandasOnRayRemotePartition(obj)
            for obj in deploy_ray_axis_func._submit(args, num_return_vals=num_splits)
        ]


class PandasOnRayColumnPartition(PandasOnRayAxisPartition):
    """The column partition implementation for Ray. All of the implementation
        for this class is in the parent class, and this class defines the axis
        to perform the computation over.
    """

    axis = 0


class PandasOnRayRowPartition(PandasOnRayAxisPartition):
    """The row partition implementation for Ray. All of the implementation
        for this class is in the parent class, and this class defines the axis
        to perform the computation over.
    """

    axis = 1


@ray.remote
def deploy_ray_axis_func(axis, func, num_splits, kwargs, *partitions):
    """Deploy a function along a full axis in Ray.

    Args:
        axis: The axis to perform the function along.
        func: The function to perform.
        num_splits: The number of splits to return
            (see `split_result_of_axis_func_pandas`)
        kwargs: A dictionary of keyword arguments.
        partitions: All partitions that make up the full axis (row or column)

    Returns:
        A list of Pandas DataFrames.
    """
    dataframe = pandas.concat(partitions, axis=axis, copy=False)
    result = func(dataframe, **kwargs)
    if isinstance(result, pandas.Series):
        return [result] + [pandas.Series([]) for _ in range(num_splits - 1)]
    if num_splits != len(partitions):
        lengths = None
    else:
        if axis == 0:
            lengths = [len(part) for part in partitions]
            if sum(lengths) != len(result):
                lengths = None
        else:
            lengths = [len(part.columns) for part in partitions]
            if sum(lengths) != len(result.columns):
                lengths = None
    return split_result_of_axis_func_pandas(axis, num_splits, result, lengths)


@ray.remote
def deploy_ray_func_between_two_axis_partitions(
    axis, func, num_splits, len_of_left, kwargs, *partitions
):
    """Deploy a function along a full axis between two data sets in Ray.

    Args:
        axis: The axis to perform the function along.
        func: The function to perform.
        num_splits: The number of splits to return
            (see `split_result_of_axis_func_pandas`).
        len_of_left: The number of values in `partitions` that belong to the
            left data set.
        kwargs: A dictionary of keyword arguments.
        partitions: All partitions that make up the full axis (row or column)
            for both data sets.

    Returns:
        A list of Pandas DataFrames.
    """
    lt_frame = pandas.concat(list(partitions[:len_of_left]), axis=axis, copy=False)
    rt_frame = pandas.concat(list(partitions[len_of_left:]), axis=axis, copy=False)

    result = func(lt_frame, rt_frame, **kwargs)
    return split_result_of_axis_func_pandas(axis, num_splits, result)


@ray.remote
def deploy_ray_shuffle_func(axis, func, numsplits, kwargs, *partitions):
    """Deploy a function that defines the partitions along this axis.

    Args:
        axis:
        func:
        numsplits:
        kwargs:
        partitions:

    Returns:
        A list of Pandas DataFrames.
    """
    dataframe = pandas.concat(partitions, axis=axis, copy=False)
    result = func(dataframe, numsplits=numsplits, **kwargs)

    assert isinstance(result, list)
    return result
