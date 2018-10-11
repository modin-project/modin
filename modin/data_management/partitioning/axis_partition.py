from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas
import ray

from .remote_partition import RayRemotePartition
from .utils import compute_chunksize


class AxisPartition(object):
    """This abstract class represents the Parent class for any
        `ColumnPartition` or `RowPartition` class. This class is intended to
        simplify the way that operations are performed

        Note 0: The procedures that use this class and its methods assume that
            they have some global knowledge about the entire axis. This may
            require the implementation to use concatenation or append on the
            list of block partitions in this object.

        Note 1: The `BlockPartitions` object that controls these objects
            (through the API exposed here) has an invariant that requires that
            this object is never returned from a function. It assumes that
            there will always be `RemotePartition` object stored and structures
            itself accordingly.

        The only abstract method needed to implement is the `apply` method.
    """

    def apply(self, func, num_splits=None, other_axis_partition=None, **kwargs):
        """Applies a function to a full axis.

        Note: The procedures that invoke this method assume full axis
            knowledge. Implement this method accordingly.

        Important: You must return a list of `RemotePartition` objects from
            this method. See Note 1 for this class above for more information.

        Args:
            func: The function to apply. This will be preprocessed according to
                the corresponding `RemotePartitions` object.
            num_splits: The number of objects to return, the number of splits
                for the resulting object. It is up to this method to choose the
                splitting at this time.
            other_axis_partition: Another `AxisPartition` object to be applied
                to func. This is for operations that are between datasets.

        Returns:
            A list of `RemotePartition` objects.
        """
        raise NotImplementedError("Must be implemented in children classes")

    def shuffle(self, func, num_splits=None, **kwargs):
        """Shuffle the order of the data in this axis based on the `func`.

        Args:
            func:
            num_splits:
            kwargs:

        Returns:
             A list of `RemotePartition` objects.
        """
        raise NotImplementedError("Must be implemented in children classes")


class RayAxisPartition(AxisPartition):
    def __init__(self, list_of_blocks):
        # Unwrap from RemotePartition object for ease of use
        self.list_of_blocks = [obj.oid for obj in list_of_blocks]

    def apply(self, func, num_splits=None, other_axis_partition=None, **kwargs):
        """Applies func to the object in the plasma store.

        See notes in Parent class about this method.

        Args:
            func: The function to apply.
            num_splits: The number of times to split the result object.
            other_axis_partition: Another `RayAxisPartition` object to apply to
                func with this one.

        Returns:
            A list of `RayRemotePartition` objects.
        """
        if num_splits is None:
            num_splits = len(self.list_of_blocks)

        if other_axis_partition is not None:
            return [
                RayRemotePartition(obj)
                for obj in deploy_ray_func_between_two_axis_partitions._submit(
                    args=(self.axis, func, num_splits, len(self.list_of_blocks), kwargs)
                    + tuple(self.list_of_blocks + other_axis_partition.list_of_blocks),
                    num_return_vals=num_splits,
                )
            ]

        args = [self.axis, func, num_splits, kwargs]
        args.extend(self.list_of_blocks)
        return [
            RayRemotePartition(obj)
            for obj in deploy_ray_axis_func._submit(args, num_return_vals=num_splits)
        ]

    def shuffle(self, func, num_splits=None, **kwargs):
        """Shuffle the order of the data in this axis based on the `func`.

        Extends `AxisPartition.shuffle`.

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
            RayRemotePartition(obj)
            for obj in deploy_ray_axis_func._submit(args, num_return_vals=num_splits)
        ]


class RayColumnPartition(RayAxisPartition):
    """The column partition implementation for Ray. All of the implementation
        for this class is in the parent class, and this class defines the axis
        to perform the computation over.
    """

    axis = 0


class RayRowPartition(RayAxisPartition):
    """The row partition implementation for Ray. All of the implementation
        for this class is in the parent class, and this class defines the axis
        to perform the computation over.
    """

    axis = 1


def split_result_of_axis_func_pandas(axis, num_splits, result, length_list=None):
    """Split the Pandas result evenly based on the provided number of splits.

    Args:
        axis: The axis to split across.
        num_splits: The number of even splits to create.
        result: The result of the computation. This should be a Pandas
            DataFrame.
        length_list: The list of lengths to split this DataFrame into. This is used to
            return the DataFrame to its original partitioning schema.

    Returns:
        A list of Pandas DataFrames.
    """
    if length_list is not None:
        length_list.insert(0, 0)
        sums = np.cumsum(length_list)
        if axis == 0:
            return [result.iloc[sums[i] : sums[i + 1]] for i in range(len(sums) - 1)]
        else:
            return [result.iloc[:, sums[i] : sums[i + 1]] for i in range(len(sums) - 1)]
    # We do this to restore block partitioning
    if axis == 0 or type(result) is pandas.Series:
        chunksize = compute_chunksize(len(result), num_splits)
        return [
            result.iloc[chunksize * i : chunksize * (i + 1)] for i in range(num_splits)
        ]
    else:
        chunksize = compute_chunksize(len(result.columns), num_splits)
        return [
            result.iloc[:, chunksize * i : chunksize * (i + 1)]
            for i in range(num_splits)
        ]


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
    if num_splits != len(partitions) or isinstance(result, pandas.Series):
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
