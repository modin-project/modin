# Licensed to Modin Development Team under one or more contributor license agreements.
# See the NOTICE file distributed with this work for additional information regarding
# copyright ownership.  The Modin Development Team licenses this file to you under the
# Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

from modin.engines.base.frame.axis_partition import BaseFrameAxisPartition
from .partition import PyarrowOnRayFramePartition

import ray
import pyarrow


class PyarrowOnRayFrameAxisPartition(BaseFrameAxisPartition):
    def __init__(self, list_of_blocks):
        # Unwrap from BaseFramePartition object for ease of use
        self.list_of_blocks = [obj.oid for obj in list_of_blocks]

    def apply(self, func, num_splits=None, other_axis_partition=None, **kwargs):
        """Applies func to the object in the plasma store.

        See notes in Parent class about this method.

        Args:
            func: The function to apply.
            num_splits: The number of times to split the result object.
            other_axis_partition: Another `PyarrowOnRayFrameAxisPartition` object to apply to
                func with this one.

        Returns:
            A list of `RayRemotePartition` objects.
        """
        if num_splits is None:
            num_splits = len(self.list_of_blocks)

        if other_axis_partition is not None:
            return [
                PyarrowOnRayFramePartition(obj)
                for obj in deploy_ray_func_between_two_axis_partitions._remote(
                    args=(self.axis, func, num_splits, len(self.list_of_blocks), kwargs)
                    + tuple(self.list_of_blocks + other_axis_partition.list_of_blocks),
                    num_returns=num_splits,
                )
            ]

        args = [self.axis, func, num_splits, kwargs]
        args.extend(self.list_of_blocks)
        return [
            PyarrowOnRayFramePartition(obj)
            for obj in deploy_ray_axis_func._remote(args, num_returns=num_splits)
        ]

    def shuffle(self, func, num_splits=None, **kwargs):
        """Shuffle the order of the data in this axis based on the `func`.

        Extends `BaseFrameAxisPartition.shuffle`.

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
            PyarrowOnRayFramePartition(obj)
            for obj in deploy_ray_axis_func._remote(args, num_returns=num_splits)
        ]


class PyarrowOnRayFrameColumnPartition(PyarrowOnRayFrameAxisPartition):
    """The column partition implementation for Ray. All of the implementation
    for this class is in the parent class, and this class defines the axis
    to perform the computation over.
    """

    axis = 0


class PyarrowOnRayFrameRowPartition(PyarrowOnRayFrameAxisPartition):
    """The row partition implementation for Ray. All of the implementation
    for this class is in the parent class, and this class defines the axis
    to perform the computation over.
    """

    axis = 1


def concat_arrow_table_partitions(axis, partitions):
    if axis == 0:
        table = pyarrow.Table.from_batches(
            [part.to_batches(part.num_rows)[0] for part in partitions]
        )
    else:
        table = partitions[0].drop([partitions[0].columns[-1].name])
        for obj in partitions[1:]:
            i = 0
            for col in obj.itercolumns():
                if i < obj.num_columns - 1:
                    table = table.append_column(col)
                i += 1
        table = table.append_column(partitions[0].columns[-1])
    return table


def split_arrow_table_result(axis, result, num_partitions, num_splits, metadata):
    chunksize = (
        num_splits // num_partitions
        if num_splits % num_partitions == 0
        else num_splits // num_partitions + 1
    )
    if axis == 0:
        return [
            pyarrow.Table.from_batches([part]) for part in result.to_batches(chunksize)
        ]
    else:
        return [
            result.drop(
                [
                    result.columns[i].name
                    for i in range(result.num_columns)
                    if i >= n * chunksize or i < (n - 1) * chunksize
                ]
            )
            .append_column(result.columns[-1])
            .replace_schema_metadata(metadata=metadata)
            for n in range(1, num_splits)
        ] + [
            result.drop(
                [
                    result.columns[i].name
                    for i in range(result.num_columns)
                    if i < (num_splits - 1) * chunksize
                ]
            ).replace_schema_metadata(metadata=metadata)
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
    table = concat_arrow_table_partitions(axis, partitions)
    try:
        result = func(table, **kwargs)
    except Exception:
        result = pyarrow.Table.from_pandas(func(table.to_pandas(), **kwargs))
    return split_arrow_table_result(
        axis, result, len(partitions), num_splits, table.schema.metadata
    )


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
    lt_table = concat_arrow_table_partitions(axis, partitions[:len_of_left])
    rt_table = concat_arrow_table_partitions(axis, partitions[len_of_left:])
    try:
        result = func(lt_table, rt_table, **kwargs)
    except Exception:
        lt_frame = lt_table.from_pandas()
        rt_frame = rt_table.from_pandas()
        result = pyarrow.Table.from_pandas(func(lt_frame, rt_frame, **kwargs))
    return split_arrow_table_result(
        axis, result, len(result.num_rows), num_splits, result.schema.metadata
    )


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
    pass
