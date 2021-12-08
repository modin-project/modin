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

"""The module defines interface for an axis partition with PyArrow storage format and Ray engine."""

from modin.core.dataframe.pandas.partitioning.axis_partition import (
    BaseDataframeAxisPartition,
)
from .partition import PyarrowOnRayDataframePartition

import ray
import pyarrow


class PyarrowOnRayDataframeAxisPartition(BaseDataframeAxisPartition):
    """
    Class defines axis partition interface with PyArrow storage format and Ray engine.

    Inherits functionality from ``BaseDataframeAxisPartition`` class.

    Parameters
    ----------
    list_of_blocks : list
        List with partition objects to create common axis partition for.
    """

    def __init__(self, list_of_blocks):
        # Unwrap from PandasDataframePartition object for ease of use
        self.list_of_blocks = [obj.oid for obj in list_of_blocks]

    def apply(self, func, num_splits=None, other_axis_partition=None, **kwargs):
        """
        Apply func to the object in the Plasma store.

        Parameters
        ----------
        func : callable or ray.ObjectRef
            The function to apply.
        num_splits : int, optional
            The number of times to split the resulting object.
        other_axis_partition : PyarrowOnRayDataframeAxisPartition, optional
            Another ``PyarrowOnRayDataframeAxisPartition`` object to apply to
            `func` with this one.
        **kwargs : dict
            Additional keyward arguments to pass with `func`.

        Returns
        -------
        list
            List with ``PyarrowOnRayDataframePartition`` objects.

        Notes
        -----
        See notes in Parent class about this method.
        """
        if num_splits is None:
            num_splits = len(self.list_of_blocks)

        if other_axis_partition is not None:
            return [
                PyarrowOnRayDataframePartition(obj)
                for obj in deploy_ray_func_between_two_axis_partitions.options(
                    num_returns=num_splits
                ).remote(
                    self.axis,
                    func,
                    num_splits,
                    len(self.list_of_blocks),
                    kwargs,
                    *(self.list_of_blocks + other_axis_partition.list_of_blocks),
                )
            ]

        args = [self.axis, func, num_splits, kwargs]
        args.extend(self.list_of_blocks)
        return [
            PyarrowOnRayDataframePartition(obj)
            for obj in deploy_ray_axis_func.options(num_returns=num_splits).remote(
                *args
            )
        ]

    def shuffle(self, func, num_splits=None, **kwargs):
        """
        Shuffle the order of the data in this axis based on the `func`.

        Parameters
        ----------
        func : callable
            The function to apply before splitting.
        num_splits : int, optional
            The number of times to split the resulting object.
        **kwargs : dict
            Additional keywords arguments to be passed in `func`.

        Returns
        -------
        list
            List with ``PyarrowOnRayDataframePartition`` objects.

        Notes
        -----
        Method extends ``BaseDataframeAxisPartition.shuffle``.
        """
        if num_splits is None:
            num_splits = len(self.list_of_blocks)

        args = [self.axis, func, num_splits, kwargs]
        args.extend(self.list_of_blocks)
        return [
            PyarrowOnRayDataframePartition(obj)
            for obj in deploy_ray_axis_func.options(num_returns=num_splits).remote(
                *args
            )
        ]


class PyarrowOnRayDataframeColumnPartition(PyarrowOnRayDataframeAxisPartition):
    """
    The column partition implementation for PyArrow storage format and Ray engine.

    All of the implementation for this class is in the ``PyarrowOnRayDataframeAxisPartition``
    parent class, and this class defines the axis to perform the computation over.

    Parameters
    ----------
    list_of_blocks : list
        List with partition objects to create common axis partition.
    """

    axis = 0


class PyarrowOnRayDataframeRowPartition(PyarrowOnRayDataframeAxisPartition):
    """
    The row partition implementation for PyArrow storage format and Ray engine.

    All of the implementation for this class is in the ``PyarrowOnRayDataframeAxisPartition``
    parent class, and this class defines the axis to perform the computation over.

    Parameters
    ----------
    list_of_blocks : list
        List with partition objects to create common axis partition.
    """

    axis = 1


def concat_arrow_table_partitions(axis, partitions):
    """
    Concatenate given `partitions` in a single table.

    Parameters
    ----------
    axis : {0, 1}
        The axis to concatenate over.
    partitions : array-like
        Array with partitions for concatenating.

    Returns
    -------
    pyarrow.Table
        ``pyarrow.Table`` constructed from the passed partitions.
    """
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
    """
    Split ``pyarrow.Table`` according to the passed parameters.

    Parameters
    ----------
    axis : {0, 1}
        The axis to perform the function along.
    result : pyarrow.Table
        Resulting table to split.
    num_partitions : int
        Number of partitions that `result` was constructed from.
    num_splits : int
        The number of splits to return.
    metadata : dict
        Dictionary with ``pyarrow.Table`` metadata.

    Returns
    -------
    list
        List of PyArrow Tables.
    """
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
    """
    Deploy a function along a full axis in Ray.

    Parameters
    ----------
    axis : {0, 1}
        The axis to perform the function along.
    func : callable
        The function to deploy.
    num_splits : int
        The number of splits to return.
    kwargs : dict
        A dictionary of keyword arguments.
    *partitions : array-like
        All partitions that make up the full axis (row or column).

    Returns
    -------
    list
        List of PyArrow Tables.
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
    """
    Deploy a function along a full axis between two data sets in Ray.

    Parameters
    ----------
    axis : {0, 1}
        The axis to perform the function along.
    func : callable
        The function to deploy.
    num_splits : int
        The number of splits to return.
    len_of_left : int
        The number of values in `partitions` that belong to the left data set.
    kwargs : dict
        A dictionary of keyword arguments.
    *partitions : array-like
        All partitions that make up the full axis (row or column)
        for both data sets.

    Returns
    -------
    list
        List of PyArrow Tables.
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
def deploy_ray_shuffle_func(axis, func, num_splits, kwargs, *partitions):
    """
    Deploy shuffle function that defines the order of the data in this axis partition.

    Parameters
    ----------
    axis : {0, 1}
        The axis to perform the function along.
    func : callable
        The function to deploy.
    num_splits : int
        The number of splits to return.
    kwargs : dict
        A dictionary of keyword arguments.
    *partitions : array-like
        All partitions that make up the full axis (row or column).

    Notes
    -----
    Function is deprecated.
    """
