import pandas

from modin.engines.base.axis_partition import BaseAxisPartition
from modin.data_management.utils import split_result_of_axis_func_pandas
from .remote_partition import DaskRemotePartition


class DaskAxisPartition(BaseAxisPartition):
    """Dask implementation for Column and Row partitions"""

    def __init__(self, list_of_blocks):
        # Unwrap from BaseRemotePartition object for ease of use
        self.list_of_blocks = [b.dask_obj for b in list_of_blocks]

    def apply(
        self,
        func,
        num_splits=None,
        other_axis_partition=None,
        maintain_partitioning=True,
        **kwargs
    ):
        """Applies func to the object.

        See notes in Parent class about this method.

        Args:
            func: The function to apply.
            num_splits: The number of times to split the result object.
            other_axis_partition: Another `DaskAxisPartition` object to apply to
                func with this one.

        Returns:
            A list of `DaskRemotePartition` objects.
        """
        import dask

        if num_splits is None:
            num_splits = len(self.list_of_blocks)

        if other_axis_partition is not None:
            return [
                DaskRemotePartition(dask.delayed(obj))
                for obj in deploy_func_between_two_axis_partitions(
                    self.axis,
                    func,
                    num_splits,
                    len(self.list_of_blocks),
                    kwargs,
                    *dask.compute(
                        *tuple(
                            self.list_of_blocks + other_axis_partition.list_of_blocks
                        )
                    )
                )
            ]

        args = [self.axis, func, num_splits, kwargs, maintain_partitioning]

        args.extend(dask.compute(*self.list_of_blocks))
        return [
            DaskRemotePartition(dask.delayed(obj)) for obj in deploy_axis_func(*args)
        ]


class DaskColumnPartition(DaskAxisPartition):
    """Dask implementation for Column partitions"""

    axis = 0


class DaskRowPartition(DaskAxisPartition):
    """Dask implementation for Row partitions"""

    axis = 1


def deploy_axis_func(
    axis, func, num_splits, kwargs, maintain_partitioning, *partitions
):
    """Deploy a function along a full axis

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
    # XXX pandas_on_python.py is slightly different here but that implementation seems wrong as
    # uncovered by test_var
    if isinstance(result, pandas.Series):
        return [result] + [pandas.Series([]) for _ in range(num_splits - 1)]
    if num_splits != len(partitions) or not maintain_partitioning:
        lengths = None

    #    if num_splits != len(partitions) or isinstance(result, pandas.Series):
    #        import pdb; pdb.set_trace()
    #        lengths = None
    else:
        if axis == 0:
            lengths = [len(part) for part in partitions]
            if sum(lengths) != len(result):
                lengths = None
        else:
            lengths = [len(part.columns) for part in partitions]
            if sum(lengths) != len(result.columns):
                lengths = None
    return [
        df.copy()
        for df in split_result_of_axis_func_pandas(axis, num_splits, result, lengths)
    ]


def deploy_func_between_two_axis_partitions(
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
    return [
        df.copy() for df in split_result_of_axis_func_pandas(axis, num_splits, result)
    ]
