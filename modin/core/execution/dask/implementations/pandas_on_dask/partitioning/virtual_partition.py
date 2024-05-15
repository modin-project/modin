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

"""Module houses classes responsible for storing a virtual partition and applying a function to it."""

import pandas
from distributed.utils import get_ip

from modin.core.dataframe.pandas.partitioning.axis_partition import (
    PandasDataframeAxisPartition,
)
from modin.core.execution.dask.common import DaskWrapper
from modin.utils import _inherit_docstrings

from .partition import PandasOnDaskDataframePartition


class PandasOnDaskDataframeVirtualPartition(PandasDataframeAxisPartition):
    """
    The class implements the interface in ``PandasDataframeAxisPartition``.

    Parameters
    ----------
    list_of_partitions : Union[list, PandasOnDaskDataframePartition]
        List of ``PandasOnDaskDataframePartition`` and
        ``PandasOnDaskDataframeVirtualPartition`` objects, or a single
        ``PandasOnDaskDataframePartition``.
    get_ip : bool, default: False
        Whether to get node IP addresses of conforming partitions or not.
    full_axis : bool, default: True
        Whether or not the virtual partition encompasses the whole axis.
    call_queue : list, optional
        A list of tuples (callable, args, kwargs) that contains deferred calls.
    length : distributed.Future or int, optional
        Length, or reference to length, of wrapped ``pandas.DataFrame``.
    width : distributed.Future or int, optional
        Width, or reference to width, of wrapped ``pandas.DataFrame``.
    """

    axis = None
    _PARTITIONS_METADATA_LEN = 3  # (length, width, ip)
    partition_type = PandasOnDaskDataframePartition

    @property
    def list_of_ips(self):
        """
        Get the IPs holding the physical objects composing this partition.

        Returns
        -------
        List
            A list of IPs as ``distributed.Future`` or str.
        """
        # Defer draining call queue until we get the ip address
        result = [None] * len(self.list_of_block_partitions)
        for idx, partition in enumerate(self.list_of_block_partitions):
            partition.drain_call_queue()
            result[idx] = partition.ip(materialize=False)
        return result

    @classmethod
    @_inherit_docstrings(PandasDataframeAxisPartition.deploy_splitting_func)
    def deploy_splitting_func(
        cls,
        axis,
        func,
        f_args,
        f_kwargs,
        num_splits,
        *partitions,
        extract_metadata=False,
    ):
        return DaskWrapper.deploy(
            func=_deploy_dask_func,
            f_args=(
                PandasDataframeAxisPartition.deploy_splitting_func,
                axis,
                func,
                f_args,
                f_kwargs,
                num_splits,
                *partitions,
            ),
            f_kwargs={"extract_metadata": extract_metadata},
            num_returns=(
                num_splits * (1 + cls._PARTITIONS_METADATA_LEN)
                if extract_metadata
                else num_splits
            ),
            pure=False,
        )

    @classmethod
    def deploy_axis_func(
        cls,
        axis,
        func,
        f_args,
        f_kwargs,
        num_splits,
        maintain_partitioning,
        *partitions,
        min_block_size,
        lengths=None,
        manual_partition=False,
    ):
        """
        Deploy a function along a full axis.

        Parameters
        ----------
        axis : {0, 1}
            The axis to perform the function along.
        func : callable
            The function to perform.
        f_args : list or tuple
            Positional arguments to pass to ``func``.
        f_kwargs : dict
            Keyword arguments to pass to ``func``.
        num_splits : int
            The number of splits to return (see `split_result_of_axis_func_pandas`).
        maintain_partitioning : bool
            If True, keep the old partitioning if possible.
            If False, create a new partition layout.
        *partitions : iterable
            All partitions that make up the full axis (row or column).
        min_block_size : int
            Minimum number of rows/columns in a single split.
        lengths : iterable, default: None
            The list of lengths to shuffle the partition into.
        manual_partition : bool, default: False
            If True, partition the result with `lengths`.

        Returns
        -------
        list
            A list of distributed.Future.
        """
        result_num_splits = len(lengths) if lengths else num_splits
        return DaskWrapper.deploy(
            func=_deploy_dask_func,
            f_args=(
                PandasDataframeAxisPartition.deploy_axis_func,
                axis,
                func,
                f_args,
                f_kwargs,
                num_splits,
                maintain_partitioning,
                *partitions,
            ),
            f_kwargs={
                "min_block_size": min_block_size,
                "lengths": lengths,
                "manual_partition": manual_partition,
            },
            num_returns=result_num_splits * (1 + cls._PARTITIONS_METADATA_LEN),
            pure=False,
        )

    @classmethod
    def deploy_func_between_two_axis_partitions(
        cls,
        axis,
        func,
        f_args,
        f_kwargs,
        num_splits,
        len_of_left,
        other_shape,
        *partitions,
        min_block_size,
    ):
        """
        Deploy a function along a full axis between two data sets.

        Parameters
        ----------
        axis : {0, 1}
            The axis to perform the function along.
        func : callable
            The function to perform.
        f_args : list or tuple
            Positional arguments to pass to ``func``.
        f_kwargs : dict
            Keyword arguments to pass to ``func``.
        num_splits : int
            The number of splits to return (see `split_result_of_axis_func_pandas`).
        len_of_left : int
            The number of values in `partitions` that belong to the left data set.
        other_shape : np.ndarray
            The shape of right frame in terms of partitions, i.e.
            (other_shape[i-1], other_shape[i]) will indicate slice to restore i-1 axis partition.
        *partitions : iterable
            All partitions that make up the full axis (row or column) for both data sets.
        min_block_size : int
            Minimum number of rows/columns in a single split.

        Returns
        -------
        list
            A list of distributed.Future.
        """
        return DaskWrapper.deploy(
            func=_deploy_dask_func,
            f_args=(
                PandasDataframeAxisPartition.deploy_func_between_two_axis_partitions,
                axis,
                func,
                f_args,
                f_kwargs,
                num_splits,
                len_of_left,
                other_shape,
                *partitions,
            ),
            f_kwargs={
                "min_block_size": min_block_size,
            },
            num_returns=num_splits * (1 + cls._PARTITIONS_METADATA_LEN),
            pure=False,
        )

    def wait(self):
        """Wait completing computations on the object wrapped by the partition."""
        self.drain_call_queue()
        DaskWrapper.wait(self.list_of_blocks)


@_inherit_docstrings(PandasOnDaskDataframeVirtualPartition)
class PandasOnDaskDataframeColumnPartition(PandasOnDaskDataframeVirtualPartition):
    axis = 0


@_inherit_docstrings(PandasOnDaskDataframeVirtualPartition)
class PandasOnDaskDataframeRowPartition(PandasOnDaskDataframeVirtualPartition):
    axis = 1


def _deploy_dask_func(
    deployer,
    axis,
    f_to_deploy,
    f_args,
    f_kwargs,
    *args,
    extract_metadata=True,
    **kwargs,
):
    """
    Execute a function on an axis partition in a worker process.

    This is ALWAYS called on either ``PandasDataframeAxisPartition.deploy_axis_func``
    or ``PandasDataframeAxisPartition.deploy_func_between_two_axis_partitions``, which both
    serve to deploy another dataframe function on a Dask worker process.

    Parameters
    ----------
    deployer : callable
        A `PandasDataFrameAxisPartition.deploy_*` method that will call `deploy_f`.
    axis : {0, 1}
        The axis to perform the function along.
    f_to_deploy : callable or RayObjectID
        The function to deploy.
    f_args : list or tuple
        Positional arguments to pass to ``f_to_deploy``.
    f_kwargs : dict
        Keyword arguments to pass to ``f_to_deploy``.
    *args : list
        Positional arguments to pass to ``func``.
    extract_metadata : bool, default: True
        Whether to return metadata (length, width, ip) of the result. Passing `False` may relax
        the load on object storage as the remote function would return 4 times fewer futures.
        Passing `False` makes sense for temporary results where you know for sure that the
        metadata will never be requested.
    **kwargs : dict
        Keyword arguments to pass to ``func``.

    Returns
    -------
    list
        The result of the function ``func`` and metadata for it.
    """
    result = deployer(axis, f_to_deploy, f_args, f_kwargs, *args, **kwargs)
    if not extract_metadata:
        return result
    ip = get_ip()
    if isinstance(result, pandas.DataFrame):
        return result, len(result), len(result.columns), ip
    elif all(isinstance(r, pandas.DataFrame) for r in result):
        return [i for r in result for i in [r, len(r), len(r.columns), ip]]
    else:
        return [i for r in result for i in [r, None, None, ip]]
