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

import warnings

import pandas
import unidist

from modin.core.dataframe.pandas.partitioning.axis_partition import (
    PandasDataframeAxisPartition,
)
from modin.core.execution.unidist.common import UnidistWrapper
from modin.core.execution.unidist.common.utils import deserialize
from modin.utils import _inherit_docstrings

from .partition import PandasOnUnidistDataframePartition


class PandasOnUnidistDataframeVirtualPartition(PandasDataframeAxisPartition):
    """
    The class implements the interface in ``PandasDataframeAxisPartition``.

    Parameters
    ----------
    list_of_partitions : Union[list, PandasOnUnidistDataframePartition]
        List of ``PandasOnUnidistDataframePartition`` and
        ``PandasOnUnidistDataframeVirtualPartition`` objects, or a single
        ``PandasOnUnidistDataframePartition``.
    get_ip : bool, default: False
        Whether to get node IP addresses to conforming partitions or not.
    full_axis : bool, default: True
        Whether or not the virtual partition encompasses the whole axis.
    call_queue : list, optional
        A list of tuples (callable, args, kwargs) that contains deferred calls.
    length : unidist.ObjectRef or int, optional
        Length, or reference to length, of wrapped ``pandas.DataFrame``.
    width : unidist.ObjectRef or int, optional
        Width, or reference to width, of wrapped ``pandas.DataFrame``.
    """

    _PARTITIONS_METADATA_LEN = 3  # (length, width, ip)
    partition_type = PandasOnUnidistDataframePartition
    axis = None

    # these variables are intentionally initialized at runtime (see #6023)
    _DEPLOY_AXIS_FUNC = None
    _DEPLOY_SPLIT_FUNC = None
    _DRAIN_FUNC = None

    @classmethod
    def _get_deploy_axis_func(cls):  # noqa: GL08
        if cls._DEPLOY_AXIS_FUNC is None:
            cls._DEPLOY_AXIS_FUNC = UnidistWrapper.put(
                PandasDataframeAxisPartition.deploy_axis_func
            )
        return cls._DEPLOY_AXIS_FUNC

    @classmethod
    def _get_deploy_split_func(cls):  # noqa: GL08
        if cls._DEPLOY_SPLIT_FUNC is None:
            cls._DEPLOY_SPLIT_FUNC = UnidistWrapper.put(
                PandasDataframeAxisPartition.deploy_splitting_func
            )
        return cls._DEPLOY_SPLIT_FUNC

    @classmethod
    def _get_drain_func(cls):  # noqa: GL08
        if cls._DRAIN_FUNC is None:
            cls._DRAIN_FUNC = UnidistWrapper.put(PandasDataframeAxisPartition.drain)
        return cls._DRAIN_FUNC

    @property
    def list_of_ips(self):
        """
        Get the IPs holding the physical objects composing this partition.

        Returns
        -------
        List
            A list of IPs as ``unidist.ObjectRef`` or str.
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
        return _deploy_unidist_func.options(
            num_returns=(
                num_splits * (1 + cls._PARTITIONS_METADATA_LEN)
                if extract_metadata
                else num_splits
            ),
        ).remote(
            cls._get_deploy_split_func(),
            axis,
            func,
            f_args,
            f_kwargs,
            num_splits,
            *partitions,
            extract_metadata=extract_metadata,
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
        max_retries=None,
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
            The number of splits to return (see ``split_result_of_axis_func_pandas``).
        maintain_partitioning : bool
            If True, keep the old partitioning if possible.
            If False, create a new partition layout.
        *partitions : iterable
            All partitions that make up the full axis (row or column).
        min_block_size : int
            Minimum number of rows/columns in a single split.
        lengths : list, optional
            The list of lengths to shuffle the object.
        manual_partition : bool, default: False
            If True, partition the result with `lengths`.
        max_retries : int, default: None
            The max number of times to retry the func.

        Returns
        -------
        list
            A list of ``unidist.ObjectRef``-s.
        """
        return _deploy_unidist_func.options(
            num_returns=(num_splits if lengths is None else len(lengths))
            * (1 + cls._PARTITIONS_METADATA_LEN),
            **({"max_retries": max_retries} if max_retries is not None else {}),
        ).remote(
            cls._get_deploy_axis_func(),
            axis,
            func,
            f_args,
            f_kwargs,
            num_splits,
            maintain_partitioning,
            *partitions,
            manual_partition=manual_partition,
            min_block_size=min_block_size,
            lengths=lengths,
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
            The number of splits to return (see ``split_result_of_axis_func_pandas``).
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
            A list of ``unidist.ObjectRef``-s.
        """
        return _deploy_unidist_func.options(
            num_returns=num_splits * (1 + cls._PARTITIONS_METADATA_LEN)
        ).remote(
            PandasDataframeAxisPartition.deploy_func_between_two_axis_partitions,
            axis,
            func,
            f_args,
            f_kwargs,
            num_splits,
            len_of_left,
            other_shape,
            *partitions,
            min_block_size=min_block_size,
        )

    def wait(self):
        """Wait completing computations on the object wrapped by the partition."""
        self.drain_call_queue()
        futures = self.list_of_blocks
        UnidistWrapper.wait(futures)


@_inherit_docstrings(PandasOnUnidistDataframeVirtualPartition)
class PandasOnUnidistDataframeColumnPartition(PandasOnUnidistDataframeVirtualPartition):
    axis = 0


@_inherit_docstrings(PandasOnUnidistDataframeVirtualPartition)
class PandasOnUnidistDataframeRowPartition(PandasOnUnidistDataframeVirtualPartition):
    axis = 1


@unidist.remote
def _deploy_unidist_func(
    deployer,
    axis,
    f_to_deploy,
    f_args,
    f_kwargs,
    *args,
    extract_metadata=True,
    **kwargs,
):  # pragma: no cover
    """
    Execute a function on an axis partition in a worker process.

    This is ALWAYS called on either ``PandasDataframeAxisPartition.deploy_axis_func``
    or ``PandasDataframeAxisPartition.deploy_func_between_two_axis_partitions``, which both
    serve to deploy another dataframe function on a unidist worker process. The provided ``f_args``
    is thus are deserialized here (on the unidist worker) before the function is called (``f_kwargs``
    will never contain more unidist objects, and thus does not require deserialization).

    Parameters
    ----------
    deployer : callable
        A `PandasDataFrameAxisPartition.deploy_*` method that will call ``f_to_deploy``.
    axis : {0, 1}
        The axis to perform the function along.
    f_to_deploy : callable or unidist.ObjectRef
        The function to deploy.
    f_args : list or tuple
        Positional arguments to pass to ``f_to_deploy``.
    f_kwargs : dict
        Keyword arguments to pass to ``f_to_deploy``.
    *args : list
        Positional arguments to pass to ``deployer``.
    extract_metadata : bool, default: True
        Whether to return metadata (length, width, ip) of the result. Passing `False` may relax
        the load on object storage as the remote function would return 4 times fewer futures.
        Passing `False` makes sense for temporary results where you know for sure that the
        metadata will never be requested.
    **kwargs : dict
        Keyword arguments to pass to ``deployer``.

    Returns
    -------
    list : Union[tuple, list]
        The result of the function call, and metadata for it.

    Notes
    -----
    Unidist functions are not detected by codecov (thus pragma: no cover).
    """
    f_args = deserialize(f_args)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        result = deployer(axis, f_to_deploy, f_args, f_kwargs, *args, **kwargs)
    if not extract_metadata:
        return result
    ip = unidist.get_ip()
    if isinstance(result, pandas.DataFrame):
        return result, len(result), len(result.columns), ip
    elif all(isinstance(r, pandas.DataFrame) for r in result):
        return [i for r in result for i in [r, len(r), len(r.columns), ip]]
    else:
        return [i for r in result for i in [r, None, None, ip]]
