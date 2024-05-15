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
import ray
from ray.util import get_node_ip_address

from modin.config import RayTaskCustomResources
from modin.core.dataframe.pandas.partitioning.axis_partition import (
    PandasDataframeAxisPartition,
)
from modin.core.execution.ray.common import RayWrapper
from modin.utils import _inherit_docstrings

from .partition import PandasOnRayDataframePartition


class PandasOnRayDataframeVirtualPartition(PandasDataframeAxisPartition):
    """
    The class implements the interface in ``PandasDataframeAxisPartition``.

    Parameters
    ----------
    list_of_partitions : Union[list, PandasOnRayDataframePartition]
        List of ``PandasOnRayDataframePartition`` and
        ``PandasOnRayDataframeVirtualPartition`` objects, or a single
        ``PandasOnRayDataframePartition``.
    get_ip : bool, default: False
        Whether to get node IP addresses to conforming partitions or not.
    full_axis : bool, default: True
        Whether or not the virtual partition encompasses the whole axis.
    call_queue : list, optional
        A list of tuples (callable, args, kwargs) that contains deferred calls.
    length : ray.ObjectRef or int, optional
        Length, or reference to length, of wrapped ``pandas.DataFrame``.
    width : ray.ObjectRef or int, optional
        Width, or reference to width, of wrapped ``pandas.DataFrame``.
    """

    _PARTITIONS_METADATA_LEN = 3  # (length, width, ip)
    partition_type = PandasOnRayDataframePartition
    axis = None

    # these variables are intentionally initialized at runtime (see #6023)
    _DEPLOY_AXIS_FUNC = None
    _DEPLOY_SPLIT_FUNC = None
    _DRAIN_FUNC = None

    @classmethod
    def _get_deploy_axis_func(cls):  # noqa: GL08
        if cls._DEPLOY_AXIS_FUNC is None:
            cls._DEPLOY_AXIS_FUNC = RayWrapper.put(
                PandasDataframeAxisPartition.deploy_axis_func
            )
        return cls._DEPLOY_AXIS_FUNC

    @classmethod
    def _get_deploy_split_func(cls):  # noqa: GL08
        if cls._DEPLOY_SPLIT_FUNC is None:
            cls._DEPLOY_SPLIT_FUNC = RayWrapper.put(
                PandasDataframeAxisPartition.deploy_splitting_func
            )
        return cls._DEPLOY_SPLIT_FUNC

    @classmethod
    def _get_drain_func(cls):  # noqa: GL08
        if cls._DRAIN_FUNC is None:
            cls._DRAIN_FUNC = RayWrapper.put(PandasDataframeAxisPartition.drain)
        return cls._DRAIN_FUNC

    @property
    def list_of_ips(self):
        """
        Get the IPs holding the physical objects composing this partition.

        Returns
        -------
        List
            A list of IPs as ``ray.ObjectRef`` or str.
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
        return _deploy_ray_func.options(
            num_returns=(
                num_splits * (1 + cls._PARTITIONS_METADATA_LEN)
                if extract_metadata
                else num_splits
            ),
            resources=RayTaskCustomResources.get(),
        ).remote(
            cls._get_deploy_split_func(),
            *f_args,
            num_splits,
            *partitions,
            axis=axis,
            f_to_deploy=func,
            f_len_args=len(f_args),
            f_kwargs=f_kwargs,
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
            A list of ``ray.ObjectRef``-s.
        """
        return _deploy_ray_func.options(
            num_returns=(num_splits if lengths is None else len(lengths))
            * (1 + cls._PARTITIONS_METADATA_LEN),
            **({"max_retries": max_retries} if max_retries is not None else {}),
            resources=RayTaskCustomResources.get(),
        ).remote(
            cls._get_deploy_axis_func(),
            *f_args,
            num_splits,
            maintain_partitioning,
            *partitions,
            axis=axis,
            f_to_deploy=func,
            f_len_args=len(f_args),
            f_kwargs=f_kwargs,
            manual_partition=manual_partition,
            min_block_size=min_block_size,
            lengths=lengths,
            return_generator=True,
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
            A list of ``ray.ObjectRef``-s.
        """
        return _deploy_ray_func.options(
            num_returns=num_splits * (1 + cls._PARTITIONS_METADATA_LEN),
            resources=RayTaskCustomResources.get(),
        ).remote(
            PandasDataframeAxisPartition.deploy_func_between_two_axis_partitions,
            *f_args,
            num_splits,
            len_of_left,
            other_shape,
            *partitions,
            axis=axis,
            f_to_deploy=func,
            f_len_args=len(f_args),
            f_kwargs=f_kwargs,
            min_block_size=min_block_size,
            return_generator=True,
        )

    def wait(self):
        """Wait completing computations on the object wrapped by the partition."""
        self.drain_call_queue()
        futures = self.list_of_blocks
        RayWrapper.wait(futures)


@_inherit_docstrings(PandasOnRayDataframeVirtualPartition)
class PandasOnRayDataframeColumnPartition(PandasOnRayDataframeVirtualPartition):
    axis = 0


@_inherit_docstrings(PandasOnRayDataframeVirtualPartition)
class PandasOnRayDataframeRowPartition(PandasOnRayDataframeVirtualPartition):
    axis = 1


@ray.remote
def _deploy_ray_func(
    deployer,
    *positional_args,
    axis,
    f_to_deploy,
    f_len_args,
    f_kwargs,
    extract_metadata=True,
    **kwargs,
):  # pragma: no cover
    """
    Execute a function on an axis partition in a worker process.

    This is ALWAYS called on either ``PandasDataframeAxisPartition.deploy_axis_func``
    or ``PandasDataframeAxisPartition.deploy_func_between_two_axis_partitions``, which both
    serve to deploy another dataframe function on a Ray worker process. The provided `positional_args`
    contains positional arguments for both: `deployer` and for `f_to_deploy`, the parameters can be separated
    using the `f_len_args` value. The parameters are combined so they will be deserialized by Ray before the
    kernel is executed (`f_kwargs` will never contain more Ray objects, and thus does not require deserialization).

    Parameters
    ----------
    deployer : callable
        A `PandasDataFrameAxisPartition.deploy_*` method that will call ``f_to_deploy``.
    *positional_args : list
        The first `f_len_args` elements in this list represent positional arguments
        to pass to the `f_to_deploy`. The rest are positional arguments that will be
        passed to `deployer`.
    axis : {0, 1}
        The axis to perform the function along. This argument is keyword only.
    f_to_deploy : callable or RayObjectID
        The function to deploy. This argument is keyword only.
    f_len_args : int
        Number of positional arguments to pass to ``f_to_deploy``. This argument is keyword only.
    f_kwargs : dict
        Keyword arguments to pass to ``f_to_deploy``. This argument is keyword only.
    extract_metadata : bool, default: True
        Whether to return metadata (length, width, ip) of the result. Passing `False` may relax
        the load on object storage as the remote function would return 4 times fewer futures.
        Passing `False` makes sense for temporary results where you know for sure that the
        metadata will never be requested. This argument is keyword only.
    **kwargs : dict
        Keyword arguments to pass to ``deployer``.

    Returns
    -------
    list : Union[tuple, list]
        The result of the function call, and metadata for it.

    Notes
    -----
    Ray functions are not detected by codecov (thus pragma: no cover).
    """
    f_args = positional_args[:f_len_args]
    deploy_args = positional_args[f_len_args:]
    result = deployer(axis, f_to_deploy, f_args, f_kwargs, *deploy_args, **kwargs)

    if not extract_metadata:
        for item in result:
            yield item
    else:
        ip = get_node_ip_address()
        for r in result:
            if isinstance(r, pandas.DataFrame):
                for item in [r, len(r), len(r.columns), ip]:
                    yield item
            else:
                for item in [r, None, None, ip]:
                    yield item
