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

"""Module houses API to operate on Modin DataFrame partitions that are pandas DataFrame(s)."""

from typing import Optional, Union, TYPE_CHECKING
import numpy as np
from pandas._typing import Axes

from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.pandas.dataframe import DataFrame, Series

if TYPE_CHECKING:
    from modin.core.execution.ray.implementations.pandas_on_ray.partitioning import (
        PandasOnRayDataframePartition,
    )
    from modin.core.execution.dask.implementations.pandas_on_dask.partitioning.partition import (
        PandasOnDaskDataframePartition,
    )
    from modin.core.execution.unidist.implementations.pandas_on_unidist.partitioning.partition import (
        PandasOnUnidistDataframePartition,
    )

    PartitionUnionType = Union[
        PandasOnRayDataframePartition,
        PandasOnDaskDataframePartition,
        PandasOnUnidistDataframePartition,
    ]
else:
    from typing import Any

    PartitionUnionType = Any


def unwrap_partitions(
    api_layer_object: Union[DataFrame, Series],
    axis: Optional[int] = None,
    get_ip: bool = False,
) -> list:
    """
    Unwrap partitions of the ``api_layer_object``.

    Parameters
    ----------
    api_layer_object : DataFrame or Series
        The API layer object.
    axis : {None, 0, 1}, default: None
        The axis to unwrap partitions for (0 - row partitions, 1 - column partitions).
        If ``axis is None``, the partitions are unwrapped as they are currently stored.
    get_ip : bool, default: False
        Whether to get node ip address to each partition or not.

    Returns
    -------
    list
        A list of Ray.ObjectRef/Dask.Future to partitions of the ``api_layer_object``
        if Ray/Dask is used as an engine.

    Notes
    -----
    If ``get_ip=True``, a list of tuples of Ray.ObjectRef/Dask.Future to node ip addresses and
    partitions of the ``api_layer_object``, respectively, is returned if Ray/Dask is used as an engine
    (i.e. ``[(Ray.ObjectRef/Dask.Future, Ray.ObjectRef/Dask.Future), ...]``).
    """
    if not hasattr(api_layer_object, "_query_compiler"):
        raise ValueError(
            f"Only API Layer objects may be passed in here, got {type(api_layer_object)} instead."
        )

    modin_frame = api_layer_object._query_compiler._modin_frame
    modin_frame._propagate_index_objs(None)
    if axis is None:

        def _unwrap_partitions() -> list:
            [p.drain_call_queue() for p in modin_frame._partitions.flatten()]

            def get_block(partition: PartitionUnionType) -> np.ndarray:
                blocks = partition.list_of_blocks
                assert (
                    len(blocks) == 1
                ), f"Implementation assumes that partition contains a single block, but {len(blocks)} recieved."
                return blocks[0]

            if get_ip:
                return [
                    [(partition._ip_cache, get_block(partition)) for partition in row]
                    for row in modin_frame._partitions
                ]
            else:
                return [
                    [get_block(partition) for partition in row]
                    for row in modin_frame._partitions
                ]

        actual_engine = type(
            api_layer_object._query_compiler._modin_frame._partitions[0][0]
        ).__name__
        if actual_engine in (
            "PandasOnRayDataframePartition",
            "PandasOnDaskDataframePartition",
            "PandasOnUnidistDataframePartition",
        ):
            return _unwrap_partitions()
        raise ValueError(
            f"Do not know how to unwrap '{actual_engine}' underlying partitions"
        )
    else:
        partitions = modin_frame._partition_mgr_cls.axis_partition(
            modin_frame._partitions, axis ^ 1
        )
        return [
            part.force_materialization(get_ip=get_ip).unwrap(
                squeeze=True, get_ip=get_ip
            )
            for part in partitions
        ]


def from_partitions(
    partitions: list,
    axis: Optional[int],
    index: Optional[Axes] = None,
    columns: Optional[Axes] = None,
    row_lengths: Optional[list] = None,
    column_widths: Optional[list] = None,
) -> DataFrame:
    """
    Create DataFrame from remote partitions.

    Parameters
    ----------
    partitions : list
        A list of Ray.ObjectRef/Dask.Future to partitions depending on the engine used.
        Or a list of tuples of Ray.ObjectRef/Dask.Future to node ip addresses and partitions
        depending on the engine used (i.e. ``[(Ray.ObjectRef/Dask.Future, Ray.ObjectRef/Dask.Future), ...]``).
    axis : {None, 0 or 1}
        The ``axis`` parameter is used to identify what are the partitions passed.
        You have to set:

        * ``axis=0`` if you want to create DataFrame from row partitions
        * ``axis=1`` if you want to create DataFrame from column partitions
        * ``axis=None`` if you want to create DataFrame from 2D list of partitions
    index : sequence, optional
        The index for the DataFrame. Is computed if not provided.
    columns : sequence, optional
        The columns for the DataFrame. Is computed if not provided.
    row_lengths : list, optional
        The length of each partition in the rows. The "height" of
        each of the block partitions. Is computed if not provided.
    column_widths : list, optional
        The width of each partition in the columns. The "width" of
        each of the block partitions. Is computed if not provided.

    Returns
    -------
    modin.pandas.DataFrame
        DataFrame instance created from remote partitions.

    Notes
    -----
    Pass `index`, `columns`, `row_lengths` and `column_widths` to avoid triggering
    extra computations of the metadata when creating a DataFrame.
    """
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    factory = FactoryDispatcher.get_factory()
    # TODO(https://github.com/modin-project/modin/issues/5127):
    # Remove these assertions once the dependencies of this function all have types.
    assert factory is not None
    assert factory.io_cls is not None
    assert factory.io_cls.frame_cls is not None
    assert factory.io_cls.frame_cls._partition_mgr_cls is not None  # type: ignore[unreachable]
    partition_class = factory.io_cls.frame_cls._partition_mgr_cls._partition_class
    partition_frame_class = factory.io_cls.frame_cls
    partition_mgr_class = factory.io_cls.frame_cls._partition_mgr_cls

    # Since we store partitions of Modin DataFrame as a 2D NumPy array we need to place
    # passed partitions to 2D NumPy array to pass it to internal Modin Frame class.
    # `axis=None` - convert 2D list to 2D NumPy array
    if axis is None:
        if isinstance(partitions[0][0], tuple):
            parts = np.array(
                [
                    [partition_class(partition, ip=ip) for ip, partition in row]
                    for row in partitions
                ]
            )
        else:
            parts = np.array(
                [
                    [partition_class(partition) for partition in row]
                    for row in partitions
                ]
            )
    # `axis=0` - place row partitions to 2D NumPy array so that each row of the array is one row partition.
    elif axis == 0:
        if isinstance(partitions[0], tuple):
            parts = np.array(
                [[partition_class(partition, ip=ip)] for ip, partition in partitions]
            )
        else:
            parts = np.array([[partition_class(partition)] for partition in partitions])
    # `axis=1` - place column partitions to 2D NumPy array so that each column of the array is one column partition.
    elif axis == 1:
        if isinstance(partitions[0], tuple):
            parts = np.array(
                [[partition_class(partition, ip=ip) for ip, partition in partitions]]
            )
        else:
            parts = np.array([[partition_class(partition) for partition in partitions]])
    else:
        raise ValueError(
            f"Got unacceptable value of axis {axis}. Possible values are {0}, {1} or {None}."
        )

    labels_axis_to_sync = None
    if index is None:
        labels_axis_to_sync = 1
        index, internal_indices = partition_mgr_class.get_indices(0, parts)
        if row_lengths is None:
            row_lengths = [len(idx) for idx in internal_indices]

    if columns is None:
        labels_axis_to_sync = 0 if labels_axis_to_sync is None else -1
        columns, internal_indices = partition_mgr_class.get_indices(1, parts)
        if column_widths is None:
            column_widths = [len(idx) for idx in internal_indices]

    frame = partition_frame_class(
        parts,
        index,
        columns,
        row_lengths=row_lengths,
        column_widths=column_widths,
    )

    if labels_axis_to_sync != -1:
        frame.synchronize_labels(axis=labels_axis_to_sync)

    return DataFrame(query_compiler=PandasQueryCompiler(frame))


def repartition(
    df: Union[DataFrame, Series], axis: Optional[int] = None
) -> Union[DataFrame, Series]:
    """
    Repartitioning Modin objects to get ideal partitions inside.

    Allows to improve performance where the query compiler cannot yet.

    Parameters
    ----------
    df : DataFrame or Series
    axis : int, optional

    Returns
    -------
    DataFrame or Series
        The repartitioned dataframe or series, depending on the original type.
    """
    if axis not in (0, 1, None):
        raise NotImplementedError

    from modin.pandas import DataFrame, Series

    if not isinstance(df, (DataFrame, Series)):
        raise NotImplementedError

    list_axis = [0, 1] if axis is None else [axis]

    new_query_compiler = df._query_compiler
    for _ax in list_axis:
        new_query_compiler = new_query_compiler.__constructor__(
            new_query_compiler._modin_frame.apply_full_axis(
                _ax, lambda df: df, keep_partitioning=False
            )
        )
    return df.__constructor__(query_compiler=new_query_compiler)  # type:ignore
