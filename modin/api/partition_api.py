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

import numpy as np

from modin.pandas.dataframe import DataFrame
from modin.backends.pandas.query_compiler import PandasQueryCompiler


def unwrap_partitions(api_layer_object, axis=None, bind_ip=False):
    """
    Unwrap partitions of the `api_layer_object`.

    Parameters
    ----------
    api_layer_object : DataFrame or Series
        The API layer object.
    axis : None, 0 or 1. Default is None
        The axis to unwrap partitions for (0 - row partitions, 1 - column partitions).
        If axis is None, all the partitions of the API layer object are unwrapped.
    bind_ip : boolean. Default is False
        Whether to bind node ip address to each partition or not.

    Returns
    -------
    list
        A list of Ray.ObjectRef/Dask.Future to partitions of the `api_layer_object`
        if Ray/Dask is used as an engine.

    Notes
    -----
    In case bind_ip=True, a list containing tuples of Ray.ObjectRef/Dask.Future to node ip addresses
    and partitions of the `api_layer_object`, respectively, is returned if Ray/Dask is used as an engine.
    """
    if not hasattr(api_layer_object, "_query_compiler"):
        raise ValueError(
            f"Only API Layer objects may be passed in here, got {type(api_layer_object)} instead."
        )

    if axis is None:

        def _unwrap_partitions(oid):
            if bind_ip:
                return [
                    [(partition.ip, getattr(partition, oid)) for partition in row]
                    for row in api_layer_object._query_compiler._modin_frame._partitions
                ]
            else:
                return [
                    [getattr(partition, oid) for partition in row]
                    for row in api_layer_object._query_compiler._modin_frame._partitions
                ]

        actual_engine = type(
            api_layer_object._query_compiler._modin_frame._partitions[0][0]
        ).__name__
        if actual_engine in ("PandasOnRayFramePartition",):
            return _unwrap_partitions("oid")
        elif actual_engine in ("PandasOnDaskFramePartition",):
            return _unwrap_partitions("future")
        raise ValueError(
            f"Do not know how to unwrap '{actual_engine}' underlying partitions"
        )
    else:
        partitions = (
            api_layer_object._query_compiler._modin_frame._frame_mgr_cls.axis_partition(
                api_layer_object._query_compiler._modin_frame._partitions, axis ^ 1
            )
        )
        return [
            part.coalesce(bind_ip=bind_ip).unwrap(squeeze=True, bind_ip=bind_ip)
            for part in partitions
        ]


def create_df_from_partitions(partitions, axis, engine):
    """
    Create DataFrame from remote partitions.

    Parameters
    ----------
    partitions : list
        List of Ray.ObjectRef/Dask.Future referencing to partitions in depend of the engine used.
        Or list containing tuples of Ray.ObjectRef/Dask.Future referencing to ip addresses of partitions
        and partitions itself in depend of the engine used.
    axis : None, 0 or 1
        The `axis` parameter is used to identify what is the partitions passed.
        You have to set:
        - `axis` to 0 if you want to create DataFrame from row partitions.
        - `axis` to 1 if you want to create DataFrame from column partitions.
        - `axis` to None if you want to create DataFrame from 2D list of partitions.
    engine : int
        The parameter is used to choose the engine for remote partitions which DataFrame will be created from.
        Possible variants: ("PandasOnRayFramePartition", "PandasOnDaskFramePartition").
        You have to specify index of the set: 0 - PandasOnRayFramePartition, 1 - PandasOnDaskFramePartition.

    Returns
    -------
    DataFrame
        DataFrame instance created from remote partitions.
    """
    if engine == 0:
        from modin.engines.ray.pandas_on_ray.frame.partition import (
            PandasOnRayFramePartition,
        )
        from modin.engines.ray.pandas_on_ray.frame.partition_manager import (
            PandasOnRayFrameManager,
        )
        from modin.engines.ray.pandas_on_ray.frame.partition_manager import (
            PandasOnRayFrame,
        )

        partition_class = PandasOnRayFramePartition
        partition_frame_class = PandasOnRayFrame
        partition_mgr_class = PandasOnRayFrameManager
    elif engine == 1:
        from modin.engines.dask.pandas_on_dask.frame.partition import (
            PandasOnDaskFramePartition,
        )
        from modin.engines.dask.pandas_on_dask.frame.partition_manager import (
            DaskFrameManager,
        )
        from modin.engines.dask.pandas_on_dask.frame.data import PandasOnDaskFrame

        partition_class = PandasOnDaskFramePartition
        partition_frame_class = PandasOnDaskFrame
        partition_mgr_class = DaskFrameManager
    else:
        raise ValueError(
            f"Got unacceptable index {engine} of the engines' list. Possible variants are {0} or {1}."
        )

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
    else:
        if axis == 0:
            if isinstance(partitions[0], tuple):
                parts = np.array(
                    [
                        [partition_class(partition, ip=ip)]
                        for ip, partition in partitions
                    ]
                )
            else:
                parts = np.array(
                    [[partition_class(partition)] for partition in partitions]
                )
        elif axis == 1:
            if isinstance(partitions[0], tuple):
                parts = np.array(
                    [
                        [
                            partition_class(partition, ip=ip)
                            for ip, partition in partitions
                        ]
                    ]
                )
            else:
                parts = np.array(
                    [[partition_class(partition) for partition in partitions]]
                )

    index = partition_mgr_class.get_indices(0, parts, lambda df: df.axes[0])
    columns = partition_mgr_class.get_indices(1, parts, lambda df: df.axes[1])
    return DataFrame(
        query_compiler=PandasQueryCompiler(partition_frame_class(parts, index, columns))
    )
