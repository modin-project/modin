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
import ray
from distributed.client import _get_global_client

from modin.backends.pandas.query_compiler import PandasQueryCompiler
from modin.pandas.dataframe import DataFrame


def unwrap_partitions(api_layer_object, axis=None, get_ip=False):
    """
    Unwrap partitions of the ``api_layer_object``.

    Parameters
    ----------
    api_layer_object : DataFrame or Series
        The API layer object.
    axis : None, 0 or 1. Default is None
        The axis to unwrap partitions for (0 - row partitions, 1 - column partitions).
        If ``axis is None``, the partitions are unwrapped as they are currently stored.
    get_ip : boolean. Default is False
        Whether to get node ip address to each partition or not.

    Returns
    -------
    list
        A list of Ray.ObjectRef/Dask.Future to partitions of the ``api_layer_object``
        if Ray/Dask is used as an engine.

    Notes
    -----
    If ``get_ip=True``, a list of tuples of node ip addresses and Ray.ObjectRef/Dask.Future to
    partitions of the ``api_layer_object``, respectively, is returned if Ray/Dask is used as an engine
    (i.e. [(str, Ray.ObjectRef/Dask.Future), ...]).
    """
    if not hasattr(api_layer_object, "_query_compiler"):
        raise ValueError(
            f"Only API Layer objects may be passed in here, got {type(api_layer_object)} instead."
        )

    engine = type(
        api_layer_object._query_compiler._modin_frame._partitions[0][0]
    ).__name__
    if engine not in (
        "PandasOnRayFramePartition",
        "PandasOnDaskFramePartition",
    ):
        raise ValueError(f"Do not know how to unwrap '{engine}' underlying partitions")

    if axis is None:

        def _unwrap_partitions(oid):
            if get_ip:
                return [
                    [(partition.ip(), getattr(partition, oid)) for partition in row]
                    for row in api_layer_object._query_compiler._modin_frame._partitions
                ]
            else:
                return [
                    [getattr(partition, oid) for partition in row]
                    for row in api_layer_object._query_compiler._modin_frame._partitions
                ]

        if engine in ("PandasOnRayFramePartition",):
            return _unwrap_partitions("oid")
        elif engine in ("PandasOnDaskFramePartition",):
            return _unwrap_partitions("future")
    else:
        partitions = (
            api_layer_object._query_compiler._modin_frame._frame_mgr_cls.axis_partition(
                api_layer_object._query_compiler._modin_frame._partitions, axis ^ 1
            )
        )
        partitions = [
            part.force_materialization(get_ip=get_ip).unwrap(
                squeeze=True, get_ip=get_ip
            )
            for part in partitions
        ]
        if get_ip:
            if engine in ("PandasOnRayFramePartition",):
                ips = ray.get([part[0] for part in partitions])
            elif engine in ("PandasOnDaskFramePartition",):
                client = _get_global_client()
                ips = client.gather([part[0] for part in partitions])
            return list(zip(ips, (part[1] for part in partitions)))
        else:
            return partitions


def map_partitions_to_ips(partitions, axis):
    """
    Map partitions to node ip addresses that hold them.

    Parameters
    ----------
    partitions : list
        List of tuples of node ip addresses and Ray.ObjectRef/Dask.Future to partitions
        depending on the engine used (i.e. [(str, Ray.ObjectRef/Dask.Future), ...]).

    axis : None, 0 or 1
        The ``axis`` parameter is used to identify what are the partitions passed.
        You have to set:

        * ``axis=0`` if row partitions are passed
        * ``axis=1`` if column partitions are passed
        * ``axis=None`` if 2D list of partitions is passed

    Returns
    -------
    dict
        A dict of node ip addresses and lists of partitions which the latters are held on
        (i.e. {'str': [Ray.ObjectRef/Dask.Future, ...], ...}).
    """
    if axis is None:
        mapped_partitions = {partitions[0][0][0]: []}
        for row in partitions:
            for ip, partition in row:
                if ip in mapped_partitions.keys():
                    mapped_partitions[ip].append(partition)
                else:
                    mapped_partitions.update({ip: [partition]})
        return mapped_partitions
    elif axis == 0 or axis == 1:
        mapped_partitions = {partitions[0][0]: []}
        for ip, partition in partitions:
            if ip in mapped_partitions.keys():
                mapped_partitions[ip].append(partition)
            else:
                mapped_partitions.update({ip: [partition]})
        return mapped_partitions
    else:
        raise ValueError(
            f"Got unacceptable value of axis {axis}. Possible values are {0}, {1} or {None}."
        )


def from_partitions(partitions, axis):
    """
    Create DataFrame from remote partitions.

    Parameters
    ----------
    partitions : list
        List of Ray.ObjectRef/Dask.Future to partitions depending on the engine used.
        Or list of tuples of node ip addresses and Ray.ObjectRef/Dask.Future to partitions
        depending on the engine used (i.e. [(str, Ray.ObjectRef/Dask.Future), ...]).
    axis : None, 0 or 1
        The ``axis`` parameter is used to identify what are the partitions passed.
        You have to set:

        * ``axis=0`` if you want to create DataFrame from row partitions
        * ``axis=1`` if you want to create DataFrame from column partitions
        * ``axis=None`` if you want to create DataFrame from 2D list of partitions

    Returns
    -------
    DataFrame
        DataFrame instance created from remote partitions.
    """
    from modin.data_management.factories.dispatcher import EngineDispatcher

    factory = EngineDispatcher.get_engine()

    partition_class = factory.io_cls.frame_cls._frame_mgr_cls._partition_class
    partition_frame_class = factory.io_cls.frame_cls
    partition_mgr_class = factory.io_cls.frame_cls._frame_mgr_cls

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

    index = partition_mgr_class.get_indices(0, parts, lambda df: df.axes[0])
    columns = partition_mgr_class.get_indices(1, parts, lambda df: df.axes[1])
    return DataFrame(
        query_compiler=PandasQueryCompiler(partition_frame_class(parts, index, columns))
    )
