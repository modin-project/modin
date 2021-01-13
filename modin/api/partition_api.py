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
                    (partition.ip, getattr(partition, oid))
                    for row in api_layer_object._query_compiler._modin_frame._partitions
                    for partition in row
                ]
            else:
                return [
                    getattr(partition, oid)
                    for row in api_layer_object._query_compiler._modin_frame._partitions
                    for partition in row
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
