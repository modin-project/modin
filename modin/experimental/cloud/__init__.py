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

from .base import ClusterError, CannotSpawnCluster, CannotDestroyCluster
from .cluster import Provider, Cluster
from .connection import Connection


def cluster(
    provider: Provider,
    project_name: str = None,
    cluster_name: str = "modin-cluster",
    worker_count: int = 4,
    head_node_type: str = None,
    worker_node_type: str = None,
    spawner: str = "rayscale",
) -> Cluster:
    if spawner == "rayscale":
        from .rayscale import RayCluster as Spawner
    else:
        raise ValueError(f"Unknown spawner: {spawner}")
    return Spawner(
        provider,
        project_name,
        cluster_name,
        worker_count,
        head_node_type,
        worker_node_type,
    )


def get_connection():
    """
    Returns an RPyC connection object to execute Python code remotely.
    """
    return Connection.get()


__all__ = [
    "ClusterError",
    "CannotSpawnCluster",
    "CannotDestroyCluster",
    "Provider",
    "Cluster",
    "get_connection",
]
