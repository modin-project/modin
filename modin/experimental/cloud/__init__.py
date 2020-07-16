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
from .cluster import Provider, create as create_cluster
from .connection import Connection


def get_connection():
    """
    Returns an RPyC connection object to execute Python code remotely on the active cluster.
    """
    return Connection.get()


__all__ = [
    "ClusterError",
    "CannotSpawnCluster",
    "CannotDestroyCluster",
    "Provider",
    "create_cluster",
    "get_connection",
]
