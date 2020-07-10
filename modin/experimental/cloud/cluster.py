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

import os
import errno
from typing import NamedTuple

from modin import set_backends

from .base import ConnectionDetails
from .connection import Connection


class _RegionZone(NamedTuple):
    region: str
    zone: str


class Provider:
    AWS = "aws"

    __KNOWN = {AWS: [_RegionZone(region="us-west-1", zone="us-west-1a")]}
    __DEFAULT_HEAD = {AWS: "m5.large"}
    __DEFAULT_WORKER = {AWS: "m5.large"}

    def __init__(
        self,
        name: str,
        credentials_file: str = None,
        region: str = None,
        zone: str = None,
    ):
        """
        Class that holds all information about particular connection to cluster provider, namely
            * provider name (must be one of known ones)
            * path to file with credentials (file format is provider-specific); omit to use global provider-default credentials
            * region and zone where cluster is to be spawned (optional, would be deduced if omitted)
        """

        if name not in self.__KNOWN:
            raise ValueError(f"Unknown provider name: {name}")
        if credentials_file is not None and not os.path.exists(credentials_file):
            raise OSError(
                errno.ENOENT, "Credentials file does not exist", credentials_file
            )

        if region is None:
            if zone is not None:
                raise ValueError("Cannot specify a zone without specifying a region")
            try:
                region, zone = self.__KNOWN[name][0]
            except IndexError:
                raise ValueError(f"No defaults for provider {name}")
        elif zone is None:
            for regzone in self.__KNOWN[name]:
                if regzone.region == region:
                    zone = regzone.zone
                    break
            else:
                raise ValueError(f"No default for region {region} for provider {name}")

        self.name = name
        self.region = region
        self.zone = zone
        self.credentials_file = (
            os.path.abspath(credentials_file) if credentials_file is not None else None
        )

    @property
    def default_head_type(self):
        return self.__DEFAULT_HEAD[self.name]

    @property
    def default_worker_type(self):
        return self.__DEFAULT_WORKER[self.name]


class Cluster:
    """
    Cluster manager for Modin. Knows how to use certain tools to spawn and destroy clusters,
    can serve as context manager to switch execution engine and partition to remote.
    """

    target_engine = None
    target_partition = None

    def __init__(
        self,
        provider: Provider,
        project_name: str = None,
        cluster_name: str = "modin-cluster",
        worker_count: int = 4,
        head_node_type: str = None,
        worker_node_type: str = None,
    ):
        """
        Prepare the cluster manager. It needs to know a few things:
            * which cloud provider to use
            * what is project name (could be omitted to use default one for account used to connect)
            * cluster name
            * worker count
            * head and worker node instance types (can be omitted to default to provider-defined)
        """

        self.provider = provider
        self.project_name = project_name
        self.cluster_name = cluster_name
        self.worker_count = worker_count
        self.head_node_type = head_node_type or provider.default_head_type
        self.worker_node_type = worker_node_type or provider.default_worker_type

        self.old_backends = None
        self.connection = None
        self.spawn(wait=False)

    def spawn(self, wait=True):
        """
        Actually spawns the cluster. When already spawned, should be a no-op.

        When wait==False it spawns cluster asynchronously.
        """
        raise NotImplementedError()

    def destroy(self, wait=True):
        """
        Destroys the cluster. When already destroyed, should be a no-op.
        When wait==False it destroys cluster asynchronously.
        """
        raise NotImplementedError()

    def _get_connection_details(self) -> ConnectionDetails:
        """
        Gets the coordinates on how to connect to cluster frontend node.
        """
        raise NotImplementedError()

    def _get_main_python(self) -> str:
        """
        Gets the path to 'main' interpreter (the one that houses created environment for running everything)
        """
        raise NotImplementedError()

    def __enter__(self):
        self.spawn(wait=True)  # make sure cluster is ready
        self.connection = Connection(
            self._get_connection_details(), self._get_main_python()
        )
        self.old_backends = set_backends(self.target_engine, self.target_partition)
        return self

    def __exit__(self, *a, **kw):
        set_backends(*self.old_backends)
        self.connection.stop()
        self.old_backends = None

    def __del__(self):
        self.destroy(wait=True)
