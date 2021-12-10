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
from typing import NamedTuple, Union
import atexit
import warnings

from modin import set_execution

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
    __DEFAULT_IMAGE = {AWS: "ami-0f56279347d2fa43e"}

    def __init__(
        self,
        name: str,
        credentials_file: str = None,
        region: str = None,
        zone: str = None,
        image: str = None,
    ):
        """
        Class that holds all information about particular connection to cluster provider, namely
            * provider name (must be one of known ones)
            * path to file with credentials (file format is provider-specific); omit to use global provider-default credentials
            * region and zone where cluster is to be spawned (optional, would be deduced if omitted)
            * image to use (optional, would use default for provider if omitted)
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
        self.image = image or self.__DEFAULT_IMAGE[name]

    @property
    def default_head_type(self):
        return self.__DEFAULT_HEAD[self.name]

    @property
    def default_worker_type(self):
        return self.__DEFAULT_WORKER[self.name]


class BaseCluster:
    """
    Cluster manager for Modin. Knows how to use certain tools to spawn and destroy clusters,
    can serve as context manager to switch execution engine and storage format to remote.
    """

    target_engine = None
    target_storage_format = None
    wrap_cmd = None
    Connector = Connection

    def __init__(
        self,
        provider: Provider,
        project_name: str = None,
        cluster_name: str = "modin-cluster",
        worker_count: int = 4,
        head_node_type: str = None,
        worker_node_type: str = None,
        add_conda_packages: list = None,
    ):
        """
        Prepare the cluster manager. It needs to know a few things:
            * which cloud provider to use
            * what is project name (could be omitted to use default one for account used to connect)
            * cluster name
            * worker count
            * head and worker node instance types (can be omitted to default to provider-defined)
            * custom conda packages for remote environment
        """

        self.provider = provider
        self.project_name = project_name
        self.cluster_name = cluster_name
        self.worker_count = worker_count
        self.head_node_type = head_node_type or provider.default_head_type
        self.worker_node_type = worker_node_type or provider.default_worker_type
        self.add_conda_packages = add_conda_packages

        self.old_execution = None
        self.connection: Connection = None

    def spawn(self, wait=False):
        """
        Actually spawns the cluster. When already spawned, should be a no-op.
        Always call .spawn(True) before assuming a cluster is ready.

        When wait==False it spawns cluster asynchronously.
        """
        self._spawn(wait=wait)
        atexit.register(self.destroy, wait=True)
        if wait:
            # cluster is ready now
            if self.connection is None:
                self.connection = self.Connector(
                    self._get_connection_details(),
                    self._get_main_python(),
                    self.wrap_cmd,
                )

    def destroy(self, wait=False):
        """
        Destroys the cluster. When already destroyed, should be a no-op.
        Always call .destroy(True) before assuming a cluster is dead.

        When wait==False it destroys cluster asynchronously.
        """
        if self.connection is not None:
            self.connection.stop()
        self._destroy(wait=wait)
        if wait:
            atexit.unregister(self.destroy)

    def _spawn(self, wait=False):
        """
        Subclass must implement the real spawning
        """
        raise NotImplementedError()

    def _destroy(self, wait=False):
        """
        Subclass must implement the real destruction
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
        self.connection.activate()
        self.old_execution = set_execution(
            self.target_engine, self.target_storage_format
        )
        return self

    def __exit__(self, *a, **kw):
        set_execution(*self.old_execution)
        self.connection.deactivate()
        self.old_execution = None

    # TODO: implement __del__() properly; naive implementation calling .destroy() crashes
    # somewhere in the innards of Ray when a cluster is destroyed during interpreter exit.


def create(
    provider: Union[Provider, str],
    credentials: str = None,
    region: str = None,
    zone: str = None,
    image: str = None,
    project_name: str = None,
    cluster_name: str = "modin-cluster",
    workers: int = 4,
    head_node: str = None,
    worker_node: str = None,
    add_conda_packages: list = None,
    cluster_type: str = "rayscale",
) -> BaseCluster:
    """
    Creates an instance of a cluster with desired characteristics in a cloud.
    Upon entering a context via with statement Modin will redirect its work to the remote cluster.
    Spawned cluster can be destroyed manually, or it will be destroyed when the program exits.

    Parameters
    ----------
    provider : str or instance of Provider class
        Specify the name of the provider to use or a Provider object.
        If Provider object is given, then credentials, region and zone are ignored.
    credentials : str, optional
        Path to the file which holds credentials used by given cloud provider.
        If not specified, cloud provider will use its default means of finding credentials on the system.
    region : str, optional
        Region in the cloud where to spawn the cluster.
        If omitted a default for given provider will be taken.
    zone : str, optional
        Availability zone (part of region) where to spawn the cluster.
        If omitted a default for given provider and region will be taken.
    image: str, optional
        Image to use for spawning head and worker nodes.
        If omitted a default for given provider will be taken.
    project_name : str, optional
        Project name to assign to the cluster in cloud, for easier manual tracking.
    cluster_name : str, optional
        Name to be given to the cluster.
        To spawn multiple clusters in single region and zone use different names.
    workers : int, optional
        How many worker nodes to spawn in the cluster. Head node is not counted for here.
    head_node : str, optional
        What machine type to use for head node in the cluster.
    worker_node : str, optional
        What machine type to use for worker nodes in the cluster.
    add_conda_packages : list, optional
        Custom conda packages for remote environments. By default remote modin version is
        the same as local version.
    cluster_type : str, optional
        How to spawn the cluster.
        Currently spawning by Ray autoscaler ("rayscale" for general and "omnisci" for Omnisci-based) is supported

    Returns
    -------
    BaseCluster descendant
        The object that knows how to destroy the cluster and how to activate it as remote context.
        Note that by default spawning and destroying of the cluster happens in the background,
        as it's usually a rather lengthy process.

    Notes
    -----
    Cluster computation actually can work when proxies are required to access the cloud.
    You should set normal "http_proxy"/"https_proxy" variables for HTTP/HTTPS proxies and
    set "MODIN_SOCKS_PROXY" variable for SOCKS proxy before calling the function.

    Using SOCKS proxy requires Ray newer than 0.8.6, which might need to be installed manually.
    """
    if not isinstance(provider, Provider) and cluster_type != "local":
        provider = Provider(
            name=provider,
            credentials_file=credentials,
            region=region,
            zone=zone,
            image=image,
        )
    else:
        if any(p is not None for p in (credentials, region, zone, image)):
            warnings.warn(
                "Ignoring credentials, region, zone and image parameters because provider is specified as Provider descriptor, not as name",
                UserWarning,
            )
    if cluster_type == "rayscale":
        from .rayscale import RayCluster as Spawner
    elif cluster_type == "omnisci":
        from .omnisci import RemoteOmnisci as Spawner
    elif cluster_type == "local":
        from .local_cluster import LocalCluster as Spawner
    else:
        raise ValueError(f"Unknown cluster type: {cluster_type}")
    instance = Spawner(
        provider,
        project_name,
        cluster_name,
        worker_count=workers,
        head_node_type=head_node,
        worker_node_type=worker_node,
        add_conda_packages=add_conda_packages,
    )
    instance.spawn(wait=False)
    return instance
