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
import subprocess

from typing import NamedTuple

class ClusterError(Exception):
    '''
    Generic cluster operating exception
    '''

    def __init__(self, *args, cause=None, **kw):
        self.cause = cause
        super().__init__(*args, **kw)

class CannotSpawnCluster(ClusterError):
    '''
    Raised when cluster cannot be spawned in the cloud
    '''

class CannotDestroyCluster(ClusterError):
    '''
    Raised when cluster cannot be destroyed in the cloud
    '''

class ConnectionDetails(NamedTuple):
    address: str
    port: int
    key_file: str
    user_name: str

class Provider:
    AWS = 'aws'
    GCP = 'gcp'
    # otherws would come later
    __KNOWN = {AWS, GCP}

    def __init__(self, name:str, credentials_file:str, region:str=None, zone:str=None):
        '''
        Class that holds all information about particular connection to cluster provider, namely
            * provider name (must be one of known ones)
            * path to file with credentials (file format is provider-specific)
            * region and zone where cluster is to be spawned (optional, would be deduced if omitted)
        '''

        if name not in self.__KNOWN:
            raise ValueError(f'Unknown provider name: {name}')
        if zone is not None and region is None:
            raise ValueError('Cannot specify a zone without specifying a region')

        self.name = name
        self.region = region
        self.zone = zone
        self.credentials_file = os.path.abspath(credentials_file)

    def _make_spawn_params(self):
        '''
        INTERNAL. Makes part of Rhoc command line for using this provider
        '''

        res = ['--provider', self.name, '--credentials', self.credentials_file]
        if self.region:
            res.extend(['--region', self.region])
            if self.zone:
                res.extend(['--zone', self.zone])
        return res

class Cluster:
    '''
    Cluster manager for Modin. Knows how to use certain tools to spawn and destroy clusters,
    can serve as context manager to keep cluster running only as long as it is needed.
    '''

    USER_NAME = 'modin'
    IMAGE_NAME = 'moding-image'
    KEY_NAME = 'modin-key'
    DEFAULT_CLUSTER_NAME = 'modin-cluster'

    def __init__(self, provider:Provider, project_name:str=None, cluster_name:str=DEFAULT_CLUSTER_NAME, worker_count:int=4, instance_type_worker_node:str=None, workdir:str=None):
        '''
        Prepare the cluster manager. It needs to know a few things:
            * how to connect to the cluster provider
            * what is project name (could be omitted to use default one for account used to connect)
            * cluster name
            * worker count
            * worker node instance type
            * where to store intermediate data, defaults to ~/.modin/cloud
        '''

        self.provider = provider
        self.project_name = project_name
        self.cluster_name = cluster_name
        self.worker_count = worker_count
        self.instance_type_worker_node = instance_type_worker_node

        self.workdir = os.path.abspath(workdir) or os.path.expanduser('~/.modin/cloud')
        os.makedirs(self.workdir, exist_ok=True)

        self.__cluster_id = None

    def _make_spawn_params(self):
        '''
        INTERNAL. Makes Rhoc "create cluster" parameters' line.
        '''

        params = {
            'user_name': self.USER_NAME,
            'cluster_name ': self.cluster_name,
            'image_name': self.IMAGE_NAME,
            'worker_count': self.worker_count,
            'ssh_key_pair_path': os.path.join(self.workdir, 'private_keys'),
            'key_name': self.KEY_NAME
        }
        if self.project_name:
            params['project_name'] = self.project_name
        if self.instance_type_worker_node:
            params['instance_type_worker_node'] = self.instance_type_worker_node
        
        joint = ','.join(f'{param_name}={param_value}' for (param_name, param_value) in params.items())

        return self.provider._make_rhoc_params() + ['--vars', joint]

    def spawn(self):
        '''
        Actually spawns the cluster.
        '''

        try:
            subprocess.check_call(['Rhoc', 'create', 'cluster'] + self._make_spawn_params(), cwd=self.workdir)
        except subprocess.CalledProcessError as err:
            raise CannotSpawnCluster(cause=err)
        try:
            state = subprocess.check_output(['Rhoc', 'state'], cwd=self.workdir)
        except subprocess.CalledProcessError as err:
            raise CannotSpawnCluster(cause=err)
        # TODO: extract cluster id from state

    def destroy(self):
        '''
        Destroys the cluster.
        '''

        try:
            subprocess.check_call(['Rhoc', 'destroy', self.__cluster_id])
        except subprocess.CalledProcessError as err:
            raise CannotDestroyCluster(cause=err)

    def get_connection_details(self) -> ConnectionDetails:
        '''
        Gets the coordinates on how to connect to cluster frontend node.
        '''

        # TODO: implement by reading 'Rhoc state'
        raise NotImplementedError()

    def __enter__(self):
        self.spawn()
        return self

    def __exit__(self, *a, **kw):
        self.destroy()
