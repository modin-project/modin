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

import warnings

from .rayscale import RayCluster
from .cluster import Provider


class RemoteOmnisci(RayCluster):
    target_engine = "Cloudnative"
    target_storage_format = "Omnisci"

    def __init__(
        self,
        provider: Provider,
        project_name: str = None,
        cluster_name: str = "modin-cluster",
        worker_count: int = 0,
        head_node_type: str = None,
        worker_node_type: str = None,
        add_conda_packages: list = None,
    ):
        if worker_count != 0:
            warnings.warn(
                "Current Omnisci on cloud does not support multi-node setups, not requesting worker nodes"
            )
        super().__init__(
            provider,
            project_name,
            cluster_name,
            0,
            head_node_type,
            worker_node_type,
            add_conda_packages,
        )
