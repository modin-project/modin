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

import subprocess
import sys
import warnings

from .base import ConnectionDetails
from .cluster import BaseCluster
from .connection import Connection
from .rpyc_proxy import WrappingConnection, WrappingService
from .tracing.tracing_connection import TracingWrappingConnection
from modin.config import DoTraceRpyc


class LocalWrappingConnection(
    TracingWrappingConnection if DoTraceRpyc.get() else WrappingConnection
):
    def _init_deliver(self):
        def ensure_modin(modin_init):
            import sys
            import os

            modin_dir = os.path.abspath(os.path.join(os.path.dirname(modin_init), ".."))
            # make sure "import modin" will be taken from current modin, not something potentially installed in the system
            if modin_dir not in sys.path:
                sys.path.insert(0, modin_dir)

        import modin

        self.teleport(ensure_modin)(modin.__file__)
        super()._init_deliver()


class LocalWrappingService(WrappingService):
    _protocol = LocalWrappingConnection


class LocalConnection(Connection):
    def _build_sshcmd(self, details: ConnectionDetails, forward_port: int = None):
        return []

    def _run(self, sshcmd: list, cmd: list, capture_out: bool = True):
        assert not sshcmd, "LocalConnection does not support running things via ssh"
        redirect = self._redirect(capture_out)
        if not capture_out and hasattr(redirect, "write"):
            redirect.write(f"Running: {cmd}\n")
        return subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=redirect,
            stderr=redirect,
        )

    @staticmethod
    def _get_service():
        return LocalWrappingService


_UNUSED = object()


class LocalCluster(BaseCluster):
    target_engine = "Cloudpython"
    target_storage_format = "Pandas"

    Connector = LocalConnection

    def __init__(
        self,
        provider,
        project_name=_UNUSED,
        cluster_name=_UNUSED,
        worker_count=_UNUSED,
        head_node_type=_UNUSED,
        worker_node_type=_UNUSED,
        add_conda_packages=_UNUSED,
    ):
        assert (
            provider == "local"
        ), "Local cluster can only be spawned with 'local' provider"
        if any(
            arg is not _UNUSED
            for arg in (
                project_name,
                cluster_name,
                worker_count,
                head_node_type,
                worker_node_type,
                add_conda_packages,
            )
        ):
            warnings.warn(
                "All parameters except 'provider' are ignored for LocalCluster, do not pass them"
            )
        super().__init__(provider, "test-project", "test-cluster", 1, "head", "worker")

    def _spawn(self, wait=False):
        pass

    def _destroy(self, wait=False):
        pass

    def _get_connection_details(self) -> ConnectionDetails:
        return ConnectionDetails()

    def _get_main_python(self) -> str:
        return sys.executable
