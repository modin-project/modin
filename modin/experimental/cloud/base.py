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

from typing import NamedTuple
import os
import sys

from modin.config import SocksProxy


class ClusterError(Exception):
    """
    Generic cluster operating exception
    """

    def __init__(self, *args, cause: BaseException = None, traceback: str = None, **kw):
        self.cause = cause
        self.traceback = traceback
        super().__init__(*args, **kw)

    def __str__(self):
        if self.cause:
            return f"cause: {self.cause}\n{super().__str__()}"
        return super().__str__()


class CannotSpawnCluster(ClusterError):
    """
    Raised when cluster cannot be spawned in the cloud
    """


class CannotDestroyCluster(ClusterError):
    """
    Raised when cluster cannot be destroyed in the cloud
    """


class ConnectionDetails(NamedTuple):
    user_name: str = "modin"
    key_file: str = None
    address: str = None
    port: int = 22


_EXT = (".exe", ".com", ".cmd", ".bat", "") if sys.platform == "win32" else ("",)


def _which(prog):
    for entry in os.environ["PATH"].split(os.pathsep):
        for ext in _EXT:
            path = os.path.join(entry, prog + ext)
            if os.access(path, os.X_OK):
                return path
    return None


def _get_ssh_proxy_command():
    socks_proxy = SocksProxy.get()
    if socks_proxy is None:
        return None
    if _which("nc"):
        return f"nc -x {socks_proxy} %h %p"
    elif _which("connect"):
        return f"connect -S {socks_proxy} %h %p"
    raise ClusterError(
        "SSH through proxy required but no supported proxying tools found"
    )
