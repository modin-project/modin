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
import signal
import os
import random

from .base import ClusterError, ConnectionDetails, _get_ssh_proxy_command


class Connection:
    __current = None
    connect_timeout = 5
    tries = 10
    rpyc_port = 18813

    def __init__(self, details: ConnectionDetails, main_python: str, log_rpyc=None):
        if log_rpyc is None:
            log_rpyc = os.environ.get("MODIN_LOG_RPYC", "").title() == "True"
        self.proc = None

        # find where rpyc_classic is located
        locator = self.__run(
            self.__build_opts(details)
            + [
                main_python,
                "-c",
                "import os; from distutils.dist import Distribution; from distutils.command.install import install; cmd = install(Distribution()); cmd.finalize_options(); print(os.path.join(cmd.install_scripts, 'rpyc_classic.py'))",
            ]
        )
        out, err = locator.communicate(timeout=5)
        if locator.returncode != 0:
            raise ClusterError(
                f"Cannot get path to rpyc_classic, return code: {locator.returncode}"
            )
        rpyc_classic = out.splitlines()[0].strip()
        if not rpyc_classic:
            raise ClusterError("Got empty path to rpyc_classic")

        port = self.rpyc_port
        for _ in range(self.tries):
            cmd = self.__build_opts(details, forward_port=port) + [
                main_python,
                rpyc_classic,
                "--port",
                str(self.rpyc_port),
            ]
            if log_rpyc:
                cmd.extend(["--logfile", "/tmp/rpyc.log"])
            proc = self.__run(cmd, capture_out=False)
            if proc.wait(1) is None:
                # started successfully
                self.proc = proc
                self.rpyc_port = port
                break
            # most likely port is busy, pick random one
            port = random.randint(1024, 65000)
        else:
            raise ClusterError("Unable to bind a local port when forwarding")

        self.activate()

    @classmethod
    def get(cls):
        if (
            not cls.__current
            or not cls.__current.proc
            or cls.__current.proc.poll() is not None
        ):
            raise ClusterError("SSH tunnel is not running")
        import rpyc

        return rpyc.classic.connect(
            "localhost", cls.__current.rpyc_port, keepalive=True
        )

    def activate(self):
        Connection.__current = self

    def deactivate(self):
        if Connection.__current is self:
            Connection.__current = None

    def stop(self):
        self.deactivate()
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)
            if self.proc.wait(self.connect_timeout) is None:
                self.proc.terminate()
                if self.proc.wait(self.connect_timeout) is None:
                    self.proc.kill()
        self.proc = None

    def __del__(self):
        self.stop()

    def __build_opts(self, details: ConnectionDetails, forward_port: int = None):
        opts = [
            ("ConnectTimeout", "{}s".format(self.connect_timeout)),
            ("StrictHostKeyChecking", "no"),
            # Try fewer extraneous key pairs.
            ("IdentitiesOnly", "yes"),
            # Abort if port forwarding fails (instead of just printing to stderr).
            ("ExitOnForwardFailure", "yes"),
            # Quickly kill the connection if network connection breaks (as opposed to hanging/blocking).
            ("ServerAliveInterval", 5),
            ("ServerAliveCountMax", 3),
        ]

        socks_proxy_cmd = _get_ssh_proxy_command()
        if socks_proxy_cmd:
            opts += [("ProxyCommand", socks_proxy_cmd)]

        cmdline = ["ssh", "-i", details.key_file]
        for oname, ovalue in opts:
            cmdline.extend(["-o", f"{oname}={ovalue}"])
        if forward_port:
            cmdline.extend(["-L", f"{forward_port}:localhost:{self.rpyc_port}"])
        cmdline.append(f"{details.user_name}@{details.address}")

        return cmdline

    @staticmethod
    def __run(cmd: list, capture_out: bool = True):
        redirect = subprocess.PIPE if capture_out else None
        return subprocess.Popen(cmd, stdout=redirect, stderr=redirect)
