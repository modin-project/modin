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
import time

from .base import ClusterError, ConnectionDetails, _get_ssh_proxy_command

RPYC_REQUEST_TIMEOUT = 2400


class Connection:
    __current = None
    connect_timeout = 10
    tries = 10
    rpyc_port = 18813

    @staticmethod
    def __wait_noexc(proc: subprocess.Popen, timeout: float):
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None

    def __init__(self, details: ConnectionDetails, main_python: str, log_rpyc=None):
        if log_rpyc is None:
            log_rpyc = os.environ.get("MODIN_LOG_RPYC", "").title() == "True"
        self.proc = None

        # find where rpyc_classic is located
        locator = self.__run(
            self.__build_sshcmd(details),
            [
                main_python,
                "-c",
                "import os; from distutils.dist import Distribution; from distutils.command.install import install; cmd = install(Distribution()); cmd.finalize_options(); print(os.path.join(cmd.install_scripts, 'rpyc_classic.py'))",
            ],
        )
        try:
            out, err = locator.communicate(timeout=self.connect_timeout)
        except subprocess.TimeoutExpired as ex:
            raise ClusterError(
                "Cannot get path to rpyc_classic: cannot connect to host", cause=ex
            )
        if locator.returncode != 0:
            raise ClusterError(
                f"Cannot get path to rpyc_classic, return code: {locator.returncode}"
            )
        rpyc_classic = out.splitlines()[0].strip().decode("utf8")
        if not rpyc_classic:
            raise ClusterError("Got empty path to rpyc_classic")

        port = self.rpyc_port
        cmd = [
            main_python,
            rpyc_classic,
            "--port",
            str(self.rpyc_port),
        ]
        if log_rpyc:
            cmd.extend(["--logfile", "/tmp/rpyc.log"])
        for _ in range(self.tries):
            proc = self.__run(
                self.__build_sshcmd(details, forward_port=port), cmd, capture_out=False
            )
            if self.__wait_noexc(proc, 1) is None:
                # started successfully
                self.proc = proc
                self.rpyc_port = port
                break
            # most likely port is busy, pick random one
            port = random.randint(1024, 65000)
        else:
            raise ClusterError("Unable to bind a local port when forwarding")
        self.__connection = None
        self.__started = time.time()

    @classmethod
    def get(cls):
        if (
            not cls.__current
            or not cls.__current.proc
            or cls.__current.proc.poll() is not None
        ):
            raise ClusterError("SSH tunnel is not running")
        if cls.__current.__connection is None:
            raise ClusterError("Connection not activated")

        return cls.__current.__connection

    def __try_connect(self):
        import rpyc
        from .rpyc_proxy import WrappingService

        try:
            stream = rpyc.SocketStream.connect(
                host="127.0.0.1", port=self.rpyc_port, nodelay=True, keepalive=True
            )
            self.__connection = rpyc.connect_stream(
                stream,
                WrappingService,
                config={"sync_request_timeout": RPYC_REQUEST_TIMEOUT},
            )
        except (ConnectionRefusedError, EOFError):
            if self.proc.poll() is not None:
                raise ClusterError(
                    f"SSH tunnel died, return code: {self.proc.returncode}"
                )

    def activate(self):
        if self.__connection is None:
            self.__try_connect()
            while (
                self.__connection is None
                and time.time() < self.__started + self.connect_timeout + 1.0
            ):
                time.sleep(1.0)
                self.__try_connect()
            if self.__connection is None:
                raise ClusterError("Timeout establishing RPyC connection")

        Connection.__current = self

    def deactivate(self):
        if Connection.__current is self:
            Connection.__current = None

    def stop(self, sigint=signal.SIGINT):
        # capture signal.SIGINT in closure so it won't get removed before __del__ is called
        self.deactivate()
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(sigint)
            if self.__wait_noexc(self.proc, self.connect_timeout) is None:
                self.proc.terminate()
                if self.__wait_noexc(self.proc, self.connect_timeout) is None:
                    self.proc.kill()
        self.proc = None

    def __del__(self):
        self.stop()

    def __build_sshcmd(self, details: ConnectionDetails, forward_port: int = None):
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
            cmdline.extend(
                ["-L", f"127.0.0.1:{forward_port}:127.0.0.1:{self.rpyc_port}"]
            )
        cmdline.append(f"{details.user_name}@{details.address}")

        return cmdline

    @staticmethod
    def __run(sshcmd: list, cmd: list, capture_out: bool = True):
        redirect = subprocess.PIPE if capture_out else subprocess.DEVNULL
        return subprocess.Popen(
            sshcmd + [subprocess.list2cmdline(cmd)],
            stdin=subprocess.DEVNULL,
            stdout=redirect,
            stderr=redirect,
        )
