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

import threading
import os
import re
import traceback
import sys
from hashlib import sha1
from typing import Callable
import subprocess

import yaml

try:
    # for ray>=1.0.1
    from ray.autoscaler.sdk import (
        create_or_update_cluster,
        teardown_cluster,
        get_head_node_ip,
        bootstrap_config,
    )
except ModuleNotFoundError:
    # for ray==1.0.0
    from ray.autoscaler.commands import (
        create_or_update_cluster,
        teardown_cluster,
        get_head_node_ip,
        _bootstrap_config as bootstrap_config,
    )

from .base import (
    CannotSpawnCluster,
    CannotDestroyCluster,
    ConnectionDetails,
    _get_ssh_proxy_command,
)
from .cluster import BaseCluster, Provider


class _ThreadTask:
    def __init__(self, target: Callable):
        self.target = target
        self.thread: threading.Thread = None
        self.exc: Exception = None
        self.silent = False


class _Immediate:
    def __init__(self, target: Callable):
        self.target = target

    def start(self):
        self.target()

    def join(self):
        pass


class RayCluster(BaseCluster):
    target_engine = "Cloudray"
    target_partition = "Pandas"

    __base_config = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "ray-autoscaler.yml"
    )
    __instance_key = {Provider.AWS: "InstanceType"}
    __image_key = {Provider.AWS: "ImageId"}
    __credentials_env = {Provider.AWS: "AWS_SHARED_CREDENTIALS_FILE"}

    def __init__(self, *a, **kw):
        self.spawner = _ThreadTask(self.__do_spawn)
        self.destroyer = _ThreadTask(self.__do_destroy)

        self.ready = False
        super().__init__(*a, **kw)

        if self.provider.credentials_file is not None:
            try:
                config_key = self.__credentials_env[self.provider.name]
            except KeyError:
                raise ValueError(f"Unsupported provider: {self.provider.name}")
            os.environ[config_key] = self.provider.credentials_file

        self.config = self.__make_config()
        self.config_file = self.__save_config(self.config)

    def _spawn(self, wait=True):
        self.__run_thread(wait, self.spawner)

    def _destroy(self, wait=True):
        self.__run_thread(wait, self.destroyer)

    def __run_thread(self, wait, task: _ThreadTask):
        if not task.thread:
            task.thread = (_Immediate if wait else threading.Thread)(target=task.target)
            task.thread.start()

        if wait:
            task.silent = True
            task.thread.join()
            exc, task.exc = task.exc, None
            if exc:
                raise exc

    def __make_config(self):
        with open(self.__base_config) as inp:
            config = yaml.safe_load(inp.read())

        # cluster and provider details
        config["cluster_name"] = self.cluster_name
        config["min_workers"] = self.worker_count
        config["max_workers"] = self.worker_count
        config["initial_workers"] = self.worker_count
        config["provider"]["type"] = self.provider.name
        if self.provider.region:
            config["provider"]["region"] = self.provider.region
        if self.provider.zone:
            config["provider"]["availability_zone"] = self.provider.zone

        # connection details
        config["auth"]["ssh_user"] = "ubuntu"
        socks_proxy_cmd = _get_ssh_proxy_command()
        if socks_proxy_cmd:
            config["auth"]["ssh_proxy_command"] = socks_proxy_cmd

        # instance types
        try:
            instance_key = self.__instance_key[self.provider.name]
            image_key = self.__image_key[self.provider.name]
        except KeyError:
            raise ValueError(f"Unsupported provider: {self.provider.name}")

        config["head_node"][instance_key] = self.head_node_type
        config["head_node"][image_key] = self.provider.image
        config["worker_nodes"][instance_key] = self.worker_node_type
        config["worker_nodes"][image_key] = self.provider.image

        # NOTE: setup_commands may be list with several sets of shell commands
        # this change only first set defining the remote environment
        res = self._update_conda_requirements(config["setup_commands"][0])
        config["setup_commands"][0] = res

        return bootstrap_config(config)

    def _conda_requirements(self):
        import shlex

        reqs = []

        reqs.extend(self._get_python_version())

        if self.add_conda_packages:
            if not any(re.match(r"modin(\W|$)", p) for p in self.add_conda_packages):
                # user didn't define modin release;
                # use automatically detected modin release from local context
                reqs.append(self._get_modin_version())

            reqs.extend(self.add_conda_packages)
        else:
            reqs.append(self._get_modin_version())

        # this is needed, for example, for dependencies that
        # looks like: "scikit-learn>=0.23"
        reqs_with_quotes = [shlex.quote(req) for req in reqs]

        return reqs_with_quotes

    def _update_conda_requirements(self, setup_commands: str):
        return setup_commands.replace(
            "{{CONDA_PACKAGES}}", " ".join(self._conda_requirements())
        )

    @staticmethod
    def _get_python_version():
        major = sys.version_info.major
        minor = sys.version_info.minor
        micro = sys.version_info.micro
        return [f"python>={major}.{minor}", f"python<={major}.{minor}.{micro}"]

    @staticmethod
    def _get_modin_version():
        from modin import __version__

        # for example: 0.8.0+116.g5e50eef.dirty
        return f"modin=={__version__.split('+')[0]}"

    @staticmethod
    def __save_config(config):
        cfgdir = os.path.abspath(os.path.expanduser("~/.modin/cloud"))
        os.makedirs(cfgdir, mode=0o700, exist_ok=True)
        namehash = sha1(repr(config).encode("utf8")).hexdigest()[:8]
        entry = os.path.join(cfgdir, f"config-{namehash}.yml")

        with open(entry, "w") as out:
            out.write(yaml.dump(config))
        return entry

    def __do_spawn(self):
        try:
            create_or_update_cluster(
                self.config_file,
                no_restart=False,
                restart_only=False,
                no_config_cache=False,
            )
            # need to re-load the config, as create_or_update_cluster() modifies it
            with open(self.config_file) as inp:
                self.config = yaml.safe_load(inp.read())
            self.ready = True
        except BaseException as ex:
            self.spawner.exc = CannotSpawnCluster(
                "Cannot spawn cluster", cause=ex, traceback=traceback.format_exc()
            )
            if not self.spawner.silent:
                sys.stderr.write(f"Cannot spawn cluster:\n{traceback.format_exc()}\n")

    def __do_destroy(self):
        try:
            teardown_cluster(self.config_file)
            self.ready = False
            self.config = None
        except BaseException as ex:
            self.destroyer.exc = CannotDestroyCluster(
                "Cannot destroy cluster", cause=ex, traceback=traceback.format_exc()
            )
            if not self.destroyer.silent:
                sys.stderr.write(f"Cannot destroy cluster:\n{traceback.format_exc()}\n")

    def _get_connection_details(self) -> ConnectionDetails:
        """
        Gets the coordinates on how to connect to cluster frontend node.
        """
        assert self.ready, "Cluster is not ready, cannot get connection details"
        return ConnectionDetails(
            user_name=self.config["auth"]["ssh_user"],
            key_file=self.config["auth"]["ssh_private_key"],
            address=get_head_node_ip(self.config_file),
        )

    def _get_main_python(self) -> str:
        """
        Gets the path to 'main' interpreter (the one that houses created environment for running everything)
        """
        return "~/miniconda/envs/modin/bin/python"

    def wrap_cmd(self, cmd: list):
        """
        Wraps command into required incantation for bash to read ~/.bashrc which is needed
        to make "conda foo" commands work
        """
        return subprocess.list2cmdline(
            [
                "bash",
                "-ic",
                # workaround for https://github.com/conda/conda/issues/8385
                subprocess.list2cmdline(["conda", "activate", "modin", "&&"] + cmd),
            ]
        )
