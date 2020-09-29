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
import warnings

import yaml
from ray.autoscaler.commands import (
    create_or_update_cluster,
    teardown_cluster,
    get_head_node_ip,
    _bootstrap_config,
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

        return _bootstrap_config(config)

    def _conda_requirements(self):
        import shlex

        reqs = []

        reqs.extend(self._get_python_version())

        if self.add_conda_packages:
            reqs.extend(self.add_conda_packages)

        # this is needed, for example, for dependencies that
        # looks like: "scikit-learn>=0.23"
        reqs_with_quotes = [shlex.quote(req) for req in reqs]

        return reqs_with_quotes

    def _update_conda_requirements(self, setup_commands: str):
        setup_commands = setup_commands.replace(
            "{{CONDA_PACKAGES}}", " ".join(self._conda_requirements())
        )

        return setup_commands.replace(
            "{{INSTALL_MODIN_COMMAND}}", self._get_modin_install_command()
        )

    @staticmethod
    def _get_python_version():
        major = sys.version_info.major
        minor = sys.version_info.minor
        micro = sys.version_info.micro
        return [f"python>={major}.{minor}", f"python<={major}.{minor}.{micro}"]

    @staticmethod
    def _git_state():
        """
        Suppose git in PATH.
        """
        import modin

        cwd = os.path.dirname(modin.__file__)
        git_branch_vv = subprocess.check_output(
            ["git", "branch", "-vv"], cwd=cwd, encoding="utf-8"
        ).strip("\n")
        git_remote_v = subprocess.check_output(
            ["git", "remote", "-v"], cwd=cwd, encoding="utf-8"
        ).strip("\n")

        local_branch = None
        remote_branch = None
        for line in re.split(r"\n+", git_branch_vv):
            # * LOCAL_BRANCH HASH [REPO_ALIAS/REMOTE_BRANCH[: STATUS]] [Commit message]
            if line.startswith("*"):
                local_branch = re.split(r"\s+", line)[1]
                if local_branch == "HEAD":
                    # example case: (HEAD detached at modin/sync-modin-between-contexts)
                    raise ValueError("You are in 'detached HEAD' state")
                if "[" not in line:
                    raise ValueError(
                        f"local branch: [{local_branch}] does not track remote"
                    )
                dirty_remote_branch = re.split(r"\s+", line)[3]
                # remove "(" and ":" or ")"
                remote_branch = dirty_remote_branch[1:][:-1]
                break

        try:
            repo_alias, remote_branch = remote_branch.split("/", maxsplit=1)
        except ValueError:
            raise ValueError(
                f"The remote branch: [{remote_branch}] obtained from \
                git: [{git_branch_vv}] for local_branch: [{local_branch}] is not valid"
            )

        repo_name = None
        for line in re.split(r"\n+", git_remote_v):
            # repo_alias repo_name (fetch/push)
            if line.startswith(repo_alias):
                repo_name = re.split(r"\s+", line)[1]
                break

        if repo_name is None:
            raise ValueError(
                f"git remote -v: [{git_remote_v}] doesn't include info for repo alias: [{repo_alias}]"
            )

        return repo_name, remote_branch

    @staticmethod
    def _get_modin_install_command():
        from modin import __version__

        if len(__version__.split("+")) == 1:
            # example version: 0.8.0
            return f"conda install --yes -c intel/label/validation \
                -c conda-forge modin=={__version__}"
        else:
            # example version: 0.8.0+103.gfe0afed.dirty
            # currently install only from last commit in any branch
            try:
                repo, branch = RayCluster._git_state()
            except Exception as er:
                warnings.warn(str(er))
                warnings.warn(
                    "failed get git repo and branch; installing latest release of modin"
                )
                return "conda install --yes -c intel/label/validation -c conda-forge modin"

            modin_install = f"""
        sudo apt-get update -y
        sudo apt-get install -y build-essential

        rm -Rf modin
        git clone --single-branch --depth 1 --branch {branch} {repo}
        (cd modin && pip install -e .[ray] --use-feature=2020-resolver)
        (cd modin && pip install -e .[remote] --use-feature=2020-resolver)
        (cd modin && pip install -e . --use-feature=2020-resolver)"""

        return modin_install

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
                override_min_workers=None,
                override_max_workers=None,
                no_restart=False,
                restart_only=False,
                yes=True,
                override_cluster_name=None,
                no_config_cache=False,
                log_old_style=False,
                log_color="auto",
                verbose=1,
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
            teardown_cluster(
                self.config_file,
                yes=True,
                workers_only=False,
                override_cluster_name=None,
                keep_min_workers=0,
                log_old_style=False,
                log_color="auto",
                verbose=1,
            )
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
            address=get_head_node_ip(self.config_file, override_cluster_name=None),
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
                subprocess.list2cmdline(["conda", "run", "-n", "modin"] + cmd),
            ]
        )
