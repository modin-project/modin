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

import unittest.mock as mock
import pytest
from collections import namedtuple
from inspect import signature
from modin.experimental.cloud.rayscale import (
    RayCluster,
    create_or_update_cluster,
    teardown_cluster,
    get_head_node_ip,
    bootstrap_config,
)
from modin.experimental.cloud.cluster import Provider


@pytest.fixture
def make_bootstrap_config_mock():
    def bootstrap_config_mock(config, *args, **kwargs):
        signature(bootstrap_config).bind(config, *args, **kwargs)
        config["auth"]["ssh_user"] = "modin"
        config["auth"]["ssh_private_key"] = "X" * 20
        return config

    return bootstrap_config_mock


@pytest.fixture
def make_get_head_node_ip_mock():
    def get_head_node_ip_mock(*args, **kwargs):
        signature(get_head_node_ip).bind(*args, **kwargs)
        return "127.0.0.1"

    return get_head_node_ip_mock


@pytest.fixture
def make_teardown_cluster_mock():
    return lambda *args, **kw: signature(teardown_cluster).bind(*args, **kw)


@pytest.fixture
def make_create_or_update_cluster_mock():
    return lambda *args, **kw: signature(create_or_update_cluster).bind(*args, **kw)


@pytest.fixture
def make_ray_cluster(make_bootstrap_config_mock):
    def ray_cluster(conda_packages=None):
        with mock.patch(
            "modin.experimental.cloud.rayscale.bootstrap_config",
            make_bootstrap_config_mock,
        ):
            ray_cluster = RayCluster(
                Provider(name="aws"),
                add_conda_packages=conda_packages,
            )
        return ray_cluster

    return ray_cluster


def test_bootstrap_config(make_ray_cluster):
    make_ray_cluster()


def test_get_head_node_ip(make_ray_cluster, make_get_head_node_ip_mock):
    ray_cluster = make_ray_cluster()

    with mock.patch(
        "modin.experimental.cloud.rayscale.get_head_node_ip", make_get_head_node_ip_mock
    ):
        ray_cluster.ready = True
        details = ray_cluster._get_connection_details()
        assert details.address == "127.0.0.1"


def test_teardown_cluster(make_ray_cluster, make_teardown_cluster_mock):
    with mock.patch(
        "modin.experimental.cloud.rayscale.teardown_cluster", make_teardown_cluster_mock
    ):
        make_ray_cluster()._destroy(wait=True)


def test_create_or_update_cluster(make_ray_cluster, make_create_or_update_cluster_mock):
    with mock.patch(
        "modin.experimental.cloud.rayscale.create_or_update_cluster",
        make_create_or_update_cluster_mock,
    ):
        make_ray_cluster()._spawn(wait=True)


@pytest.mark.parametrize(
    "setup_commands_source",
    [
        r"""conda create --clone base --name modin --yes
        conda activate modin
        conda install --yes {{CONDA_PACKAGES}}
        """
    ],
)
@pytest.mark.parametrize(
    "user_packages",
    [
        ["scikit-learn>=0.23", "modin==0.8.0"],
        None,
    ],
)
def test_update_conda_requirements(
    make_ray_cluster,
    setup_commands_source,
    user_packages,
):
    fake_version = namedtuple("FakeVersion", "major minor micro")(7, 12, 45)
    with mock.patch("sys.version_info", fake_version):
        setup_commands_result = make_ray_cluster(
            user_packages
        )._update_conda_requirements(setup_commands_source)

    assert f"python>={fake_version.major}.{fake_version.minor}" in setup_commands_result
    assert (
        f"python<={fake_version.major}.{fake_version.minor}.{fake_version.micro}"
        in setup_commands_result
    )
    assert "{{CONDA_PACKAGES}}" not in setup_commands_result

    if user_packages:
        for package in user_packages:
            assert package in setup_commands_result
    else:
        assert "modin=" in setup_commands_result
