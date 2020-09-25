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
from modin.experimental.cloud.rayscale import RayCluster
from modin.experimental.cloud.cluster import Provider


@pytest.mark.parametrize(
    "setup_commands_source",
    [
        r"""conda create --clone base --name modin --yes
        conda activate modin
        conda install --yes {{CONDA_PACKAGES}}

        pip install modin "ray==0.8.7" cloudpickle
        """
    ],
)
def test_update_conda_requirements(setup_commands_source):
    with mock.patch(
        "modin.experimental.cloud.rayscale._bootstrap_config", lambda config: config
    ):
        ray_cluster = RayCluster(
            Provider(name="aws"), add_conda_packages=["scikit-learn>=0.23"]
        )

    fake_version = namedtuple("FakeVersion", "major minor micro")(7, 12, 45)
    with mock.patch("sys.version_info", fake_version):
        setup_commands_result = ray_cluster._update_conda_requirements(
            setup_commands_source
        )

    assert f"python>={fake_version.major}.{fake_version.minor}" in setup_commands_result
    assert (
        f"python<={fake_version.major}.{fake_version.minor}.{fake_version.micro}"
        in setup_commands_result
    )
    assert "scikit-learn>=0.23" in setup_commands_result
    assert "{{CONDA_PACKAGES}}" not in setup_commands_result
