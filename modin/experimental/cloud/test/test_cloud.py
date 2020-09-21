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

import sys
import pytest
from modin.experimental.cloud.rayscale import RayCluster


@pytest.mark.parametrize(
    "setup_commands_source",
    [
        r"""conda create --clone base --name modin --yes
        conda activate modin
        conda install {{PYTHON_VERSION}}

        pip install modin "ray==0.8.7" cloudpickle
        """
    ],
)
def test_sync_python(setup_commands_source):
    setup_commands_result = RayCluster._sync_python(setup_commands_source)

    major = sys.version_info.major
    minor = sys.version_info.minor
    micro = sys.version_info.micro
    python_version = f"python=={major}.{minor}.{micro}"

    assert python_version in setup_commands_result
    assert "{{PYTHON_VERSION}}" not in setup_commands_result
