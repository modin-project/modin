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

import os
import sys
import pytest
from modin.experimental.cloud.rayscale import RayCluster


@pytest.mark.parametrize(
    "setup_commands_source",
    [
        "# setup_commands section of ray_autoscaler.yml define from setup_commands.sh.in in runtime",
        r"""# setup_commands section of ray_autoscaler.yml define from setup_commands.sh.in in runtime
# FROM SETUP_COMMANDS.SH.IN with defined template variable
python==2.7.9
# FROM SETUP_COMMANDS.SH.IN with defined template variable
""",
    ],
)
def test_sync_python(setup_commands_source):
    template_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "../setup_commands.sh.in"
    )
    setup_commands_result = RayCluster._sync_python(
        setup_commands_source, template_path
    )

    # first line is header; should be unchanged
    assert setup_commands_result.split("\n")[0] == setup_commands_source.split("\n")[0]

    major = sys.version_info.major
    minor = sys.version_info.minor
    micro = sys.version_info.micro
    python_version = f"python=={major}.{minor}.{micro}"

    assert python_version in setup_commands_result
    assert "{{PYTHON_VERSION}}" not in setup_commands_result
