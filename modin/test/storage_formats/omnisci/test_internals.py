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

import pytest
import sys
import subprocess


@pytest.mark.parametrize(
    "import_strategy",
    [
        pytest.param(
            """
import modin.config as cfg
cfg.Engine.put('Native') # 'omniscidbe'/'dbe' would be imported with dlopen flags first time
cfg.StorageFormat.put('Omnisci')
cfg.IsExperimental.put(True)
import modin.pandas as pd
""",
            id="config_first-import_second",
        ),
        pytest.param(
            """
import modin.pandas as pd
import modin.config as cfg
cfg.Engine.put('Native')
cfg.StorageFormat.put('Omnisci')
cfg.IsExperimental.put(True)
""",
            id="import_first-config_second",
        ),
    ],
)
@pytest.mark.parametrize("has_other_engines", [True, False])
def test_omnisci_import(import_strategy, has_other_engines):
    """
    Test import of OmniSci engine.

    The import of PyDBEngine requires to set special open flags which makes it then
    incompatible to import some other libraries further (like ``pyarrow.gandiva``).
    This test verifies that it's not the case when a user naturally imports Modin
    with OmniSci engine.

    Parameters
    ----------
    import_strategy : str
        There are several scenarios of how user can import Modin with OmniSci engine:
        configure Modin first and then import ``modin.pandas`` or vice versa.
        This parameters holds a python code, implementing one of these scenarious.
    has_other_engines : bool
        The problem with import may appear depending on whether other engines are
        installed. This parameter indicates of whether to remove modules for
        non-omnisci engines before the test.

    Notes
    -----
    The failed import flow may cause segfault, which causes to crash the pytest itself.
    This makes us to run the test in a separate process and check its exit-code to
    decide the success of the test.
    """

    remove_other_engines = """
import sys
sys.modules['ray'] = None
sys.modules['dask'] = None
"""

    if not has_other_engines:
        import_strategy = f"{remove_other_engines}\n{import_strategy}"

    res = subprocess.run(
        [sys.executable, "-c", import_strategy],
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )

    if res.returncode != 0:
        pytest.fail(str(res.stderr))


def test_compatibility_omnisci_with_pyarrow_gandiva():
    """
    Test that PyDbEngine and pyarrow.gandiva packages are still incompatible.

    At the moment of writing this test PyDbEngine (5.8.0) and pyarrow.gandiva (3.0.0) are incompatible.
    If this test appears to fail, it means that these packages are now compatible, if it's so,
    change the failing assert statement and this doc-string accordingly.
    """
    res = subprocess.run(
        [
            sys.executable,
            "-c",
            """
from modin.experimental.core.execution.native.implementations.omnisci_on_native.utils import PyDbEngine
import pyarrow.gandiva
""",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert (
        res.returncode != 0
    ), "PyDbEngine and pyarrow.gandiva are now compatible! Please check the test's doc-string for further instructions."

    if res.returncode != 0:
        error_msg = res.stderr.decode("utf-8")
        assert (
            error_msg.find("LLVM ERROR") != -1
        ), f"Expected to fail because LLVM error, but failed because of:\n{error_msg}"
