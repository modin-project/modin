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
            id="config_omnisci_first-import_modin_second",
        ),
        pytest.param(
            """
import modin.pandas as pd
import modin.config as cfg
cfg.Engine.put('Native')
cfg.StorageFormat.put('Omnisci')
cfg.IsExperimental.put(True)
""",
            id="import_modin_first-config_omnisci_second",
        ),
    ],
)
@pytest.mark.parametrize("has_other_engines", [True, False])
def test_omnisci_import(import_strategy, has_other_engines):
    """
    Test import of OmniSci engine.

    The import of PyDBEngine requires to set special dlopen flags which make it then
    incompatible to import some other libraries further (like ``pyarrow.gandiva``).
    This test verifies that it's not the case when a user naturally imports Modin
    with OmniSci engine.

    Parameters
    ----------
    import_strategy : str
        There are several scenarios of how a user can import Modin with OmniSci engine:
        configure Modin first to use OmniSci engine and then import ``modin.pandas`` or vice versa.
        This parameters holds a python code, implementing one of these scenarios.
    has_other_engines : bool
        The problem with import may appear depending on whether other engines are
        installed. This parameter indicates whether to remove modules for
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


@pytest.mark.parametrize(
    "import_strategy, expected_to_fail",
    [
        pytest.param(
            """
from modin.experimental.core.execution.native.implementations.omnisci_on_native.utils import PyDbEngine
import pyarrow.gandiva
""",
            True,
            id="import_pydbe_first-pyarrow_gandiva_second",
        ),
        pytest.param(
            """
import pyarrow.gandiva
from modin.experimental.core.execution.native.implementations.omnisci_on_native.utils import PyDbEngine
""",
            False,
            id="import_pyarrow_gandiva_first-pydbe_second",
        ),
    ],
)
def test_omnisci_compatibility_with_pyarrow_gandiva(import_strategy, expected_to_fail):
    """
    Test the current status of compatibility of PyDbEngine and pyarrow.gandiva packages.

    If this test appears to fail, it means that these packages are now compatible/incopmatible,
    if it's so, please post the actual compatibility status to the issue:
    https://github.com/modin-project/modin/issues/3865
    And then inverse `expected_to_fail` parameter for the scenario that has changed its behaviour.

    Parameters
    ----------
    import_strategy : str
        There are several scenarios of how a user can import PyDbEngine and pyarrow.gandiva.
        This parameters holds a python code, implementing one of the scenarios.
    expected_to_fail : bool
        Indicates the estimated compatibility status for the specified `import_strategy`.
        True - the strategy expected to fail, False - the strategy expected to pass.
        Note: we can't use built-in ``pytest.marks.xfail`` as we need to check that the
        expected failure was caused by LLVM error.
    """
    res = subprocess.run(
        [sys.executable, "-c", import_strategy],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if expected_to_fail:
        assert (
            res.returncode != 0
        ), "PyDbEngine and pyarrow.gandiva are now compatible! Please check the test's doc-string for further instructions."
    else:
        assert (
            res.returncode == 0
        ), "PyDbEngine and pyarrow.gandiva are now incompatible! Please check the test's doc-string for further instructions."

    if res.returncode != 0:
        error_msg = res.stderr.decode("utf-8")
        assert (
            error_msg.find("LLVM ERROR") != -1
        ), f"Expected to fail because of LLVM error, but failed because of:\n{error_msg}"
