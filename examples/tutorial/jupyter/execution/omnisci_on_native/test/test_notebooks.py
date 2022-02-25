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

import nbformat

MODIN_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), *[".." for _ in range(6)])
)
sys.path.insert(0, MODIN_DIR)
from examples.tutorial.jupyter.execution.test.utils import (  # noqa: E402
    _replace_str,
    _execute_notebook,
    _find_code_cell_idx,
    test_dataset_path,
    download_taxi_dataset,
)

local_notebooks_dir = "examples/tutorial/jupyter/execution/omnisci_on_native/local"


# in this notebook user should replace 'import pandas as pd' with
# 'import modin.pandas as pd' to make notebook work
def test_exercise_1():
    modified_notebook_path = os.path.join(local_notebooks_dir, "exercise_1_test.ipynb")
    nb = nbformat.read(
        os.path.join(local_notebooks_dir, "exercise_1.ipynb"),
        as_version=nbformat.NO_CONVERT,
    )

    _replace_str(nb, "import pandas as pd", "import modin.pandas as pd")

    nbformat.write(nb, modified_notebook_path)
    _execute_notebook(modified_notebook_path)


# this notebook works "as is" but for testing purposes we can use smaller dataset
def test_exercise_2():
    modified_notebook_path = os.path.join(local_notebooks_dir, "exercise_2_test.ipynb")
    nb = nbformat.read(
        os.path.join(local_notebooks_dir, "exercise_2.ipynb"),
        as_version=nbformat.NO_CONVERT,
    )

    new_optional_cell = f'path = "{test_dataset_path}"\n' + download_taxi_dataset

    optional_cell_idx = _find_code_cell_idx(nb, "[Optional] Download data locally.")
    nb["cells"][optional_cell_idx]["source"] = new_optional_cell

    nbformat.write(nb, modified_notebook_path)
    _execute_notebook(modified_notebook_path)


# this notebook works "as is"
def test_exercise_3():
    modified_notebook_path = os.path.join(local_notebooks_dir, "exercise_3_test.ipynb")
    nb = nbformat.read(
        os.path.join(local_notebooks_dir, "exercise_3.ipynb"),
        as_version=nbformat.NO_CONVERT,
    )

    nbformat.write(nb, modified_notebook_path)
    _execute_notebook(modified_notebook_path)
