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

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

test_dataset_path = "taxi.csv"
ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

download_taxi_dataset = f"""import os
import urllib.request
if not os.path.exists("{test_dataset_path}"):
    url_path = "https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2021-01.csv"
    urllib.request.urlretrieve(url_path, "{test_dataset_path}")
    """


def _execute_notebook(notebook):
    """
    Execute a jupyter notebook.

    Parameters
    ----------
    notebook : file-like or str
        File-like object or path to the notebook to execute.
    """
    nb = nbformat.read(notebook, as_version=nbformat.NO_CONVERT)
    ep.preprocess(nb)


def _find_code_cell_idx(nb, identifier):
    """
    Find code cell index by provided ``identifier``.

    Parameters
    ----------
    nb : dict
        Dictionary representation of the notebook to look for.
    identifier : str
        Unique string which target code cell should contain.

    Returns
    -------
    int
        Code cell index by provided ``identifier``.

    Notes
    -----
    Assertion will be raised if ``identifier`` is found in
    several code cells or isn't found at all.
    """
    import_cell_idx = [
        idx
        for idx, cell in enumerate(nb["cells"])
        if cell["cell_type"] == "code" and identifier in cell["source"]
    ]
    assert len(import_cell_idx) == 1
    return import_cell_idx[0]


def _replace_str(nb, original_str, str_to_replace):
    """
    Replace ``original_str`` with ``str_to_replace`` in the provided notebook.

    Parameters
    ----------
    nb : dict
        Dictionary representation of the notebook which requires replacement.
    original_str : str
        Original string which should be replaced.
    str_to_replace : str
        String to replace original string.

    Notes
    -----
    Assertion will be raised if ``original_str`` is found in
    several code cells or isn't found at all.
    """
    import_cell_idx = _find_code_cell_idx(nb, original_str)
    nb["cells"][import_cell_idx]["source"] = nb["cells"][import_cell_idx][
        "source"
    ].replace(original_str, str_to_replace)


def test_omnisci_exercise_1():
    modified_notebook_path = (
        "examples/tutorial/jupyter/execution/omnisci_on_native/local/exercise_1_test.ipynb"
    )
    nb = nbformat.read(
        "examples/tutorial/jupyter/execution/omnisci_on_native/local/exercise_1.ipynb",
        as_version=nbformat.NO_CONVERT,
    )
    _replace_str(nb, "import pandas as pd", "import modin.pandas as pd")

    nbformat.write(nb, modified_notebook_path)
    _execute_notebook(modified_notebook_path)


def test_omnisci_exercise_2():
    modified_notebook_path = (
        "examples/tutorial/jupyter/execution/omnisci_on_native/local/exercise_2_test.ipynb"
    )
    nb = nbformat.read(
        "examples/tutorial/jupyter/execution/omnisci_on_native/local/exercise_2.ipynb",
        as_version=nbformat.NO_CONVERT,
    )

    _replace_str(
        nb,
        'path = "s3://dask-data/nyc-taxi/2015/yellow_tripdata_2015-01.csv"',
        '# path = "s3://dask-data/nyc-taxi/2015/yellow_tripdata_2015-01.csv"',
    )

    new_optional_cell = f'path = "{test_dataset_path}"\n' + download_taxi_dataset

    optional_cell_idx = _find_code_cell_idx(nb, "[Optional] Download data locally.")
    nb["cells"][optional_cell_idx]["source"] = new_optional_cell

    nbformat.write(nb, modified_notebook_path)
    _execute_notebook(modified_notebook_path)


def test_omnisci_exercise_3():
    modified_notebook_path = (
        "examples/tutorial/jupyter/execution/omnisci_on_native/local/exercise_3_test.ipynb"
    )
    nb = nbformat.read(
        "examples/tutorial/jupyter/execution/omnisci_on_native/local/exercise_3.ipynb",
        as_version=nbformat.NO_CONVERT,
    )

    nbformat.write(nb, modified_notebook_path)
    _execute_notebook(modified_notebook_path)
