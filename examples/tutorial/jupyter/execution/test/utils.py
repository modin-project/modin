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
download_taxi_dataset = f"""import os
import urllib.request
if not os.path.exists("{test_dataset_path}"):
    url_path = "https://modin-datasets.intel.com/testing/yellow_tripdata_2015-01.csv"
    urllib.request.urlretrieve(url_path, "{test_dataset_path}")
    """


# Default kernel name for ``ExecutePreprocessor`` to be created
_default_kernel_name = "python3"


def set_kernel(kernel_name):
    """
    Set custom kernel for ``ExecutePreprocessor`` to be created.

    Parameters
    ----------
    kernel_name : str
        Kernel name.
    """
    global _default_kernel_name
    _default_kernel_name = kernel_name


def make_execute_preprocessor():
    """
    Make ``ExecutePreprocessor`` with the `_default_kernel_name`.

    Returns
    -------
    nbconvert.preprocessors.ExecutePreprocessor
        Execute processor entity.

    Notes
    -----
    Note that `_default_kernel_name` can be changed for the concrete executions
    (e.g., ``PandasOnUnidist`` with MPI backend).
    """
    return ExecutePreprocessor(timeout=600, kernel_name=_default_kernel_name)


def _execute_notebook(notebook):
    """
    Execute a jupyter notebook.

    Parameters
    ----------
    notebook : file-like or str
        File-like object or path to the notebook to execute.
    """
    nb = nbformat.read(notebook, as_version=nbformat.NO_CONVERT)
    ep = make_execute_preprocessor()
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
