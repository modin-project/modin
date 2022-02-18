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

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import pytest

test_dataset_path = "taxi.csv"
ep = ExecutePreprocessor(timeout=600, kernel_name="python3")

download_taxi_dataset = f"""import os
import urllib.request
if not os.path.exists("{test_dataset_path}"):
    url_path = "https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2021-01.csv"
    urllib.request.urlretrieve(url_path, "{test_dataset_path}")
    """
notebooks_dir = (
    "examples/tutorial/tutorial_notebooks/introduction/pandas_storage_format"
)


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


def _set_dask_engine(nb):
    _replace_str(nb, "# import modin.config as cfg", "import modin.config as cfg")
    _replace_str(nb, '# cfg.Engine.put("dask")', 'cfg.Engine.put("dask")')


# in this notebook user should replace 'import pandas as pd' with
# 'import modin.pandas as pd' to make notebook work
@pytest.mark.parametrize("use_dask_engine", [True, False])
def test_exercise_1(use_dask_engine):
    modified_notebook_path = os.path.join(notebooks_dir, "exercise_1_test.ipynb")
    nb = nbformat.read(
        os.path.join(notebooks_dir, "exercise_1.ipynb"),
        as_version=nbformat.NO_CONVERT,
    )
    if use_dask_engine:
        _set_dask_engine(nb)

    _replace_str(nb, "import pandas as pd", "import modin.pandas as pd")

    nbformat.write(nb, modified_notebook_path)
    _execute_notebook(modified_notebook_path)


# this notebook works "as is" but for testing purposes we can use smaller dataset
@pytest.mark.parametrize("use_dask_engine", [True, False])
def test_exercise_2(use_dask_engine):
    modified_notebook_path = os.path.join(notebooks_dir, "exercise_2_test.ipynb")
    nb = nbformat.read(
        os.path.join(notebooks_dir, "exercise_2.ipynb"),
        as_version=nbformat.NO_CONVERT,
    )
    if use_dask_engine:
        _set_dask_engine(nb)

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


# in this notebook user should add custom mad implementation
# to make notebook work
@pytest.mark.parametrize("use_dask_engine", [True, False])
def test_exercise_3(use_dask_engine):
    modified_notebook_path = os.path.join(notebooks_dir, "exercise_3_test.ipynb")
    nb = nbformat.read(
        os.path.join(notebooks_dir, "exercise_3.ipynb"),
        as_version=nbformat.NO_CONVERT,
    )
    if use_dask_engine:
        _set_dask_engine(nb)

    user_mad_implementation = """PandasQueryCompiler.sq_mad_custom = TreeReduce.register(lambda cell_value, **kwargs: cell_value ** 2,
                                                             pandas.DataFrame.mad)

def sq_mad_func(self, axis=None, skipna=True, level=None, **kwargs):
    if axis is None:
        axis = 0

    return self._reduce_dimension(
        self._query_compiler.sq_mad_custom(
            axis=axis, skipna=skipna, level=level, **kwargs
        )
    )

pd.DataFrame.sq_mad_custom = sq_mad_func

modin_mad_custom = df.sq_mad_custom()
    """

    _replace_str(nb, "modin_mad_custom = ...", user_mad_implementation)

    nbformat.write(nb, modified_notebook_path)
    _execute_notebook(modified_notebook_path)


# this notebook works "as is" but for testing purposes we can use smaller dataset
@pytest.mark.parametrize("use_dask_engine", [True, False])
def test_exercise_4(use_dask_engine):
    modified_notebook_path = os.path.join(notebooks_dir, "exercise_4_test.ipynb")
    nb = nbformat.read(
        os.path.join(notebooks_dir, "exercise_4.ipynb"),
        as_version=nbformat.NO_CONVERT,
    )
    if use_dask_engine:
        _replace_str(nb, '# Engine.put("dask")', 'Engine.put("dask")')

    s3_path_cell = f's3_path = "{test_dataset_path}"\n' + download_taxi_dataset
    _replace_str(
        nb,
        's3_path = "s3://dask-data/nyc-taxi/2015/yellow_tripdata_2015-01.csv"',
        s3_path_cell,
    )

    nbformat.write(nb, modified_notebook_path)
    _execute_notebook(modified_notebook_path)
