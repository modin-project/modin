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
    _execute_notebook,
    _find_code_cell_idx,
    _replace_str,
    download_taxi_dataset,
    test_dataset_path,
)

local_notebooks_dir = "examples/tutorial/jupyter/execution/pandas_on_ray/local"


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
def test_exercise_3():
    modified_notebook_path = os.path.join(local_notebooks_dir, "exercise_3_test.ipynb")
    nb = nbformat.read(
        os.path.join(local_notebooks_dir, "exercise_3.ipynb"),
        as_version=nbformat.NO_CONVERT,
    )

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
    # need to update example, `.mad` doesn't exist
    # _execute_notebook(modified_notebook_path)


# this notebook works "as is" but for testing purposes we can use smaller dataset
def test_exercise_4():
    modified_notebook_path = os.path.join(local_notebooks_dir, "exercise_4_test.ipynb")
    nb = nbformat.read(
        os.path.join(local_notebooks_dir, "exercise_4.ipynb"),
        as_version=nbformat.NO_CONVERT,
    )

    s3_path_cell = f's3_path = "{test_dataset_path}"\n' + download_taxi_dataset
    _replace_str(
        nb,
        's3_path = "s3://dask-data/nyc-taxi/2015/yellow_tripdata_2015-01.csv"',
        s3_path_cell,
    )

    nbformat.write(nb, modified_notebook_path)
    _execute_notebook(modified_notebook_path)
