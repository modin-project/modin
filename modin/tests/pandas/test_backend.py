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

import platform
import re
from unittest.mock import patch

import pytest
import tqdm.auto

import modin.pandas as pd
from modin.config import Backend
from modin.config import context as config_context
from modin.tests.pandas.utils import df_equals

WINDOWS_RAY_SKIP_MARK = pytest.mark.skipif(
    platform.system() == "Windows",
    reason=(
        "Some windows tests with engine != ray use 2 cores, but that "
        + "doesn't work with ray due to "
        + "https://github.com/modin-project/modin/issues/7387"
    ),
)


def test_new_dataframe_uses_default_backend():
    # We run this test with `Backend` set to just one value (instead of
    # trying to look for every possible `Backend` value in the same pytest
    # process) because switching to the MPI backend within a test process
    # that's not set up to run MPI (i.e. because the test process has been
    # started `mpiexec` instead of just `pytest`) would cause errors. We assume
    # that CI runs this test file once with every possible `Backend`.
    assert pd.DataFrame([1]).get_backend() == Backend.get()


@pytest.mark.parametrize("setter_method", ["set_backend", "move_to"])
@pytest.mark.parametrize(
    "inplace_kwargs",
    [
        pytest.param({"inplace": True}, id="inplace"),
        pytest.param({"inplace": False}, id="not_inplace"),
        pytest.param({}, id="no_inplace_kwargs"),
    ],
)
@pytest.mark.parametrize(
    "starting_backend, new_backend, expected_result_backend",
    [
        pytest.param(Backend.get(), "pandas", "Pandas", id="current_to_pandas"),
        pytest.param("pandas", Backend.get(), Backend.get(), id="pandas_to_current"),
        pytest.param(
            Backend.get(), "python_test", "Python_Test", id="current_to_python"
        ),
        pytest.param(
            "python_test", Backend.get(), Backend.get(), id="python_to_current"
        ),
        pytest.param("python_test", "pandas", "Pandas", id="python_to_pandas1"),
        pytest.param("PYTHON_test", "PANDAS", "Pandas", id="python_to_pandas2"),
        pytest.param("pandas", "python_test", "Python_Test", id="pandas_to_python"),
        pytest.param("pandas", "pandas", "Pandas", id="pandas_to_pandas"),
        pytest.param(
            "python_test", "python_test", "Python_Test", id="python_to_python"
        ),
        pytest.param(
            "ray",
            "dask",
            "Dask",
            id="ray_to_dask",
            marks=WINDOWS_RAY_SKIP_MARK,
        ),
        pytest.param(
            "dask",
            "ray",
            "Ray",
            id="dask_to_ray",
            marks=WINDOWS_RAY_SKIP_MARK,
        ),
        pytest.param(
            "ray",
            "python_test",
            "Python_Test",
            id="ray_to_python",
            marks=WINDOWS_RAY_SKIP_MARK,
        ),
        pytest.param("dask", "python_test", "Python_Test", id="dask_to_python"),
        pytest.param(
            "python_test",
            "ray",
            "Ray",
            id="python_to_ray",
            marks=WINDOWS_RAY_SKIP_MARK,
        ),
        pytest.param("python_test", "dask", "Dask", id="python_to_dask"),
        pytest.param("ray", "ray", "Ray", id="ray_to_ray", marks=WINDOWS_RAY_SKIP_MARK),
        pytest.param("dask", "dask", "Dask", id="dask_to_dask"),
    ],
)
@pytest.mark.parametrize(
    "data_class",
    [
        pytest.param(pd.DataFrame, id="dataframe"),
        pytest.param(pd.Series, id="series"),
    ],
)
def test_set_valid_backend(
    setter_method,
    inplace_kwargs,
    starting_backend,
    new_backend,
    data_class,
    expected_result_backend,
):
    progress_iter_count = 2
    with patch.object(
        tqdm.auto, "trange", return_value=range(progress_iter_count)
    ) as mock_trange, config_context(Backend=starting_backend):
        original_df = data_class([1])
        # convert to pandas for comparison while still on the `starting_backend`.
        original_df_as_pandas = original_df.modin.to_pandas()
        method_result = getattr(original_df, setter_method)(
            new_backend, **inplace_kwargs
        )
        if inplace_kwargs.get("inplace", False):
            assert method_result is None
            result_df = original_df
        else:
            assert method_result is not None
            result_df = method_result
        assert result_df.get_backend() == expected_result_backend
        df_equals(result_df, original_df_as_pandas)
        # The global Backend should remain the same even if we change the
        # backend for a single dataframe.
        assert Backend.get() == Backend.normalize(starting_backend)
        if Backend.normalize(starting_backend) == Backend.normalize(
            expected_result_backend
        ):
            mock_trange.assert_not_called()
        else:
            # trange constructor is only called once and the iterator is consumed
            # progress_iter_count times, but we can't easily assert on the number of iterations
            mock_trange.assert_called_once()


def test_same_backend():
    with patch.object(
        tqdm.auto, "trange", return_value=range(2)
    ) as mock_trange, config_context(Backend="Python_Test"):
        df = pd.DataFrame([1])
        new_df = df.set_backend("Python_Test")
        mock_trange.assert_not_called()
        assert new_df.get_backend() == "Python_Test"
        new_df = df.set_backend("Python_Test", inplace=True)
        mock_trange.assert_not_called()
        assert new_df is None
        assert df.get_backend() == "Python_Test"


def test_set_nonexistent_backend():
    backend_choice_string = ", ".join(f"'{choice}'" for choice in Backend.choices)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Unknown backend 'does_not_exist'. "
            + f"Available backends are: {backend_choice_string}"
        ),
    ):
        pd.DataFrame([1]).set_backend("does_not_exist")


@pytest.mark.parametrize("backend", [None, 1, [], {}])
def test_wrong_backend_type(backend):
    with pytest.raises(
        TypeError,
        match=re.escape(
            "Backend value should be a string, but instead it is "
            + f"{repr(backend)} of type {type(backend)}"
        ),
    ):
        pd.DataFrame([1]).set_backend(backend)


def test_get_backend_docstrings():
    dataframe_method = pd.DataFrame.get_backend
    series_method = pd.Series.get_backend
    assert dataframe_method.__doc__ != series_method.__doc__
    assert dataframe_method.__doc__ == series_method.__doc__.replace(
        "Series", "DataFrame"
    )


@pytest.mark.parametrize("setter_method", ["set_backend", "move_to"])
def test_set_backend_docstrings(setter_method):
    dataframe_method = getattr(pd.DataFrame, setter_method)
    series_method = getattr(pd.Series, setter_method)
    assert dataframe_method.__doc__ != series_method.__doc__
    assert dataframe_method.__doc__ == series_method.__doc__.replace(
        "Series", "DataFrame"
    )
