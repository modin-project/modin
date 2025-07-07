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

import pandas
import pytest
import tqdm.auto

import modin.pandas as pd
from modin.config import Backend
from modin.config import context as config_context
from modin.tests.pandas.utils import (
    create_test_dfs,
    default_to_pandas_ignore_string,
    df_equals,
)

WINDOWS_RAY_SKIP_MARK = pytest.mark.skipif(
    platform.system() == "Windows",
    reason=(
        "Some windows tests with engine != ray use 2 cores, but that "
        + "doesn't work with ray due to "
        + "https://github.com/modin-project/modin/issues/7387"
    ),
)

# Some modin methods warn about defaulting to pandas at the API layer. That's
# expected and not an error as it would be normally.
pytestmark = pytest.mark.filterwarnings(default_to_pandas_ignore_string)


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


class TestGroupbySetBackend:
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
        "starting_backend, new_backend",
        [
            pytest.param(Backend.get(), "Pandas", id="current_to_pandas"),
            pytest.param("Pandas", Backend.get(), id="pandas_to_current"),
            pytest.param(Backend.get(), "Python_Test", id="current_to_python"),
            pytest.param("Python_Test", Backend.get(), id="python_to_current"),
            pytest.param("Python_Test", "Pandas", id="python_to_pandas"),
            pytest.param("Pandas", "Python_Test", id="pandas_to_python"),
        ],
    )
    @pytest.mark.parametrize(
        "by_level_factory",
        [
            pytest.param(lambda df: ("C", None), id="by_string_column"),
            pytest.param(lambda df: (["C", "D"], None), id="by_list_of_strings"),
            pytest.param(lambda df: (df["C"], None), id="by_series"),
            pytest.param(lambda df: (["C", df["D"]], None), id="by_list_mixed"),
            pytest.param(lambda df: (pandas.Grouper(key="C"), None), id="by_grouper"),
            pytest.param(lambda df: (None, 0), id="level_scalar"),
            pytest.param(lambda df: (None, [0, 1]), id="level_list"),
            pytest.param(
                lambda df: (["C", df["D"]], None), id="by_mixed_string_series"
            ),
        ],
    )
    def test_dataframe(
        self,
        setter_method,
        inplace_kwargs,
        starting_backend,
        new_backend,
        by_level_factory,
    ):
        """Test set_backend functionality for DataFrame groupby objects with various 'by' and 'level' combinations."""
        with config_context(Backend=starting_backend):

            def do_groupby(df):
                by, level = by_level_factory(df)
                return df.groupby(by=by, level=level)

            inplace = inplace_kwargs.get("inplace", False)
            original_modin_df, original_pandas_df = create_test_dfs(
                pandas.DataFrame(
                    data={
                        "A": [1, 2, 3, 4, 5, 6],
                        "B": [10, 20, 30, 40, 50, 60],
                        "C": ["x", "y", "x", "y", "x", "y"],
                        "D": ["p", "p", "q", "q", "r", "r"],
                    },
                    index=pd.MultiIndex.from_tuples(
                        [
                            ("foo", 1),
                            ("foo", 2),
                            ("bar", 1),
                            ("bar", 2),
                            ("baz", 1),
                            ("baz", 2),
                        ],
                        names=["first", "second"],
                    ),
                )
            )

            # Create DataFrame groupby object
            original_groupby = do_groupby(original_modin_df)

            setter_result = getattr(original_groupby, setter_method)(
                new_backend, **inplace_kwargs
            )

            if inplace:
                assert setter_result is None
                result_groupby = original_groupby
                # Verify that the underlying DataFrame's backend was also changed
                assert original_groupby._df.get_backend() == new_backend
            else:
                assert setter_result is not original_groupby
                result_groupby = setter_result
                # Verify original DataFrame's backend was not changed
                assert original_groupby._df.get_backend() == starting_backend

            # Verify backend was changed
            assert result_groupby.get_backend() == new_backend

            # Verify that groupby still works correctly after backend switch
            # Create a fresh groupby for comparison to avoid mixed backend states
            pandas_groupby_sum = do_groupby(original_pandas_df).sum()
            df_equals(
                result_groupby.sum(),
                pandas_groupby_sum,
            )
            if not inplace:
                df_equals(
                    original_groupby.sum(),
                    pandas_groupby_sum,
                )

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
        "starting_backend, new_backend",
        [
            pytest.param(Backend.get(), "Pandas", id="current_to_pandas"),
            pytest.param("Pandas", Backend.get(), id="pandas_to_current"),
            pytest.param(Backend.get(), "Python_Test", id="current_to_python"),
            pytest.param("Python_Test", Backend.get(), id="python_to_current"),
            pytest.param("Python_Test", "Pandas", id="python_to_pandas"),
            pytest.param("Pandas", "Python_Test", id="pandas_to_python"),
        ],
    )
    @pytest.mark.parametrize(
        "by_level_factory",
        [
            pytest.param(lambda series: (None, 0), id="by_index_level_0"),
            pytest.param(
                lambda series: (None, [0, 1]),
                id="by_index_levels_list",
            ),
            pytest.param(
                lambda series: (pandas.Grouper(level=0), None),
                id="by_grouper_level",
            ),
            pytest.param(lambda series: (None, 0), id="level_scalar"),
            pytest.param(lambda series: (None, [0, 1]), id="level_list"),
            pytest.param(lambda series: (series, None), id="by_self"),
            pytest.param(lambda series: (series % 2, None), id="by_self_modulo_2"),
        ],
    )
    def test_series(
        self,
        setter_method,
        inplace_kwargs,
        starting_backend,
        new_backend,
        by_level_factory,
    ):
        """Test set_backend functionality for Series groupby objects with various 'by' and 'level' combinations."""
        with config_context(Backend=starting_backend):
            inplace = inplace_kwargs.get("inplace", False)
            # Create test data with MultiIndex to support level-based grouping
            idx = pd.MultiIndex.from_tuples(
                [
                    ("foo", 1),
                    ("foo", 2),
                    ("bar", 1),
                    ("bar", 2),
                    ("baz", 1),
                    ("baz", 2),
                ],
                names=["first", "second"],
            )
            original_pandas_series = pandas.Series([1, 2, 1, 3, 4, 5], index=idx)
            original_modin_series = pd.Series([1, 2, 1, 3, 4, 5], index=idx)

            def do_groupby(series):
                by, level = by_level_factory(series)
                return series.groupby(by=by, level=level)

            # Create Series groupby object
            original_groupby = do_groupby(original_modin_series)

            setter_result = getattr(original_groupby, setter_method)(
                new_backend, **inplace_kwargs
            )

            if inplace:
                assert setter_result is None
                result_groupby = original_groupby
                # Verify that the underlying Series's backend was also changed
                assert original_groupby._df.get_backend() == new_backend
            else:
                assert setter_result is not original_groupby
                result_groupby = setter_result
                # Verify original Series's backend was not changed
                assert original_groupby._df.get_backend() == starting_backend

            assert result_groupby.get_backend() == new_backend

            pandas_groupby_sum = do_groupby(original_pandas_series).sum()
            df_equals(result_groupby.sum(), pandas_groupby_sum)
            if not inplace:
                df_equals(original_groupby.sum(), pandas_groupby_sum)


# Tests for fallback progress printing when tqdm is not available
@pytest.mark.parametrize(
    "switch_operation,expected_output",
    [
        (None, "Transferring data from Python_Test to Pandas with max estimated shape"),
        (
            "test_operation",
            "Transferring data from Python_Test to Pandas for 'test_operation' with max estimated shape",
        ),
    ],
)
@patch("tqdm.auto.trange", side_effect=ImportError("tqdm not available"))
@config_context(Backend="python_test")
def test_fallback_progress_printing(
    mock_trange, capsys, switch_operation, expected_output
):
    """Test that fallback progress printing works when tqdm is not available and ShowBackendSwitchProgress is enabled."""
    df = pd.DataFrame([1, 2, 3])

    df.set_backend("pandas", switch_operation=switch_operation)

    captured = capsys.readouterr()
    assert expected_output in captured.err
    assert captured.out == ""  # Nothing should go to stdout


@patch("tqdm.auto.trange", side_effect=ImportError("tqdm not available"))
@config_context(Backend="python_test")
def test_fallback_progress_printing_silent_when_disabled(mock_trange, capsys):
    """Test that fallback progress printing is silent when ShowBackendSwitchProgress is disabled."""

    df = pd.DataFrame([1, 2, 3])

    with config_context(ShowBackendSwitchProgress=False):
        df.set_backend("pandas")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


@config_context(Backend="python_test")
def test_tqdm_progress_bar_disabled_when_backend_switch_progress_false(capsys):
    """Test that tqdm progress bar doesn't appear when ShowBackendSwitchProgress is disabled."""
    df = pd.DataFrame([1, 2, 3])

    with config_context(ShowBackendSwitchProgress=False), patch(
        "tqdm.auto.trange"
    ) as mock_trange:
        df.set_backend("pandas")

    mock_trange.assert_not_called()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
