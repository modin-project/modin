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

# Tests interactions between a modin frame and a parent or child native pandas frame when one
# object's metadata or data is modified.
# Only valid on the native pandas backend.

import functools

import pandas
import pytest

import modin.pandas as pd
from modin.config import Backend
from modin.config import context as config_context


@pytest.fixture(scope="module", autouse=True)
def mutation_cow_test():
    if Backend.get() != "Pandas":
        pytest.skip(
            reason="tests are only meaningful with pandas backend",
            allow_module_level=True,
        )


@pytest.fixture(scope="function")
def copy_on_write(request):
    # Indirect fixture for toggling copy-on-write when tests are run
    with config_context(
        Backend="Pandas", NativePandasDeepCopy=False
    ), pandas.option_context("mode.copy_on_write", request.param):
        yield request.param


def get_mutation_fixtures(data, **kwargs):
    # Return a fixture that sets the copy_on_write fixture, then passes a modin and native DF together for mutation testing.
    # One parameter combination creates a modin DF from a native DF.
    # The other creates a native DF by calling to_pandas on a modin DF.
    def wrapper(f):
        # Need to create separate functions so parametrized runs don't affect each other.
        def native_first():
            native_input = pandas.DataFrame(data, **kwargs)
            return native_input, pd.DataFrame(native_input)

        def modin_first():
            modin_input = pd.DataFrame(data, **kwargs)
            return modin_input, modin_input.modin.to_pandas()

        @pytest.mark.parametrize("df_factory", [native_first, modin_first])
        @pytest.mark.parametrize(
            "copy_on_write",
            [pytest.param(True, id="CoW"), pytest.param(False, id="no_CoW")],
            indirect=True,
        )
        @functools.wraps(f)
        def test_runner(*args, **kwargs):
            return f(*args, **kwargs)

        return test_runner

    return wrapper


@pytest.mark.parametrize(
    "axis", [pytest.param(0, id="index"), pytest.param(1, id="columns")]
)
@get_mutation_fixtures({"A": [0, 1], "B": [2, 3]})
def test_set_axis_name(axis, copy_on_write, df_factory):
    df1, df2 = df_factory()
    df1.axes[axis].name = "x"
    assert df1.axes[axis].name == "x"
    # Changes do not propagate when copy-on-write is enabled.
    if copy_on_write:
        assert df2.axes[axis].name is None
    else:
        assert df2.axes[axis].name == "x"
    df2.axes[axis].name = "y"
    assert df1.axes[axis].name == ("x" if copy_on_write else "y")
    assert df2.axes[axis].name == "y"


@pytest.mark.parametrize(
    "axis", [pytest.param(0, id="index"), pytest.param(1, id="columns")]
)
@get_mutation_fixtures({"A": [0, 1], "B": [2, 3]}, index=["A", "B"])
def test_rename_axis(axis, copy_on_write, df_factory):
    df1, df2 = df_factory()
    # Renames don't propagate, regardless of CoW.
    df1.rename({"A": "aprime"}, axis=axis, inplace=True)
    assert df1.axes[axis].tolist() == ["aprime", "B"]
    assert df2.axes[axis].tolist() == ["A", "B"]
    df2.rename({"B": "bprime"}, axis=axis, inplace=True)
    assert df1.axes[axis].tolist() == ["aprime", "B"]
    assert df2.axes[axis].tolist() == ["A", "bprime"]


@get_mutation_fixtures({"A": [0, 1], "B": [2, 3]})
def test_locset(copy_on_write, df_factory):
    df1, df2 = df_factory()
    df1.loc[0, "A"] = -1
    assert df1.loc[0, "A"] == -1
    assert df2.loc[0, "A"] == (0 if copy_on_write else -1)
    df2.loc[1, "B"] = 999
    assert df1.loc[1, "B"] == (3 if copy_on_write else 999)
    assert df2.loc[1, "B"] == 999


@get_mutation_fixtures({"A": [0, 1], "B": [2, 3]})
def test_add_column(copy_on_write, df_factory):
    df1, df2 = df_factory()
    df1["C"] = [4, 5]
    assert df1["C"].tolist() == [4, 5]
    # Even with CoW disabled, the new column is not added to df2.
    assert df2.columns.tolist() == ["A", "B"]
    df2["D"] = [6, 7]
    assert df2["D"].tolist() == [6, 7]
    assert df1.columns.tolist() == ["A", "B", "C"]


@get_mutation_fixtures({"A": [0, 1], "B": [2, 3]})
def test_add_row(copy_on_write, df_factory):
    df1, df2 = df_factory()
    df1.loc[9] = [4, 5]
    assert df1.loc[9].tolist() == [4, 5]
    # Even with CoW disabled, the new row is not added to df2.
    assert df2.index.tolist() == [0, 1]
    df2.loc[10] = [6, 7]
    assert df2.loc[10].tolist() == [6, 7]
    assert df1.index.tolist() == [0, 1, 9]


@pytest.mark.filterwarnings("ignore::FutureWarning")
@pytest.mark.filterwarnings("ignore::pandas.errors.ChainedAssignmentError")
@get_mutation_fixtures({"A": [0, 1], "B": [2, 3]})
def test_chained_assignment(copy_on_write, df_factory):
    df1, df2 = df_factory()
    is_assign_noop = copy_on_write and isinstance(df1, pandas.DataFrame)
    df1["A"][0] = -1
    assert df1["A"][0] == (0 if is_assign_noop else -1)
    assert df2["A"][0] == (
        0 if copy_on_write or isinstance(df2, pandas.DataFrame) else -1
    )
    is_assign_noop = copy_on_write and isinstance(df2, pandas.DataFrame)
    df2["B"][1] = 999
    assert df1["B"][1] == (
        3 if copy_on_write or isinstance(df1, pandas.DataFrame) else 999
    )
    assert df2["B"][1] == (3 if is_assign_noop else 999)


@get_mutation_fixtures({"A": [0, 1], "B": [2, 3]})
def test_column_reassign(copy_on_write, df_factory):
    df1, df2 = df_factory()
    df1["A"] = df1["A"] - 1
    assert df1["A"].tolist() == [-1, 0]
    assert df2["A"].tolist() == [0, 1]
    df2["B"] = df2["B"] + 1
    assert df1["B"].tolist() == [2, 3]
    assert df2["B"].tolist() == [3, 4]


@pytest.mark.parametrize("always_deep", [True, False])
def test_explicit_copy(always_deep):
    # Test that making an explicit copy with deep=True actually makes a deep copy.
    with config_context(NativePandasDeepCopy=always_deep):
        df = pd.DataFrame([[0]])
        # We don't really care about behavior with shallow copy, since modin semantics don't line up
        # perfectly with native pandas.
        df_copy = df.copy(deep=True)
        df.loc[0, 0] = -1
        assert df.loc[0, 0] == -1
        assert df_copy.loc[0, 0] == 0
