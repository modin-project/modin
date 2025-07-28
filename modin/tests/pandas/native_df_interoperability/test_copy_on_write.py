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

# Tests that ensure copy-on-write is properly enforced when the Backend environment variable
# is set to pandas.


import pandas
from pandas import option_context
import pytest

from modin.config import Backend, context as config_context
import modin.pandas as pd


@pytest.fixture(scope="function", autouse=True)
def mutation_cow_test():
    if Backend.get() != "Pandas":
        pytest.skip(reason="test is only meaningful with pandas backend")
    with config_context(Backend="Pandas"), option_context("mode.copy_on_write", False):
        yield


def test_mutate_input_metadata():
    # When constructing a modin frame from a native pandas frame, changes to the native pandas
    # frame should not reflect in the modin frame's data or vice versa.
    input_native_df = pandas.DataFrame({"A": [0, 1], "B": [2, 3]})
    modin_df = pd.DataFrame(input_native_df)
    input_native_df.columns.name = "x"
    assert input_native_df.columns.name == "x"
    assert modin_df.columns.name is None
    modin_df.columns.name = "y"
    assert input_native_df.columns.name == "x"
    assert modin_df.columns.name == "y"


def test_mutate_input_data():
    # When constructing a modin frame from a native pandas frame, changes to the native pandas
    # frame should not reflect in the modin frame's data or vice versa.
    input_native_df = pandas.DataFrame({"A": [0, 1], "B": [2, 3]})
    modin_df = pd.DataFrame(input_native_df)
    input_native_df.loc[0, "A"] = -1
    assert input_native_df.loc[0, "A"] == -1
    # Fails when copy_on_write is False
    assert modin_df.loc[0, "A"] == 0
    modin_df.loc[1, "B"] = 999
    assert input_native_df.loc[1, "B"] == 3
    assert modin_df.loc[1, "B"] == 999


def test_mutate_to_pandas_metadata():
    # After calling to_pandas on a modin frame, changes to the native pandas frame should not reflect
    # in the the modin frame's data or vice versa.
    modin_df = pd.DataFrame({"A": [0, 1], "B": [2, 3]})
    output_native_df = modin_df.modin.to_pandas()
    output_native_df.columns.name = "x"
    assert output_native_df.columns.name == "x"
    assert modin_df.columns.name is None
    modin_df.columns.name = "y"
    assert output_native_df.columns.name == "x"
    assert modin_df.columns.name == "y"


def test_mutate_to_pandas_data():
    # After calling to_pandas on a modin frame, changes to the native pandas frame should not reflect
    # in the the modin frame's data or vice versa.
    modin_df = pd.DataFrame({"A": [0, 1], "B": [2, 3]})
    output_native_df = modin_df.modin.to_pandas()
    output_native_df.loc[0, "A"] = -1
    assert output_native_df.loc[0, "A"] == -1
    # Fails when copy_on_write is False
    assert modin_df.loc[0, "A"] == 0
    modin_df.loc[1, "B"] = 999
    assert output_native_df.loc[1, "B"] == 3
    assert modin_df.loc[1, "B"] == 999
