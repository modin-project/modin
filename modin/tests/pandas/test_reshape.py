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

import contextlib

import numpy as np
import pandas
import pytest

import modin.pandas as pd
from modin.config import StorageFormat
from modin.tests.test_utils import (
    df_or_series_using_native_execution,
    warns_that_defaulting_to_pandas,
    warns_that_defaulting_to_pandas_if,
)

from .utils import df_equals, test_data_values


def test_get_dummies():
    s = pd.Series(list("abca"))
    with warns_that_defaulting_to_pandas():
        pd.get_dummies(s)

    s1 = ["a", "b", np.nan]
    with warns_that_defaulting_to_pandas():
        pd.get_dummies(s1)

    with warns_that_defaulting_to_pandas():
        pd.get_dummies(s1, dummy_na=True)

    data = {"A": ["a", "b", "a"], "B": ["b", "a", "c"], "C": [1, 2, 3]}
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    modin_result = pd.get_dummies(modin_df, prefix=["col1", "col2"])
    pandas_result = pandas.get_dummies(pandas_df, prefix=["col1", "col2"])
    df_equals(modin_result, pandas_result)
    assert modin_result._to_pandas().columns.equals(pandas_result.columns)
    assert modin_result.shape == pandas_result.shape

    modin_result = pd.get_dummies(pd.DataFrame(pd.Series(list("abcdeabac"))))
    pandas_result = pandas.get_dummies(
        pandas.DataFrame(pandas.Series(list("abcdeabac")))
    )
    df_equals(modin_result, pandas_result)
    assert modin_result._to_pandas().columns.equals(pandas_result.columns)
    assert modin_result.shape == pandas_result.shape

    with pytest.raises(NotImplementedError):
        pd.get_dummies(modin_df, prefix=["col1", "col2"], sparse=True)

    with warns_that_defaulting_to_pandas():
        pd.get_dummies(pd.Series(list("abcaa")))

    with warns_that_defaulting_to_pandas():
        pd.get_dummies(pd.Series(list("abcaa")), drop_first=True)

    with warns_that_defaulting_to_pandas():
        pd.get_dummies(pd.Series(list("abc")), dtype=float)

    with warns_that_defaulting_to_pandas():
        pd.get_dummies(1)

    # test from #5184
    pandas_df = pandas.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": ["7", "8", "9"]})
    modin_df = pd.DataFrame(pandas_df)
    pandas_result = pandas.get_dummies(pandas_df, columns=["a", "b"])
    modin_result = pd.get_dummies(modin_df, columns=["a", "b"])
    df_equals(modin_result, pandas_result)


def test_melt():
    data = test_data_values[0]

    with (
        pytest.warns(
            UserWarning, match=r"`melt` implementation has mismatches with pandas"
        )
        if StorageFormat.get() == "Pandas"
        else contextlib.nullcontext()
    ):
        pd.melt(pd.DataFrame(data))


def test_crosstab():
    a = np.array(
        ["foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar", "foo", "foo", "foo"],
        dtype=object,
    )
    b = np.array(
        ["one", "one", "one", "two", "one", "one", "one", "two", "two", "two", "one"],
        dtype=object,
    )
    c = np.array(
        [
            "dull",
            "dull",
            "shiny",
            "dull",
            "dull",
            "shiny",
            "shiny",
            "dull",
            "shiny",
            "shiny",
            "shiny",
        ],
        dtype=object,
    )

    with warns_that_defaulting_to_pandas():
        df = pd.crosstab(a, [b, c], rownames=["a"], colnames=["b", "c"])
        assert isinstance(df, pd.DataFrame)

    foo = pd.Categorical(["a", "b"], categories=["a", "b", "c"])
    bar = pd.Categorical(["d", "e"], categories=["d", "e", "f"])

    with warns_that_defaulting_to_pandas():
        df = pd.crosstab(foo, bar)
        assert isinstance(df, pd.DataFrame)

    with warns_that_defaulting_to_pandas():
        df = pd.crosstab(foo, bar, dropna=False)
        assert isinstance(df, pd.DataFrame)


def test_lreshape():
    data = pd.DataFrame(
        {
            "hr1": [514, 573],
            "hr2": [545, 526],
            "team": ["Red Sox", "Yankees"],
            "year1": [2007, 2008],
            "year2": [2008, 2008],
        }
    )

    with warns_that_defaulting_to_pandas():
        df = pd.lreshape(data, {"year": ["year1", "year2"], "hr": ["hr1", "hr2"]})
        assert isinstance(df, pd.DataFrame)

    with pytest.raises(ValueError):
        pd.lreshape(data.to_numpy(), {"year": ["year1", "year2"], "hr": ["hr1", "hr2"]})


def test_wide_to_long():
    data = pd.DataFrame(
        {
            "hr1": [514, 573],
            "hr2": [545, 526],
            "team": ["Red Sox", "Yankees"],
            "year1": [2007, 2008],
            "year2": [2008, 2008],
        }
    )

    with warns_that_defaulting_to_pandas_if(
        not df_or_series_using_native_execution(data)
    ):
        df = pd.wide_to_long(data, ["hr", "year"], "team", "index")
        assert isinstance(df, pd.DataFrame)

    with pytest.raises(ValueError):
        pd.wide_to_long(data.to_numpy(), ["hr", "year"], "team", "index")
