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

import pandas
import pytest
import modin.pandas as pd
import numpy as np
from numpy.testing import assert_array_equal

from .utils import test_data_values, test_data_keys, df_equals


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_isna(data):
    pandas_df = pandas.DataFrame(data)
    modin_df = pd.DataFrame(data)

    pandas_result = pandas.isna(pandas_df)
    modin_result = pd.isna(modin_df)
    df_equals(modin_result, pandas_result)

    modin_result = pd.isna(pd.Series([1, np.nan, 2]))
    pandas_result = pandas.isna(pandas.Series([1, np.nan, 2]))
    df_equals(modin_result, pandas_result)

    assert pd.isna(np.nan) == pandas.isna(np.nan)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_isnull(data):
    pandas_df = pandas.DataFrame(data)
    modin_df = pd.DataFrame(data)

    pandas_result = pandas.isnull(pandas_df)
    modin_result = pd.isnull(modin_df)
    df_equals(modin_result, pandas_result)

    modin_result = pd.isnull(pd.Series([1, np.nan, 2]))
    pandas_result = pandas.isnull(pandas.Series([1, np.nan, 2]))
    df_equals(modin_result, pandas_result)

    assert pd.isna(np.nan) == pandas.isna(np.nan)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_notna(data):
    pandas_df = pandas.DataFrame(data)
    modin_df = pd.DataFrame(data)

    pandas_result = pandas.notna(pandas_df)
    modin_result = pd.notna(modin_df)
    df_equals(modin_result, pandas_result)

    modin_result = pd.notna(pd.Series([1, np.nan, 2]))
    pandas_result = pandas.notna(pandas.Series([1, np.nan, 2]))
    df_equals(modin_result, pandas_result)

    assert pd.isna(np.nan) == pandas.isna(np.nan)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_notnull(data):
    pandas_df = pandas.DataFrame(data)
    modin_df = pd.DataFrame(data)

    pandas_result = pandas.notnull(pandas_df)
    modin_result = pd.notnull(modin_df)
    df_equals(modin_result, pandas_result)

    modin_result = pd.notnull(pd.Series([1, np.nan, 2]))
    pandas_result = pandas.notnull(pandas.Series([1, np.nan, 2]))
    df_equals(modin_result, pandas_result)

    assert pd.isna(np.nan) == pandas.isna(np.nan)


def test_merge():
    frame_data = {
        "col1": [0, 1, 2, 3],
        "col2": [4, 5, 6, 7],
        "col3": [8, 9, 0, 1],
        "col4": [2, 4, 5, 6],
    }

    modin_df = pd.DataFrame(frame_data)
    pandas_df = pandas.DataFrame(frame_data)

    frame_data2 = {"col1": [0, 1, 2], "col2": [1, 5, 6]}
    modin_df2 = pd.DataFrame(frame_data2)
    pandas_df2 = pandas.DataFrame(frame_data2)

    join_types = ["outer", "inner"]
    for how in join_types:
        # Defaults
        modin_result = pd.merge(modin_df, modin_df2, how=how)
        pandas_result = pandas.merge(pandas_df, pandas_df2, how=how)
        df_equals(modin_result, pandas_result)

        # left_on and right_index
        modin_result = pd.merge(
            modin_df, modin_df2, how=how, left_on="col1", right_index=True
        )
        pandas_result = pandas.merge(
            pandas_df, pandas_df2, how=how, left_on="col1", right_index=True
        )
        df_equals(modin_result, pandas_result)

        # left_index and right_on
        modin_result = pd.merge(
            modin_df, modin_df2, how=how, left_index=True, right_on="col1"
        )
        pandas_result = pandas.merge(
            pandas_df, pandas_df2, how=how, left_index=True, right_on="col1"
        )
        df_equals(modin_result, pandas_result)

        # left_on and right_on col1
        modin_result = pd.merge(
            modin_df, modin_df2, how=how, left_on="col1", right_on="col1"
        )
        pandas_result = pandas.merge(
            pandas_df, pandas_df2, how=how, left_on="col1", right_on="col1"
        )
        df_equals(modin_result, pandas_result)

        # left_on and right_on col2
        modin_result = pd.merge(
            modin_df, modin_df2, how=how, left_on="col2", right_on="col2"
        )
        pandas_result = pandas.merge(
            pandas_df, pandas_df2, how=how, left_on="col2", right_on="col2"
        )
        df_equals(modin_result, pandas_result)

        # left_index and right_index
        modin_result = pd.merge(
            modin_df, modin_df2, how=how, left_index=True, right_index=True
        )
        pandas_result = pandas.merge(
            pandas_df, pandas_df2, how=how, left_index=True, right_index=True
        )
        df_equals(modin_result, pandas_result)

        with pytest.raises(ValueError):
            pd.merge(modin_df["col1"], modin_df2)


def test_merge_ordered():
    data_a = {
        "key": list("aceace"),
        "lvalue": [1, 2, 3, 1, 2, 3],
        "group": list("aaabbb"),
    }
    data_b = {"key": list("bcd"), "rvalue": [1, 2, 3]}

    modin_df_a = pd.DataFrame(data_a)
    modin_df_b = pd.DataFrame(data_b)

    with pytest.warns(UserWarning):
        df = pd.merge_ordered(
            modin_df_a, modin_df_b, fill_method="ffill", left_by="group"
        )
        assert isinstance(df, pd.DataFrame)

    with pytest.raises(ValueError):
        pd.merge_ordered(data_a, data_b, fill_method="ffill", left_by="group")


def test_merge_asof():
    left = pd.DataFrame({"a": [1, 5, 10], "left_val": ["a", "b", "c"]})
    right = pd.DataFrame({"a": [1, 2, 3, 6, 7], "right_val": [1, 2, 3, 6, 7]})

    with pytest.warns(UserWarning):
        df = pd.merge_asof(left, right, on="a")
        assert isinstance(df, pd.DataFrame)

    with pytest.warns(UserWarning):
        df = pd.merge_asof(left, right, on="a", allow_exact_matches=False)
        assert isinstance(df, pd.DataFrame)

    with pytest.warns(UserWarning):
        df = pd.merge_asof(left, right, on="a", direction="forward")
        assert isinstance(df, pd.DataFrame)

    with pytest.warns(UserWarning):
        df = pd.merge_asof(left, right, on="a", direction="nearest")
        assert isinstance(df, pd.DataFrame)

    left = pd.DataFrame({"left_val": ["a", "b", "c"]}, index=[1, 5, 10])
    right = pd.DataFrame({"right_val": [1, 2, 3, 6, 7]}, index=[1, 2, 3, 6, 7])

    with pytest.warns(UserWarning):
        df = pd.merge_asof(left, right, left_index=True, right_index=True)
        assert isinstance(df, pd.DataFrame)

    with pytest.raises(ValueError):
        pd.merge_asof(
            {"left_val": ["a", "b", "c"]},
            {"right_val": [1, 2, 3, 6, 7]},
            left_index=True,
            right_index=True,
        )


def test_pivot():
    test_df = pd.DataFrame(
        {
            "foo": ["one", "one", "one", "two", "two", "two"],
            "bar": ["A", "B", "C", "A", "B", "C"],
            "baz": [1, 2, 3, 4, 5, 6],
            "zoo": ["x", "y", "z", "q", "w", "t"],
        }
    )
    with pytest.warns(UserWarning):
        df = pd.pivot(test_df, index="foo", columns="bar", values="baz")
        assert isinstance(df, pd.DataFrame)

    with pytest.raises(ValueError):
        pd.pivot(test_df["bar"], index="foo", columns="bar", values="baz")


def test_pivot_table():
    test_df = pd.DataFrame(
        {
            "A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
            "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
            "C": [
                "small",
                "large",
                "large",
                "small",
                "small",
                "large",
                "small",
                "small",
                "large",
            ],
            "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
            "E": [2, 4, 5, 5, 6, 6, 8, 9, 9],
        }
    )
    with pytest.warns(UserWarning):
        df = pd.pivot_table(
            test_df, values="D", index=["A", "B"], columns=["C"], aggfunc=np.sum
        )
        assert isinstance(df, pd.DataFrame)

    with pytest.raises(ValueError):
        pd.pivot_table(
            test_df["C"], values="D", index=["A", "B"], columns=["C"], aggfunc=np.sum
        )


def test_unique():
    modin_result = pd.unique([2, 1, 3, 3])
    pandas_result = pandas.unique([2, 1, 3, 3])
    assert_array_equal(modin_result, pandas_result)

    modin_result = pd.unique(pd.Series([2] + [1] * 5))
    pandas_result = pandas.unique(pandas.Series([2] + [1] * 5))
    assert_array_equal(modin_result, pandas_result)

    modin_result = pd.unique(
        pd.Series([pd.Timestamp("20160101"), pd.Timestamp("20160101")])
    )
    pandas_result = pandas.unique(
        pandas.Series([pandas.Timestamp("20160101"), pandas.Timestamp("20160101")])
    )
    assert_array_equal(modin_result, pandas_result)

    modin_result = pd.unique(
        pd.Series(
            [
                pd.Timestamp("20160101", tz="US/Eastern"),
                pd.Timestamp("20160101", tz="US/Eastern"),
            ]
        )
    )
    pandas_result = pandas.unique(
        pandas.Series(
            [
                pandas.Timestamp("20160101", tz="US/Eastern"),
                pandas.Timestamp("20160101", tz="US/Eastern"),
            ]
        )
    )
    assert_array_equal(modin_result, pandas_result)

    modin_result = pd.unique(
        pd.Index(
            [
                pd.Timestamp("20160101", tz="US/Eastern"),
                pd.Timestamp("20160101", tz="US/Eastern"),
            ]
        )
    )
    pandas_result = pandas.unique(
        pandas.Index(
            [
                pandas.Timestamp("20160101", tz="US/Eastern"),
                pandas.Timestamp("20160101", tz="US/Eastern"),
            ]
        )
    )
    assert_array_equal(modin_result, pandas_result)

    modin_result = pd.unique(pd.Series(pd.Categorical(list("baabc"))))
    pandas_result = pandas.unique(pandas.Series(pandas.Categorical(list("baabc"))))
    assert_array_equal(modin_result, pandas_result)


def test_to_datetime():
    # DataFrame input for to_datetime
    modin_df = pd.DataFrame({"year": [2015, 2016], "month": [2, 3], "day": [4, 5]})
    pandas_df = pandas.DataFrame({"year": [2015, 2016], "month": [2, 3], "day": [4, 5]})
    df_equals(pd.to_datetime(modin_df), pandas.to_datetime(pandas_df))

    # Series input for to_datetime
    modin_s = pd.Series(["3/11/2000", "3/12/2000", "3/13/2000"] * 1000)
    pandas_s = pandas.Series(["3/11/2000", "3/12/2000", "3/13/2000"] * 1000)
    df_equals(pd.to_datetime(modin_s), pandas.to_datetime(pandas_s))

    # Other inputs for to_datetime
    value = 1490195805
    assert pd.to_datetime(value, unit="s") == pandas.to_datetime(value, unit="s")
    value = 1490195805433502912
    assert pd.to_datetime(value, unit="ns") == pandas.to_datetime(value, unit="ns")
    value = [1, 2, 3]
    assert pd.to_datetime(value, unit="D", origin=pd.Timestamp("2000-01-01")).equals(
        pandas.to_datetime(value, unit="D", origin=pandas.Timestamp("2000-01-01"))
    )
