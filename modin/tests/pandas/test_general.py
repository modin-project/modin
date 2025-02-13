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

import numpy as np
import pandas
import pytest
from numpy.testing import assert_array_equal

import modin.pandas as pd
from modin.pandas.io import to_pandas
from modin.pandas.testing import assert_frame_equal
from modin.tests.test_utils import (
    current_execution_is_native,
    df_or_series_using_native_execution,
    warns_that_defaulting_to_pandas,
    warns_that_defaulting_to_pandas_if,
)
from modin.utils import get_current_execution

from .utils import (
    bool_arg_keys,
    bool_arg_values,
    create_test_dfs,
    df_equals,
    eval_general,
    sort_if_range_partitioning,
    sort_index_for_equal_values,
    test_data_keys,
    test_data_values,
)

pytestmark = pytest.mark.filterwarnings(
    "default:`DataFrame.insert` for empty DataFrame is not currently supported.*:UserWarning"
)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("append_na", [True, False])
@pytest.mark.parametrize("op", ["isna", "isnull", "notna", "notnull"])
def test_isna_isnull_notna_notnull(data, append_na, op):
    pandas_df = pandas.DataFrame(data)
    modin_df = pd.DataFrame(pandas_df)
    if append_na:
        pandas_df["NONE_COL"] = None
        pandas_df["NAN_COL"] = np.nan
        modin_df["NONE_COL"] = None
        modin_df["NAN_COL"] = np.nan

    pandas_result = getattr(pandas, op)(pandas_df)
    modin_result = getattr(pd, op)(modin_df)
    df_equals(modin_result, pandas_result)

    modin_result = getattr(pd, op)(pd.Series([1, np.nan, 2]))
    pandas_result = getattr(pandas, op)(pandas.Series([1, np.nan, 2]))
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
        with warns_that_defaulting_to_pandas_if(
            how == "outer" and not df_or_series_using_native_execution(modin_df)
        ):
            modin_result = pd.merge(modin_df, modin_df2, how=how)
        pandas_result = pandas.merge(pandas_df, pandas_df2, how=how)
        df_equals(modin_result, pandas_result)

        # left_on and right_index
        with warns_that_defaulting_to_pandas_if(
            not df_or_series_using_native_execution(modin_df)
        ):
            modin_result = pd.merge(
                modin_df, modin_df2, how=how, left_on="col1", right_index=True
            )
        pandas_result = pandas.merge(
            pandas_df, pandas_df2, how=how, left_on="col1", right_index=True
        )
        df_equals(modin_result, pandas_result)

        # left_index and right_on
        with warns_that_defaulting_to_pandas_if(
            not df_or_series_using_native_execution(modin_df)
        ):
            modin_result = pd.merge(
                modin_df, modin_df2, how=how, left_index=True, right_on="col1"
            )
        pandas_result = pandas.merge(
            pandas_df, pandas_df2, how=how, left_index=True, right_on="col1"
        )
        df_equals(modin_result, pandas_result)

        # left_on and right_on col1
        with warns_that_defaulting_to_pandas_if(
            how == "outer" and not df_or_series_using_native_execution(modin_df)
        ):
            modin_result = pd.merge(
                modin_df, modin_df2, how=how, left_on="col1", right_on="col1"
            )
        pandas_result = pandas.merge(
            pandas_df, pandas_df2, how=how, left_on="col1", right_on="col1"
        )
        df_equals(modin_result, pandas_result)

        # left_on and right_on col2
        with warns_that_defaulting_to_pandas_if(
            how == "outer" and not df_or_series_using_native_execution(modin_df)
        ):
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

    s = pd.Series(frame_data.get("col1"))
    with pytest.raises(ValueError):
        pd.merge(s, modin_df2)

    with pytest.raises(TypeError):
        pd.merge("Non-valid type", modin_df2)


def test_merge_ordered():
    data_a = {
        "key": list("aceace"),
        "lvalue": [1, 2, 3, 1, 2, 3],
        "group": list("aaabbb"),
    }
    data_b = {"key": list("bcd"), "rvalue": [1, 2, 3]}

    modin_df_a = pd.DataFrame(data_a)
    modin_df_b = pd.DataFrame(data_b)

    with warns_that_defaulting_to_pandas_if(not current_execution_is_native()):
        df = pd.merge_ordered(
            modin_df_a, modin_df_b, fill_method="ffill", left_by="group"
        )
        assert isinstance(df, pd.DataFrame)

    with pytest.raises(TypeError):
        pd.merge_ordered(data_a, data_b, fill_method="ffill", left_by="group")


@pytest.mark.parametrize("right_index", [None, [0] * 5], ids=["default", "non_unique"])
def test_merge_asof(right_index):
    left = pd.DataFrame({"a": [1, 5, 10], "left_val": ["a", "b", "c"]})
    right = pd.DataFrame(
        {"a": [1, 2, 3, 6, 7], "right_val": [1, 2, 3, 6, 7]}, index=right_index
    )

    with warns_that_defaulting_to_pandas():
        df = pd.merge_asof(left, right, on="a")
        assert isinstance(df, pd.DataFrame)

    with warns_that_defaulting_to_pandas():
        df = pd.merge_asof(left, right, on="a", allow_exact_matches=False)
        assert isinstance(df, pd.DataFrame)

    with warns_that_defaulting_to_pandas():
        df = pd.merge_asof(left, right, on="a", direction="forward")
        assert isinstance(df, pd.DataFrame)

    with warns_that_defaulting_to_pandas():
        df = pd.merge_asof(left, right, on="a", direction="nearest")
        assert isinstance(df, pd.DataFrame)

    left = pd.DataFrame({"left_val": ["a", "b", "c"]}, index=[1, 5, 10])
    right = pd.DataFrame({"right_val": [1, 2, 3, 6, 7]}, index=[1, 2, 3, 6, 7])

    with warns_that_defaulting_to_pandas():
        df = pd.merge_asof(left, right, left_index=True, right_index=True)
        assert isinstance(df, pd.DataFrame)

    with pytest.raises(ValueError):
        pd.merge_asof(
            {"left_val": ["a", "b", "c"]},
            {"right_val": [1, 2, 3, 6, 7]},
            left_index=True,
            right_index=True,
        )


def test_merge_asof_on_variations():
    """on=,left_on=,right_on=,right_index=,left_index= options match Pandas."""
    left = {"a": [1, 5, 10], "left_val": ["a", "b", "c"]}
    left_index = [6, 8, 12]
    right = {"a": [1, 2, 3, 6, 7], "right_val": ["d", "e", "f", "g", "h"]}
    right_index = [6, 7, 8, 9, 15]
    pandas_left, pandas_right = (
        pandas.DataFrame(left, index=left_index),
        pandas.DataFrame(right, index=right_index),
    )
    modin_left, modin_right = (
        pd.DataFrame(left, index=left_index),
        pd.DataFrame(right, index=right_index),
    )
    for on_arguments in [
        {"on": "a"},
        {"left_on": "a", "right_on": "a"},
        {"left_on": "a", "right_index": True},
        {"left_index": True, "right_on": "a"},
        {"left_index": True, "right_index": True},
    ]:
        pandas_merged = pandas.merge_asof(pandas_left, pandas_right, **on_arguments)
        with warns_that_defaulting_to_pandas():
            modin_merged = pd.merge_asof(modin_left, modin_right, **on_arguments)
        df_equals(pandas_merged, modin_merged)


def test_merge_asof_suffixes():
    """Suffix variations are handled the same as Pandas."""
    left = {"a": [1, 5, 10]}
    right = {"a": [2, 3, 6]}
    pandas_left, pandas_right = (pandas.DataFrame(left), pandas.DataFrame(right))
    modin_left, modin_right = pd.DataFrame(left), pd.DataFrame(right)
    for suffixes in [("a", "b"), (False, "c"), ("d", False)]:
        pandas_merged = pandas.merge_asof(
            pandas_left,
            pandas_right,
            left_index=True,
            right_index=True,
            suffixes=suffixes,
        )
        with warns_that_defaulting_to_pandas():
            modin_merged = pd.merge_asof(
                modin_left,
                modin_right,
                left_index=True,
                right_index=True,
                suffixes=suffixes,
            )
        df_equals(pandas_merged, modin_merged)

    with pytest.raises(ValueError):
        pandas.merge_asof(
            pandas_left,
            pandas_right,
            left_index=True,
            right_index=True,
            suffixes=(False, False),
        )
    with pytest.raises(ValueError), warns_that_defaulting_to_pandas():
        pd.merge_asof(
            modin_left,
            modin_right,
            left_index=True,
            right_index=True,
            suffixes=(False, False),
        )


def test_merge_asof_bad_arguments():
    left = {"a": [1, 5, 10], "b": [5, 7, 9]}
    right = {"a": [2, 3, 6], "b": [6, 5, 20]}
    pandas_left, pandas_right = (pandas.DataFrame(left), pandas.DataFrame(right))
    modin_left, modin_right = pd.DataFrame(left), pd.DataFrame(right)

    # Can't mix by with left_by/right_by
    with pytest.raises(ValueError):
        pandas.merge_asof(
            pandas_left, pandas_right, on="a", by="b", left_by="can't do with by"
        )
    with pytest.raises(ValueError), warns_that_defaulting_to_pandas():
        pd.merge_asof(
            modin_left, modin_right, on="a", by="b", left_by="can't do with by"
        )
    with pytest.raises(ValueError):
        pandas.merge_asof(
            pandas_left, pandas_right, by="b", on="a", right_by="can't do with by"
        )
    with pytest.raises(ValueError), warns_that_defaulting_to_pandas():
        pd.merge_asof(
            modin_left, modin_right, by="b", on="a", right_by="can't do with by"
        )

    # Can't mix on with left_on/right_on
    with pytest.raises(ValueError):
        pandas.merge_asof(pandas_left, pandas_right, on="a", left_on="can't do with by")
    with pytest.raises(ValueError), warns_that_defaulting_to_pandas():
        pd.merge_asof(modin_left, modin_right, on="a", left_on="can't do with by")
    with pytest.raises(ValueError):
        pandas.merge_asof(
            pandas_left, pandas_right, on="a", right_on="can't do with by"
        )
    with pytest.raises(ValueError), warns_that_defaulting_to_pandas():
        pd.merge_asof(modin_left, modin_right, on="a", right_on="can't do with by")

    # Can't mix left_index with left_on or on, similarly for right.
    with pytest.raises(ValueError), warns_that_defaulting_to_pandas():
        pd.merge_asof(modin_left, modin_right, on="a", right_index=True)
    with pytest.raises(ValueError), warns_that_defaulting_to_pandas():
        pd.merge_asof(
            modin_left, modin_right, left_on="a", right_on="a", right_index=True
        )
    with pytest.raises(ValueError), warns_that_defaulting_to_pandas():
        pd.merge_asof(modin_left, modin_right, on="a", left_index=True)
    with pytest.raises(ValueError), warns_that_defaulting_to_pandas():
        pd.merge_asof(
            modin_left, modin_right, left_on="a", right_on="a", left_index=True
        )

    # Need both left and right
    with pytest.raises(Exception):  # Pandas bug, didn't validate inputs sufficiently
        pandas.merge_asof(pandas_left, pandas_right, left_on="a")
    with pytest.raises(ValueError), warns_that_defaulting_to_pandas():
        pd.merge_asof(modin_left, modin_right, left_on="a")
    with pytest.raises(Exception):  # Pandas bug, didn't validate inputs sufficiently
        pandas.merge_asof(pandas_left, pandas_right, right_on="a")
    with pytest.raises(ValueError), warns_that_defaulting_to_pandas():
        pd.merge_asof(modin_left, modin_right, right_on="a")
    with pytest.raises(ValueError):
        pandas.merge_asof(pandas_left, pandas_right)
    with pytest.raises(ValueError), warns_that_defaulting_to_pandas():
        pd.merge_asof(modin_left, modin_right)


def test_merge_asof_merge_options():
    modin_quotes = pd.DataFrame(
        {
            "time": [
                pd.Timestamp("2016-05-25 13:30:00.023"),
                pd.Timestamp("2016-05-25 13:30:00.023"),
                pd.Timestamp("2016-05-25 13:30:00.030"),
                pd.Timestamp("2016-05-25 13:30:00.041"),
                pd.Timestamp("2016-05-25 13:30:00.048"),
                pd.Timestamp("2016-05-25 13:30:00.049"),
                pd.Timestamp("2016-05-25 13:30:00.072"),
                pd.Timestamp("2016-05-25 13:30:00.075"),
            ],
            "ticker": ["GOOG", "MSFT", "MSFT", "MSFT", "GOOG", "AAPL", "GOOG", "MSFT"],
            "bid": [720.50, 51.95, 51.97, 51.99, 720.50, 97.99, 720.50, 52.01],
            "ask": [720.93, 51.96, 51.98, 52.00, 720.93, 98.01, 720.88, 52.03],
        }
    )
    modin_trades = pd.DataFrame(
        {
            "time": [
                pd.Timestamp("2016-05-25 13:30:00.023"),
                pd.Timestamp("2016-05-25 13:30:00.038"),
                pd.Timestamp("2016-05-25 13:30:00.048"),
                pd.Timestamp("2016-05-25 13:30:00.048"),
                pd.Timestamp("2016-05-25 13:30:00.048"),
            ],
            "ticker2": ["MSFT", "MSFT", "GOOG", "GOOG", "AAPL"],
            "price": [51.95, 51.95, 720.77, 720.92, 98.0],
            "quantity": [75, 155, 100, 100, 100],
        }
    )
    pandas_quotes, pandas_trades = to_pandas(modin_quotes), to_pandas(modin_trades)

    # left_by + right_by
    with warns_that_defaulting_to_pandas():
        modin_result = pd.merge_asof(
            modin_quotes,
            modin_trades,
            on="time",
            left_by="ticker",
            right_by="ticker2",
        )
    df_equals(
        pandas.merge_asof(
            pandas_quotes,
            pandas_trades,
            on="time",
            left_by="ticker",
            right_by="ticker2",
        ),
        modin_result,
    )

    # Just by:
    pandas_trades["ticker"] = pandas_trades["ticker2"]
    modin_trades["ticker"] = modin_trades["ticker2"]
    with warns_that_defaulting_to_pandas():
        modin_result = pd.merge_asof(
            modin_quotes,
            modin_trades,
            on="time",
            by="ticker",
        )
    df_equals(
        pandas.merge_asof(
            pandas_quotes,
            pandas_trades,
            on="time",
            by="ticker",
        ),
        modin_result,
    )

    # Tolerance
    with warns_that_defaulting_to_pandas():
        modin_result = pd.merge_asof(
            modin_quotes,
            modin_trades,
            on="time",
            by="ticker",
            tolerance=pd.Timedelta("2ms"),
        )
    df_equals(
        pandas.merge_asof(
            pandas_quotes,
            pandas_trades,
            on="time",
            by="ticker",
            tolerance=pd.Timedelta("2ms"),
        ),
        modin_result,
    )

    # Direction
    with warns_that_defaulting_to_pandas():
        modin_result = pd.merge_asof(
            modin_quotes,
            modin_trades,
            on="time",
            by="ticker",
            direction="forward",
        )
    df_equals(
        pandas.merge_asof(
            pandas_quotes,
            pandas_trades,
            on="time",
            by="ticker",
            direction="forward",
        ),
        modin_result,
    )

    # Allow exact matches
    with warns_that_defaulting_to_pandas():
        modin_result = pd.merge_asof(
            modin_quotes,
            modin_trades,
            on="time",
            by="ticker",
            tolerance=pd.Timedelta("10ms"),
            allow_exact_matches=False,
        )
    df_equals(
        pandas.merge_asof(
            pandas_quotes,
            pandas_trades,
            on="time",
            by="ticker",
            tolerance=pd.Timedelta("10ms"),
            allow_exact_matches=False,
        ),
        modin_result,
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

    df = pd.pivot(test_df, index="foo", columns="bar", values="baz")
    assert isinstance(df, pd.DataFrame)

    with pytest.raises(ValueError):
        pd.pivot(test_df["bar"], index="foo", columns="bar", values="baz")

    if not (get_current_execution() == "BaseOnPython" or current_execution_is_native()):
        # FIXME: Failed for some reason on 'BaseOnPython' and 'NativeOnNative'
        # https://github.com/modin-project/modin/issues/6240
        df_equals(
            pd.pivot(test_df, columns="bar"),
            pandas.pivot(test_df._to_pandas(), columns="bar"),
        )

        df_equals(
            pd.pivot(test_df, index="foo", columns="bar"),
            pandas.pivot(test_df._to_pandas(), index="foo", columns="bar"),
        )


def test_pivot_values_is_none():
    test_df = pd.DataFrame(
        {
            "foo": ["one", "one", "one", "two", "two", "two"],
            "bar": ["A", "B", "C", "A", "B", "C"],
            "baz": [1, 2, 3, 4, 5, 6],
            "zoo": ["x", "y", "z", "q", "w", "t"],
        }
    )
    df = pd.pivot(test_df, index="foo", columns="bar")
    assert isinstance(df, pd.DataFrame)


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

    df = pd.pivot_table(
        test_df, values="D", index=["A", "B"], columns=["C"], aggfunc=np.sum
    )
    assert isinstance(df, pd.DataFrame)

    with pytest.raises(ValueError):
        pd.pivot_table(
            test_df["C"], values="D", index=["A", "B"], columns=["C"], aggfunc=np.sum
        )


def test_unique():
    comparator = lambda *args: sort_if_range_partitioning(  # noqa: E731
        *args, comparator=assert_array_equal
    )

    modin_result = pd.unique([2, 1, 3, 3])
    pandas_result = pandas.unique([2, 1, 3, 3])
    comparator(modin_result, pandas_result)
    assert modin_result.shape == pandas_result.shape

    modin_result = pd.unique(pd.Series([2] + [1] * 5))
    pandas_result = pandas.unique(pandas.Series([2] + [1] * 5))
    comparator(modin_result, pandas_result)
    assert modin_result.shape == pandas_result.shape

    modin_result = pd.unique(
        pd.Series([pd.Timestamp("20160101"), pd.Timestamp("20160101")])
    )
    pandas_result = pandas.unique(
        pandas.Series([pandas.Timestamp("20160101"), pandas.Timestamp("20160101")])
    )
    comparator(modin_result, pandas_result)
    assert modin_result.shape == pandas_result.shape

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
    comparator(modin_result, pandas_result)
    assert modin_result.shape == pandas_result.shape

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
    comparator(modin_result, pandas_result)
    assert modin_result.shape == pandas_result.shape

    modin_result = pd.unique(pd.Series(pd.Categorical(list("baabc"))))
    pandas_result = pandas.unique(pandas.Series(pandas.Categorical(list("baabc"))))
    comparator(modin_result, pandas_result)
    assert modin_result.shape == pandas_result.shape


@pytest.mark.parametrize("normalize, bins, dropna", [(True, 3, False)])
def test_value_counts(normalize, bins, dropna):
    # We sort indices for Modin and pandas result because of issue #1650
    values = np.array([3, 1, 2, 3, 4, np.nan])
    modin_result = sort_index_for_equal_values(
        pd.value_counts(values, normalize=normalize, ascending=False), False
    )
    pandas_result = sort_index_for_equal_values(
        pandas.value_counts(values, normalize=normalize, ascending=False), False
    )
    df_equals(modin_result, pandas_result)

    with warns_that_defaulting_to_pandas():
        modin_result = sort_index_for_equal_values(
            pd.value_counts(values, bins=bins, ascending=False), False
        )
    pandas_result = sort_index_for_equal_values(
        pandas.value_counts(values, bins=bins, ascending=False), False
    )
    df_equals(modin_result, pandas_result)

    modin_result = sort_index_for_equal_values(
        pd.value_counts(values, dropna=dropna, ascending=True), True
    )
    pandas_result = sort_index_for_equal_values(
        pandas.value_counts(values, dropna=dropna, ascending=True), True
    )
    df_equals(modin_result, pandas_result)


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


def test_to_datetime_inplace_side_effect():
    # See GH#3063
    times = list(range(1617993360, 1618193360))
    values = list(range(215441, 415441))
    modin_df = pd.DataFrame({"time": times, "value": values})
    pandas_df = pandas.DataFrame({"time": times, "value": values})
    df_equals(
        pd.to_datetime(modin_df["time"], unit="s"),
        pandas.to_datetime(pandas_df["time"], unit="s"),
    )


@pytest.mark.parametrize(
    "data, errors, downcast",
    [
        (["1.0", "2", -3], "raise", None),
        (["1.0", "2", -3], "raise", "float"),
        (["1.0", "2", -3], "raise", "signed"),
        (["apple", "1.0", "2", -3], "ignore", None),
        (["apple", "1.0", "2", -3], "coerce", None),
    ],
)
def test_to_numeric(data, errors, downcast):
    modin_series = pd.Series(data)
    pandas_series = pandas.Series(data)
    modin_result = pd.to_numeric(modin_series, errors=errors, downcast=downcast)
    pandas_result = pandas.to_numeric(pandas_series, errors=errors, downcast=downcast)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("retbins", bool_arg_values, ids=bool_arg_keys)
def test_qcut(retbins):
    # test case from https://github.com/modin-project/modin/issues/5610
    pandas_series = pandas.Series(range(10))
    modin_series = pd.Series(range(10))
    pandas_result = pandas.qcut(pandas_series, 4, retbins=retbins)
    # NOTE that qcut() defaults to pandas at the API layer.
    with warns_that_defaulting_to_pandas():
        modin_result = pd.qcut(modin_series, 4, retbins=retbins)
    if retbins:
        df_equals(modin_result[0], pandas_result[0])
        df_equals(modin_result[0].cat.categories, pandas_result[0].cat.categories)
        assert_array_equal(modin_result[1], pandas_result[1])
    else:
        df_equals(modin_result, pandas_result)
        df_equals(modin_result.cat.categories, pandas_result.cat.categories)

    # test case for fallback to pandas, taken from pandas docs
    pandas_result = pandas.qcut(range(5), 4)
    modin_result = pd.qcut(range(5), 4)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "bins, labels",
    [
        pytest.param(
            [-int(1e18), -1000, 0, 1000, 2000, int(1e18)],
            [
                "-inf_to_-1000",
                "-1000_to_0",
                "0_to_1000",
                "1000_to_2000",
                "2000_to_inf",
            ],
            id="bin_list_spanning_entire_range_with_custom_labels",
        ),
        pytest.param(
            [-int(1e18), -1000, 0, 1000, 2000, int(1e18)],
            None,
            id="bin_list_spanning_entire_range_with_default_labels",
        ),
        pytest.param(
            [-1000, 0, 1000, 2000], None, id="bin_list_not_spanning_entire_range"
        ),
        pytest.param(
            10,
            [f"custom_label{i}" for i in range(9)],
            id="int_bin_10_with_custom_labels",
        ),
        pytest.param(1, None, id="int_bin_1_with_default_labels"),
        pytest.param(-1, None, id="int_bin_-1_with_default_labels"),
        pytest.param(111, None, id="int_bin_111_with_default_labels"),
    ],
)
@pytest.mark.parametrize("retbins", bool_arg_values, ids=bool_arg_keys)
def test_cut(retbins, bins, labels):
    # Would use `eval_general` here, but `eval_general` expects the operation
    # to be supported by Modin, and so errors out when we give the defaulting
    # to pandas UserWarning. We could get around this by using
    # @pytest.mark.filterwarnings("ignore"), but then `eval_general` fails because
    # sometimes the return type of pd.cut is an np.ndarray, and `eval_general` does
    # not know how to handle that.
    try:
        pd_result = pandas.cut(
            pandas.Series(range(1000)), retbins=retbins, bins=bins, labels=labels
        )
    except Exception as pd_e:
        with pytest.raises(Exception) as md_e:
            with warns_that_defaulting_to_pandas_if(not current_execution_is_native()):
                md_result = pd.cut(
                    pd.Series(range(1000)), retbins=retbins, bins=bins, labels=labels
                )
        assert isinstance(
            md_e.value, type(pd_e)
        ), f"Got Modin Exception type {type(md_e.value)}, but pandas Exception type {type(pd_e)} was expected"
    else:
        with warns_that_defaulting_to_pandas_if(not current_execution_is_native()):
            md_result = pd.cut(
                pd.Series(range(1000)), retbins=retbins, bins=bins, labels=labels
            )
        if not isinstance(pd_result, tuple):
            df_equals(md_result, pd_result)
        else:
            assert isinstance(
                md_result, tuple
            ), "Modin returned single value, but pandas returned tuple of values"
            for pd_res, md_res in zip(pd_result, md_result):
                if isinstance(pd_res, pandas.Series):
                    df_equals(pd_res, md_res)
                else:
                    np.testing.assert_array_equal(pd_res, md_res)


def test_cut_fallback():
    # Test case for falling back to pandas for cut.
    pandas_result = pandas.cut(range(5), 4)
    # note that we default to pandas at the API layer here, so we warn
    # regardless of whether we are on native execution.
    with warns_that_defaulting_to_pandas():
        modin_result = pd.cut(range(5), 4)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "data", [test_data_values[0], []], ids=["test_data_values[0]", "[]"]
)
def test_to_pandas_indices(data):
    md_df = pd.DataFrame(data)
    index = pandas.MultiIndex.from_tuples(
        [(i, i * 2) for i in np.arange(len(md_df) + 1)], names=["A", "B"]
    ).drop(0)
    columns = pandas.MultiIndex.from_tuples(
        [(i, i * 2) for i in np.arange(len(md_df.columns) + 1)], names=["A", "B"]
    ).drop(0)

    md_df.index = index
    md_df.columns = columns

    pd_df = md_df._to_pandas()

    for axis in [0, 1]:
        assert md_df.axes[axis].equals(
            pd_df.axes[axis]
        ), f"Indices at axis {axis} are different!"
        assert not hasattr(md_df.axes[axis], "equal_levels") or md_df.axes[
            axis
        ].equal_levels(
            pd_df.axes[axis]
        ), f"Levels of indices at axis {axis} are different!"


def test_to_pandas_read_only_issue():
    df = pd.DataFrame(
        [
            [np.nan, 2, np.nan, 0],
            [3, 4, np.nan, 1],
            [np.nan, np.nan, np.nan, np.nan],
            [np.nan, 3, np.nan, 4],
        ],
        columns=list("ABCD"),
    )
    pdf = df._to_pandas()
    # there shouldn't be `ValueError: putmask: output array is read-only`
    pdf.fillna(0, inplace=True)


def test_to_numpy_read_only_issue():
    df = pd.DataFrame(
        [
            [np.nan, 2, np.nan, 0],
            [3, 4, np.nan, 1],
            [np.nan, np.nan, np.nan, np.nan],
            [np.nan, 3, np.nan, 4],
        ],
        columns=list("ABCD"),
    )
    arr = df.to_numpy()
    # there shouldn't be `ValueError: putmask: output array is read-only`
    np.putmask(arr, np.isnan(arr), 0)


def test_create_categorical_dataframe_with_duplicate_column_name():
    # This tests for https://github.com/modin-project/modin/issues/4312
    pd_df = pandas.DataFrame(
        {
            "a": pandas.Categorical([1, 2]),
            "b": [4, 5],
            "c": pandas.Categorical([7, 8]),
        }
    )
    pd_df.columns = ["a", "b", "a"]
    md_df = pd.DataFrame(pd_df)
    # Use assert_frame_equal instead of the common modin util df_equals because
    # we should check dtypes of the new categorical with check_dtype=True.
    # TODO(https://github.com/modin-project/modin/issues/3804): Make
    # df_equals set check_dtype=True and use df_equals instead.
    assert_frame_equal(
        md_df._to_pandas(),
        pd_df,
        check_dtype=True,
        check_index_type=True,
        check_column_type=True,
        check_names=True,
        check_categorical=True,
    )


@pytest.mark.skipif(
    get_current_execution() != "BaseOnPython",
    reason="This test make sense only on BaseOnPython execution.",
)
@pytest.mark.parametrize(
    "func, regex",
    [
        (lambda df: df.mean(), r"DataFrame\.mean"),
        (lambda df: df + df, r"DataFrame\.add"),
        (lambda df: df.index, r"DataFrame\.get_axis\(0\)"),
        (
            lambda df: df.drop(columns="col1").squeeze().repeat(2),
            r"Series\.repeat",
        ),
        (lambda df: df.groupby("col1").prod(), r"GroupBy\.prod"),
        (lambda df: df.rolling(1).count(), r"Rolling\.count"),
    ],
)
def test_default_to_pandas_warning_message(func, regex):
    data = {"col1": [1, 2, 3], "col2": [4, 5, 6]}
    df = pd.DataFrame(data)

    with pytest.warns(UserWarning, match=regex):
        func(df)


def test_empty_dataframe():
    df = pd.DataFrame(columns=["a", "b"])
    # NOTE that we default to pandas at the API layer.
    with warns_that_defaulting_to_pandas():
        df[(df.a == 1) & (df.b == 2)]


def test_empty_series():
    s = pd.Series([])
    pd.to_numeric(s)


@pytest.mark.parametrize(
    "arg",
    [[1, 2], ["a"], 1, "a"],
    ids=["list_of_ints", "list_of_invalid_strings", "scalar", "invalid_scalar"],
)
def test_to_timedelta(arg, request):
    # This test case comes from
    # https://github.com/modin-project/modin/issues/4966
    expected_exception = None
    if request.node.callspec.id == "list_of_invalid_strings":
        expected_exception = ValueError("Could not convert 'a' to NumPy timedelta")
    elif request.node.callspec.id == "invalid_scalar":
        expected_exception = ValueError("unit abbreviation w/o a number")
    eval_general(
        pd,
        pandas,
        lambda lib: lib.to_timedelta(arg),
        expected_exception=expected_exception,
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_series_to_timedelta(data):
    def make_frame(lib):
        series = lib.Series(
            next(iter(data.values())) if isinstance(data, dict) else data
        )
        return lib.to_timedelta(series).to_frame(name="timedelta")

    eval_general(pd, pandas, make_frame)


@pytest.mark.parametrize(
    "key",
    [["col0"], "col0", "col1"],
    ids=["valid_list_of_string", "valid_string", "invalid_string"],
)
def test_get(key):
    modin_df, pandas_df = create_test_dfs({"col0": [0, 1]})
    eval_general(modin_df, pandas_df, lambda df: df.get(key))


def test_df_immutability():
    """
    Verify that modifications of the source data doesn't propagate to Modin's DataFrame objects.
    """
    src_data = pandas.DataFrame({"a": [1]})

    md_df = pd.DataFrame(src_data)
    src_data.iloc[0, 0] = 100

    assert md_df._to_pandas().iloc[0, 0] == 1
