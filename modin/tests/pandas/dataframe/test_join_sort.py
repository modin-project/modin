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

import warnings

import matplotlib
import numpy as np
import pandas
import pytest

import modin.pandas as pd
from modin.config import Engine, NativeDataframeMode, NPartitions, StorageFormat
from modin.pandas.io import to_pandas
from modin.tests.pandas.utils import (
    arg_keys,
    axis_keys,
    axis_values,
    bool_arg_keys,
    bool_arg_values,
    create_test_dfs,
    default_to_pandas_ignore_string,
    df_equals,
    eval_general,
    generate_multiindex,
    random_state,
    rotate_decimal_digits_or_symbols,
    test_data,
    test_data_keys,
    test_data_values,
)
from modin.tests.test_utils import warns_that_defaulting_to_pandas

NPartitions.put(4)

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")

# Our configuration in pytest.ini requires that we explicitly catch all
# instances of defaulting to pandas, but some test modules, like this one,
# have too many such instances.
pytestmark = pytest.mark.filterwarnings(default_to_pandas_ignore_string)

# Initialize env for storage format detection in @pytest.mark.*
pd.DataFrame()


def df_equals_and_sort(df1, df2):
    """Sort dataframe's rows and run ``df_equals()`` for them."""
    df1 = df1.sort_values(by=df1.columns.tolist(), ignore_index=True)
    df2 = df2.sort_values(by=df2.columns.tolist(), ignore_index=True)
    df_equals(df1, df2)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_combine(data):
    pandas_df = pandas.DataFrame(data)
    modin_df = pd.DataFrame(data)

    modin_df.combine(modin_df + 1, lambda s1, s2: s1 if s1.count() < s2.count() else s2)
    pandas_df.combine(
        pandas_df + 1, lambda s1, s2: s1 if s1.count() < s2.count() else s2
    )


@pytest.mark.parametrize(
    "test_data, test_data2",
    [
        (
            np.random.randint(0, 100, size=(64, 64)),
            np.random.randint(0, 100, size=(128, 64)),
        ),
        (
            np.random.randint(0, 100, size=(128, 64)),
            np.random.randint(0, 100, size=(64, 64)),
        ),
        (
            np.random.randint(0, 100, size=(64, 64)),
            np.random.randint(0, 100, size=(64, 128)),
        ),
        (
            np.random.randint(0, 100, size=(64, 128)),
            np.random.randint(0, 100, size=(64, 64)),
        ),
    ],
)
def test_join(test_data, test_data2):
    modin_df = pd.DataFrame(
        test_data,
        columns=["col{}".format(i) for i in range(test_data.shape[1])],
        index=pd.Index([i for i in range(1, test_data.shape[0] + 1)], name="key"),
    )
    pandas_df = pandas.DataFrame(
        test_data,
        columns=["col{}".format(i) for i in range(test_data.shape[1])],
        index=pandas.Index([i for i in range(1, test_data.shape[0] + 1)], name="key"),
    )
    modin_df2 = pd.DataFrame(
        test_data2,
        columns=["col{}".format(i) for i in range(test_data2.shape[1])],
        index=pd.Index([i for i in range(1, test_data2.shape[0] + 1)], name="key"),
    )
    pandas_df2 = pandas.DataFrame(
        test_data2,
        columns=["col{}".format(i) for i in range(test_data2.shape[1])],
        index=pandas.Index([i for i in range(1, test_data2.shape[0] + 1)], name="key"),
    )

    hows = ["inner", "left", "right", "outer"]
    ons = ["col33", "col34"]
    sorts = [False, True]
    assert len(ons) == len(sorts), "the loop below is designed for this condition"
    for i in range(len(hows)):
        for j in range(len(ons)):
            modin_result = modin_df.join(
                modin_df2,
                how=hows[i],
                on=ons[j],
                sort=sorts[j],
                lsuffix="_caller",
                rsuffix="_other",
            )
            pandas_result = pandas_df.join(
                pandas_df2,
                how=hows[i],
                on=ons[j],
                sort=sorts[j],
                lsuffix="_caller",
                rsuffix="_other",
            )
            if sorts[j]:
                # sorting in `join` is implemented through range partitioning technique
                # therefore the order of the rows after it does not match the pandas,
                # so additional sorting is needed in order to get the same result as for pandas
                df_equals_and_sort(modin_result, pandas_result)
            else:
                df_equals(modin_result, pandas_result)

    frame_data = {
        "col1": [0, 1, 2, 3],
        "col2": [4, 5, 6, 7],
        "col3": [8, 9, 0, 1],
        "col4": [2, 4, 5, 6],
    }

    modin_df = pd.DataFrame(frame_data)
    pandas_df = pandas.DataFrame(frame_data)

    frame_data2 = {"col5": [0], "col6": [1]}
    modin_df2 = pd.DataFrame(frame_data2)
    pandas_df2 = pandas.DataFrame(frame_data2)

    join_types = ["left", "right", "outer", "inner"]
    for how in join_types:
        modin_join = modin_df.join(modin_df2, how=how)
        pandas_join = pandas_df.join(pandas_df2, how=how)
        df_equals(modin_join, pandas_join)

    frame_data3 = {"col7": [1, 2, 3, 5, 6, 7, 8]}

    modin_df3 = pd.DataFrame(frame_data3)
    pandas_df3 = pandas.DataFrame(frame_data3)

    join_types = ["left", "outer", "inner"]
    for how in join_types:
        modin_join = modin_df.join([modin_df2, modin_df3], how=how)
        pandas_join = pandas_df.join([pandas_df2, pandas_df3], how=how)
        df_equals(modin_join, pandas_join)


@pytest.mark.parametrize("how", ["left", "inner", "right"])
def test_join_empty(how):
    data = np.random.randint(0, 100, size=(64, 64))
    eval_general(
        *create_test_dfs(data),
        lambda df: df.join(df.iloc[:0], on=1, how=how, lsuffix="_caller"),
    )


def test_join_cross_6786():
    data = [[7, 8, 9], [10, 11, 12]]
    modin_df, pandas_df = create_test_dfs(data, columns=["x", "y", "z"])

    modin_join = modin_df.join(
        modin_df[["x"]].set_axis(["p", "q"], axis=0), how="cross", lsuffix="p"
    )
    pandas_join = pandas_df.join(
        pandas_df[["x"]].set_axis(["p", "q"], axis=0), how="cross", lsuffix="p"
    )
    df_equals(modin_join, pandas_join)


def test_join_5203():
    data = np.ones([2, 4])
    kwargs = {"columns": ["a", "b", "c", "d"]}
    modin_dfs, pandas_dfs = [None] * 3, [None] * 3
    for idx in range(len(modin_dfs)):
        modin_dfs[idx], pandas_dfs[idx] = create_test_dfs(data, **kwargs)

    for dfs in (modin_dfs, pandas_dfs):
        with pytest.raises(
            ValueError,
            match="Joining multiple DataFrames only supported for joining on index",
        ):
            dfs[0].join([dfs[1], dfs[2]], how="inner", on="a")


def test_join_6602():
    abbreviations = pd.Series(
        ["Major League Baseball", "National Basketball Association"],
        index=["MLB", "NBA"],
    )
    teams = pd.DataFrame(
        {
            "name": ["Mariners", "Lakers"] * 50,
            "league_abbreviation": ["MLB", "NBA"] * 50,
        }
    )

    with warnings.catch_warnings():
        # check that join doesn't show UserWarning
        warnings.filterwarnings(
            "error", "Distributing <class 'dict'> object", category=UserWarning
        )
        teams.set_index("league_abbreviation").join(abbreviations.rename("league_name"))


@pytest.mark.parametrize(
    "test_data, test_data2",
    [
        (
            np.random.randint(0, 100, size=(64, 64)),
            np.random.randint(0, 100, size=(128, 64)),
        ),
        (
            np.random.randint(0, 100, size=(128, 64)),
            np.random.randint(0, 100, size=(64, 64)),
        ),
        (
            np.random.randint(0, 100, size=(64, 64)),
            np.random.randint(0, 100, size=(64, 128)),
        ),
        (
            np.random.randint(0, 100, size=(64, 128)),
            np.random.randint(0, 100, size=(64, 64)),
        ),
    ],
)
def test_merge(test_data, test_data2):
    modin_df = pd.DataFrame(
        test_data,
        columns=["col{}".format(i) for i in range(test_data.shape[1])],
        index=pd.Index([i for i in range(1, test_data.shape[0] + 1)], name="key"),
    )
    pandas_df = pandas.DataFrame(
        test_data,
        columns=["col{}".format(i) for i in range(test_data.shape[1])],
        index=pandas.Index([i for i in range(1, test_data.shape[0] + 1)], name="key"),
    )
    modin_df2 = pd.DataFrame(
        test_data2,
        columns=["col{}".format(i) for i in range(test_data2.shape[1])],
        index=pd.Index([i for i in range(1, test_data2.shape[0] + 1)], name="key"),
    )
    pandas_df2 = pandas.DataFrame(
        test_data2,
        columns=["col{}".format(i) for i in range(test_data2.shape[1])],
        index=pandas.Index([i for i in range(1, test_data2.shape[0] + 1)], name="key"),
    )

    hows = ["left", "inner", "right"]
    ons = ["col33", ["col33", "col34"]]
    sorts = [False, True]
    assert len(ons) == len(sorts), "the loop below is designed for this condition"
    for i in range(len(hows)):
        for j in range(len(ons)):
            modin_result = modin_df.merge(
                modin_df2, how=hows[i], on=ons[j], sort=sorts[j]
            )
            pandas_result = pandas_df.merge(
                pandas_df2, how=hows[i], on=ons[j], sort=sorts[j]
            )
            # FIXME: https://github.com/modin-project/modin/issues/2246
            df_equals_and_sort(modin_result, pandas_result)

            modin_result = modin_df.merge(
                modin_df2,
                how=hows[i],
                left_on="key",
                right_on="key",
                sort=sorts[j],
            )
            pandas_result = pandas_df.merge(
                pandas_df2,
                how=hows[i],
                left_on="key",
                right_on="key",
                sort=sorts[j],
            )
            # FIXME: https://github.com/modin-project/modin/issues/2246
            df_equals_and_sort(modin_result, pandas_result)

    # Test for issue #1771
    modin_df = pd.DataFrame({"name": np.arange(40)})
    modin_df2 = pd.DataFrame({"name": [39], "position": [0]})
    pandas_df = pandas.DataFrame({"name": np.arange(40)})
    pandas_df2 = pandas.DataFrame({"name": [39], "position": [0]})
    modin_result = modin_df.merge(modin_df2, on="name", how="inner")
    pandas_result = pandas_df.merge(pandas_df2, on="name", how="inner")
    # FIXME: https://github.com/modin-project/modin/issues/2246
    df_equals_and_sort(modin_result, pandas_result)

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
        modin_result = modin_df.merge(modin_df2, how=how)
        pandas_result = pandas_df.merge(pandas_df2, how=how)
        # FIXME: https://github.com/modin-project/modin/issues/2246
        df_equals_and_sort(modin_result, pandas_result)

        # left_on and right_index
        modin_result = modin_df.merge(
            modin_df2, how=how, left_on="col1", right_index=True
        )
        pandas_result = pandas_df.merge(
            pandas_df2, how=how, left_on="col1", right_index=True
        )
        # FIXME: https://github.com/modin-project/modin/issues/2246
        df_equals_and_sort(modin_result, pandas_result)

        # left_index and right_on
        modin_result = modin_df.merge(
            modin_df2, how=how, left_index=True, right_on="col1"
        )
        pandas_result = pandas_df.merge(
            pandas_df2, how=how, left_index=True, right_on="col1"
        )
        # FIXME: https://github.com/modin-project/modin/issues/2246
        df_equals_and_sort(modin_result, pandas_result)

        # left_on and right_on col1
        modin_result = modin_df.merge(
            modin_df2, how=how, left_on="col1", right_on="col1"
        )
        pandas_result = pandas_df.merge(
            pandas_df2, how=how, left_on="col1", right_on="col1"
        )
        # FIXME: https://github.com/modin-project/modin/issues/2246
        df_equals_and_sort(modin_result, pandas_result)

        # left_on and right_on col2
        modin_result = modin_df.merge(
            modin_df2, how=how, left_on="col2", right_on="col2"
        )
        pandas_result = pandas_df.merge(
            pandas_df2, how=how, left_on="col2", right_on="col2"
        )
        # FIXME: https://github.com/modin-project/modin/issues/2246
        df_equals_and_sort(modin_result, pandas_result)

        # left_index and right_index
        modin_result = modin_df.merge(
            modin_df2, how=how, left_index=True, right_index=True
        )
        pandas_result = pandas_df.merge(
            pandas_df2, how=how, left_index=True, right_index=True
        )
        # FIXME: https://github.com/modin-project/modin/issues/2246
        df_equals_and_sort(modin_result, pandas_result)

    # Cannot merge a Series without a name
    ps = pandas.Series(frame_data2.get("col1"))
    ms = pd.Series(frame_data2.get("col1"))
    eval_general(
        modin_df,
        pandas_df,
        lambda df: df.merge(ms if isinstance(df, pd.DataFrame) else ps),
        # FIXME: https://github.com/modin-project/modin/issues/2246
        comparator=df_equals_and_sort,
        expected_exception=ValueError("Cannot merge a Series without a name"),
    )

    # merge a Series with a name
    ps = pandas.Series(frame_data2.get("col1"), name="col1")
    ms = pd.Series(frame_data2.get("col1"), name="col1")
    eval_general(
        modin_df,
        pandas_df,
        lambda df: df.merge(ms if isinstance(df, pd.DataFrame) else ps),
        # FIXME: https://github.com/modin-project/modin/issues/2246
        comparator=df_equals_and_sort,
    )

    with pytest.raises(TypeError):
        modin_df.merge("Non-valid type")


@pytest.mark.parametrize("how", ["left", "inner", "right"])
def test_merge_empty(how):
    data = np.random.randint(0, 100, size=(64, 64))
    eval_general(*create_test_dfs(data), lambda df: df.merge(df.iloc[:0], how=how))


def test_merge_with_mi_columns():
    modin_df1, pandas_df1 = create_test_dfs(
        {
            ("col0", "a"): [1, 2, 3, 4],
            ("col0", "b"): [2, 3, 4, 5],
            ("col1", "a"): [3, 4, 5, 6],
        }
    )

    modin_df2, pandas_df2 = create_test_dfs(
        {
            ("col0", "a"): [1, 2, 3, 4],
            ("col0", "c"): [2, 3, 4, 5],
            ("col1", "a"): [3, 4, 5, 6],
        }
    )

    eval_general(
        (modin_df1, modin_df2),
        (pandas_df1, pandas_df2),
        lambda dfs: dfs[0].merge(dfs[1], on=[("col0", "a")]),
    )


@pytest.mark.parametrize("has_index_cache", [True, False])
def test_merge_on_index(has_index_cache):
    modin_df1, pandas_df1 = create_test_dfs(
        {
            "idx_key1": [1, 2, 3, 4],
            "idx_key2": [2, 3, 4, 5],
            "idx_key3": [3, 4, 5, 6],
            "data_col1": [10, 2, 3, 4],
            "col_key1": [3, 4, 5, 6],
            "col_key2": [3, 4, 5, 6],
        }
    )

    modin_df1 = modin_df1.set_index(["idx_key1", "idx_key2"])
    pandas_df1 = pandas_df1.set_index(["idx_key1", "idx_key2"])

    modin_df2, pandas_df2 = create_test_dfs(
        {
            "idx_key1": [4, 3, 2, 1],
            "idx_key2": [5, 4, 3, 2],
            "idx_key3": [6, 5, 4, 3],
            "data_col2": [10, 2, 3, 4],
            "col_key1": [6, 5, 4, 3],
            "col_key2": [6, 5, 4, 3],
        }
    )

    modin_df2 = modin_df2.set_index(["idx_key2", "idx_key3"])
    pandas_df2 = pandas_df2.set_index(["idx_key2", "idx_key3"])

    def setup_cache():
        if has_index_cache:
            modin_df1.index  # triggering index materialization
            modin_df2.index
            assert modin_df1._query_compiler.frame_has_index_cache
            assert modin_df2._query_compiler.frame_has_index_cache
        else:
            # Propagate deferred indices to partitions
            # The change in index is not automatically handled by Modin. See #3941.
            modin_df1.index = modin_df1.index
            modin_df1._to_pandas()
            modin_df1._query_compiler.set_frame_index_cache(None)
            modin_df2.index = modin_df2.index
            modin_df2._to_pandas()
            modin_df2._query_compiler.set_frame_index_cache(None)

    for on in (
        ["col_key1", "idx_key1"],
        ["col_key1", "idx_key2"],
        ["col_key1", "idx_key3"],
        ["idx_key1"],
        ["idx_key2"],
        ["idx_key3"],
    ):
        setup_cache()
        eval_general(
            (modin_df1, modin_df2),
            (pandas_df1, pandas_df2),
            lambda dfs: dfs[0].merge(dfs[1], on=on),
        )

    for left_on, right_on in (
        (["idx_key1"], ["col_key1"]),
        (["col_key1"], ["idx_key3"]),
        (["idx_key1"], ["idx_key3"]),
        (["idx_key2"], ["idx_key2"]),
        (["col_key1", "idx_key2"], ["col_key2", "idx_key2"]),
    ):
        setup_cache()
        eval_general(
            (modin_df1, modin_df2),
            (pandas_df1, pandas_df2),
            lambda dfs: dfs[0].merge(dfs[1], left_on=left_on, right_on=right_on),
        )


@pytest.mark.parametrize(
    "left_index", [[], ["key"], ["key", "b"], ["key", "b", "c"], ["b"], ["b", "c"]]
)
@pytest.mark.parametrize(
    "right_index", [[], ["key"], ["key", "e"], ["key", "e", "f"], ["e"], ["e", "f"]]
)
def test_merge_on_single_index(left_index, right_index):
    """
    Test ``.merge()`` method when merging on a single column, that is located in an index level of one of the frames.
    """
    modin_df1, pandas_df1 = create_test_dfs(
        {"b": [3, 4, 4, 5], "key": [1, 1, 2, 2], "c": [2, 3, 2, 2], "d": [2, 1, 3, 1]}
    )
    if len(left_index):
        modin_df1 = modin_df1.set_index(left_index)
        pandas_df1 = pandas_df1.set_index(left_index)

    modin_df2, pandas_df2 = create_test_dfs(
        {"e": [3, 4, 4, 5], "f": [2, 3, 2, 2], "key": [1, 1, 2, 2], "h": [2, 1, 3, 1]}
    )
    if len(right_index):
        modin_df2 = modin_df2.set_index(right_index)
        pandas_df2 = pandas_df2.set_index(right_index)
    eval_general(
        (modin_df1, modin_df2),
        (pandas_df1, pandas_df2),
        lambda dfs: dfs[0].merge(dfs[1], on="key"),
    )


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("ascending", [False, True])
@pytest.mark.parametrize("na_position", ["first", "last"], ids=["first", "last"])
def test_sort_index(axis, ascending, na_position):
    data = test_data["float_nan_data"]
    modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)

    # Change index value so sorting will actually make a difference
    if axis == 0:
        length = len(modin_df.index)
        for df in [modin_df, pandas_df]:
            df.index = [(i - length / 2) % length for i in range(length)]

    dfs = [modin_df, pandas_df]
    # Add NaNs to sorted index
    for idx in range(len(dfs)):
        sort_index = dfs[idx].axes[axis]
        dfs[idx] = dfs[idx].set_axis(
            [np.nan if i % 2 == 0 else sort_index[i] for i in range(len(sort_index))],
            axis=axis,
            copy=False,
        )
    modin_df, pandas_df = dfs

    eval_general(
        modin_df,
        pandas_df,
        lambda df: df.sort_index(
            axis=axis, ascending=ascending, na_position=na_position
        ),
    )


@pytest.mark.parametrize("axis", ["rows", "columns"])
def test_sort_index_inplace(axis):
    data = test_data["int_data"]
    modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)

    for df in [modin_df, pandas_df]:
        df.sort_index(axis=axis, inplace=True)
    df_equals(modin_df, pandas_df)


@pytest.mark.parametrize(
    "sort_remaining", bool_arg_values, ids=arg_keys("sort_remaining", bool_arg_keys)
)
def test_sort_multiindex(sort_remaining):
    data = test_data["int_data"]
    modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)

    for index in ["index", "columns"]:
        new_index = generate_multiindex(len(getattr(modin_df, index)))
        for df in [modin_df, pandas_df]:
            setattr(df, index, new_index)

    for kwargs in [{"level": 0}, {"axis": 0}, {"axis": 1}]:
        with warns_that_defaulting_to_pandas():
            df_equals(
                modin_df.sort_index(sort_remaining=sort_remaining, **kwargs),
                pandas_df.sort_index(sort_remaining=sort_remaining, **kwargs),
            )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize(
    "by",
    [
        pytest.param(
            "first",
            marks=pytest.mark.exclude_by_default,
        ),
        pytest.param(
            "first,last",
            marks=pytest.mark.exclude_by_default,
        ),
        "first,last,middle",
    ],
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "ascending",
    [False, True] + ["list_first_True", "list_first_False"],
    ids=arg_keys(
        "ascending", ["False", "True"] + ["list_first_True", "list_first_False"]
    ),
)
@pytest.mark.parametrize(
    "inplace", bool_arg_values, ids=arg_keys("inplace", bool_arg_keys)
)
@pytest.mark.parametrize(
    "kind",
    [
        pytest.param(
            "mergesort",
            marks=pytest.mark.exclude_by_default,
        ),
        "quicksort",
        pytest.param(
            "heapsort",
            marks=pytest.mark.exclude_by_default,
        ),
    ],
)
@pytest.mark.parametrize("na_position", ["first", "last"], ids=["first", "last"])
@pytest.mark.parametrize(
    "ignore_index",
    bool_arg_values,
    ids=arg_keys("ignore_index", bool_arg_keys),
)
@pytest.mark.parametrize("key", [None, rotate_decimal_digits_or_symbols])
def test_sort_values(
    data, by, axis, ascending, inplace, kind, na_position, ignore_index, key
):
    if ascending is None:
        pytest.skip("None is not a valid value for ascending.")
    if (axis == 1 or axis == "columns") and ignore_index:
        pytest.skip("Pandas bug #39426 which is fixed in Pandas 1.3")

    if ascending is None and key is not None:
        pytest.skip("Pandas bug #41318")

    if "multiindex" in by:
        index = generate_multiindex(len(data[list(data.keys())[0]]), nlevels=2)
        columns = generate_multiindex(len(data.keys()), nlevels=2)
        data = {columns[ind]: data[key] for ind, key in enumerate(data)}
    else:
        index = None
        columns = None

    modin_df = pd.DataFrame(data, index=index, columns=columns)
    pandas_df = pandas.DataFrame(data, index=index, columns=columns)

    index = modin_df.index if axis == 1 or axis == "columns" else modin_df.columns

    # Parse "by" spec
    by_list = []
    for b in by.split(","):
        if b == "first":
            by_list.append(index[0])
        elif b == "last":
            by_list.append(index[-1])
        elif b == "middle":
            by_list.append(index[len(index) // 2])
        elif b.startswith("multiindex_level"):
            by_list.append(index.names[int(b[len("multiindex_level") :])])
        else:
            raise Exception('Unknown "by" specifier:' + b)

    # Create "ascending" list
    if ascending in ["list_first_True", "list_first_False"]:
        start = 0 if ascending == "list_first_False" else 1
        ascending = [i & 1 > 0 for i in range(start, len(by_list) + start)]

    eval_general(
        modin_df,
        pandas_df,
        lambda df: df.sort_values(
            by_list,
            axis=axis,
            ascending=ascending,
            inplace=inplace,
            kind=kind,
            na_position=na_position,
            ignore_index=ignore_index,
            key=key,
        ),
        __inplace__=inplace,
    )


def test_sort_values_descending_with_only_two_bins():
    # test case from https://github.com/modin-project/modin/issues/5781
    part1 = pd.DataFrame({"a": [1, 2, 3, 4]})
    part2 = pd.DataFrame({"a": [5, 6, 7, 8]})

    modin_df = pd.concat([part1, part2])
    pandas_df = modin_df._to_pandas()

    if StorageFormat.get() == "Pandas" and NativeDataframeMode.get() == "Default":
        assert modin_df._query_compiler._modin_frame._partitions.shape == (2, 1)

    eval_general(
        modin_df, pandas_df, lambda df: df.sort_values(by="a", ascending=False)
    )


@pytest.mark.parametrize("ignore_index", [True, False])
def test_sort_values_preserve_index_names(ignore_index):
    modin_df, pandas_df = create_test_dfs(
        np.random.choice(128, 128, replace=False).reshape((128, 1))
    )

    pandas_df.index.names, pandas_df.columns.names = ["custom_name"], ["custom_name"]
    modin_df.index.names, modin_df.columns.names = ["custom_name"], ["custom_name"]
    # workaround for #1618 to actually propagate index change
    modin_df.index = modin_df.index
    modin_df.columns = modin_df.columns

    def comparator(df1, df2):
        assert df1.index.names == df2.index.names
        assert df1.columns.names == df2.columns.names
        df_equals(df1, df2)

    eval_general(
        modin_df,
        pandas_df,
        lambda df: df.sort_values(df.columns[0], ignore_index=ignore_index),
        comparator=comparator,
    )


@pytest.mark.parametrize("ascending", [True, False])
def test_sort_values_with_one_partition(ascending):
    # Test case from https://github.com/modin-project/modin/issues/5859
    modin_df, pandas_df = create_test_dfs(
        np.array([["hello", "goodbye"], ["hello", "Hello"]])
    )

    if StorageFormat.get() == "Pandas" and NativeDataframeMode.get() == "Default":
        assert modin_df._query_compiler._modin_frame._partitions.shape == (1, 1)

    eval_general(
        modin_df, pandas_df, lambda df: df.sort_values(by=1, ascending=ascending)
    )


def test_sort_overpartitioned_df():
    # First we test when the final df will have only 1 row and column partition.
    data = [[4, 5, 6], [1, 2, 3]]
    modin_df = pd.concat([pd.DataFrame(row).T for row in data]).reset_index(drop=True)
    pandas_df = pandas.DataFrame(data)

    eval_general(modin_df, pandas_df, lambda df: df.sort_values(by=0))

    # Next we test when the final df will only have 1 row, but starts with multiple column
    # partitions.
    data = [list(range(100)), list(range(100, 200))]
    modin_df = pd.concat([pd.DataFrame(row).T for row in data]).reset_index(drop=True)
    pandas_df = pandas.DataFrame(data)

    eval_general(modin_df, pandas_df, lambda df: df.sort_values(by=0))

    # Next we test when the final df will have multiple row partitions.
    data = np.random.choice(650, 650, replace=False).reshape((65, 10))
    modin_df = pd.concat([pd.DataFrame(row).T for row in data]).reset_index(drop=True)
    pandas_df = pandas.DataFrame(data)

    eval_general(modin_df, pandas_df, lambda df: df.sort_values(by=0))

    old_nptns = NPartitions.get()
    NPartitions.put(24)
    try:
        # Next we test when there's only one row per partition.
        data = np.random.choice(650, 650, replace=False).reshape((65, 10))
        modin_df = pd.concat([pd.DataFrame(row).T for row in data]).reset_index(
            drop=True
        )
        pandas_df = pandas.DataFrame(data)

        eval_general(modin_df, pandas_df, lambda df: df.sort_values(by=0))

        # And again, when there's more than one column partition.
        data = np.random.choice(6500, 6500, replace=False).reshape((65, 100))
        modin_df = pd.concat([pd.DataFrame(row).T for row in data]).reset_index(
            drop=True
        )
        pandas_df = pandas.DataFrame(data)

        eval_general(modin_df, pandas_df, lambda df: df.sort_values(by=0))

        # Additionally, we should test when we have a number of partitions
        # that doesn't divide cleanly into our desired number of partitions.
        # In this case, we start with 17 partitions, and want 2.
        NPartitions.put(21)
        data = np.random.choice(6500, 6500, replace=False).reshape((65, 100))
        modin_df = pd.concat([pd.DataFrame(row).T for row in data]).reset_index(
            drop=True
        )
        pandas_df = pandas.DataFrame(data)

        eval_general(modin_df, pandas_df, lambda df: df.sort_values(by=0))

    finally:
        NPartitions.put(old_nptns)


def test_sort_values_with_duplicates():
    modin_df = pd.DataFrame({"col": [2, 1, 1]}, index=[1, 1, 0])
    pandas_df = pandas.DataFrame({"col": [2, 1, 1]}, index=[1, 1, 0])

    key = modin_df.columns[0]
    modin_result = modin_df.sort_values(key, inplace=False)
    pandas_result = pandas_df.sort_values(key, inplace=False)
    df_equals(modin_result, pandas_result)

    modin_df.sort_values(key, inplace=True)
    pandas_df.sort_values(key, inplace=True)
    df_equals(modin_df, pandas_df)


def test_sort_values_with_string_index():
    modin_df = pd.DataFrame({"col": [25, 17, 1]}, index=["ccc", "bbb", "aaa"])
    pandas_df = pandas.DataFrame({"col": [25, 17, 1]}, index=["ccc", "bbb", "aaa"])

    key = modin_df.columns[0]
    modin_result = modin_df.sort_values(key, inplace=False)
    pandas_result = pandas_df.sort_values(key, inplace=False)
    df_equals(modin_result, pandas_result)

    modin_df.sort_values(key, inplace=True)
    pandas_df.sort_values(key, inplace=True)
    df_equals(modin_df, pandas_df)


@pytest.mark.skipif(
    StorageFormat.get() != "Pandas",
    reason="We only need to test this case where sort does not default to pandas.",
)
@pytest.mark.parametrize("ascending", [True, False], ids=["True", "False"])
@pytest.mark.parametrize("na_position", ["first", "last"], ids=["first", "last"])
def test_sort_values_with_only_one_non_na_row_in_partition(ascending, na_position):
    pandas_df = pandas.DataFrame(
        np.random.rand(1000, 100), columns=[f"col {i}" for i in range(100)]
    )
    # Need to ensure that one of the partitions has all NA values except for one row
    pandas_df.iloc[340:] = np.nan
    pandas_df.iloc[-1] = -4.0
    modin_df = pd.DataFrame(pandas_df)
    eval_general(
        modin_df,
        pandas_df,
        lambda df: df.sort_values(
            "col 3", ascending=ascending, na_position=na_position
        ),
    )


@pytest.mark.skipif(
    Engine.get() not in ("Ray", "Unidist", "Dask")
    or NativeDataframeMode.get() == "Pandas",
    reason="We only need to test this case where sort does not default to pandas.",
)
def test_sort_values_with_sort_key_on_partition_boundary():
    modin_df = pd.DataFrame(
        np.random.rand(1000, 100), columns=[f"col {i}" for i in range(100)]
    )
    sort_key = modin_df.columns[modin_df._query_compiler._modin_frame.column_widths[0]]
    eval_general(modin_df, modin_df._to_pandas(), lambda df: df.sort_values(sort_key))


def test_where():
    columns = list("abcdefghij")

    frame_data = random_state.randn(100, 10)
    modin_df, pandas_df = create_test_dfs(frame_data, columns=columns)
    pandas_cond_df = pandas_df % 5 < 2
    modin_cond_df = modin_df % 5 < 2

    pandas_result = pandas_df.where(pandas_cond_df, -pandas_df)
    modin_result = modin_df.where(modin_cond_df, -modin_df)
    assert all((to_pandas(modin_result) == pandas_result).all())

    # test case when other is Series
    other_data = random_state.randn(len(pandas_df))
    modin_other, pandas_other = pd.Series(other_data), pandas.Series(other_data)
    pandas_result = pandas_df.where(pandas_cond_df, pandas_other, axis=0)
    modin_result = modin_df.where(modin_cond_df, modin_other, axis=0)
    df_equals(modin_result, pandas_result)

    # Test that we choose the right values to replace when `other` == `True`
    # everywhere.
    other_data = np.full(shape=pandas_df.shape, fill_value=True)
    modin_other, pandas_other = create_test_dfs(other_data, columns=columns)
    pandas_result = pandas_df.where(pandas_cond_df, pandas_other)
    modin_result = modin_df.where(modin_cond_df, modin_other)
    df_equals(modin_result, pandas_result)

    other = pandas_df.loc[3]
    pandas_result = pandas_df.where(pandas_cond_df, other, axis=1)
    modin_result = modin_df.where(modin_cond_df, other, axis=1)
    assert all((to_pandas(modin_result) == pandas_result).all())

    other = pandas_df["e"]
    pandas_result = pandas_df.where(pandas_cond_df, other, axis=0)
    modin_result = modin_df.where(modin_cond_df, other, axis=0)
    assert all((to_pandas(modin_result) == pandas_result).all())

    pandas_result = pandas_df.where(pandas_df < 2, True)
    modin_result = modin_df.where(modin_df < 2, True)
    assert all((to_pandas(modin_result) == pandas_result).all())


def test_where_different_axis_order():
    # Test `where` when `cond`, `df`, and `other` each have columns and index
    # in different orders.
    data = test_data["float_nan_data"]
    pandas_df = pandas.DataFrame(data)
    pandas_cond_df = pandas_df % 5 < 2
    pandas_cond_df = pandas_cond_df.reindex(
        columns=pandas_df.columns[::-1], index=pandas_df.index[::-1]
    )
    pandas_other_df = -pandas_df
    pandas_other_df = pandas_other_df.reindex(
        columns=pandas_df.columns[-1:].append(pandas_df.columns[:-1]),
        index=pandas_df.index[-1:].append(pandas_df.index[:-1]),
    )

    modin_df = pd.DataFrame(pandas_df)
    modin_cond_df = pd.DataFrame(pandas_cond_df)
    modin_other_df = pd.DataFrame(pandas_other_df)

    pandas_result = pandas_df.where(pandas_cond_df, pandas_other_df)
    modin_result = modin_df.where(modin_cond_df, modin_other_df)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("align_axis", ["index", "columns"])
@pytest.mark.parametrize("keep_shape", [False, True])
@pytest.mark.parametrize("keep_equal", [False, True])
def test_compare(align_axis, keep_shape, keep_equal):
    kwargs = {
        "align_axis": align_axis,
        "keep_shape": keep_shape,
        "keep_equal": keep_equal,
    }
    frame_data1 = random_state.randn(100, 10)
    frame_data2 = random_state.randn(100, 10)
    pandas_df = pandas.DataFrame(frame_data1, columns=list("abcdefghij"))
    pandas_df2 = pandas.DataFrame(frame_data2, columns=list("abcdefghij"))
    modin_df = pd.DataFrame(frame_data1, columns=list("abcdefghij"))
    modin_df2 = pd.DataFrame(frame_data2, columns=list("abcdefghij"))

    modin_result = modin_df.compare(modin_df2, **kwargs)
    pandas_result = pandas_df.compare(pandas_df2, **kwargs)
    assert to_pandas(modin_result).equals(pandas_result)

    modin_result = modin_df2.compare(modin_df, **kwargs)
    pandas_result = pandas_df2.compare(pandas_df, **kwargs)
    assert to_pandas(modin_result).equals(pandas_result)

    series_data1 = ["a", "b", "c", "d", "e"]
    series_data2 = ["a", "a", "c", "b", "e"]
    pandas_series1 = pandas.Series(series_data1)
    pandas_series2 = pandas.Series(series_data2)
    modin_series1 = pd.Series(series_data1)
    modin_series2 = pd.Series(series_data2)

    modin_result = modin_series1.compare(modin_series2, **kwargs)
    pandas_result = pandas_series1.compare(pandas_series2, **kwargs)
    assert to_pandas(modin_result).equals(pandas_result)

    modin_result = modin_series2.compare(modin_series1, **kwargs)
    pandas_result = pandas_series2.compare(pandas_series1, **kwargs)
    assert to_pandas(modin_result).equals(pandas_result)
