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

import pytest
import numpy as np
import pandas
import matplotlib
import modin.pandas as pd
from modin.pandas.utils import to_pandas

from modin.pandas.test.utils import (
    random_state,
    df_equals,
    arg_keys,
    name_contains,
    test_data_values,
    test_data_keys,
    numeric_dfs,
    axis_keys,
    axis_values,
    bool_arg_keys,
    bool_arg_values,
)

pd.DEFAULT_NPARTITIONS = 4

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_combine(data):
    pandas_df = pandas.DataFrame(data)
    modin_df = pd.DataFrame(data)

    modin_df.combine(modin_df + 1, lambda s1, s2: s1 if s1.count() < s2.count() else s2)
    pandas_df.combine(
        pandas_df + 1, lambda s1, s2: s1 if s1.count() < s2.count() else s2
    )


def test_join():
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


@pytest.mark.parametrize(
    "test_data, test_data2",
    [
        (
            np.random.uniform(0, 100, size=(2 ** 6, 2 ** 6)),
            np.random.uniform(0, 100, size=(2 ** 7, 2 ** 6)),
        ),
        (
            np.random.uniform(0, 100, size=(2 ** 7, 2 ** 6)),
            np.random.uniform(0, 100, size=(2 ** 6, 2 ** 6)),
        ),
        (
            np.random.uniform(0, 100, size=(2 ** 6, 2 ** 6)),
            np.random.uniform(0, 100, size=(2 ** 6, 2 ** 7)),
        ),
        (
            np.random.uniform(0, 100, size=(2 ** 6, 2 ** 7)),
            np.random.uniform(0, 100, size=(2 ** 6, 2 ** 6)),
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

    hows = ["left", "inner"]
    ons = ["col33", ["col33", "col34"]]
    sorts = [False, True]
    for i in range(2):
        for j in range(2):
            modin_result = modin_df.merge(
                modin_df2, how=hows[i], on=ons[j], sort=sorts[j]
            )
            pandas_result = pandas_df.merge(
                pandas_df2, how=hows[i], on=ons[j], sort=sorts[j]
            )
            df_equals(modin_result, pandas_result)

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
            df_equals(modin_result, pandas_result)

    # Test for issue #1771
    modin_df = pd.DataFrame({"name": np.arange(40)})
    modin_df2 = pd.DataFrame({"name": [39], "position": [0]})
    pandas_df = pandas.DataFrame({"name": np.arange(40)})
    pandas_df2 = pandas.DataFrame({"name": [39], "position": [0]})
    modin_result = modin_df.merge(modin_df2, on="name", how="inner")
    pandas_result = pandas_df.merge(pandas_df2, on="name", how="inner")
    df_equals(modin_result, pandas_result)

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
        df_equals(modin_result, pandas_result)

        # left_on and right_index
        modin_result = modin_df.merge(
            modin_df2, how=how, left_on="col1", right_index=True
        )
        pandas_result = pandas_df.merge(
            pandas_df2, how=how, left_on="col1", right_index=True
        )
        df_equals(modin_result, pandas_result)

        # left_index and right_on
        modin_result = modin_df.merge(
            modin_df2, how=how, left_index=True, right_on="col1"
        )
        pandas_result = pandas_df.merge(
            pandas_df2, how=how, left_index=True, right_on="col1"
        )
        df_equals(modin_result, pandas_result)

        # left_on and right_on col1
        modin_result = modin_df.merge(
            modin_df2, how=how, left_on="col1", right_on="col1"
        )
        pandas_result = pandas_df.merge(
            pandas_df2, how=how, left_on="col1", right_on="col1"
        )
        df_equals(modin_result, pandas_result)

        # left_on and right_on col2
        modin_result = modin_df.merge(
            modin_df2, how=how, left_on="col2", right_on="col2"
        )
        pandas_result = pandas_df.merge(
            pandas_df2, how=how, left_on="col2", right_on="col2"
        )
        df_equals(modin_result, pandas_result)

        # left_index and right_index
        modin_result = modin_df.merge(
            modin_df2, how=how, left_index=True, right_index=True
        )
        pandas_result = pandas_df.merge(
            pandas_df2, how=how, left_index=True, right_index=True
        )
        df_equals(modin_result, pandas_result)

    # Named Series promoted to DF
    s = pd.Series(frame_data2.get("col1"))
    with pytest.raises(ValueError):
        modin_df.merge(s)

    s = pd.Series(frame_data2.get("col1"), name="col1")
    df_equals(modin_df.merge(s), modin_df.merge(modin_df2[["col1"]]))

    with pytest.raises(TypeError):
        modin_df.merge("Non-valid type")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "ascending", bool_arg_values, ids=arg_keys("ascending", bool_arg_keys)
)
@pytest.mark.parametrize("na_position", ["first", "last"], ids=["first", "last"])
@pytest.mark.parametrize(
    "sort_remaining", bool_arg_values, ids=arg_keys("sort_remaining", bool_arg_keys)
)
def test_sort_index(data, axis, ascending, na_position, sort_remaining):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    # Change index value so sorting will actually make a difference
    if axis == "rows" or axis == 0:
        length = len(modin_df.index)
        modin_df.index = [(i - length / 2) % length for i in range(length)]
        pandas_df.index = [(i - length / 2) % length for i in range(length)]
    # Add NaNs to sorted index
    if axis == "rows" or axis == 0:
        length = len(modin_df.index)
        modin_df.index = [
            np.nan if i % 2 == 0 else modin_df.index[i] for i in range(length)
        ]
        pandas_df.index = [
            np.nan if i % 2 == 0 else pandas_df.index[i] for i in range(length)
        ]
    else:
        length = len(modin_df.columns)
        modin_df.columns = [
            np.nan if i % 2 == 0 else modin_df.columns[i] for i in range(length)
        ]
        pandas_df.columns = [
            np.nan if i % 2 == 0 else pandas_df.columns[i] for i in range(length)
        ]

    modin_result = modin_df.sort_index(
        axis=axis, ascending=ascending, na_position=na_position, inplace=False
    )
    pandas_result = pandas_df.sort_index(
        axis=axis, ascending=ascending, na_position=na_position, inplace=False
    )
    df_equals(modin_result, pandas_result)

    modin_df_cp = modin_df.copy()
    pandas_df_cp = pandas_df.copy()
    modin_df_cp.sort_index(
        axis=axis, ascending=ascending, na_position=na_position, inplace=True
    )
    pandas_df_cp.sort_index(
        axis=axis, ascending=ascending, na_position=na_position, inplace=True
    )
    df_equals(modin_df_cp, pandas_df_cp)

    # MultiIndex
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    modin_df.index = pd.MultiIndex.from_tuples(
        [(i // 10, i // 5, i) for i in range(len(modin_df))]
    )
    pandas_df.index = pandas.MultiIndex.from_tuples(
        [(i // 10, i // 5, i) for i in range(len(pandas_df))]
    )
    modin_df.columns = pd.MultiIndex.from_tuples(
        [(i // 10, i // 5, i) for i in range(len(modin_df.columns))]
    )
    pandas_df.columns = pd.MultiIndex.from_tuples(
        [(i // 10, i // 5, i) for i in range(len(pandas_df.columns))]
    )

    with pytest.warns(UserWarning):
        df_equals(modin_df.sort_index(level=0), pandas_df.sort_index(level=0))
    with pytest.warns(UserWarning):
        df_equals(modin_df.sort_index(axis=0), pandas_df.sort_index(axis=0))
    with pytest.warns(UserWarning):
        df_equals(modin_df.sort_index(axis=1), pandas_df.sort_index(axis=1))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "ascending", bool_arg_values, ids=arg_keys("ascending", bool_arg_keys)
)
@pytest.mark.parametrize("na_position", ["first", "last"], ids=["first", "last"])
def test_sort_values(request, data, axis, ascending, na_position):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    if "empty_data" not in request.node.name and (
        (axis == 0 or axis == "over rows")
        or name_contains(request.node.name, numeric_dfs)
    ):
        index = modin_df.index if axis == 1 or axis == "columns" else modin_df.columns
        key = index[0]
        modin_result = modin_df.sort_values(
            key,
            axis=axis,
            ascending=ascending,
            na_position=na_position,
            inplace=False,
        )
        pandas_result = pandas_df.sort_values(
            key,
            axis=axis,
            ascending=ascending,
            na_position=na_position,
            inplace=False,
        )
        df_equals(modin_result, pandas_result)

        modin_df_cp = modin_df.copy()
        pandas_df_cp = pandas_df.copy()
        modin_df_cp.sort_values(
            key,
            axis=axis,
            ascending=ascending,
            na_position=na_position,
            inplace=True,
        )
        pandas_df_cp.sort_values(
            key,
            axis=axis,
            ascending=ascending,
            na_position=na_position,
            inplace=True,
        )
        df_equals(modin_df_cp, pandas_df_cp)

        keys = [key, index[-1]]
        modin_result = modin_df.sort_values(
            keys,
            axis=axis,
            ascending=ascending,
            na_position=na_position,
            inplace=False,
        )
        pandas_result = pandas_df.sort_values(
            keys,
            axis=axis,
            ascending=ascending,
            na_position=na_position,
            inplace=False,
        )
        df_equals(modin_result, pandas_result)

        modin_df_cp = modin_df.copy()
        pandas_df_cp = pandas_df.copy()
        modin_df_cp.sort_values(
            keys,
            axis=axis,
            ascending=ascending,
            na_position=na_position,
            inplace=True,
        )
        pandas_df_cp.sort_values(
            keys,
            axis=axis,
            ascending=ascending,
            na_position=na_position,
            inplace=True,
        )
        df_equals(modin_df_cp, pandas_df_cp)


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


def test_where():
    frame_data = random_state.randn(100, 10)
    pandas_df = pandas.DataFrame(frame_data, columns=list("abcdefghij"))
    modin_df = pd.DataFrame(frame_data, columns=list("abcdefghij"))
    pandas_cond_df = pandas_df % 5 < 2
    modin_cond_df = modin_df % 5 < 2

    pandas_result = pandas_df.where(pandas_cond_df, -pandas_df)
    modin_result = modin_df.where(modin_cond_df, -modin_df)
    assert all((to_pandas(modin_result) == pandas_result).all())

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
