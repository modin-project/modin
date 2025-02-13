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

import matplotlib
import numpy as np
import pandas
import pytest

import modin.pandas as pd
from modin.config import NPartitions
from modin.pandas.io import to_pandas
from modin.tests.pandas.native_df_interoperability.utils import (
    create_test_df_in_defined_mode,
    create_test_series_in_defined_mode,
    eval_general_interop,
)
from modin.tests.pandas.utils import (
    default_to_pandas_ignore_string,
    df_equals,
    eval_general,
    random_state,
    test_data_keys,
    test_data_values,
)

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
def test_combine(data, df_mode_pair):
    modin_df_1, pandas_df_1 = create_test_df_in_defined_mode(
        data, native=df_mode_pair[0]
    )
    modin_df_2, pandas_df_2 = create_test_df_in_defined_mode(
        data, native=df_mode_pair[1]
    )
    modin_df_1.combine(
        modin_df_2 + 1, lambda s1, s2: s1 if s1.count() < s2.count() else s2
    )
    pandas_df_1.combine(
        pandas_df_2 + 1, lambda s1, s2: s1 if s1.count() < s2.count() else s2
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
def test_join(test_data, test_data2, df_mode_pair):
    modin_df, pandas_df = create_test_df_in_defined_mode(
        test_data,
        columns=["col{}".format(i) for i in range(test_data.shape[1])],
        index=pd.Index([i for i in range(1, test_data.shape[0] + 1)], name="key"),
        native=df_mode_pair[0],
    )
    modin_df2, pandas_df2 = create_test_df_in_defined_mode(
        test_data2,
        columns=["col{}".format(i) for i in range(test_data2.shape[1])],
        index=pd.Index([i for i in range(1, test_data2.shape[0] + 1)], name="key"),
        native=df_mode_pair[1],
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


def test_join_cross_6786(df_mode_pair):
    data = [[7, 8, 9], [10, 11, 12]]
    modin_df_1, pandas_df_1 = create_test_df_in_defined_mode(
        data, columns=["x", "y", "z"], native=df_mode_pair[0]
    )
    modin_df_2, pandas_df_2 = create_test_df_in_defined_mode(
        data, columns=["x", "y", "z"], native=df_mode_pair[1]
    )
    modin_join = modin_df_1.join(
        modin_df_2[["x"]].set_axis(["p", "q"], axis=0), how="cross", lsuffix="p"
    )
    pandas_join = pandas_df_1.join(
        pandas_df_2[["x"]].set_axis(["p", "q"], axis=0), how="cross", lsuffix="p"
    )
    df_equals(modin_join, pandas_join)


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
def test_merge(test_data, test_data2, df_mode_pair):
    modin_df, pandas_df = create_test_df_in_defined_mode(
        test_data,
        columns=["col{}".format(i) for i in range(test_data.shape[1])],
        index=pd.Index([i for i in range(1, test_data.shape[0] + 1)], name="key"),
        native=df_mode_pair[0],
    )
    modin_df2, pandas_df2 = create_test_df_in_defined_mode(
        test_data2,
        columns=["col{}".format(i) for i in range(test_data2.shape[1])],
        index=pd.Index([i for i in range(1, test_data2.shape[0] + 1)], name="key"),
        native=df_mode_pair[1],
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


@pytest.mark.parametrize("how", ["left", "inner", "right"])
def test_merge_empty(
    how,
    df_mode_pair,
):
    data = np.random.randint(0, 100, size=(64, 64))
    eval_general_interop(
        data,
        None,
        lambda df1, df2: df1.merge(df2.iloc[:0], how=how),
        df_mode_pair,
    )


def test_merge_with_mi_columns(df_mode_pair):
    modin_df1, pandas_df1 = create_test_df_in_defined_mode(
        {
            ("col0", "a"): [1, 2, 3, 4],
            ("col0", "b"): [2, 3, 4, 5],
            ("col1", "a"): [3, 4, 5, 6],
        },
        native=df_mode_pair[0],
    )

    modin_df2, pandas_df2 = create_test_df_in_defined_mode(
        {
            ("col0", "a"): [1, 2, 3, 4],
            ("col0", "c"): [2, 3, 4, 5],
            ("col1", "a"): [3, 4, 5, 6],
        },
        native=df_mode_pair[1],
    )

    eval_general(
        (modin_df1, modin_df2),
        (pandas_df1, pandas_df2),
        lambda dfs: dfs[0].merge(dfs[1], on=[("col0", "a")]),
    )


def test_where(df_mode_pair):
    columns = list("abcdefghij")

    frame_data = random_state.randn(100, 10)
    modin_df_1, pandas_df_1 = create_test_df_in_defined_mode(
        frame_data, columns=columns, native=df_mode_pair[0]
    )
    modin_df_2, pandas_df_2 = create_test_df_in_defined_mode(
        frame_data, columns=columns, native=df_mode_pair[1]
    )
    pandas_cond_df = pandas_df_2 % 5 < 2
    modin_cond_df = modin_df_2 % 5 < 2

    pandas_result = pandas_df_1.where(pandas_cond_df, -pandas_df_2)
    modin_result = modin_df_1.where(modin_cond_df, -modin_df_2)
    assert all((to_pandas(modin_result) == pandas_result).all())

    # test case when other is Series
    other_data = random_state.randn(len(pandas_df_1))
    modin_other, pandas_other = create_test_series_in_defined_mode(
        other_data, native=df_mode_pair[0]
    )
    pandas_result = pandas_df_1.where(pandas_cond_df, pandas_other, axis=0)
    modin_result = modin_df_1.where(modin_cond_df, modin_other, axis=0)
    df_equals(modin_result, pandas_result)

    # Test that we choose the right values to replace when `other` == `True`
    # everywhere.
    other_data = np.full(shape=pandas_df_1.shape, fill_value=True)
    modin_other, pandas_other = create_test_df_in_defined_mode(
        other_data, columns=columns, native=df_mode_pair[0]
    )
    pandas_result = pandas_df_1.where(pandas_cond_df, pandas_other)
    modin_result = modin_df_1.where(modin_cond_df, modin_other)
    df_equals(modin_result, pandas_result)

    other = pandas_df_1.loc[3]
    pandas_result = pandas_df_1.where(pandas_cond_df, other, axis=1)
    modin_result = modin_df_1.where(modin_cond_df, other, axis=1)
    assert all((to_pandas(modin_result) == pandas_result).all())

    other = pandas_df_1["e"]
    pandas_result = pandas_df_1.where(pandas_cond_df, other, axis=0)
    modin_result = modin_df_1.where(modin_cond_df, other, axis=0)
    assert all((to_pandas(modin_result) == pandas_result).all())

    pandas_result = pandas_df_1.where(pandas_df_2 < 2, True)
    modin_result = modin_df_1.where(modin_df_2 < 2, True)
    assert all((to_pandas(modin_result) == pandas_result).all())


@pytest.mark.parametrize("align_axis", ["index", "columns"])
@pytest.mark.parametrize("keep_shape", [False, True])
@pytest.mark.parametrize("keep_equal", [False, True])
def test_compare(align_axis, keep_shape, keep_equal, df_mode_pair):
    kwargs = {
        "align_axis": align_axis,
        "keep_shape": keep_shape,
        "keep_equal": keep_equal,
    }
    frame_data1 = random_state.randn(100, 10)
    frame_data2 = random_state.randn(100, 10)
    modin_df, pandas_df = create_test_df_in_defined_mode(
        frame_data1, columns=list("abcdefghij"), native=df_mode_pair[0]
    )
    modin_df2, pandas_df2 = create_test_df_in_defined_mode(
        frame_data2, columns=list("abcdefghij"), native=df_mode_pair[0]
    )
    modin_result = modin_df.compare(modin_df2, **kwargs)
    pandas_result = pandas_df.compare(pandas_df2, **kwargs)
    assert to_pandas(modin_result).equals(pandas_result)

    modin_result = modin_df2.compare(modin_df, **kwargs)
    pandas_result = pandas_df2.compare(pandas_df, **kwargs)
    assert to_pandas(modin_result).equals(pandas_result)

    series_data1 = ["a", "b", "c", "d", "e"]
    series_data2 = ["a", "a", "c", "b", "e"]
    modin_series1, pandas_series1 = create_test_series_in_defined_mode(
        series_data1, native=df_mode_pair[0]
    )
    modin_series2, pandas_series2 = create_test_series_in_defined_mode(
        series_data2, native=df_mode_pair[1]
    )

    modin_result = modin_series1.compare(modin_series2, **kwargs)
    pandas_result = pandas_series1.compare(pandas_series2, **kwargs)
    assert to_pandas(modin_result).equals(pandas_result)

    modin_result = modin_series2.compare(modin_series1, **kwargs)
    pandas_result = pandas_series2.compare(pandas_series1, **kwargs)
    assert to_pandas(modin_result).equals(pandas_result)
