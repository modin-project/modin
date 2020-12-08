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
import pytest
import pandas

import modin.pandas as pd
from modin.pandas.utils import from_pandas
from .utils import (
    df_equals,
    generate_dfs,
    generate_multiindex_dfs,
    generate_none_dfs,
    create_test_dfs,
)

pd.DEFAULT_NPARTITIONS = 4


def test_df_concat():
    df, df2 = generate_dfs()

    df_equals(pd.concat([df, df2]), pandas.concat([df, df2]))


def test_concat():
    df, df2 = generate_dfs()
    modin_df, modin_df2 = from_pandas(df), from_pandas(df2)

    df_equals(pd.concat([modin_df, modin_df2]), pandas.concat([df, df2]))


def test_concat_with_series():
    df, df2 = generate_dfs()
    modin_df, modin_df2 = from_pandas(df), from_pandas(df2)
    pandas_series = pandas.Series([1, 2, 3, 4], name="new_col")

    df_equals(
        pd.concat([modin_df, modin_df2, pandas_series], axis=0),
        pandas.concat([df, df2, pandas_series], axis=0),
    )

    df_equals(
        pd.concat([modin_df, modin_df2, pandas_series], axis=1),
        pandas.concat([df, df2, pandas_series], axis=1),
    )


def test_concat_on_index():
    df, df2 = generate_dfs()
    modin_df, modin_df2 = from_pandas(df), from_pandas(df2)

    df_equals(
        pd.concat([modin_df, modin_df2], axis="index"),
        pandas.concat([df, df2], axis="index"),
    )

    df_equals(
        pd.concat([modin_df, modin_df2], axis="rows"),
        pandas.concat([df, df2], axis="rows"),
    )

    df_equals(
        pd.concat([modin_df, modin_df2], axis=0), pandas.concat([df, df2], axis=0)
    )


def test_concat_on_column():
    df, df2 = generate_dfs()
    modin_df, modin_df2 = from_pandas(df), from_pandas(df2)

    df_equals(
        pd.concat([modin_df, modin_df2], axis=1), pandas.concat([df, df2], axis=1)
    )

    df_equals(
        pd.concat([modin_df, modin_df2], axis="columns"),
        pandas.concat([df, df2], axis="columns"),
    )

    modin_result = pd.concat(
        [pd.Series(np.ones(10)), pd.Series(np.ones(10))], axis=1, ignore_index=True
    )
    pandas_result = pandas.concat(
        [pandas.Series(np.ones(10)), pandas.Series(np.ones(10))],
        axis=1,
        ignore_index=True,
    )
    df_equals(modin_result, pandas_result)
    assert modin_result.dtypes.equals(pandas_result.dtypes)


def test_invalid_axis_errors():
    df, df2 = generate_dfs()
    modin_df, modin_df2 = from_pandas(df), from_pandas(df2)

    with pytest.raises(ValueError):
        pd.concat([modin_df, modin_df2], axis=2)


def test_mixed_concat():
    df, df2 = generate_dfs()
    df3 = df.copy()

    mixed_dfs = [from_pandas(df), from_pandas(df2), df3]

    df_equals(pd.concat(mixed_dfs), pandas.concat([df, df2, df3]))


def test_mixed_inner_concat():
    df, df2 = generate_dfs()
    df3 = df.copy()

    mixed_dfs = [from_pandas(df), from_pandas(df2), df3]

    df_equals(
        pd.concat(mixed_dfs, join="inner"), pandas.concat([df, df2, df3], join="inner")
    )


def test_mixed_none_concat():
    df, df2 = generate_none_dfs()
    df3 = df.copy()

    mixed_dfs = [from_pandas(df), from_pandas(df2), df3]

    df_equals(pd.concat(mixed_dfs), pandas.concat([df, df2, df3]))


def test_ignore_index_concat():
    df, df2 = generate_dfs()

    df_equals(
        pd.concat([df, df2], ignore_index=True),
        pandas.concat([df, df2], ignore_index=True),
    )


def test_concat_non_subscriptable_keys():
    frame_data = np.random.randint(0, 100, size=(2 ** 10, 2 ** 6))
    df = pd.DataFrame(frame_data).add_prefix("col")
    pdf = pandas.DataFrame(frame_data).add_prefix("col")

    modin_dict = {"c": df.copy(), "b": df.copy()}
    pandas_dict = {"c": pdf.copy(), "b": pdf.copy()}
    modin_result = pd.concat(modin_dict.values(), keys=modin_dict.keys())
    pandas_result = pandas.concat(pandas_dict.values(), keys=pandas_dict.keys())
    df_equals(modin_result, pandas_result)


def test_concat_series_only():
    modin_series = pd.Series(list(range(1000)))
    pandas_series = pandas.Series(list(range(1000)))

    df_equals(
        pd.concat([modin_series, modin_series]),
        pandas.concat([pandas_series, pandas_series]),
    )


def test_concat_with_empty_frame():
    modin_empty_df = pd.DataFrame()
    pandas_empty_df = pandas.DataFrame()
    modin_row = pd.Series({0: "a", 1: "b"})
    pandas_row = pandas.Series({0: "a", 1: "b"})
    df_equals(
        pd.concat([modin_empty_df, modin_row]),
        pandas.concat([pandas_empty_df, pandas_row]),
    )

    md_empty1, pd_empty1 = create_test_dfs(index=[1, 2, 3])
    md_empty2, pd_empty2 = create_test_dfs(index=[2, 3, 4])

    df_equals(
        pd.concat([md_empty1, md_empty2], axis=0),
        pandas.concat([pd_empty1, pd_empty2], axis=0),
    )
    df_equals(
        pd.concat([md_empty1, md_empty2], axis=1),
        pandas.concat([pd_empty1, pd_empty2], axis=1),
    )


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("names", [False, True])
def test_concat_multiindex(axis, names):
    pd_df1, pd_df2 = generate_multiindex_dfs(axis=axis)
    md_df1, md_df2 = map(from_pandas, [pd_df1, pd_df2])

    keys = ["first", "second"]
    if names:
        names = [str(i) for i in np.arange(pd_df1.axes[axis].nlevels + 1)]
    else:
        names = None

    df_equals(
        pd.concat([md_df1, md_df2], keys=keys, axis=axis, names=names),
        pandas.concat([pd_df1, pd_df2], keys=keys, axis=axis, names=names),
    )


@pytest.mark.parametrize("axis", [0, 1])
def test_concat_dictionary(axis):
    pandas_df, pandas_df2 = generate_dfs()
    modin_df, modin_df2 = from_pandas(pandas_df), from_pandas(pandas_df2)

    df_equals(
        pd.concat({"A": modin_df, "B": modin_df2}, axis=axis),
        pandas.concat({"A": pandas_df, "B": pandas_df2}, axis=axis),
    )


@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("join", ["inner", "outer"])
@pytest.mark.parametrize("axis", [0, 1])
def test_sort_order(sort, join, axis):
    pandas_df = pandas.DataFrame({"c": [3], "d": [4]}, columns=["d", "c"])
    pandas_df2 = pandas.DataFrame({"a": [1], "b": [2]}, columns=["b", "a"])
    modin_df, modin_df2 = from_pandas(pandas_df), from_pandas(pandas_df2)
    pandas_concat = pandas.concat([pandas_df, pandas_df2], join=join, sort=sort)
    modin_concat = pd.concat([modin_df, modin_df2], join=join, sort=sort)
    df_equals(
        pandas_concat,
        modin_concat,
    )
    assert list(pandas_concat.columns) == list(modin_concat.columns)
