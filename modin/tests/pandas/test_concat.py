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

import modin.pandas as pd
from modin.config import NPartitions, StorageFormat
from modin.pandas.io import from_pandas
from modin.utils import get_current_execution

from .utils import (
    create_test_dfs,
    default_to_pandas_ignore_string,
    df_equals,
    generate_dfs,
    generate_multiindex_dfs,
    generate_none_dfs,
)

NPartitions.put(4)

pytestmark = pytest.mark.filterwarnings(default_to_pandas_ignore_string)

# Initialize env for storage format detection in @pytest.mark.*
pd.DataFrame()


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


@pytest.mark.parametrize("no_dup_cols", [True, False])
@pytest.mark.parametrize("different_len", [True, False])
def test_concat_on_column(no_dup_cols, different_len):
    df, df2 = generate_dfs()
    if no_dup_cols:
        df = df.drop(set(df.columns) & set(df2.columns), axis="columns")
    if different_len:
        df = pandas.concat([df, df], ignore_index=True)

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
        pd.concat(mixed_dfs, join="inner"),
        pandas.concat([df, df2, df3], join="inner"),
        # https://github.com/modin-project/modin/issues/5963
        check_dtypes=False,
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
    frame_data = np.random.randint(0, 100, size=(2**10, 2**6))
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


def test_concat_5776():
    modin_data = {key: pd.Series(index=range(3)) for key in ["a", "b"]}
    pandas_data = {key: pandas.Series(index=range(3)) for key in ["a", "b"]}
    df_equals(
        pd.concat(modin_data, axis="columns"),
        pandas.concat(pandas_data, axis="columns"),
    )


def test_concat_6840():
    groupby_objs = []
    for idx, lib in enumerate((pd, pandas)):
        df1 = lib.DataFrame(
            [["a", 1], ["b", 2], ["b", 4]], columns=["letter", "number"]
        )
        df1_g = df1.groupby("letter", as_index=False)["number"].agg("sum")

        df2 = lib.DataFrame(
            [["a", 3], ["a", 4], ["b", 1]], columns=["letter", "number"]
        )
        df2_g = df2.groupby("letter", as_index=False)["number"].agg("sum")
        groupby_objs.append([df1_g, df2_g])

    df_equals(
        pd.concat(groupby_objs[0]),
        pandas.concat(groupby_objs[1]),
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
        # https://github.com/modin-project/modin/issues/5963
        check_dtypes=join != "inner",
    )
    assert list(pandas_concat.columns) == list(modin_concat.columns)


@pytest.mark.parametrize(
    "data1, index1, data2, index2",
    [
        (None, None, None, None),
        (None, None, {"A": [1, 2, 3]}, pandas.Index([1, 2, 3], name="Test")),
        ({"A": [1, 2, 3]}, pandas.Index([1, 2, 3], name="Test"), None, None),
        ({"A": [1, 2, 3]}, None, None, None),
        (None, None, {"A": [1, 2, 3]}, None),
        (None, pandas.Index([1, 2, 3], name="Test"), None, None),
        (None, None, None, pandas.Index([1, 2, 3], name="Test")),
    ],
)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("join", ["inner", "outer"])
def test_concat_empty(data1, index1, data2, index2, axis, join):
    pdf1 = pandas.DataFrame(data1, index=index1)
    pdf2 = pandas.DataFrame(data2, index=index2)
    pdf = pandas.concat((pdf1, pdf2), axis=axis, join=join)
    mdf1 = pd.DataFrame(data1, index=index1)
    mdf2 = pd.DataFrame(data2, index=index2)
    mdf = pd.concat((mdf1, mdf2), axis=axis, join=join)
    df_equals(
        pdf,
        mdf,
        # https://github.com/modin-project/modin/issues/5963
        check_dtypes=join != "inner",
    )


def test_concat_empty_df_series():
    pdf = pandas.concat((pandas.DataFrame({"A": [1, 2, 3]}), pandas.Series()))
    mdf = pd.concat((pd.DataFrame({"A": [1, 2, 3]}), pd.Series()))
    df_equals(
        pdf,
        mdf,
        # https://github.com/modin-project/modin/issues/5964
        check_dtypes=False,
    )
    pdf = pandas.concat((pandas.DataFrame(), pandas.Series([1, 2, 3])))
    mdf = pd.concat((pd.DataFrame(), pd.Series([1, 2, 3])))
    df_equals(
        pdf,
        mdf,
        # https://github.com/modin-project/modin/issues/5964
        check_dtypes=False,
    )


@pytest.mark.skipif(
    StorageFormat.get() != "Base",
    reason="https://github.com/modin-project/modin/issues/5696",
)
@pytest.mark.parametrize("col_type", [None, "str"])
@pytest.mark.parametrize("df1_cols", [0, 90, 100])
@pytest.mark.parametrize("df2_cols", [0, 90, 100])
@pytest.mark.parametrize("df1_rows", [0, 100])
@pytest.mark.parametrize("df2_rows", [0, 100])
@pytest.mark.parametrize("idx_type", [None, "str"])
@pytest.mark.parametrize("ignore_index", [True, False])
@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("join", ["inner", "outer"])
def test_concat_different_num_cols(
    col_type,
    df1_cols,
    df2_cols,
    df1_rows,
    df2_rows,
    idx_type,
    ignore_index,
    sort,
    join,
):
    def create_frame(frame_type, ncols, nrows):
        def to_str(val):
            return f"str_{val}"

        off = 0
        data = {}
        for n in range(1, ncols + 1):
            row = range(off + 1, off + nrows + 1)
            if col_type == "str":
                row = map(to_str, row)
            data[f"Col_{n}"] = list(row)
            off += nrows

        idx = None
        if idx_type == "str":
            idx = pandas.Index(map(to_str, range(1, nrows + 1)), name=f"Index_{nrows}")
        df = frame_type(data=data, index=idx)
        return df

    def concat(frame_type, lib):
        df1 = create_frame(frame_type, df1_cols, df1_rows)
        df2 = create_frame(frame_type, df2_cols, df2_rows)
        return lib.concat([df1, df2], ignore_index=ignore_index, sort=sort, join=join)

    mdf = concat(pd.DataFrame, pd)
    pdf = concat(pandas.DataFrame, pandas)
    df_equals(
        pdf,
        mdf,
        # Empty slicing causes this bug:
        # https://github.com/modin-project/modin/issues/5974
        check_dtypes=not (
            get_current_execution() == "BaseOnPython"
            and any(o == 0 for o in (df1_cols, df2_cols, df1_rows, df2_rows))
        ),
    )
