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
import pandas
import numpy as np
import modin.pandas as pd
from modin.utils import try_cast_to_pandas, to_pandas, get_current_backend
from modin.pandas.utils import from_pandas
from .utils import (
    df_equals,
    check_df_columns_have_nans,
    create_test_dfs,
    eval_general,
    df_categories_equals,
    test_data_values,
)

pd.DEFAULT_NPARTITIONS = 4


def modin_df_almost_equals_pandas(modin_df, pandas_df):
    df_categories_equals(modin_df._to_pandas(), pandas_df)

    modin_df = to_pandas(modin_df)

    if hasattr(modin_df, "select_dtypes"):
        modin_df = modin_df.select_dtypes(exclude=["category"])
    if hasattr(pandas_df, "select_dtypes"):
        pandas_df = pandas_df.select_dtypes(exclude=["category"])

    difference = modin_df - pandas_df
    diff_max = difference.max()
    if isinstance(diff_max, pandas.Series):
        diff_max = diff_max.max()
    assert (
        modin_df.equals(pandas_df)
        or diff_max < 0.0001
        or (all(modin_df.isna().all()) and all(pandas_df.isna().all()))
    )


def modin_groupby_equals_pandas(modin_groupby, pandas_groupby):
    for g1, g2 in zip(modin_groupby, pandas_groupby):
        assert g1[0] == g2[0]
        df_equals(g1[1], g2[1])


def eval_aggregation(md_df, pd_df, operation=None, by=None, *args, **kwargs):
    if by is None:
        by = md_df.columns[0]
    if operation is None:
        operation = {}
    return eval_general(
        md_df,
        pd_df,
        lambda df, *args, **kwargs: df.groupby(by=by).agg(operation, *args, **kwargs),
        *args,
        **kwargs,
    )


@pytest.mark.parametrize("as_index", [True, False])
def test_mixed_dtypes_groupby(as_index):
    frame_data = np.random.randint(97, 198, size=(2 ** 6, 2 ** 4))
    pandas_df = pandas.DataFrame(frame_data).add_prefix("col")
    # Convert every other column to string
    for col in pandas_df.iloc[
        :, [i for i in range(len(pandas_df.columns)) if i % 2 == 0]
    ]:
        pandas_df[col] = [str(chr(i)) for i in pandas_df[col]]
    modin_df = from_pandas(pandas_df)

    n = 1

    by_values = [
        ("col1",),
        (lambda x: x % 2,),
        (modin_df["col0"].copy(), pandas_df["col0"].copy()),
        ("col3",),
    ]

    for by in by_values:
        if by_values[0] == "col3":
            modin_groupby = modin_df.set_index(by[0]).groupby(
                by=by[0], as_index=as_index
            )
            pandas_groupby = pandas_df.set_index(by[0]).groupby(
                by=by[-1], as_index=as_index
            )
        else:
            modin_groupby = modin_df.groupby(by=by[0], as_index=as_index)
            pandas_groupby = pandas_df.groupby(by=by[-1], as_index=as_index)

        modin_groupby_equals_pandas(modin_groupby, pandas_groupby)
        eval_ngroups(modin_groupby, pandas_groupby)
        eval_general(
            modin_groupby, pandas_groupby, lambda df: df.ffill(), is_default=True
        )
        eval_general(
            modin_groupby,
            pandas_groupby,
            lambda df: df.sem(),
            modin_df_almost_equals_pandas,
            is_default=True,
        )
        eval_mean(modin_groupby, pandas_groupby)
        eval_any(modin_groupby, pandas_groupby)
        eval_min(modin_groupby, pandas_groupby)
        eval_general(
            modin_groupby, pandas_groupby, lambda df: df.idxmax(), is_default=True
        )
        eval_ndim(modin_groupby, pandas_groupby)
        eval_cumsum(modin_groupby, pandas_groupby)
        eval_general(
            modin_groupby,
            pandas_groupby,
            lambda df: df.pct_change(),
            modin_df_almost_equals_pandas,
            is_default=True,
        )
        eval_cummax(modin_groupby, pandas_groupby)

        # TODO Add more apply functions
        apply_functions = [lambda df: df.sum(), min]
        # Workaround for Pandas bug #34656. Recreate groupby object for Pandas
        pandas_groupby = pandas_df.groupby(by=by[-1], as_index=as_index)
        for func in apply_functions:
            eval_apply(modin_groupby, pandas_groupby, func)

        eval_dtypes(modin_groupby, pandas_groupby)
        eval_general(
            modin_groupby, pandas_groupby, lambda df: df.first(), is_default=True
        )
        eval_general(
            modin_groupby, pandas_groupby, lambda df: df.backfill(), is_default=True
        )
        eval_cummin(modin_groupby, pandas_groupby)
        eval_general(
            modin_groupby, pandas_groupby, lambda df: df.bfill(), is_default=True
        )
        eval_general(
            modin_groupby, pandas_groupby, lambda df: df.idxmin(), is_default=True
        )
        eval_prod(modin_groupby, pandas_groupby)
        if as_index:
            eval_std(modin_groupby, pandas_groupby)
            eval_var(modin_groupby, pandas_groupby)
            eval_skew(modin_groupby, pandas_groupby)

        agg_functions = ["min", "max"]
        for func in agg_functions:
            eval_agg(modin_groupby, pandas_groupby, func)
            eval_aggregate(modin_groupby, pandas_groupby, func)

        eval_general(
            modin_groupby, pandas_groupby, lambda df: df.last(), is_default=True
        )
        eval_general(
            modin_groupby,
            pandas_groupby,
            lambda df: df.mad(),
            modin_df_almost_equals_pandas,
            is_default=True,
        )
        eval_max(modin_groupby, pandas_groupby)
        eval_len(modin_groupby, pandas_groupby)
        eval_sum(modin_groupby, pandas_groupby)
        eval_ngroup(modin_groupby, pandas_groupby)
        eval_nunique(modin_groupby, pandas_groupby)
        eval_median(modin_groupby, pandas_groupby)
        eval_general(
            modin_groupby, pandas_groupby, lambda df: df.head(n), is_default=True
        )
        eval_cumprod(modin_groupby, pandas_groupby)
        eval_general(
            modin_groupby,
            pandas_groupby,
            lambda df: df.cov(),
            modin_df_almost_equals_pandas,
            is_default=True,
        )

        transform_functions = [lambda df: df, lambda df: df + df]
        for func in transform_functions:
            eval_transform(modin_groupby, pandas_groupby, func)

        pipe_functions = [lambda dfgb: dfgb.sum()]
        for func in pipe_functions:
            eval_pipe(modin_groupby, pandas_groupby, func)

        eval_general(
            modin_groupby,
            pandas_groupby,
            lambda df: df.corr(),
            modin_df_almost_equals_pandas,
        )
        eval_fillna(modin_groupby, pandas_groupby)
        eval_count(modin_groupby, pandas_groupby)
        eval_general(
            modin_groupby, pandas_groupby, lambda df: df.tail(n), is_default=True
        )
        eval_quantile(modin_groupby, pandas_groupby)
        eval_general(
            modin_groupby, pandas_groupby, lambda df: df.take(), is_default=True
        )
        eval___getattr__(modin_groupby, pandas_groupby, "col2")
        eval_groups(modin_groupby, pandas_groupby)


class GetColumn:
    """Indicate to the test that it should do gc(df)."""

    def __init__(self, name):
        self.name = name

    def __call__(self, df):
        return df[self.name]


@pytest.mark.parametrize(
    "by",
    [
        [1, 2, 1, 2],
        lambda x: x % 3,
        "col1",
        ["col1"],
        # col2 contains NaN, is it necessary to test functions like size()
        "col2",
        ["col2"],
        pytest.param(
            ["col1", "col2"],
            marks=pytest.mark.xfail(reason="Excluded because of bug #1554"),
        ),
        pytest.param(
            ["col2", "col4"],
            marks=pytest.mark.xfail(reason="Excluded because of bug #1554"),
        ),
        pytest.param(
            ["col4", "col2"],
            marks=pytest.mark.xfail(reason="Excluded because of bug #1554"),
        ),
        pytest.param(
            ["col3", "col4", "col2"],
            marks=pytest.mark.xfail(reason="Excluded because of bug #1554"),
        ),
        # but cum* functions produce undefined results with NaNs so we need to test the same combinations without NaN too
        ["col5"],
        ["col1", "col5"],
        ["col5", "col4"],
        ["col4", "col5"],
        ["col5", "col4", "col1"],
        ["col1", pd.Series([1, 5, 7, 8])],
        [pd.Series([1, 5, 7, 8])],
        [
            pd.Series([1, 5, 7, 8]),
            pd.Series([1, 5, 7, 8]),
            pd.Series([1, 5, 7, 8]),
            pd.Series([1, 5, 7, 8]),
            pd.Series([1, 5, 7, 8]),
        ],
        ["col1", GetColumn("col5")],
        [GetColumn("col1"), GetColumn("col5")],
        [GetColumn("col1")],
    ],
)
@pytest.mark.parametrize("as_index", [True, False])
@pytest.mark.parametrize("col1_category", [True, False])
def test_simple_row_groupby(by, as_index, col1_category):
    pandas_df = pandas.DataFrame(
        {
            "col1": [0, 1, 2, 3],
            "col2": [4, 5, np.NaN, 7],
            "col3": [np.NaN, np.NaN, 12, 10],
            "col4": [17, 13, 16, 15],
            "col5": [-4, -5, -6, -7],
        }
    )

    if col1_category:
        pandas_df = pandas_df.astype({"col1": "category"})

    modin_df = from_pandas(pandas_df)
    n = 1

    def maybe_get_columns(df, by):
        if isinstance(by, list):
            return [o(df) if isinstance(o, GetColumn) else o for o in by]
        else:
            return by

    modin_groupby = modin_df.groupby(
        by=maybe_get_columns(modin_df, by), as_index=as_index
    )

    pandas_by = maybe_get_columns(pandas_df, try_cast_to_pandas(by))
    pandas_groupby = pandas_df.groupby(by=pandas_by, as_index=as_index)

    modin_groupby_equals_pandas(modin_groupby, pandas_groupby)
    eval_ngroups(modin_groupby, pandas_groupby)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.ffill(), is_default=True)
    eval_general(
        modin_groupby,
        pandas_groupby,
        lambda df: df.sem(),
        modin_df_almost_equals_pandas,
        is_default=True,
    )
    eval_mean(modin_groupby, pandas_groupby)
    eval_any(modin_groupby, pandas_groupby)
    eval_min(modin_groupby, pandas_groupby)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.idxmax(), is_default=True)
    eval_ndim(modin_groupby, pandas_groupby)
    if not check_df_columns_have_nans(modin_df, by):
        # cum* functions produce undefined results for columns with NaNs so we run them only when "by" columns contain no NaNs
        eval_general(modin_groupby, pandas_groupby, lambda df: df.cumsum(axis=0))
        eval_general(modin_groupby, pandas_groupby, lambda df: df.cummax(axis=0))
        eval_general(modin_groupby, pandas_groupby, lambda df: df.cummin(axis=0))
        eval_general(modin_groupby, pandas_groupby, lambda df: df.cumprod(axis=0))

    eval_general(
        modin_groupby,
        pandas_groupby,
        lambda df: df.pct_change(),
        modin_df_almost_equals_pandas,
        is_default=True,
    )

    # Workaround for Pandas bug #34656. Recreate groupby object for Pandas
    pandas_groupby = pandas_df.groupby(by=pandas_by, as_index=as_index)
    apply_functions = [lambda df: df.sum(), min]
    for func in apply_functions:
        eval_apply(modin_groupby, pandas_groupby, func)

    eval_dtypes(modin_groupby, pandas_groupby)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.first(), is_default=True)
    eval_general(
        modin_groupby, pandas_groupby, lambda df: df.backfill(), is_default=True
    )
    eval_general(modin_groupby, pandas_groupby, lambda df: df.bfill(), is_default=True)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.idxmin(), is_default=True)
    eval_prod(modin_groupby, pandas_groupby)
    if as_index:
        eval_std(modin_groupby, pandas_groupby)
        eval_var(modin_groupby, pandas_groupby)
        eval_skew(modin_groupby, pandas_groupby)

    agg_functions = ["min", "max"]
    for func in agg_functions:
        eval_agg(modin_groupby, pandas_groupby, func)
        eval_aggregate(modin_groupby, pandas_groupby, func)

    eval_general(modin_groupby, pandas_groupby, lambda df: df.last(), is_default=True)
    eval_general(
        modin_groupby,
        pandas_groupby,
        lambda df: df.mad(),
        modin_df_almost_equals_pandas,
        is_default=True,
    )
    eval_general(modin_groupby, pandas_groupby, lambda df: df.rank())
    eval_max(modin_groupby, pandas_groupby)
    eval_len(modin_groupby, pandas_groupby)
    eval_sum(modin_groupby, pandas_groupby)
    eval_ngroup(modin_groupby, pandas_groupby)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.nunique())
    eval_median(modin_groupby, pandas_groupby)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.head(n), is_default=True)
    eval_general(
        modin_groupby,
        pandas_groupby,
        lambda df: df.cov(),
        modin_df_almost_equals_pandas,
        is_default=True,
    )

    if not check_df_columns_have_nans(modin_df, by):
        # Pandas groupby.transform does not work correctly with NaN values in grouping columns. See Pandas bug 17093.
        transform_functions = [lambda df: df + 4, lambda df: -df - 10]
        for func in transform_functions:
            eval_general(
                modin_groupby,
                pandas_groupby,
                lambda df: df.transform(func),
                check_exception_type=None,
            )

    pipe_functions = [lambda dfgb: dfgb.sum()]
    for func in pipe_functions:
        eval_pipe(modin_groupby, pandas_groupby, func)

    eval_general(
        modin_groupby,
        pandas_groupby,
        lambda df: df.corr(),
        modin_df_almost_equals_pandas,
        is_default=True,
    )
    eval_fillna(modin_groupby, pandas_groupby)
    eval_count(modin_groupby, pandas_groupby)
    if get_current_backend() != "BaseOnPython":
        eval_general(
            modin_groupby,
            pandas_groupby,
            lambda df: df.size(),
            check_exception_type=None,
        )
    eval_general(modin_groupby, pandas_groupby, lambda df: df.tail(n), is_default=True)
    eval_quantile(modin_groupby, pandas_groupby)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.take(), is_default=True)
    if isinstance(by, list) and not any(
        isinstance(o, (pd.Series, pandas.Series)) for o in by
    ):
        # Not yet supported for non-original-column-from-dataframe Series in by:
        eval___getattr__(modin_groupby, pandas_groupby, "col3")
    eval_groups(modin_groupby, pandas_groupby)


def test_single_group_row_groupby():
    pandas_df = pandas.DataFrame(
        {
            "col1": [0, 1, 2, 3],
            "col2": [4, 5, 36, 7],
            "col3": [3, 8, 12, 10],
            "col4": [17, 3, 16, 15],
            "col5": [-4, 5, -6, -7],
        }
    )

    modin_df = from_pandas(pandas_df)

    by = ["1", "1", "1", "1"]
    n = 6

    modin_groupby = modin_df.groupby(by=by)
    pandas_groupby = pandas_df.groupby(by=by)

    modin_groupby_equals_pandas(modin_groupby, pandas_groupby)
    eval_ngroups(modin_groupby, pandas_groupby)
    eval_skew(modin_groupby, pandas_groupby)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.ffill(), is_default=True)
    eval_general(
        modin_groupby,
        pandas_groupby,
        lambda df: df.sem(),
        modin_df_almost_equals_pandas,
        is_default=True,
    )
    eval_mean(modin_groupby, pandas_groupby)
    eval_any(modin_groupby, pandas_groupby)
    eval_min(modin_groupby, pandas_groupby)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.idxmax(), is_default=True)
    eval_ndim(modin_groupby, pandas_groupby)
    eval_cumsum(modin_groupby, pandas_groupby)
    eval_general(
        modin_groupby,
        pandas_groupby,
        lambda df: df.pct_change(),
        modin_df_almost_equals_pandas,
        is_default=True,
    )
    eval_cummax(modin_groupby, pandas_groupby)

    apply_functions = [lambda df: df.sum(), lambda df: -df]
    for func in apply_functions:
        eval_apply(modin_groupby, pandas_groupby, func)

    eval_dtypes(modin_groupby, pandas_groupby)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.first(), is_default=True)
    eval_general(
        modin_groupby, pandas_groupby, lambda df: df.backfill(), is_default=True
    )
    eval_cummin(modin_groupby, pandas_groupby)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.bfill(), is_default=True)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.idxmin(), is_default=True)
    eval_prod(modin_groupby, pandas_groupby)
    eval_std(modin_groupby, pandas_groupby)

    agg_functions = ["min", "max"]
    for func in agg_functions:
        eval_agg(modin_groupby, pandas_groupby, func)
        eval_aggregate(modin_groupby, pandas_groupby, func)

    eval_general(modin_groupby, pandas_groupby, lambda df: df.last(), is_default=True)
    eval_general(
        modin_groupby,
        pandas_groupby,
        lambda df: df.mad(),
        modin_df_almost_equals_pandas,
        is_default=True,
    )
    eval_rank(modin_groupby, pandas_groupby)
    eval_max(modin_groupby, pandas_groupby)
    eval_var(modin_groupby, pandas_groupby)
    eval_len(modin_groupby, pandas_groupby)
    eval_sum(modin_groupby, pandas_groupby)
    eval_ngroup(modin_groupby, pandas_groupby)
    eval_nunique(modin_groupby, pandas_groupby)
    eval_median(modin_groupby, pandas_groupby)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.head(n), is_default=True)
    eval_cumprod(modin_groupby, pandas_groupby)
    eval_general(
        modin_groupby,
        pandas_groupby,
        lambda df: df.cov(),
        modin_df_almost_equals_pandas,
        is_default=True,
    )

    transform_functions = [lambda df: df + 4, lambda df: -df - 10]
    for func in transform_functions:
        eval_transform(modin_groupby, pandas_groupby, func)

    pipe_functions = [lambda dfgb: dfgb.sum()]
    for func in pipe_functions:
        eval_pipe(modin_groupby, pandas_groupby, func)

    eval_general(
        modin_groupby,
        pandas_groupby,
        lambda df: df.corr(),
        modin_df_almost_equals_pandas,
        is_default=True,
    )
    eval_fillna(modin_groupby, pandas_groupby)
    eval_count(modin_groupby, pandas_groupby)
    eval_size(modin_groupby, pandas_groupby)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.tail(n), is_default=True)
    eval_quantile(modin_groupby, pandas_groupby)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.take(), is_default=True)
    eval___getattr__(modin_groupby, pandas_groupby, "col2")
    eval_groups(modin_groupby, pandas_groupby)


@pytest.mark.parametrize("is_by_category", [True, False])
def test_large_row_groupby(is_by_category):
    pandas_df = pandas.DataFrame(
        np.random.randint(0, 8, size=(100, 4)), columns=list("ABCD")
    )

    modin_df = from_pandas(pandas_df)

    by = [str(i) for i in pandas_df["A"].tolist()]

    if is_by_category:
        by = pandas.Categorical(by)

    n = 4

    modin_groupby = modin_df.groupby(by=by)
    pandas_groupby = pandas_df.groupby(by=by)

    modin_groupby_equals_pandas(modin_groupby, pandas_groupby)
    eval_ngroups(modin_groupby, pandas_groupby)
    eval_skew(modin_groupby, pandas_groupby)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.ffill(), is_default=True)
    eval_general(
        modin_groupby,
        pandas_groupby,
        lambda df: df.sem(),
        modin_df_almost_equals_pandas,
        is_default=True,
    )
    eval_mean(modin_groupby, pandas_groupby)
    eval_any(modin_groupby, pandas_groupby)
    eval_min(modin_groupby, pandas_groupby)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.idxmax(), is_default=True)
    eval_ndim(modin_groupby, pandas_groupby)
    eval_cumsum(modin_groupby, pandas_groupby)
    eval_general(
        modin_groupby,
        pandas_groupby,
        lambda df: df.pct_change(),
        modin_df_almost_equals_pandas,
        is_default=True,
    )
    eval_cummax(modin_groupby, pandas_groupby)

    apply_functions = [lambda df: df.sum(), lambda df: -df]
    for func in apply_functions:
        eval_apply(modin_groupby, pandas_groupby, func)

    eval_dtypes(modin_groupby, pandas_groupby)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.first(), is_default=True)
    eval_general(
        modin_groupby, pandas_groupby, lambda df: df.backfill(), is_default=True
    )
    eval_cummin(modin_groupby, pandas_groupby)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.bfill(), is_default=True)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.idxmin(), is_default=True)
    # eval_prod(modin_groupby, pandas_groupby) causes overflows
    eval_std(modin_groupby, pandas_groupby)

    agg_functions = ["min", "max"]
    for func in agg_functions:
        eval_agg(modin_groupby, pandas_groupby, func)
        eval_aggregate(modin_groupby, pandas_groupby, func)

    eval_general(modin_groupby, pandas_groupby, lambda df: df.last(), is_default=True)
    eval_general(
        modin_groupby,
        pandas_groupby,
        lambda df: df.mad(),
        modin_df_almost_equals_pandas,
        is_default=True,
    )
    eval_rank(modin_groupby, pandas_groupby)
    eval_max(modin_groupby, pandas_groupby)
    eval_var(modin_groupby, pandas_groupby)
    eval_len(modin_groupby, pandas_groupby)
    eval_sum(modin_groupby, pandas_groupby)
    eval_ngroup(modin_groupby, pandas_groupby)
    eval_nunique(modin_groupby, pandas_groupby)
    eval_median(modin_groupby, pandas_groupby)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.head(n), is_default=True)
    # eval_cumprod(modin_groupby, pandas_groupby) causes overflows
    eval_general(
        modin_groupby,
        pandas_groupby,
        lambda df: df.cov(),
        modin_df_almost_equals_pandas,
        is_default=True,
    )

    transform_functions = [lambda df: df + 4, lambda df: -df - 10]
    for func in transform_functions:
        eval_transform(modin_groupby, pandas_groupby, func)

    pipe_functions = [lambda dfgb: dfgb.sum()]
    for func in pipe_functions:
        eval_pipe(modin_groupby, pandas_groupby, func)

    eval_general(
        modin_groupby,
        pandas_groupby,
        lambda df: df.corr(),
        modin_df_almost_equals_pandas,
        is_default=True,
    )
    eval_fillna(modin_groupby, pandas_groupby)
    eval_count(modin_groupby, pandas_groupby)
    eval_size(modin_groupby, pandas_groupby)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.tail(n), is_default=True)
    eval_quantile(modin_groupby, pandas_groupby)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.take(), is_default=True)
    eval_groups(modin_groupby, pandas_groupby)


def test_simple_col_groupby():
    pandas_df = pandas.DataFrame(
        {
            "col1": [0, 3, 2, 3],
            "col2": [4, 1, 6, 7],
            "col3": [3, 8, 2, 10],
            "col4": [1, 13, 6, 15],
            "col5": [-4, 5, 6, -7],
        }
    )

    modin_df = from_pandas(pandas_df)

    by = [1, 2, 3, 2, 1]

    modin_groupby = modin_df.groupby(axis=1, by=by)
    pandas_groupby = pandas_df.groupby(axis=1, by=by)

    modin_groupby_equals_pandas(modin_groupby, pandas_groupby)
    eval_ngroups(modin_groupby, pandas_groupby)
    eval_skew(modin_groupby, pandas_groupby)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.ffill(), is_default=True)
    eval_general(
        modin_groupby,
        pandas_groupby,
        lambda df: df.sem(),
        modin_df_almost_equals_pandas,
        is_default=True,
    )
    eval_mean(modin_groupby, pandas_groupby)
    eval_any(modin_groupby, pandas_groupby)
    eval_min(modin_groupby, pandas_groupby)
    eval_ndim(modin_groupby, pandas_groupby)

    eval_general(modin_groupby, pandas_groupby, lambda df: df.idxmax(), is_default=True)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.idxmin(), is_default=True)
    eval_quantile(modin_groupby, pandas_groupby)

    # https://github.com/pandas-dev/pandas/issues/21127
    # eval_cumsum(modin_groupby, pandas_groupby)
    # eval_cummax(modin_groupby, pandas_groupby)
    # eval_cummin(modin_groupby, pandas_groupby)
    # eval_cumprod(modin_groupby, pandas_groupby)

    eval_general(
        modin_groupby,
        pandas_groupby,
        lambda df: df.pct_change(),
        modin_df_almost_equals_pandas,
        is_default=True,
    )
    apply_functions = [lambda df: -df, lambda df: df.sum(axis=1)]
    for func in apply_functions:
        eval_apply(modin_groupby, pandas_groupby, func)

    eval_general(modin_groupby, pandas_groupby, lambda df: df.first(), is_default=True)
    eval_general(
        modin_groupby, pandas_groupby, lambda df: df.backfill(), is_default=True
    )
    eval_general(modin_groupby, pandas_groupby, lambda df: df.bfill(), is_default=True)
    eval_prod(modin_groupby, pandas_groupby)
    eval_std(modin_groupby, pandas_groupby)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.last(), is_default=True)
    eval_general(
        modin_groupby,
        pandas_groupby,
        lambda df: df.mad(),
        modin_df_almost_equals_pandas,
        is_default=True,
    )
    eval_max(modin_groupby, pandas_groupby)
    eval_var(modin_groupby, pandas_groupby)
    eval_len(modin_groupby, pandas_groupby)
    eval_sum(modin_groupby, pandas_groupby)

    # Pandas fails on this case with ValueError
    # eval_ngroup(modin_groupby, pandas_groupby)
    # eval_nunique(modin_groupby, pandas_groupby)
    eval_median(modin_groupby, pandas_groupby)
    eval_general(
        modin_groupby,
        pandas_groupby,
        lambda df: df.cov(),
        modin_df_almost_equals_pandas,
        is_default=True,
    )

    transform_functions = [lambda df: df + 4, lambda df: -df - 10]
    for func in transform_functions:
        eval_transform(modin_groupby, pandas_groupby, func)

    pipe_functions = [lambda dfgb: dfgb.sum()]
    for func in pipe_functions:
        eval_pipe(modin_groupby, pandas_groupby, func)

    eval_general(
        modin_groupby,
        pandas_groupby,
        lambda df: df.corr(),
        modin_df_almost_equals_pandas,
        is_default=True,
    )
    eval_fillna(modin_groupby, pandas_groupby)
    eval_count(modin_groupby, pandas_groupby)
    eval_size(modin_groupby, pandas_groupby)
    eval_general(modin_groupby, pandas_groupby, lambda df: df.take(), is_default=True)
    eval_groups(modin_groupby, pandas_groupby)


@pytest.mark.parametrize(
    "by", [np.random.randint(0, 100, size=2 ** 8), lambda x: x % 3, None]
)
@pytest.mark.parametrize("as_index_series_or_dataframe", [0, 1, 2])
def test_series_groupby(by, as_index_series_or_dataframe):
    if as_index_series_or_dataframe <= 1:
        as_index = as_index_series_or_dataframe == 1
        series_data = np.random.randint(97, 198, size=2 ** 8)
        modin_series = pd.Series(series_data)
        pandas_series = pandas.Series(series_data)
    else:
        as_index = True
        pandas_series = pandas.DataFrame(
            {
                "col1": [0, 1, 2, 3],
                "col2": [4, 5, 6, 7],
                "col3": [3, 8, 12, 10],
                "col4": [17, 13, 16, 15],
                "col5": [-4, -5, -6, -7],
            }
        )
        modin_series = from_pandas(pandas_series)
        if isinstance(by, np.ndarray) or by is None:
            by = np.random.randint(0, 100, size=len(pandas_series.index))

    n = 1

    try:
        pandas_groupby = pandas_series.groupby(by, as_index=as_index)
        if as_index_series_or_dataframe == 2:
            pandas_groupby = pandas_groupby["col1"]
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.groupby(by, as_index=as_index)
    else:
        modin_groupby = modin_series.groupby(by, as_index=as_index)
        if as_index_series_or_dataframe == 2:
            modin_groupby = modin_groupby["col1"]

        modin_groupby_equals_pandas(modin_groupby, pandas_groupby)
        eval_ngroups(modin_groupby, pandas_groupby)
        eval_general(
            modin_groupby, pandas_groupby, lambda df: df.ffill(), is_default=True
        )
        eval_general(
            modin_groupby,
            pandas_groupby,
            lambda df: df.sem(),
            modin_df_almost_equals_pandas,
            is_default=True,
        )
        eval_mean(modin_groupby, pandas_groupby)
        eval_any(modin_groupby, pandas_groupby)
        eval_min(modin_groupby, pandas_groupby)
        eval_general(
            modin_groupby, pandas_groupby, lambda df: df.idxmax(), is_default=True
        )
        eval_ndim(modin_groupby, pandas_groupby)
        eval_cumsum(modin_groupby, pandas_groupby)
        eval_general(
            modin_groupby,
            pandas_groupby,
            lambda df: df.pct_change(),
            modin_df_almost_equals_pandas,
            is_default=True,
        )
        eval_cummax(modin_groupby, pandas_groupby)

        apply_functions = [lambda df: df.sum(), min]
        for func in apply_functions:
            eval_apply(modin_groupby, pandas_groupby, func)

        eval_general(
            modin_groupby, pandas_groupby, lambda df: df.first(), is_default=True
        )
        eval_general(
            modin_groupby, pandas_groupby, lambda df: df.backfill(), is_default=True
        )
        eval_cummin(modin_groupby, pandas_groupby)
        eval_general(
            modin_groupby, pandas_groupby, lambda df: df.bfill(), is_default=True
        )
        eval_general(
            modin_groupby, pandas_groupby, lambda df: df.idxmin(), is_default=True
        )
        eval_prod(modin_groupby, pandas_groupby)
        if as_index:
            eval_std(modin_groupby, pandas_groupby)
            eval_var(modin_groupby, pandas_groupby)
            eval_skew(modin_groupby, pandas_groupby)

        agg_functions = ["min", "max"]
        for func in agg_functions:
            eval_agg(modin_groupby, pandas_groupby, func)
            eval_aggregate(modin_groupby, pandas_groupby, func)

        eval_general(
            modin_groupby, pandas_groupby, lambda df: df.last(), is_default=True
        )
        eval_general(
            modin_groupby,
            pandas_groupby,
            lambda df: df.mad(),
            modin_df_almost_equals_pandas,
            is_default=True,
        )
        eval_rank(modin_groupby, pandas_groupby)
        eval_max(modin_groupby, pandas_groupby)
        eval_len(modin_groupby, pandas_groupby)
        eval_sum(modin_groupby, pandas_groupby)
        eval_ngroup(modin_groupby, pandas_groupby)
        eval_nunique(modin_groupby, pandas_groupby)
        eval_median(modin_groupby, pandas_groupby)
        eval_general(
            modin_groupby, pandas_groupby, lambda df: df.head(n), is_default=True
        )
        eval_cumprod(modin_groupby, pandas_groupby)

        transform_functions = [lambda df: df + 4, lambda df: -df - 10]
        for func in transform_functions:
            eval_transform(modin_groupby, pandas_groupby, func)

        pipe_functions = [lambda dfgb: dfgb.sum()]
        for func in pipe_functions:
            eval_pipe(modin_groupby, pandas_groupby, func)

        eval_fillna(modin_groupby, pandas_groupby)
        eval_count(modin_groupby, pandas_groupby)
        eval_general(
            modin_groupby, pandas_groupby, lambda df: df.tail(n), is_default=True
        )
        eval_quantile(modin_groupby, pandas_groupby)
        eval_general(
            modin_groupby, pandas_groupby, lambda df: df.take(), is_default=True
        )
        eval_groups(modin_groupby, pandas_groupby)


def test_multi_column_groupby():
    pandas_df = pandas.DataFrame(
        {
            "col1": np.random.randint(0, 100, size=1000),
            "col2": np.random.randint(0, 100, size=1000),
            "col3": np.random.randint(0, 100, size=1000),
            "col4": np.random.randint(0, 100, size=1000),
            "col5": np.random.randint(0, 100, size=1000),
        },
        index=["row{}".format(i) for i in range(1000)],
    )

    modin_df = from_pandas(pandas_df)
    by = ["col1", "col2"]

    df_equals(modin_df.groupby(by).count(), pandas_df.groupby(by).count())

    with pytest.warns(UserWarning):
        for k, _ in modin_df.groupby(by):
            assert isinstance(k, tuple)

    by = ["row0", "row1"]
    with pytest.raises(KeyError):
        modin_df.groupby(by, axis=1).count()


def eval_ngroups(modin_groupby, pandas_groupby):
    assert modin_groupby.ngroups == pandas_groupby.ngroups


def eval_skew(modin_groupby, pandas_groupby):
    modin_df_almost_equals_pandas(modin_groupby.skew(), pandas_groupby.skew())


def eval_mean(modin_groupby, pandas_groupby):
    modin_df_almost_equals_pandas(modin_groupby.mean(), pandas_groupby.mean())


def eval_any(modin_groupby, pandas_groupby):
    df_equals(modin_groupby.any(), pandas_groupby.any())


def eval_min(modin_groupby, pandas_groupby):
    df_equals(modin_groupby.min(), pandas_groupby.min())


def eval_ndim(modin_groupby, pandas_groupby):
    assert modin_groupby.ndim == pandas_groupby.ndim


def eval_cumsum(modin_groupby, pandas_groupby, axis=0):
    df_equals(modin_groupby.cumsum(axis=axis), pandas_groupby.cumsum(axis=axis))


def eval_cummax(modin_groupby, pandas_groupby, axis=0):
    df_equals(modin_groupby.cummax(axis=axis), pandas_groupby.cummax(axis=axis))


def eval_apply(modin_groupby, pandas_groupby, func):
    df_equals(modin_groupby.apply(func), pandas_groupby.apply(func))


def eval_dtypes(modin_groupby, pandas_groupby):
    df_equals(modin_groupby.dtypes, pandas_groupby.dtypes)


def eval_cummin(modin_groupby, pandas_groupby, axis=0):
    df_equals(modin_groupby.cummin(axis=axis), pandas_groupby.cummin(axis=axis))


def eval_prod(modin_groupby, pandas_groupby):
    df_equals(modin_groupby.prod(), pandas_groupby.prod())


def eval_std(modin_groupby, pandas_groupby):
    modin_df_almost_equals_pandas(modin_groupby.std(), pandas_groupby.std())


def eval_aggregate(modin_groupby, pandas_groupby, func):
    df_equals(modin_groupby.aggregate(func), pandas_groupby.aggregate(func))


def eval_agg(modin_groupby, pandas_groupby, func):
    df_equals(modin_groupby.agg(func), pandas_groupby.agg(func))


def eval_rank(modin_groupby, pandas_groupby):
    df_equals(modin_groupby.rank(), pandas_groupby.rank())


def eval_max(modin_groupby, pandas_groupby):
    df_equals(modin_groupby.max(), pandas_groupby.max())


def eval_var(modin_groupby, pandas_groupby):
    modin_df_almost_equals_pandas(modin_groupby.var(), pandas_groupby.var())


def eval_len(modin_groupby, pandas_groupby):
    assert len(modin_groupby) == len(pandas_groupby)


def eval_sum(modin_groupby, pandas_groupby):
    df_equals(modin_groupby.sum(), pandas_groupby.sum())


def eval_ngroup(modin_groupby, pandas_groupby):
    df_equals(modin_groupby.ngroup(), pandas_groupby.ngroup())


def eval_nunique(modin_groupby, pandas_groupby):
    df_equals(modin_groupby.nunique(), pandas_groupby.nunique())


def eval_median(modin_groupby, pandas_groupby):
    modin_df_almost_equals_pandas(modin_groupby.median(), pandas_groupby.median())


def eval_cumprod(modin_groupby, pandas_groupby, axis=0):
    df_equals(modin_groupby.cumprod(), pandas_groupby.cumprod())
    df_equals(modin_groupby.cumprod(axis=axis), pandas_groupby.cumprod(axis=axis))


def eval_transform(modin_groupby, pandas_groupby, func):
    df_equals(modin_groupby.transform(func), pandas_groupby.transform(func))


def eval_fillna(modin_groupby, pandas_groupby):
    df_equals(
        modin_groupby.fillna(method="ffill"), pandas_groupby.fillna(method="ffill")
    )


def eval_count(modin_groupby, pandas_groupby):
    df_equals(modin_groupby.count(), pandas_groupby.count())


def eval_size(modin_groupby, pandas_groupby):
    df_equals(modin_groupby.size(), pandas_groupby.size())


def eval_pipe(modin_groupby, pandas_groupby, func):
    df_equals(modin_groupby.pipe(func), pandas_groupby.pipe(func))


def eval_quantile(modin_groupby, pandas_groupby):
    try:
        pandas_result = pandas_groupby.quantile(q=0.4)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_groupby.quantile(q=0.4)
    else:
        df_equals(modin_groupby.quantile(q=0.4), pandas_result)


def eval___getattr__(modin_groupby, pandas_groupby, item):
    try:
        pandas_groupby = pandas_groupby[item]
        pandas_result = pandas_groupby.count()
    except Exception as e:
        with pytest.raises(type(e)):
            modin_groupby[item].count()
    else:
        modin_groupby = modin_groupby[item]
        modin_result = modin_groupby.count()
        df_equals(modin_result, pandas_result)


def eval_groups(modin_groupby, pandas_groupby):
    for k, v in modin_groupby.groups.items():
        assert v.equals(pandas_groupby.groups[k])


def eval_shift(modin_groupby, pandas_groupby):
    assert modin_groupby.groups == pandas_groupby.groups


def test_groupby_on_index_values_with_loop():
    length = 2 ** 6
    data = {
        "a": np.random.randint(0, 100, size=length),
        "b": np.random.randint(0, 100, size=length),
        "c": np.random.randint(0, 100, size=length),
    }
    idx = ["g1" if i % 3 != 0 else "g2" for i in range(length)]
    modin_df = pd.DataFrame(data, index=idx, columns=list("aba"))
    pandas_df = pandas.DataFrame(data, index=idx, columns=list("aba"))
    modin_groupby_obj = modin_df.groupby(modin_df.index)
    pandas_groupby_obj = pandas_df.groupby(pandas_df.index)

    modin_dict = {k: v for k, v in modin_groupby_obj}
    pandas_dict = {k: v for k, v in pandas_groupby_obj}

    for k in modin_dict:
        df_equals(modin_dict[k], pandas_dict[k])

    modin_groupby_obj = modin_df.groupby(modin_df.columns, axis=1)
    pandas_groupby_obj = pandas_df.groupby(pandas_df.columns, axis=1)

    modin_dict = {k: v for k, v in modin_groupby_obj}
    pandas_dict = {k: v for k, v in pandas_groupby_obj}

    for k in modin_dict:
        df_equals(modin_dict[k], pandas_dict[k])


def test_groupby_multiindex():
    frame_data = np.random.randint(0, 100, size=(2 ** 6, 2 ** 4))
    modin_df = pd.DataFrame(frame_data)
    pandas_df = pandas.DataFrame(frame_data)

    new_columns = pandas.MultiIndex.from_tuples(
        [(i // 4, i // 2, i) for i in modin_df.columns], names=["four", "two", "one"]
    )
    modin_df.columns = new_columns
    pandas_df.columns = new_columns
    modin_df.groupby(level=1, axis=1).sum()

    modin_df = modin_df.T
    pandas_df = pandas_df.T
    df_equals(modin_df.groupby(level=1).count(), pandas_df.groupby(level=1).count())
    df_equals(modin_df.groupby(by="four").count(), pandas_df.groupby(by="four").count())

    by = ["one", "two"]
    df_equals(modin_df.groupby(by=by).count(), pandas_df.groupby(by=by).count())


def test_agg_func_None_rename():
    pandas_df = pandas.DataFrame(
        {
            "col1": np.random.randint(0, 100, size=1000),
            "col2": np.random.randint(0, 100, size=1000),
            "col3": np.random.randint(0, 100, size=1000),
            "col4": np.random.randint(0, 100, size=1000),
        },
        index=["row{}".format(i) for i in range(1000)],
    )
    modin_df = from_pandas(pandas_df)

    modin_result = modin_df.groupby(["col1", "col2"]).agg(
        max=("col3", np.max), min=("col3", np.min)
    )
    pandas_result = pandas_df.groupby(["col1", "col2"]).agg(
        max=("col3", np.max), min=("col3", np.min)
    )
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "operation", ["quantile", "mean", "sum", "median", "unique", "cumprod"]
)
def test_agg_exceptions(operation):
    N = 256
    fill_data = [
        ("nan_column", [None, np.datetime64("2010")] * (N // 2)),
        (
            "date_column",
            [
                np.datetime64("2010"),
                np.datetime64("2011"),
                np.datetime64("2011-06-15T00:00"),
                np.datetime64("2009-01-01"),
            ]
            * (N // 4),
        ),
    ]

    data1 = {
        "column_to_by": ["foo", "bar", "baz", "bar"] * (N // 4),
        "nan_column": [None] * N,
    }

    data2 = {
        f"{key}{i}": value
        for key, value in fill_data
        for i in range(N // len(fill_data))
    }

    data = {**data1, **data2}

    eval_aggregation(*create_test_dfs(data), operation=operation)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "Max": ("cnt", np.max),
            "Sum": ("cnt", np.sum),
            "Num": ("c", pd.Series.nunique),
            "Num1": ("c", pandas.Series.nunique),
        },
        {
            "func": {
                "Max": ("cnt", np.max),
                "Sum": ("cnt", np.sum),
                "Num": ("c", pd.Series.nunique),
                "Num1": ("c", pandas.Series.nunique),
            }
        },
    ],
)
def test_to_pandas_convertion(kwargs):
    data = {"a": [1, 2], "b": [3, 4], "c": [5, 6]}
    by = ["a", "b"]

    eval_aggregation(*create_test_dfs(data), by=by, **kwargs)


@pytest.mark.parametrize(
    # When True, do df[name], otherwise just use name
    "columns",
    [
        [(False, "a"), (False, "b"), (False, "c")],
        [(False, "a"), (False, "b")],
        [(True, "a"), (True, "b"), (True, "c")],
        [(True, "a"), (True, "b")],
        [(False, "a"), (False, "b"), (True, "c")],
        [(False, "a"), (True, "c")],
    ],
)
def test_mixed_columns(columns):
    def get_columns(df):
        return [df[name] if lookup else name for (lookup, name) in columns]

    data = {"a": [1, 1, 2], "b": [11, 11, 22], "c": [111, 111, 222]}

    df1 = pandas.DataFrame(data)
    df1 = pandas.concat([df1])
    ref = df1.groupby(get_columns(df1)).size()

    df2 = pd.DataFrame(data)
    df2 = pd.concat([df2])
    exp = df2.groupby(get_columns(df2)).size()
    df_equals(ref, exp)


@pytest.mark.parametrize(
    # When True, use (df[name] + 1), otherwise just use name
    "columns",
    [
        [(True, "a"), (True, "b"), (True, "c")],
        [(True, "a"), (True, "b")],
        [(False, "a"), (False, "b"), (True, "c")],
        [(False, "a"), (True, "c")],
    ],
)
def test_mixed_columns_not_from_df(columns):
    """
    Unlike the previous test, in this case the Series is not just a column from
    the original DataFrame, so you can't use a fasttrack.
    """

    def get_columns(df):
        return [(df[name] + 1) if lookup else name for (lookup, name) in columns]

    data = {"a": [1, 1, 2], "b": [11, 11, 22], "c": [111, 111, 222]}

    df1 = pandas.DataFrame(data)
    df1 = pandas.concat([df1])
    ref = df1.groupby(get_columns(df1)).size()

    df2 = pd.DataFrame(data)
    df2 = pd.concat([df2])
    exp = df2.groupby(get_columns(df2)).size()
    df_equals(ref, exp)


@pytest.mark.parametrize(
    # When True, do df[obj], otherwise just use the obj
    "columns",
    [
        [(False, "a")],
        [(False, "a"), (False, "b"), (False, "c")],
        [(False, "a"), (False, "b")],
        [(False, "b"), (False, "a")],
        [(True, "a"), (True, "b"), (True, "c")],
        [(True, "a"), (True, "b")],
        [(False, "a"), (False, "b"), (True, "c")],
        [(False, "a"), (True, "c")],
        [(False, "a"), (False, pd.Series([5, 6, 7, 8]))],
    ],
)
def test_unknown_groupby(columns):
    def get_columns(df):
        return [df[name] if lookup else name for (lookup, name) in columns]

    data = {"b": [11, 11, 22, 200], "c": [111, 111, 222, 7000]}
    modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)

    with pytest.raises(KeyError):
        pandas_df.groupby(by=get_columns(pandas_df))
    with pytest.raises(KeyError):
        modin_df.groupby(by=get_columns(modin_df))


@pytest.mark.parametrize(
    "func_to_apply",
    [
        lambda df: df.sum(),
        lambda df: df.count(),
        lambda df: df.size(),
        lambda df: df.mean(),
        lambda df: df.quantile(),
    ],
)
def test_multi_column_groupby_different_partitions(func_to_apply):
    data = test_data_values[0]
    md_df, pd_df = create_test_dfs(data)

    # columns that will be located in a different partitions
    by = [pd_df.columns[0], pd_df.columns[-1]]

    md_grp, pd_grp = md_df.groupby(by), pd_df.groupby(by)
    eval_general(md_grp, pd_grp, func_to_apply)
