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
from modin.utils import try_cast_to_pandas, get_current_backend
from modin.pandas.utils import from_pandas, is_scalar
from .utils import (
    df_equals,
    check_df_columns_have_nans,
    create_test_dfs,
    eval_general,
    test_data,
    test_data_values,
    modin_df_almost_equals_pandas,
    generate_multiindex,
)

pd.DEFAULT_NPARTITIONS = 4


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


def build_types_asserter(comparator):
    def wrapper(obj1, obj2, *args, **kwargs):
        error_str = f"obj1 and obj2 has incorrect types: {type(obj1)} and {type(obj2)}"
        assert not (is_scalar(obj1) ^ is_scalar(obj2)), error_str
        assert obj1.__module__.split(".")[0] == "modin", error_str
        assert obj2.__module__.split(".")[0] == "pandas", error_str
        comparator(obj1, obj2, *args, **kwargs)

    return wrapper


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
        eval_shift(modin_groupby, pandas_groupby)
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

        agg_functions = [
            lambda df: df.sum(),
            "min",
            min,
            "max",
            max,
            sum,
            {"col2": "sum"},
            {"col2": sum},
            {"col2": "max", "col4": "sum", "col5": "min"},
            {"col2": max, "col4": sum, "col5": "min"},
        ]
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
        ["col2"],  # 5
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
        ["col5"],  # 10
        ["col1", "col5"],
        ["col5", "col4"],
        ["col4", "col5"],
        ["col5", "col4", "col1"],
        ["col1", pd.Series([1, 5, 7, 8])],  # 15
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
        [GetColumn("col1")],  # 20
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
    eval_shift(modin_groupby, pandas_groupby)
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
    apply_functions = [
        lambda df: df.sum(),
        lambda df: pandas.Series([1, 2, 3, 4], name="result"),
        min,
    ]
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

    agg_functions = [lambda df: df.sum(), "min", "max", min, sum]
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
    # Pandas raising exception when 'by' contains categorical key and `as_index=False`
    # because of a bug: https://github.com/pandas-dev/pandas/issues/36698
    # Modin correctly processes the result, so that's why `check_exception_type=None` in some cases
    eval_general(
        modin_groupby,
        pandas_groupby,
        lambda df: df.nunique(),
        check_exception_type=None if (col1_category and not as_index) else True,
    )
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
    eval_shift(modin_groupby, pandas_groupby)
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

    agg_functions = [
        lambda df: df.sum(),
        "min",
        "max",
        max,
        sum,
        {"col2": "sum"},
        {"col2": "max", "col4": "sum", "col5": "min"},
    ]
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
    eval_shift(modin_groupby, pandas_groupby)
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

    agg_functions = [
        lambda df: df.sum(),
        "min",
        "max",
        min,
        sum,
        {"A": "sum"},
        {"A": lambda df: df.sum()},
        {"A": "max", "B": "sum", "C": "min"},
    ]
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
    eval_shift(modin_groupby, pandas_groupby)
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
        eval_shift(modin_groupby, pandas_groupby)
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

        agg_functions = [lambda df: df.sum(), "min", "max", max, sum]
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
    eval_general(
        modin_groupby,
        pandas_groupby,
        lambda grp: grp[item].count(),
        comparator=build_types_asserter(df_equals),
    )
    eval_general(
        modin_groupby,
        pandas_groupby,
        lambda grp: getattr(grp, item).count(),
        comparator=build_types_asserter(df_equals),
    )


def eval_groups(modin_groupby, pandas_groupby):
    for k, v in modin_groupby.groups.items():
        assert v.equals(pandas_groupby.groups[k])


def eval_shift(modin_groupby, pandas_groupby):
    eval_general(
        modin_groupby,
        pandas_groupby,
        lambda groupby: groupby.shift(),
    )
    eval_general(
        modin_groupby,
        pandas_groupby,
        lambda groupby: groupby.shift(periods=0),
    )
    eval_general(
        modin_groupby,
        pandas_groupby,
        lambda groupby: groupby.shift(periods=-3),
    )
    eval_general(
        modin_groupby,
        pandas_groupby,
        lambda groupby: groupby.shift(axis=1, fill_value=777),
    )


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


@pytest.mark.parametrize("groupby_axis", [0, 1])
@pytest.mark.parametrize("shift_axis", [0, 1])
def test_shift_freq(groupby_axis, shift_axis):
    pandas_df = pandas.DataFrame(
        {
            "col1": [1, 0, 2, 3],
            "col2": [4, 5, np.NaN, 7],
            "col3": [np.NaN, np.NaN, 12, 10],
            "col4": [17, 13, 16, 15],
        }
    )
    modin_df = from_pandas(pandas_df)

    new_index = pandas.date_range("1/12/2020", periods=4, freq="S")
    if groupby_axis == 0 and shift_axis == 0:
        pandas_df.index = modin_df.index = new_index
        by = [["col2", "col3"], ["col2"], ["col4"], [0, 1, 0, 2]]
    else:
        pandas_df.index = modin_df.index = new_index
        pandas_df.columns = modin_df.columns = new_index
        by = [[0, 1, 0, 2]]

    for _by in by:
        pandas_groupby = pandas_df.groupby(by=_by, axis=groupby_axis)
        modin_groupby = modin_df.groupby(by=_by, axis=groupby_axis)
        eval_general(
            modin_groupby,
            pandas_groupby,
            lambda groupby: groupby.shift(axis=shift_axis, freq="S"),
        )


@pytest.mark.parametrize(
    "by_and_agg_dict",
    [
        {
            "by": [
                list(test_data["int_data"].keys())[0],
                list(test_data["int_data"].keys())[1],
            ],
            "agg_dict": {
                "max": (list(test_data["int_data"].keys())[2], np.max),
                "min": (list(test_data["int_data"].keys())[2], np.min),
            },
        },
        {
            "by": ["col1"],
            "agg_dict": {
                "max": (list(test_data["int_data"].keys())[0], np.max),
                "min": (list(test_data["int_data"].keys())[-1], np.min),
            },
        },
        {
            "by": [
                list(test_data["int_data"].keys())[0],
                list(test_data["int_data"].keys())[-1],
            ],
            "agg_dict": {
                "max": (list(test_data["int_data"].keys())[1], max),
                "min": (list(test_data["int_data"].keys())[-2], min),
            },
        },
        pytest.param(
            {
                "by": [
                    list(test_data["int_data"].keys())[0],
                    list(test_data["int_data"].keys())[-1],
                ],
                "agg_dict": {
                    "max": (list(test_data["int_data"].keys())[1], max),
                    "min": (list(test_data["int_data"].keys())[-1], min),
                },
            },
            marks=pytest.mark.skip("See Modin issue #2542"),
        ),
    ],
)
@pytest.mark.parametrize("as_index", [True, False])
def test_agg_func_None_rename(by_and_agg_dict, as_index):
    modin_df, pandas_df = create_test_dfs(test_data["int_data"])

    modin_result = modin_df.groupby(by_and_agg_dict["by"], as_index=as_index).agg(
        **by_and_agg_dict["agg_dict"]
    )
    pandas_result = pandas_df.groupby(by_and_agg_dict["by"], as_index=as_index).agg(
        **by_and_agg_dict["agg_dict"]
    )
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "as_index",
    [
        True,
        pytest.param(
            False,
            marks=pytest.mark.xfail_backends(
                ["BaseOnPython"], reason="See Pandas issue #39103"
            ),
        ),
    ],
)
@pytest.mark.parametrize("by_length", [1, 3])
@pytest.mark.parametrize(
    "agg_fns",
    [["sum", "min", "max"], ["mean", "quantile"]],
    ids=["reduction", "aggregation"],
)
def test_dict_agg_rename_mi_columns(as_index, by_length, agg_fns):
    md_df, pd_df = create_test_dfs(test_data["int_data"])
    mi_columns = generate_multiindex(len(md_df.columns), nlevels=4)

    md_df.columns, pd_df.columns = mi_columns, mi_columns

    by = list(md_df.columns[:by_length])
    agg_cols = list(md_df.columns[by_length : by_length + 3])

    agg_dict = {
        f"custom-{i}" + str(agg_fns[i % len(agg_fns)]): (col, agg_fns[i % len(agg_fns)])
        for i, col in enumerate(agg_cols)
    }

    md_res = md_df.groupby(by, as_index=as_index).agg(**agg_dict)
    pd_res = md_df.groupby(by, as_index=as_index).agg(**agg_dict)

    df_equals(md_res, pd_res)


@pytest.mark.parametrize(
    "operation",
    [
        "quantile",
        "mean",
        pytest.param(
            "sum", marks=pytest.mark.skip("See Modin issue #2255 for details")
        ),
        "median",
        "unique",
        "cumprod",
    ],
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
        [(False, "a"), (True, "c"), (False, [1, 1, 2])],
    ],
)
@pytest.mark.parametrize("as_index", [True, False])
def test_mixed_columns_not_from_df(columns, as_index):
    """
    Unlike the previous test, in this case the Series is not just a column from
    the original DataFrame, so you can't use a fasttrack.
    """

    def get_columns(df):
        return [(df[name] + 1) if lookup else name for (lookup, name) in columns]

    data = {"a": [1, 1, 2], "b": [11, 11, 22], "c": [111, 111, 222]}
    groupby_kw = {"as_index": as_index}

    md_df, pd_df = create_test_dfs(data)
    by_md, by_pd = map(get_columns, [md_df, pd_df])

    pd_grp = pd_df.groupby(by_pd, **groupby_kw)
    md_grp = md_df.groupby(by_md, **groupby_kw)

    modin_groupby_equals_pandas(md_grp, pd_grp)
    eval_general(md_grp, pd_grp, lambda grp: grp.size())
    eval_general(md_grp, pd_grp, lambda grp: grp.apply(lambda df: df.sum()))
    eval_general(md_grp, pd_grp, lambda grp: grp.first())


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
        lambda df: df.size(),
        lambda df: df.quantile(),
        lambda df: df.dtypes,
        lambda df: df.apply(lambda df: df.sum()),
        pytest.param(
            lambda df: df.apply(lambda df: pandas.Series([1, 2, 3, 4])),
            marks=pytest.mark.skip("See modin issue #2511"),
        ),
        lambda grp: grp.agg(
            {
                list(test_data_values[0].keys())[1]: (max, min, sum),
                list(test_data_values[0].keys())[-2]: (sum, min, max),
            }
        ),
        lambda grp: grp.agg(
            {
                list(test_data_values[0].keys())[1]: [
                    ("new_sum", "sum"),
                    ("new_min", "min"),
                ],
                list(test_data_values[0].keys())[-2]: np.sum,
            }
        ),
        pytest.param(
            lambda grp: grp.agg(
                {
                    list(test_data_values[0].keys())[1]: (max, min, sum),
                    list(test_data_values[0].keys())[-1]: (sum, min, max),
                }
            ),
            marks=pytest.mark.skip("See modin issue #2542"),
        ),
    ],
)
@pytest.mark.parametrize("as_index", [True, False])
@pytest.mark.parametrize("by_length", [1, 2])
@pytest.mark.parametrize(
    "categorical_by",
    [pytest.param(True, marks=pytest.mark.skip("See modin issue #2513")), False],
)
def test_multi_column_groupby_different_partitions(
    func_to_apply, as_index, by_length, categorical_by
):
    data = test_data_values[0]
    md_df, pd_df = create_test_dfs(data)

    by = [pd_df.columns[-i if i % 2 else i] for i in range(by_length)]

    if categorical_by:
        md_df = md_df.astype({by[0]: "category"})
        pd_df = pd_df.astype({by[0]: "category"})

    md_grp, pd_grp = md_df.groupby(by, as_index=as_index), pd_df.groupby(
        by, as_index=as_index
    )
    eval_general(md_grp, pd_grp, func_to_apply)


@pytest.mark.parametrize(
    "by",
    [
        0,
        1.5,
        "str",
        pandas.Timestamp("2020-02-02"),
        [None],
        [0, "str"],
        [None, 0],
        [pandas.Timestamp("2020-02-02"), 1.5],
    ],
)
@pytest.mark.parametrize("as_index", [True, False])
def test_not_str_by(by, as_index):
    data = {f"col{i}": np.arange(5) for i in range(5)}
    columns = pandas.Index([0, 1.5, "str", pandas.Timestamp("2020-02-02"), None])

    md_df, pd_df = create_test_dfs(data, columns=columns)
    md_grp, pd_grp = md_df.groupby(by, as_index=as_index), pd_df.groupby(
        by, as_index=as_index
    )

    modin_groupby_equals_pandas(md_grp, pd_grp)
    df_equals(md_grp.sum(), pd_grp.sum())
    df_equals(md_grp.size(), pd_grp.size())
    df_equals(md_grp.agg(lambda df: df.mean()), pd_grp.agg(lambda df: df.mean()))
    df_equals(md_grp.dtypes, pd_grp.dtypes)
    df_equals(md_grp.first(), pd_grp.first())
