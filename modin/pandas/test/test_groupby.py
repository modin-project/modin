import pytest
import pandas
import numpy as np
import modin.pandas as pd
from modin.pandas.utils import from_pandas, to_pandas

pd.DEFAULT_NPARTITIONS = 4


def ray_df_equals_pandas(ray_df, pandas_df):
    assert isinstance(ray_df, pd.DataFrame)
    # Order may not match here, but pandas behavior can change, so we will be consistent
    # ourselves in keeping the columns in the order they were in before the groupby
    try:
        assert (
            to_pandas(ray_df).equals(pandas_df)
            or (all(ray_df.isna().all()) and all(pandas_df.isna().all()))
            or to_pandas(ray_df)[list(pandas_df.columns)].equals(pandas_df)
        )
    # Pandas does not seem to be consistent with the way that it handles as_index.
    # Because the behavior is determined to be non-deterministic we will at least check
    # that everything else matches if we drop that column from the pandas side.
    except KeyError:
        assert to_pandas(ray_df).equals(pandas_df.drop(columns=[ray_df.index.name]))


def ray_df_almost_equals_pandas(ray_df, pandas_df):
    assert isinstance(ray_df, pd.DataFrame)
    difference = to_pandas(ray_df) - pandas_df
    diff_max = difference.max().max()
    assert (
        to_pandas(ray_df).equals(pandas_df)
        or diff_max < 0.0001
        or (all(ray_df.isna().all()) and all(pandas_df.isna().all()))
    )


def ray_series_equals_pandas(ray_df, pandas_df):
    assert ray_df.equals(pandas_df)


def ray_df_equals(ray_df1, ray_df2):
    assert to_pandas(ray_df1).equals(to_pandas(ray_df2))


def ray_groupby_equals_pandas(ray_groupby, pandas_groupby):
    for g1, g2 in zip(ray_groupby, pandas_groupby):
        assert g1[0] == g2[0]
        ray_df_equals_pandas(g1[1], g2[1])


def test_mixed_dtypes_groupby():
    frame_data = np.random.randint(97, 198, size=(2 ** 6, 2 ** 4))
    pandas_df = pandas.DataFrame(frame_data).add_prefix("col")
    # Convert every other column to string
    for col in pandas_df.iloc[
        :, [i for i in range(len(pandas_df.columns)) if i % 2 == 0]
    ]:
        pandas_df[col] = [str(chr(i)) for i in pandas_df[col]]
    ray_df = from_pandas(pandas_df)

    n = 1

    by_values = ["col1", lambda x: x % 2, pandas_df.col1]

    for by in by_values:
        ray_groupby = ray_df.groupby(by=by)
        pandas_groupby = pandas_df.groupby(by=by)

        ray_groupby_equals_pandas(ray_groupby, pandas_groupby)
        eval_ngroups(ray_groupby, pandas_groupby)
        eval_skew(ray_groupby, pandas_groupby)
        eval_ffill(ray_groupby, pandas_groupby)
        eval_sem(ray_groupby, pandas_groupby)
        eval_mean(ray_groupby, pandas_groupby)
        eval_any(ray_groupby, pandas_groupby)
        eval_min(ray_groupby, pandas_groupby)
        eval_idxmax(ray_groupby, pandas_groupby)
        eval_ndim(ray_groupby, pandas_groupby)
        eval_cumsum(ray_groupby, pandas_groupby)
        eval_pct_change(ray_groupby, pandas_groupby)
        eval_cummax(ray_groupby, pandas_groupby)

        # TODO Add more apply functions
        apply_functions = [lambda df: df.sum(), min]
        for func in apply_functions:
            eval_apply(ray_groupby, pandas_groupby, func)

        eval_dtypes(ray_groupby, pandas_groupby)
        eval_first(ray_groupby, pandas_groupby)
        eval_backfill(ray_groupby, pandas_groupby)
        eval_cummin(ray_groupby, pandas_groupby)
        eval_bfill(ray_groupby, pandas_groupby)
        eval_idxmin(ray_groupby, pandas_groupby)
        eval_prod(ray_groupby, pandas_groupby)
        eval_std(ray_groupby, pandas_groupby)

        agg_functions = ["min", "max"]
        for func in agg_functions:
            eval_agg(ray_groupby, pandas_groupby, func)
            eval_aggregate(ray_groupby, pandas_groupby, func)

        eval_last(ray_groupby, pandas_groupby)
        eval_mad(ray_groupby, pandas_groupby)
        eval_max(ray_groupby, pandas_groupby)
        eval_var(ray_groupby, pandas_groupby)
        eval_len(ray_groupby, pandas_groupby)
        eval_sum(ray_groupby, pandas_groupby)
        eval_ngroup(ray_groupby, pandas_groupby)
        eval_nunique(ray_groupby, pandas_groupby)
        eval_median(ray_groupby, pandas_groupby)
        eval_head(ray_groupby, pandas_groupby, n)
        eval_cumprod(ray_groupby, pandas_groupby)
        eval_cov(ray_groupby, pandas_groupby)

        transform_functions = [lambda df: df + 4, lambda df: -df - 10]
        for func in transform_functions:
            eval_transform(ray_groupby, pandas_groupby, func)

        pipe_functions = [lambda dfgb: dfgb.sum()]
        for func in pipe_functions:
            eval_pipe(ray_groupby, pandas_groupby, func)

        eval_corr(ray_groupby, pandas_groupby)
        eval_fillna(ray_groupby, pandas_groupby)
        eval_count(ray_groupby, pandas_groupby)
        eval_tail(ray_groupby, pandas_groupby, n)
        eval_quantile(ray_groupby, pandas_groupby)
        eval_take(ray_groupby, pandas_groupby)
        eval___getattr__(ray_groupby, pandas_groupby)
        eval_groups(ray_groupby, pandas_groupby)


def test_simple_row_groupby():
    pandas_df = pandas.DataFrame(
        {
            "col1": [0, 1, 2, 3],
            "col2": [4, 5, 6, 7],
            "col3": [3, 8, 12, 10],
            "col4": [17, 13, 16, 15],
            "col5": [-4, -5, -6, -7],
        }
    )

    ray_df = from_pandas(pandas_df)

    by_values = [[1, 2, 1, 2], lambda x: x % 3, pandas_df.col1]
    n = 1
    for by in by_values:
        ray_groupby = ray_df.groupby(by=by)
        pandas_groupby = pandas_df.groupby(by=by)

        ray_groupby_equals_pandas(ray_groupby, pandas_groupby)
        eval_ngroups(ray_groupby, pandas_groupby)
        eval_skew(ray_groupby, pandas_groupby)
        eval_ffill(ray_groupby, pandas_groupby)
        eval_sem(ray_groupby, pandas_groupby)
        eval_mean(ray_groupby, pandas_groupby)
        eval_any(ray_groupby, pandas_groupby)
        eval_min(ray_groupby, pandas_groupby)
        eval_idxmax(ray_groupby, pandas_groupby)
        eval_ndim(ray_groupby, pandas_groupby)
        eval_cumsum(ray_groupby, pandas_groupby)
        eval_pct_change(ray_groupby, pandas_groupby)
        eval_cummax(ray_groupby, pandas_groupby)

        apply_functions = [lambda df: df.sum(), lambda df: -df]
        for func in apply_functions:
            eval_apply(ray_groupby, pandas_groupby, func)

        eval_dtypes(ray_groupby, pandas_groupby)
        eval_first(ray_groupby, pandas_groupby)
        eval_backfill(ray_groupby, pandas_groupby)
        eval_cummin(ray_groupby, pandas_groupby)
        eval_bfill(ray_groupby, pandas_groupby)
        eval_idxmin(ray_groupby, pandas_groupby)
        eval_prod(ray_groupby, pandas_groupby)
        eval_std(ray_groupby, pandas_groupby)

        agg_functions = ["min", "max"]
        for func in agg_functions:
            eval_agg(ray_groupby, pandas_groupby, func)
            eval_aggregate(ray_groupby, pandas_groupby, func)

        eval_last(ray_groupby, pandas_groupby)
        eval_mad(ray_groupby, pandas_groupby)
        eval_rank(ray_groupby, pandas_groupby)
        eval_max(ray_groupby, pandas_groupby)
        eval_var(ray_groupby, pandas_groupby)
        eval_len(ray_groupby, pandas_groupby)
        eval_sum(ray_groupby, pandas_groupby)
        eval_ngroup(ray_groupby, pandas_groupby)
        eval_nunique(ray_groupby, pandas_groupby)
        eval_median(ray_groupby, pandas_groupby)
        eval_head(ray_groupby, pandas_groupby, n)
        eval_cumprod(ray_groupby, pandas_groupby)
        eval_cov(ray_groupby, pandas_groupby)

        transform_functions = [lambda df: df + 4, lambda df: -df - 10]
        for func in transform_functions:
            eval_transform(ray_groupby, pandas_groupby, func)

        pipe_functions = [lambda dfgb: dfgb.sum()]
        for func in pipe_functions:
            eval_pipe(ray_groupby, pandas_groupby, func)

        eval_corr(ray_groupby, pandas_groupby)
        eval_fillna(ray_groupby, pandas_groupby)
        eval_count(ray_groupby, pandas_groupby)
        eval_tail(ray_groupby, pandas_groupby, n)
        eval_quantile(ray_groupby, pandas_groupby)
        eval_take(ray_groupby, pandas_groupby)
        eval___getattr__(ray_groupby, pandas_groupby)
        eval_groups(ray_groupby, pandas_groupby)


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

    ray_df = from_pandas(pandas_df)

    by = ["1", "1", "1", "1"]
    n = 6

    ray_groupby = ray_df.groupby(by=by)
    pandas_groupby = pandas_df.groupby(by=by)

    ray_groupby_equals_pandas(ray_groupby, pandas_groupby)
    eval_ngroups(ray_groupby, pandas_groupby)
    eval_skew(ray_groupby, pandas_groupby)
    eval_ffill(ray_groupby, pandas_groupby)
    eval_sem(ray_groupby, pandas_groupby)
    eval_mean(ray_groupby, pandas_groupby)
    eval_any(ray_groupby, pandas_groupby)
    eval_min(ray_groupby, pandas_groupby)
    eval_idxmax(ray_groupby, pandas_groupby)
    eval_ndim(ray_groupby, pandas_groupby)
    eval_cumsum(ray_groupby, pandas_groupby)
    eval_pct_change(ray_groupby, pandas_groupby)
    eval_cummax(ray_groupby, pandas_groupby)

    apply_functions = [lambda df: df.sum(), lambda df: -df]
    for func in apply_functions:
        eval_apply(ray_groupby, pandas_groupby, func)

    eval_dtypes(ray_groupby, pandas_groupby)
    eval_first(ray_groupby, pandas_groupby)
    eval_backfill(ray_groupby, pandas_groupby)
    eval_cummin(ray_groupby, pandas_groupby)
    eval_bfill(ray_groupby, pandas_groupby)
    eval_idxmin(ray_groupby, pandas_groupby)
    eval_prod(ray_groupby, pandas_groupby)
    eval_std(ray_groupby, pandas_groupby)

    agg_functions = ["min", "max"]
    for func in agg_functions:
        eval_agg(ray_groupby, pandas_groupby, func)
        eval_aggregate(ray_groupby, pandas_groupby, func)

    eval_last(ray_groupby, pandas_groupby)
    eval_mad(ray_groupby, pandas_groupby)
    eval_rank(ray_groupby, pandas_groupby)
    eval_max(ray_groupby, pandas_groupby)
    eval_var(ray_groupby, pandas_groupby)
    eval_len(ray_groupby, pandas_groupby)
    eval_sum(ray_groupby, pandas_groupby)
    eval_ngroup(ray_groupby, pandas_groupby)
    eval_nunique(ray_groupby, pandas_groupby)
    eval_median(ray_groupby, pandas_groupby)
    eval_head(ray_groupby, pandas_groupby, n)
    eval_cumprod(ray_groupby, pandas_groupby)
    eval_cov(ray_groupby, pandas_groupby)

    transform_functions = [lambda df: df + 4, lambda df: -df - 10]
    for func in transform_functions:
        eval_transform(ray_groupby, pandas_groupby, func)

    pipe_functions = [lambda dfgb: dfgb.sum()]
    for func in pipe_functions:
        eval_pipe(ray_groupby, pandas_groupby, func)

    eval_corr(ray_groupby, pandas_groupby)
    eval_fillna(ray_groupby, pandas_groupby)
    eval_count(ray_groupby, pandas_groupby)
    eval_tail(ray_groupby, pandas_groupby, n)
    eval_quantile(ray_groupby, pandas_groupby)
    eval_take(ray_groupby, pandas_groupby)
    eval___getattr__(ray_groupby, pandas_groupby)
    eval_groups(ray_groupby, pandas_groupby)


def test_large_row_groupby():
    pandas_df = pandas.DataFrame(
        np.random.randint(0, 8, size=(100, 4)), columns=list("ABCD")
    )

    ray_df = from_pandas(pandas_df)

    by = [str(i) for i in pandas_df["A"].tolist()]
    n = 4

    ray_groupby = ray_df.groupby(by=by)
    pandas_groupby = pandas_df.groupby(by=by)

    ray_groupby_equals_pandas(ray_groupby, pandas_groupby)
    eval_ngroups(ray_groupby, pandas_groupby)
    eval_skew(ray_groupby, pandas_groupby)
    eval_ffill(ray_groupby, pandas_groupby)
    eval_sem(ray_groupby, pandas_groupby)
    eval_mean(ray_groupby, pandas_groupby)
    eval_any(ray_groupby, pandas_groupby)
    eval_min(ray_groupby, pandas_groupby)
    eval_idxmax(ray_groupby, pandas_groupby)
    eval_ndim(ray_groupby, pandas_groupby)
    eval_cumsum(ray_groupby, pandas_groupby)
    eval_pct_change(ray_groupby, pandas_groupby)
    eval_cummax(ray_groupby, pandas_groupby)

    apply_functions = [lambda df: df.sum(), lambda df: -df]
    for func in apply_functions:
        eval_apply(ray_groupby, pandas_groupby, func)

    eval_dtypes(ray_groupby, pandas_groupby)
    eval_first(ray_groupby, pandas_groupby)
    eval_backfill(ray_groupby, pandas_groupby)
    eval_cummin(ray_groupby, pandas_groupby)
    eval_bfill(ray_groupby, pandas_groupby)
    eval_idxmin(ray_groupby, pandas_groupby)
    # eval_prod(ray_groupby, pandas_groupby) causes overflows
    eval_std(ray_groupby, pandas_groupby)

    agg_functions = ["min", "max"]
    for func in agg_functions:
        eval_agg(ray_groupby, pandas_groupby, func)
        eval_aggregate(ray_groupby, pandas_groupby, func)

    eval_last(ray_groupby, pandas_groupby)
    eval_mad(ray_groupby, pandas_groupby)
    eval_rank(ray_groupby, pandas_groupby)
    eval_max(ray_groupby, pandas_groupby)
    eval_var(ray_groupby, pandas_groupby)
    eval_len(ray_groupby, pandas_groupby)
    eval_sum(ray_groupby, pandas_groupby)
    eval_ngroup(ray_groupby, pandas_groupby)
    eval_nunique(ray_groupby, pandas_groupby)
    eval_median(ray_groupby, pandas_groupby)
    eval_head(ray_groupby, pandas_groupby, n)
    # eval_cumprod(ray_groupby, pandas_groupby) causes overflows
    eval_cov(ray_groupby, pandas_groupby)

    transform_functions = [lambda df: df + 4, lambda df: -df - 10]
    for func in transform_functions:
        eval_transform(ray_groupby, pandas_groupby, func)

    pipe_functions = [lambda dfgb: dfgb.sum()]
    for func in pipe_functions:
        eval_pipe(ray_groupby, pandas_groupby, func)

    eval_corr(ray_groupby, pandas_groupby)
    eval_fillna(ray_groupby, pandas_groupby)
    eval_count(ray_groupby, pandas_groupby)
    eval_tail(ray_groupby, pandas_groupby, n)
    eval_quantile(ray_groupby, pandas_groupby)
    eval_take(ray_groupby, pandas_groupby)
    eval_groups(ray_groupby, pandas_groupby)


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

    ray_df = from_pandas(pandas_df)

    by = [1, 2, 3, 2, 1]

    ray_groupby = ray_df.groupby(axis=1, by=by)
    pandas_groupby = pandas_df.groupby(axis=1, by=by)

    ray_groupby_equals_pandas(ray_groupby, pandas_groupby)
    eval_ngroups(ray_groupby, pandas_groupby)
    eval_skew(ray_groupby, pandas_groupby)
    eval_ffill(ray_groupby, pandas_groupby)
    eval_sem(ray_groupby, pandas_groupby)
    eval_mean(ray_groupby, pandas_groupby)
    eval_any(ray_groupby, pandas_groupby)
    eval_min(ray_groupby, pandas_groupby)
    eval_ndim(ray_groupby, pandas_groupby)

    eval_idxmax(ray_groupby, pandas_groupby)
    eval_idxmin(ray_groupby, pandas_groupby)
    eval_quantile(ray_groupby, pandas_groupby)

    # https://github.com/pandas-dev/pandas/issues/21127
    # eval_cumsum(ray_groupby, pandas_groupby)
    # eval_cummax(ray_groupby, pandas_groupby)
    # eval_cummin(ray_groupby, pandas_groupby)
    # eval_cumprod(ray_groupby, pandas_groupby)

    eval_pct_change(ray_groupby, pandas_groupby)
    apply_functions = [lambda df: -df, lambda df: df.sum(axis=1)]
    for func in apply_functions:
        eval_apply(ray_groupby, pandas_groupby, func)

    eval_first(ray_groupby, pandas_groupby)
    eval_backfill(ray_groupby, pandas_groupby)
    eval_bfill(ray_groupby, pandas_groupby)
    eval_prod(ray_groupby, pandas_groupby)
    eval_std(ray_groupby, pandas_groupby)
    eval_last(ray_groupby, pandas_groupby)
    eval_mad(ray_groupby, pandas_groupby)
    eval_max(ray_groupby, pandas_groupby)
    eval_var(ray_groupby, pandas_groupby)
    eval_len(ray_groupby, pandas_groupby)
    eval_sum(ray_groupby, pandas_groupby)

    # Pandas fails on this case with ValueError
    # eval_ngroup(ray_groupby, pandas_groupby)
    # eval_nunique(ray_groupby, pandas_groupby)
    eval_median(ray_groupby, pandas_groupby)
    eval_cov(ray_groupby, pandas_groupby)

    transform_functions = [lambda df: df + 4, lambda df: -df - 10]
    for func in transform_functions:
        eval_transform(ray_groupby, pandas_groupby, func)

    pipe_functions = [lambda dfgb: dfgb.sum()]
    for func in pipe_functions:
        eval_pipe(ray_groupby, pandas_groupby, func)

    eval_corr(ray_groupby, pandas_groupby)
    eval_fillna(ray_groupby, pandas_groupby)
    eval_count(ray_groupby, pandas_groupby)
    eval_take(ray_groupby, pandas_groupby)
    eval___getattr__(ray_groupby, pandas_groupby)
    eval_groups(ray_groupby, pandas_groupby)


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

    ray_df = from_pandas(pandas_df)
    by = ["col1", "col2"]

    ray_df_equals_pandas(ray_df.groupby(by).count(), pandas_df.groupby(by).count())

    with pytest.warns(UserWarning):
        for k, _ in ray_df.groupby(by):
            assert isinstance(k, tuple)

    by = ["row0", "row1"]
    with pytest.raises(KeyError):
        ray_df.groupby(by, axis=1).count()


def eval_ngroups(ray_groupby, pandas_groupby):
    assert ray_groupby.ngroups == pandas_groupby.ngroups


def eval_skew(ray_groupby, pandas_groupby):
    ray_df_almost_equals_pandas(ray_groupby.skew(), pandas_groupby.skew())


def eval_ffill(ray_groupby, pandas_groupby):
    with pytest.warns(UserWarning):
        try:
            ray_groupby.ffill()
        except Exception:
            pass


def eval_sem(ray_groupby, pandas_groupby):
    with pytest.warns(UserWarning):
        ray_groupby.sem()


def eval_mean(ray_groupby, pandas_groupby):
    ray_df_almost_equals_pandas(ray_groupby.mean(), pandas_groupby.mean())


def eval_any(ray_groupby, pandas_groupby):
    ray_df_equals_pandas(ray_groupby.any(), pandas_groupby.any())


def eval_min(ray_groupby, pandas_groupby):
    ray_df_equals_pandas(ray_groupby.min(), pandas_groupby.min())


def eval_idxmax(ray_groupby, pandas_groupby):
    with pytest.warns(UserWarning):
        ray_groupby.idxmax()


def eval_ndim(ray_groupby, pandas_groupby):
    assert ray_groupby.ndim == pandas_groupby.ndim


def eval_cumsum(ray_groupby, pandas_groupby, axis=0):
    ray_df_equals_pandas(
        ray_groupby.cumsum(axis=axis), pandas_groupby.cumsum(axis=axis)
    )


def eval_pct_change(ray_groupby, pandas_groupby):
    with pytest.warns(UserWarning):
        try:
            ray_groupby.pct_change()
        except Exception:
            pass


def eval_cummax(ray_groupby, pandas_groupby, axis=0):
    ray_df_equals_pandas(
        ray_groupby.cummax(axis=axis), pandas_groupby.cummax(axis=axis)
    )


def eval_apply(ray_groupby, pandas_groupby, func):
    ray_df_equals_pandas(ray_groupby.apply(func), pandas_groupby.apply(func))


def eval_dtypes(ray_groupby, pandas_groupby):
    ray_df_equals_pandas(ray_groupby.dtypes, pandas_groupby.dtypes)


def eval_first(ray_groupby, pandas_groupby):
    with pytest.warns(UserWarning):
        ray_groupby.first()


def eval_backfill(ray_groupby, pandas_groupby):
    with pytest.warns(UserWarning):
        try:
            ray_groupby.backfill()
        except Exception:
            pass


def eval_cummin(ray_groupby, pandas_groupby, axis=0):
    ray_df_equals_pandas(
        ray_groupby.cummin(axis=axis), pandas_groupby.cummin(axis=axis)
    )


def eval_bfill(ray_groupby, pandas_groupby):
    with pytest.warns(UserWarning):
        try:
            ray_groupby.bfill()
        except Exception:
            pass


def eval_idxmin(ray_groupby, pandas_groupby):
    with pytest.warns(UserWarning):
        ray_groupby.idxmin()


def eval_prod(ray_groupby, pandas_groupby):
    ray_df_equals_pandas(ray_groupby.prod(), pandas_groupby.prod())


def eval_std(ray_groupby, pandas_groupby):
    ray_df_almost_equals_pandas(ray_groupby.std(), pandas_groupby.std())


def eval_aggregate(ray_groupby, pandas_groupby, func):
    ray_df_equals_pandas(ray_groupby.aggregate(func), pandas_groupby.aggregate(func))


def eval_agg(ray_groupby, pandas_groupby, func):
    ray_df_equals_pandas(ray_groupby.agg(func), pandas_groupby.agg(func))


def eval_last(ray_groupby, pandas_groupby):
    with pytest.warns(UserWarning):
        ray_groupby.last()


def eval_mad(ray_groupby, pandas_groupby):
    with pytest.warns(UserWarning):
        ray_groupby.mad()


def eval_rank(ray_groupby, pandas_groupby):
    ray_df_equals_pandas(ray_groupby.rank(), pandas_groupby.rank())


def eval_max(ray_groupby, pandas_groupby):
    ray_df_equals_pandas(ray_groupby.max(), pandas_groupby.max())


def eval_var(ray_groupby, pandas_groupby):
    ray_df_almost_equals_pandas(ray_groupby.var(), pandas_groupby.var())


def eval_len(ray_groupby, pandas_groupby):
    assert len(ray_groupby) == len(pandas_groupby)


def eval_sum(ray_groupby, pandas_groupby):
    ray_df_equals_pandas(ray_groupby.sum(), pandas_groupby.sum())


def eval_ngroup(ray_groupby, pandas_groupby):
    ray_series_equals_pandas(ray_groupby.ngroup(), pandas_groupby.ngroup())


def eval_nunique(ray_groupby, pandas_groupby):
    ray_df_equals_pandas(ray_groupby.nunique(), pandas_groupby.nunique())


def eval_median(ray_groupby, pandas_groupby):
    ray_df_almost_equals_pandas(ray_groupby.median(), pandas_groupby.median())


def eval_head(ray_groupby, pandas_groupby, n):
    with pytest.warns(UserWarning):
        ray_groupby.head()


def eval_cumprod(ray_groupby, pandas_groupby, axis=0):
    ray_df_equals_pandas(ray_groupby.cumprod(), pandas_groupby.cumprod())
    ray_df_equals_pandas(
        ray_groupby.cumprod(axis=axis), pandas_groupby.cumprod(axis=axis)
    )


def eval_cov(ray_groupby, pandas_groupby):
    with pytest.warns(UserWarning):
        ray_groupby.cov()


def eval_transform(ray_groupby, pandas_groupby, func):
    ray_df_equals_pandas(ray_groupby.transform(func), pandas_groupby.transform(func))


def eval_corr(ray_groupby, pandas_groupby):
    with pytest.warns(UserWarning):
        ray_groupby.corr()


def eval_fillna(ray_groupby, pandas_groupby):
    ray_df_equals_pandas(
        ray_groupby.fillna(method="ffill"), pandas_groupby.fillna(method="ffill")
    )


def eval_count(ray_groupby, pandas_groupby):
    ray_df_equals_pandas(ray_groupby.count(), pandas_groupby.count())


def eval_pipe(ray_groupby, pandas_groupby, func):
    ray_df_equals_pandas(ray_groupby.pipe(func), pandas_groupby.pipe(func))


def eval_tail(ray_groupby, pandas_groupby, n):
    with pytest.warns(UserWarning):
        ray_groupby.tail()


def eval_quantile(ray_groupby, pandas_groupby):
    try:
        pandas_result = pandas_groupby.quantile(q=0.4)
    except Exception as e:
        with pytest.raises(type(e)):
            ray_groupby.quantile(q=0.4)
    else:
        ray_df_equals_pandas(ray_groupby.quantile(q=0.4), pandas_result)


def eval_take(ray_groupby, pandas_groupby):
    with pytest.warns(UserWarning):
        try:
            ray_groupby.take()
        except Exception:
            pass


def eval___getattr__(ray_groupby, pandas_groupby):
    with pytest.warns(UserWarning):
        ray_groupby["col1"]

    with pytest.warns(UserWarning):
        ray_groupby.col1


def eval_groups(ray_groupby, pandas_groupby):
    for k, v in ray_groupby.groups.items():
        assert v.equals(pandas_groupby.groups[k])


def eval_shift(ray_groupby, pandas_groupby):
    assert ray_groupby.groups == pandas_groupby.groups


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
        ray_df_equals_pandas(modin_dict[k], pandas_dict[k])

    modin_groupby_obj = modin_df.groupby(modin_df.columns, axis=1)
    pandas_groupby_obj = pandas_df.groupby(pandas_df.columns, axis=1)

    modin_dict = {k: v for k, v in modin_groupby_obj}
    pandas_dict = {k: v for k, v in pandas_groupby_obj}

    for k in modin_dict:
        ray_df_equals_pandas(modin_dict[k], pandas_dict[k])


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
    ray_df_equals_pandas(
        modin_df.groupby(level=1).count(), pandas_df.groupby(level=1).count()
    )
    ray_df_equals_pandas(
        modin_df.groupby(by="four").count(), pandas_df.groupby(by="four").count()
    )

    by = ["one", "two"]
    ray_df_equals_pandas(
        modin_df.groupby(by=by).count(), pandas_df.groupby(by=by).count()
    )
