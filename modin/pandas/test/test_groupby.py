from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import sys
import pandas
import modin.pandas as pd

from .utils import (
    df_equals,
    name_contains,
    arg_keys,
    test_data,
    numeric_dfs,
    axis_keys,
    axis_values,
    bool_arg_keys,
    bool_arg_values,
    bool_none_arg_keys,
    bool_none_arg_values,
    int_arg_keys,
    int_arg_values,
    quantiles_keys,
    quantiles_values,
    groupby_apply_func_keys,
    groupby_apply_func_values,
    groupby_agg_func_keys,
    groupby_agg_func_values,
    groupby_transform_func_keys,
    groupby_transform_func_values,
    groupby_pipe_func_keys,
    groupby_pipe_func_values,
)

PY2 = False
if sys.version_info.major < 3:
    PY2 = True

# Create test_groupby objects
test_groupby = {}
for axis_name, axis in zip(axis_keys, axis_values):
    for df_name, frame_data in test_data.items():
        if "empty_data" not in df_name and not (
            "over rows" in axis_name and "columns_only" in df_name
        ):
            modin_df, pandas_df = (
                pd.DataFrame(frame_data),
                pandas.DataFrame(frame_data),
            )
            index = modin_df.columns if "over columns" in axis_name else modin_df.index
            vals = (
                modin_df.groupby([str(i) for i in index], axis=axis),
                pandas_df.groupby([str(i) for i in index], axis=axis),
            )
            test_groupby["{}-{}".format(df_name, axis_name)] = vals

test_groupby_keys = list(test_groupby.keys())
test_groupby_values = list(test_groupby.values())


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
def test_ngroups(modin_groupby, pandas_groupby):
    assert modin_groupby.ngroups == pandas_groupby.ngroups


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "numeric_only",
    bool_none_arg_values,
    ids=arg_keys("numeric_only", bool_none_arg_keys),
)
def test_skew(modin_groupby, pandas_groupby, axis, skipna, numeric_only):
    modin_result = modin_groupby.skew(
        axis=axis, skipna=skipna, numeric_only=numeric_only
    )
    pandas_result = pandas_groupby.skew(
        axis=axis, skipna=skipna, numeric_only=numeric_only
    )
    assert df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
def test_ffill(modin_groupby, pandas_groupby):
    with pytest.raises(NotImplementedError):
        modin_groupby.ffill()


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
def test_sem(modin_groupby, pandas_groupby):
    with pytest.raises(NotImplementedError):
        modin_groupby.sem()


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "numeric_only",
    bool_none_arg_values,
    ids=arg_keys("numeric_only", bool_none_arg_keys),
)
def test_mean(modin_groupby, pandas_groupby, axis, skipna, numeric_only):
    modin_result = modin_groupby.mean(
        axis=axis, skipna=skipna, numeric_only=numeric_only
    )
    pandas_result = pandas_groupby.mean(
        axis=axis, skipna=skipna, numeric_only=numeric_only
    )
    assert df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "bool_only", bool_none_arg_values, ids=arg_keys("bool_only", bool_none_arg_keys)
)
def test_any(modin_groupby, pandas_groupby, axis, skipna, bool_only):
    modin_result = modin_groupby.any(axis=axis, skipna=skipna, bool_only=bool_only)
    pandas_result = pandas_groupby.any(axis=axis, skipna=skipna, bool_only=bool_only)
    assert df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "numeric_only",
    bool_none_arg_values,
    ids=arg_keys("numeric_only", bool_none_arg_keys),
)
def test_min(modin_groupby, pandas_groupby, axis, skipna, numeric_only):
    modin_result = modin_groupby.min(
        axis=axis, skipna=skipna, numeric_only=numeric_only
    )
    pandas_result = pandas_groupby.min(
        axis=axis, skipna=skipna, numeric_only=numeric_only
    )
    assert df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
def test_idxmax(modin_groupby, pandas_groupby):
    with pytest.raises(NotImplementedError):
        assert df_equals(modin_groupby.idxmax(), pandas_groupby.idxmax())


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
def test_ndim(modin_groupby, pandas_groupby):
    assert modin_groupby.ndim == pandas_groupby.ndim


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_cumsum(request, modin_groupby, pandas_groupby, axis, skipna):
    if name_contains(request.node.name, numeric_dfs):
        modin_result = modin_groupby.cumsum(axis=axis, skipna=skipna)
        pandas_result = pandas_groupby.cumsum(axis=axis, skipna=skipna)
        assert df_equals(modin_result, pandas_result)
    else:
        with pytest.raises(TypeError):
            modin_result = modin_groupby.cumsum(axis=axis, skipna=skipna)


@pytest.fixture
def test_pct_change(modin_groupby, pandas_groupby):
    with pytest.raises(NotImplementedError):
        modin_groupby.pct_change()


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_cummax(request, modin_groupby, pandas_groupby, axis, skipna):
    if name_contains(request.node.name, numeric_dfs):
        modin_result = modin_groupby.cummax(axis=axis, skipna=skipna)
        pandas_result = pandas_groupby.cummax(axis=axis, skipna=skipna)
        assert df_equals(modin_result, pandas_result)
    else:
        with pytest.raises(TypeError):
            modin_result = modin_groupby.cummax(axis=axis, skipna=skipna)


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("func", groupby_apply_func_values, ids=groupby_apply_func_keys)
def test_apply(request, modin_groupby, pandas_groupby, func, axis):
    if name_contains(request.node.name, ["over rows"]) or not name_contains(
        request.node.name, numeric_dfs
    ):
        modin_result = modin_groupby.apply(func, axis)
        pandas_result = pandas_groupby.apply(func, axis)
        assert df_equals(modin_result, pandas_result)
    else:
        with pytest.raises(TypeError):
            modin_result = modin_groupby.apply(func, axis)


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
def test_dtypes(modin_groupby, pandas_groupby):
    assert df_equals(modin_groupby.dtypes, pandas_groupby.dtypes)


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
def test_first(modin_groupby, pandas_groupby):
    with pytest.raises(NotImplementedError):
        modin_groupby.first()


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
def test_backfill(modin_groupby, pandas_groupby):
    with pytest.raises(NotImplementedError):
        modin_groupby.backfill()


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_cummin(modin_groupby, pandas_groupby, axis, skipna):
    modin_result = modin_groupby.cummin(axis=axis, skipna=skipna)
    pandas_result = pandas_groupby.cummin(axis=axis, skipna=skipna)
    assert df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
def test_bfill(modin_groupby, pandas_groupby):
    with pytest.raises(NotImplementedError):
        modin_groupby.bfill()


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
def test_idxmin(modin_groupby, pandas_groupby):
    with pytest.raises(NotImplementedError):
        assert df_equals(modin_groupby.idxmin(), pandas_groupby.idxmin())


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
)
@pytest.mark.parametrize(
    "min_count", int_arg_values, ids=arg_keys("min_count", int_arg_keys)
)
def test_prod(
    request, modin_groupby, pandas_groupby, axis, skipna, numeric_only, min_count
):
    if numeric_only or name_contains(request.node.name, numeric_dfs):
        modin_result = modin_groupby.prod(
            axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
        )
        pandas_result = pandas_groupby.prod(
            axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
        )
        assert df_equals(modin_result, pandas_result)
    else:
        with pytest.raises(TypeError):
            modin_groupby.prod(
                axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
            )


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "numeric_only",
    bool_none_arg_values,
    ids=arg_keys("numeric_only", bool_none_arg_keys),
)
@pytest.mark.parametrize("ddof", int_arg_values, ids=arg_keys("ddof", int_arg_keys))
def test_std(modin_groupby, pandas_groupby, axis, skipna, numeric_only, ddof):
    modin_result = modin_groupby.std(
        axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof
    )
    pandas_result = pandas_groupby.std(
        axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof
    )
    assert df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("func", groupby_agg_func_values, ids=groupby_agg_func_keys)
def test_aggregate(request, modin_groupby, pandas_groupby, axis, func):
    # if (name_contains(request.node.name, ["over rows"]) or
    #         not name_contains(request.node.name, numeric_dfs)):
    #     modin_result = modin_groupby.aggregate(func, axis)
    #     pandas_result = pandas_groupby.aggregate(func, axis)
    #     assert df_equals(modin_result, pandas_result)
    # else:
    #     with pytest.raises(TypeError):
    #         modin_result = modin_groupby.aggregate(func, axis)
    with pytest.raises(NotImplementedError):
        modin_groupby.aggregate()


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("func", groupby_agg_func_values, ids=groupby_agg_func_keys)
def test_agg(request, modin_groupby, pandas_groupby, axis, func):
    # if (name_contains(request.node.name, ["over rows"]) or
    #         not name_contains(request.node.name, numeric_dfs)):
    #     modin_result = modin_groupby.agg(func, axis)
    #     pandas_result = pandas_groupby.agg(func, axis)
    #     assert df_equals(modin_result, pandas_result)
    # else:
    #     with pytest.raises(TypeError):
    #         modin_result = modin_groupby.agg(func, axis)
    with pytest.raises(NotImplementedError):
        modin_groupby.agg()


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
def test_last(modin_groupby, pandas_groupby):
    with pytest.raises(NotImplementedError):
        modin_groupby.last()


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
def test_mad(modin_groupby, pandas_groupby):
    with pytest.raises(NotImplementedError):
        modin_groupby.mad()


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "method",
    ["average", "min", "max", "first", "dense"],
    ids=["average", "min", "max", "first", "dense"],
)
@pytest.mark.parametrize(
    "numeric_only",
    bool_none_arg_values,
    ids=arg_keys("numeric_only", bool_none_arg_keys),
)
@pytest.mark.parametrize(
    "na_option", ["keep", "top", "bottom"], ids=["keep", "top", "bottom"]
)
@pytest.mark.parametrize(
    "ascending", bool_arg_values, ids=arg_keys("ascending", bool_arg_keys)
)
@pytest.mark.parametrize("pct", bool_arg_values, ids=arg_keys("pct", bool_arg_keys))
def test_rank(
    modin_groupby, pandas_groupby, axis, method, numeric_only, na_option, ascending, pct
):
    modin_result = modin_groupby.rank(
        axis=axis,
        method=method,
        numeric_only=numeric_only,
        na_option=na_option,
        ascending=ascending,
        pct=pct,
    )
    pandas_result = pandas_groupby.rank(
        axis=axis,
        method=method,
        numeric_only=numeric_only,
        na_option=na_option,
        ascending=ascending,
        pct=pct,
    )
    assert df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
)
def test_max(modin_groupby, pandas_groupby, axis, skipna, numeric_only):
    modin_result = modin_groupby.max(
        axis=axis, skipna=skipna, numeric_only=numeric_only
    )
    pandas_result = pandas_groupby.max(
        axis=axis, skipna=skipna, numeric_only=numeric_only
    )
    assert df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "numeric_only",
    bool_none_arg_values,
    ids=arg_keys("numeric_only", bool_none_arg_keys),
)
@pytest.mark.parametrize("ddof", int_arg_values, ids=arg_keys("ddof", int_arg_keys))
def test_var(modin_groupby, pandas_groupby, axis, skipna, numeric_only, ddof):
    modin_result = modin_groupby.var(
        axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof
    )
    pandas_result = pandas_groupby.var(
        axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof
    )
    assert df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
def test_len(modin_groupby, pandas_groupby):
    assert len(modin_groupby) == len(pandas_groupby)


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
)
@pytest.mark.parametrize(
    "min_count", int_arg_values, ids=arg_keys("min_count", int_arg_keys)
)
def test_sum(
    request, modin_groupby, pandas_groupby, axis, skipna, numeric_only, min_count
):
    if numeric_only or name_contains(request.node.name, numeric_dfs):
        modin_result = modin_groupby.sum(
            axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
        )
        pandas_result = pandas_groupby.sum(
            axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
        )
        assert df_equals(modin_result, pandas_result)
    else:
        with pytest.raises(TypeError):
            modin_groupby.sum(
                axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
            )


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
def test_ngroup(modin_groupby, pandas_groupby):
    assert df_equals(modin_groupby.ngroup(), pandas_groupby.ngroup())


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
def test_nunique(modin_groupby, pandas_groupby):
    assert df_equals(modin_groupby.nunique(), pandas_groupby.nunique())


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "numeric_only",
    bool_none_arg_values,
    ids=arg_keys("numeric_only", bool_none_arg_keys),
)
def test_median(modin_groupby, pandas_groupby, axis, skipna, numeric_only):
    modin_result = modin_groupby.median(
        axis=axis, skipna=skipna, numeric_only=numeric_only
    )
    pandas_result = pandas_groupby.median(
        axis=axis, skipna=skipna, numeric_only=numeric_only
    )
    assert df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("n", int_arg_values, ids=arg_keys("n", int_arg_keys))
def test_head(modin_groupby, pandas_groupby, n):
    with pytest.raises(NotImplementedError):
        assert df_equals(modin_groupby.head(n=n), pandas_groupby.head(n=n))


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_cumprod(request, modin_groupby, pandas_groupby, axis, skipna):
    if name_contains(request.node.name, numeric_dfs):
        modin_result = modin_groupby.cumprod(axis=axis, skipna=skipna)
        pandas_result = pandas_groupby.cumprod(axis=axis, skipna=skipna)
        assert df_equals(modin_result, pandas_result)
    else:
        with pytest.raises(TypeError):
            modin_result = modin_groupby.cumprod(axis=axis, skipna=skipna)


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
def test_cov(modin_groupby, pandas_groupby):
    with pytest.raises(NotImplementedError):
        modin_groupby.cov()


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize(
    "func", groupby_transform_func_values, ids=groupby_transform_func_keys
)
def test_transform(request, modin_groupby, pandas_groupby, func):
    if "empty_data" not in request.node.name:
        modin_result = modin_groupby.agg(func)
        pandas_result = pandas_groupby.agg(func)
        assert df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
def test_corr(modin_groupby, pandas_groupby):
    with pytest.raises(NotImplementedError):
        modin_groupby.corr()


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize(
    "method",
    ["backfill", "bfill", "pad", "ffill", None],
    ids=["backfill", "bfill", "pad", "ffill", "None"],
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
def test_fillna(modin_groupby, pandas_groupby, method, axis):
    modin_result = modin_groupby.fillna(method=method, axis=axis, inplace=False)
    pandas_result = pandas_groupby.fillna(method=method, axis=axis, inplace=False)
    assert df_equals(modin_result, pandas_result)

    modin_groupby.fillna(method=method, axis=axis, inplace=True)
    pandas_groupby.fillna(method=method, axis=axis, inplace=True)
    assert df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
)
def test_count(modin_groupby, pandas_groupby, axis, numeric_only):
    modin_result = modin_groupby.count(axis=axis, numeric_only=numeric_only)
    pandas_result = pandas_groupby.count(axis=axis, numeric_only=numeric_only)
    assert df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("func", groupby_pipe_func_values, ids=groupby_pipe_func_keys)
def test_pipe(modin_groupby, pandas_groupby, func):
    assert df_equals(modin_groupby.pipe(func), pandas_groupby.pipe(func))


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("n", int_arg_values, ids=arg_keys("n", int_arg_keys))
def test_tail(modin_groupby, pandas_groupby, n):
    with pytest.raises(NotImplementedError):
        assert df_equals(modin_groupby.tail(n=n), pandas_groupby.tail(n=n))


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
@pytest.mark.parametrize("q", quantiles_values, ids=quantiles_keys)
def test_quantile(modin_groupby, pandas_groupby, q):
    assert df_equals(modin_groupby.quantile(q), pandas_groupby.quantile(q))


@pytest.mark.parametrize(
    "modin_groupby, pandas_groupby", test_groupby_values, ids=test_groupby_keys
)
def test_take(modin_groupby, pandas_groupby):
    with pytest.raises(NotImplementedError):
        modin_groupby.take(indices=[1])
