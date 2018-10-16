from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import io
import numpy as np
import pandas
import pandas.util.testing as tm
from pandas.tests.frame.common import TestData
import matplotlib
import modin.pandas as pd
from modin.pandas.utils import to_pandas
from numpy.testing import assert_array_equal
import sys

from .utils import (
    random_state,
    RAND_LOW,
    RAND_HIGH,
    df_equals,
    df_is_empty,
    arg_keys,
    name_contains,
    test_dfs_keys,
    test_dfs_values,
    numeric_dfs,
    no_numeric_dfs,
    test_func_keys,
    test_func_values,
    numeric_test_funcs,
    query_func_keys,
    query_func_values,
    agg_func_keys,
    agg_func_values,
    numeric_agg_funcs,
    quantiles_keys,
    quantiles_values,
    indices_keys,
    indices_values,
    axis_keys,
    axis_values,
    bool_arg_keys,
    bool_arg_values,
    bool_none_arg_keys,
    bool_none_arg_values,
    int_arg_keys,
    int_arg_values,
)

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")

if sys.version_info[0] < 3:
    PY2 = True
else:
    PY2 = False


# Test inter df math functions
def inter_df_math_helper(op):
    frame_data = {
        "col1": [0, 1, 2, 3],
        "col2": [4, 5, 6, 7],
        "col3": [8, 9, 0, 1],
        "col4": [2, 4, 5, 6],
    }

    modin_df = pd.DataFrame(frame_data)
    pandas_df = pandas.DataFrame(frame_data)

    df_equals(getattr(modin_df, op)(modin_df), getattr(pandas_df, op)(pandas_df))
    df_equals(getattr(modin_df, op)(4), getattr(pandas_df, op)(4))
    df_equals(getattr(modin_df, op)(4.0), getattr(pandas_df, op)(4.0))

    frame_data = {"A": [0, 2], "col1": [0, 19], "col2": [1, 1]}
    modin_df2 = pd.DataFrame(frame_data)
    pandas_df2 = pandas.DataFrame(frame_data)

    df_equals(
        getattr(modin_df, op)(modin_df2), getattr(pandas_df, op)(pandas_df2)
    )

    list_test = [0, 1, 2, 4]

    df_equals(
        getattr(modin_df, op)(list_test, axis=1),
        getattr(pandas_df, op)(list_test, axis=1),
    )

    df_equals(
        getattr(modin_df, op)(list_test, axis=0),
        getattr(pandas_df, op)(list_test, axis=0),
    )


def test_add():
    inter_df_math_helper("add")


def test_div():
    inter_df_math_helper("div")


def test_divide():
    inter_df_math_helper("divide")


def test_floordiv():
    inter_df_math_helper("floordiv")


def test_mod():
    inter_df_math_helper("mod")


def test_mul():
    inter_df_math_helper("mul")


def test_multiply():
    inter_df_math_helper("multiply")


def test_pow():
    inter_df_math_helper("pow")


def test_sub():
    inter_df_math_helper("sub")


def test_subtract():
    inter_df_math_helper("subtract")


def test_truediv():
    inter_df_math_helper("truediv")


def test___div__():
    inter_df_math_helper("__div__")


# END test inter df math functions


# Test comparison of inter operation functions
def comparison_inter_ops_helper(op):
    frame_data = {
        "col1": [0, 1, 2, 3],
        "col2": [4, 5, 6, 7],
        "col3": [8, 9, 0, 1],
        "col4": [2, 4, 5, 6],
    }

    modin_df = pd.DataFrame(frame_data)
    pandas_df = pandas.DataFrame(frame_data)

    df_equals(getattr(modin_df, op)(modin_df), getattr(pandas_df, op)(pandas_df))
    df_equals(getattr(modin_df, op)(4), getattr(pandas_df, op)(4))
    df_equals(getattr(modin_df, op)(4.0), getattr(pandas_df, op)(4.0))

    frame_data = {"A": [0, 2], "col1": [0, 19], "col2": [1, 1]}

    modin_df2 = pd.DataFrame(frame_data)
    pandas_df2 = pandas.DataFrame(frame_data)

    df_equals(
        getattr(modin_df2, op)(modin_df2), getattr(pandas_df2, op)(pandas_df2)
    )


def test_eq():
    comparison_inter_ops_helper("eq")


def test_ge():
    comparison_inter_ops_helper("ge")


def test_gt():
    comparison_inter_ops_helper("gt")


def test_le():
    comparison_inter_ops_helper("le")


def test_lt():
    comparison_inter_ops_helper("lt")


def test_ne():
    comparison_inter_ops_helper("ne")


# END test comparison of inter operation functions


# Test dataframe right operations
def inter_df_math_right_ops_helper(op):
    frame_data = {
        "col1": [0, 1, 2, 3],
        "col2": [4, 5, 6, 7],
        "col3": [8, 9, 0, 1],
        "col4": [2, 4, 5, 6],
    }

    modin_df = pd.DataFrame(frame_data)
    pandas_df = pandas.DataFrame(frame_data)

    df_equals(getattr(modin_df, op)(4), getattr(pandas_df, op)(4))
    df_equals(getattr(modin_df, op)(4.0), getattr(pandas_df, op)(4.0))


def test_radd():
    inter_df_math_right_ops_helper("radd")


def test_rdiv():
    inter_df_math_right_ops_helper("rdiv")


def test_rfloordiv():
    inter_df_math_right_ops_helper("rfloordiv")


def test_rmod():
    inter_df_math_right_ops_helper("rmod")


def test_rmul():
    inter_df_math_right_ops_helper("rmul")


def test_rpow():
    inter_df_math_right_ops_helper("rpow")


def test_rsub():
    inter_df_math_right_ops_helper("rsub")


def test_rtruediv():
    inter_df_math_right_ops_helper("rtruediv")


def test___rsub__():
    inter_df_math_right_ops_helper("__rsub__")


# END test dataframe right operations


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_abs(request, modin_df, pandas_df):
    try:
        pandas_result = pandas_df.abs()
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.abs()
        return
    modin_result = modin_df.abs()
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_add_prefix(modin_df, pandas_df):
    test_prefix = "TEST"
    new_modin_df = modin_df.add_prefix(test_prefix)
    new_pandas_df = pandas_df.add_prefix(test_prefix)
    df_equals(new_modin_df.columns, new_pandas_df.columns)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("testfunc", test_func_values, ids=test_func_keys)
def test_applymap(request, modin_df, pandas_df, testfunc):
    try:
        pandas_result = pandas_df.applymap(testfunc)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.applymap(testfunc)
        return
    modin_result = modin_df.applymap(testfunc)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_add_suffix(modin_df, pandas_df):
    test_suffix = "TEST"
    new_modin_df = modin_df.add_suffix(test_suffix)
    new_pandas_df = pandas_df.add_suffix(test_suffix)

    df_equals(new_modin_df.columns, new_pandas_df.columns)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_at(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.at()


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_axes(modin_df, pandas_df):
    for modin_axis, pd_axis in zip(modin_df.axes, pandas_df.axes):
        assert np.array_equal(modin_axis, pd_axis)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_copy(modin_df, pandas_df):
    # pandas_df is unused but there so there won't be confusing list comprehension
    # stuff in the pytest.mark.parametrize
    modin_df_cp = modin_df.copy()

    assert modin_df_cp is not modin_df
    df_equals(modin_df_cp, modin_df)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_dtypes(modin_df, pandas_df):
    df_equals(modin_df.dtypes, pandas_df.dtypes)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_ftypes(modin_df, pandas_df):
    df_equals(modin_df.ftypes, pandas_df.ftypes)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("key", indices_values, ids=indices_keys)
def test_get(modin_df, pandas_df, key):
    df_equals(modin_df.get(key), pandas_df.get(key))
    df_equals(
        modin_df.get(key, default="default"), pandas_df.get(key, default="default")
    )


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_get_dtype_counts(modin_df, pandas_df):
    df_equals(modin_df.get_dtype_counts(), pandas_df.get_dtype_counts())


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize(
    "dummy_na", bool_arg_values, ids=arg_keys("dummy_na", bool_arg_keys)
)
@pytest.mark.parametrize(
    "drop_first", bool_arg_values, ids=arg_keys("drop_first", bool_arg_keys)
)
def test_get_dummies(request, modin_df, pandas_df, dummy_na, drop_first):
    try:
        pandas_result = pandas.get_dummies(
            pandas_df, dummy_na=dummy_na, drop_first=drop_first
        )
    except Exception as e:
        with pytest.raises(type(e)):
            pd.get_dummies(modin_df, dummy_na=dummy_na, drop_first=drop_first)
        return
    modin_result = pd.get_dummies(modin_df, dummy_na=dummy_na, drop_first=drop_first)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_get_ftype_counts(modin_df, pandas_df):
    df_equals(modin_df.get_ftype_counts(), pandas_df.get_ftype_counts())


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
def test_agg(request, modin_df, pandas_df, axis, func):
    try:
        pandas_result = pandas_df.agg(func, axis)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_result = modin_df.agg(func, axis)
        return
    modin_result = modin_df.agg(func, axis)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
def test_aggregate(request, modin_df, pandas_df, func, axis):
    try:
        pandas_result = pandas_df.aggregate(func, axis)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_result = modin_df.aggregate(func, axis)
        return
    modin_result = modin_df.aggregate(func, axis)
    df_equals(modin_result, pandas_result)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_align(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.align(None)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "bool_only", bool_none_arg_values, ids=arg_keys("bool_only", bool_none_arg_keys)
)
def test_all(modin_df, pandas_df, axis, skipna, bool_only):
    modin_result = modin_df.all(axis=axis, skipna=skipna, bool_only=bool_only)
    pandas_result = pandas_df.all(axis=axis, skipna=skipna, bool_only=bool_only)
    df_equals(modin_result, pandas_result)

    # Test when axis is None. This will get repeated but easier than using list in parameterize decorator
    modin_result = modin_df.all(axis=None, skipna=skipna, bool_only=bool_only)
    pandas_result = pandas_df.all(axis=None, skipna=skipna, bool_only=bool_only)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "bool_only", bool_none_arg_values, ids=arg_keys("bool_only", bool_none_arg_keys)
)
def test_any(modin_df, pandas_df, axis, skipna, bool_only):
    modin_result = modin_df.any(axis=axis, skipna=skipna, bool_only=bool_only)
    pandas_result = pandas_df.any(axis=axis, skipna=skipna, bool_only=bool_only)
    df_equals(modin_result, pandas_result)

    # Test when axis is None. This will get repeated but easier than using list in parameterize decorator
    modin_result = modin_df.any(axis=None, skipna=skipna, bool_only=bool_only)
    pandas_result = pandas_df.any(axis=None, skipna=skipna, bool_only=bool_only)
    df_equals(modin_result, pandas_result)


def test_append():
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

    df_equals(modin_df.append(modin_df2), pandas_df.append(pandas_df2))

    with pytest.raises(ValueError):
        modin_df.append(modin_df2, verify_integrity=True)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
def test_apply(request, modin_df, pandas_df, func, axis):
    try:
        pandas_result = pandas_df.apply(func, axis)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_result = modin_df.apply(func, axis)
        return
    modin_result = modin_df.apply(func, axis)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_apply_special(request, modin_df, pandas_df):
    if name_contains(request.node.name, numeric_dfs):
        modin_result = modin_df.apply(lambda df: -df, axis=0)
        pandas_result = pandas_df.apply(lambda df: -df, axis=0)
        df_equals(modin_result, pandas_result)
        modin_result = modin_df.apply(lambda df: -df, axis=1)
        pandas_result = pandas_df.apply(lambda df: -df, axis=1)
        df_equals(modin_result, pandas_result)
    elif "empty_data" not in request.node.name:
        key = modin_df.columns[0]
        modin_result = modin_df.apply(lambda df: df.drop(key), axis=1)
        pandas_result = pandas_df.apply(lambda df: df.drop(key), axis=1)
        df_equals(modin_result, pandas_result)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_as_blocks(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.as_blocks()


def test_as_matrix():
    test_data = TestData()
    frame = pd.DataFrame(test_data.frame)
    mat = frame.as_matrix()

    frame_columns = frame.columns
    for i, row in enumerate(mat):
        for j, value in enumerate(row):
            col = frame_columns[j]
            if np.isnan(value):
                assert np.isnan(frame[col][i])
            else:
                assert value == frame[col][i]

    # mixed type
    mat = pd.DataFrame(test_data.mixed_frame).as_matrix(["foo", "A"])
    assert mat[0, 0] == "bar"

    df = pd.DataFrame({"real": [1, 2, 3], "complex": [1j, 2j, 3j]})
    mat = df.as_matrix()
    if PY2:
        assert mat[0, 0] == 1j
    else:
        assert mat[0, 1] == 1j

    # single block corner case
    mat = pd.DataFrame(test_data.frame).as_matrix(["A", "B"])
    expected = test_data.frame.reindex(columns=["A", "B"]).values
    tm.assert_almost_equal(mat, expected)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_asfreq(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.asfreq(None)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_asof(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.asof(None)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_assign(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.assign()


def test_astype():
    td = TestData()
    modin_df = pd.DataFrame(
        td.frame.values, index=td.frame.index, columns=td.frame.columns
    )
    expected_df = pandas.DataFrame(
        td.frame.values, index=td.frame.index, columns=td.frame.columns
    )

    modin_df_casted = modin_df.astype(np.int32)
    expected_df_casted = expected_df.astype(np.int32)
    df_equals(modin_df_casted, expected_df_casted)

    modin_df_casted = modin_df.astype(np.float64)
    expected_df_casted = expected_df.astype(np.float64)
    df_equals(modin_df_casted, expected_df_casted)

    modin_df_casted = modin_df.astype(str)
    expected_df_casted = expected_df.astype(str)
    df_equals(modin_df_casted, expected_df_casted)

    dtype_dict = {
            "A": np.int32,
            "B": np.int64,
            "C": str
            }
    modin_df_casted = modin_df.astype(dtype_dict)
    expected_df_casted = expected_df.astype(dtype_dict)
    df_equals(modin_df_casted, expected_df_casted)

    bad_dtype_dict = {
            "B": np.int32,
            "B": np.int64,
            "B": str
            }
    modin_df_casted = modin_df.astype(bad_dtype_dict)
    expected_df_casted = expected_df.astype(bad_dtype_dict)
    df_equals(modin_df_casted, expected_df_casted)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_at_time(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.at_time(None)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_between_time(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.between_time(None, None)


def test_bfill():
    test_data = TestData()
    test_data.tsframe["A"][:5] = np.nan
    test_data.tsframe["A"][-5:] = np.nan
    modin_df = pd.DataFrame(test_data.tsframe)
    df_equals(modin_df.bfill(), test_data.tsframe.bfill())


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_blocks(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.blocks


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_bool(modin_df, pandas_df):
    with pytest.raises(ValueError):
        modin_df.bool()
        modin_df.__bool__()

    single_bool_pandas_df = pandas.DataFrame([True])
    single_bool_modin_df = pd.DataFrame([True])

    assert single_bool_pandas_df.bool() == single_bool_modin_df.bool()

    with pytest.raises(ValueError):
        # __bool__ always raises this error for DataFrames
        single_bool_modin_df.__bool__()


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_boxplot(modin_df, pandas_df):
    assert modin_df.boxplot() == to_pandas(modin_df).boxplot()


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
def test_clip(request, modin_df, pandas_df, axis):
    if name_contains(request.node.name, numeric_dfs):
        ind_len = len(modin_df.columns) if axis == 1 or axis == 'columns' else len(modin_df.index)
        # set bounds
        lower, upper = np.sort(random_state.random_integers(RAND_LOW, RAND_HIGH, 2))
        lower_list = random_state.random_integers(RAND_LOW, RAND_HIGH, ind_len)
        upper_list = random_state.random_integers(RAND_LOW, RAND_HIGH, ind_len)

        # test only upper scalar bound
        modin_result = modin_df.clip(None, upper, axis=axis)
        pandas_result = pandas_df.clip(None, upper, axis=axis)
        df_equals(modin_result, pandas_result)

        # test lower and upper scalar bound
        modin_result = modin_df.clip(lower, upper, axis=axis)
        pandas_result = pandas_df.clip(lower, upper, axis=axis)
        df_equals(modin_result, pandas_result)

        # test lower and upper list bound on each column
        modin_result = modin_df.clip(lower_list, upper_list, axis=axis)
        pandas_result = pandas_df.clip(lower_list, upper_list, axis=axis)
        df_equals(modin_result, pandas_result)

        # test only upper list bound on each column
        modin_result = modin_df.clip(np.nan, upper_list, axis=axis)
        pandas_result = pandas_df.clip(np.nan, upper_list, axis=axis)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
def test_clip_lower(modin_df, pandas_df, axis):
    if name_contains(request.node.name, numeric_dfs):
        ind_len = len(modin_df.index) if axis else len(modin_df.columns)
        # set bounds
        lower = random_state.random_integer(RAND_LOW, RAND_HIGH)
        lower_list = random_state.random_integer(RAND_LOW, RAND_HIGH, ind_len)

        # test lower scalar bound
        modin_result = modin_df.clip_lower(lower, axis=axis)
        pandas_result = pandas_df.clip_lower(lower, axis=axis)
        df_equals(modin_result, pandas_result)

        # test lower list bound on each column
        modin_result = modin_df.clip_lower(lower_list, axis=axis)
        pandas_result = pandas_df.clip_lower(lower_list, axis=axis)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
def test_clip_upper(modin_df, pandas_df, axis):
    if name_contains(request.node.name, numeric_dfs):
        ind_len = len(modin_df.index) if axis else len(modin_df.columns)
        # set bounds
        upper = random_state.random_integer(RAND_LOW, RAND_HIGH)
        upper_list = random_state.random_integer(RAND_LOW, RAND_HIGH, ind_len)

        # test upper scalar bound
        modin_result = modin_df.clip_upper(upper, axis=axis)
        pandas_result = pandas_df.clip_upper(upper, axis=axis)
        df_equals(modin_result, pandas_result)

        # test upper list bound on each column
        modin_result = modin_df.clip_upper(upper_list, axis=axis)
        pandas_result = pandas_df.clip_upper(upper_list, axis=axis)
        df_equals(modin_result, pandas_result)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_combine(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.combine(None, None)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_combine_first(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.combine_first(None)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_compound(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.compound()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_consolidate(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.consolidate()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_convert_objects(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.convert_objects()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_corr(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.corr()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_corrwith(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.corrwith(None)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
)
def test_count(request, modin_df, pandas_df, axis, numeric_only):
    modin_result = modin_df.count(axis=axis, numeric_only=numeric_only)
    pandas_result = pandas_df.count(axis=axis, numeric_only=numeric_only)
    df_equals(modin_result, pandas_result)



@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_cov(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.cov()


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_cummax(request, modin_df, pandas_df, axis, skipna):
    try:
        pandas_result = pandas_df.cummax(axis=axis, skipna=skipna)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.cummax(axis=axis, skipna=skipna)
        return
    modin_result = modin_df.cummax(axis=axis, skipna=skipna)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_cummin(request, modin_df, pandas_df, axis, skipna):
    try:
        pandas_result = pandas_df.cummin(axis=axis, skipna=skipna)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.cummin(axis=axis, skipna=skipna)
        return
    modin_result = modin_df.cummin(axis=axis, skipna=skipna)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_cumprod(request, modin_df, pandas_df, axis, skipna):
    try:
        pandas_result = pandas_df.cumprod(axis=axis, skipna=skipna)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.cumprod(axis=axis, skipna=skipna)
        return
    modin_result = modin_df.cumprod(axis=axis, skipna=skipna)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_cumsum(request, modin_df, pandas_df, axis, skipna):
    try:
        pandas_result = pandas_df.cumsum(axis=axis, skipna=skipna)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.cumsum(axis=axis, skipna=skipna)
        return
    modin_result = modin_df.cumsum(axis=axis, skipna=skipna)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_describe(modin_df, pandas_df):
    df_equals(modin_df.describe(), pandas_df.describe())


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "periods", int_arg_values, ids=arg_keys("periods", int_arg_keys)
)
def test_diff(request, modin_df, pandas_df, axis, periods):
    try:
        pandas_result = pandas_df.diff(axis=axis, periods=periods)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.diff(axis=axis, periods=periods)
        return
    modin_result = modin_df.diff(axis=axis, periods=periods)
    df_equals(modin_result, pandas_result)


def test_drop():
    frame_data = {"A": [1, 2, 3, 4], "B": [0, 1, 2, 3]}
    simple = pandas.DataFrame(frame_data)
    modin_simple = pd.DataFrame(frame_data)
    df_equals(modin_simple.drop("A", axis=1), simple[["B"]])
    df_equals(modin_simple.drop(["A", "B"], axis="columns"), simple[[]])
    df_equals(modin_simple.drop([0, 1, 3], axis=0), simple.loc[[2], :])
    df_equals(modin_simple.drop([0, 3], axis="index"), simple.loc[[1, 2], :])

    pytest.raises(ValueError, modin_simple.drop, 5)
    pytest.raises(ValueError, modin_simple.drop, "C", 1)
    pytest.raises(ValueError, modin_simple.drop, [1, 5])
    pytest.raises(ValueError, modin_simple.drop, ["A", "C"], 1)

    # errors = 'ignore'
    df_equals(modin_simple.drop(5, errors="ignore"), simple)
    df_equals(
        modin_simple.drop([0, 5], errors="ignore"), simple.loc[[1, 2, 3], :]
    )
    df_equals(modin_simple.drop("C", axis=1, errors="ignore"), simple)
    df_equals(
        modin_simple.drop(["A", "C"], axis=1, errors="ignore"), simple[["B"]]
    )

    # non-unique
    nu_df = pandas.DataFrame(
        pandas.compat.lzip(range(3), range(-3, 1), list("abc")), columns=["a", "a", "b"]
    )
    modin_nu_df = pd.DataFrame(nu_df)
    df_equals(modin_nu_df.drop("a", axis=1), nu_df[["b"]])
    df_equals(modin_nu_df.drop("b", axis="columns"), nu_df["a"])
    df_equals(modin_nu_df.drop([]), nu_df)

    nu_df = nu_df.set_index(pandas.Index(["X", "Y", "X"]))
    nu_df.columns = list("abc")
    modin_nu_df = pd.DataFrame(nu_df)
    df_equals(modin_nu_df.drop("X", axis="rows"), nu_df.loc[["Y"], :])
    df_equals(modin_nu_df.drop(["X", "Y"], axis=0), nu_df.loc[[], :])

    # inplace cache issue
    frame_data = random_state.randn(10, 3)
    df = pandas.DataFrame(frame_data, columns=list("abc"))
    modin_df = pd.DataFrame(frame_data, columns=list("abc"))
    expected = df[~(df.b > 0)]
    modin_df.drop(labels=df[df.b > 0].index, inplace=True)
    df_equals(modin_df, expected)


def test_drop_api_equivalence():
    # equivalence of the labels/axis and index/columns API's
    frame_data = [[1, 2, 3], [3, 4, 5], [5, 6, 7]]

    modin_df = pd.DataFrame(frame_data, index=["a", "b", "c"], columns=["d", "e", "f"])

    modin_df1 = modin_df.drop("a")
    modin_df2 = modin_df.drop(index="a")
    df_equals(modin_df1, modin_df2)

    modin_df1 = modin_df.drop("d", 1)
    modin_df2 = modin_df.drop(columns="d")
    df_equals(modin_df1, modin_df2)

    modin_df1 = modin_df.drop(labels="e", axis=1)
    modin_df2 = modin_df.drop(columns="e")
    df_equals(modin_df1, modin_df2)

    modin_df1 = modin_df.drop(["a"], axis=0)
    modin_df2 = modin_df.drop(index=["a"])
    df_equals(modin_df1, modin_df2)

    modin_df1 = modin_df.drop(["a"], axis=0).drop(["d"], axis=1)
    modin_df2 = modin_df.drop(index=["a"], columns=["d"])
    df_equals(modin_df1, modin_df2)

    with pytest.raises(ValueError):
        modin_df.drop(labels="a", index="b")

    with pytest.raises(ValueError):
        modin_df.drop(labels="a", columns="b")

    with pytest.raises(ValueError):
        modin_df.drop(axis=1)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_drop_duplicates(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.drop_duplicates()


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("how", ["any", "all"], ids=["any", "all"])
def test_dropna(modin_df, pandas_df, axis, how):
    modin_result = modin_df.dropna(axis=axis, how=how)
    pandas_result = pandas_df.dropna(axis=axis, how=how)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_dropna_inplace(modin_df, pandas_df):
    pandas_result = pandas_df.dropna()
    modin_df.dropna(inplace=True)
    df_equals(modin_df, pandas_result)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_dropna_multiple_axes(modin_df, pandas_df):
    df_equals(
        modin_df.dropna(how="all", axis=[0, 1]),
        pandas_df.dropna(how="all", axis=[0, 1]),
    )
    df_equals(
        modin_df.dropna(how="all", axis=(0, 1)),
        pandas_df.dropna(how="all", axis=(0, 1)),
    )


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_dropna_multiple_axes_inplace(modin_df, pandas_df):
    modin_df_copy = modin_df.copy()
    pandas_df_copy = pandas_df.copy()

    modin_df_copy.dropna(how="all", axis=[0, 1], inplace=True)
    pandas_df_copy.dropna(how="all", axis=[0, 1], inplace=True)

    df_equals(modin_df_copy, pandas_df_copy)

    modin_df_copy = modin_df.copy()
    pandas_df_copy = pandas_df.copy()

    modin_df_copy.dropna(how="all", axis=(0, 1), inplace=True)
    pandas_df_copy.dropna(how="all", axis=(0, 1), inplace=True)

    df_equals(modin_df_copy, pandas_df_copy)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_dropna_subset(request, modin_df, pandas_df):
    if "empty_data" not in request.node.name:
        column_subset = modin_df.columns[0:2]
        df_equals(
            modin_df.dropna(how="all", subset=column_subset),
            pandas_df.dropna(how="all", subset=column_subset),
        )
        df_equals(
            modin_df.dropna(how="any", subset=column_subset),
            pandas_df.dropna(how="any", subset=column_subset),
        )

        row_subset = modin_df.index[0:2]
        df_equals(
            modin_df.dropna(how="all", axis=1, subset=row_subset),
            pandas_df.dropna(how="all", axis=1, subset=row_subset),
        )
        df_equals(
            modin_df.dropna(how="any", axis=1, subset=row_subset),
            pandas_df.dropna(how="any", axis=1, subset=row_subset),
        )


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_dropna_subset_error(modin_df, pandas_df):
    # pandas_df is unused but there so there won't be confusing list comprehension
    # stuff in the pytest.mark.parametrize
    with pytest.raises(KeyError):
        modin_df.dropna(subset=list("EF"))

    with pytest.raises(KeyError):
        modin_df.dropna(axis=1, subset=[4, 5])


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_dot(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.dot(None)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_duplicated(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.duplicated()


def test_empty_df():
    df = pd.DataFrame(index=["a", "b"])
    df_is_empty(df)
    tm.assert_index_equal(df.index, pd.Index(["a", "b"]))
    assert len(df.columns) == 0

    df = pd.DataFrame(columns=["a", "b"])
    df_is_empty(df)
    assert len(df.index) == 0
    tm.assert_index_equal(df.columns, pd.Index(["a", "b"]))

    df = pd.DataFrame()
    df_is_empty(df)
    assert len(df.index) == 0
    assert len(df.columns) == 0

    df = pd.DataFrame(index=["a", "b"])
    df_is_empty(df)
    tm.assert_index_equal(df.index, pd.Index(["a", "b"]))
    assert len(df.columns) == 0

    df = pd.DataFrame(columns=["a", "b"])
    df_is_empty(df)
    assert len(df.index) == 0
    tm.assert_index_equal(df.columns, pd.Index(["a", "b"]))

    df = pd.DataFrame()
    df_is_empty(df)
    assert len(df.index) == 0
    assert len(df.columns) == 0


def test_equals():
    frame_data = {"col1": [2.9, 3, 3, 3], "col2": [2, 3, 4, 1]}
    modin_df1 = pd.DataFrame(frame_data)
    modin_df2 = pd.DataFrame(frame_data)

    df_equals(modin_df1, modin_df2)

    frame_data = {"col1": [2.9, 3, 3, 3], "col2": [2, 3, 5, 1]}
    modin_df3 = pd.DataFrame(frame_data)

    try:
        df_equals(modin_df3, modin_df1)
    except AssertionError:
        pass
    else:
        raise AssertionError

    try:
        df_equals(modin_df3, modin_df2)
    except AssertionError:
        pass
    else:
        raise AssertionError


def test_eval_df_use_case():
    frame_data = {"a": random_state.randn(10), "b": random_state.randn(10)}
    df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)

    # test eval for series results
    tmp_pandas = df.eval("arctan2(sin(a), b)", engine="python", parser="pandas")
    tmp_modin = modin_df.eval("arctan2(sin(a), b)", engine="python", parser="pandas")

    assert isinstance(tmp_modin, pandas.Series)
    df_equals(tmp_modin, tmp_pandas)

    # Test not inplace assignments
    tmp_pandas = df.eval("e = arctan2(sin(a), b)", engine="python", parser="pandas")
    tmp_modin = modin_df.eval(
        "e = arctan2(sin(a), b)", engine="python", parser="pandas"
    )
    df_equals(tmp_modin, tmp_pandas)

    # Test inplace assignments
    df.eval("e = arctan2(sin(a), b)", engine="python", parser="pandas", inplace=True)
    modin_df.eval(
        "e = arctan2(sin(a), b)", engine="python", parser="pandas", inplace=True
    )
    # TODO: Use a series equality validator.
    df_equals(modin_df, df)


def test_eval_df_arithmetic_subexpression():
    frame_data = {"a": random_state.randn(10), "b": random_state.randn(10)}
    df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)
    df.eval("not_e = sin(a + b)", engine="python", parser="pandas", inplace=True)
    modin_df.eval("not_e = sin(a + b)", engine="python", parser="pandas", inplace=True)
    # TODO: Use a series equality validator.
    df_equals(modin_df, df)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_ewm(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.ewm()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_expanding(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.expanding()


def test_ffill():
    test_data = TestData()
    test_data.tsframe["A"][:5] = np.nan
    test_data.tsframe["A"][-5:] = np.nan
    modin_df = pd.DataFrame(test_data.tsframe)

    df_equals(modin_df.ffill(), test_data.tsframe.ffill())


def test_fillna_sanity():
    test_data = TestData()
    tf = test_data.tsframe
    tf.loc[tf.index[:5], "A"] = np.nan
    tf.loc[tf.index[-5:], "A"] = np.nan

    zero_filled = test_data.tsframe.fillna(0)
    modin_df = pd.DataFrame(test_data.tsframe).fillna(0)
    df_equals(modin_df, zero_filled)

    padded = test_data.tsframe.fillna(method="pad")
    modin_df = pd.DataFrame(test_data.tsframe).fillna(method="pad")
    df_equals(modin_df, padded)

    # mixed type
    mf = test_data.mixed_frame
    mf.loc[mf.index[5:20], "foo"] = np.nan
    mf.loc[mf.index[-10:], "A"] = np.nan

    result = test_data.mixed_frame.fillna(value=0)
    modin_df = pd.DataFrame(test_data.mixed_frame).fillna(value=0)
    df_equals(modin_df, result)

    result = test_data.mixed_frame.fillna(method="pad")
    modin_df = pd.DataFrame(test_data.mixed_frame).fillna(method="pad")
    df_equals(modin_df, result)

    pytest.raises(ValueError, test_data.tsframe.fillna)
    pytest.raises(ValueError, pd.DataFrame(test_data.tsframe).fillna)
    with pytest.raises(ValueError):
        pd.DataFrame(test_data.tsframe).fillna(5, method="ffill")

    # mixed numeric (but no float16)
    mf = test_data.mixed_float.reindex(columns=["A", "B", "D"])
    mf.loc[mf.index[-10:], "A"] = np.nan
    result = mf.fillna(value=0)
    modin_df = pd.DataFrame(mf).fillna(value=0)
    df_equals(modin_df, result)

    result = mf.fillna(method="pad")
    modin_df = pd.DataFrame(mf).fillna(method="pad")
    df_equals(modin_df, result)

    # TODO: Use this when Arrow issue resolves:
    # (https://issues.apache.org/jira/browse/ARROW-2122)
    # empty frame
    # df = DataFrame(columns=['x'])
    # for m in ['pad', 'backfill']:
    #     df.x.fillna(method=m, inplace=True)
    #     df.x.fillna(method=m)

    # with different dtype
    frame_data = [
        ["a", "a", np.nan, "a"],
        ["b", "b", np.nan, "b"],
        ["c", "c", np.nan, "c"],
    ]
    df = pandas.DataFrame(frame_data)

    result = df.fillna({2: "foo"})
    modin_df = pd.DataFrame(frame_data).fillna({2: "foo"})

    df_equals(modin_df, result)

    modin_df = pd.DataFrame(df)
    df.fillna({2: "foo"}, inplace=True)
    modin_df.fillna({2: "foo"}, inplace=True)
    df_equals(modin_df, result)

    frame_data = {
        "Date": [pandas.NaT, pandas.Timestamp("2014-1-1")],
        "Date2": [pandas.Timestamp("2013-1-1"), pandas.NaT],
    }
    df = pandas.DataFrame(frame_data)
    result = df.fillna(value={"Date": df["Date2"]})
    modin_df = pd.DataFrame(frame_data).fillna(value={"Date": df["Date2"]})
    df_equals(modin_df, result)

    # TODO: Use this when Arrow issue resolves:
    # (https://issues.apache.org/jira/browse/ARROW-2122)
    # with timezone
    """
    frame_data = {'A': [pandas.Timestamp('2012-11-11 00:00:00+01:00'),
                        pandas.NaT]}
    df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)
    df_equals(modin_df.fillna(method='pad'), df.fillna(method='pad'))

    frame_data = {'A': [pandas.NaT,
                        pandas.Timestamp('2012-11-11 00:00:00+01:00')]}
    df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data).fillna(method='bfill')
    df_equals(modin_df, df.fillna(method='bfill'))
    """


def test_fillna_downcast():
    # infer int64 from float64
    frame_data = {"a": [1.0, np.nan]}
    df = pandas.DataFrame(frame_data)
    result = df.fillna(0, downcast="infer")
    modin_df = pd.DataFrame(frame_data).fillna(0, downcast="infer")
    df_equals(modin_df, result)

    # infer int64 from float64 when fillna value is a dict
    df = pandas.DataFrame(frame_data)
    result = df.fillna({"a": 0}, downcast="infer")
    modin_df = pd.DataFrame(frame_data).fillna({"a": 0}, downcast="infer")
    df_equals(modin_df, result)


def test_ffill2():
    test_data = TestData()
    test_data.tsframe["A"][:5] = np.nan
    test_data.tsframe["A"][-5:] = np.nan
    modin_df = pd.DataFrame(test_data.tsframe)
    df_equals(
        modin_df.fillna(method="ffill"), test_data.tsframe.fillna(method="ffill")
    )


def test_bfill2():
    test_data = TestData()
    test_data.tsframe["A"][:5] = np.nan
    test_data.tsframe["A"][-5:] = np.nan
    modin_df = pd.DataFrame(test_data.tsframe)
    df_equals(
        modin_df.fillna(method="bfill"), test_data.tsframe.fillna(method="bfill")
    )


def test_fillna_inplace():
    frame_data = random_state.randn(10, 4)
    df = pandas.DataFrame(frame_data)
    df[1][:4] = np.nan
    df[3][-4:] = np.nan

    modin_df = pd.DataFrame(df)
    df.fillna(value=0, inplace=True)
    assert not df_equals(modin_df, df)

    modin_df.fillna(value=0, inplace=True)
    df_equals(modin_df, df)

    modin_df = pd.DataFrame(df).fillna(value={0: 0}, inplace=True)
    assert modin_df is None

    df[1][:4] = np.nan
    df[3][-4:] = np.nan
    modin_df = pd.DataFrame(df)
    df.fillna(method="ffill", inplace=True)

    assert not df_equals(modin_df, df)

    modin_df.fillna(method="ffill", inplace=True)
    df_equals(modin_df, df)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_frame_fillna_limit(modin_df, pandas_df):
    index = pandas_df.index

    result = pandas_df[:2].reindex(index)
    modin_df = pd.DataFrame(result)
    df_equals(
        modin_df.fillna(method="pad", limit=2), result.fillna(method="pad", limit=2)
    )

    result = pandas_df[-2:].reindex(index)
    modin_df = pd.DataFrame(result)
    df_equals(
        modin_df.fillna(method="backfill", limit=2),
        result.fillna(method="backfill", limit=2),
    )


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_frame_pad_backfill_limit(modin_df, pandas_df):
    index = pandas_df.index

    result = pandas_df[:2].reindex(index)
    modin_df = pd.DataFrame(result)
    df_equals(
        modin_df.fillna(method="pad", limit=2), result.fillna(method="pad", limit=2)
    )

    result = pandas_df[-2:].reindex(index)
    modin_df = pd.DataFrame(result)
    df_equals(
        modin_df.fillna(method="backfill", limit=2),
        result.fillna(method="backfill", limit=2),
    )


def test_fillna_dtype_conversion():
    # make sure that fillna on an empty frame works
    df = pandas.DataFrame(index=range(3), columns=["A", "B"], dtype="float64")
    modin_df = pd.DataFrame(index=range(3), columns=["A", "B"], dtype="float64")
    df_equals(modin_df.fillna("nan"), df.fillna("nan"))

    frame_data = {"A": [1, np.nan], "B": [1.0, 2.0]}
    df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)
    for v in ["", 1, np.nan, 1.0]:
        df_equals(modin_df.fillna(v), df.fillna(v))


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_fillna_skip_certain_blocks(modin_df, pandas_df):
    # don't try to fill boolean, int blocks
    df_equals(modin_df.fillna(np.nan), pandas_df.fillna(np.nan))


def test_fillna_dict_series():
    frame_data = {
        "a": [np.nan, 1, 2, np.nan, np.nan],
        "b": [1, 2, 3, np.nan, np.nan],
        "c": [np.nan, 1, 2, 3, 4],
    }
    df = pandas.DataFrame(frame_data)
    ray_df = pd.DataFrame(frame_data)

    df_equals(ray_df.fillna({"a": 0, "b": 5}), df.fillna({"a": 0, "b": 5}))

    df_equals(ray_df.fillna({"a": 0, "b": 5, "d": 7}), df.fillna({"a": 0, "b": 5, "d": 7}))

    # Series treated same as dict
    with pytest.raises(NotImplementedError):
        df_equals(ray_df.fillna(df.max()), df.fillna(df.max()))


def test_fillna_dataframe():
    frame_data = {
        "a": [np.nan, 1, 2, np.nan, np.nan],
        "b": [1, 2, 3, np.nan, np.nan],
        "c": [np.nan, 1, 2, 3, 4],
    }
    df = pandas.DataFrame(frame_data, index=list("VWXYZ"))
    modin_df = pd.DataFrame(frame_data, index=list("VWXYZ"))

    # df2 may have different index and columns
    df2 = pandas.DataFrame(
        {"a": [np.nan, 10, 20, 30, 40], "b": [50, 60, 70, 80, 90], "foo": ["bar"] * 5},
        index=list("VWXuZ"),
    )

    # only those columns and indices which are shared get filled
    with pytest.raises(NotImplementedError):
        df_equals(modin_df.fillna(df2), df.fillna(df2))


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_fillna_columns(modin_df, pandas_df):
    df_equals(
        modin_df.fillna(method="ffill", axis=1),
        pandas_df.fillna(method="ffill", axis=1),
    )

    df_equals(
        modin_df.fillna(method="ffill", axis=1),
        pandas_df.fillna(method="ffill", axis=1),
    )


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_fillna_invalid_method(modin_df, pandas_df):
    with tm.assert_raises_regex(ValueError, "ffil"):
        modin_df.fillna(method="ffil")


def test_fillna_invalid_value():
    test_data = TestData()
    ray_df = pd.DataFrame(test_data.frame)
    # list
    pytest.raises(TypeError, ray_df.fillna, [1, 2])
    # tuple
    pytest.raises(TypeError, ray_df.fillna, (1, 2))
    # frame with series
    pytest.raises(TypeError, ray_df.iloc[:, 0].fillna, ray_df)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_fillna_col_reordering(modin_df, pandas_df):
    df_equals(modin_df.fillna(method="ffill"), pandas_df.fillna(method="ffill"))


"""
TODO: Use this when Arrow issue resolves:
(https://issues.apache.org/jira/browse/ARROW-2122)
def test_fillna_datetime_columns():
    frame_data = {'A': [-1, -2, np.nan],
                  'B': date_range('20130101', periods=3),
                  'C': ['foo', 'bar', None],
                  'D': ['foo2', 'bar2', None]}
    df = pandas.DataFrame(frame_data, index=date_range('20130110', periods=3))
    modin_df = pd.DataFrame(frame_data, index=date_range('20130110', periods=3))
    df_equals(modin_df.fillna('?'), df.fillna('?'))

    frame_data = {'A': [-1, -2, np.nan],
                  'B': [pandas.Timestamp('2013-01-01'),
                        pandas.Timestamp('2013-01-02'), pandas.NaT],
                  'C': ['foo', 'bar', None],
                  'D': ['foo2', 'bar2', None]}
    df = pandas.DataFrame(frame_data, index=date_range('20130110', periods=3))
    modin_df = pd.DataFrame(frame_data, index=date_range('20130110', periods=3))
    df_equals(modin_df.fillna('?'), df.fillna('?'))
"""


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_filter(modin_df, pandas_df):
    by = {"items": ["col1", "col5"], "regex": "4$|3$", "like": "col"}
    df_equals(
        modin_df.filter(items=by["items"]), pandas_df.filter(items=by["items"])
    )

    df_equals(
        modin_df.filter(regex=by["regex"], axis=0),
        pandas_df.filter(regex=by["regex"], axis=0),
    )
    df_equals(
        modin_df.filter(regex=by["regex"], axis=1),
        pandas_df.filter(regex=by["regex"], axis=1),
    )

    df_equals(
        modin_df.filter(like=by["like"]), pandas_df.filter(like=by["like"])
    )


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_first(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.first(None)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_first_valid_index(modin_df, pandas_df):
    assert modin_df.first_valid_index() == (pandas_df.first_valid_index())


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_from_csv(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        pd.DataFrame.from_csv(None)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_from_dict(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        pd.DataFrame.from_dict(None)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_from_items(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        pd.DataFrame.from_items(None)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_from_records(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        pd.DataFrame.from_records(None)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_get_value(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.get_value(None, None)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_get_values(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.get_values()


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("n", int_arg_values, ids=arg_keys("n", int_arg_keys))
def test_head(modin_df, pandas_df, n):
    df_equals(modin_df.head(n), pandas_df.head(n))


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_hist(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.hist(None)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_iat(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.iat()


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_idxmax(modin_df, pandas_df, axis, skipna):
    modin_result = modin_df.all(axis=axis, skipna=skipna)
    pandas_result = pandas_df.all(axis=axis, skipna=skipna)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_idxmin(modin_df, pandas_df, axis, skipna):
    modin_result = modin_df.all(axis=axis, skipna=skipna)
    pandas_result = pandas_df.all(axis=axis, skipna=skipna)
    df_equals(modin_result, pandas_result)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_infer_objects(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.infer_objects()


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_iloc(request, modin_df, pandas_df):
    if not name_contains(request.node.name, ["empty_data"]):
        # Scaler
        assert modin_df.iloc[0, 1] == pandas_df.iloc[0, 1]

        # Series
        df_equals(modin_df.iloc[0], pandas_df.iloc[0])
        df_equals(modin_df.iloc[1:, 0], pandas_df.iloc[1:, 0])
        df_equals(modin_df.iloc[1:2, 0], pandas_df.iloc[1:2, 0])

        # DataFrame
        df_equals(modin_df.iloc[[1, 2]], pandas_df.iloc[[1, 2]])
        # See issue #80
        # df_equals(modin_df.iloc[[1, 2], [1, 0]], pandas_df.iloc[[1, 2], [1, 0]])
        df_equals(modin_df.iloc[1:2, 0:2], pandas_df.iloc[1:2, 0:2])

        # Issue #43
        modin_df.iloc[0:3, :]

        # Write Item
        modin_df.iloc[[1, 2]] = 42
        pandas_df.iloc[[1, 2]] = 42
        df_equals(modin_df, pandas_df)
    else:
        with pytest.raises(IndexError):
            modin_df.iloc[0, 1]


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_index(modin_df, pandas_df):
    df_equals(modin_df.index, pandas_df.index)
    modin_df_cp = modin_df.copy()
    pandas_df_cp = pandas_df.copy()

    modin_df_cp.index = [str(i) for i in modin_df_cp.index]
    pandas_df_cp.index = [str(i) for i in pandas_df_cp.index]
    df_equals(modin_df_cp.index, pandas_df_cp.index)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_info(request, modin_df, pandas_df):
    # Test to make sure that it does not crash
    modin_df.info(memory_usage="deep")

    if not name_contains(request.node.name, ["empty_data"]):
        with io.StringIO() as buf:
            modin_df.info(buf=buf)
            info_string = buf.getvalue()
            assert "<class 'modin.pandas.dataframe.DataFrame'>\n" in info_string
            assert "memory usage: " in info_string
            assert (
                "Data columns (total {} columns):".format(modin_df.shape[1])
                in info_string
            )

        with io.StringIO() as buf:
            modin_df.info(buf=buf, verbose=False, memory_usage=False)
            info_string = buf.getvalue()
            assert "memory usage: " not in info_string
            assert (
                "Columns: {0} entries, {1} to {2}".format(
                    modin_df.shape[1], modin_df.columns[0], modin_df.columns[-1]
                )
                in info_string
            )


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("loc", int_arg_values, ids=arg_keys("loc", int_arg_keys))
def test_insert(modin_df, pandas_df, loc):
    modin_df = modin_df.copy()
    pandas_df = pandas_df.copy()
    loc %= modin_df.shape[1] + 1
    column = "New Column"
    key = loc if loc < modin_df.shape[1] else loc - 1
    value = modin_df.iloc[:, key]
    modin_df.insert(loc, column, value)
    pandas_df.insert(loc, column, value)
    df_equals(modin_df, pandas_df)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_interpolate(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.interpolate()


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_is_copy(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.is_copy


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_items(modin_df, pandas_df):
    modin_items = modin_df.items()
    pandas_items = pandas_df.items()
    for modin_item, pandas_item in zip(modin_items, pandas_items):
        modin_index, modin_series = modin_item
        pandas_index, pandas_series = pandas_item
        df_equals(pandas_series, modin_series)
        assert pandas_index == modin_index


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_iteritems(modin_df, pandas_df):
    modin_items = modin_df.iteritems()
    pandas_items = pandas_df.iteritems()
    for modin_item, pandas_item in zip(modin_items, pandas_items):
        modin_index, modin_series = modin_item
        pandas_index, pandas_series = pandas_item
        df_equals(pandas_series, modin_series)
        assert pandas_index == modin_index


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_iterrows(modin_df, pandas_df):
    modin_iterrows = modin_df.iterrows()
    pandas_iterrows = pandas_df.iterrows()
    for modin_row, pandas_row in zip(modin_iterrows, pandas_iterrows):
        modin_index, modin_series = modin_row
        pandas_index, pandas_series = pandas_row
        df_equals(pandas_series, modin_series)
        assert pandas_index == modin_index


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_itertuples(modin_df, pandas_df):
    # test default
    modin_it_default = modin_df.itertuples()
    pandas_it_default = pandas_df.itertuples()
    for modin_row, pandas_row in zip(modin_it_default, pandas_it_default):
        np.testing.assert_equal(modin_row, pandas_row)

    # test all combinations of custom params
    indices = [True, False]
    names = [None, "NotPandas", "Pandas"]

    for index in indices:
        for name in names:
            modin_it_custom = modin_df.itertuples(index=index, name=name)
            pandas_it_custom = pandas_df.itertuples(index=index, name=name)
            for modin_row, pandas_row in zip(modin_it_custom, pandas_it_custom):
                np.testing.assert_equal(modin_row, pandas_row)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_ix(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.ix()


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


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_keys(modin_df, pandas_df):
    df_equals(modin_df.keys(), pandas_df.keys())


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_kurt(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.kurt()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_kurtosis(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.kurtosis()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_last(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.last(None)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_last_valid_index(modin_df, pandas_df):
    assert modin_df.last_valid_index() == (pandas_df.last_valid_index())


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_loc(request, modin_df, pandas_df):
    # We skip nan datasets because nan != nan
    if "nan" not in request.node.name:
        key1 = modin_df.columns[0]
        key2 = modin_df.columns[1]
        # Scaler
        assert modin_df.loc[0, key1] == pandas_df.loc[0, key1]

        # Series
        df_equals(modin_df.loc[0], pandas_df.loc[0])
        df_equals(modin_df.loc[1:, key1], pandas_df.loc[1:, key1])
        df_equals(modin_df.loc[1:2, key1], pandas_df.loc[1:2, key1])

        # DataFrame
        df_equals(modin_df.loc[[1, 2]], pandas_df.loc[[1, 2]])

        # See issue #80
        # df_equals(modin_df.loc[[1, 2], ['col1']], pandas_df.loc[[1, 2], ['col1']])
        df_equals(modin_df.loc[1:2, key1:key2], pandas_df.loc[1:2, key1:key2])

        # Write Item
        modin_df_copy = modin_df.copy()
        pandas_df_copy = pandas_df.copy()
        modin_df_copy.loc[[1, 2]] = 42
        pandas_df_copy.loc[[1, 2]] = 42
        df_equals(modin_df_copy, pandas_df_copy)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_lookup(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.lookup(None, None)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_mad(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.mad()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_mask(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.mask(None)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "numeric_only",
    bool_none_arg_values,
    ids=arg_keys("numeric_only", bool_none_arg_keys),
)
def test_max(request, modin_df, pandas_df, axis, skipna, numeric_only):
    try:
        pandas_result = pandas_df.max(axis=axis, skipna=skipna, numeric_only=numeric_only)
    except Exception:
        with pytest.raises(TypeError):
            modin_result = modin_df.max(axis=axis, skipna=skipna, numeric_only=numeric_only)
        return
    modin_result = modin_df.max(axis=axis, skipna=skipna, numeric_only=numeric_only)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "numeric_only", bool_none_arg_values, ids=arg_keys("numeric_only", bool_none_arg_keys),
)
def test_mean(request, modin_df, pandas_df, axis, skipna, numeric_only):
    try:
        pandas_result = pandas_df.mean(axis=axis, skipna=skipna, numeric_only=numeric_only)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.mean(axis=axis, skipna=skipna, numeric_only=numeric_only)
        return
    modin_result = modin_df.mean(axis=axis, skipna=skipna, numeric_only=numeric_only)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_none_arg_values, ids=arg_keys("skipna", bool_none_arg_keys)
)
@pytest.mark.parametrize(
    "numeric_only",
    bool_arg_values,
    ids=arg_keys("numeric_only", bool_arg_keys),
)
def test_median(request, modin_df, pandas_df, axis, skipna, numeric_only):
    try:
        pandas_result = pandas_df.median(
            axis=axis, skipna=skipna, numeric_only=numeric_only
        )
    except Exception:
        with pytest.raises(TypeError):
            modin_df.median(axis=axis, skipna=skipna, numeric_only=numeric_only)
        return
    modin_result = modin_df.median(axis=axis, skipna=skipna, numeric_only=numeric_only)
    df_equals(modin_result, pandas_result)

@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_melt(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.melt()


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_memory_usage(modin_df, pandas_df):
    assert modin_df.memory_usage(index=True).at["Index"] is not None
    assert (
        modin_df.memory_usage(deep=True).sum()
        >= modin_df.memory_usage(deep=False).sum()
    )


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
        with pytest.raises(NotImplementedError):
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


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "numeric_only",
    bool_none_arg_values,
    ids=arg_keys("numeric_only", bool_none_arg_keys),
)
def test_min(modin_df, pandas_df, axis, skipna, numeric_only):
    try:
        pandas_result = pandas_df.min(axis=axis, skipna=skipna, numeric_only=numeric_only)
    except Exception:
        with pytest.raises(TypeError):
            modin_result = modin_df.min(axis=axis, skipna=skipna, numeric_only=numeric_only)
        return
    modin_result = modin_df.min(axis=axis, skipna=skipna, numeric_only=numeric_only)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
)
def test_mode(request, modin_df, pandas_df, axis, numeric_only):
    try:
        pandas_result = pandas_df.mode(axis=axis, numeric_only=numeric_only)
    except Exception:
        with pytest.raises(TypeError):
            modin_df.mode(axis=axis, numeric_only=numeric_only)
        return
    modin_result = modin_df.mode(axis=axis, numeric_only=numeric_only)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_ndim(modin_df, pandas_df):
    assert modin_df.ndim == pandas_df.ndim


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_nlargest(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.nlargest(None, None)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_notna(modin_df, pandas_df):
    df_equals(modin_df.notna(), pandas_df.notna())


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_notnull(modin_df, pandas_df):
    df_equals(modin_df.notnull(), pandas_df.notnull())


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_nsmallest(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.nsmallest(None, None)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "dropna", bool_arg_values, ids=arg_keys("dropna", bool_arg_keys)
)
def test_nunique(modin_df, pandas_df, axis, dropna):
    modin_result = modin_df.nunique(axis=axis, dropna=dropna)
    pandas_result = pandas_df.nunique(axis=axis, dropna=dropna)
    df_equals(modin_result, pandas_result)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_pct_change(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.pct_change()


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_pipe(modin_df, pandas_df):
    n = len(modin_df.index)
    a, b, c = 2 % n, 0, 3 % n
    col = modin_df.columns[3 % len(modin_df.columns)]

    def h(x):
        return x.drop(columns=[col])

    def g(x, arg1=0):
        for _ in range(arg1):
            x = x.append(x)
        return x

    def f(x, arg2=0, arg3=0):
        return x.drop([arg2, arg3])

    df_equals(
        f(g(h(modin_df), arg1=a), arg2=b, arg3=c),
        (modin_df.pipe(h).pipe(g, arg1=a).pipe(f, arg2=b, arg3=c)),
    )

    df_equals(
        (modin_df.pipe(h).pipe(g, arg1=a).pipe(f, arg2=b, arg3=c)),
        (pandas_df.pipe(h).pipe(g, arg1=a).pipe(f, arg2=b, arg3=c)),
    )


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_pivot(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.pivot()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_pivot_table(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.pivot_table()


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_plot(request, modin_df, pandas_df):
    if name_contains(request.node.name, numeric_dfs):
        # We have to test this way because equality in plots means same object.
        zipped_plot_lines = zip(modin_df.plot().lines, pandas_df.plot().lines)
        for l, r in zipped_plot_lines:
            if isinstance(l.get_xdata(), np.ma.core.MaskedArray) and isinstance(r.get_xdata(), np.ma.core.MaskedArray):
                assert all((l.get_xdata() == r.get_xdata()).data)
            else:
                assert np.array_equal(l.get_xdata(), r.get_xdata())
            if isinstance(l.get_ydata(), np.ma.core.MaskedArray) and isinstance(r.get_ydata(), np.ma.core.MaskedArray):
                assert all((l.get_ydata() == r.get_ydata()).data)
            else:
                assert np.array_equal(l.get_xdata(), r.get_xdata())


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_pop(request, modin_df, pandas_df):
    if "empty_data" not in request.node.name:
        key = modin_df.columns[0]
        temp_modin_df = modin_df.copy()
        temp_pandas_df = pandas_df.copy()
        modin_popped = temp_modin_df.pop(key)
        pandas_popped = temp_pandas_df.pop(key)
        df_equals(modin_popped, pandas_popped)
        df_equals(temp_modin_df, temp_pandas_df)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
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
def test_prod(request, modin_df, pandas_df, axis, skipna, numeric_only, min_count):
    try:
        pandas_result = pandas_df.prod(
            axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
        )
    except Exception:
        with pytest.raises(TypeError):
            modin_df.prod(
                axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
            )
        return
    modin_result = modin_df.prod(
        axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
    )
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
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
def test_product(request, modin_df, pandas_df, axis, skipna, numeric_only, min_count):
    try:
        pandas_result = pandas_df.product(
            axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
        )
    except Exception:
        with pytest.raises(TypeError):
            modin_df.product(
                axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
            )
        return
    modin_result = modin_df.product(
        axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
    )
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("q", quantiles_values, ids=quantiles_keys)
def test_quantile(request, modin_df, pandas_df, q):
    if not name_contains(request.node.name, no_numeric_dfs):
        df_equals(modin_df.quantile(q), pandas_df.quantile(q))
    else:
        with pytest.raises(ValueError):
            modin_df.quantile(q)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("funcs", query_func_values, ids=query_func_keys)
def test_query(request, modin_df, pandas_df, funcs):
    try:
        pandas_result = pandas_df.query(funcs)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.query(funcs)
        return
    modin_result = modin_df.query(funcs)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
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
    modin_df, pandas_df, axis, method, numeric_only, na_option, ascending, pct
):
    modin_result = modin_df.rank(
        axis=axis,
        method=method,
        numeric_only=numeric_only,
        na_option=na_option,
        ascending=ascending,
        pct=pct,
    )
    pandas_result = pandas_df.rank(
        axis=axis,
        method=method,
        numeric_only=numeric_only,
        na_option=na_option,
        ascending=ascending,
        pct=pct,
    )
    df_equals(modin_result, pandas_result)


def test_reindex():
    frame_data = {
        "col1": [0, 1, 2, 3],
        "col2": [4, 5, 6, 7],
        "col3": [8, 9, 10, 11],
        "col4": [12, 13, 14, 15],
        "col5": [0, 0, 0, 0],
    }
    pandas_df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)

    df_equals(modin_df.reindex([0, 3, 2, 1]), pandas_df.reindex([0, 3, 2, 1]))

    df_equals(modin_df.reindex([0, 6, 2]), pandas_df.reindex([0, 6, 2]))

    df_equals(
        modin_df.reindex(["col1", "col3", "col4", "col2"], axis=1),
        pandas_df.reindex(["col1", "col3", "col4", "col2"], axis=1),
    )

    df_equals(
        modin_df.reindex(["col1", "col7", "col4", "col8"], axis=1),
        pandas_df.reindex(["col1", "col7", "col4", "col8"], axis=1),
    )

    df_equals(
        modin_df.reindex(index=[0, 1, 5], columns=["col1", "col7", "col4", "col8"]),
        pandas_df.reindex(index=[0, 1, 5], columns=["col1", "col7", "col4", "col8"]),
    )


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_reindex_axis(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.reindex_axis(None)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_reindex_like(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.reindex_like(None)


def test_rename_sanity():
    test_data = TestData()
    mapping = {"A": "a", "B": "b", "C": "c", "D": "d"}

    modin_df = pd.DataFrame(test_data.frame)
    df_equals(
        modin_df.rename(columns=mapping), test_data.frame.rename(columns=mapping)
    )

    renamed2 = test_data.frame.rename(columns=str.lower)
    df_equals(modin_df.rename(columns=str.lower), renamed2)

    modin_df = pd.DataFrame(renamed2)
    df_equals(
        modin_df.rename(columns=str.upper), renamed2.rename(columns=str.upper)
    )

    # index
    data = {"A": {"foo": 0, "bar": 1}}

    # gets sorted alphabetical
    df = pandas.DataFrame(data)
    modin_df = pd.DataFrame(data)
    tm.assert_index_equal(
        modin_df.rename(index={"foo": "bar", "bar": "foo"}).index,
        df.rename(index={"foo": "bar", "bar": "foo"}).index,
    )

    tm.assert_index_equal(
        modin_df.rename(index=str.upper).index, df.rename(index=str.upper).index
    )

    # have to pass something
    pytest.raises(TypeError, modin_df.rename)

    # partial columns
    renamed = test_data.frame.rename(columns={"C": "foo", "D": "bar"})
    modin_df = pd.DataFrame(test_data.frame)
    tm.assert_index_equal(
        modin_df.rename(columns={"C": "foo", "D": "bar"}).index,
        test_data.frame.rename(columns={"C": "foo", "D": "bar"}).index,
    )

    # TODO: Uncomment when transpose works
    # other axis
    # renamed = test_data.frame.T.rename(index={'C': 'foo', 'D': 'bar'})
    # tm.assert_index_equal(
    #     test_data.frame.T.rename(index={'C': 'foo', 'D': 'bar'}).index,
    #     modin_df.T.rename(index={'C': 'foo', 'D': 'bar'}).index)

    # index with name
    index = pandas.Index(["foo", "bar"], name="name")
    renamer = pandas.DataFrame(data, index=index)
    modin_df = pd.DataFrame(data, index=index)

    renamed = renamer.rename(index={"foo": "bar", "bar": "foo"})
    modin_renamed = modin_df.rename(index={"foo": "bar", "bar": "foo"})
    tm.assert_index_equal(renamed.index, modin_renamed.index)

    assert renamed.index.name == modin_renamed.index.name


def test_rename_multiindex():
    tuples_index = [("foo1", "bar1"), ("foo2", "bar2")]
    tuples_columns = [("fizz1", "buzz1"), ("fizz2", "buzz2")]
    index = pandas.MultiIndex.from_tuples(tuples_index, names=["foo", "bar"])
    columns = pandas.MultiIndex.from_tuples(tuples_columns, names=["fizz", "buzz"])

    frame_data = [(0, 0), (1, 1)]
    df = pandas.DataFrame(frame_data, index=index, columns=columns)
    modin_df = pd.DataFrame(frame_data, index=index, columns=columns)

    #
    # without specifying level -> accross all levels
    renamed = df.rename(
        index={"foo1": "foo3", "bar2": "bar3"},
        columns={"fizz1": "fizz3", "buzz2": "buzz3"},
    )
    modin_renamed = modin_df.rename(
        index={"foo1": "foo3", "bar2": "bar3"},
        columns={"fizz1": "fizz3", "buzz2": "buzz3"},
    )
    tm.assert_index_equal(renamed.index, modin_renamed.index)

    renamed = df.rename(
        index={"foo1": "foo3", "bar2": "bar3"},
        columns={"fizz1": "fizz3", "buzz2": "buzz3"},
    )
    tm.assert_index_equal(renamed.columns, modin_renamed.columns)
    assert renamed.index.names == modin_renamed.index.names
    assert renamed.columns.names == modin_renamed.columns.names

    #
    # with specifying a level

    # dict
    renamed = df.rename(columns={"fizz1": "fizz3", "buzz2": "buzz3"}, level=0)
    modin_renamed = modin_df.rename(
        columns={"fizz1": "fizz3", "buzz2": "buzz3"}, level=0
    )
    tm.assert_index_equal(renamed.columns, modin_renamed.columns)
    renamed = df.rename(columns={"fizz1": "fizz3", "buzz2": "buzz3"}, level="fizz")
    modin_renamed = modin_df.rename(
        columns={"fizz1": "fizz3", "buzz2": "buzz3"}, level="fizz"
    )
    tm.assert_index_equal(renamed.columns, modin_renamed.columns)

    renamed = df.rename(columns={"fizz1": "fizz3", "buzz2": "buzz3"}, level=1)
    modin_renamed = modin_df.rename(
        columns={"fizz1": "fizz3", "buzz2": "buzz3"}, level=1
    )
    tm.assert_index_equal(renamed.columns, modin_renamed.columns)
    renamed = df.rename(columns={"fizz1": "fizz3", "buzz2": "buzz3"}, level="buzz")
    modin_renamed = modin_df.rename(
        columns={"fizz1": "fizz3", "buzz2": "buzz3"}, level="buzz"
    )
    tm.assert_index_equal(renamed.columns, modin_renamed.columns)

    # function
    func = str.upper
    renamed = df.rename(columns=func, level=0)
    modin_renamed = modin_df.rename(columns=func, level=0)
    tm.assert_index_equal(renamed.columns, modin_renamed.columns)
    renamed = df.rename(columns=func, level="fizz")
    modin_renamed = modin_df.rename(columns=func, level="fizz")
    tm.assert_index_equal(renamed.columns, modin_renamed.columns)

    renamed = df.rename(columns=func, level=1)
    modin_renamed = modin_df.rename(columns=func, level=1)
    tm.assert_index_equal(renamed.columns, modin_renamed.columns)
    renamed = df.rename(columns=func, level="buzz")
    modin_renamed = modin_df.rename(columns=func, level="buzz")
    tm.assert_index_equal(renamed.columns, modin_renamed.columns)

    # index
    renamed = df.rename(index={"foo1": "foo3", "bar2": "bar3"}, level=0)
    modin_renamed = modin_df.rename(index={"foo1": "foo3", "bar2": "bar3"}, level=0)
    tm.assert_index_equal(modin_renamed.index, renamed.index)


def test_rename_nocopy():
    test_data = TestData().frame
    modin_df = pd.DataFrame(test_data)
    modin_renamed = modin_df.rename(columns={"C": "foo"}, copy=False)
    modin_renamed["foo"] = 1
    assert (modin_df["C"] == 1).all()


def test_rename_inplace():
    test_data = TestData().frame
    modin_df = pd.DataFrame(test_data)

    df_equals(
        modin_df.rename(columns={"C": "foo"}), test_data.rename(columns={"C": "foo"})
    )

    frame = test_data.copy()
    modin_frame = modin_df.copy()
    frame.rename(columns={"C": "foo"}, inplace=True)
    modin_frame.rename(columns={"C": "foo"}, inplace=True)

    df_equals(modin_frame, frame)


def test_rename_bug():
    # rename set ref_locs, and set_index was not resetting
    frame_data = {0: ["foo", "bar"], 1: ["bah", "bas"], 2: [1, 2]}
    df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)
    df = df.rename(columns={0: "a"})
    df = df.rename(columns={1: "b"})
    # TODO: Uncomment when set_index is implemented
    # df = df.set_index(['a', 'b'])
    # df.columns = ['2001-01-01']

    modin_df = modin_df.rename(columns={0: "a"})
    modin_df = modin_df.rename(columns={1: "b"})
    # TODO: Uncomment when set_index is implemented
    # modin_df = modin_df.set_index(['a', 'b'])
    # modin_df.columns = ['2001-01-01']

    df_equals(modin_df, df)


def test_rename_axis_inplace():
    test_frame = TestData().frame
    modin_df = pd.DataFrame(test_frame)

    result = test_frame.copy()
    modin_result = modin_df.copy()
    no_return = result.rename_axis("foo", inplace=True)
    modin_no_return = modin_result.rename_axis("foo", inplace=True)

    assert no_return is modin_no_return
    df_equals(modin_result, result)

    result = test_frame.copy()
    modin_result = modin_df.copy()
    no_return = result.rename_axis("bar", axis=1, inplace=True)
    modin_no_return = modin_result.rename_axis("bar", axis=1, inplace=True)

    assert no_return is modin_no_return
    df_equals(modin_result, result)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_reorder_levels(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.reorder_levels(None)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_replace(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.replace()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_resample(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.resample(None)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_reset_index(modin_df, pandas_df):
    modin_result = modin_df.reset_index(inplace=False)
    pandas_result = pandas_df.reset_index(inplace=False)
    df_equals(modin_result, pandas_result)

    modin_df_cp = modin_df.copy()
    pd_df_cp = pandas_df.copy()
    modin_df_cp.reset_index(inplace=True)
    pd_df_cp.reset_index(inplace=True)
    df_equals(modin_df_cp, pd_df_cp)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_rolling(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.rolling(None)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_round(modin_df, pandas_df):
    df_equals(modin_df.round(), pandas_df.round())
    df_equals(modin_df.round(1), pandas_df.round(1))


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
def test_sample(modin_df, pandas_df, axis):
    with pytest.raises(ValueError):
        modin_df.sample(n=3, frac=0.4, axis=axis)

    modin_result = modin_df.sample(frac=0.5, random_state=42, axis=axis)
    pandas_result = pandas_df.sample(frac=0.5, random_state=42, axis=axis)
    df_equals(modin_result, pandas_result)

    modin_result = modin_df.sample(n=2, random_state=42, axis=axis)
    pandas_result = pandas_df.sample(n=2, random_state=42, axis=axis)
    df_equals(modin_result, pandas_result)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_select(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.select(None)


def test_select_dtypes():
    frame_data = {
        "test1": list("abc"),
        "test2": np.arange(3, 6).astype("u1"),
        "test3": np.arange(8.0, 11.0, dtype="float64"),
        "test4": [True, False, True],
        "test5": pandas.date_range("now", periods=3).values,
        "test6": list(range(5, 8)),
    }
    df = pandas.DataFrame(frame_data)
    rd = pd.DataFrame(frame_data)

    include = np.float, "integer"
    exclude = (np.bool_,)
    r = rd.select_dtypes(include=include, exclude=exclude)

    e = df[["test2", "test3", "test6"]]
    df_equals(r, e)

    try:
        pd.DataFrame().select_dtypes()
        assert False
    except ValueError:
        assert True


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_sem(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.sem()


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
def test_set_axis(modin_df, pandas_df, axis):
    x = pandas.DataFrame()._get_axis_number(axis)
    index = modin_df.columns if x else modin_df.index
    labels = ["{0}_{1}".format(index[i], i) for i in range(modin_df.shape[x])]

    modin_result = modin_df.set_axis(labels, axis=axis, inplace=False)
    pandas_result = pandas_df.set_axis(labels, axis=axis, inplace=False)
    df_equals(modin_result, pandas_result)

    modin_df_copy = modin_df.copy()
    pandas_df_copy = pandas_df.copy()
    modin_df_copy.set_axis(labels, axis=axis, inplace=True)
    # Test difference
    try:
        df_equals(modin_df, modin_df_copy)
    except AssertionError:
        pass
    else:
        raise AssertionError
    pandas_df_copy.set_axis(labels, axis=axis, inplace=True)
    df_equals(modin_df_copy, pandas_df_copy)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("drop", bool_arg_values, ids=arg_keys("drop", bool_arg_keys))
@pytest.mark.parametrize(
    "append", bool_arg_values, ids=arg_keys("append", bool_arg_keys)
)
def test_set_index(request, modin_df, pandas_df, drop, append):
    if "empty_data" not in request.node.name:
        key = modin_df.columns[0]
        modin_result = modin_df.set_index(key, drop=drop, append=append, inplace=False)
        pandas_result = pandas_df.set_index(
            key, drop=drop, append=append, inplace=False
        )
        df_equals(modin_result, pandas_result)

        modin_df_copy = modin_df.copy()
        pandas_df_copy = pandas_df.copy()
        modin_df_copy.set_index(key, drop=drop, append=append, inplace=True)
        # Test difference
        try:
            df_equals(modin_df, modin_df_copy)
        except AssertionError:
            pass
        else:
            raise AssertionError
        pandas_df_copy.set_index(key, drop=drop, append=append, inplace=True)
        df_equals(modin_df_copy, pandas_df_copy)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_set_value(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.set_value(None, None, None)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_shape(modin_df, pandas_df):
    assert modin_df.shape == pandas_df.shape


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_shift(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.shift()


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_size(modin_df, pandas_df):
    assert modin_df.size == pandas_df.size


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "numeric_only",
    bool_none_arg_values,
    ids=arg_keys("numeric_only", bool_none_arg_keys),
)
def test_skew(request, modin_df, pandas_df, axis, skipna, numeric_only):
    try:
        pandas_result = pandas_df.skew(axis=axis, skipna=skipna, numeric_only=numeric_only)
    except Exception:
        with pytest.raises(TypeError):
            modin_df.skew(axis=axis, skipna=skipna, numeric_only=numeric_only)
        return
    modin_result = modin_df.skew(axis=axis, skipna=skipna, numeric_only=numeric_only)
    df_equals(modin_result, pandas_result)

@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_slice_shift(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.slice_shift()


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "ascending", bool_arg_values, ids=arg_keys("ascending", bool_arg_keys)
)
@pytest.mark.parametrize("na_position", ["first", "last"], ids=["first", "last"])
@pytest.mark.parametrize(
    "sort_remaining", bool_arg_values, ids=arg_keys("sort_remaining", bool_arg_keys)
)
def test_sort_index(modin_df, pandas_df, axis, ascending, na_position, sort_remaining):
    # Change index value so sorting will actually make a difference
    if axis == 'rows' or axis == 0:
        length = len(modin_df.index)
        modin_df.index = [(i-length/2)%length for i in range(length)]
        pandas_df.index = [(i-length/2)%length for i in range(length)]
    # Add NaNs to sorted index
    if axis == 'rows' or axis == 0:
        length = len(modin_df.index)
        modin_df.index = [np.nan if i % 2 == 0 else modin_df.index[i] for i in range(length)]
        pandas_df.index = [np.nan if i % 2 == 0 else pandas_df.index[i] for i in range(length)]
    else:
        length = len(modin_df.columns)
        modin_df.columns = [np.nan if i % 2 == 0 else modin_df.columns[i] for i in range(length)]
        pandas_df.columns = [np.nan if i % 2 == 0 else pandas_df.columns[i] for i in range(length)]

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


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "ascending", bool_arg_values, ids=arg_keys("ascending", bool_arg_keys)
)
@pytest.mark.parametrize("na_position", ["first", "last"], ids=["first", "last"])
def test_sort_values(request, modin_df, pandas_df, axis, ascending, na_position):
    if "empty_data" not in request.node.name and (
        (axis == 0 or axis == "over rows")
        or name_contains(request.node.name, numeric_dfs)
    ):
        index = modin_df.index if axis == 1 or axis == "columns" else modin_df.columns
        key = index[0]
        modin_result = modin_df.sort_values(
            key, axis=axis, ascending=ascending, na_position=na_position, inplace=False
        )
        pandas_result = pandas_df.sort_values(
            key, axis=axis, ascending=ascending, na_position=na_position, inplace=False
        )
        df_equals(modin_result, pandas_result)

        modin_df_cp = modin_df.copy()
        pandas_df_cp = pandas_df.copy()
        modin_df_cp.sort_values(
            key, axis=axis, ascending=ascending, na_position=na_position, inplace=True
        )
        pandas_df_cp.sort_values(
            key, axis=axis, ascending=ascending, na_position=na_position, inplace=True
        )
        df_equals(modin_df_cp, pandas_df_cp)

        keys = [key, index[-1]]
        modin_result = modin_df.sort_values(
            keys, axis=axis, ascending=ascending, na_position=na_position, inplace=False
        )
        pandas_result = pandas_df.sort_values(
            keys, axis=axis, ascending=ascending, na_position=na_position, inplace=False
        )
        df_equals(modin_result, pandas_result)

        modin_df_cp = modin_df.copy()
        pandas_df_cp = pandas_df.copy()
        modin_df_cp.sort_values(
            keys, axis=axis, ascending=ascending, na_position=na_position, inplace=True
        )
        pandas_df_cp.sort_values(
            keys, axis=axis, ascending=ascending, na_position=na_position, inplace=True
        )
        df_equals(modin_df_cp, pandas_df_cp)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_sortlevel(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.sortlevel()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_squeeze(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.squeeze()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_stack(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.stack()


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
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
def test_std(request, modin_df, pandas_df, axis, skipna, numeric_only, ddof):
    try:
        pandas_result = pandas_df.std(
            axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof
        )
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.std(axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof)
        return
    modin_result = modin_df.std(
        axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof
    )
    df_equals(modin_result, pandas_result)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_style(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.style


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
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
def test_sum(request, modin_df, pandas_df, axis, skipna, numeric_only, min_count):
    try:
        pandas_result = pandas_df.sum(
            axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
        )
    except Exception:
        with pytest.raises(TypeError):
            modin_df.sum(
                axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
            )
        return
    modin_result = modin_df.sum(
        axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
    )
    df_equals(modin_result, pandas_result)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_swapaxes(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.swapaxes(None, None)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_swaplevel(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.swaplevel()


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("n", int_arg_values, ids=arg_keys("n", int_arg_keys))
def test_tail(modin_df, pandas_df, n):
    df_equals(modin_df.tail(n), pandas_df.tail(n))


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_take(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.take(None)


def test_to_datetime():
    frame_data = {"year": [2015, 2016], "month": [2, 3], "day": [4, 5]}
    modin_df = pd.DataFrame(frame_data)
    pd_df = pandas.DataFrame(frame_data)

    df_equals(pd.to_datetime(modin_df), pandas.to_datetime(pd_df))


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_to_records(request, modin_df, pandas_df):
    # Skips nan because only difference is nan instead of NaN
    if not name_contains(request.node.name, ['nan']):
        assert np.array_equal(modin_df.to_records(), pandas_df.to_records())


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_to_sparse(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.to_sparse()


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_to_string(request, modin_df, pandas_df):
    # Skips nan because only difference is nan instead of NaN
    if not name_contains(request.node.name, ['nan']):
        assert modin_df.to_string() == to_pandas(modin_df).to_string()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_to_timestamp(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.to_timestamp()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_to_xarray(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.to_xarray()


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
@pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
def test_transform(request, modin_df, pandas_df, func):
    try:
        pandas_result = pandas_df.agg(func)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.agg(func)
        return
    modin_result = modin_df.agg(func)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_transpose(modin_df, pandas_df):
    df_equals(modin_df.T, pandas_df.T)
    df_equals(modin_df.transpose(), pandas_df.transpose())
    # Test for map across full axis for select indices
    df_equals(modin_df.T.reset_index(), pandas_df.T.reset_index())
    # Test for map across full axis
    df_equals(modin_df.T.nunique(), pandas_df.T.nunique())
    # Test for map across blocks
    df_equals(modin_df.T.notna(), pandas_df.T.notna())


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_truncate(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.truncate()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_tshift(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.tshift()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_tz_convert(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.tz_convert(None)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_tz_localize(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.tz_localize(None)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_unstack(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.unstack()


def test_update():
    df = pd.DataFrame(
        [[1.5, np.nan, 3.0], [1.5, np.nan, 3.0], [1.5, np.nan, 3], [1.5, np.nan, 3]]
    )
    other = pd.DataFrame([[3.6, 2.0, np.nan], [np.nan, np.nan, 7]], index=[1, 3])

    df.update(other)
    expected = pd.DataFrame(
        [[1.5, np.nan, 3], [3.6, 2, 3], [1.5, np.nan, 3], [1.5, np.nan, 7.0]]
    )
    df_equals(df, expected)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_values(modin_df, pandas_df):
    np.testing.assert_equal(modin_df.values, pandas_df.values)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
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
def test_var(request, modin_df, pandas_df, axis, skipna, numeric_only, ddof):
    try:
        pandas_result = pandas_df.var(
            axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof
        )
    except Exception:
        with pytest.raises(TypeError):
            modin_df.var(axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof)
        return
    modin_result = modin_df.var(
        axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof
    )
    df_equals(modin_result, pandas_result)


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


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test_xs(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.xs(None)


def test__doc__():
    assert pd.DataFrame.__doc__ != pandas.DataFrame.__doc__
    assert pd.DataFrame.__init__ != pandas.DataFrame.__init__
    for attr, obj in pd.DataFrame.__dict__.items():
        if (callable(obj) or isinstance(obj, property)) and attr != "__init__":
            pd_obj = getattr(pandas.DataFrame, attr, None)
            if callable(pd_obj) or isinstance(pd_obj, property):
                assert obj.__doc__ == pd_obj.__doc__


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___getitem__(request, modin_df, pandas_df):
    if "empty_data" not in request.node.name:
        key = modin_df.columns[0]
        modin_col = modin_df.__getitem__(key)
        assert isinstance(modin_col, pandas.Series)

        pd_col = pandas_df[key]
        df_equals(pd_col, modin_col)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___getattr__(request, modin_df, pandas_df):
    if "empty_data" not in request.node.name:
        key = modin_df.columns[0]
        col = modin_df.__getattr__(key)
        assert isinstance(col, pandas.Series)

        col = getattr(modin_df, key)
        assert isinstance(col, pandas.Series)

        col = modin_df.col1
        assert isinstance(col, pandas.Series)

        # Check that lookup in column doesn't override other attributes
        df2 = modin_df.rename(index=str, columns={key: "columns"})
        assert isinstance(df2.columns, pandas.Index)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___setitem__(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.__setitem__(None, None)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___len__(modin_df, pandas_df):
    assert len(modin_df) == len(pandas_df)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___unicode__(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.__unicode__()


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___neg__(request, modin_df, pandas_df):
    try:
        pandas_result = pandas_df.__neg__()
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.__neg__()
        return
    modin_result = modin_df.__neg__()
    df_equals(modin_result, pandas_result)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___invert__(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.__invert__()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___hash__(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.__hash__()


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___iter__(modin_df, pandas_df):
    modin_iterator = modin_df.__iter__()

    # Check that modin_iterator implements the iterator interface
    assert hasattr(modin_iterator, "__iter__")
    assert hasattr(modin_iterator, "next") or hasattr(modin_iterator, "__next__")

    pd_iterator = pandas_df.__iter__()
    assert list(modin_iterator) == list(pd_iterator)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___contains__(request, modin_df, pandas_df):
    result = False
    key = "Not Exist"
    assert result == modin_df.__contains__(key)
    assert result == (key in modin_df)

    if "empty_data" not in request.node.name:
        result = True
        key = pandas_df.columns[0]
        assert result == modin_df.__contains__(key)
        assert result == (key in modin_df)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___nonzero__(modin_df, pandas_df):
    with pytest.raises(ValueError):
        # Always raises ValueError
        modin_df.__nonzero__()


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___abs__(request, modin_df, pandas_df):
    try:
        pandas_result = abs(pandas_df)
    except Exception as e:
        with pytest.raises(type(e)):
            abs(modin_df)
        return
    modin_result = abs(modin_df)
    df_equals(modin_df, pandas_df)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___round__(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.__round__()


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___array__(modin_df, pandas_df):
    assert_array_equal(modin_df.__array__(), pandas_df.__array__())


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___bool__(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.__bool__()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___getstate__(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.__getstate__()


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___setstate__(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.__setstate__(None)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___delitem__(request, modin_df, pandas_df):
    if "empty_data" not in request.node.name:
        key = pandas_df.columns[0]

        modin_df = modin_df.copy()
        pandas_df = pandas_df.copy()
        modin_df.__delitem__(key)
        pandas_df.__delitem__(key)
        df_equals(modin_df, pandas_df)

        # Issue 2027
        last_label = pandas_df.iloc[:, -1].name
        modin_df.__delitem__(last_label)
        pandas_df.__delitem__(last_label)
        df_equals(modin_df, pandas_df)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___finalize__(modin_df, pandas_df):
    with pytest.raises(NotImplementedError):
        modin_df.__finalize__(None)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___copy__(modin_df, pandas_df):
    modin_df_copy, pandas_df_copy = modin_df.__copy__(), pandas_df.__copy__()
    df_equals(modin_df_copy, pandas_df_copy)


@pytest.mark.parametrize("modin_df, pandas_df", test_dfs_values, ids=test_dfs_keys)
def test___deepcopy__(modin_df, pandas_df):
    modin_df_copy, pandas_df_copy = modin_df.__deepcopy__(), pandas_df.__deepcopy__()
    df_equals(modin_df_copy, pandas_df_copy)


def test___repr__():
    frame_data = random_state.randint(RAND_LOW, RAND_HIGH, size=(1000, 100))
    pandas_df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)
    assert repr(pandas_df) == repr(modin_df)

    frame_data = random_state.randint(RAND_LOW, RAND_HIGH, size=(1000, 99))
    pandas_df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)
    assert repr(pandas_df) == repr(modin_df)

    frame_data = random_state.randint(RAND_LOW, RAND_HIGH, size=(1000, 101))
    pandas_df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)
    assert repr(pandas_df) == repr(modin_df)

    frame_data = random_state.randint(RAND_LOW, RAND_HIGH, size=(1000, 102))
    pandas_df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)
    assert repr(pandas_df) == repr(modin_df)

    # ___repr___ method has a different code path depending on
    # whether the number of rows is >60; and a different code path
    # depending on the number of columns is >20.
    # Previous test cases already check the case when cols>20
    # and rows>60. The cases that follow exercise the other three
    # combinations.
    # rows <= 60, cols > 20
    frame_data = random_state.randint(RAND_LOW, RAND_HIGH, size=(10, 100))
    pandas_df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)

    assert repr(pandas_df) == repr(modin_df)

    # rows <= 60, cols <= 20
    frame_data = random_state.randint(RAND_LOW, RAND_HIGH, size=(10, 10))
    pandas_df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)

    assert repr(pandas_df) == repr(modin_df)

    # rows > 60, cols <= 20
    frame_data = random_state.randint(RAND_LOW, RAND_HIGH, size=(100, 10))
    pandas_df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)

    assert repr(pandas_df) == repr(modin_df)
