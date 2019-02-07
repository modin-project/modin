from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import numpy as np
import pandas
import pandas.util.testing as tm
from pandas.tests.frame.common import TestData
import matplotlib
import modin.pandas as pd
from modin.pandas.utils import to_pandas
from modin.pandas.series import SeriesView
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
    test_data_values,
    test_data_keys,
    numeric_dfs,
    no_numeric_dfs,
    test_func_keys,
    test_func_values,
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
    int_arg_keys,
    int_arg_values,
)

pd.DEFAULT_NPARTITIONS = 4

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")

if sys.version_info[0] < 3:
    PY2 = True
else:
    PY2 = False


# Test inter df math functions
def inter_df_math_helper(modin_df, pandas_df, op):
    try:
        pandas_result = getattr(pandas_df, op)(pandas_df)
    except Exception as e:
        with pytest.raises(type(e)):
            getattr(modin_df, op)(modin_df)
    else:
        modin_result = getattr(modin_df, op)(modin_df)
        df_equals(modin_result, pandas_result)

    try:
        pandas_result = getattr(pandas_df, op)(4)
    except Exception as e:
        with pytest.raises(type(e)):
            getattr(modin_df, op)(4)
    else:
        modin_result = getattr(modin_df, op)(4)
        df_equals(modin_result, pandas_result)

    try:
        pandas_result = getattr(pandas_df, op)(4.0)
    except Exception as e:
        with pytest.raises(type(e)):
            getattr(modin_df, op)(4.0)
    else:
        modin_result = getattr(modin_df, op)(4.0)
        df_equals(modin_result, pandas_result)

    frame_data = {
        "{}_other".format(modin_df.columns[0]): [0, 2],
        modin_df.columns[0]: [0, 19],
        modin_df.columns[1]: [1, 1],
    }
    modin_df2 = pd.DataFrame(frame_data)
    pandas_df2 = pandas.DataFrame(frame_data)

    try:
        pandas_result = getattr(pandas_df, op)(pandas_df2)
    except Exception as e:
        with pytest.raises(type(e)):
            getattr(modin_df, op)(modin_df2)
    else:
        modin_result = getattr(modin_df, op)(modin_df2)
        df_equals(modin_result, pandas_result)

    list_test = random_state.randint(RAND_LOW, RAND_HIGH, size=(modin_df.shape[1]))
    try:
        pandas_result = getattr(pandas_df, op)(list_test, axis=1)
    except Exception as e:
        with pytest.raises(type(e)):
            getattr(modin_df, op)(list_test, axis=1)
    else:
        modin_result = getattr(modin_df, op)(list_test, axis=1)
        df_equals(modin_result, pandas_result)

    list_test = random_state.randint(RAND_LOW, RAND_HIGH, size=(modin_df.shape[0]))
    try:
        pandas_result = getattr(pandas_df, op)(list_test, axis=0)
    except Exception as e:
        with pytest.raises(type(e)):
            getattr(modin_df, op)(list_test, axis=0)
    else:
        modin_result = getattr(modin_df, op)(list_test, axis=0)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_add(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    inter_df_math_helper(modin_df, pandas_df, "add")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_div(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    inter_df_math_helper(modin_df, pandas_df, "div")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_divide(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    inter_df_math_helper(modin_df, pandas_df, "divide")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_floordiv(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    inter_df_math_helper(modin_df, pandas_df, "floordiv")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_mod(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    inter_df_math_helper(modin_df, pandas_df, "mod")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_mul(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    inter_df_math_helper(modin_df, pandas_df, "mul")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_multiply(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    inter_df_math_helper(modin_df, pandas_df, "multiply")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_pow(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    # TODO: Revert to others once we have an efficient way of preprocessing for positive
    # values
    try:
        pandas_df = pandas_df.abs()
    except Exception:
        pass
    else:
        modin_df = modin_df.abs()
        inter_df_math_helper(modin_df, pandas_df, "pow")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_sub(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    inter_df_math_helper(modin_df, pandas_df, "sub")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_subtract(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    inter_df_math_helper(modin_df, pandas_df, "subtract")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_truediv(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    inter_df_math_helper(modin_df, pandas_df, "truediv")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___div__(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    inter_df_math_helper(modin_df, pandas_df, "__div__")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___add__(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    inter_df_math_helper(modin_df, pandas_df, "__add__")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___radd__(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    inter_df_math_helper(modin_df, pandas_df, "__radd__")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___mul__(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    inter_df_math_helper(modin_df, pandas_df, "__mul__")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___rmul__(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    inter_df_math_helper(modin_df, pandas_df, "__rmul__")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___pow__(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    inter_df_math_helper(modin_df, pandas_df, "__pow__")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___rpow__(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    inter_df_math_helper(modin_df, pandas_df, "__rpow__")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___sub__(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    inter_df_math_helper(modin_df, pandas_df, "__sub__")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___floordiv__(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    inter_df_math_helper(modin_df, pandas_df, "__floordiv__")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___rfloordiv__(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    inter_df_math_helper(modin_df, pandas_df, "__rfloordiv__")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___truediv__(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    inter_df_math_helper(modin_df, pandas_df, "__truediv__")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___rtruediv__(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    inter_df_math_helper(modin_df, pandas_df, "__rtruediv__")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___mod__(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    inter_df_math_helper(modin_df, pandas_df, "__mod__")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___rmod__(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    inter_df_math_helper(modin_df, pandas_df, "__rmod__")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___rdiv__(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    inter_df_math_helper(modin_df, pandas_df, "__rdiv__")


# END test inter df math functions


# Test comparison of inter operation functions
def comparison_inter_ops_helper(modin_df, pandas_df, op):
    try:
        pandas_result = getattr(pandas_df, op)(pandas_df)
    except Exception as e:
        with pytest.raises(type(e)):
            getattr(modin_df, op)(modin_df)
    else:
        modin_result = getattr(modin_df, op)(modin_df)
        df_equals(modin_result, pandas_result)

    try:
        pandas_result = getattr(pandas_df, op)(4)
    except TypeError:
        with pytest.raises(TypeError):
            getattr(modin_df, op)(4)
    else:
        modin_result = getattr(modin_df, op)(4)
        df_equals(modin_result, pandas_result)

    try:
        pandas_result = getattr(pandas_df, op)(4.0)
    except TypeError:
        with pytest.raises(TypeError):
            getattr(modin_df, op)(4.0)
    else:
        modin_result = getattr(modin_df, op)(4.0)
        df_equals(modin_result, pandas_result)

    frame_data = {
        "{}_other".format(modin_df.columns[0]): [0, 2],
        modin_df.columns[0]: [0, 19],
        modin_df.columns[1]: [1, 1],
    }
    modin_df2 = pd.DataFrame(frame_data)
    pandas_df2 = pandas.DataFrame(frame_data)

    try:
        pandas_result = getattr(pandas_df, op)(pandas_df2)
    except Exception as e:
        with pytest.raises(type(e)):
            getattr(modin_df, op)(modin_df2)
    else:
        modin_result = getattr(modin_df, op)(modin_df2)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_eq(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    comparison_inter_ops_helper(modin_df, pandas_df, "eq")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_ge(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    comparison_inter_ops_helper(modin_df, pandas_df, "ge")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_gt(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    comparison_inter_ops_helper(modin_df, pandas_df, "gt")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_le(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    comparison_inter_ops_helper(modin_df, pandas_df, "le")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_lt(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    comparison_inter_ops_helper(modin_df, pandas_df, "lt")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_ne(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    comparison_inter_ops_helper(modin_df, pandas_df, "ne")


# END test comparison of inter operation functions


# Test dataframe right operations
def inter_df_math_right_ops_helper(modin_df, pandas_df, op):
    try:
        pandas_result = getattr(pandas_df, op)(4)
    except Exception as e:
        with pytest.raises(type(e)):
            getattr(modin_df, op)(4)
    else:
        modin_result = getattr(modin_df, op)(4)
        df_equals(modin_result, pandas_result)

    try:
        pandas_result = getattr(pandas_df, op)(4.0)
    except Exception as e:
        with pytest.raises(type(e)):
            getattr(modin_df, op)(4.0)
    else:
        modin_result = getattr(modin_df, op)(4.0)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_radd(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    inter_df_math_right_ops_helper(modin_df, pandas_df, "radd")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_rdiv(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    inter_df_math_right_ops_helper(modin_df, pandas_df, "rdiv")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_rfloordiv(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    inter_df_math_right_ops_helper(modin_df, pandas_df, "rfloordiv")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_rmod(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    inter_df_math_right_ops_helper(modin_df, pandas_df, "rmod")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_rmul(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    inter_df_math_right_ops_helper(modin_df, pandas_df, "rmul")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_rpow(request, data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    # TODO: Revert to others once we have an efficient way of preprocessing for positive values
    # We need to check that negative integers are not used efficiently
    if "100x100" not in request.node.name:
        inter_df_math_right_ops_helper(modin_df, pandas_df, "rpow")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_rsub(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    inter_df_math_right_ops_helper(modin_df, pandas_df, "rsub")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_rtruediv(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    inter_df_math_right_ops_helper(modin_df, pandas_df, "rtrudiv")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___rsub__(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    inter_df_math_right_ops_helper(modin_df, pandas_df, "__rsub__")


# END test dataframe right operations


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_abs(request, data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.abs()
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.abs()
    else:
        modin_result = modin_df.abs()
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_add_prefix(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    test_prefix = "TEST"
    new_modin_df = modin_df.add_prefix(test_prefix)
    new_pandas_df = pandas_df.add_prefix(test_prefix)
    df_equals(new_modin_df.columns, new_pandas_df.columns)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("testfunc", test_func_values, ids=test_func_keys)
def test_applymap(request, data, testfunc):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    with pytest.raises(ValueError):
        x = 2
        modin_df.applymap(x)

    try:
        pandas_result = pandas_df.applymap(testfunc)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.applymap(testfunc)
    else:
        modin_result = modin_df.applymap(testfunc)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("testfunc", test_func_values, ids=test_func_keys)
def test_applymap_numeric(request, data, testfunc):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    if name_contains(request.node.name, numeric_dfs):
        try:
            pandas_result = pandas_df.applymap(testfunc)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.applymap(testfunc)
        else:
            modin_result = modin_df.applymap(testfunc)
            df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_add_suffix(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    test_suffix = "TEST"
    new_modin_df = modin_df.add_suffix(test_suffix)
    new_pandas_df = pandas_df.add_suffix(test_suffix)

    df_equals(new_modin_df.columns, new_pandas_df.columns)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_at(request, data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    # We skip nan datasets because nan != nan
    if "nan" not in request.node.name:
        key1 = modin_df.columns[0]
        # Scaler
        assert modin_df.at[0, key1] == pandas_df.at[0, key1]

        # Series
        df_equals(modin_df.loc[0].at[key1], pandas_df.loc[0].at[key1])

        # Write Item
        modin_df_copy = modin_df.copy()
        pandas_df_copy = pandas_df.copy()
        modin_df_copy.at[1, key1] = modin_df.at[0, key1]
        pandas_df_copy.at[1, key1] = pandas_df.at[0, key1]
        df_equals(modin_df_copy, pandas_df_copy)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_axes(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    for modin_axis, pd_axis in zip(modin_df.axes, pandas_df.axes):
        assert np.array_equal(modin_axis, pd_axis)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_copy(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)  # noqa F841

    # pandas_df is unused but there so there won't be confusing list comprehension
    # stuff in the pytest.mark.parametrize
    new_modin_df = modin_df.copy()

    assert new_modin_df is not modin_df
    assert np.array_equal(
        new_modin_df._query_compiler.data.partitions,
        modin_df._query_compiler.data.partitions,
    )
    assert new_modin_df is not modin_df
    df_equals(new_modin_df, modin_df)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_dtypes(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    df_equals(modin_df.dtypes, pandas_df.dtypes)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_ftypes(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    df_equals(modin_df.ftypes, pandas_df.ftypes)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("key", indices_values, ids=indices_keys)
def test_get(data, key):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    df_equals(modin_df.get(key), pandas_df.get(key))
    df_equals(
        modin_df.get(key, default="default"), pandas_df.get(key, default="default")
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_get_dtype_counts(data):
    modin_result = pd.DataFrame(data).get_dtype_counts().sort_index()
    pandas_result = pandas.DataFrame(data).get_dtype_counts().sort_index()

    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize(
    "dummy_na", bool_arg_values, ids=arg_keys("dummy_na", bool_arg_keys)
)
@pytest.mark.parametrize(
    "drop_first", bool_arg_values, ids=arg_keys("drop_first", bool_arg_keys)
)
def test_get_dummies(request, data, dummy_na, drop_first):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas.get_dummies(
            pandas_df, dummy_na=dummy_na, drop_first=drop_first
        )
    except Exception as e:
        with pytest.raises(type(e)):
            pd.get_dummies(modin_df, dummy_na=dummy_na, drop_first=drop_first)
    else:
        modin_result = pd.get_dummies(
            modin_df, dummy_na=dummy_na, drop_first=drop_first
        )
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_get_ftype_counts(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    df_equals(modin_df.get_ftype_counts(), pandas_df.get_ftype_counts())


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
def test_agg(data, axis, func):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.agg(func, axis)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.agg(func, axis)
    else:
        modin_result = modin_df.agg(func, axis)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
def test_agg_numeric(request, data, axis, func):
    if name_contains(request.node.name, numeric_agg_funcs) and name_contains(
        request.node.name, numeric_dfs
    ):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        try:
            pandas_result = pandas_df.agg(func, axis)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.agg(func, axis)
        else:
            modin_result = modin_df.agg(func, axis)
            df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
def test_aggregate(request, data, func, axis):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.aggregate(func, axis)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.aggregate(func, axis)
    else:
        modin_result = modin_df.aggregate(func, axis)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
def test_aggregate_numeric(request, data, axis, func):
    if name_contains(request.node.name, numeric_agg_funcs) and name_contains(
        request.node.name, numeric_dfs
    ):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        try:
            pandas_result = pandas_df.agg(func, axis)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.agg(func, axis)
        else:
            modin_result = modin_df.agg(func, axis)
            df_equals(modin_result, pandas_result)


def test_align():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).align(pd.DataFrame(data))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "bool_only", bool_arg_values, ids=arg_keys("bool_only", bool_arg_keys)
)
def test_all(data, axis, skipna, bool_only):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.all(axis=axis, skipna=skipna, bool_only=bool_only)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.all(axis=axis, skipna=skipna, bool_only=bool_only)
    else:
        modin_result = modin_df.all(axis=axis, skipna=skipna, bool_only=bool_only)
        df_equals(modin_result, pandas_result)

    # Test when axis is None. This will get repeated but easier than using list in parameterize decorator
    try:
        pandas_result = pandas_df.all(axis=None, skipna=skipna, bool_only=bool_only)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.all(axis=None, skipna=skipna, bool_only=bool_only)
    else:
        modin_result = modin_df.all(axis=None, skipna=skipna, bool_only=bool_only)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "bool_only", bool_arg_values, ids=arg_keys("bool_only", bool_arg_keys)
)
def test_any(data, axis, skipna, bool_only):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.any(axis=axis, skipna=skipna, bool_only=bool_only)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.any(axis=axis, skipna=skipna, bool_only=bool_only)
    else:
        modin_result = modin_df.any(axis=axis, skipna=skipna, bool_only=bool_only)
        df_equals(modin_result, pandas_result)

    try:
        pandas_result = pandas_df.any(axis=None, skipna=skipna, bool_only=bool_only)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.any(axis=None, skipna=skipna, bool_only=bool_only)
    else:
        modin_result = modin_df.any(axis=None, skipna=skipna, bool_only=bool_only)
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


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
def test_apply(request, data, func, axis):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    with pytest.raises(TypeError):
        modin_df.apply({"row": func}, axis=1)

    try:
        pandas_result = pandas_df.apply(func, axis)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_result = modin_df.apply(func, axis)
    else:
        modin_result = modin_df.apply(func, axis)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
def test_apply_numeric(request, data, func, axis):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    if name_contains(request.node.name, numeric_dfs):
        try:
            pandas_result = pandas_df.apply(func, axis)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.apply(func, axis)
        else:
            modin_result = modin_df.apply(func, axis)
            df_equals(modin_result, pandas_result)

    if "empty_data" not in request.node.name:
        key = modin_df.columns[0]
        modin_result = modin_df.apply(lambda df: df.drop(key), axis=1)
        pandas_result = pandas_df.apply(lambda df: df.drop(key), axis=1)
        df_equals(modin_result, pandas_result)


def test_as_blocks():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).as_blocks()


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


def test_asfreq():
    index = pd.date_range("1/1/2000", periods=4, freq="T")
    series = pd.Series([0.0, None, 2.0, 3.0], index=index)
    df = pd.DataFrame({"s": series})
    with pytest.warns(UserWarning):
        # We are only testing that this defaults to pandas, so we will just check for
        # the warning
        df.asfreq(freq="30S")


def test_asof():
    df = pd.DataFrame(
        {"a": [10, 20, 30, 40, 50], "b": [None, None, None, None, 500]},
        index=pd.DatetimeIndex(
            [
                "2018-02-27 09:01:00",
                "2018-02-27 09:02:00",
                "2018-02-27 09:03:00",
                "2018-02-27 09:04:00",
                "2018-02-27 09:05:00",
            ]
        ),
    )
    with pytest.warns(UserWarning):
        df.asof(pd.DatetimeIndex(["2018-02-27 09:03:30", "2018-02-27 09:04:30"]))


def test_assign():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).assign()


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

    dtype_dict = {"A": np.int32, "B": np.int64, "C": str}
    modin_df_casted = modin_df.astype(dtype_dict)
    expected_df_casted = expected_df.astype(dtype_dict)
    df_equals(modin_df_casted, expected_df_casted)

    # Ignore lint because this is testing bad input
    bad_dtype_dict = {"B": np.int32, "B": np.int64, "B": str}  # noqa F601
    modin_df_casted = modin_df.astype(bad_dtype_dict)
    expected_df_casted = expected_df.astype(bad_dtype_dict)
    df_equals(modin_df_casted, expected_df_casted)


def test_at_time():
    i = pd.date_range("2018-04-09", periods=4, freq="12H")
    ts = pd.DataFrame({"A": [1, 2, 3, 4]}, index=i)
    with pytest.warns(UserWarning):
        ts.at_time("12:00")


def test_between_time():
    i = pd.date_range("2018-04-09", periods=4, freq="12H")
    ts = pd.DataFrame({"A": [1, 2, 3, 4]}, index=i)
    with pytest.warns(UserWarning):
        ts.between_time("0:15", "0:45")


def test_bfill():
    test_data = TestData()
    test_data.tsframe["A"][:5] = np.nan
    test_data.tsframe["A"][-5:] = np.nan
    modin_df = pd.DataFrame(test_data.tsframe)
    df_equals(modin_df.bfill(), test_data.tsframe.bfill())


def test_blocks():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).blocks


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_bool(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)  # noqa F841

    with pytest.raises(ValueError):
        modin_df.bool()
        modin_df.__bool__()

    single_bool_pandas_df = pandas.DataFrame([True])
    single_bool_modin_df = pd.DataFrame([True])

    assert single_bool_pandas_df.bool() == single_bool_modin_df.bool()

    with pytest.raises(ValueError):
        # __bool__ always raises this error for DataFrames
        single_bool_modin_df.__bool__()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_boxplot(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)  # noqa F841

    assert modin_df.boxplot() == to_pandas(modin_df).boxplot()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
def test_clip(request, data, axis):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    if name_contains(request.node.name, numeric_dfs):
        ind_len = (
            len(modin_df.index)
            if not pandas.DataFrame()._get_axis_number(axis)
            else len(modin_df.columns)
        )
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


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
def test_clip_lower(request, data, axis):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    if name_contains(request.node.name, numeric_dfs):
        ind_len = (
            len(modin_df.index)
            if not pandas.DataFrame()._get_axis_number(axis)
            else len(modin_df.columns)
        )
        # set bounds
        lower = random_state.random_integers(RAND_LOW, RAND_HIGH, 1)[0]
        lower_list = random_state.random_integers(RAND_LOW, RAND_HIGH, ind_len)

        # test lower scalar bound
        pandas_result = pandas_df.clip_lower(lower, axis=axis)
        modin_result = modin_df.clip_lower(lower, axis=axis)
        df_equals(modin_result, pandas_result)

        # test lower list bound on each column
        pandas_result = pandas_df.clip_lower(lower_list, axis=axis)
        modin_result = modin_df.clip_lower(lower_list, axis=axis)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
def test_clip_upper(request, data, axis):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    if name_contains(request.node.name, numeric_dfs):
        ind_len = (
            len(modin_df.index)
            if not pandas.DataFrame()._get_axis_number(axis)
            else len(modin_df.columns)
        )
        # set bounds
        upper = random_state.random_integers(RAND_LOW, RAND_HIGH, 1)[0]
        upper_list = random_state.random_integers(RAND_LOW, RAND_HIGH, ind_len)

        # test upper scalar bound
        modin_result = modin_df.clip_upper(upper, axis=axis)
        pandas_result = pandas_df.clip_upper(upper, axis=axis)
        df_equals(modin_result, pandas_result)

        # test upper list bound on each column
        modin_result = modin_df.clip_upper(upper_list, axis=axis)
        pandas_result = pandas_df.clip_upper(upper_list, axis=axis)
        df_equals(modin_result, pandas_result)


def test_combine():
    df1 = pd.DataFrame({"A": [0, 0], "B": [4, 4]})
    df2 = pd.DataFrame({"A": [1, 1], "B": [3, 3]})

    with pytest.warns(UserWarning):
        df1.combine(df2, lambda s1, s2: s1 if s1.sum() < s2.sum() else s2)


def test_combine_first():
    df1 = pd.DataFrame({"A": [None, 0], "B": [None, 4]})
    df2 = pd.DataFrame({"A": [1, 1], "B": [3, 3]})

    with pytest.warns(UserWarning):
        df1.combine_first(df2)


def test_compound():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).compound()


def test_consolidate():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).consolidate()


def test_convert_objects():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).convert_objects()


def test_corr():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).corr()


def test_corrwith():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).corrwith(pd.DataFrame(data))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
)
def test_count(request, data, axis, numeric_only):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    modin_result = modin_df.count(axis=axis, numeric_only=numeric_only)
    pandas_result = pandas_df.count(axis=axis, numeric_only=numeric_only)
    df_equals(modin_result, pandas_result)


def test_cov():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).cov()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_cummax(request, data, axis, skipna):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.cummax(axis=axis, skipna=skipna)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.cummax(axis=axis, skipna=skipna)
    else:
        modin_result = modin_df.cummax(axis=axis, skipna=skipna)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_cummin(request, data, axis, skipna):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.cummin(axis=axis, skipna=skipna)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.cummin(axis=axis, skipna=skipna)
    else:
        modin_result = modin_df.cummin(axis=axis, skipna=skipna)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_cumprod(request, data, axis, skipna):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.cumprod(axis=axis, skipna=skipna)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.cumprod(axis=axis, skipna=skipna)
    else:
        modin_result = modin_df.cumprod(axis=axis, skipna=skipna)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_cumsum(request, data, axis, skipna):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    # pandas exhibits weird behavior for this case
    # Remove this case when we can pull the error messages from backend
    if name_contains(request.node.name, ["datetime_timedelta_data"]) and (
        axis == 0 or axis == "rows"
    ):
        with pytest.raises(TypeError):
            modin_df.cumsum(axis=axis, skipna=skipna)
    else:
        try:
            pandas_result = pandas_df.cumsum(axis=axis, skipna=skipna)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.cumsum(axis=axis, skipna=skipna)
        else:
            modin_result = modin_df.cumsum(axis=axis, skipna=skipna)
            df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_describe(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    df_equals(modin_df.describe(), pandas_df.describe())


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "periods", int_arg_values, ids=arg_keys("periods", int_arg_keys)
)
def test_diff(request, data, axis, periods):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.diff(axis=axis, periods=periods)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.diff(axis=axis, periods=periods)
    else:
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
    df_equals(modin_simple.drop([0, 5], errors="ignore"), simple.loc[[1, 2, 3], :])
    df_equals(modin_simple.drop("C", axis=1, errors="ignore"), simple)
    df_equals(modin_simple.drop(["A", "C"], axis=1, errors="ignore"), simple[["B"]])

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


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_drop_duplicates(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    df_equals(
        modin_df.drop_duplicates(keep="first", inplace=False),
        pandas_df.drop_duplicates(keep="first", inplace=False),
    )

    df_equals(
        modin_df.drop_duplicates(keep="last", inplace=False),
        pandas_df.drop_duplicates(keep="last", inplace=False),
    )

    df_equals(
        modin_df.drop_duplicates(keep=False, inplace=False),
        pandas_df.drop_duplicates(keep=False, inplace=False),
    )

    df_equals(
        modin_df.drop_duplicates(inplace=False),
        pandas_df.drop_duplicates(inplace=False),
    )

    modin_df.drop_duplicates(inplace=True)
    df_equals(modin_df, pandas_df.drop_duplicates(inplace=False))

    frame_data = {
        "A": list(range(3)) * 2,
        "B": list(range(1, 4)) * 2,
        "C": list(range(6)),
    }
    modin_df = pd.DataFrame(frame_data)
    modin_df.drop_duplicates(subset=["A", "B"], keep=False, inplace=True)
    df_equals(modin_df, pandas.DataFrame({"A": [], "B": [], "C": []}))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("how", ["any", "all"], ids=["any", "all"])
def test_dropna(data, axis, how):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    with pytest.raises(ValueError):
        modin_df.dropna(axis=axis, how="invalid")

    with pytest.raises(TypeError):
        modin_df.dropna(axis=axis, how=None, thresh=None)

    with pytest.raises(KeyError):
        modin_df.dropna(axis=axis, subset=["NotExists"], how=how)

    modin_result = modin_df.dropna(axis=axis, how=how)
    pandas_result = pandas_df.dropna(axis=axis, how=how)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_dropna_inplace(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    pandas_result = pandas_df.dropna()
    modin_df.dropna(inplace=True)
    df_equals(modin_df, pandas_result)

    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    pandas_df.dropna(thresh=2, inplace=True)
    modin_df.dropna(thresh=2, inplace=True)
    df_equals(modin_df, pandas_df)

    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    pandas_df.dropna(axis=1, how="any", inplace=True)
    modin_df.dropna(axis=1, how="any", inplace=True)
    df_equals(modin_df, pandas_df)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_dropna_multiple_axes(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    df_equals(
        modin_df.dropna(how="all", axis=[0, 1]),
        pandas_df.dropna(how="all", axis=[0, 1]),
    )
    df_equals(
        modin_df.dropna(how="all", axis=(0, 1)),
        pandas_df.dropna(how="all", axis=(0, 1)),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_dropna_multiple_axes_inplace(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

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


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_dropna_subset(request, data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

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


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_dropna_subset_error(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)  # noqa F841

    # pandas_df is unused so there won't be confusing list comprehension
    # stuff in the pytest.mark.parametrize
    with pytest.raises(KeyError):
        modin_df.dropna(subset=list("EF"))

    if len(modin_df.columns) < 5:
        with pytest.raises(KeyError):
            modin_df.dropna(axis=1, subset=[4, 5])


def test_dot():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).dot(pd.DataFrame(data).T)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_duplicated(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    pandas_result = pandas_df.duplicated()
    modin_result = modin_df.duplicated()

    df_equals(modin_result, pandas_result)


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
    df_equals(modin_df1, pd.DataFrame(modin_df1))

    frame_data = {"col1": [2.9, 3, 3, 3], "col2": [2, 3, 5, 1]}
    modin_df3 = pd.DataFrame(frame_data)

    with pytest.raises(AssertionError):
        df_equals(modin_df3, modin_df1)

    with pytest.raises(AssertionError):
        df_equals(modin_df3, modin_df2)


def test_eval_df_use_case():
    frame_data = {"a": random_state.randn(10), "b": random_state.randn(10)}
    df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)

    # test eval for series results
    tmp_pandas = df.eval("arctan2(sin(a), b)", engine="python", parser="pandas")
    tmp_modin = modin_df.eval("arctan2(sin(a), b)", engine="python", parser="pandas")

    assert isinstance(tmp_modin, (pandas.Series, SeriesView))
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


def test_ewm():
    df = pd.DataFrame({"B": [0, 1, 2, np.nan, 4]})
    with pytest.warns(UserWarning):
        df.ewm(com=0.5).mean()


def test_expanding():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).expanding()


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
    df_equals(modin_df.fillna(method="ffill"), test_data.tsframe.fillna(method="ffill"))


def test_bfill2():
    test_data = TestData()
    test_data.tsframe["A"][:5] = np.nan
    test_data.tsframe["A"][-5:] = np.nan
    modin_df = pd.DataFrame(test_data.tsframe)
    df_equals(modin_df.fillna(method="bfill"), test_data.tsframe.fillna(method="bfill"))


def test_fillna_inplace():
    frame_data = random_state.randn(10, 4)
    df = pandas.DataFrame(frame_data)
    df[1][:4] = np.nan
    df[3][-4:] = np.nan

    modin_df = pd.DataFrame(df)
    df.fillna(value=0, inplace=True)
    try:
        df_equals(modin_df, df)
    except AssertionError:
        pass
    else:
        assert False

    modin_df.fillna(value=0, inplace=True)
    df_equals(modin_df, df)

    modin_df = pd.DataFrame(df).fillna(value={0: 0}, inplace=True)
    assert modin_df is None

    df[1][:4] = np.nan
    df[3][-4:] = np.nan
    modin_df = pd.DataFrame(df)
    df.fillna(method="ffill", inplace=True)
    try:
        df_equals(modin_df, df)
    except AssertionError:
        pass
    else:
        assert False

    modin_df.fillna(method="ffill", inplace=True)
    df_equals(modin_df, df)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_frame_fillna_limit(data):
    pandas_df = pandas.DataFrame(data)

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


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_frame_pad_backfill_limit(data):
    pandas_df = pandas.DataFrame(data)

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


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_fillna_skip_certain_blocks(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    # don't try to fill boolean, int blocks
    df_equals(modin_df.fillna(np.nan), pandas_df.fillna(np.nan))


def test_fillna_dict_series():
    frame_data = {
        "a": [np.nan, 1, 2, np.nan, np.nan],
        "b": [1, 2, 3, np.nan, np.nan],
        "c": [np.nan, 1, 2, 3, 4],
    }
    df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)

    df_equals(modin_df.fillna({"a": 0, "b": 5}), df.fillna({"a": 0, "b": 5}))

    df_equals(
        modin_df.fillna({"a": 0, "b": 5, "d": 7}), df.fillna({"a": 0, "b": 5, "d": 7})
    )

    # Series treated same as dict
    df_equals(modin_df.fillna(df.max()), df.fillna(df.max()))


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
    df_equals(modin_df.fillna(df2), df.fillna(df2))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_fillna_columns(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    df_equals(
        modin_df.fillna(method="ffill", axis=1),
        pandas_df.fillna(method="ffill", axis=1),
    )

    df_equals(
        modin_df.fillna(method="ffill", axis=1),
        pandas_df.fillna(method="ffill", axis=1),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_fillna_invalid_method(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)  # noqa F841

    with tm.assert_raises_regex(ValueError, "ffil"):
        modin_df.fillna(method="ffil")


def test_fillna_invalid_value():
    test_data = TestData()
    modin_df = pd.DataFrame(test_data.frame)
    # list
    pytest.raises(TypeError, modin_df.fillna, [1, 2])
    # tuple
    pytest.raises(TypeError, modin_df.fillna, (1, 2))
    # frame with series
    pytest.raises(TypeError, modin_df.iloc[:, 0].fillna, modin_df)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_fillna_col_reordering(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

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


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_filter(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    by = {"items": ["col1", "col5"], "regex": "4$|3$", "like": "col"}
    df_equals(modin_df.filter(items=by["items"]), pandas_df.filter(items=by["items"]))

    df_equals(
        modin_df.filter(regex=by["regex"], axis=0),
        pandas_df.filter(regex=by["regex"], axis=0),
    )
    df_equals(
        modin_df.filter(regex=by["regex"], axis=1),
        pandas_df.filter(regex=by["regex"], axis=1),
    )

    df_equals(modin_df.filter(like=by["like"]), pandas_df.filter(like=by["like"]))


def test_first():
    i = pd.date_range("2018-04-09", periods=4, freq="2D")
    ts = pd.DataFrame({"A": [1, 2, 3, 4]}, index=i)
    with pytest.warns(UserWarning):
        ts.first("3D")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_first_valid_index(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    assert modin_df.first_valid_index() == (pandas_df.first_valid_index())


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_from_dict(data):
    modin_df = pd.DataFrame(data)  # noqa F841
    pandas_df = pandas.DataFrame(data)  # noqa F841

    with pytest.raises(NotImplementedError):
        pd.DataFrame.from_dict(None)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_from_items(data):
    modin_df = pd.DataFrame(data)  # noqa F841
    pandas_df = pandas.DataFrame(data)  # noqa F841

    with pytest.raises(NotImplementedError):
        pd.DataFrame.from_items(None)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_from_records(data):
    modin_df = pd.DataFrame(data)  # noqa F841
    pandas_df = pandas.DataFrame(data)  # noqa F841

    with pytest.raises(NotImplementedError):
        pd.DataFrame.from_records(None)


def test_get_value():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).get_value(0, "col1")


def test_get_values():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).get_values()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("n", int_arg_values, ids=arg_keys("n", int_arg_keys))
def test_head(data, n):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    df_equals(modin_df.head(n), pandas_df.head(n))


@pytest.mark.skip(reason="Skip plotting")
def test_hist():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).hist(None)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_iat(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)  # noqa F841

    with pytest.raises(NotImplementedError):
        modin_df.iat()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_idxmax(data, axis, skipna):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    modin_result = modin_df.all(axis=axis, skipna=skipna)
    pandas_result = pandas_df.all(axis=axis, skipna=skipna)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_idxmin(data, axis, skipna):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    modin_result = modin_df.all(axis=axis, skipna=skipna)
    pandas_result = pandas_df.all(axis=axis, skipna=skipna)
    df_equals(modin_result, pandas_result)


def test_infer_objects():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).infer_objects()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_iloc(request, data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    if not name_contains(request.node.name, ["empty_data"]):
        # Scaler
        np.testing.assert_equal(modin_df.iloc[0, 1], pandas_df.iloc[0, 1])

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


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_index(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    df_equals(modin_df.index, pandas_df.index)
    modin_df_cp = modin_df.copy()
    pandas_df_cp = pandas_df.copy()

    modin_df_cp.index = [str(i) for i in modin_df_cp.index]
    pandas_df_cp.index = [str(i) for i in pandas_df_cp.index]
    df_equals(modin_df_cp.index, pandas_df_cp.index)


def test_info():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).info(memory_usage="deep")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("loc", int_arg_values, ids=arg_keys("loc", int_arg_keys))
def test_insert(data, loc):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    modin_df = modin_df.copy()
    pandas_df = pandas_df.copy()
    column = "New Column"
    value = modin_df.iloc[:, 0]

    try:
        pandas_df.insert(loc, column, value)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.insert(loc, column, value)
    else:
        modin_df.insert(loc, column, value)
        df_equals(modin_df, pandas_df)


def test_interpolate():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).interpolate()


def test_is_copy():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).is_copy


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_items(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    modin_items = modin_df.items()
    pandas_items = pandas_df.items()
    for modin_item, pandas_item in zip(modin_items, pandas_items):
        modin_index, modin_series = modin_item
        pandas_index, pandas_series = pandas_item
        df_equals(pandas_series, modin_series)
        assert pandas_index == modin_index


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_iteritems(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    modin_items = modin_df.iteritems()
    pandas_items = pandas_df.iteritems()
    for modin_item, pandas_item in zip(modin_items, pandas_items):
        modin_index, modin_series = modin_item
        pandas_index, pandas_series = pandas_item
        df_equals(pandas_series, modin_series)
        assert pandas_index == modin_index


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_iterrows(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    modin_iterrows = modin_df.iterrows()
    pandas_iterrows = pandas_df.iterrows()
    for modin_row, pandas_row in zip(modin_iterrows, pandas_iterrows):
        modin_index, modin_series = modin_row
        pandas_index, pandas_series = pandas_row
        df_equals(pandas_series, modin_series)
        assert pandas_index == modin_index


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_itertuples(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

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


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_ix(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)  # noqa F841

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


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_keys(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    df_equals(modin_df.keys(), pandas_df.keys())


def test_kurt():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).kurt()


def test_kurtosis():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).kurtosis()


def test_last():
    i = pd.date_range("2018-04-09", periods=4, freq="2D")
    ts = pd.DataFrame({"A": [1, 2, 3, 4]}, index=i)
    with pytest.warns(UserWarning):
        ts.last("3D")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_last_valid_index(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    assert modin_df.last_valid_index() == (pandas_df.last_valid_index())


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_loc(request, data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

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

        # From issue #421
        df_equals(modin_df.loc[:, [key2, key1]], pandas_df.loc[:, [key2, key1]])
        df_equals(modin_df.loc[[2, 1], :], pandas_df.loc[[2, 1], :])

        # Write Item
        modin_df_copy = modin_df.copy()
        pandas_df_copy = pandas_df.copy()
        modin_df_copy.loc[[1, 2]] = 42
        pandas_df_copy.loc[[1, 2]] = 42
        df_equals(modin_df_copy, pandas_df_copy)


def test_lookup():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).lookup([0, 1], ["col1", "col2"])


def test_mad():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).mad()


def test_mask():
    df = pd.DataFrame(np.arange(10).reshape(-1, 2), columns=["A", "B"])
    m = df % 3 == 0
    with pytest.warns(UserWarning):
        try:
            df.mask(~m, -df)
        except ValueError:
            pass


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
)
def test_max(request, data, axis, skipna, numeric_only):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.max(
            axis=axis, skipna=skipna, numeric_only=numeric_only
        )
    except Exception:
        with pytest.raises(TypeError):
            modin_result = modin_df.max(
                axis=axis, skipna=skipna, numeric_only=numeric_only
            )
    else:
        modin_result = modin_df.max(axis=axis, skipna=skipna, numeric_only=numeric_only)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
)
def test_mean(request, data, axis, skipna, numeric_only):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.mean(
            axis=axis, skipna=skipna, numeric_only=numeric_only
        )
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.mean(axis=axis, skipna=skipna, numeric_only=numeric_only)
    else:
        modin_result = modin_df.mean(
            axis=axis, skipna=skipna, numeric_only=numeric_only
        )
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
)
def test_median(request, data, axis, skipna, numeric_only):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.median(
            axis=axis, skipna=skipna, numeric_only=numeric_only
        )
    except Exception:
        with pytest.raises(TypeError):
            modin_df.median(axis=axis, skipna=skipna, numeric_only=numeric_only)
    else:
        modin_result = modin_df.median(
            axis=axis, skipna=skipna, numeric_only=numeric_only
        )
        df_equals(modin_result, pandas_result)


def test_melt():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).melt()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_memory_usage(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)  # noqa F841

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


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
)
def test_min(data, axis, skipna, numeric_only):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.min(
            axis=axis, skipna=skipna, numeric_only=numeric_only
        )
    except Exception:
        with pytest.raises(TypeError):
            modin_df.min(axis=axis, skipna=skipna, numeric_only=numeric_only)
    else:
        modin_result = modin_df.min(axis=axis, skipna=skipna, numeric_only=numeric_only)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
)
def test_mode(request, data, axis, numeric_only):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.mode(axis=axis, numeric_only=numeric_only)
    except Exception:
        with pytest.raises(TypeError):
            modin_df.mode(axis=axis, numeric_only=numeric_only)
    else:
        modin_result = modin_df.mode(axis=axis, numeric_only=numeric_only)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_ndim(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    assert modin_df.ndim == pandas_df.ndim


def test_nlargest():
    df = pd.DataFrame(
        {
            "population": [
                59000000,
                65000000,
                434000,
                434000,
                434000,
                337000,
                11300,
                11300,
                11300,
            ],
            "GDP": [1937894, 2583560, 12011, 4520, 12128, 17036, 182, 38, 311],
            "alpha-2": ["IT", "FR", "MT", "MV", "BN", "IS", "NR", "TV", "AI"],
        },
        index=[
            "Italy",
            "France",
            "Malta",
            "Maldives",
            "Brunei",
            "Iceland",
            "Nauru",
            "Tuvalu",
            "Anguilla",
        ],
    )
    with pytest.warns(UserWarning):
        df.nlargest(3, "population")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_notna(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    df_equals(modin_df.notna(), pandas_df.notna())


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_notnull(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    df_equals(modin_df.notnull(), pandas_df.notnull())


def test_nsmallest():
    df = pd.DataFrame(
        {
            "population": [
                59000000,
                65000000,
                434000,
                434000,
                434000,
                337000,
                11300,
                11300,
                11300,
            ],
            "GDP": [1937894, 2583560, 12011, 4520, 12128, 17036, 182, 38, 311],
            "alpha-2": ["IT", "FR", "MT", "MV", "BN", "IS", "NR", "TV", "AI"],
        },
        index=[
            "Italy",
            "France",
            "Malta",
            "Maldives",
            "Brunei",
            "Iceland",
            "Nauru",
            "Tuvalu",
            "Anguilla",
        ],
    )
    with pytest.warns(UserWarning):
        df.nsmallest(3, "population")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "dropna", bool_arg_values, ids=arg_keys("dropna", bool_arg_keys)
)
def test_nunique(data, axis, dropna):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    modin_result = modin_df.nunique(axis=axis, dropna=dropna)
    pandas_result = pandas_df.nunique(axis=axis, dropna=dropna)
    df_equals(modin_result, pandas_result)


def test_pct_change():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).pct_change()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_pipe(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

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


def test_pivot():
    df = pd.DataFrame(
        {
            "foo": ["one", "one", "one", "two", "two", "two"],
            "bar": ["A", "B", "C", "A", "B", "C"],
            "baz": [1, 2, 3, 4, 5, 6],
            "zoo": ["x", "y", "z", "q", "w", "t"],
        }
    )
    with pytest.warns(UserWarning):
        df.pivot(index="foo", columns="bar", values="baz")


def test_pivot_table():
    df = pd.DataFrame(
        {
            "A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
            "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
            "C": [
                "small",
                "large",
                "large",
                "small",
                "small",
                "large",
                "small",
                "small",
                "large",
            ],
            "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
            "E": [2, 4, 5, 5, 6, 6, 8, 9, 9],
        }
    )
    with pytest.warns(UserWarning):
        df.pivot_table(values="D", index=["A", "B"], columns=["C"], aggfunc=np.sum)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_plot(request, data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    if name_contains(request.node.name, numeric_dfs):
        # We have to test this way because equality in plots means same object.
        zipped_plot_lines = zip(modin_df.plot().lines, pandas_df.plot().lines)
        for l, r in zipped_plot_lines:
            if isinstance(l.get_xdata(), np.ma.core.MaskedArray) and isinstance(
                r.get_xdata(), np.ma.core.MaskedArray
            ):
                assert all((l.get_xdata() == r.get_xdata()).data)
            else:
                assert np.array_equal(l.get_xdata(), r.get_xdata())
            if isinstance(l.get_ydata(), np.ma.core.MaskedArray) and isinstance(
                r.get_ydata(), np.ma.core.MaskedArray
            ):
                assert all((l.get_ydata() == r.get_ydata()).data)
            else:
                assert np.array_equal(l.get_xdata(), r.get_xdata())


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_pop(request, data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    if "empty_data" not in request.node.name:
        key = modin_df.columns[0]
        temp_modin_df = modin_df.copy()
        temp_pandas_df = pandas_df.copy()
        modin_popped = temp_modin_df.pop(key)
        pandas_popped = temp_pandas_df.pop(key)
        df_equals(modin_popped, pandas_popped)
        df_equals(temp_modin_df, temp_pandas_df)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
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
def test_prod(request, data, axis, skipna, numeric_only, min_count):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.prod(
            axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
        )
    except Exception:
        with pytest.raises(TypeError):
            modin_df.prod(
                axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
            )
    else:
        modin_result = modin_df.prod(
            axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
        )
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
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
def test_product(request, data, axis, skipna, numeric_only, min_count):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.product(
            axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
        )
    except Exception:
        with pytest.raises(TypeError):
            modin_df.product(
                axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
            )
    else:
        modin_result = modin_df.product(
            axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
        )
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("q", quantiles_values, ids=quantiles_keys)
def test_quantile(request, data, q):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    if not name_contains(request.node.name, no_numeric_dfs):
        df_equals(modin_df.quantile(q), pandas_df.quantile(q))
    else:
        with pytest.raises(ValueError):
            modin_df.quantile(q)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("funcs", query_func_values, ids=query_func_keys)
def test_query(data, funcs):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    with pytest.raises(ValueError):
        modin_df.query("")
    with pytest.raises(NotImplementedError):
        x = 2  # noqa F841
        modin_df.query("col1 < @x")

    try:
        pandas_result = pandas_df.query(funcs)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.query(funcs)
    else:
        modin_result = modin_df.query(funcs)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
)
@pytest.mark.parametrize(
    "na_option", ["keep", "top", "bottom"], ids=["keep", "top", "bottom"]
)
def test_rank(data, axis, numeric_only, na_option):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.rank(
            axis=axis, numeric_only=numeric_only, na_option=na_option
        )
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.rank(axis=axis, numeric_only=numeric_only, na_option=na_option)
    else:
        modin_result = modin_df.rank(
            axis=axis, numeric_only=numeric_only, na_option=na_option
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


def test_reindex_axis():
    df = pd.DataFrame({"num_legs": [4, 2], "num_wings": [0, 2]}, index=["dog", "hawk"])
    with pytest.warns(UserWarning):
        df.reindex_axis(["num_wings", "num_legs", "num_heads"], axis="columns")


def test_reindex_like():
    df1 = pd.DataFrame(
        [
            [24.3, 75.7, "high"],
            [31, 87.8, "high"],
            [22, 71.6, "medium"],
            [35, 95, "medium"],
        ],
        columns=["temp_celsius", "temp_fahrenheit", "windspeed"],
        index=pd.date_range(start="2014-02-12", end="2014-02-15", freq="D"),
    )
    df2 = pd.DataFrame(
        [[28, "low"], [30, "low"], [35.1, "medium"]],
        columns=["temp_celsius", "windspeed"],
        index=pd.DatetimeIndex(["2014-02-12", "2014-02-13", "2014-02-15"]),
    )
    with pytest.warns(UserWarning):
        df2.reindex_like(df1)


def test_rename_sanity():
    test_data = TestData()
    mapping = {"A": "a", "B": "b", "C": "c", "D": "d"}

    modin_df = pd.DataFrame(test_data.frame)
    df_equals(modin_df.rename(columns=mapping), test_data.frame.rename(columns=mapping))

    renamed2 = test_data.frame.rename(columns=str.lower)
    df_equals(modin_df.rename(columns=str.lower), renamed2)

    modin_df = pd.DataFrame(renamed2)
    df_equals(modin_df.rename(columns=str.upper), renamed2.rename(columns=str.upper))

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


@pytest.mark.skip(reason="Pandas does not pass this test")
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


def test_reorder_levels():
    df = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(
            [
                (num, letter, color)
                for num in range(1, 3)
                for letter in ["a", "b", "c"]
                for color in ["Red", "Green"]
            ],
            names=["Number", "Letter", "Color"],
        )
    )
    df["Value"] = np.random.randint(1, 100, len(df))
    with pytest.warns(UserWarning):
        df.reorder_levels(["Letter", "Color", "Number"])


def test_replace():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).replace()


def test_resample():
    d = dict(
        {
            "price": [10, 11, 9, 13, 14, 18, 17, 19],
            "volume": [50, 60, 40, 100, 50, 100, 40, 50],
        }
    )
    df = pd.DataFrame(d)
    df["week_starting"] = pd.date_range("01/01/2018", periods=8, freq="W")
    with pytest.warns(UserWarning):
        df.resample("M", on="week_starting")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_reset_index(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    modin_result = modin_df.reset_index(inplace=False)
    pandas_result = pandas_df.reset_index(inplace=False)
    df_equals(modin_result, pandas_result)

    modin_df_cp = modin_df.copy()
    pd_df_cp = pandas_df.copy()
    modin_df_cp.reset_index(inplace=True)
    pd_df_cp.reset_index(inplace=True)
    df_equals(modin_df_cp, pd_df_cp)


def test_rolling():
    df = pd.DataFrame({"B": [0, 1, 2, np.nan, 4]})
    with pytest.warns(UserWarning):
        df.rolling(2, win_type="triang")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_round(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    df_equals(modin_df.round(), pandas_df.round())
    df_equals(modin_df.round(1), pandas_df.round(1))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
def test_sample(data, axis):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    with pytest.raises(ValueError):
        modin_df.sample(n=3, frac=0.4, axis=axis)

    with pytest.raises(KeyError):
        modin_df.sample(frac=0.5, weights="CoLuMn_No_ExIsT", axis=0)

    with pytest.raises(ValueError):
        modin_df.sample(frac=0.5, weights=modin_df.columns[0], axis=1)

    with pytest.raises(ValueError):
        modin_df.sample(
            frac=0.5, weights=[0.5 for _ in range(len(modin_df.index[:-1]))], axis=0
        )

    with pytest.raises(ValueError):
        modin_df.sample(
            frac=0.5, weights=[0.5 for _ in range(len(modin_df.columns[:-1]))], axis=1
        )

    with pytest.raises(ValueError):
        modin_df.sample(n=-3, axis=axis)

    with pytest.raises(ValueError):
        modin_df.sample(frac=0.2, weights=pandas.Series(), axis=axis)

    if isinstance(axis, str):
        num_axis = pandas.DataFrame()._get_axis_number(axis)
    else:
        num_axis = axis

    # weights that sum to 1
    sums = sum(i % 2 for i in range(len(modin_df.axes[num_axis])))
    weights = [i % 2 / sums for i in range(len(modin_df.axes[num_axis]))]

    modin_result = modin_df.sample(
        frac=0.5, random_state=42, weights=weights, axis=axis
    )
    pandas_result = pandas_df.sample(
        frac=0.5, random_state=42, weights=weights, axis=axis
    )
    df_equals(modin_result, pandas_result)

    # weights that don't sum to 1
    weights = [i % 2 for i in range(len(modin_df.axes[num_axis]))]
    modin_result = modin_df.sample(
        frac=0.5, random_state=42, weights=weights, axis=axis
    )
    pandas_result = pandas_df.sample(
        frac=0.5, random_state=42, weights=weights, axis=axis
    )
    df_equals(modin_result, pandas_result)

    modin_result = modin_df.sample(n=0, axis=axis)
    pandas_result = pandas_df.sample(n=0, axis=axis)
    df_equals(modin_result, pandas_result)

    modin_result = modin_df.sample(frac=0.5, random_state=42, axis=axis)
    pandas_result = pandas_df.sample(frac=0.5, random_state=42, axis=axis)
    df_equals(modin_result, pandas_result)

    modin_result = modin_df.sample(n=2, random_state=42, axis=axis)
    pandas_result = pandas_df.sample(n=2, random_state=42, axis=axis)
    df_equals(modin_result, pandas_result)


def test_select():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).select(lambda x: x % 2 == 0)


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


def test_sem():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).sem()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
def test_set_axis(data, axis):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    x = pandas.DataFrame()._get_axis_number(axis)
    index = modin_df.columns if x else modin_df.index
    labels = ["{0}_{1}".format(index[i], i) for i in range(modin_df.shape[x])]

    modin_result = modin_df.set_axis(labels, axis=axis, inplace=False)
    pandas_result = pandas_df.set_axis(labels, axis=axis, inplace=False)
    df_equals(modin_result, pandas_result)

    modin_df_copy = modin_df.copy()
    modin_df.set_axis(labels, axis=axis, inplace=True)

    # Check that the copy and original are different
    try:
        df_equals(modin_df, modin_df_copy)
    except AssertionError:
        assert True
    else:
        assert False

    pandas_df.set_axis(labels, axis=axis, inplace=True)
    df_equals(modin_df, pandas_df)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("drop", bool_arg_values, ids=arg_keys("drop", bool_arg_keys))
@pytest.mark.parametrize(
    "append", bool_arg_values, ids=arg_keys("append", bool_arg_keys)
)
def test_set_index(request, data, drop, append):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    if "empty_data" not in request.node.name:
        key = modin_df.columns[0]
        modin_result = modin_df.set_index(key, drop=drop, append=append, inplace=False)
        pandas_result = pandas_df.set_index(
            key, drop=drop, append=append, inplace=False
        )
        df_equals(modin_result, pandas_result)

        modin_df_copy = modin_df.copy()
        modin_df.set_index(key, drop=drop, append=append, inplace=True)

        # Check that the copy and original are different
        try:
            df_equals(modin_df, modin_df_copy)
        except AssertionError:
            assert True
        else:
            assert False

        pandas_df.set_index(key, drop=drop, append=append, inplace=True)
        df_equals(modin_df, pandas_df)


def test_set_value():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).set_value(0, 0, 0)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_shape(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    assert modin_df.shape == pandas_df.shape


def test_shift():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).shift()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_size(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    assert modin_df.size == pandas_df.size


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
)
def test_skew(request, data, axis, skipna, numeric_only):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.skew(
            axis=axis, skipna=skipna, numeric_only=numeric_only
        )
    except Exception:
        with pytest.raises(TypeError):
            modin_df.skew(axis=axis, skipna=skipna, numeric_only=numeric_only)
    else:
        modin_result = modin_df.skew(
            axis=axis, skipna=skipna, numeric_only=numeric_only
        )
        df_equals(modin_result, pandas_result)


def test_slice_shift():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).slice_shift()


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


def test_sortlevel():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).sortlevel()


def test_squeeze():
    frame_data = {
        "col1": [0, 1, 2, 3],
        "col2": [4, 5, 6, 7],
        "col3": [8, 9, 10, 11],
        "col4": [12, 13, 14, 15],
        "col5": [0, 0, 0, 0],
    }
    frame_data_2 = {"col1": [0, 1, 2, 3]}
    frame_data_3 = {"col1": [0], "col2": [4], "col3": [8], "col4": [12], "col5": [0]}
    frame_data_4 = {"col1": [2]}
    frame_data_5 = {"col1": ["string"]}
    # Different data for different cases
    pandas_df = pandas.DataFrame(frame_data).squeeze()
    ray_df = pd.DataFrame(frame_data).squeeze()
    df_equals(ray_df, pandas_df)

    pandas_df_2 = pandas.DataFrame(frame_data_2).squeeze()
    ray_df_2 = pd.DataFrame(frame_data_2).squeeze()
    df_equals(ray_df_2, pandas_df_2)

    pandas_df_3 = pandas.DataFrame(frame_data_3).squeeze()
    ray_df_3 = pd.DataFrame(frame_data_3).squeeze()
    df_equals(ray_df_3, pandas_df_3)

    pandas_df_4 = pandas.DataFrame(frame_data_4).squeeze()
    ray_df_4 = pd.DataFrame(frame_data_4).squeeze()
    df_equals(ray_df_4, pandas_df_4)

    pandas_df_5 = pandas.DataFrame(frame_data_5).squeeze()
    ray_df_5 = pd.DataFrame(frame_data_5).squeeze()
    df_equals(ray_df_5, pandas_df_5)


def test_stack():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).stack()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
)
@pytest.mark.parametrize("ddof", int_arg_values, ids=arg_keys("ddof", int_arg_keys))
def test_std(request, data, axis, skipna, numeric_only, ddof):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.std(
            axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof
        )
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.std(axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof)
    else:
        modin_result = modin_df.std(
            axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof
        )
        df_equals(modin_result, pandas_result)


def test_style():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).style


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
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
def test_sum(request, data, axis, skipna, numeric_only, min_count):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.sum(
            axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
        )
    except Exception:
        with pytest.raises(TypeError):
            modin_df.sum(
                axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
            )
    else:
        modin_result = modin_df.sum(
            axis=axis, skipna=skipna, numeric_only=numeric_only, min_count=min_count
        )
        df_equals(modin_result, pandas_result)


def test_swapaxes():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).swapaxes(0, 1)


def test_swaplevel():
    df = pd.DataFrame(
        index=pd.MultiIndex.from_tuples(
            [
                (num, letter, color)
                for num in range(1, 3)
                for letter in ["a", "b", "c"]
                for color in ["Red", "Green"]
            ],
            names=["Number", "Letter", "Color"],
        )
    )
    df["Value"] = np.random.randint(1, 100, len(df))
    with pytest.warns(UserWarning):
        df.swaplevel("Number", "Color")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("n", int_arg_values, ids=arg_keys("n", int_arg_keys))
def test_tail(data, n):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    df_equals(modin_df.tail(n), pandas_df.tail(n))
    df_equals(modin_df.tail(len(modin_df)), pandas_df.tail(len(pandas_df)))


def test_take():
    df = pd.DataFrame(
        [
            ("falcon", "bird", 389.0),
            ("parrot", "bird", 24.0),
            ("lion", "mammal", 80.5),
            ("monkey", "mammal", np.nan),
        ],
        columns=["name", "class", "max_speed"],
        index=[0, 2, 3, 1],
    )
    with pytest.warns(UserWarning):
        df.take([0, 3])


def test_to_datetime():
    frame_data = {"year": [2015, 2016], "month": [2, 3], "day": [4, 5]}
    modin_df = pd.DataFrame(frame_data)
    pd_df = pandas.DataFrame(frame_data)

    df_equals(pd.to_datetime(modin_df), pandas.to_datetime(pd_df))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_to_records(request, data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    # Skips nan because only difference is nan instead of NaN
    if not name_contains(request.node.name, ["nan"]):
        assert np.array_equal(modin_df.to_records(), pandas_df.to_records())


def test_to_sparse():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).to_sparse()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_to_string(request, data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)  # noqa F841

    # Skips nan because only difference is nan instead of NaN
    if not name_contains(request.node.name, ["nan"]):
        assert modin_df.to_string() == to_pandas(modin_df).to_string()


def test_to_timestamp():
    idx = pd.date_range("1/1/2012", periods=5, freq="M")
    df = pd.DataFrame(np.random.randint(0, 100, size=(len(idx), 4)), index=idx)

    with pytest.warns(UserWarning):
        df.to_period().to_timestamp()


def test_to_xarray():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).to_xarray()


@pytest.mark.skip(
    reason="We do not have support to check if a UDF can only take in numeric functions"
)
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
def test_transform(request, data, func):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.agg(func)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.agg(func)
    else:
        modin_result = modin_df.agg(func)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
def test_transform_numeric(request, data, func):
    if name_contains(request.node.name, numeric_agg_funcs) and name_contains(
        request.node.name, numeric_dfs
    ):
        modin_df = pd.DataFrame(data)
        pandas_df = pandas.DataFrame(data)

        try:
            pandas_result = pandas_df.agg(func)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_df.agg(func)
        else:
            modin_result = modin_df.agg(func)
            df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_transpose(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    df_equals(modin_df.T, pandas_df.T)
    df_equals(modin_df.transpose(), pandas_df.transpose())
    # Uncomment below once #165 is merged
    # Test for map across full axis for select indices
    # df_equals(modin_df.T.dropna(), pandas_df.T.dropna())
    # Test for map across full axis
    # df_equals(modin_df.T.nunique(), pandas_df.T.nunique())
    # Test for map across blocks
    # df_equals(modin_df.T.notna(), pandas_df.T.notna())


def test_truncate():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).truncate()


def test_tshift():
    idx = pd.date_range("1/1/2012", periods=5, freq="M")
    df = pd.DataFrame(np.random.randint(0, 100, size=(len(idx), 4)), index=idx)

    with pytest.warns(UserWarning):
        df.to_period().tshift()


def test_tz_convert():
    idx = pd.date_range("1/1/2012", periods=5, freq="M")
    df = pd.DataFrame(np.random.randint(0, 100, size=(len(idx), 4)), index=idx)

    with pytest.warns(UserWarning):
        df.tz_localize("America/Los_Angeles").tz_convert("America/Los_Angeles")


def test_tz_localize():
    idx = pd.date_range("1/1/2012", periods=5, freq="M")
    df = pd.DataFrame(np.random.randint(0, 100, size=(len(idx), 4)), index=idx)

    with pytest.warns(UserWarning):
        df.tz_localize("America/Los_Angeles")


def test_unstack():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).unstack()


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


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_values(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    np.testing.assert_equal(modin_df.values, pandas_df.values)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
)
@pytest.mark.parametrize("ddof", int_arg_values, ids=arg_keys("ddof", int_arg_keys))
def test_var(request, data, axis, skipna, numeric_only, ddof):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.var(
            axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof
        )
    except Exception:
        with pytest.raises(TypeError):
            modin_df.var(axis=axis, skipna=skipna, numeric_only=numeric_only, ddof=ddof)
    else:
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


def test_xs():
    d = {
        "num_legs": [4, 4, 2, 2],
        "num_wings": [0, 0, 2, 2],
        "class": ["mammal", "mammal", "mammal", "bird"],
        "animal": ["cat", "dog", "bat", "penguin"],
        "locomotion": ["walks", "walks", "flies", "walks"],
    }
    df = pd.DataFrame(data=d)
    df = df.set_index(["class", "animal", "locomotion"])
    with pytest.warns(UserWarning):
        df.xs("mammal")


def test__doc__():
    assert pd.DataFrame.__doc__ != pandas.DataFrame.__doc__
    assert pd.DataFrame.__init__ != pandas.DataFrame.__init__
    for attr, obj in pd.DataFrame.__dict__.items():
        if (callable(obj) or isinstance(obj, property)) and attr != "__init__":
            pd_obj = getattr(pandas.DataFrame, attr, None)
            if callable(pd_obj) or isinstance(pd_obj, property):
                assert obj.__doc__ == pd_obj.__doc__


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___getitem__(request, data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    if "empty_data" not in request.node.name:
        key = modin_df.columns[0]
        modin_col = modin_df.__getitem__(key)
        assert isinstance(modin_col, (pandas.Series, SeriesView))

        pd_col = pandas_df[key]
        df_equals(pd_col, modin_col)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___getattr__(request, data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)  # noqa F841

    if "empty_data" not in request.node.name:
        key = modin_df.columns[0]
        col = modin_df.__getattr__(key)

        col = modin_df.__getattr__("col1")
        assert isinstance(col, (pandas.Series, SeriesView))

        col = getattr(modin_df, "col1")
        assert isinstance(col, (pandas.Series, SeriesView))

        # Check that lookup in column doesn't override other attributes
        df2 = modin_df.rename(index=str, columns={key: "columns"})
        assert isinstance(df2.columns, pandas.Index)


@pytest.mark.skip(reason="Defaulting to Pandas")
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___setitem__(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)  # noqa F841

    with pytest.raises(NotImplementedError):
        modin_df.__setitem__(None, None)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___len__(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    assert len(modin_df) == len(pandas_df)


def test___unicode__():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).__unicode__()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___neg__(request, data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.__neg__()
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.__neg__()
    else:
        modin_result = modin_df.__neg__()
        df_equals(modin_result, pandas_result)


def test___invert__():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        try:
            pd.DataFrame(data).__invert__()
        except TypeError:
            pass


def test___hash__():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        try:
            pd.DataFrame(data).__hash__()
        except TypeError:
            pass


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___iter__(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    modin_iterator = modin_df.__iter__()

    # Check that modin_iterator implements the iterator interface
    assert hasattr(modin_iterator, "__iter__")
    assert hasattr(modin_iterator, "next") or hasattr(modin_iterator, "__next__")

    pd_iterator = pandas_df.__iter__()
    assert list(modin_iterator) == list(pd_iterator)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___contains__(request, data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    result = False
    key = "Not Exist"
    assert result == modin_df.__contains__(key)
    assert result == (key in modin_df)

    if "empty_data" not in request.node.name:
        result = True
        key = pandas_df.columns[0]
        assert result == modin_df.__contains__(key)
        assert result == (key in modin_df)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___nonzero__(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)  # noqa F841

    with pytest.raises(ValueError):
        # Always raises ValueError
        modin_df.__nonzero__()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___abs__(request, data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = abs(pandas_df)
    except Exception as e:
        with pytest.raises(type(e)):
            abs(modin_df)
    else:
        modin_result = abs(modin_df)
        df_equals(modin_result, pandas_result)


def test___round__():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).__round__()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___array__(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    assert_array_equal(modin_df.__array__(), pandas_df.__array__())


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___bool__(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    try:
        pandas_result = pandas_df.__bool__()
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.__bool__()
    else:
        modin_result = modin_df.__bool__()
        df_equals(modin_result, pandas_result)


def test___getstate__():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).__getstate__()


def test___setstate__():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        try:
            pd.DataFrame(data).__setstate__(None)
        except TypeError:
            pass


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___delitem__(request, data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

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


def test___finalize__():
    data = test_data_values[0]
    with pytest.warns(UserWarning):
        pd.DataFrame(data).__finalize__(None)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___copy__(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    modin_df_copy, pandas_df_copy = modin_df.__copy__(), pandas_df.__copy__()
    df_equals(modin_df_copy, pandas_df_copy)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___deepcopy__(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

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


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_reset_index_with_multi_index(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    if len(modin_df.columns) > len(pandas_df.columns):
        col0 = modin_df.columns[0]
        col1 = modin_df.columns[1]
        modin_cols = modin_df.groupby([col0, col1]).count().reset_index().columns
        pandas_cols = pandas_df.groupby([col0, col1]).count().reset_index().columns

        assert modin_cols.equals(pandas_cols)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_inplace_series_ops(data):
    pandas_df = pandas.DataFrame(data)
    modin_df = pd.DataFrame(data)

    if len(modin_df.columns) > len(pandas_df.columns):
        col0 = modin_df.columns[0]
        col1 = modin_df.columns[1]
        pandas_df[col1].dropna(inplace=True)
        modin_df[col1].dropna(inplace=True)
        df_equals(modin_df, pandas_df)

        pandas_df[col0].fillna(0, inplace=True)
        modin_df[col0].fillna(0, inplace=True)
        df_equals(modin_df, pandas_df)


def test___setattr__():
    pandas_df = pandas.DataFrame([1, 2, 3])
    modin_df = pd.DataFrame([1, 2, 3])

    pandas_df.new_col = [4, 5, 6]
    modin_df.new_col = [4, 5, 6]

    df_equals(modin_df, pandas_df)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_isin(data):
    pandas_df = pandas.DataFrame(data)
    modin_df = pd.DataFrame(data)

    val = [1, 2, 3, 4]
    pandas_result = pandas_df.isin(val)
    modin_result = modin_df.isin(val)

    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_isna(data):
    pandas_df = pandas.DataFrame(data)
    modin_df = pd.DataFrame(data)

    pandas_result = pandas_df.isna()
    modin_result = modin_df.isna()

    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_isnull(data):
    pandas_df = pandas.DataFrame(data)
    modin_df = pd.DataFrame(data)

    pandas_result = pandas_df.isnull()
    modin_result = modin_df.isnull()

    df_equals(modin_result, pandas_result)
