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


def inter_df_math_helper(modin_series, pandas_series, op):
    try:
        pandas_result = getattr(pandas_series, op)(4)
    except Exception as e:
        with pytest.raises(type(e)):
            getattr(modin_series, op)(4)
    else:
        modin_result = getattr(modin_series, op)(4)
        df_equals(modin_result, pandas_result)

    try:
        pandas_result = getattr(pandas_series, op)(4.0)
    except Exception as e:
        with pytest.raises(type(e)):
            getattr(modin_series, op)(4.0)
    else:
        modin_result = getattr(modin_series, op)(4.0)
        df_equals(modin_result, pandas_result)

    # These operations don't support non-scalar `other`
    if op in ["__divmod__", "divmod", "rdivmod"]:
        return

    try:
        pandas_result = getattr(pandas_series, op)(pandas_series)
    except Exception as e:
        with pytest.raises(type(e)):
            getattr(modin_series, op)(modin_series)
    else:
        modin_result = getattr(modin_series, op)(modin_series)
        df_equals(modin_result, pandas_result)

    list_test = random_state.randint(RAND_LOW, RAND_HIGH, size=(modin_series.shape[0]))
    try:
        pandas_result = getattr(pandas_series, op)(list_test)
    except Exception as e:
        with pytest.raises(type(e)):
            getattr(modin_series, op)(list_test)
    else:
        modin_result = getattr(modin_series, op)(list_test)
        df_equals(modin_result, pandas_result)

    series_test_modin = pd.Series(list_test, index=modin_series.index)
    series_test_pandas = pandas.Series(list_test, index=pandas_series.index)
    try:
        pandas_result = getattr(pandas_series, op)(series_test_pandas)
    except Exception as e:
        with pytest.raises(type(e)):
            getattr(modin_series, op)(series_test_modin)
    else:
        modin_result = getattr(modin_series, op)(series_test_modin)
        df_equals(modin_result, pandas_result)

    # Level test
    new_idx = pandas.MultiIndex.from_tuples(
        [(i // 4, i // 2, i) for i in modin_series.index]
    )
    modin_df_multi_level = modin_series.copy()
    modin_df_multi_level.index = new_idx

    try:
        # Defaults to pandas
        with pytest.warns(UserWarning):
            # Operation against self for sanity check
            getattr(modin_df_multi_level, op)(modin_df_multi_level, level=1)
    except TypeError:
        # Some operations don't support multilevel `level` parameter
        pass


def create_test_series(dict_vals):
    modin_series = pd.Series(dict_vals[next(iter(dict_vals.keys()))])
    pandas_series = pandas.Series(dict_vals[next(iter(dict_vals.keys()))])
    return modin_series, pandas_series


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_T(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.T, pandas_series.T)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___abs__(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.__abs__(), pandas_series.__abs__())


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___add__(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "__add__")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___and__(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "__and__")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___array__(data):
    modin_series, pandas_series = create_test_series(data)
    with pytest.warns(UserWarning):
        modin_result = modin_series.__array__()
    assert_array_equal(modin_result, pandas_series.__array__())


@pytest.mark.skip(reason="Defaulting to pandas")
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___array_prepare__(data):
    modin_series, pandas_series = create_test_series(data)
    with pytest.warns(UserWarning):
        modin_result = modin_series.__array_prepare__()
    assert_array_equal(modin_result, pandas_series.__array_prepare__())


@pytest.mark.skip(reason="Defaulting to pandas")
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___array_priority__(data):
    modin_series, pandas_series = create_test_series(data)
    with pytest.warns(UserWarning):
        modin_result = modin_series.__array_priority__()
    assert_array_equal(modin_result, pandas_series.__array_priority__())


@pytest.mark.skip(reason="Defaulting to pandas")
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___array_wrap__(data):
    modin_series, pandas_series = create_test_series(data)
    with pytest.warns(UserWarning):
        modin_result = modin_series.__array_wrap__()
    assert_array_equal(modin_result, pandas_series.__array_wrap__())


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___bool__(data):
    modin_series, pandas_series = create_test_series(data)
    try:
        pandas_result = pandas_series.__bool__()
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.__bool__()
    else:
        modin_result = modin_series.__bool__()
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___contains__(request, data):
    modin_series, pandas_series = create_test_series(data)

    result = False
    key = "Not Exist"
    assert result == modin_series.__contains__(key)
    assert result == (key in modin_series)

    if "empty_data" not in request.node.name:
        result = True
        key = pandas_series.keys()[0]
        assert result == modin_series.__contains__(key)
        assert result == (key in modin_series)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_copy(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.copy(), modin_series)
    df_equals(modin_series.copy(), pandas_series.copy())
    df_equals(modin_series.copy(), pandas_series)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___deepcopy__(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.__deepcopy__(), modin_series)
    df_equals(modin_series.__deepcopy__(), pandas_series.__deepcopy__())
    df_equals(modin_series.__deepcopy__(), pandas_series)


@pytest.mark.skip(reason="Using pandas Series.")
def test___delitem__():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.__delitem__(None)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___div__(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "__div__")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_divmod(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "divmod")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_rdivmod(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "rdivmod")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___eq__(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "__eq__")


@pytest.mark.skip(reason="Come back to fix")
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___floordiv__(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "__floordiv__")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___ge__(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "__ge__")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___getitem__(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series[0], pandas_series[0])
    df_equals(
        modin_series[modin_series.index[-1]],
        pandas_series[pandas_series.index[-1]]
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___gt__(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "__gt__")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___int__(data):
    modin_series, pandas_series = create_test_series(data)
    try:
        pandas_result = int(pandas_series[0])
    except Exception as e:
        with pytest.raises(type(e)):
            int(modin_series[0])
    else:
        assert int(modin_series[0]) == pandas_result


@pytest.mark.skip(reason="Using pandas Series.")
def test___invert__():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.__invert__()


@pytest.mark.skip(reason="Using pandas Series.")
def test___iter__():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.__iter__()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___le__(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "__le__")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___len__(data):
    modin_series, pandas_series = create_test_series(data)
    assert len(modin_series) == len(pandas_series)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___long__(data):
    modin_series, pandas_series = create_test_series(data)
    try:
        pandas_result = long(pandas_series[0])
    except Exception as e:
        with pytest.raises(type(e)):
            long(modin_series[0])
    else:
        assert long(modin_series[0]) == pandas_result


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___lt__(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "__lt__")


@pytest.mark.skip(reason="Come back to fix")
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___mod__(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "__mod__")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___mul__(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "__mul__")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___ne__(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "__ne__")


@pytest.mark.skip(reason="Using pandas Series.")
def test___neg__():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.__neg__()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___or__(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "__or__")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___pow__(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "__pow__")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___repr__(data):
    modin_series, pandas_series = create_test_series(data)
    assert repr(modin_series) == repr(pandas_series)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___round__(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(round(modin_series), round(pandas_series))


@pytest.mark.skip(reason="Using pandas Series.")
def test___setitem__():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.__setitem__(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test___sizeof__():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.__sizeof__()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___str__(data):
    modin_series, pandas_series = create_test_series(data)
    assert str(modin_series) == str(pandas_series)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___sub__(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "__sub__")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___truediv__(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "__truediv__")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___xor__(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "__xor__")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_abs(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.abs(), pandas_series.abs())


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_add(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "add")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_add_prefix(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.add_prefix("PREFIX_ADD_"), pandas_series.add_prefix("PREFIX_ADD_"))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_add_suffix(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.add_suffix("SUFFIX_ADD_"), pandas_series.add_suffix("SUFFIX_ADD_"))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
def test_agg(data, func):
    modin_series, pandas_series = create_test_series(data)
    try:
        pandas_result = pandas_series.agg(func)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.agg(func)
    else:
        modin_result = modin_series.agg(func)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
def test_agg_numeric(request, data, func):
    if name_contains(request.node.name, numeric_agg_funcs) and name_contains(
        request.node.name, numeric_dfs
    ):
        axis = 0
        modin_series, pandas_series = create_test_series(data)
        try:
            pandas_result = pandas_series.agg(func, axis)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_series.agg(func, axis)
        else:
            modin_result = modin_series.agg(func, axis)
            df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
def test_aggregate(request, data, func):
    axis = 0
    modin_series, pandas_series = create_test_series(data)
    try:
        pandas_result = pandas_series.aggregate(func, axis)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.aggregate(func, axis)
    else:
        modin_result = modin_series.aggregate(func, axis)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
def test_aggregate_numeric(request, data, func):
    if name_contains(request.node.name, numeric_agg_funcs) and name_contains(
        request.node.name, numeric_dfs
    ):
        axis = 0
        modin_series, pandas_series = create_test_series(data)
        try:
            pandas_result = pandas_series.agg(func, axis)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_series.agg(func, axis)
        else:
            modin_result = modin_series.agg(func, axis)
            df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_aggregate_error_checking(data):
    modin_series, _ = create_test_series(data)

    assert modin_series.aggregate("ndim") == 1
    with pytest.warns(UserWarning):
        modin_series.aggregate("cumproduct")
    with pytest.raises(ValueError):
        modin_series.aggregate("NOT_EXISTS")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_align(data):
    modin_series, _ = create_test_series(data)
    with pytest.warns(UserWarning):
        modin_series.align(modin_series)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_all(data, skipna):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.all(skipna=skipna), pandas_series.all(skipna=skipna))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_any(data, skipna):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.any(skipna=skipna), pandas_series.any(skipna=skipna))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_append(data):
    modin_series, pandas_series = create_test_series(data)

    data_to_append = {"append_a": 2, "append_b": 1000}

    ignore_idx_values = [True, False]

    for ignore in ignore_idx_values:
        try:
            pandas_result = pandas_series.append(data_to_append, ignore_index=ignore)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_series.append(data_to_append, ignore_index=ignore)
        else:
            modin_result = modin_series.append(data_to_append, ignore_index=ignore)
            df_equals(modin_result, pandas_result)

    try:
        pandas_result = pandas_series.append(pandas_series.iloc[-1])
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.append(modin_series.iloc[-1])
    else:
        modin_result = modin_series.append(modin_series.iloc[-1])
        df_equals(modin_result, pandas_result)

    try:
        pandas_result = pandas_series.append([pandas_series.iloc[-1]])
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.append([modin_series.iloc[-1]])
    else:
        modin_result = modin_series.append([modin_series.iloc[-1]])
        df_equals(modin_result, pandas_result)

    verify_integrity_values = [True, False]

    for verify_integrity in verify_integrity_values:
        try:
            pandas_result = pandas_series.append(
                [pandas_series, pandas_series], verify_integrity=verify_integrity
            )
        except Exception as e:
            with pytest.raises(type(e)):
                modin_series.append([modin_series, modin_series], verify_integrity=verify_integrity)
        else:
            modin_result = modin_series.append(
                [modin_series, modin_series], verify_integrity=verify_integrity
            )
            df_equals(modin_result, pandas_result)

        try:
            pandas_result = pandas_series.append(
                pandas_series, verify_integrity=verify_integrity
            )
        except Exception as e:
            with pytest.raises(type(e)):
                modin_series.append(modin_series, verify_integrity=verify_integrity)
        else:
            modin_result = modin_series.append(modin_series, verify_integrity=verify_integrity)
            df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
def test_apply(request, data, func):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.apply(func)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.apply(func)
    else:
        modin_result = modin_series.apply(func)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
def test_apply_numeric(request, data, func):
    modin_series, pandas_series = create_test_series(data)

    if name_contains(request.node.name, numeric_dfs):
        try:
            pandas_result = pandas_series.apply(func)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_series.apply(func)
        else:
            modin_result = modin_series.apply(func)
            df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_argmax(data, skipna):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.argmax(skipna=skipna), pandas_series.argmax(skipna=skipna))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_argmin(data, skipna):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.argmin(skipna=skipna), pandas_series.argmin(skipna=skipna))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_argsort(data):
    modin_series, pandas_series = create_test_series(data)
    with pytest.warns(UserWarning):
        modin_result = modin_series.argsort()
    df_equals(modin_result, pandas_series.argsort())


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_as_blocks(data):
    modin_series, _ = create_test_series(data)
    with pytest.warns(UserWarning):
        modin_series.as_blocks()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_as_matrix(data):
    modin_series, _ = create_test_series(data)
    with pytest.warns(UserWarning):
        modin_series.as_matrix()


def test_asfreq():
    index = pd.date_range("1/1/2000", periods=4, freq="T")
    series = pd.Series([0.0, None, 2.0, 3.0], index=index)
    with pytest.warns(UserWarning):
        # We are only testing that this defaults to pandas, so we will just check for
        # the warning
        series.asfreq(freq="30S")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_asobject(data):
    modin_series, _ = create_test_series(data)

    with pytest.warns(UserWarning):
        _ = modin_series.asobject


def test_asof():
    series = pd.Series(
        [10, 20, 30, 40, 50],
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
        series.asof(pd.DatetimeIndex(["2018-02-27 09:03:30", "2018-02-27 09:04:30"]))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_astype(data):
    modin_series, pandas_series = create_test_series(data)
    try:
        pandas_result = pandas_series.astype(str)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.astype(str)
    else:
        df_equals(modin_series.astype(str), pandas_result)

    try:
        pandas_result = pandas_series.astype(np.int64)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.astype(np.int64)
    else:
        df_equals(modin_series.astype(np.int64), pandas_result)

    try:
        pandas_result = pandas_series.astype(np.float64)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.astype(np.float64)
    else:
        df_equals(modin_series.astype(np.float64), pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_at(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.at[modin_series.index[0]], pandas_series.at[pandas_series.index[0]])
    df_equals(modin_series.at[modin_series.index[-1]], pandas_series[pandas_series.index[-1]])


def test_at_time():
    i = pd.date_range("2018-04-09", periods=4, freq="12H")
    ts = pd.Series([1, 2, 3, 4], index=i)
    with pytest.warns(UserWarning):
        ts.at_time("12:00")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_autocorr(data):
    modin_series, _ = create_test_series(data)

    with pytest.warns(UserWarning):
        modin_series.autocorr()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_axes(data):
    modin_series, pandas_series = create_test_series(data)
    assert modin_series.axes[0].equals(pandas_series.axes[0])
    assert len(modin_series.axes) == len(pandas_series.axes)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_base(data):
    modin_series, _ = create_test_series(data)

    with pytest.warns(UserWarning):
        modin_series.base


@pytest.mark.skip(reason="Using pandas Series.")
def test_between():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.between(None, None)


def test_between_time():
    i = pd.date_range("2018-04-09", periods=4, freq="12H")
    ts = pd.Series([1, 2, 3, 4], index=i)
    with pytest.warns(UserWarning):
        ts.between_time("0:15", "0:45")


@pytest.mark.skip(reason="Using pandas Series.")
def test_bfill():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.bfill(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_blocks():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.blocks


@pytest.mark.skip(reason="Using pandas Series.")
def test_bool():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.bool()


@pytest.mark.skip(reason="Using pandas Series.")
def test_clip():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.clip(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_clip_lower():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.clip_lower(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_clip_upper():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.clip_upper(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_combine():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.combine(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_combine_first():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.combine_first(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_compound():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.compound(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_compress():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.compress(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_consolidate():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.consolidate(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_convert_objects():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.convert_objects(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_copy():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.copy(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_corr():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.corr(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_count():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.count(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_cov():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.cov(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_cummax():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.cummax(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_cummin():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.cummin(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_cumprod():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.cumprod(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_cumsum():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.cumsum(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_data():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.data


@pytest.mark.skip(reason="Using pandas Series.")
def test_describe():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.describe(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_diff():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.diff(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_div():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.div(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_divide():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.divide(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_dot():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.dot(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_drop():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.drop(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_drop_duplicates():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.drop_duplicates(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_dropna():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.dropna(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_dtype():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.dtype


@pytest.mark.skip(reason="Using pandas Series.")
def test_dtypes():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.dtypes


@pytest.mark.skip(reason="Using pandas Series.")
def test_duplicated():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.duplicated(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_empty():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.empty


@pytest.mark.skip(reason="Using pandas Series.")
def test_eq():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.eq(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_equals():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.equals(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_ewm():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.ewm(None, None, None, None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_expanding():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.expanding(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_factorize():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.factorize(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_ffill():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.ffill(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_fillna():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.fillna(None, None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_filter():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.filter(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_first():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.first(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_first_valid_index():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.first_valid_index()


@pytest.mark.skip(reason="Using pandas Series.")
def test_flags():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.flags


@pytest.mark.skip(reason="Using pandas Series.")
def test_floordiv():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.floordiv(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_from_array():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.from_array(None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_from_csv():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.from_csv(None, None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_ftype():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.ftype


@pytest.mark.skip(reason="Using pandas Series.")
def test_ftypes():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.ftypes


@pytest.mark.skip(reason="Using pandas Series.")
def test_ge():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.ge(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_get():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.get(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_get_dtype_counts():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.get_dtype_counts()


@pytest.mark.skip(reason="Using pandas Series.")
def test_get_ftype_counts():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.get_ftype_counts()


@pytest.mark.skip(reason="Using pandas Series.")
def test_get_value():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.get_value(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_get_values():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.get_values()


@pytest.mark.skip(reason="Using pandas Series.")
def test_groupby():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.groupby(None, None, None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_gt():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.gt(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_hasnans():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.hasnans


@pytest.mark.skip(reason="Using pandas Series.")
def test_head():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.head(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_hist():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.hist(None, None, None, None, None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_iat():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.iat(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_idxmax():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.idxmax(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_idxmin():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.idxmin(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_iloc():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.iloc(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_imag():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.imag


@pytest.mark.skip(reason="Using pandas Series.")
def test_index():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.index


@pytest.mark.skip(reason="Using pandas Series.")
def test_interpolate():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.interpolate(None, None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_is_copy():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.is_copy


@pytest.mark.skip(reason="Using pandas Series.")
def test_is_monotonic():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.is_monotonic


@pytest.mark.skip(reason="Using pandas Series.")
def test_is_monotonic_decreasing():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.is_monotonic_decreasing


@pytest.mark.skip(reason="Using pandas Series.")
def test_is_monotonic_increasing():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.is_monotonic_increasing


@pytest.mark.skip(reason="Using pandas Series.")
def test_is_unique():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.is_unique


@pytest.mark.skip(reason="Using pandas Series.")
def test_isin():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.isin(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_isnull():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.isnull()


@pytest.mark.skip(reason="Using pandas Series.")
def test_item():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.item()


@pytest.mark.skip(reason="Using pandas Series.")
def test_items():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.items()


@pytest.mark.skip(reason="Using pandas Series.")
def test_itemsize():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.itemsize


@pytest.mark.skip(reason="Using pandas Series.")
def test_iteritems():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.iteritems()


@pytest.mark.skip(reason="Using pandas Series.")
def test_ix():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.ix(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_keys():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.keys()


@pytest.mark.skip(reason="Using pandas Series.")
def test_kurt():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.kurt(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_kurtosis():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.kurtosis(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_last():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.last(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_last_valid_index():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.last_valid_index()


@pytest.mark.skip(reason="Using pandas Series.")
def test_le():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.le(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_loc():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.loc(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_lt():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.lt(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_mad():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.mad(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_map():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.map(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_mask():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.mask(None, None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_max():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.max(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_mean():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.mean(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_median():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.median(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_memory_usage():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.memory_usage(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_min():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.min(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_mod():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.mod(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_mode():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.mode()


@pytest.mark.skip(reason="Using pandas Series.")
def test_mul():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.mul(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_multiply():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.multiply(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_name():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.name


@pytest.mark.skip(reason="Using pandas Series.")
def test_nbytes():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.nbytes


@pytest.mark.skip(reason="Using pandas Series.")
def test_ndim():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.ndim


@pytest.mark.skip(reason="Using pandas Series.")
def test_ne():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.ne(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_nlargest():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.nlargest(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_nonzero():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.nonzero()


@pytest.mark.skip(reason="Using pandas Series.")
def test_notnull():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.notnull()


@pytest.mark.skip(reason="Using pandas Series.")
def test_nsmallest():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.nsmallest(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_nunique():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.nunique(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_pct_change():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.pct_change(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_pipe():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.pipe(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_plot():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.plot(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@pytest.mark.skip(reason="Using pandas Series.")
def test_pop():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.pop(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_pow():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.pow(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_prod():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.prod(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_product():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.product(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_ptp():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.ptp(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_put():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.put(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_quantile():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.quantile(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_radd():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.radd(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_rank():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.rank(None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_ravel():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.ravel(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_rdiv():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.rdiv(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_real():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.real


@pytest.mark.skip(reason="Using pandas Series.")
def test_reindex():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.reindex(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_reindex_axis():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.reindex_axis(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_reindex_like():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.reindex_like(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_rename():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.rename(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_rename_axis():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.rename_axis(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_reorder_levels():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.reorder_levels(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_repeat():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.repeat(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_replace():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.replace(None, None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_resample():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.resample(
            None, None, None, None, None, None, None, None, None, None, None, None
        )


@pytest.mark.skip(reason="Using pandas Series.")
def test_reset_index():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.reset_index(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_reshape():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.reshape(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_rfloordiv():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.rfloordiv(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_rmod():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.rmod(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_rmul():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.rmul(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_rolling():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.rolling(None, None, None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_round():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.round(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_rpow():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.rpow(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_rsub():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.rsub(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_rtruediv():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.rtruediv(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_sample():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.sample(None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_searchsorted():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.searchsorted(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_select():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.select(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_sem():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.sem(None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_set_axis():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.set_axis(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_set_value():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.set_value(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_shape():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.shape


@pytest.mark.skip(reason="Using pandas Series.")
def test_shift():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.shift(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_size():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.size


@pytest.mark.skip(reason="Using pandas Series.")
def test_skew():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.skew(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_slice_shift():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.slice_shift(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_sort_index():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.sort_index(None, None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_sort_values():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.sort_values(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_sortlevel():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.sortlevel(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_squeeze():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.squeeze(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_std():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.std(None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_strides():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.strides


@pytest.mark.skip(reason="Using pandas Series.")
def test_sub():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.sub(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_subtract():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.subtract(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_sum():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.sum(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_swapaxes():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.swapaxes(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_swaplevel():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.swaplevel(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_tail():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.tail(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_take():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.take(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_to_clipboard():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.to_clipboard(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_to_csv():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.to_csv(None, None, None, None, None, None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_to_dense():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.to_dense()


@pytest.mark.skip(reason="Using pandas Series.")
def test_to_dict():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.to_dict()


@pytest.mark.skip(reason="Using pandas Series.")
def test_to_excel():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.to_excel(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@pytest.mark.skip(reason="Using pandas Series.")
def test_to_frame():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.to_frame(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_to_hdf():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.to_hdf(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_to_json():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.to_json(None, None, None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_to_latex():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.to_latex(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


@pytest.mark.skip(reason="Using pandas Series.")
def test_to_msgpack():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.to_msgpack(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_to_period():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.to_period(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_to_pickle():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.to_pickle(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_to_sparse():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.to_sparse(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_to_sql():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.to_sql(None, None, None, None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_to_string():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.to_string(None, None, None, None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_to_timestamp():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.to_timestamp(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_to_xarray():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.to_xarray()


@pytest.mark.skip(reason="Using pandas Series.")
def test_tolist():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.tolist()


@pytest.mark.skip(reason="Using pandas Series.")
def test_transform():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.transform(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_transpose():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.transpose(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_truediv():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.truediv(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_truncate():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.truncate(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_tshift():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.tshift(None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_tz_convert():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.tz_convert(None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_tz_localize():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.tz_localize(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_unique():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.unique()


@pytest.mark.skip(reason="Using pandas Series.")
def test_unstack():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.unstack(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_update():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.update(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_valid():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.valid(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_value_counts():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.value_counts(None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_values():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.values


@pytest.mark.skip(reason="Using pandas Series.")
def test_var():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.var(None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_view():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.view(None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_where():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.where(None, None, None, None, None, None)


@pytest.mark.skip(reason="Using pandas Series.")
def test_xs():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.xs(None, None, None)
