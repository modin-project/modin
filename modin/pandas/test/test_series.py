from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import numpy as np
import pandas
import matplotlib
import modin.pandas as pd
from numpy.testing import assert_array_equal
import sys

from modin.pandas.utils import to_pandas
from .utils import (
    random_state,
    RAND_LOW,
    RAND_HIGH,
    df_equals,
    arg_keys,
    name_contains,
    test_data_values,
    test_data_keys,
    numeric_dfs,
    no_numeric_dfs,
    agg_func_keys,
    agg_func_values,
    numeric_agg_funcs,
    quantiles_keys,
    quantiles_values,
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
            repr(getattr(modin_series, op)(4))  # repr to force materialization
    else:
        modin_result = getattr(modin_series, op)(4)
        df_equals(modin_result, pandas_result)

    try:
        pandas_result = getattr(pandas_series, op)(4.0)
    except Exception as e:
        with pytest.raises(type(e)):
            repr(getattr(modin_series, op)(4.0))  # repr to force materialization
    else:
        modin_result = getattr(modin_series, op)(4.0)
        df_equals(modin_result, pandas_result)

    # These operations don't support non-scalar `other` or have a strange behavior in
    # the testing environment
    if op in [
        "__divmod__",
        "divmod",
        "rdivmod",
        "floordiv",
        "__floordiv__",
        "rfloordiv",
        "__rfloordiv__",
        "mod",
        "__mod__",
        "rmod",
        "__rmod__",
    ]:
        return

    try:
        pandas_result = getattr(pandas_series, op)(pandas_series)
    except Exception as e:
        with pytest.raises(type(e)):
            repr(
                getattr(modin_series, op)(modin_series)
            )  # repr to force materialization
    else:
        modin_result = getattr(modin_series, op)(modin_series)
        df_equals(modin_result, pandas_result)

    list_test = random_state.randint(RAND_LOW, RAND_HIGH, size=(modin_series.shape[0]))
    try:
        pandas_result = getattr(pandas_series, op)(list_test)
    except Exception as e:
        with pytest.raises(type(e)):
            repr(getattr(modin_series, op)(list_test))  # repr to force materialization
    else:
        modin_result = getattr(modin_series, op)(list_test)
        df_equals(modin_result, pandas_result)

    series_test_modin = pd.Series(list_test, index=modin_series.index)
    series_test_pandas = pandas.Series(list_test, index=pandas_series.index)
    try:
        pandas_result = getattr(pandas_series, op)(series_test_pandas)
    except Exception as e:
        with pytest.raises(type(e)):
            repr(
                getattr(modin_series, op)(series_test_modin)
            )  # repr to force materialization
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
def test___copy__(data):
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


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___delitem__(data):
    modin_series, pandas_series = create_test_series(data)
    del modin_series[modin_series.index[0]]
    del pandas_series[pandas_series.index[0]]
    df_equals(modin_series, pandas_series)

    del modin_series[modin_series.index[-1]]
    del pandas_series[pandas_series.index[-1]]
    df_equals(modin_series, pandas_series)

    del modin_series[modin_series.index[0]]
    del pandas_series[pandas_series.index[0]]
    df_equals(modin_series, pandas_series)


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
        modin_series[modin_series.index[-1]], pandas_series[pandas_series.index[-1]]
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


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___invert__(data):
    modin_series, pandas_series = create_test_series(data)
    try:
        pandas_result = pandas_series.__invert__()
    except Exception as e:
        with pytest.raises(type(e)):
            repr(modin_series.__invert__())
    else:
        df_equals(modin_series.__invert__(), pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___iter__(data):
    modin_series, pandas_series = create_test_series(data)
    for m, p in zip(modin_series.__iter__(), pandas_series.__iter__()):
        np.testing.assert_equal(m, p)


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
        pandas_result = pandas_series[0].__long__()
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series[0].__long__()
    else:
        assert modin_series[0].__long__() == pandas_result


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___lt__(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "__lt__")


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


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___neg__(request, data):
    modin_series, pandas_series = create_test_series(data)
    try:
        pandas_result = pandas_series.__neg__()
    except Exception as e:
        with pytest.raises(type(e)):
            repr(modin_series.__neg__())
    else:
        df_equals(modin_series.__neg__(), pandas_result)


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
    if not PY2:
        df_equals(round(modin_series), round(pandas_series))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___setitem__(data):
    modin_series, pandas_series = create_test_series(data)
    for key in modin_series.keys():
        modin_series[key] = 0
        pandas_series[key] = 0
        df_equals(modin_series, pandas_series)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___sizeof__(data):
    modin_series, pandas_series = create_test_series(data)
    with pytest.warns(UserWarning):
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
    df_equals(
        modin_series.add_prefix("PREFIX_ADD_"), pandas_series.add_prefix("PREFIX_ADD_")
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_add_suffix(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(
        modin_series.add_suffix("SUFFIX_ADD_"), pandas_series.add_suffix("SUFFIX_ADD_")
    )


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
    modin_series, _ = create_test_series(data)  # noqa: F841

    assert modin_series.aggregate("ndim") == 1
    with pytest.warns(UserWarning):
        modin_series.aggregate("cumproduct")
    with pytest.raises(ValueError):
        modin_series.aggregate("NOT_EXISTS")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_align(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
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
                modin_series.append(
                    [modin_series, modin_series], verify_integrity=verify_integrity
                )
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
            modin_result = modin_series.append(
                modin_series, verify_integrity=verify_integrity
            )
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
    modin_series, _ = create_test_series(data)  # noqa: F841
    with pytest.warns(UserWarning):
        modin_series.as_blocks()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_as_matrix(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
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
    modin_series, _ = create_test_series(data)  # noqa: F841
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
            repr(modin_series.astype(str))  # repr to force materialization
    else:
        df_equals(modin_series.astype(str), pandas_result)

    try:
        pandas_result = pandas_series.astype(np.int64)
    except Exception as e:
        with pytest.raises(type(e)):
            repr(modin_series.astype(np.int64))  # repr to force materialization
    else:
        df_equals(modin_series.astype(np.int64), pandas_result)

    try:
        pandas_result = pandas_series.astype(np.float64)
    except Exception as e:
        with pytest.raises(type(e)):
            repr(modin_series.astype(np.float64))  # repr to force materialization
    else:
        df_equals(modin_series.astype(np.float64), pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_at(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(
        modin_series.at[modin_series.index[0]], pandas_series.at[pandas_series.index[0]]
    )
    df_equals(
        modin_series.at[modin_series.index[-1]], pandas_series[pandas_series.index[-1]]
    )


def test_at_time():
    i = pd.date_range("2018-04-09", periods=4, freq="12H")
    ts = pd.Series([1, 2, 3, 4], index=i)
    with pytest.warns(UserWarning):
        ts.at_time("12:00")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_autocorr(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with pytest.warns(UserWarning):
        modin_series.autocorr()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_axes(data):
    modin_series, pandas_series = create_test_series(data)
    assert modin_series.axes[0].equals(pandas_series.axes[0])
    assert len(modin_series.axes) == len(pandas_series.axes)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_base(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with pytest.warns(UserWarning):
        _ = modin_series.base


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


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_bfill(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.bfill(), pandas_series.bfill())
    # inplace
    modin_series_cp = modin_series.copy()
    pandas_series_cp = pandas_series.copy()
    modin_series_cp.bfill(inplace=True)
    pandas_series_cp.bfill(inplace=True)
    df_equals(modin_series_cp, pandas_series_cp)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_blocks(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with pytest.warns(UserWarning):
        _ = modin_series.blocks


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_bool(data):
    modin_series, pandas_series = create_test_series(data)

    with pytest.raises(ValueError):
        modin_series.bool()
    with pytest.raises(ValueError):
        modin_series.__bool__()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_clip(request, data):
    modin_series, pandas_series = create_test_series(data)

    if name_contains(request.node.name, numeric_dfs):
        # set bounds
        lower, upper = np.sort(random_state.random_integers(RAND_LOW, RAND_HIGH, 2))

        # test only upper scalar bound
        modin_result = modin_series.clip(None, upper)
        pandas_result = pandas_series.clip(None, upper)
        df_equals(modin_result, pandas_result)

        # test lower and upper scalar bound
        modin_result = modin_series.clip(lower, upper)
        pandas_result = pandas_series.clip(lower, upper)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_clip_lower(request, data):
    modin_series, pandas_series = create_test_series(data)

    if name_contains(request.node.name, numeric_dfs):
        # set bounds
        lower = random_state.random_integers(RAND_LOW, RAND_HIGH, 1)[0]

        # test lower scalar bound
        pandas_result = pandas_series.clip_lower(lower)
        modin_result = modin_series.clip_lower(lower)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_clip_upper(request, data):
    modin_series, pandas_series = create_test_series(data)

    if name_contains(request.node.name, numeric_dfs):
        # set bounds
        upper = random_state.random_integers(RAND_LOW, RAND_HIGH, 1)[0]

        # test upper scalar bound
        modin_result = modin_series.clip_upper(upper)
        pandas_result = pandas_series.clip_upper(upper)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_combine(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    modin_series2 = modin_series % (max(modin_series) // 2)
    with pytest.warns(UserWarning):
        modin_series.combine(modin_series2, lambda s1, s2: s1 if s1 < s2 else s2)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_combine_first(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    modin_series2 = modin_series % (max(modin_series) // 2)
    with pytest.warns(UserWarning):
        modin_series.combine_first(modin_series2)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_compound(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with pytest.warns(UserWarning):
        modin_series.compound()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_compress(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with pytest.warns(UserWarning):
        modin_series.compress(modin_series > 30)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_convert_objects(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with pytest.warns(UserWarning):
        modin_series.convert_objects()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_copy(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series, modin_series.copy())
    df_equals(modin_series.copy(), pandas_series)
    df_equals(modin_series.copy(), pandas_series.copy())


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_corr(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with pytest.warns(UserWarning):
        modin_series.corr(modin_series)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_count(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.count(), pandas_series.count())


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_cov(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with pytest.warns(UserWarning):
        modin_series.cov(modin_series)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_cummax(data, skipna):
    modin_series, pandas_series = create_test_series(data)
    try:
        pandas_result = pandas_series.cummax(skipna=skipna)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.cummax(skipna=skipna)
    else:
        df_equals(modin_series.cummax(skipna=skipna), pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_cummin(data, skipna):
    modin_series, pandas_series = create_test_series(data)
    try:
        pandas_result = pandas_series.cummin(skipna=skipna)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.cummin(skipna=skipna)
    else:
        df_equals(modin_series.cummin(skipna=skipna), pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_cumprod(data, skipna):
    modin_series, pandas_series = create_test_series(data)
    try:
        pandas_result = pandas_series.cumprod(skipna=skipna)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.cumprod(skipna=skipna)
    else:
        df_equals(modin_series.cumprod(skipna=skipna), pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_cumsum(data, skipna):
    modin_series, pandas_series = create_test_series(data)
    try:
        pandas_result = pandas_series.cumsum(skipna=skipna)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.cumsum(skipna=skipna)
    else:
        df_equals(modin_series.cumsum(skipna=skipna), pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_data(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with pytest.warns(UserWarning):
        _ = modin_series.data


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_describe(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.describe(), pandas_series.describe())
    percentiles = [0.10, 0.11, 0.44, 0.78, 0.99]
    df_equals(
        modin_series.describe(percentiles=percentiles),
        pandas_series.describe(percentiles=percentiles),
    )

    try:
        pandas_result = pandas_series.describe(exclude=[np.float64])
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.describe(exclude=[np.float64])
    else:
        modin_result = modin_series.describe(exclude=[np.float64])
        df_equals(modin_result, pandas_result)

    try:
        pandas_result = pandas_series.describe(exclude=np.float64)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.describe(exclude=np.float64)
    else:
        modin_result = modin_series.describe(exclude=np.float64)
        df_equals(modin_result, pandas_result)

    try:
        pandas_result = pandas_series.describe(
            include=[np.timedelta64, np.datetime64, np.object, np.bool]
        )
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.describe(
                include=[np.timedelta64, np.datetime64, np.object, np.bool]
            )
    else:
        modin_result = modin_series.describe(
            include=[np.timedelta64, np.datetime64, np.object, np.bool]
        )
        df_equals(modin_result, pandas_result)

    modin_result = modin_series.describe(include=str(modin_series.dtypes))
    pandas_result = pandas_series.describe(include=str(pandas_series.dtypes))
    df_equals(modin_result, pandas_result)

    modin_result = modin_series.describe(include=[np.number])
    pandas_result = pandas_series.describe(include=[np.number])
    df_equals(modin_result, pandas_result)

    df_equals(
        modin_series.describe(include="all"), pandas_series.describe(include="all")
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize(
    "periods", int_arg_values, ids=arg_keys("periods", int_arg_keys)
)
def test_diff(data, periods):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.diff(periods=periods)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.diff(periods=periods)
    else:
        modin_result = modin_series.diff(periods=periods)
        df_equals(modin_result, pandas_result)

    try:
        pandas_result = pandas_series.T.diff(periods=periods)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.T.diff(periods=periods)
    else:
        modin_result = modin_series.T.diff(periods=periods)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_div(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "div")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_divide(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "divide")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_dot(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with pytest.warns(UserWarning):
        modin_series.dot(modin_series)


@pytest.mark.skip(reason="Using pandas Series.")
def test_drop():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.drop(None, None, None, None)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_drop_duplicates(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(
        modin_series.drop_duplicates(keep="first", inplace=False),
        pandas_series.drop_duplicates(keep="first", inplace=False),
    )
    df_equals(
        modin_series.drop_duplicates(keep="last", inplace=False),
        pandas_series.drop_duplicates(keep="last", inplace=False),
    )
    df_equals(
        modin_series.drop_duplicates(keep=False, inplace=False),
        pandas_series.drop_duplicates(keep=False, inplace=False),
    )
    df_equals(
        modin_series.drop_duplicates(inplace=False),
        pandas_series.drop_duplicates(inplace=False),
    )
    modin_series.drop_duplicates(inplace=True)
    df_equals(modin_series, pandas_series.drop_duplicates(inplace=False))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("how", ["any", "all"], ids=["any", "all"])
def test_dropna(data, how):
    modin_series, pandas_series = create_test_series(data)

    with pytest.raises(TypeError):
        modin_series.dropna(how=None, thresh=None)

    modin_result = modin_series.dropna(how=how)
    pandas_result = pandas_series.dropna(how=how)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_dropna_inplace(data):
    modin_series, pandas_series = create_test_series(data)
    pandas_result = pandas_series.dropna()
    modin_series.dropna(inplace=True)
    df_equals(modin_series, pandas_result)

    modin_series, pandas_series = create_test_series(data)
    with pytest.raises(TypeError):
        modin_series.dropna(thresh=2, inplace=True)

    modin_series, pandas_series = create_test_series(data)
    pandas_series.dropna(how="any", inplace=True)
    modin_series.dropna(how="any", inplace=True)
    df_equals(modin_series, pandas_series)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_dtype(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.dtype, modin_series.dtypes)
    df_equals(modin_series.dtype, pandas_series.dtype)
    df_equals(modin_series.dtype, pandas_series.dtypes)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("keep", ["last", "first"], ids=["last", "first"])
def test_duplicated(data, keep):
    modin_series, pandas_series = create_test_series(data)
    with pytest.warns(UserWarning):
        modin_result = modin_series.duplicated(keep=keep)
    df_equals(modin_result, pandas_series.duplicated(keep=keep))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_empty(data):
    modin_series, pandas_series = create_test_series(data)
    assert modin_series.empty == pandas_series.empty


def test_empty_series():
    modin_series = pd.Series()
    assert modin_series.empty


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_eq(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "eq")


def test_equals():
    series_data = [2.9, 3, 3, 3]
    modin_df1 = pd.Series(series_data)
    modin_df2 = pd.Series(series_data)

    assert modin_df1.equals(modin_df2)
    assert modin_df1.equals(pd.Series(modin_df1))
    df_equals(modin_df1, modin_df2)
    df_equals(modin_df1, pd.Series(modin_df1))

    series_data = [2, 3, 5, 1]
    modin_df3 = pd.Series(series_data, index=list("abcd"))

    assert not modin_df1.equals(modin_df3)

    with pytest.raises(AssertionError):
        df_equals(modin_df3, modin_df1)

    with pytest.raises(AssertionError):
        df_equals(modin_df3, modin_df2)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_ewm(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with pytest.warns(UserWarning):
        modin_series.ewm(halflife=6)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_expanding(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with pytest.warns(UserWarning):
        modin_series.expanding()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_factorize(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with pytest.warns(UserWarning):
        modin_series.factorize()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_ffill(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.ffill(), pandas_series.ffill())
    # inplace
    modin_series_cp = modin_series.copy()
    pandas_series_cp = pandas_series.copy()
    modin_series_cp.ffill(inplace=True)
    pandas_series_cp.ffill(inplace=True)
    df_equals(modin_series_cp, pandas_series_cp)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_fillna(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.fillna(0), pandas_series.fillna(0))
    df_equals(modin_series.fillna(method="bfill"), pandas_series.fillna(method="bfill"))
    df_equals(modin_series.fillna(method="ffill"), pandas_series.fillna(method="ffill"))
    df_equals(modin_series.fillna(0, limit=1), pandas_series.fillna(0, limit=1))


@pytest.mark.skip(reason="Using pandas Series.")
def test_filter():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.filter(None, None, None)


def test_first():
    i = pd.date_range("2018-04-09", periods=4, freq="2D")
    ts = pd.Series([1, 2, 3, 4], index=i)
    with pytest.warns(UserWarning):
        ts.first("3D")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_first_valid_index(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.first_valid_index(), pandas_series.first_valid_index())


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_flags(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with pytest.warns(UserWarning):
        _ = modin_series.flags


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_floordiv(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "floordiv")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_ftype(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.ftype, modin_series.ftypes)
    df_equals(modin_series.ftype, pandas_series.ftype)
    df_equals(modin_series.ftype, pandas_series.ftypes)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_ge(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "ge")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_get(data):
    modin_series, pandas_series = create_test_series(data)
    for key in modin_series.keys():
        df_equals(modin_series.get(key), pandas_series.get(key))
    df_equals(
        modin_series.get("NO_EXIST", "DEFAULT"),
        pandas_series.get("NO_EXIST", "DEFAULT"),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_get_dtype_counts(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.get_dtype_counts(), pandas_series.get_dtype_counts())


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_get_ftype_counts(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.get_ftype_counts(), pandas_series.get_ftype_counts())


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_get_value(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with pytest.warns(UserWarning):
        modin_series.get_value(0)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_get_values(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with pytest.warns(UserWarning):
        modin_series.get_values()


@pytest.mark.skip(reason="Using pandas Series.")
def test_groupby():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.groupby(None, None, None, None, None, None, None)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_gt(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "gt")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_hasnans(data):
    modin_series, pandas_series = create_test_series(data)
    assert modin_series.hasnans == pandas_series.hasnans


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("n", int_arg_values, ids=arg_keys("n", int_arg_keys))
def test_head(data, n):
    modin_series, pandas_series = create_test_series(data)

    df_equals(modin_series.head(n), pandas_series.head(n))
    df_equals(
        modin_series.head(len(modin_series)), pandas_series.head(len(pandas_series))
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_hist(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with pytest.warns(UserWarning):
        modin_series.hist(None)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_iat(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.iat[0], pandas_series.iat[0])


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_idxmax(data, skipna):
    modin_series, pandas_series = create_test_series(data)
    pandas_result = pandas_series.idxmax(skipna=skipna)
    modin_result = modin_series.idxmax(skipna=skipna)
    df_equals(modin_result, pandas_result)

    pandas_result = pandas_series.T.idxmax(skipna=skipna)
    modin_result = modin_series.T.idxmax(skipna=skipna)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_idxmin(data, skipna):
    modin_series, pandas_series = create_test_series(data)
    pandas_result = pandas_series.idxmin(skipna=skipna)
    modin_result = modin_series.idxmin(skipna=skipna)
    df_equals(modin_result, pandas_result)

    pandas_result = pandas_series.T.idxmin(skipna=skipna)
    modin_result = modin_series.T.idxmin(skipna=skipna)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_iloc(request, data):
    modin_series, pandas_series = create_test_series(data)

    if not name_contains(request.node.name, ["empty_data"]):
        # Scaler
        np.testing.assert_equal(modin_series.iloc[0], pandas_series.iloc[0])

        # Series
        df_equals(modin_series.iloc[1:], pandas_series.iloc[1:])
        df_equals(modin_series.iloc[1:2], pandas_series.iloc[1:2])
        df_equals(modin_series.iloc[[1, 2]], pandas_series.iloc[[1, 2]])

        # Write Item
        modin_series.iloc[[1, 2]] = 42
        pandas_series.iloc[[1, 2]] = 42
        df_equals(modin_series, pandas_series)

        with pytest.raises(IndexError):
            modin_series.iloc[1:, 1]
    else:
        with pytest.raises(IndexError):
            modin_series.iloc[0]


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_imag(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with pytest.warns(UserWarning):
        modin_series.imag


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_index(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.index, pandas_series.index)
    with pytest.raises(ValueError):
        modin_series.index = list(modin_series.index) + [999]

    modin_series.index = modin_series.index.map(str)
    pandas_series.index = pandas_series.index.map(str)
    df_equals(modin_series.index, pandas_series.index)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_interpolate(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with pytest.warns(UserWarning):
        modin_series.interpolate()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_is_copy(data):
    modin_series, pandas_series = create_test_series(data)
    with pytest.warns(FutureWarning):
        assert modin_series.is_copy is pandas_series.is_copy


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_is_monotonic(data):
    modin_series, pandas_series = create_test_series(data)
    with pytest.warns(UserWarning):
        assert modin_series.is_monotonic == pandas_series.is_monotonic


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_is_monotonic_decreasing(data):
    modin_series, pandas_series = create_test_series(data)
    with pytest.warns(UserWarning):
        assert (
            modin_series.is_monotonic_decreasing
            == pandas_series.is_monotonic_decreasing
        )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_is_monotonic_increasing(data):
    modin_series, pandas_series = create_test_series(data)
    with pytest.warns(UserWarning):
        assert (
            modin_series.is_monotonic_increasing
            == pandas_series.is_monotonic_increasing
        )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_is_unique(data):
    modin_series, pandas_series = create_test_series(data)
    with pytest.warns(UserWarning):
        assert modin_series.is_unique == pandas_series.is_unique


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_isin(data):
    modin_series, pandas_series = create_test_series(data)
    val = [1, 2, 3, 4]
    pandas_result = pandas_series.isin(val)
    modin_result = modin_series.isin(val)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_isnull(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.isnull(), pandas_series.isnull())


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_items(data):
    modin_series, pandas_series = create_test_series(data)

    modin_items = modin_series.items()
    pandas_items = pandas_series.items()
    for modin_item, pandas_item in zip(modin_items, pandas_items):
        modin_index, modin_scalar = modin_item
        pandas_index, pandas_scalar = pandas_item
        df_equals(modin_scalar, pandas_scalar)
        assert pandas_index == modin_index


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_itemsize(data):
    modin_series, pandas_series = create_test_series(data)
    with pytest.warns(UserWarning):
        assert modin_series.itemsize == pandas_series.itemsize


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_iteritems(data):
    modin_series, pandas_series = create_test_series(data)

    modin_items = modin_series.iteritems()
    pandas_items = pandas_series.iteritems()
    for modin_item, pandas_item in zip(modin_items, pandas_items):
        modin_index, modin_scalar = modin_item
        pandas_index, pandas_scalar = pandas_item
        df_equals(modin_scalar, pandas_scalar)
        assert pandas_index == modin_index


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_ix(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with pytest.raises(NotImplementedError):
        modin_series.ix[0]


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_keys(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.keys(), pandas_series.keys())


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_kurt(data):
    modin_series, pandas_series = create_test_series(data)
    with pytest.warns(UserWarning):
        modin_series.kurt()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_kurtosis(data):
    modin_series, pandas_series = create_test_series(data)
    with pytest.warns(UserWarning):
        modin_series.kurtosis()


def test_last():
    i = pd.date_range("2018-04-09", periods=4, freq="2D")
    ts = pd.Series([1, 2, 3, 4], index=i)
    with pytest.warns(UserWarning):
        ts.last("3D")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_last_valid_index(data):
    modin_series, pandas_series = create_test_series(data)
    assert modin_series.last_valid_index() == (pandas_series.last_valid_index())


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_le(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "le")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_loc(data):
    modin_series, pandas_series = create_test_series(data)
    for v in modin_series.index:
        df_equals(modin_series.loc[v], pandas_series.loc[v])
        df_equals(modin_series.loc[v:], pandas_series.loc[v:])


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_lt(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "lt")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_mad(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with pytest.warns(UserWarning):
        modin_series.mad()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_map(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.map(str), pandas_series.map(str))


def test_mask():
    modin_series = pd.Series(np.arange(10))
    m = modin_series % 3 == 0
    with pytest.warns(UserWarning):
        try:
            modin_series.mask(~m, -modin_series)
        except ValueError:
            pass


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_max(data, skipna):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.max(skipna=skipna), pandas_series.max(skipna=skipna))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_mean(data, skipna):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.mean(skipna=skipna), pandas_series.mean(skipna=skipna))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_median(data, skipna):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.median(skipna=skipna), pandas_series.median(skipna=skipna))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("index", [True, False], ids=["True", "False"])
def test_memory_usage(data, index):
    modin_series, pandas_series = create_test_series(data)
    df_equals(
        modin_series.memory_usage(index=index), pandas_series.memory_usage(index=index)
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_min(data, skipna):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.min(skipna=skipna), pandas_series.min(skipna=skipna))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_mod(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "mod")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_mode(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.mode(), pandas_series.mode())


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_mul(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "mul")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_multiply(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "multiply")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_name(data):
    modin_series, pandas_series = create_test_series(data)
    assert modin_series.name == pandas_series.name
    modin_series.name = pandas_series.name = "New_name"
    assert modin_series.name == pandas_series.name
    assert modin_series._query_compiler.columns == ["New_name"]


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_nbytes(data):
    modin_series, pandas_series = create_test_series(data)
    with pytest.warns(UserWarning):
        assert modin_series.nbytes == pandas_series.nbytes


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_ndim(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    assert modin_series.ndim == 1


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_ne(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "ne")


@pytest.mark.skip(reason="Using pandas Series.")
def test_nlargest():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.nlargest(None)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_notnull(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.notnull(), pandas_series.notnull())


@pytest.mark.skip(reason="Using pandas Series.")
def test_nsmallest():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.nsmallest(None)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("dropna", [True, False], ids=["True", "False"])
def test_nunique(data, dropna):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.nunique(dropna=dropna), pandas_series.nunique(dropna=dropna))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_pct_change(data):
    modin_series, pandas_series = create_test_series(data)
    with pytest.warns(UserWarning):
        modin_series.pct_change()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_pipe(data):
    modin_series, pandas_series = create_test_series(data)
    n = len(modin_series.index)
    a, b, c = 2 % n, 0, 3 % n

    def h(x):
        return x.dropna()

    def g(x, arg1=0):
        for _ in range(arg1):
            x = x.append(x)
        return x

    def f(x, arg2=0, arg3=0):
        return x.drop(x.index[[arg2, arg3]])

    df_equals(
        f(g(h(modin_series), arg1=a), arg2=b, arg3=c),
        (modin_series.pipe(h).pipe(g, arg1=a).pipe(f, arg2=b, arg3=c)),
    )
    df_equals(
        (modin_series.pipe(h).pipe(g, arg1=a).pipe(f, arg2=b, arg3=c)),
        (pandas_series.pipe(h).pipe(g, arg1=a).pipe(f, arg2=b, arg3=c)),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_plot(request, data):
    modin_series, pandas_series = create_test_series(data)

    if name_contains(request.node.name, numeric_dfs):
        # We have to test this way because equality in plots means same object.
        zipped_plot_lines = zip(modin_series.plot().lines, pandas_series.plot().lines)
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
def test_pop(data):
    modin_series, pandas_series = create_test_series(data)

    for key in modin_series.keys():
        df_equals(modin_series.pop(key), pandas_series.pop(key))
        df_equals(modin_series, pandas_series)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_pow(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "pow")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_prod(data):
    modin_series, pandas_series = create_test_series(data)
    # Wrap in Series to test almost_equal because of overflow
    df_equals(pd.Series([modin_series.prod()]), pandas.Series([pandas_series.prod()]))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_product(data):
    modin_series, pandas_series = create_test_series(data)
    # Wrap in Series to test almost_equal because of overflow
    df_equals(
        pd.Series([modin_series.product()]), pandas.Series([pandas_series.product()])
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_ptp(data):
    modin_series, pandas_series = create_test_series(data)
    with pytest.warns(UserWarning):
        modin_series.ptp()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_put(data):
    modin_series, pandas_series = create_test_series(data)
    with pytest.warns(UserWarning):
        assert modin_series.put(0, 3) == pandas_series.put(0, 3)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("q", quantiles_values, ids=quantiles_keys)
def test_quantile(request, data, q):
    modin_series, pandas_series = create_test_series(data)
    if not name_contains(request.node.name, no_numeric_dfs):
        df_equals(modin_series.quantile(q), pandas_series.quantile(q))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_radd(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "radd")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize(
    "na_option", ["keep", "top", "bottom"], ids=["keep", "top", "bottom"]
)
def test_rank(data, na_option):
    modin_series, pandas_series = create_test_series(data)
    try:
        pandas_result = pandas_series.rank(na_option=na_option)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.rank(na_option=na_option)
    else:
        modin_result = modin_series.rank(na_option=na_option)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_ravel(data):
    modin_series, pandas_series = create_test_series(data)
    with pytest.warns(UserWarning):
        np.testing.assert_equal(modin_series.ravel(), pandas_series.ravel())


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_rdiv(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "rdiv")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_real(data):
    modin_series, pandas_series = create_test_series(data)
    with pytest.warns(UserWarning):
        np.testing.assert_equal(modin_series.real, pandas_series.real)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_reindex(data):
    modin_series, pandas_series = create_test_series(data)
    pandas_result = pandas_series.reindex(
        list(pandas_series.index) + ["_A_NEW_ROW"], fill_value=0
    )
    modin_result = modin_series.reindex(
        list(modin_series.index) + ["_A_NEW_ROW"], fill_value=0
    )
    df_equals(pandas_result, modin_result)

    frame_data = {
        "col1": [0, 1, 2, 3],
        "col2": [4, 5, 6, 7],
        "col3": [8, 9, 10, 11],
        "col4": [12, 13, 14, 15],
        "col5": [0, 0, 0, 0],
    }
    pandas_df = pandas.DataFrame(frame_data)
    modin_df = pd.DataFrame(frame_data)

    for col in pandas_df.columns:
        modin_series = modin_df[col]
        pandas_series = pandas_df[col]
        df_equals(
            modin_series.reindex([0, 3, 2, 1]), pandas_series.reindex([0, 3, 2, 1])
        )
        df_equals(modin_series.reindex([0, 6, 2]), pandas_series.reindex([0, 6, 2]))
        df_equals(
            modin_series.reindex(index=[0, 1, 5]),
            pandas_series.reindex(index=[0, 1, 5]),
        )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_reindex_axis(data):
    modin_series, pandas_series = create_test_series(data)
    modin_series.reindex_axis(
        [i for i in modin_series.index[: len(modin_series.index) // 2]]
    )


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

    series1 = df1["windspeed"]
    series2 = df2["windspeed"]
    with pytest.warns(UserWarning):
        series2.reindex_like(series1)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_rename(data):
    modin_series, pandas_series = create_test_series(data)
    new_name = "NEW_NAME"
    df_equals(modin_series.rename(new_name), pandas_series.rename(new_name))

    modin_series_cp = modin_series.copy()
    pandas_series_cp = pandas_series.copy()
    modin_series_cp.rename(new_name, inplace=True)
    pandas_series_cp.rename(new_name, inplace=True)
    df_equals(modin_series_cp, pandas_series_cp)

    modin_result = modin_series.rename("{}__".format)
    pandas_result = pandas_series.rename("{}__".format)
    df_equals(modin_result, pandas_result)


def test_reorder_levels():
    series = pd.Series(
        np.random.randint(1, 100, 12),
        index=pd.MultiIndex.from_tuples(
            [
                (num, letter, color)
                for num in range(1, 3)
                for letter in ["a", "b", "c"]
                for color in ["Red", "Green"]
            ],
            names=["Number", "Letter", "Color"],
        ),
    )
    with pytest.warns(UserWarning):
        series.reorder_levels(["Letter", "Color", "Number"])


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize(
    "repeats", [2, 3, 4], ids=["repeats_{}".format(i) for i in [2, 3, 4]]
)
def test_repeat(data, repeats):
    modin_series, pandas_series = create_test_series(data)
    with pytest.warns(UserWarning):
        df_equals(modin_series.repeat(repeats), pandas_series.repeat(repeats))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_replace(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with pytest.warns(UserWarning):
        modin_series.replace(0, 5)


def test_resample():
    modin_series = pd.Series(
        [10, 11, 9, 13, 14, 18, 17, 19],
        index=pd.date_range("01/01/2018", periods=8, freq="W"),
    )
    with pytest.warns(UserWarning):
        modin_series.resample("M")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("drop", [True, False], ids=["True", "False"])
def test_reset_index(data, drop):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.reset_index(drop=drop), pandas_series.reset_index(drop=drop))

    modin_series_cp = modin_series.copy()
    pandas_series_cp = pandas_series.copy()
    try:
        pandas_result = pandas_series_cp.reset_index(drop=drop, inplace=True)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series_cp.reset_index(drop=drop, inplace=True)
    else:
        modin_result = modin_series_cp.reset_index(drop=drop, inplace=True)
        df_equals(pandas_result, modin_result)


@pytest.mark.skip(reason="Using pandas Series.")
def test_reshape():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.reshape(None)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_rfloordiv(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "rfloordiv")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_rmod(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "rmod")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_rmul(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "rmul")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_rolling(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with pytest.warns(UserWarning):
        modin_series.rolling(10)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_round(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.round(), pandas_series.round())


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_rpow(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "rpow")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_rsub(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "rsub")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_rtruediv(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "rtruediv")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_sample(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(
        modin_series.sample(frac=0.5, random_state=21019),
        pandas_series.sample(frac=0.5, random_state=21019),
    )
    df_equals(
        modin_series.sample(n=12, random_state=21019),
        pandas_series.sample(n=12, random_state=21019),
    )
    with pytest.warns(UserWarning):
        df_equals(
            modin_series.sample(n=0, random_state=21019),
            pandas_series.sample(n=0, random_state=21019),
        )
    with pytest.raises(ValueError):
        modin_series.sample(n=-3)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_searchsorted(data):
    modin_series, pandas_series = create_test_series(data)
    with pytest.warns(UserWarning):
        modin_series.searchsorted(3)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_select(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with pytest.warns(UserWarning):
        modin_series.select(lambda x: x == 4)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_sem(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with pytest.warns(UserWarning):
        modin_series.sem()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_set_axis(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    modin_series.set_axis(labels=["{}_{}".format(i, i + 1) for i in modin_series.index])


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_set_value(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with pytest.warns(UserWarning):
        modin_series.set_value(5, 6)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_shape(data):
    modin_series, pandas_series = create_test_series(data)
    assert modin_series.shape == pandas_series.shape


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_shift(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with pytest.warns(UserWarning):
        modin_series.shift()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_size(data):
    modin_series, pandas_series = create_test_series(data)
    assert modin_series.size == pandas_series.size


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_skew(data, skipna):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.skew(skipna=skipna), pandas_series.skew(skipna=skipna))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_slice_shift(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with pytest.warns(UserWarning):
        modin_series.slice_shift()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize(
    "ascending", bool_arg_values, ids=arg_keys("ascending", bool_arg_keys)
)
@pytest.mark.parametrize(
    "sort_remaining", bool_arg_values, ids=arg_keys("sort_remaining", bool_arg_keys)
)
@pytest.mark.parametrize("na_position", ["first", "last"], ids=["first", "last"])
def test_sort_index(data, ascending, sort_remaining, na_position):
    modin_series, pandas_series = create_test_series(data)
    df_equals(
        modin_series.sort_index(
            ascending=ascending, sort_remaining=sort_remaining, na_position=na_position
        ),
        pandas_series.sort_index(
            ascending=ascending, sort_remaining=sort_remaining, na_position=na_position
        ),
    )

    modin_series_cp = modin_series.copy()
    pandas_series_cp = pandas_series.copy()
    modin_series_cp.sort_index(
        ascending=ascending,
        sort_remaining=sort_remaining,
        na_position=na_position,
        inplace=True,
    )
    pandas_series_cp.sort_index(
        ascending=ascending,
        sort_remaining=sort_remaining,
        na_position=na_position,
        inplace=True,
    )
    df_equals(modin_series_cp, pandas_series_cp)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("ascending", [True, False], ids=["True", "False"])
@pytest.mark.parametrize("na_position", ["first", "last"], ids=["first", "last"])
def test_sort_values(data, ascending, na_position):
    modin_series, pandas_series = create_test_series(data)
    modin_result = modin_series.sort_values(
        ascending=ascending, na_position=na_position
    )
    pandas_result = pandas_series.sort_values(
        ascending=ascending, na_position=na_position
    )
    # Note: For `ascending=False` only
    # For some reason, the indexing of Series and DataFrame differ in the underlying
    # algorithm. The order of values is the same, but the index values are shuffled.
    # Since we use `DataFrame.sort_values` even for Series, the index can be different
    # between `pandas.Series.sort_values`. For this reason, we check that the values are
    # identical instead of the index as well.
    if ascending:
        df_equals(modin_result, pandas_result)
    else:
        np.testing.assert_equal(modin_result.values, pandas_result.values)

    modin_series_cp = modin_series.copy()
    pandas_series_cp = pandas_series.copy()
    modin_series_cp.sort_values(
        ascending=ascending, na_position=na_position, inplace=True
    )
    pandas_series_cp.sort_values(
        ascending=ascending, na_position=na_position, inplace=True
    )
    # See above about `ascending=False`
    if ascending:
        df_equals(modin_series_cp, pandas_series_cp)
    else:
        np.testing.assert_equal(modin_series_cp.values, pandas_series_cp.values)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_squeeze(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.squeeze(None), pandas_series.squeeze(None))
    df_equals(modin_series.squeeze(0), pandas_series.squeeze(0))
    with pytest.raises(ValueError):
        modin_series.squeeze(1)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize("ddof", int_arg_values, ids=arg_keys("ddof", int_arg_keys))
def test_std(request, data, skipna, ddof):
    modin_series, pandas_series = create_test_series(data)
    try:
        pandas_result = pandas_series.std(skipna=skipna, ddof=ddof)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.std(skipna=skipna, ddof=ddof)
    else:
        modin_result = modin_series.std(skipna=skipna, ddof=ddof)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_strides(data):
    modin_series, pandas_series = create_test_series(data)
    with pytest.warns(UserWarning):
        modin_series.strides


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_sub(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "sub")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_subtract(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "subtract")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "min_count", int_arg_values, ids=arg_keys("min_count", int_arg_keys)
)
def test_sum(data, skipna, min_count):
    modin_series, pandas_series = create_test_series(data)
    try:
        pandas_result = pandas_series.sum(skipna=skipna, min_count=min_count)
    except Exception:
        with pytest.raises(TypeError):
            modin_series.sum(skipna=skipna, min_count=min_count)
    else:
        modin_result = modin_series.sum(skipna=skipna, min_count=min_count)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_swapaxes(data):
    modin_series, pandas_series = create_test_series(data)
    with pytest.warns(UserWarning):
        modin_series.swapaxes(0, 0)


def test_swaplevel():
    s = pd.Series(
        np.random.randint(1, 100, 12),
        index=pd.MultiIndex.from_tuples(
            [
                (num, letter, color)
                for num in range(1, 3)
                for letter in ["a", "b", "c"]
                for color in ["Red", "Green"]
            ],
            names=["Number", "Letter", "Color"],
        ),
    )
    with pytest.warns(UserWarning):
        s.swaplevel("Number", "Color")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("n", int_arg_values, ids=arg_keys("n", int_arg_keys))
def test_tail(data, n):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.tail(n), pandas_series.tail(n))
    df_equals(
        modin_series.tail(len(modin_series)), pandas_series.tail(len(pandas_series))
    )


def test_take():
    series = pd.Series([1, 2, 3, 4], index=[0, 2, 3, 1])
    with pytest.warns(UserWarning):
        series.take([0, 3])


def test_to_period():
    idx = pd.date_range("1/1/2012", periods=5, freq="M")
    series = pd.Series(np.random.randint(0, 100, size=(len(idx))), index=idx)
    with pytest.warns(UserWarning):
        series.to_period()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_to_sparse(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with pytest.warns(UserWarning):
        modin_series.to_sparse()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_to_string(request, data):
    modin_series, pandas_series = create_test_series(data)
    # Skips nan because only difference is nan instead of NaN
    if not name_contains(request.node.name, ["nan"]):
        assert modin_series.to_string() == pandas_series.to_string()


def test_to_timestamp():
    idx = pd.date_range("1/1/2012", periods=5, freq="M")
    series = pd.Series(np.random.randint(0, 100, size=(len(idx))), index=idx)
    with pytest.warns(UserWarning):
        series.to_period().to_timestamp()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_to_xarray(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with pytest.warns(UserWarning):
        modin_series.to_xarray()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_tolist(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with pytest.warns(UserWarning):
        modin_series.tolist()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
def test_transform(data, func):
    modin_series, pandas_series = create_test_series(data)
    try:
        pandas_result = pandas_series.transform(func)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.transform(func)
    else:
        df_equals(modin_series.transform(func), pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_transpose(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.transpose(), modin_series)
    df_equals(modin_series.transpose(), pandas_series.transpose())
    df_equals(modin_series.transpose(), pandas_series)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_truediv(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "truediv")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_truncate(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with pytest.warns(UserWarning):
        modin_series.truncate()


def test_tshift():
    idx = pd.date_range("1/1/2012", periods=5, freq="M")
    modin_series = pd.Series(np.random.randint(0, 100, size=len(idx)), index=idx)
    with pytest.warns(UserWarning):
        modin_series.to_period().tshift()


def test_tz_convert():
    idx = pd.date_range("1/1/2012", periods=5, freq="M")
    modin_series = pd.Series(np.random.randint(0, 100, size=len(idx)), index=idx)
    with pytest.warns(UserWarning):
        modin_series.tz_localize("America/Los_Angeles").tz_convert(
            "America/Los_Angeles"
        )


def test_tz_localize():
    idx = pd.date_range("1/1/2012", periods=5, freq="M")
    modin_series = pd.Series(np.random.randint(0, 100, size=len(idx)), index=idx)
    with pytest.warns(UserWarning):
        modin_series.tz_localize("America/Los_Angeles")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_unique(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with pytest.warns(UserWarning):
        modin_series.unique()


def test_unstack():
    s = pd.Series(
        np.random.randint(1, 100, 12),
        index=pd.MultiIndex.from_tuples(
            [
                (num, letter, color)
                for num in range(1, 3)
                for letter in ["a", "b", "c"]
                for color in ["Red", "Green"]
            ],
            names=["Number", "Letter", "Color"],
        ),
    )
    with pytest.warns(UserWarning):
        s.unstack()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_update(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with pytest.warns(UserWarning):
        try:
            modin_series.update(pd.Series([4.1 for _ in modin_series]))
        except Exception:
            pass


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_valid(data):
    modin_series, pandas_series = create_test_series(data)

    with pytest.warns(UserWarning):
        modin_series.valid()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_value_counts(data):
    modin_series, pandas_series = create_test_series(data)

    with pytest.warns(UserWarning):
        modin_series.value_counts()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_values(data):
    modin_series, pandas_series = create_test_series(data)

    np.testing.assert_equal(modin_series.values, pandas_series.values)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize("ddof", int_arg_values, ids=arg_keys("ddof", int_arg_keys))
def test_var(data, skipna, ddof):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.var(skipna=skipna, ddof=ddof)
    except Exception:
        with pytest.raises(TypeError):
            modin_series.var(skipna=skipna, ddof=ddof)
    else:
        modin_result = modin_series.var(skipna=skipna, ddof=ddof)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_view(data):
    modin_series, pandas_series = create_test_series(data)

    with pytest.warns(UserWarning):
        modin_series.view(None)


def test_where():
    frame_data = random_state.randn(100)
    pandas_series = pandas.Series(frame_data)
    modin_series = pd.Series(frame_data)
    pandas_cond_series = pandas_series % 5 < 2
    modin_cond_series = modin_series % 5 < 2

    pandas_result = pandas_series.where(pandas_cond_series, -pandas_series)
    modin_result = modin_series.where(modin_cond_series, -modin_series)
    assert all((to_pandas(modin_result) == pandas_result))

    other = pandas.Series(random_state.randn(100))
    pandas_result = pandas_series.where(pandas_cond_series, other, axis=0)
    modin_result = modin_series.where(modin_cond_series, other, axis=0)
    assert all(to_pandas(modin_result) == pandas_result)

    pandas_result = pandas_series.where(pandas_series < 2, True)
    modin_result = modin_series.where(modin_series < 2, True)
    assert all(to_pandas(modin_result) == pandas_result)


@pytest.mark.skip("Deprecated in pandas.")
def test_xs():
    series = pd.Series([4, 0, "mammal", "cat", "walks"])
    with pytest.warns(UserWarning):
        series.xs("mammal")
