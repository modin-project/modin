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
import numpy as np
import json
import pandas
import matplotlib
import modin.pandas as pd
from numpy.testing import assert_array_equal
from pandas.core.base import SpecificationError
import sys

from modin.utils import to_pandas
from .utils import (
    random_state,
    RAND_LOW,
    RAND_HIGH,
    df_equals,
    arg_keys,
    name_contains,
    test_data,
    test_data_values,
    test_data_keys,
    test_data_with_duplicates_values,
    test_data_with_duplicates_keys,
    test_string_data_values,
    test_string_data_keys,
    test_string_list_data_values,
    test_string_list_data_keys,
    string_sep_values,
    string_sep_keys,
    string_na_rep_values,
    string_na_rep_keys,
    numeric_dfs,
    no_numeric_dfs,
    agg_func_keys,
    agg_func_values,
    agg_func_except_keys,
    agg_func_except_values,
    numeric_agg_funcs,
    quantiles_keys,
    quantiles_values,
    axis_keys,
    axis_values,
    bool_arg_keys,
    bool_arg_values,
    int_arg_keys,
    int_arg_values,
    encoding_types,
    categories_equals,
    eval_general,
    test_data_small_values,
    test_data_small_keys,
    test_data_categorical_values,
    test_data_categorical_keys,
    generate_multiindex,
    test_data_diff_dtype,
    sort_index_for_equal_values,
)
from modin.config import NPartitions

NPartitions.put(4)

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")


def get_rop(op):
    if op.startswith("__") and op.endswith("__"):
        return "__r" + op[2:]
    else:
        return None


def inter_df_math_helper(modin_series, pandas_series, op):
    inter_df_math_helper_one_side(modin_series, pandas_series, op)
    rop = get_rop(op)
    if rop:
        inter_df_math_helper_one_side(modin_series, pandas_series, rop)


def inter_df_math_helper_one_side(modin_series, pandas_series, op):
    try:
        pandas_attr = getattr(pandas_series, op)
    except Exception as e:
        with pytest.raises(type(e)):
            _ = getattr(modin_series, op)
        return
    modin_attr = getattr(modin_series, op)

    try:
        pandas_result = pandas_attr(4)
    except Exception as e:
        with pytest.raises(type(e)):
            repr(modin_attr(4))  # repr to force materialization
    else:
        modin_result = modin_attr(4)
        df_equals(modin_result, pandas_result)

    try:
        pandas_result = pandas_attr(4.0)
    except Exception as e:
        with pytest.raises(type(e)):
            repr(modin_attr(4.0))  # repr to force materialization
    else:
        modin_result = modin_attr(4.0)
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
        pandas_result = pandas_attr(pandas_series)
    except Exception as e:
        with pytest.raises(type(e)):
            repr(modin_attr(modin_series))  # repr to force materialization
    else:
        modin_result = modin_attr(modin_series)
        df_equals(modin_result, pandas_result)

    list_test = random_state.randint(RAND_LOW, RAND_HIGH, size=(modin_series.shape[0]))
    try:
        pandas_result = pandas_attr(list_test)
    except Exception as e:
        with pytest.raises(type(e)):
            repr(modin_attr(list_test))  # repr to force materialization
    else:
        modin_result = modin_attr(list_test)
        df_equals(modin_result, pandas_result)

    series_test_modin = pd.Series(list_test, index=modin_series.index)
    series_test_pandas = pandas.Series(list_test, index=pandas_series.index)
    try:
        pandas_result = pandas_attr(series_test_pandas)
    except Exception as e:
        with pytest.raises(type(e)):
            repr(modin_attr(series_test_modin))  # repr to force materialization
    else:
        modin_result = modin_attr(series_test_modin)
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


def create_test_series(vals, sort=False, **kwargs):
    if isinstance(vals, dict):
        modin_series = pd.Series(vals[next(iter(vals.keys()))], **kwargs)
        pandas_series = pandas.Series(vals[next(iter(vals.keys()))], **kwargs)
    else:
        modin_series = pd.Series(vals, **kwargs)
        pandas_series = pandas.Series(vals, **kwargs)
    if sort:
        modin_series = modin_series.sort_values().reset_index(drop=True)
        pandas_series = pandas_series.sort_values().reset_index(drop=True)
    return modin_series, pandas_series


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_to_frame(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.to_frame(name="miao"), pandas_series.to_frame(name="miao"))


def test_accessing_index_element_as_property():
    s = pd.Series([10, 20, 30], index=["a", "b", "c"])
    assert s.b == 20
    with pytest.raises(Exception):
        _ = s.d


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_callable_key_in_getitem(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(
        modin_series[lambda s: s.index % 2 == 0],
        pandas_series[lambda s: s.index % 2 == 0],
    )


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
    modin_series = pd.Series(list(range(1000)))
    pandas_series = pandas.Series(list(range(1000)))
    df_equals(modin_series[:30], pandas_series[:30])
    df_equals(modin_series[modin_series > 500], pandas_series[pandas_series > 500])

    # Test empty series
    df_equals(pd.Series([])[:30], pandas.Series([])[:30])


def test___getitem__1383():
    # see #1383 for more details
    data = ["", "a", "b", "c", "a"]
    modin_series = pd.Series(data)
    pandas_series = pandas.Series(data)
    df_equals(modin_series[3:7], pandas_series[3:7])


@pytest.mark.parametrize("start", [-7, -5, -3, 0, None, 3, 5, 7])
@pytest.mark.parametrize("stop", [-7, -5, -3, 0, None, 3, 5, 7])
def test___getitem_edge_cases(start, stop):
    data = ["", "a", "b", "c", "a"]
    modin_series = pd.Series(data)
    pandas_series = pandas.Series(data)
    df_equals(modin_series[start:stop], pandas_series[start:stop])


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


@pytest.mark.parametrize("name", ["Dates", None])
@pytest.mark.parametrize(
    "dt_index", [True, False], ids=["dt_index_true", "dt_index_false"]
)
@pytest.mark.parametrize(
    "data",
    [
        *test_data_values,
        pytest.param(
            "empty",
            marks=pytest.mark.xfail_backends(
                ["BaseOnPython"],
                reason="Empty Series has a missmatched from Pandas dtype.",
            ),
        ),
    ],
    ids=[*test_data_keys, "empty"],
)
def test___repr__(name, dt_index, data):
    if data == "empty":
        modin_series, pandas_series = pd.Series(), pandas.Series()
    else:
        modin_series, pandas_series = create_test_series(data)
    pandas_series.name = modin_series.name = name
    if dt_index:
        index = pandas.date_range(
            "1/1/2000", periods=len(pandas_series.index), freq="T"
        )
        pandas_series.index = modin_series.index = index
    assert repr(modin_series) == repr(pandas_series)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___round__(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(round(modin_series), round(pandas_series))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___setitem__(data):
    modin_series, pandas_series = create_test_series(data)
    for key in modin_series.keys():
        modin_series[key] = 0
        pandas_series[key] = 0
        df_equals(modin_series, pandas_series)


@pytest.mark.parametrize(
    "key",
    [
        pytest.param(lambda idx: slice(1, 3), id="location_based_slice"),
        pytest.param(lambda idx: slice(idx[1], idx[-1]), id="index_based_slice"),
        pytest.param(lambda idx: [idx[0], idx[2], idx[-1]], id="list_of_labels"),
        pytest.param(
            lambda idx: [True if i % 2 else False for i in range(len(idx))],
            id="boolean_mask",
        ),
    ],
)
@pytest.mark.parametrize(
    "index",
    [
        pytest.param(
            lambda idx_len: [chr(x) for x in range(ord("a"), ord("a") + idx_len)],
            id="str_index",
        ),
        pytest.param(lambda idx_len: list(range(1, idx_len + 1)), id="int_index"),
    ],
)
def test___setitem___non_hashable(key, index):
    data = np.arange(5)
    index = index(len(data))
    key = key(index)
    md_sr, pd_sr = create_test_series(data, index=index)

    md_sr[key] = 10
    pd_sr[key] = 10
    df_equals(md_sr, pd_sr)


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
    eval_general(
        *create_test_series(data),
        lambda df: df.agg(func),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("func", agg_func_except_values, ids=agg_func_except_keys)
def test_agg_except(data, func):
    # SpecificationError is arisen because we treat a Series as a DataFrame.
    # See details in pandas issue 36036.
    with pytest.raises(SpecificationError):
        eval_general(
            *create_test_series(data),
            lambda df: df.agg(func),
        )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
def test_agg_numeric(request, data, func):
    if name_contains(request.node.name, numeric_agg_funcs) and name_contains(
        request.node.name, numeric_dfs
    ):
        axis = 0
        eval_general(
            *create_test_series(data),
            lambda df: df.agg(func, axis),
        )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("func", agg_func_except_values, ids=agg_func_except_keys)
def test_agg_numeric_except(request, data, func):
    if name_contains(request.node.name, numeric_agg_funcs) and name_contains(
        request.node.name, numeric_dfs
    ):
        axis = 0
        # SpecificationError is arisen because we treat a Series as a DataFrame.
        # See details in pandas issue 36036.
        with pytest.raises(SpecificationError):
            eval_general(
                *create_test_series(data),
                lambda df: df.agg(func, axis),
            )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
def test_aggregate(data, func):
    axis = 0
    eval_general(
        *create_test_series(data),
        lambda df: df.aggregate(func, axis),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("func", agg_func_except_values, ids=agg_func_except_keys)
def test_aggregate_except(data, func):
    axis = 0
    # SpecificationError is arisen because we treat a Series as a DataFrame.
    # See details in pandas issues 36036.
    with pytest.raises(SpecificationError):
        eval_general(
            *create_test_series(data),
            lambda df: df.aggregate(func, axis),
        )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
def test_aggregate_numeric(request, data, func):
    if name_contains(request.node.name, numeric_agg_funcs) and name_contains(
        request.node.name, numeric_dfs
    ):
        axis = 0
        eval_general(
            *create_test_series(data),
            lambda df: df.agg(func, axis),
        )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("func", agg_func_except_values, ids=agg_func_except_keys)
def test_aggregate_numeric_except(request, data, func):
    if name_contains(request.node.name, numeric_agg_funcs) and name_contains(
        request.node.name, numeric_dfs
    ):
        axis = 0
        # SpecificationError is arisen because we treat a Series as a DataFrame.
        # See details in pandas issues 36036.
        with pytest.raises(SpecificationError):
            eval_general(
                *create_test_series(data),
                lambda df: df.agg(func, axis),
            )


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
def test_apply(data, func):
    eval_general(
        *create_test_series(data),
        lambda df: df.apply(func),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("func", agg_func_except_values, ids=agg_func_except_keys)
def test_apply_except(data, func):
    # SpecificationError is arisen because we treat a Series as a DataFrame.
    # See details in pandas issues 36036.
    with pytest.raises(SpecificationError):
        eval_general(
            *create_test_series(data),
            lambda df: df.apply(func),
        )


def test_apply_external_lib():
    json_string = """
    {
        "researcher": {
            "name": "Ford Prefect",
            "species": "Betelgeusian",
            "relatives": [
                {
                    "name": "Zaphod Beeblebrox",
                    "species": "Betelgeusian"
                }
            ]
        }
    }
    """
    modin_result = pd.DataFrame.from_dict({"a": [json_string]}).a.apply(json.loads)
    pandas_result = pandas.DataFrame.from_dict({"a": [json_string]}).a.apply(json.loads)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
def test_apply_numeric(request, data, func):
    if name_contains(request.node.name, numeric_dfs):
        eval_general(
            *create_test_series(data),
            lambda df: df.apply(func),
        )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("func", agg_func_except_values, ids=agg_func_except_keys)
def test_apply_numeric_except(request, data, func):
    if name_contains(request.node.name, numeric_dfs):
        # SpecificationError is arisen because we treat a Series as a DataFrame.
        # See details in pandas issues 36036.
        with pytest.raises(SpecificationError):
            eval_general(
                *create_test_series(data),
                lambda df: df.apply(func),
            )


@pytest.mark.parametrize("axis", [None, 0, 1])
@pytest.mark.parametrize("level", [None, -1, 0, 1])
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("func", ["count", "all", "kurt", "array", "searchsorted"])
def test_apply_text_func(level, data, func, axis):
    func_kwargs = {}
    if level:
        func_kwargs.update({"level": level})
    if axis:
        func_kwargs.update({"axis": axis})
    rows_number = len(next(iter(data.values())))  # length of the first data column
    level_0 = np.random.choice([0, 1, 2], rows_number)
    level_1 = np.random.choice([3, 4, 5], rows_number)
    index = pd.MultiIndex.from_arrays([level_0, level_1])

    modin_series, pandas_series = create_test_series(data)
    modin_series.index = index
    pandas_series.index = index

    eval_general(modin_series, pandas_series, lambda df: df.apply(func), **func_kwargs)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("skipna", [True, False])
def test_argmax(data, skipna):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.argmax(skipna=skipna), pandas_series.argmax(skipna=skipna))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("skipna", [True, False])
def test_argmin(data, skipna):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.argmin(skipna=skipna), pandas_series.argmin(skipna=skipna))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_argsort(data):
    modin_series, pandas_series = create_test_series(data)
    with pytest.warns(UserWarning):
        modin_result = modin_series.argsort()
    df_equals(modin_result, pandas_series.argsort())


def test_asfreq():
    index = pd.date_range("1/1/2000", periods=4, freq="T")
    series = pd.Series([0.0, None, 2.0, 3.0], index=index)
    with pytest.warns(UserWarning):
        # We are only testing that this defaults to pandas, so we will just check for
        # the warning
        series.asfreq(freq="30S")


@pytest.mark.parametrize(
    "where",
    [
        20,
        30,
        [10, 40],
        [20, 30],
        [20],
        25,
        [25, 45],
        [25, 30],
        pandas.Index([20, 30]),
        pandas.Index([10]),
    ],
)
def test_asof(where):
    # With NaN:
    values = [1, 2, np.nan, 4]
    index = [10, 20, 30, 40]
    modin_series, pandas_series = pd.Series(values, index=index), pandas.Series(
        values, index=index
    )
    df_equals(modin_series.asof(where), pandas_series.asof(where))

    # No NaN:
    values = [1, 2, 7, 4]
    modin_series, pandas_series = pd.Series(values, index=index), pandas.Series(
        values, index=index
    )
    df_equals(modin_series.asof(where), pandas_series.asof(where))


@pytest.mark.parametrize(
    "where",
    [
        20,
        30,
        [10.5, 40.5],
        [10],
        pandas.Index([20, 30]),
        pandas.Index([10.5]),
    ],
)
def test_asof_large(where):
    values = test_data["float_nan_data"]["col1"]
    index = list(range(len(values)))
    modin_series, pandas_series = pd.Series(values, index=index), pandas.Series(
        values, index=index
    )
    df_equals(modin_series.asof(where), pandas_series.asof(where))


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


def test_astype_categorical():
    modin_df = pd.Series(["A", "A", "B", "B", "A"])
    pandas_df = pandas.Series(["A", "A", "B", "B", "A"])

    modin_result = modin_df.astype("category")
    pandas_result = pandas_df.astype("category")
    df_equals(modin_result, pandas_result)
    assert modin_result.dtype == pandas_result.dtype

    modin_df = pd.Series([1, 1, 2, 1, 2, 2, 3, 1, 2, 1, 2])
    pandas_df = pandas.Series([1, 1, 2, 1, 2, 2, 3, 1, 2, 1, 2])
    df_equals(modin_result, pandas_result)
    assert modin_result.dtype == pandas_result.dtype


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
    i = pd.date_range("2008-01-01", periods=1000, freq="12H")
    modin_series = pd.Series(list(range(1000)), index=i)
    pandas_series = pandas.Series(list(range(1000)), index=i)
    df_equals(modin_series.at_time("12:00"), pandas_series.at_time("12:00"))
    df_equals(modin_series.at_time("3:00"), pandas_series.at_time("3:00"))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("lag", [1, 2, 3])
def test_autocorr(data, lag):
    modin_series, pandas_series = create_test_series(data)
    modin_result = modin_series.autocorr(lag=lag)
    pandas_result = pandas_series.autocorr(lag=lag)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_axes(data):
    modin_series, pandas_series = create_test_series(data)
    assert modin_series.axes[0].equals(pandas_series.axes[0])
    assert len(modin_series.axes) == len(pandas_series.axes)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_attrs(data):
    modin_series, pandas_series = create_test_series(data)
    eval_general(modin_series, pandas_series, lambda df: df.attrs)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_array(data):
    modin_series, pandas_series = create_test_series(data)
    eval_general(modin_series, pandas_series, lambda df: df.array)


@pytest.mark.skip(reason="Using pandas Series.")
def test_between():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.between(None, None)


def test_between_time():
    i = pd.date_range("2008-01-01", periods=1000, freq="12H")
    modin_series = pd.Series(list(range(1000)), index=i)
    pandas_series = pandas.Series(list(range(1000)), index=i)
    df_equals(
        modin_series.between_time("12:00", "17:00"),
        pandas_series.between_time("12:00", "17:00"),
    )
    df_equals(
        modin_series.between_time("3:00", "8:00"),
        pandas_series.between_time("3:00", "8:00"),
    )
    df_equals(
        modin_series.between_time("3:00", "8:00", False),
        pandas_series.between_time("3:00", "8:00", False),
    )


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
def test_combine(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    modin_series2 = modin_series % (max(modin_series) // 2)
    modin_series.combine(modin_series2, lambda s1, s2: s1 if s1 < s2 else s2)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_combine_first(data):
    modin_series, pandas_series = create_test_series(data)
    modin_series2 = modin_series % (max(modin_series) // 2)
    pandas_series2 = pandas_series % (max(pandas_series) // 2)
    modin_result = modin_series.combine_first(modin_series2)
    pandas_result = pandas_series.combine_first(pandas_series2)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_compress(data):
    modin_series, pandas_series = create_test_series(data)  # noqa: F841
    try:
        pandas_series.compress(pandas_series > 30)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.compress(modin_series > 30)
    else:
        modin_series.compress(modin_series > 30)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_constructor(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series, pandas_series)
    df_equals(pd.Series(modin_series), pandas.Series(pandas_series))


def test_constructor_columns_and_index():
    modin_series = pd.Series([1, 1, 10], index=[1, 2, 3], name="health")
    pandas_series = pandas.Series([1, 1, 10], index=[1, 2, 3], name="health")
    df_equals(modin_series, pandas_series)
    df_equals(pd.Series(modin_series), pandas.Series(pandas_series))
    df_equals(
        pd.Series(modin_series, name="max_speed"),
        pandas.Series(pandas_series, name="max_speed"),
    )
    df_equals(
        pd.Series(modin_series, index=[1, 2]),
        pandas.Series(pandas_series, index=[1, 2]),
    )
    with pytest.raises(NotImplementedError):
        pd.Series(modin_series, index=[1, 2, 99999])


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_copy(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series, modin_series.copy())
    df_equals(modin_series.copy(), pandas_series)
    df_equals(modin_series.copy(), pandas_series.copy())


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_corr(data):
    modin_series, pandas_series = create_test_series(data)
    modin_result = modin_series.corr(modin_series)
    pandas_result = pandas_series.corr(pandas_series)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_count(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.count(), pandas_series.count())


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_cov(data):
    modin_series, pandas_series = create_test_series(data)
    modin_result = modin_series.cov(modin_series)
    pandas_result = pandas_series.cov(pandas_series)
    df_equals(modin_result, pandas_result)


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
    modin_series, pandas_series = create_test_series(data)
    ind_len = len(modin_series)

    # Test 1D array input
    arr = np.arange(ind_len)
    modin_result = modin_series.dot(arr)
    pandas_result = pandas_series.dot(arr)
    df_equals(modin_result, pandas_result)

    # Test 2D array input
    arr = np.arange(ind_len * 2).reshape(ind_len, 2)
    modin_result = modin_series.dot(arr)
    pandas_result = pandas_series.dot(arr)
    assert_array_equal(modin_result, pandas_result)

    # Test bad dimensions
    with pytest.raises(ValueError):
        modin_result = modin_series.dot(np.arange(ind_len + 10))

    # Test dataframe input
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    modin_result = modin_series.dot(modin_df)
    pandas_result = pandas_series.dot(pandas_df)
    df_equals(modin_result, pandas_result)

    # Test series input
    modin_series_2 = pd.Series(np.arange(ind_len), index=modin_series.index)
    pandas_series_2 = pandas.Series(np.arange(ind_len), index=pandas_series.index)
    modin_result = modin_series.dot(modin_series_2)
    pandas_result = pandas_series.dot(pandas_series_2)
    df_equals(modin_result, pandas_result)

    # Test when input series index doesn't line up with columns
    with pytest.raises(ValueError):
        modin_result = modin_series.dot(
            pd.Series(
                np.arange(ind_len), index=["a" for _ in range(len(modin_series.index))]
            )
        )

    # Test case when left series has size (1 x 1)
    # and right dataframe has size (1 x n)
    modin_result = pd.Series([1]).dot(pd.DataFrame(modin_series).T)
    pandas_result = pandas.Series([1]).dot(pandas.DataFrame(pandas_series).T)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_matmul(data):
    modin_series, pandas_series = create_test_series(data)  # noqa: F841
    ind_len = len(modin_series)

    # Test 1D array input
    arr = np.arange(ind_len)
    modin_result = modin_series @ arr
    pandas_result = pandas_series @ arr
    df_equals(modin_result, pandas_result)

    # Test 2D array input
    arr = np.arange(ind_len * 2).reshape(ind_len, 2)
    modin_result = modin_series @ arr
    pandas_result = pandas_series @ arr
    assert_array_equal(modin_result, pandas_result)

    # Test bad dimensions
    with pytest.raises(ValueError):
        modin_result = modin_series @ np.arange(ind_len + 10)

    # Test dataframe input
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)
    modin_result = modin_series @ modin_df
    pandas_result = pandas_series @ pandas_df
    df_equals(modin_result, pandas_result)

    # Test series input
    modin_series_2 = pd.Series(np.arange(ind_len), index=modin_series.index)
    pandas_series_2 = pandas.Series(np.arange(ind_len), index=pandas_series.index)
    modin_result = modin_series @ modin_series_2
    pandas_result = pandas_series @ pandas_series_2
    df_equals(modin_result, pandas_result)

    # Test when input series index doesn't line up with columns
    with pytest.raises(ValueError):
        modin_result = modin_series @ pd.Series(
            np.arange(ind_len), index=["a" for _ in range(len(modin_series.index))]
        )


@pytest.mark.skip(reason="Using pandas Series.")
def test_drop():
    modin_series = create_test_series()

    with pytest.raises(NotImplementedError):
        modin_series.drop(None, None, None, None)


@pytest.mark.parametrize(
    "data", test_data_with_duplicates_values, ids=test_data_with_duplicates_keys
)
@pytest.mark.parametrize(
    "keep", ["last", "first", False], ids=["last", "first", "False"]
)
@pytest.mark.parametrize("inplace", [True, False], ids=["True", "False"])
def test_drop_duplicates(data, keep, inplace):
    modin_series, pandas_series = create_test_series(data)
    df_equals(
        modin_series.drop_duplicates(keep=keep, inplace=inplace),
        pandas_series.drop_duplicates(keep=keep, inplace=inplace),
    )


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


@pytest.mark.xfail_backends(
    ["BaseOnPython"], reason="Empty Series has a missmatched from Pandas dtype."
)
def test_dtype_empty():
    modin_series, pandas_series = pd.Series(), pandas.Series()
    assert modin_series.dtype == pandas_series.dtype


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_dtype(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.dtype, modin_series.dtypes)
    df_equals(modin_series.dtype, pandas_series.dtype)
    df_equals(modin_series.dtype, pandas_series.dtypes)


def test_dt():
    data = pd.date_range("2016-12-31", periods=128, freq="D", tz="Europe/Berlin")
    modin_series = pd.Series(data)
    pandas_series = pandas.Series(data)

    df_equals(modin_series.dt.date, pandas_series.dt.date)
    df_equals(modin_series.dt.time, pandas_series.dt.time)
    df_equals(modin_series.dt.timetz, pandas_series.dt.timetz)
    df_equals(modin_series.dt.year, pandas_series.dt.year)
    df_equals(modin_series.dt.month, pandas_series.dt.month)
    df_equals(modin_series.dt.day, pandas_series.dt.day)
    df_equals(modin_series.dt.hour, pandas_series.dt.hour)
    df_equals(modin_series.dt.minute, pandas_series.dt.minute)
    df_equals(modin_series.dt.second, pandas_series.dt.second)
    df_equals(modin_series.dt.microsecond, pandas_series.dt.microsecond)
    df_equals(modin_series.dt.nanosecond, pandas_series.dt.nanosecond)
    df_equals(modin_series.dt.week, pandas_series.dt.week)
    df_equals(modin_series.dt.weekofyear, pandas_series.dt.weekofyear)
    df_equals(modin_series.dt.dayofweek, pandas_series.dt.dayofweek)
    df_equals(modin_series.dt.weekday, pandas_series.dt.weekday)
    df_equals(modin_series.dt.dayofyear, pandas_series.dt.dayofyear)
    df_equals(modin_series.dt.quarter, pandas_series.dt.quarter)
    df_equals(modin_series.dt.is_month_start, pandas_series.dt.is_month_start)
    df_equals(modin_series.dt.is_month_end, pandas_series.dt.is_month_end)
    df_equals(modin_series.dt.is_quarter_start, pandas_series.dt.is_quarter_start)
    df_equals(modin_series.dt.is_quarter_end, pandas_series.dt.is_quarter_end)
    df_equals(modin_series.dt.is_year_start, pandas_series.dt.is_year_start)
    df_equals(modin_series.dt.is_year_end, pandas_series.dt.is_year_end)
    df_equals(modin_series.dt.is_leap_year, pandas_series.dt.is_leap_year)
    df_equals(modin_series.dt.daysinmonth, pandas_series.dt.daysinmonth)
    df_equals(modin_series.dt.days_in_month, pandas_series.dt.days_in_month)
    assert modin_series.dt.tz == pandas_series.dt.tz
    assert modin_series.dt.freq == pandas_series.dt.freq
    df_equals(modin_series.dt.to_period("W"), pandas_series.dt.to_period("W"))
    assert_array_equal(
        modin_series.dt.to_pydatetime(), pandas_series.dt.to_pydatetime()
    )
    df_equals(
        modin_series.dt.tz_localize(None),
        pandas_series.dt.tz_localize(None),
    )
    df_equals(
        modin_series.dt.tz_convert(tz="Europe/Berlin"),
        pandas_series.dt.tz_convert(tz="Europe/Berlin"),
    )

    df_equals(modin_series.dt.normalize(), pandas_series.dt.normalize())
    df_equals(
        modin_series.dt.strftime("%B %d, %Y, %r"),
        pandas_series.dt.strftime("%B %d, %Y, %r"),
    )
    df_equals(modin_series.dt.round("H"), pandas_series.dt.round("H"))
    df_equals(modin_series.dt.floor("H"), pandas_series.dt.floor("H"))
    df_equals(modin_series.dt.ceil("H"), pandas_series.dt.ceil("H"))
    df_equals(modin_series.dt.month_name(), pandas_series.dt.month_name())
    df_equals(modin_series.dt.day_name(), pandas_series.dt.day_name())

    modin_series = pd.Series(pd.to_timedelta(np.arange(128), unit="d"))
    pandas_series = pandas.Series(pandas.to_timedelta(np.arange(128), unit="d"))

    assert_array_equal(
        modin_series.dt.to_pytimedelta(), pandas_series.dt.to_pytimedelta()
    )
    df_equals(modin_series.dt.total_seconds(), pandas_series.dt.total_seconds())
    df_equals(modin_series.dt.days, pandas_series.dt.days)
    df_equals(modin_series.dt.seconds, pandas_series.dt.seconds)
    df_equals(modin_series.dt.microseconds, pandas_series.dt.microseconds)
    df_equals(modin_series.dt.nanoseconds, pandas_series.dt.nanoseconds)
    df_equals(modin_series.dt.components, pandas_series.dt.components)

    data_per = pd.date_range("1/1/2012", periods=128, freq="M")
    pandas_series = pandas.Series(data_per, index=data_per).dt.to_period()
    modin_series = pd.Series(data_per, index=data_per).dt.to_period()

    df_equals(modin_series.dt.qyear, pandas_series.dt.qyear)
    df_equals(modin_series.dt.start_time, pandas_series.dt.start_time)
    df_equals(modin_series.dt.end_time, pandas_series.dt.end_time)
    df_equals(modin_series.dt.to_timestamp(), pandas_series.dt.to_timestamp())


@pytest.mark.parametrize(
    "data", test_data_with_duplicates_values, ids=test_data_with_duplicates_keys
)
@pytest.mark.parametrize(
    "keep", ["last", "first", False], ids=["last", "first", "False"]
)
def test_duplicated(data, keep):
    modin_series, pandas_series = create_test_series(data)
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
    i = pd.date_range("2010-04-09", periods=400, freq="2D")
    modin_series = pd.Series(list(range(400)), index=i)
    pandas_series = pandas.Series(list(range(400)), index=i)
    df_equals(modin_series.first("3D"), pandas_series.first("3D"))
    df_equals(modin_series.first("20D"), pandas_series.first("20D"))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_first_valid_index(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.first_valid_index(), pandas_series.first_valid_index())


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_floordiv(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "floordiv")


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
def test_is_monotonic(data):
    modin_series, pandas_series = create_test_series(data)
    assert modin_series.is_monotonic == pandas_series.is_monotonic


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_is_monotonic_decreasing(data):
    modin_series, pandas_series = create_test_series(data)
    assert modin_series.is_monotonic_decreasing == pandas_series.is_monotonic_decreasing


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_is_monotonic_increasing(data):
    modin_series, pandas_series = create_test_series(data)
    assert modin_series.is_monotonic_increasing == pandas_series.is_monotonic_increasing


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_is_unique(data):
    modin_series, pandas_series = create_test_series(data)
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
def test_keys(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.keys(), pandas_series.keys())


def test_kurtosis_alias():
    # It's optimization. If failed, Series.kurt should be tested explicitly
    # in tests: `test_kurt_kurtosis`, `test_kurt_kurtosis_level`.
    assert pd.Series.kurt == pd.Series.kurtosis


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("skipna", bool_arg_values, ids=bool_arg_keys)
def test_kurtosis(axis, skipna):
    eval_general(
        *create_test_series(test_data["float_nan_data"]),
        lambda df: df.kurtosis(axis=axis, skipna=skipna),
    )


@pytest.mark.parametrize("axis", ["rows", "columns"])
@pytest.mark.parametrize(
    "numeric_only",
    [
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                reason="Modin - DID NOT RAISE <class 'NotImplementedError'>"
            ),
        ),
        False,
        None,
    ],
)
def test_kurtosis_numeric_only(axis, numeric_only):
    eval_general(
        *create_test_series(test_data_diff_dtype),
        lambda df: df.kurtosis(axis=axis, numeric_only=numeric_only),
    )


@pytest.mark.parametrize("level", [-1, 0, 1])
def test_kurtosis_level(level):
    data = test_data["int_data"]
    modin_s, pandas_s = create_test_series(data)

    index = generate_multiindex(len(data.keys()))
    modin_s.columns = index
    pandas_s.columns = index

    eval_general(
        modin_s,
        pandas_s,
        lambda s: s.kurtosis(axis=1, level=level),
    )


def test_last():
    modin_index = pd.date_range("2010-04-09", periods=400, freq="2D")
    pandas_index = pandas.date_range("2010-04-09", periods=400, freq="2D")
    modin_series = pd.Series(list(range(400)), index=modin_index)
    pandas_series = pandas.Series(list(range(400)), index=pandas_index)
    df_equals(modin_series.last("3D"), pandas_series.last("3D"))
    df_equals(modin_series.last("20D"), pandas_series.last("20D"))


@pytest.mark.parametrize("func", ["all", "any", "mad", "count"])
def test_index_order(func):
    # see #1708 and #1869 for details
    s_modin, s_pandas = create_test_series(test_data["float_nan_data"])
    rows_number = len(s_modin.index)
    level_0 = np.random.choice([x for x in range(10)], rows_number)
    level_1 = np.random.choice([x for x in range(10)], rows_number)
    index = pandas.MultiIndex.from_arrays([level_0, level_1])

    s_modin.index = index
    s_pandas.index = index

    df_equals(
        getattr(s_modin, func)(level=0).index,
        getattr(s_pandas, func)(level=0).index,
    )


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

    indices = [True if i % 3 == 0 else False for i in range(len(modin_series.index))]
    modin_result = modin_series.loc[indices]
    pandas_result = pandas_series.loc[indices]
    df_equals(modin_result, pandas_result)

    # From issue #1988
    index = pd.MultiIndex.from_product([np.arange(10), np.arange(10)], names=["f", "s"])
    data = np.arange(100)
    modin_series = pd.Series(data, index=index).sort_index()
    pandas_series = pandas.Series(data, index=index).sort_index()
    modin_result = modin_series.loc[
        (slice(None), 1),
    ]
    pandas_result = pandas_series.loc[
        (slice(None), 1),
    ]
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_lt(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "lt")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", [None, 0])
@pytest.mark.parametrize("skipna", [None, True, False])
@pytest.mark.parametrize("level", [0, -1, None])
def test_mad(level, data, axis, skipna):
    modin_series, pandas_series = create_test_series(data)
    df_equals(
        modin_series.mad(axis=axis, skipna=skipna, level=level),
        pandas_series.mad(axis=axis, skipna=skipna, level=level),
    )


@pytest.mark.parametrize("na_values", ["ignore", None], ids=["na_ignore", "na_none"])
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_map(data, na_values):
    modin_series, pandas_series = create_test_series(data)
    df_equals(
        modin_series.map(str, na_action=na_values),
        pandas_series.map(str, na_action=na_values),
    )
    mapper = {i: str(i) for i in range(100)}
    df_equals(
        modin_series.map(mapper, na_action=na_values),
        pandas_series.map(mapper, na_action=na_values),
    )

    # Return list objects
    modin_series_lists = modin_series.map(lambda s: [s, s, s])
    pandas_series_lists = pandas_series.map(lambda s: [s, s, s])
    df_equals(modin_series_lists, pandas_series_lists)

    # Index into list objects
    df_equals(
        modin_series_lists.map(lambda l: l[0]), pandas_series_lists.map(lambda l: l[0])
    )


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


@pytest.mark.parametrize(
    "method", ["median", "skew", "std", "sum", "var", "prod", "sem"]
)
def test_median_skew_std_sum_var_prod_sem_1953(method):
    # See #1953 for details
    data = [3, 3, 3, 3, 3, 3, 3, 3, 3]
    arrays = [
        ["1", "1", "1", "2", "2", "2", "3", "3", "3"],
        ["1", "2", "3", "4", "5", "6", "7", "8", "9"],
    ]
    modin_s = pd.Series(data, index=arrays)
    pandas_s = pandas.Series(data, index=arrays)
    eval_general(modin_s, pandas_s, lambda s: getattr(s, method)(level=0))


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


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_nsmallest(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(
        modin_series.nsmallest(n=5, keep="first"),
        pandas_series.nsmallest(n=5, keep="first"),
    )
    df_equals(
        modin_series.nsmallest(n=10, keep="first"),
        pandas_series.nsmallest(n=10, keep="first"),
    )
    df_equals(
        modin_series.nsmallest(n=10, keep="last"),
        pandas_series.nsmallest(n=10, keep="last"),
    )
    df_equals(modin_series.nsmallest(keep="all"), pandas_series.nsmallest(keep="all"))


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
        for left, right in zipped_plot_lines:
            if isinstance(left.get_xdata(), np.ma.core.MaskedArray) and isinstance(
                right.get_xdata(), np.ma.core.MaskedArray
            ):
                assert all((left.get_xdata() == right.get_xdata()).data)
            else:
                assert np.array_equal(left.get_xdata(), right.get_xdata())
            if isinstance(left.get_ydata(), np.ma.core.MaskedArray) and isinstance(
                right.get_ydata(), np.ma.core.MaskedArray
            ):
                assert all((left.get_ydata() == right.get_ydata()).data)
            else:
                assert np.array_equal(left.get_xdata(), right.get_xdata())


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


def test_product_alias():
    assert pd.Series.prod == pd.Series.product


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_prod(axis, skipna):
    eval_general(
        *create_test_series(test_data["float_nan_data"]),
        lambda s: s.prod(axis=axis, skipna=skipna),
    )


@pytest.mark.parametrize(
    "numeric_only",
    [
        None,
        False,
        pytest.param(True, marks=pytest.mark.xfail(reason="didn't raise Exception")),
    ],
)
@pytest.mark.parametrize(
    "min_count", int_arg_values, ids=arg_keys("min_count", int_arg_keys)
)
def test_prod_specific(min_count, numeric_only):
    eval_general(
        *create_test_series(test_data_diff_dtype),
        lambda df: df.prod(min_count=min_count, numeric_only=numeric_only),
    )


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
@pytest.mark.parametrize("order", [None, "C", "F", "A", "K"])
def test_ravel(data, order):
    modin_series, pandas_series = create_test_series(data)
    np.testing.assert_equal(
        modin_series.ravel(order=order), pandas_series.ravel(order=order)
    )


# TODO: remove xfail mark then #1628 will be fixed
@pytest.mark.xfail(
    reason="Modin Series with category dtype is buggy for now. See #1628 for more details."
)
@pytest.mark.parametrize(
    "data",
    [
        pandas.Categorical(np.arange(1000), ordered=True),
        pandas.Categorical(np.arange(1000), ordered=False),
        pandas.Categorical(np.arange(1000), categories=np.arange(500), ordered=True),
        pandas.Categorical(np.arange(1000), categories=np.arange(500), ordered=False),
    ],
)
@pytest.mark.parametrize("order", [None, "C", "F", "A", "K"])
def test_ravel_category(data, order):
    modin_series, pandas_series = create_test_series(data)
    categories_equals(modin_series.ravel(order=order), pandas_series.ravel(order=order))


@pytest.mark.parametrize(
    "data",
    [
        pandas.Categorical(np.arange(10), ordered=True),
        pandas.Categorical(np.arange(10), ordered=False),
        pandas.Categorical(np.arange(10), categories=np.arange(5), ordered=True),
        pandas.Categorical(np.arange(10), categories=np.arange(5), ordered=False),
    ],
)
@pytest.mark.parametrize("order", [None, "C", "F", "A", "K"])
def test_ravel_simple_category(data, order):
    modin_series, pandas_series = create_test_series(data)
    categories_equals(modin_series.ravel(order=order), pandas_series.ravel(order=order))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_rdiv(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "rdiv")


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

    # MultiIndex
    modin_series, pandas_series = create_test_series(data)
    modin_series.index, pandas_series.index = [
        generate_multiindex(len(pandas_series))
    ] * 2
    pandas_result = pandas_series.reindex(list(reversed(pandas_series.index)))
    modin_result = modin_series.reindex(list(reversed(modin_series.index)))
    df_equals(pandas_result, modin_result)


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
    data = np.random.randint(1, 100, 12)
    modin_series = pd.Series(
        data,
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
    pandas_series = pandas.Series(
        data,
        index=pandas.MultiIndex.from_tuples(
            [
                (num, letter, color)
                for num in range(1, 3)
                for letter in ["a", "b", "c"]
                for color in ["Red", "Green"]
            ],
            names=["Number", "Letter", "Color"],
        ),
    )
    modin_result = modin_series.reorder_levels(["Letter", "Color", "Number"])
    pandas_result = pandas_series.reorder_levels(["Letter", "Color", "Number"])
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize(
    "repeats", [0, 2, 3, 4], ids=["repeats_{}".format(i) for i in [0, 2, 3, 4]]
)
def test_repeat(data, repeats):
    eval_general(pd.Series(data), pandas.Series(data), lambda df: df.repeat(repeats))


@pytest.mark.parametrize("data", [np.arange(256)])
@pytest.mark.parametrize(
    "repeats",
    [
        [0],
        [2],
        [3],
        [4],
        np.arange(256),
        [0] * 64 + [2] * 64 + [3] * 32 + [4] * 32 + [5] * 64,
        [2] * 257,
        [2] * 128,
    ],
)
def test_repeat_lists(data, repeats):
    eval_general(
        pd.Series(data),
        pandas.Series(data),
        lambda df: df.repeat(repeats),
    )


def test_replace():
    modin_series = pd.Series([0, 1, 2, 3, 4])
    pandas_series = pandas.Series([0, 1, 2, 3, 4])
    modin_result = modin_series.replace(0, 5)
    pandas_result = pandas_series.replace(0, 5)
    df_equals(modin_result, pandas_result)

    modin_result = modin_series.replace([1, 2], method="bfill")
    pandas_result = pandas_series.replace([1, 2], method="bfill")
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("closed", ["left", "right"])
@pytest.mark.parametrize("label", ["right", "left"])
@pytest.mark.parametrize("level", [None, 1])
def test_resample(closed, label, level):
    rule = "5T"
    freq = "H"
    base = 2

    index = pandas.date_range("1/1/2000", periods=12, freq=freq)
    pandas_series = pandas.Series(range(12), index=index)
    modin_series = pd.Series(range(12), index=index)

    if level is not None:
        index = pandas.MultiIndex.from_product(
            [["a", "b", "c"], pandas.date_range("31/12/2000", periods=4, freq=freq)]
        )
        pandas_series.index = index
        modin_series.index = index
    pandas_resampler = pandas_series.resample(
        rule, closed=closed, label=label, base=base, level=level
    )
    modin_resampler = modin_series.resample(
        rule, closed=closed, label=label, base=base, level=level
    )

    df_equals(modin_resampler.count(), pandas_resampler.count())
    df_equals(modin_resampler.var(0), pandas_resampler.var(0))
    df_equals(modin_resampler.sum(), pandas_resampler.sum())
    df_equals(modin_resampler.std(), pandas_resampler.std())
    df_equals(modin_resampler.sem(), pandas_resampler.sem())
    df_equals(modin_resampler.size(), pandas_resampler.size())
    df_equals(modin_resampler.prod(), pandas_resampler.prod())
    df_equals(modin_resampler.ohlc(), pandas_resampler.ohlc())
    df_equals(modin_resampler.min(), pandas_resampler.min())
    df_equals(modin_resampler.median(), pandas_resampler.median())
    df_equals(modin_resampler.mean(), pandas_resampler.mean())
    df_equals(modin_resampler.max(), pandas_resampler.max())
    df_equals(modin_resampler.last(), pandas_resampler.last())
    df_equals(modin_resampler.first(), pandas_resampler.first())
    df_equals(modin_resampler.nunique(), pandas_resampler.nunique())
    df_equals(
        modin_resampler.pipe(lambda x: x.max() - x.min()),
        pandas_resampler.pipe(lambda x: x.max() - x.min()),
    )
    df_equals(
        modin_resampler.transform(lambda x: (x - x.mean()) / x.std()),
        pandas_resampler.transform(lambda x: (x - x.mean()) / x.std()),
    )
    df_equals(
        modin_resampler.aggregate("max"),
        pandas_resampler.aggregate("max"),
    )
    df_equals(
        modin_resampler.apply("sum"),
        pandas_resampler.apply("sum"),
    )
    df_equals(
        modin_resampler.get_group(name=list(modin_resampler.groups)[0]),
        pandas_resampler.get_group(name=list(pandas_resampler.groups)[0]),
    )
    assert pandas_resampler.indices == modin_resampler.indices
    assert pandas_resampler.groups == modin_resampler.groups
    df_equals(modin_resampler.quantile(), pandas_resampler.quantile())
    # Upsampling from level= or on= selection is not supported
    if level is None:
        df_equals(
            modin_resampler.interpolate(),
            pandas_resampler.interpolate(),
        )
        df_equals(modin_resampler.asfreq(), pandas_resampler.asfreq())
        df_equals(
            modin_resampler.fillna(method="nearest"),
            pandas_resampler.fillna(method="nearest"),
        )
        df_equals(modin_resampler.pad(), pandas_resampler.pad())
        df_equals(modin_resampler.nearest(), pandas_resampler.nearest())
        df_equals(modin_resampler.bfill(), pandas_resampler.bfill())
        df_equals(modin_resampler.backfill(), pandas_resampler.backfill())
        df_equals(modin_resampler.ffill(), pandas_resampler.ffill())
    df_equals(
        modin_resampler.apply(["sum", "mean", "max"]),
        pandas_resampler.apply(["sum", "mean", "max"]),
    )
    df_equals(
        modin_resampler.aggregate(["sum", "mean", "max"]),
        pandas_resampler.aggregate(["sum", "mean", "max"]),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("drop", [True, False], ids=["True", "False"])
@pytest.mark.parametrize("name", [None, "Custom name"])
@pytest.mark.parametrize("inplace", [True, False])
def test_reset_index(data, drop, name, inplace):
    eval_general(
        *create_test_series(data),
        lambda df, *args, **kwargs: df.reset_index(*args, **kwargs),
        drop=drop,
        name=name,
        inplace=inplace,
        __inplace__=inplace,
    )


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
    try:
        pandas_result = pandas_series.sample(frac=0.5, random_state=21019)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.sample(frac=0.5, random_state=21019)
    else:
        modin_result = modin_series.sample(frac=0.5, random_state=21019)
        df_equals(pandas_result, modin_result)

    try:
        pandas_result = pandas_series.sample(n=12, random_state=21019)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.sample(n=12, random_state=21019)
    else:
        modin_result = modin_series.sample(n=12, random_state=21019)
        df_equals(pandas_result, modin_result)

    with pytest.warns(UserWarning):
        df_equals(
            modin_series.sample(n=0, random_state=21019),
            pandas_series.sample(n=0, random_state=21019),
        )
    with pytest.raises(ValueError):
        modin_series.sample(n=-3)


@pytest.mark.parametrize("single_value_data", [True, False])
@pytest.mark.parametrize(
    "use_multiindex",
    [
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                reason="When use_multiindex=True, test is failing."
            ),
        ),
        False,
    ],
)
@pytest.mark.parametrize("sorter", [True, None])
@pytest.mark.parametrize("values_number", [1, 2, 5])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_searchsorted(
    data, side, values_number, sorter, use_multiindex, single_value_data
):
    data = data if not single_value_data else data[next(iter(data.keys()))][0]
    if not sorter:
        modin_series, pandas_series = create_test_series(vals=data, sort=True)
    else:
        modin_series, pandas_series = create_test_series(vals=data)
        sorter = np.argsort(list(modin_series))

    if use_multiindex:
        rows_number = len(modin_series.index)
        level_0_series = random_state.choice([0, 1], rows_number)
        level_1_series = random_state.choice([2, 3], rows_number)
        index_series = pd.MultiIndex.from_arrays(
            [level_0_series, level_1_series], names=["first", "second"]
        )
        modin_series.index = index_series
        pandas_series.index = index_series

    min_sample = modin_series.min(skipna=True)
    max_sample = modin_series.max(skipna=True)

    if single_value_data:
        values = [data]
    else:
        values = []
        values.append(pandas_series.sample(n=values_number, random_state=random_state))
        values.append(
            random_state.uniform(low=min_sample, high=max_sample, size=values_number)
        )
        values.append(
            random_state.uniform(
                low=max_sample, high=2 * max_sample, size=values_number
            )
        )
        values.append(
            random_state.uniform(
                low=min_sample - max_sample, high=min_sample, size=values_number
            )
        )
        pure_float = random_state.uniform(float(min_sample), float(max_sample))
        pure_int = int(pure_float)
        values.append(pure_float)
        values.append(pure_int)

    test_cases = [
        modin_series.searchsorted(value=value, side=side, sorter=sorter)
        == pandas_series.searchsorted(value=value, side=side, sorter=sorter)
        for value in values
    ]
    test_cases = [
        case.all() if not isinstance(case, bool) else case for case in test_cases
    ]

    for case in test_cases:
        assert case


@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize("ddof", int_arg_values, ids=arg_keys("ddof", int_arg_keys))
def test_sem_float_nan_only(skipna, ddof):
    eval_general(
        *create_test_series(test_data["float_nan_data"]),
        lambda df: df.sem(skipna=skipna, ddof=ddof),
    )


@pytest.mark.parametrize("ddof", int_arg_values, ids=arg_keys("ddof", int_arg_keys))
def test_sem_int_only(ddof):
    eval_general(
        *create_test_series(test_data["int_data"]),
        lambda df: df.sem(ddof=ddof),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_set_axis(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    modin_series.set_axis(labels=["{}_{}".format(i, i + 1) for i in modin_series.index])


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_shape(data):
    modin_series, pandas_series = create_test_series(data)
    assert modin_series.shape == pandas_series.shape


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_shift(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.shift(), pandas_series.shift())
    df_equals(modin_series.shift(fill_value=777), pandas_series.shift(fill_value=777))
    df_equals(modin_series.shift(periods=7), pandas_series.shift(periods=7))
    df_equals(modin_series.shift(periods=-3), pandas_series.shift(periods=-3))
    eval_general(modin_series, pandas_series, lambda df: df.shift(axis=1))


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
@pytest.mark.parametrize("index", ["default", "ndarray"])
@pytest.mark.parametrize("periods", [0, 1, -1, 10, -10, 1000000000, -1000000000])
def test_slice_shift(data, index, periods):
    if index == "default":
        modin_series, pandas_series = create_test_series(data)
    elif index == "ndarray":
        modin_series, pandas_series = create_test_series(data)
        data_column_length = len(data[next(iter(data))])
        index_data = np.arange(2, data_column_length + 2)
        modin_series.index = index_data
        pandas_series.index = index_data

    df_equals(
        modin_series.slice_shift(periods=periods),
        pandas_series.slice_shift(periods=periods),
    )


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
    eval_general(
        modin_series,
        pandas_series,
        lambda df: df.sort_index(
            ascending=ascending,
            sort_remaining=sort_remaining,
            na_position=na_position,
        ),
    )

    eval_general(
        modin_series.copy(),
        pandas_series.copy(),
        lambda df: df.sort_index(
            ascending=ascending,
            sort_remaining=sort_remaining,
            na_position=na_position,
            inplace=True,
        ),
        __inplace__=True,
    )


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
def test_sub(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "sub")


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_subtract(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "subtract")


@pytest.mark.parametrize(
    "data",
    test_data_values + test_data_small_values,
    ids=test_data_keys + test_data_small_keys,
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize(
    "numeric_only",
    [
        None,
        False,
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                reason="numeric_only not implemented for pandas.Series"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "min_count", int_arg_values, ids=arg_keys("min_count", int_arg_keys)
)
def test_sum(data, axis, skipna, numeric_only, min_count):
    eval_general(
        *create_test_series(data),
        lambda df, *args, **kwargs: df.sum(*args, **kwargs),
        axis=axis,
        skipna=skipna,
        numeric_only=numeric_only,
        min_count=min_count,
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis1", [0, 1, "columns", "index"])
@pytest.mark.parametrize("axis2", [0, 1, "columns", "index"])
def test_swapaxes(data, axis1, axis2):
    modin_series, pandas_series = create_test_series(data)
    try:
        pandas_result = pandas_series.swapaxes(axis1, axis2)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.swapaxes(axis1, axis2)
    else:
        modin_result = modin_series.swapaxes(axis1, axis2)
        df_equals(modin_result, pandas_result)


def test_swaplevel():
    data = np.random.randint(1, 100, 12)
    modin_s = pd.Series(
        data,
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
    pandas_s = pandas.Series(
        data,
        index=pandas.MultiIndex.from_tuples(
            [
                (num, letter, color)
                for num in range(1, 3)
                for letter in ["a", "b", "c"]
                for color in ["Red", "Green"]
            ],
            names=["Number", "Letter", "Color"],
        ),
    )
    df_equals(
        modin_s.swaplevel("Number", "Color"), pandas_s.swaplevel("Number", "Color")
    )
    df_equals(modin_s.swaplevel(), pandas_s.swaplevel())
    df_equals(modin_s.swaplevel(1, 0), pandas_s.swaplevel(1, 0))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("n", int_arg_values, ids=arg_keys("n", int_arg_keys))
def test_tail(data, n):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.tail(n), pandas_series.tail(n))
    df_equals(
        modin_series.tail(len(modin_series)), pandas_series.tail(len(pandas_series))
    )


def test_take():
    modin_s = pd.Series(["falcon", "parrot", "lion", "cat"], index=[0, 2, 3, 1])
    pandas_s = pandas.Series(["falcon", "parrot", "lion", "cat"], index=[0, 2, 3, 1])
    a = modin_s.take([0, 3])
    df_equals(a, pandas_s.take([0, 3]))
    try:
        pandas_s.take([2], axis=1)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_s.take([2], axis=1)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_explode(data):
    modin_series, pandas_series = create_test_series(data)
    with pytest.warns(UserWarning):
        modin_result = modin_series.explode()
    pandas_result = pandas_series.explode()
    df_equals(modin_result, pandas_result)


def test_to_period():
    idx = pd.date_range("1/1/2012", periods=5, freq="M")
    series = pd.Series(np.random.randint(0, 100, size=(len(idx))), index=idx)
    with pytest.warns(UserWarning):
        series.to_period()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_to_numpy(data):
    modin_series, pandas_series = create_test_series(data)
    assert_array_equal(modin_series.values, pandas_series.values)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_to_string(request, data):
    eval_general(
        *create_test_series(data),
        lambda df: df.to_string(),
    )


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
    eval_general(
        *create_test_series(data),
        lambda df: df.transform(func),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("func", agg_func_except_values, ids=agg_func_except_keys)
def test_transform_except(data, func):
    eval_general(
        *create_test_series(data),
        lambda df: df.transform(func),
    )


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
    modin_series, pandas_series = create_test_series(data)

    before = 1
    after = len(modin_series - 3)
    df_equals(
        modin_series.truncate(before, after), pandas_series.truncate(before, after)
    )

    before = 1
    after = 3
    df_equals(
        modin_series.truncate(before, after), pandas_series.truncate(before, after)
    )

    before = None
    after = None
    df_equals(
        modin_series.truncate(before, after), pandas_series.truncate(before, after)
    )


def test_tshift():
    idx = pd.date_range("1/1/2012", periods=5, freq="M")
    data = np.random.randint(0, 100, size=len(idx))
    modin_series = pd.Series(data, index=idx)
    pandas_series = pandas.Series(data, index=idx)
    df_equals(modin_series.tshift(4), pandas_series.tshift(4))


def test_tz_convert():
    modin_idx = pd.date_range(
        "1/1/2012", periods=400, freq="2D", tz="America/Los_Angeles"
    )
    pandas_idx = pandas.date_range(
        "1/1/2012", periods=400, freq="2D", tz="America/Los_Angeles"
    )
    data = np.random.randint(0, 100, size=len(modin_idx))
    modin_series = pd.Series(data, index=modin_idx)
    pandas_series = pandas.Series(data, index=pandas_idx)
    modin_result = modin_series.tz_convert("UTC", axis=0)
    pandas_result = pandas_series.tz_convert("UTC", axis=0)
    df_equals(modin_result, pandas_result)

    modin_multi = pd.MultiIndex.from_arrays([modin_idx, range(len(modin_idx))])
    pandas_multi = pandas.MultiIndex.from_arrays([pandas_idx, range(len(modin_idx))])
    modin_series = pd.Series(data, index=modin_multi)
    pandas_series = pandas.Series(data, index=pandas_multi)
    df_equals(
        modin_series.tz_convert("UTC", axis=0, level=0),
        pandas_series.tz_convert("UTC", axis=0, level=0),
    )


def test_tz_localize():
    idx = pd.date_range("1/1/2012", periods=400, freq="2D")
    data = np.random.randint(0, 100, size=len(idx))
    modin_series = pd.Series(data, index=idx)
    pandas_series = pandas.Series(data, index=idx)
    df_equals(
        modin_series.tz_localize("America/Los_Angeles"),
        pandas_series.tz_localize("America/Los_Angeles"),
    )
    df_equals(
        modin_series.tz_localize("UTC"),
        pandas_series.tz_localize("UTC"),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_unique(data):
    modin_series, pandas_series = create_test_series(data)
    modin_result = modin_series.unique()
    pandas_result = pandas_series.unique()
    assert_array_equal(modin_result, pandas_result)
    assert modin_result.shape == pandas_result.shape

    modin_result = pd.Series([2, 1, 3, 3], name="A").unique()
    pandas_result = pandas.Series([2, 1, 3, 3], name="A").unique()
    assert_array_equal(modin_result, pandas_result)
    assert modin_result.shape == pandas_result.shape

    modin_result = pd.Series([pd.Timestamp("2016-01-01") for _ in range(3)]).unique()
    pandas_result = pandas.Series(
        [pd.Timestamp("2016-01-01") for _ in range(3)]
    ).unique()
    assert_array_equal(modin_result, pandas_result)
    assert modin_result.shape == pandas_result.shape

    modin_result = pd.Series(
        [pd.Timestamp("2016-01-01", tz="US/Eastern") for _ in range(3)]
    ).unique()
    pandas_result = pandas.Series(
        [pd.Timestamp("2016-01-01", tz="US/Eastern") for _ in range(3)]
    ).unique()
    assert_array_equal(modin_result, pandas_result)
    assert modin_result.shape == pandas_result.shape

    modin_result = pandas.Series(pd.Categorical(list("baabc"))).unique()
    pandas_result = pd.Series(pd.Categorical(list("baabc"))).unique()
    assert_array_equal(modin_result, pandas_result)
    assert modin_result.shape == pandas_result.shape

    modin_result = pd.Series(
        pd.Categorical(list("baabc"), categories=list("abc"), ordered=True)
    ).unique()
    pandas_result = pandas.Series(
        pd.Categorical(list("baabc"), categories=list("abc"), ordered=True)
    ).unique()
    assert_array_equal(modin_result, pandas_result)
    assert modin_result.shape == pandas_result.shape


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_unstack(data):
    modin_series, pandas_series = create_test_series(data)
    index = generate_multiindex(len(pandas_series), nlevels=4, is_tree_like=True)

    modin_series = pd.Series(data[next(iter(data.keys()))], index=index)
    pandas_series = pandas.Series(data[next(iter(data.keys()))], index=index)

    df_equals(modin_series.unstack(), pandas_series.unstack())
    df_equals(modin_series.unstack(level=0), pandas_series.unstack(level=0))
    df_equals(modin_series.unstack(level=[0, 1]), pandas_series.unstack(level=[0, 1]))
    df_equals(
        modin_series.unstack(level=[0, 1, 2]), pandas_series.unstack(level=[0, 1, 2])
    )


@pytest.mark.parametrize(
    "data, other_data",
    [([1, 2, 3], [4, 5, 6]), ([1, 2, 3], [4, 5, 6, 7, 8]), ([1, 2, 3], [4, np.nan, 6])],
)
def test_update(data, other_data):
    modin_series, pandas_series = pd.Series(data), pandas.Series(data)
    modin_series.update(pd.Series(other_data))
    pandas_series.update(pandas.Series(other_data))
    df_equals(modin_series, pandas_series)


@pytest.mark.parametrize("normalize, bins, dropna", [(True, 3, False)])
def test_value_counts(normalize, bins, dropna):
    # We sort indices for Modin and pandas result because of issue #1650
    modin_series, pandas_series = create_test_series(test_data_values[0])

    modin_result = sort_index_for_equal_values(
        modin_series.value_counts(normalize=normalize, ascending=False), False
    )
    pandas_result = sort_index_for_equal_values(
        pandas_series.value_counts(normalize=normalize, ascending=False), False
    )
    df_equals(modin_result, pandas_result)

    modin_result = sort_index_for_equal_values(
        modin_series.value_counts(bins=bins, ascending=False), False
    )
    pandas_result = sort_index_for_equal_values(
        pandas_series.value_counts(bins=bins, ascending=False), False
    )
    df_equals(modin_result, pandas_result)

    modin_result = sort_index_for_equal_values(
        modin_series.value_counts(dropna=dropna, ascending=True), True
    )
    pandas_result = sort_index_for_equal_values(
        pandas_series.value_counts(dropna=dropna, ascending=True), True
    )
    df_equals(modin_result, pandas_result)

    # from issue #2365
    arr = np.random.rand(2 ** 6)
    arr[::10] = np.nan
    modin_series, pandas_series = create_test_series(arr)
    modin_result = sort_index_for_equal_values(
        modin_series.value_counts(dropna=False, ascending=True), True
    )
    pandas_result = sort_index_for_equal_values(
        pandas_series.value_counts(dropna=False, ascending=True), True
    )
    df_equals(modin_result, pandas_result)

    modin_result = sort_index_for_equal_values(
        modin_series.value_counts(dropna=False, ascending=False), False
    )
    pandas_result = sort_index_for_equal_values(
        pandas_series.value_counts(dropna=False, ascending=False), False
    )
    df_equals(modin_result, pandas_result)


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


def test_view():
    modin_series = pd.Series([-2, -1, 0, 1, 2], dtype="int8")
    pandas_series = pandas.Series([-2, -1, 0, 1, 2], dtype="int8")
    modin_result = modin_series.view(dtype="uint8")
    pandas_result = pandas_series.view(dtype="uint8")
    df_equals(modin_result, pandas_result)

    modin_series = pd.Series([-20, -10, 0, 10, 20], dtype="int32")
    pandas_series = pandas.Series([-20, -10, 0, 10, 20], dtype="int32")
    modin_result = modin_series.view(dtype="float32")
    pandas_result = pandas_series.view(dtype="float32")
    df_equals(modin_result, pandas_result)

    modin_series = pd.Series([-200, -100, 0, 100, 200], dtype="int64")
    pandas_series = pandas.Series([-200, -100, 0, 100, 200], dtype="int64")
    modin_result = modin_series.view(dtype="float64")
    pandas_result = pandas_series.view(dtype="float64")
    df_equals(modin_result, pandas_result)


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


# Test str operations
def test_str_cat():
    data = ["abC|DeF,Hik", "gSaf,qWer|Gre", "asd3,4sad|", np.NaN]
    modin_series, pandas_series = create_test_series(data)
    others = data

    with pytest.warns(UserWarning):
        # We are only testing that this defaults to pandas, so we will just check for
        # the warning
        modin_series.str.cat(others)

    with pytest.warns(UserWarning):
        # We are only testing that this defaults to pandas, so we will just check for
        # the warning
        modin_series.str.cat(None)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("pat", string_sep_values, ids=string_sep_keys)
@pytest.mark.parametrize("n", int_arg_values, ids=int_arg_keys)
@pytest.mark.parametrize("expand", bool_arg_values, ids=bool_arg_keys)
def test_str_split(data, pat, n, expand):
    # Empty pattern not supported on Python 3.7+
    if sys.version_info[0] == 3 and sys.version_info[1] >= 7 and pat == "":
        return

    modin_series, pandas_series = create_test_series(data)

    if n >= -1:
        if expand and pat:
            with pytest.warns(UserWarning):
                # We are only testing that this defaults to pandas, so we will just check for
                # the warning
                modin_series.str.split(pat, n=n, expand=expand)
        elif not expand:
            try:
                pandas_result = pandas_series.str.split(pat, n=n, expand=expand)
            except Exception as e:
                with pytest.raises(type(e)):
                    modin_series.str.split(pat, n=n, expand=expand)
            else:
                modin_result = modin_series.str.split(pat, n=n, expand=expand)
                df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("pat", string_sep_values, ids=string_sep_keys)
@pytest.mark.parametrize("n", int_arg_values, ids=int_arg_keys)
@pytest.mark.parametrize("expand", bool_arg_values, ids=bool_arg_keys)
def test_str_rsplit(data, pat, n, expand):
    modin_series, pandas_series = create_test_series(data)

    if n >= -1:
        if expand and pat:
            with pytest.warns(UserWarning):
                # We are only testing that this defaults to pandas, so we will just check for
                # the warning
                modin_series.str.rsplit(pat, n=n, expand=expand)
        elif not expand:
            try:
                pandas_result = pandas_series.str.rsplit(pat, n=n, expand=expand)
            except Exception as e:
                with pytest.raises(type(e)):
                    modin_series.str.rsplit(pat, n=n, expand=expand)
            else:
                modin_result = modin_series.str.rsplit(pat, n=n, expand=expand)
                df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("i", int_arg_values, ids=int_arg_keys)
def test_str_get(data, i):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.get(i)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.get(i)
    else:
        modin_result = modin_series.str.get(i)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "data", test_string_list_data_values, ids=test_string_list_data_keys
)
@pytest.mark.parametrize("sep", string_sep_values, ids=string_sep_keys)
def test_str_join(data, sep):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.join(sep)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.join(sep)
    else:
        modin_result = modin_series.str.join(sep)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "data", test_string_list_data_values, ids=test_string_list_data_keys
)
@pytest.mark.parametrize("sep", string_sep_values, ids=string_sep_keys)
def test_str_get_dummies(data, sep):
    modin_series, pandas_series = create_test_series(data)

    if sep:
        with pytest.warns(UserWarning):
            # We are only testing that this defaults to pandas, so we will just check for
            # the warning
            modin_series.str.get_dummies(sep)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("pat", string_sep_values, ids=string_sep_keys)
@pytest.mark.parametrize("case", bool_arg_values, ids=bool_arg_keys)
@pytest.mark.parametrize("na", string_na_rep_values, ids=string_na_rep_keys)
def test_str_contains(data, pat, case, na):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.contains(pat, case=case, na=na, regex=False)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.contains(pat, case=case, na=na, regex=False)
    else:
        modin_result = modin_series.str.contains(pat, case=case, na=na, regex=False)
        df_equals(modin_result, pandas_result)

    # Test regex
    pat = ",|b"
    try:
        pandas_result = pandas_series.str.contains(pat, case=case, na=na, regex=True)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.contains(pat, case=case, na=na, regex=True)
    else:
        modin_result = modin_series.str.contains(pat, case=case, na=na, regex=True)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("pat", string_sep_values, ids=string_sep_keys)
@pytest.mark.parametrize("repl", string_sep_values, ids=string_sep_keys)
@pytest.mark.parametrize("n", int_arg_values, ids=int_arg_keys)
@pytest.mark.parametrize("case", bool_arg_values, ids=bool_arg_keys)
def test_str_replace(data, pat, repl, n, case):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.replace(
            pat, repl, n=n, case=case, regex=False
        )
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.replace(pat, repl, n=n, case=case, regex=False)
    else:
        modin_result = modin_series.str.replace(pat, repl, n=n, case=case, regex=False)
        df_equals(modin_result, pandas_result)

    # Test regex
    pat = ",|b"
    try:
        pandas_result = pandas_series.str.replace(pat, repl, n=n, case=case, regex=True)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.replace(pat, repl, n=n, case=case, regex=True)
    else:
        modin_result = modin_series.str.replace(pat, repl, n=n, case=case, regex=True)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("repeats", int_arg_values, ids=int_arg_keys)
def test_str_repeats(data, repeats):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.repeats(repeats)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.repeats(repeats)
    else:
        modin_result = modin_series.str.repeats(repeats)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("width", int_arg_values, ids=int_arg_keys)
@pytest.mark.parametrize(
    "side", ["left", "right", "both"], ids=["left", "right", "both"]
)
@pytest.mark.parametrize("fillchar", string_sep_values, ids=string_sep_keys)
def test_str_pad(data, width, side, fillchar):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.pad(width, side=side, fillchar=fillchar)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.pad(width, side=side, fillchar=fillchar)
    else:
        modin_result = modin_series.str.pad(width, side=side, fillchar=fillchar)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("width", int_arg_values, ids=int_arg_keys)
@pytest.mark.parametrize("fillchar", string_sep_values, ids=string_sep_keys)
def test_str_center(data, width, fillchar):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.center(width, fillchar=fillchar)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.center(width, fillchar=fillchar)
    else:
        modin_result = modin_series.str.center(width, fillchar=fillchar)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("width", int_arg_values, ids=int_arg_keys)
@pytest.mark.parametrize("fillchar", string_sep_values, ids=string_sep_keys)
def test_str_ljust(data, width, fillchar):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.ljust(width, fillchar=fillchar)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.ljust(width, fillchar=fillchar)
    else:
        modin_result = modin_series.str.ljust(width, fillchar=fillchar)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("width", int_arg_values, ids=int_arg_keys)
@pytest.mark.parametrize("fillchar", string_sep_values, ids=string_sep_keys)
def test_str_rjust(data, width, fillchar):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.rjust(width, fillchar=fillchar)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.rjust(width, fillchar=fillchar)
    else:
        modin_result = modin_series.str.rjust(width, fillchar=fillchar)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("width", int_arg_values, ids=int_arg_keys)
def test_str_zfill(data, width):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.zfill(width)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.zfill(width)
    else:
        modin_result = modin_series.str.zfill(width)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("width", int_arg_values, ids=int_arg_keys)
def test_str_wrap(data, width):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.wrap(width)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.wrap(width)
    else:
        modin_result = modin_series.str.wrap(width)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("start", int_arg_values, ids=int_arg_keys)
@pytest.mark.parametrize("stop", int_arg_values, ids=int_arg_keys)
@pytest.mark.parametrize("step", int_arg_values, ids=int_arg_keys)
def test_str_slice(data, start, stop, step):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.slice(start=start, stop=stop, step=step)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.slice(start=start, stop=stop, step=step)
    else:
        modin_result = modin_series.str.slice(start=start, stop=stop, step=step)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("start", int_arg_values, ids=int_arg_keys)
@pytest.mark.parametrize("stop", int_arg_values, ids=int_arg_keys)
@pytest.mark.parametrize("repl", string_sep_values, ids=string_sep_keys)
def test_str_slice_replace(data, start, stop, repl):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.slice_replace(
            start=start, stop=stop, repl=repl
        )
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.slice_replace(start=start, stop=stop, repl=repl)
    else:
        modin_result = modin_series.str.slice_replace(start=start, stop=stop, repl=repl)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("pat", string_sep_values, ids=string_sep_keys)
def test_str_count(data, pat):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.count(pat)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.count(pat)
    else:
        modin_result = modin_series.str.count(pat)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("pat", string_sep_values, ids=string_sep_keys)
@pytest.mark.parametrize("na", string_na_rep_values, ids=string_na_rep_keys)
def test_str_startswith(data, pat, na):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.startswith(pat, na=na)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.startswith(pat, na=na)
    else:
        modin_result = modin_series.str.startswith(pat, na=na)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("pat", string_sep_values, ids=string_sep_keys)
@pytest.mark.parametrize("na", string_na_rep_values, ids=string_na_rep_keys)
def test_str_endswith(data, pat, na):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.endswith(pat, na=na)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.endswith(pat, na=na)
    else:
        modin_result = modin_series.str.endswith(pat, na=na)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("pat", string_sep_values, ids=string_sep_keys)
def test_str_findall(data, pat):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.findall(pat)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.findall(pat)
    else:
        modin_result = modin_series.str.findall(pat)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("pat", string_sep_values, ids=string_sep_keys)
@pytest.mark.parametrize("case", bool_arg_values, ids=bool_arg_keys)
@pytest.mark.parametrize("na", string_na_rep_values, ids=string_na_rep_keys)
def test_str_match(data, pat, case, na):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.match(pat, case=case, na=na)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.match(pat, case=case, na=na)
    else:
        modin_result = modin_series.str.match(pat, case=case, na=na)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("expand", bool_arg_values, ids=bool_arg_keys)
def test_str_extract(data, expand):
    modin_series, pandas_series = create_test_series(data)

    if expand is not None:
        with pytest.warns(UserWarning):
            # We are only testing that this defaults to pandas, so we will just check for
            # the warning
            modin_series.str.extract(r"([ab])(\d)", expand=expand)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_str_extractall(data):
    modin_series, pandas_series = create_test_series(data)

    with pytest.warns(UserWarning):
        # We are only testing that this defaults to pandas, so we will just check for
        # the warning
        modin_series.str.extractall(r"([ab])(\d)")


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_str_len(data):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.len()
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.len()
    else:
        modin_result = modin_series.str.len()
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("to_strip", string_sep_values, ids=string_sep_keys)
def test_str_strip(data, to_strip):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.strip(to_strip=to_strip)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.strip(to_strip=to_strip)
    else:
        modin_result = modin_series.str.strip(to_strip=to_strip)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("to_strip", string_sep_values, ids=string_sep_keys)
def test_str_rstrip(data, to_strip):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.rstrip(to_strip=to_strip)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.rstrip(to_strip=to_strip)
    else:
        modin_result = modin_series.str.rstrip(to_strip=to_strip)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("to_strip", string_sep_values, ids=string_sep_keys)
def test_str_lstrip(data, to_strip):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.lstrip(to_strip=to_strip)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.lstrip(to_strip=to_strip)
    else:
        modin_result = modin_series.str.lstrip(to_strip=to_strip)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("sep", string_sep_values, ids=string_sep_keys)
@pytest.mark.parametrize("expand", bool_arg_values, ids=bool_arg_keys)
def test_str_partition(data, sep, expand):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.partition(sep, expand=expand)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.partition(sep, expand=expand)
    else:
        modin_result = modin_series.str.partition(sep, expand=expand)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("sep", string_sep_values, ids=string_sep_keys)
@pytest.mark.parametrize("expand", bool_arg_values, ids=bool_arg_keys)
def test_str_rpartition(data, sep, expand):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.rpartition(sep, expand=expand)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.rpartition(sep, expand=expand)
    else:
        modin_result = modin_series.str.rpartition(sep, expand=expand)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_str_lower(data):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.lower()
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.lower()
    else:
        modin_result = modin_series.str.lower()
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_str_upper(data):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.upper()
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.upper()
    else:
        modin_result = modin_series.str.upper()
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_str_title(data):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.title()
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.title()
    else:
        modin_result = modin_series.str.title()
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("sub", string_sep_values, ids=string_sep_keys)
@pytest.mark.parametrize("start", int_arg_values, ids=int_arg_keys)
@pytest.mark.parametrize("end", int_arg_values, ids=int_arg_keys)
def test_str_find(data, sub, start, end):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.find(sub, start=start, end=end)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.find(sub, start=start, end=end)
    else:
        modin_result = modin_series.str.find(sub, start=start, end=end)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("sub", string_sep_values, ids=string_sep_keys)
@pytest.mark.parametrize("start", int_arg_values, ids=int_arg_keys)
@pytest.mark.parametrize("end", int_arg_values, ids=int_arg_keys)
def test_str_rfind(data, sub, start, end):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.rfind(sub, start=start, end=end)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.rfind(sub, start=start, end=end)
    else:
        modin_result = modin_series.str.rfind(sub, start=start, end=end)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("sub", string_sep_values, ids=string_sep_keys)
@pytest.mark.parametrize("start", int_arg_values, ids=int_arg_keys)
@pytest.mark.parametrize("end", int_arg_values, ids=int_arg_keys)
def test_str_index(data, sub, start, end):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.index(sub, start=start, end=end)
    except ValueError:
        # pytest does not get the RayGetErrors
        assert True
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.index(sub, start=start, end=end)
    else:
        modin_result = modin_series.str.index(sub, start=start, end=end)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("sub", string_sep_values, ids=string_sep_keys)
@pytest.mark.parametrize("start", int_arg_values, ids=int_arg_keys)
@pytest.mark.parametrize("end", int_arg_values, ids=int_arg_keys)
def test_str_rindex(data, sub, start, end):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.rindex(sub, start=start, end=end)
    except ValueError:
        # pytest does not get the RayGetErrors
        assert True
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.rindex(sub, start=start, end=end)
    else:
        modin_result = modin_series.str.rindex(sub, start=start, end=end)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_str_capitalize(data):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.capitalize()
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.capitalize()
    else:
        modin_result = modin_series.str.capitalize()
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_str_swapcase(data):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.swapcase()
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.swapcase()
    else:
        modin_result = modin_series.str.swapcase()
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize(
    "form", ["NFC", "NFKC", "NFD", "NFKD"], ids=["NFC", "NFKC", "NFD", "NFKD"]
)
def test_str_normalize(data, form):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.normalize(form)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.normalize(form)
    else:
        modin_result = modin_series.str.normalize(form)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("pat", string_sep_values, ids=string_sep_keys)
def test_str_translate(data, pat):
    modin_series, pandas_series = create_test_series(data)

    # Test none table
    try:
        pandas_result = pandas_series.str.translate(None)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.translate(None)
    else:
        modin_result = modin_series.str.translate(None)
        df_equals(modin_result, pandas_result)

    # Translation dictionary
    table = {pat: "DDD"}
    try:
        pandas_result = pandas_series.str.translate(table)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.translate(table)
    else:
        modin_result = modin_series.str.translate(table)
        df_equals(modin_result, pandas_result)

    # Translation table with maketrans (python3 only)
    if pat is not None:
        table = str.maketrans(pat, "d" * len(pat))
        try:
            pandas_result = pandas_series.str.translate(table)
        except Exception as e:
            with pytest.raises(type(e)):
                modin_series.str.translate(table)
        else:
            modin_result = modin_series.str.translate(table)
            df_equals(modin_result, pandas_result)

    # Test delete chars
    deletechars = "|"
    try:
        pandas_result = pandas_series.str.translate(table, deletechars)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.translate(table, deletechars)
    else:
        modin_result = modin_series.str.translate(table, deletechars)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_str_isalnum(data):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.isalnum()
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.isalnum()
    else:
        modin_result = modin_series.str.isalnum()
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_str_isalpha(data):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.isalpha()
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.isalpha()
    else:
        modin_result = modin_series.str.isalpha()
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_str_isdigit(data):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.isdigit()
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.isdigit()
    else:
        modin_result = modin_series.str.isdigit()
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_str_isspace(data):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.isspace()
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.isspace()
    else:
        modin_result = modin_series.str.isspace()
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_str_islower(data):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.islower()
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.islower()
    else:
        modin_result = modin_series.str.islower()
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_str_isupper(data):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.isupper()
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.isupper()
    else:
        modin_result = modin_series.str.isupper()
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_str_istitle(data):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.istitle()
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.istitle()
    else:
        modin_result = modin_series.str.istitle()
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_str_isnumeric(data):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.isnumeric()
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.isnumeric()
    else:
        modin_result = modin_series.str.isnumeric()
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_str_isdecimal(data):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.isdecimal()
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.isdecimal()
    else:
        modin_result = modin_series.str.isdecimal()
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_casefold(data):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.casefold()
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.casefold()
    else:
        modin_result = modin_series.str.casefold()
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("encoding_type", encoding_types)
@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_encode(data, encoding_type):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.str.encode(encoding=encoding_type)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_series.str.encode(encoding=encoding_type)
    else:
        modin_result = modin_series.str.encode(encoding=encoding_type)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "is_sparse_data", [True, False], ids=["is_sparse", "is_not_sparse"]
)
def test_hasattr_sparse(is_sparse_data):
    modin_df, pandas_df = (
        create_test_series(
            pandas.arrays.SparseArray(test_data["float_nan_data"].values())
        )
        if is_sparse_data
        else create_test_series(test_data["float_nan_data"])
    )
    eval_general(modin_df, pandas_df, lambda df: hasattr(df, "sparse"))


@pytest.mark.parametrize(
    "data", test_data_categorical_values, ids=test_data_categorical_keys
)
def test_cat_categories(data):
    modin_series, pandas_series = create_test_series(data.copy())
    df_equals(modin_series.cat.categories, pandas_series.cat.categories)
    pandas_series.cat.categories = list("qwert")
    modin_series.cat.categories = list("qwert")
    df_equals(modin_series, pandas_series)


@pytest.mark.parametrize(
    "data", test_data_categorical_values, ids=test_data_categorical_keys
)
def test_cat_ordered(data):
    modin_series, pandas_series = create_test_series(data.copy())
    assert modin_series.cat.ordered == pandas_series.cat.ordered


@pytest.mark.parametrize(
    "data", test_data_categorical_values, ids=test_data_categorical_keys
)
def test_cat_codes(data):
    modin_series, pandas_series = create_test_series(data.copy())
    pandas_result = pandas_series.cat.codes
    modin_result = modin_series.cat.codes
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "data", test_data_categorical_values, ids=test_data_categorical_keys
)
@pytest.mark.parametrize("inplace", [True, False])
def test_cat_rename_categories(data, inplace):
    modin_series, pandas_series = create_test_series(data.copy())
    pandas_result = pandas_series.cat.rename_categories(list("qwert"), inplace=inplace)
    modin_result = modin_series.cat.rename_categories(list("qwert"), inplace=inplace)
    df_equals(modin_series, pandas_series)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "data", test_data_categorical_values, ids=test_data_categorical_keys
)
@pytest.mark.parametrize("ordered", bool_arg_values, ids=bool_arg_keys)
@pytest.mark.parametrize("inplace", [True, False])
def test_cat_reorder_categories(data, ordered, inplace):
    modin_series, pandas_series = create_test_series(data.copy())
    pandas_result = pandas_series.cat.reorder_categories(
        list("tades"), ordered=ordered, inplace=inplace
    )
    modin_result = modin_series.cat.reorder_categories(
        list("tades"), ordered=ordered, inplace=inplace
    )
    df_equals(modin_series, pandas_series)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "data", test_data_categorical_values, ids=test_data_categorical_keys
)
@pytest.mark.parametrize("inplace", [True, False])
def test_cat_add_categories(data, inplace):
    modin_series, pandas_series = create_test_series(data.copy())
    pandas_result = pandas_series.cat.add_categories(list("qw"), inplace=inplace)
    modin_result = modin_series.cat.add_categories(list("qw"), inplace=inplace)
    df_equals(modin_series, pandas_series)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "data", test_data_categorical_values, ids=test_data_categorical_keys
)
@pytest.mark.parametrize("inplace", [True, False])
def test_cat_remove_categories(data, inplace):
    modin_series, pandas_series = create_test_series(data.copy())
    pandas_result = pandas_series.cat.remove_categories(list("at"), inplace=inplace)
    modin_result = modin_series.cat.remove_categories(list("at"), inplace=inplace)
    df_equals(modin_series, pandas_series)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "data", test_data_categorical_values, ids=test_data_categorical_keys
)
@pytest.mark.parametrize("inplace", [True, False])
def test_cat_remove_unused_categories(data, inplace):
    modin_series, pandas_series = create_test_series(data.copy())
    pandas_series[1] = np.nan
    pandas_result = pandas_series.cat.remove_unused_categories(inplace=inplace)
    modin_series[1] = np.nan
    modin_result = modin_series.cat.remove_unused_categories(inplace=inplace)
    df_equals(modin_series, pandas_series)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "data", test_data_categorical_values, ids=test_data_categorical_keys
)
@pytest.mark.parametrize("ordered", bool_arg_values, ids=bool_arg_keys)
@pytest.mark.parametrize("rename", [True, False])
@pytest.mark.parametrize("inplace", [True, False])
def test_cat_set_categories(data, ordered, rename, inplace):
    modin_series, pandas_series = create_test_series(data.copy())
    pandas_result = pandas_series.cat.set_categories(
        list("qwert"), ordered=ordered, rename=rename, inplace=inplace
    )
    modin_result = modin_series.cat.set_categories(
        list("qwert"), ordered=ordered, rename=rename, inplace=inplace
    )
    df_equals(modin_series, pandas_series)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "data", test_data_categorical_values, ids=test_data_categorical_keys
)
@pytest.mark.parametrize("inplace", [True, False])
def test_cat_as_ordered(data, inplace):
    modin_series, pandas_series = create_test_series(data.copy())
    pandas_result = pandas_series.cat.as_ordered(inplace=inplace)
    modin_result = modin_series.cat.as_ordered(inplace=inplace)
    df_equals(modin_series, pandas_series)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "data", test_data_categorical_values, ids=test_data_categorical_keys
)
@pytest.mark.parametrize("inplace", [True, False])
def test_cat_as_unordered(data, inplace):
    modin_series, pandas_series = create_test_series(data.copy())
    pandas_result = pandas_series.cat.as_unordered(inplace=inplace)
    modin_result = modin_series.cat.as_unordered(inplace=inplace)
    df_equals(modin_series, pandas_series)
    df_equals(modin_result, pandas_result)
