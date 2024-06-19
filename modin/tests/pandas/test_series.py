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

from __future__ import annotations

import datetime
import itertools
import json
import sys
import unittest.mock as mock
import warnings

import matplotlib
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
from numpy.testing import assert_array_equal
from pandas.core.indexing import IndexingError
from pandas.errors import PerformanceWarning, SpecificationError

import modin.pandas as pd
from modin.config import Engine, NPartitions, StorageFormat
from modin.pandas.io import to_pandas
from modin.tests.core.storage_formats.pandas.test_internals import (
    construct_modin_df_by_scheme,
)
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution, try_cast_to_pandas

from .utils import (
    RAND_HIGH,
    RAND_LOW,
    CustomIntegerForAddition,
    NonCommutativeMultiplyInteger,
    agg_func_except_keys,
    agg_func_except_values,
    agg_func_keys,
    agg_func_values,
    arg_keys,
    bool_arg_keys,
    bool_arg_values,
    categories_equals,
    create_test_dfs,
    create_test_series,
    default_to_pandas_ignore_string,
    df_equals,
    df_equals_with_non_stable_indices,
    encoding_types,
    eval_general,
    generate_multiindex,
    int_arg_keys,
    int_arg_values,
    name_contains,
    no_numeric_dfs,
    numeric_dfs,
    quantiles_keys,
    quantiles_values,
    random_state,
    sort_if_range_partitioning,
    string_na_rep_keys,
    string_na_rep_values,
    string_sep_keys,
    string_sep_values,
    test_data,
    test_data_categorical_keys,
    test_data_categorical_values,
    test_data_diff_dtype,
    test_data_keys,
    test_data_large_categorical_series_keys,
    test_data_large_categorical_series_values,
    test_data_small_keys,
    test_data_small_values,
    test_data_values,
    test_data_with_duplicates_keys,
    test_data_with_duplicates_values,
    test_string_data_keys,
    test_string_data_values,
    test_string_list_data_keys,
    test_string_list_data_values,
)

# Our configuration in pytest.ini requires that we explicitly catch all
# instances of defaulting to pandas, but some test modules, like this one,
# have too many such instances.
# TODO(https://github.com/modin-project/modin/issues/3655): catch all instances
# of defaulting to pandas.
pytestmark = [
    pytest.mark.filterwarnings(default_to_pandas_ignore_string),
    # IGNORE FUTUREWARNINGS MARKS TO CLEANUP OUTPUT
    pytest.mark.filterwarnings(
        "ignore:.*bool is now deprecated and will be removed:FutureWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:first is deprecated and will be removed:FutureWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:last is deprecated and will be removed:FutureWarning"
    ),
]

NPartitions.put(4)

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")

# Initialize the environment
pd.DataFrame()


def get_rop(op):
    if op.startswith("__") and op.endswith("__"):
        return "__r" + op[2:]
    else:
        return None


def inter_df_math_helper(
    modin_series, pandas_series, op, comparator_kwargs=None, expected_exception=None
):
    inter_df_math_helper_one_side(
        modin_series, pandas_series, op, comparator_kwargs, expected_exception
    )
    rop = get_rop(op)
    if rop:
        inter_df_math_helper_one_side(
            modin_series, pandas_series, rop, comparator_kwargs, expected_exception
        )


def inter_df_math_helper_one_side(
    modin_series,
    pandas_series,
    op,
    comparator_kwargs=None,
    expected_exception=None,
):
    if comparator_kwargs is None:
        comparator_kwargs = {}

    try:
        pandas_attr = getattr(pandas_series, op)
    except Exception as err:
        with pytest.raises(type(err)):
            _ = getattr(modin_series, op)
        return
    modin_attr = getattr(modin_series, op)

    try:
        pandas_result = pandas_attr(4)
    except Exception as err:
        with pytest.raises(type(err)):
            try_cast_to_pandas(modin_attr(4))  # force materialization
    else:
        modin_result = modin_attr(4)
        df_equals(modin_result, pandas_result, **comparator_kwargs)

    try:
        pandas_result = pandas_attr(4.0)
    except Exception as err:
        with pytest.raises(type(err)):
            try_cast_to_pandas(modin_attr(4.0))  # force materialization
    else:
        modin_result = modin_attr(4.0)
        df_equals(modin_result, pandas_result, **comparator_kwargs)

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

    eval_general(
        modin_series,
        pandas_series,
        lambda df: (pandas_attr if isinstance(df, pandas.Series) else modin_attr)(df),
        comparator_kwargs=comparator_kwargs,
        expected_exception=expected_exception,
    )

    list_test = random_state.randint(RAND_LOW, RAND_HIGH, size=(modin_series.shape[0]))
    try:
        pandas_result = pandas_attr(list_test)
    except Exception as err:
        with pytest.raises(type(err)):
            try_cast_to_pandas(modin_attr(list_test))  # force materialization
    else:
        modin_result = modin_attr(list_test)
        df_equals(modin_result, pandas_result, **comparator_kwargs)

    series_test_modin = pd.Series(list_test, index=modin_series.index)
    series_test_pandas = pandas.Series(list_test, index=pandas_series.index)

    eval_general(
        series_test_modin,
        series_test_pandas,
        lambda df: (pandas_attr if isinstance(df, pandas.Series) else modin_attr)(df),
        comparator_kwargs=comparator_kwargs,
        expected_exception=expected_exception,
    )

    # Level test
    new_idx = pandas.MultiIndex.from_tuples(
        [(i // 4, i // 2, i) for i in modin_series.index]
    )
    modin_df_multi_level = modin_series.copy()
    modin_df_multi_level.index = new_idx
    # When 'level' parameter is passed, modin's implementation must raise a default-to-pandas warning,
    # here we first detect whether 'op' takes 'level' parameter at all and only then perform the warning check
    # reasoning: https://github.com/modin-project/modin/issues/6893
    try:
        getattr(modin_df_multi_level, op)(modin_df_multi_level, level=1)
    except TypeError:
        # Operation doesn't support 'level' parameter
        pass
    else:
        # Operation supports 'level' parameter, so it makes sense to check for a warning
        with warns_that_defaulting_to_pandas():
            getattr(modin_df_multi_level, op)(modin_df_multi_level, level=1)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_to_frame(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.to_frame(name="miao"), pandas_series.to_frame(name="miao"))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_to_list(data):
    modin_series, pandas_series = create_test_series(data)
    pd_res = pandas_series.to_list()
    md_res = modin_series.to_list()
    assert type(pd_res) is type(md_res)
    assert np.array_equal(pd_res, md_res, equal_nan=True)


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
def test___and__(data, request):
    modin_series, pandas_series = create_test_series(data)
    expected_exception = None
    if "float_nan_data" in request.node.callspec.id:
        # FIXME: https://github.com/modin-project/modin/issues/7037
        expected_exception = False
    inter_df_math_helper(
        modin_series,
        pandas_series,
        "__and__",
        # https://github.com/modin-project/modin/issues/5966
        comparator_kwargs={"check_dtypes": False},
        expected_exception=expected_exception,
    )


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
    except Exception as err:
        with pytest.raises(type(err)):
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
    df_equals(modin_series[::2], pandas_series[::2])
    # Test getting an invalid string key
    # FIXME: https://github.com/modin-project/modin/issues/7038
    eval_general(
        modin_series, pandas_series, lambda s: s["a"], expected_exception=False
    )
    eval_general(
        modin_series, pandas_series, lambda s: s[["a"]], expected_exception=False
    )

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


@pytest.mark.parametrize("count_elements", [0, 1, 10])
def test___int__(count_elements):
    expected_exception = None
    if count_elements != 1:
        expected_exception = TypeError("cannot convert the series to <class 'int'>")
    eval_general(
        *create_test_series([1.5] * count_elements),
        int,
        expected_exception=expected_exception,
    )


@pytest.mark.parametrize("count_elements", [0, 1, 10])
def test___float__(count_elements):
    expected_exception = None
    if count_elements != 1:
        expected_exception = TypeError("cannot convert the series to <class 'float'>")
    eval_general(
        *create_test_series([1] * count_elements),
        float,
        expected_exception=expected_exception,
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___invert__(data, request):
    modin_series, pandas_series = create_test_series(data)
    expected_exception = None
    if "float_nan_data" in request.node.callspec.id:
        # FIXME: https://github.com/modin-project/modin/issues/7081
        expected_exception = False
    eval_general(
        modin_series,
        pandas_series,
        lambda ser: ser.__invert__(),
        expected_exception=expected_exception,
    )


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
    except Exception as err:
        with pytest.raises(type(err)):
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
def test___neg__(data):
    modin_series, pandas_series = create_test_series(data)
    eval_general(modin_series, pandas_series, lambda ser: ser.__neg__())


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test___or__(data, request):
    modin_series, pandas_series = create_test_series(data)
    expected_exception = None
    if "float_nan_data" in request.node.callspec.id:
        # FIXME: https://github.com/modin-project/modin/issues/7081
        expected_exception = False
    inter_df_math_helper(
        modin_series,
        pandas_series,
        "__or__",
        # https://github.com/modin-project/modin/issues/5966
        comparator_kwargs={"check_dtypes": False},
        expected_exception=expected_exception,
    )


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
    [*test_data_values, "empty"],
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
            "1/1/2000", periods=len(pandas_series.index), freq="min"
        )
        pandas_series.index = modin_series.index = index

    assert repr(modin_series) == repr(pandas_series)


def test___repr__4186():
    modin_series, pandas_series = create_test_series(
        ["a", "b", "c", "a"], dtype="category"
    )
    assert repr(modin_series) == repr(pandas_series)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.exclude_in_sanity
def test___round__(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(round(modin_series), round(pandas_series))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.exclude_in_sanity
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
    with warns_that_defaulting_to_pandas():
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
def test___xor__(data, request):
    modin_series, pandas_series = create_test_series(data)
    expected_exception = None
    if "float_nan_data" in request.node.callspec.id:
        # FIXME: https://github.com/modin-project/modin/issues/7081
        expected_exception = False
    inter_df_math_helper(
        modin_series,
        pandas_series,
        "__xor__",
        # https://github.com/modin-project/modin/issues/5966
        comparator_kwargs={"check_dtypes": False},
        expected_exception=expected_exception,
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_abs(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.abs(), pandas_series.abs())


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_add(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "add")


def test_add_does_not_change_original_series_name():
    # See https://github.com/modin-project/modin/issues/5232
    s1 = pd.Series(1, name=1)
    s2 = pd.Series(2, name=2)
    original_s1 = s1.copy(deep=True)
    original_s2 = s2.copy(deep=True)
    _ = s1 + s2
    df_equals(s1, original_s1)
    df_equals(s2, original_s2)


@pytest.mark.parametrize("axis", [None, 0, 1])
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_add_prefix(data, axis):
    expected_exception = None
    if axis:
        expected_exception = ValueError("No axis named 1 for object type Series")
    eval_general(
        *create_test_series(data),
        lambda df: df.add_prefix("PREFIX_ADD_", axis=axis),
        expected_exception=expected_exception,
    )


@pytest.mark.parametrize("axis", [None, 0, 1])
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_add_suffix(data, axis):
    expected_exception = None
    if axis:
        expected_exception = ValueError("No axis named 1 for object type Series")
    eval_general(
        *create_test_series(data),
        lambda df: df.add_suffix("SUFFIX_ADD_", axis=axis),
        expected_exception=expected_exception,
    )


def test_add_custom_class():
    # see https://github.com/modin-project/modin/issues/5236
    # Test that we can add any object that is addable to pandas object data
    # via "+".
    eval_general(
        *create_test_series(test_data["int_data"]),
        lambda df: df + CustomIntegerForAddition(4),
    )


def test_aggregate_alias():
    # It's optimization. If failed, Series.agg should be tested explicitly
    assert pd.Series.aggregate == pd.Series.agg


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
def test_aggregate(data, func, request):
    expected_exception = None
    if "should raise AssertionError" in request.node.callspec.id:
        # FIXME: https://github.com/modin-project/modin/issues/7031
        expected_exception = False
    eval_general(
        *create_test_series(data),
        lambda df: df.aggregate(func),
        expected_exception=expected_exception,
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("func", agg_func_except_values, ids=agg_func_except_keys)
def test_aggregate_except(data, func):
    # SpecificationError is arisen because we treat a Series as a DataFrame.
    # See details in pandas issues 36036.
    with pytest.raises(SpecificationError):
        eval_general(
            *create_test_series(data),
            lambda df: df.aggregate(func),
        )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_aggregate_error_checking(data):
    modin_series, pandas_series = create_test_series(data)

    assert pandas_series.aggregate("ndim") == 1
    assert modin_series.aggregate("ndim") == 1

    eval_general(
        modin_series,
        pandas_series,
        lambda series: series.aggregate("cumprod"),
    )
    eval_general(
        modin_series,
        pandas_series,
        lambda series: series.aggregate("NOT_EXISTS"),
        expected_exception=AttributeError(
            "'NOT_EXISTS' is not a valid function for 'Series' object"
        ),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_align(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with warns_that_defaulting_to_pandas():
        modin_series.align(modin_series)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("skipna", [False, True])
def test_all(data, skipna):
    eval_general(*create_test_series(data), lambda df: df.all(skipna=skipna))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("skipna", [False, True])
def test_any(data, skipna):
    eval_general(*create_test_series(data), lambda df: df.any(skipna=skipna))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_append(data):
    modin_series, pandas_series = create_test_series(data)

    data_to_append = {"append_a": 2, "append_b": 1000}

    ignore_idx_values = [True, False]

    for ignore in ignore_idx_values:
        try:
            pandas_result = pandas_series.append(data_to_append, ignore_index=ignore)
        except Exception as err:
            with pytest.raises(type(err)):
                modin_series.append(data_to_append, ignore_index=ignore)
        else:
            modin_result = modin_series.append(data_to_append, ignore_index=ignore)
            df_equals(modin_result, pandas_result)

    try:
        pandas_result = pandas_series.append(pandas_series.iloc[-1])
    except Exception as err:
        with pytest.raises(type(err)):
            modin_series.append(modin_series.iloc[-1])
    else:
        modin_result = modin_series.append(modin_series.iloc[-1])
        df_equals(modin_result, pandas_result)

    try:
        pandas_result = pandas_series.append([pandas_series.iloc[-1]])
    except Exception as err:
        with pytest.raises(type(err)):
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
        except Exception as err:
            with pytest.raises(type(err)):
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
        except Exception as err:
            with pytest.raises(type(err)):
                modin_series.append(modin_series, verify_integrity=verify_integrity)
        else:
            modin_result = modin_series.append(
                modin_series, verify_integrity=verify_integrity
            )
            df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("func", agg_func_values, ids=agg_func_keys)
def test_apply(data, func, request):
    expected_exception = None
    if "should raise AssertionError" in request.node.callspec.id:
        # FIXME: https://github.com/modin-project/modin/issues/7031
        expected_exception = False
    elif "df sum" in request.node.callspec.id:
        _type = "int" if "int_data" in request.node.callspec.id else "float"
        expected_exception = AttributeError(f"'{_type}' object has no attribute 'sum'")
    eval_general(
        *create_test_series(data),
        lambda df: df.apply(func),
        expected_exception=expected_exception,
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("func", agg_func_except_values, ids=agg_func_except_keys)
def test_apply_except(data, func):
    eval_general(
        *create_test_series(data),
        lambda df: df.apply(func),
        expected_exception=pandas.errors.SpecificationError(
            "Function names must be unique if there is no new column names assigned"
        ),
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


@pytest.mark.parametrize("axis", [None, 0, 1])
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("func", ["count", "all", "kurt", "array", "searchsorted"])
def test_apply_text_func(data, func, axis):
    func_kwargs = {}
    if func not in ("count", "searchsorted"):
        func_kwargs["axis"] = axis
    elif not axis:
        # FIXME: https://github.com/modin-project/modin/issues/7000
        return
    rows_number = len(next(iter(data.values())))  # length of the first data column
    level_0 = np.random.choice([0, 1, 2], rows_number)
    level_1 = np.random.choice([3, 4, 5], rows_number)
    index = pd.MultiIndex.from_arrays([level_0, level_1])

    modin_series, pandas_series = create_test_series(data)
    modin_series.index = index
    pandas_series.index = index

    if func == "searchsorted":
        # required parameter
        func_kwargs["value"] = pandas_series[1]

    eval_general(modin_series, pandas_series, lambda df: df.apply(func, **func_kwargs))


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
    with warns_that_defaulting_to_pandas():
        modin_result = modin_series.argsort()
    df_equals(modin_result, pandas_series.argsort())


def test_asfreq():
    index = pd.date_range("1/1/2000", periods=4, freq="min")
    series = pd.Series([0.0, None, 2.0, 3.0], index=index)
    with warns_that_defaulting_to_pandas():
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
    modin_series, pandas_series = (
        pd.Series(values, index=index),
        pandas.Series(values, index=index),
    )
    df_equals(modin_series.asof(where), pandas_series.asof(where))

    # No NaN:
    values = [1, 2, 7, 4]
    modin_series, pandas_series = (
        pd.Series(values, index=index),
        pandas.Series(values, index=index),
    )
    df_equals(modin_series.asof(where), pandas_series.asof(where))


@pytest.mark.parametrize(
    "where",
    [20, 30, [10.5, 40.5], [10], pandas.Index([20, 30]), pandas.Index([10.5])],
)
def test_asof_large(where):
    values = test_data["float_nan_data"]["col1"]
    index = list(range(len(values)))
    modin_series, pandas_series = (
        pd.Series(values, index=index),
        pandas.Series(values, index=index),
    )
    df_equals(modin_series.asof(where), pandas_series.asof(where))


@pytest.mark.parametrize(
    "data",
    [
        test_data["int_data"],
        test_data["float_nan_data"],
    ],
    ids=test_data_keys,
)
def test_astype(data, request):
    modin_series, pandas_series = create_test_series(data)
    series_name = "test_series"
    modin_series.name = pandas_series.name = series_name

    eval_general(modin_series, pandas_series, lambda df: df.astype(str))
    expected_exception = None
    if "float_nan_data" in request.node.callspec.id:
        expected_exception = pd.errors.IntCastingNaNError(
            "Cannot convert non-finite values (NA or inf) to integer"
        )
    eval_general(
        modin_series,
        pandas_series,
        lambda ser: ser.astype(np.int64),
        expected_exception=expected_exception,
    )
    eval_general(modin_series, pandas_series, lambda ser: ser.astype(np.float64))
    eval_general(
        modin_series, pandas_series, lambda ser: ser.astype({series_name: str})
    )
    # FIXME: https://github.com/modin-project/modin/issues/7039
    eval_general(
        modin_series,
        pandas_series,
        lambda ser: ser.astype({"wrong_name": str}),
        expected_exception=False,
    )

    # TODO(https://github.com/modin-project/modin/issues/4317): Test passing a
    # dict to astype() for a series with no name.


@pytest.mark.parametrize("dtype", ["int32", "float32"])
def test_astype_32_types(dtype):
    # https://github.com/modin-project/modin/issues/6881
    assert pd.Series([1, 2, 6]).astype(dtype).dtype == dtype


@pytest.mark.parametrize(
    "data", [["A", "A", "B", "B", "A"], [1, 1, 2, 1, 2, 2, 3, 1, 2, 1, 2]]
)
def test_astype_categorical(data):
    modin_df, pandas_df = create_test_series(data)

    modin_result = modin_df.astype("category")
    pandas_result = pandas_df.astype("category")
    df_equals(modin_result, pandas_result)
    assert modin_result.dtype == pandas_result.dtype


@pytest.mark.parametrize("data", [["a", "a", "b", "c", "c", "d", "b", "d"]])
@pytest.mark.parametrize(
    "set_min_row_partition_size",
    [2, 4],
    ids=["four_row_partitions", "two_row_partitions"],
    indirect=True,
)
def test_astype_categorical_issue5722(data, set_min_row_partition_size):
    modin_series, pandas_series = create_test_series(data)

    modin_result = modin_series.astype("category")
    pandas_result = pandas_series.astype("category")
    df_equals(modin_result, pandas_result)
    assert modin_result.dtype == pandas_result.dtype

    pandas_result1, pandas_result2 = pandas_result.iloc[:4], pandas_result.iloc[4:]
    modin_result1, modin_result2 = modin_result.iloc[:4], modin_result.iloc[4:]

    # check categories
    assert pandas_result1.cat.categories.equals(pandas_result2.cat.categories)
    assert modin_result1.cat.categories.equals(modin_result2.cat.categories)
    assert pandas_result1.cat.categories.equals(modin_result1.cat.categories)
    assert pandas_result2.cat.categories.equals(modin_result2.cat.categories)

    # check codes
    assert_array_equal(pandas_result1.cat.codes.values, modin_result1.cat.codes.values)
    assert_array_equal(pandas_result2.cat.codes.values, modin_result2.cat.codes.values)


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


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_between(data):
    modin_series, pandas_series = create_test_series(data)

    df_equals(
        modin_series.between(1, 4),
        pandas_series.between(1, 4),
    )


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
        modin_series.between_time("3:00", "8:00", inclusive="right"),
        pandas_series.between_time("3:00", "8:00", inclusive="right"),
    )


def test_add_series_to_timedeltaindex():
    # Make a pandas.core.indexes.timedeltas.TimedeltaIndex
    deltas = pd.to_timedelta([1], unit="h")
    test_series = create_test_series(np.datetime64("2000-12-12"))
    eval_general(*test_series, lambda s: s + deltas)
    eval_general(*test_series, lambda s: s - deltas)


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
    modin_series, _ = create_test_series(data)

    with pytest.warns(
        FutureWarning, match="bool is now deprecated and will be removed"
    ):
        with pytest.raises(ValueError):
            modin_series.bool()
    with pytest.raises(ValueError):
        modin_series.__bool__()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("bound_type", ["list", "series"], ids=["list", "series"])
def test_clip_scalar(request, data, bound_type):
    modin_series, pandas_series = create_test_series(
        data,
    )

    if name_contains(request.node.name, numeric_dfs):
        # set bounds
        lower, upper = np.sort(random_state.randint(RAND_LOW, RAND_HIGH, 2))

        # test only upper scalar bound
        modin_result = modin_series.clip(None, upper)
        pandas_result = pandas_series.clip(None, upper)
        df_equals(modin_result, pandas_result)

        # test lower and upper scalar bound
        modin_result = modin_series.clip(lower, upper)
        pandas_result = pandas_series.clip(lower, upper)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("bound_type", ["list", "series"], ids=["list", "series"])
def test_clip_sequence(request, data, bound_type):
    modin_series, pandas_series = create_test_series(
        data,
    )

    if name_contains(request.node.name, numeric_dfs):
        lower = random_state.randint(RAND_LOW, RAND_HIGH, len(pandas_series))
        upper = random_state.randint(RAND_LOW, RAND_HIGH, len(pandas_series))

        if bound_type == "series":
            modin_lower = pd.Series(lower)
            pandas_lower = pandas.Series(lower)
            modin_upper = pd.Series(upper)
            pandas_upper = pandas.Series(upper)
        else:
            modin_lower = pandas_lower = lower
            modin_upper = pandas_upper = upper

        # test lower and upper list bound
        modin_result = modin_series.clip(modin_lower, modin_upper, axis=0)
        pandas_result = pandas_series.clip(pandas_lower, pandas_upper)
        df_equals(modin_result, pandas_result)

        # test only upper list bound
        modin_result = modin_series.clip(np.nan, modin_upper, axis=0)
        pandas_result = pandas_series.clip(np.nan, pandas_upper)
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
    except Exception as err:
        with pytest.raises(type(err)):
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


def test_constructor_arrow_extension_array():
    # example from pandas docs
    pa = pytest.importorskip("pyarrow")
    array = pd.arrays.ArrowExtensionArray(
        pa.array(
            [{"1": "2"}, {"10": "20"}, None],
            type=pa.map_(pa.string(), pa.string()),
        )
    )
    md_ser, pd_ser = create_test_series(array)
    df_equals(md_ser, pd_ser)
    df_equals(md_ser.dtypes, pd_ser.dtypes)


def test_pyarrow_backed_constructor():
    pa = pytest.importorskip("pyarrow")
    data = list("abcd")
    df_equals(*create_test_series(data, dtype="string[pyarrow]"))
    df_equals(*create_test_series(data, dtype=pd.ArrowDtype(pa.string())))

    data = [["hello"], ["there"]]
    list_str_type = pa.list_(pa.string())
    df_equals(*create_test_series(data, dtype=pd.ArrowDtype(list_str_type)))


def test_pyarrow_backed_functions():
    pytest.importorskip("pyarrow")
    modin_series, pandas_series = create_test_series(
        [-1.545, 0.211, None], dtype="float32[pyarrow]"
    )
    df_equals(modin_series.mean(), pandas_series.mean())

    def comparator(df1, df2):
        df_equals(df1, df2)
        df_equals(df1.dtypes, df2.dtypes)

    eval_general(
        modin_series,
        pandas_series,
        lambda ser: ser
        + (modin_series if isinstance(ser, pd.Series) else pandas_series),
        comparator=comparator,
    )

    eval_general(
        modin_series,
        pandas_series,
        lambda ser: ser > (ser + 1),
        comparator=comparator,
    )

    eval_general(
        modin_series,
        pandas_series,
        lambda ser: ser.dropna(),
        comparator=comparator,
    )

    eval_general(
        modin_series,
        pandas_series,
        lambda ser: ser.isna(),
        comparator=comparator,
    )

    eval_general(
        modin_series,
        pandas_series,
        lambda ser: ser.fillna(0),
        comparator=comparator,
    )


def test_pyarrow_array_retrieve():
    pa = pytest.importorskip("pyarrow")
    modin_series, pandas_series = create_test_series(
        [1, 2, None], dtype="uint8[pyarrow]"
    )
    eval_general(
        modin_series,
        pandas_series,
        lambda ser: pa.array(ser),
    )


def test___arrow_array__():
    # https://github.com/modin-project/modin/issues/6808
    pa = pytest.importorskip("pyarrow")
    mpd_df_1 = pd.DataFrame({"a": ["1", "2", "3"], "b": ["4", "5", "6"]})
    mpd_df_2 = pd.DataFrame({"a": ["7", "8", "9"], "b": ["10", "11", "12"]})
    test_df = pd.concat([mpd_df_1, mpd_df_2])

    res_from_md = pa.Table.from_pandas(df=test_df)
    res_from_pd = pa.Table.from_pandas(df=test_df._to_pandas())
    assert res_from_md.equals(res_from_pd)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_copy(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series, modin_series.copy())
    df_equals(modin_series.copy(), pandas_series)
    df_equals(modin_series.copy(), pandas_series.copy())


def test_copy_empty_series():
    ser = pd.Series(range(3))
    res = ser[:0].copy()
    assert res.dtype == ser.dtype


@pytest.mark.parametrize("method", ["pearson", "kendall"])
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_corr(data, method):
    modin_series, pandas_series = create_test_series(data)
    modin_result = modin_series.corr(modin_series, method=method)
    pandas_result = pandas_series.corr(pandas_series, method=method)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "data",
    test_data_values + test_data_large_categorical_series_values,
    ids=test_data_keys + test_data_large_categorical_series_keys,
)
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
@pytest.mark.parametrize("skipna", [False, True])
def test_cummax(data, skipna):
    modin_series, pandas_series = create_test_series(data)
    try:
        pandas_result = pandas_series.cummax(skipna=skipna)
    except Exception as err:
        with pytest.raises(type(err)):
            modin_series.cummax(skipna=skipna)
    else:
        df_equals(modin_series.cummax(skipna=skipna), pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("skipna", [False, True])
def test_cummin(data, skipna):
    modin_series, pandas_series = create_test_series(data)
    try:
        pandas_result = pandas_series.cummin(skipna=skipna)
    except Exception as err:
        with pytest.raises(type(err)):
            modin_series.cummin(skipna=skipna)
    else:
        df_equals(modin_series.cummin(skipna=skipna), pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("skipna", [False, True])
def test_cumprod(data, skipna):
    modin_series, pandas_series = create_test_series(data)
    try:
        pandas_result = pandas_series.cumprod(skipna=skipna)
    except Exception as err:
        with pytest.raises(type(err)):
            modin_series.cumprod(skipna=skipna)
    else:
        df_equals(modin_series.cumprod(skipna=skipna), pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("skipna", [False, True])
def test_cumsum(data, skipna):
    modin_series, pandas_series = create_test_series(data)
    try:
        pandas_result = pandas_series.cumsum(skipna=skipna)
    except Exception as err:
        with pytest.raises(type(err)):
            modin_series.cumsum(skipna=skipna)
    else:
        df_equals(modin_series.cumsum(skipna=skipna), pandas_result)


def test_cumsum_6771():
    _ = to_pandas(pd.Series([1, 2, 3], dtype="Int64").cumsum())


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
    except Exception as err:
        with pytest.raises(type(err)):
            modin_series.describe(exclude=[np.float64])
    else:
        modin_result = modin_series.describe(exclude=[np.float64])
        df_equals(modin_result, pandas_result)

    try:
        pandas_result = pandas_series.describe(exclude=np.float64)
    except Exception as err:
        with pytest.raises(type(err)):
            modin_series.describe(exclude=np.float64)
    else:
        modin_result = modin_series.describe(exclude=np.float64)
        df_equals(modin_result, pandas_result)

    try:
        pandas_result = pandas_series.describe(
            include=[np.timedelta64, np.datetime64, np.object_, np.bool_]
        )
    except Exception as err:
        with pytest.raises(type(err)):
            modin_series.describe(
                include=[np.timedelta64, np.datetime64, np.object_, np.bool_]
            )
    else:
        modin_result = modin_series.describe(
            include=[np.timedelta64, np.datetime64, np.object_, np.bool_]
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
    except Exception as err:
        with pytest.raises(type(err)):
            modin_series.diff(periods=periods)
    else:
        modin_result = modin_series.diff(periods=periods)
        df_equals(modin_result, pandas_result)

    try:
        pandas_result = pandas_series.T.diff(periods=periods)
    except Exception as err:
        with pytest.raises(type(err)):
            modin_series.T.diff(periods=periods)
    else:
        modin_result = modin_series.T.diff(periods=periods)
        df_equals(modin_result, pandas_result)


def test_diff_with_dates():
    data = pandas.date_range("2018-01-01", periods=15, freq="h").values
    pandas_series = pandas.Series(data)
    modin_series = pd.Series(pandas_series)

    # Check that `diff` with datetime types works correctly.
    pandas_result = pandas_series.diff()
    modin_result = modin_series.diff()
    df_equals(modin_result, pandas_result)

    # Check that `diff` with timedelta types works correctly.
    td_pandas_result = pandas_result.diff()
    td_modin_result = modin_result.diff()
    df_equals(td_modin_result, td_pandas_result)


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
        modin_series.dot(np.arange(ind_len + 10))

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
        modin_series.dot(
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
        modin_series @ np.arange(ind_len + 10)

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
        modin_series @ pd.Series(
            np.arange(ind_len), index=["a" for _ in range(len(modin_series.index))]
        )


@pytest.mark.xfail(reason="Using pandas Series.")
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_drop(data):
    modin_series = create_test_series(data)

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
    modin_res = modin_series.drop_duplicates(keep=keep, inplace=inplace)
    pandas_res = pandas_series.drop_duplicates(keep=keep, inplace=inplace)
    if inplace:
        sort_if_range_partitioning(modin_series, pandas_series)
    else:
        sort_if_range_partitioning(modin_res, pandas_res)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("how", ["any", "all"], ids=["any", "all"])
def test_dropna(data, how):
    modin_series, pandas_series = create_test_series(data)
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
    pandas_series.dropna(how="any", inplace=True)
    modin_series.dropna(how="any", inplace=True)
    df_equals(modin_series, pandas_series)


def test_dtype_empty():
    modin_series, pandas_series = pd.Series(), pandas.Series()
    assert modin_series.dtype == pandas_series.dtype


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_dtype(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.dtype, modin_series.dtypes)
    df_equals(modin_series.dtype, pandas_series.dtype)
    df_equals(modin_series.dtype, pandas_series.dtypes)


# Bug https://github.com/modin-project/modin/issues/4436 in
# Series.dt.to_pydatetime is only reproducible when the date range out of which
# the frame is created has timezone None, so that its dtype is datetime64[ns]
# as opposed to, e.g. datetime64[ns, Europe/Berlin]. To reproduce that bug, we
# use timezones None and Europe/Berlin.
@pytest.mark.parametrize(
    "timezone",
    [
        pytest.param(None),
        pytest.param("Europe/Berlin"),
    ],
)
def test_dt(timezone):
    data = pd.date_range("2016-12-31", periods=128, freq="D", tz=timezone)
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
    df_equals(modin_series.dt.dayofweek, pandas_series.dt.dayofweek)
    df_equals(modin_series.dt.day_of_week, pandas_series.dt.day_of_week)
    df_equals(modin_series.dt.weekday, pandas_series.dt.weekday)
    df_equals(modin_series.dt.dayofyear, pandas_series.dt.dayofyear)
    df_equals(modin_series.dt.day_of_year, pandas_series.dt.day_of_year)
    df_equals(modin_series.dt.unit, pandas_series.dt.unit)
    df_equals(modin_series.dt.as_unit("s"), pandas_series.dt.as_unit("s"))
    df_equals(modin_series.dt.isocalendar(), pandas_series.dt.isocalendar())
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
    if timezone:
        df_equals(
            modin_series.dt.tz_convert(tz="Europe/Berlin"),
            pandas_series.dt.tz_convert(tz="Europe/Berlin"),
        )

    df_equals(modin_series.dt.normalize(), pandas_series.dt.normalize())
    df_equals(
        modin_series.dt.strftime("%B %d, %Y, %r"),
        pandas_series.dt.strftime("%B %d, %Y, %r"),
    )
    df_equals(modin_series.dt.round("h"), pandas_series.dt.round("h"))
    df_equals(modin_series.dt.floor("h"), pandas_series.dt.floor("h"))
    df_equals(modin_series.dt.ceil("h"), pandas_series.dt.ceil("h"))
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

    def dt_with_empty_partition(lib):
        # For context, see https://github.com/modin-project/modin/issues/5112
        df = (
            pd.concat(
                [pd.DataFrame([None]), pd.DataFrame([pd.to_timedelta(1)])], axis=1
            )
            .dropna(axis=1)
            .squeeze(1)
        )
        # BaseOnPython had a single partition after the concat, and it
        # maintains that partition after dropna and squeeze. In other execution modes,
        # the series should have two column partitions, one of which is empty.
        if isinstance(df, pd.DataFrame) and get_current_execution() != "BaseOnPython":
            assert df._query_compiler._modin_frame._partitions.shape == (1, 2)
        return df.dt.days

    eval_general(pd, pandas, dt_with_empty_partition)

    if timezone is None:
        data = pd.period_range("2016-12-31", periods=128, freq="D")
        modin_series = pd.Series(data)
        pandas_series = pandas.Series(data)
        df_equals(modin_series.dt.asfreq("min"), pandas_series.dt.asfreq("min"))


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


@pytest.mark.parametrize(
    "series1_data,series2_data,expected_pandas_equals",
    [
        pytest.param([1], [0], False, id="single_unequal_values"),
        pytest.param([None], [None], True, id="single_none_values"),
        pytest.param(
            pandas.Series(1, name="series1"),
            pandas.Series(1, name="series2"),
            True,
            id="different_names",
        ),
        pytest.param(
            pandas.Series([1], index=[1]),
            pandas.Series([1], index=[1.0]),
            True,
            id="different_index_types",
        ),
        pytest.param(
            pandas.Series([1], index=[1]),
            pandas.Series([1], index=[2]),
            False,
            id="different_index_values",
        ),
        pytest.param([1], [1.0], False, id="different_value_types"),
        pytest.param(
            [1, 2],
            [1, 2],
            True,
            id="equal_series_of_length_two",
        ),
        pytest.param(
            [1, 2],
            [1, 3],
            False,
            id="unequal_series_of_length_two",
        ),
        pytest.param(
            [[1, 2]],
            [[1]],
            False,
            id="different_lengths",
        ),
    ],
)
def test_equals(series1_data, series2_data, expected_pandas_equals):
    modin_series1, pandas_df1 = create_test_series(series1_data)
    modin_series2, pandas_df2 = create_test_series(series2_data)

    pandas_equals = pandas_df1.equals(pandas_df2)
    assert pandas_equals == expected_pandas_equals, (
        "Test expected pandas to say the series were"
        + f"{'' if expected_pandas_equals else ' not'} equal, but they were"
        + f"{' not' if expected_pandas_equals else ''} equal."
    )
    assert modin_series1.equals(modin_series2) == pandas_equals
    assert modin_series1.equals(pandas_df2) == pandas_equals


def test_equals_several_partitions():
    modin_series1 = pd.concat([pd.Series([0, 1]), pd.Series([None, 1])])
    modin_series2 = pd.concat([pd.Series([0, 1]), pd.Series([1, None])])
    assert not modin_series1.equals(modin_series2)


def test_equals_with_nans():
    ser1 = pd.Series([0, 1, None], dtype="uint8[pyarrow]")
    ser2 = pd.Series([None, None, None], dtype="uint8[pyarrow]")
    assert not ser1.equals(ser2)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_ewm(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with warns_that_defaulting_to_pandas():
        modin_series.ewm(halflife=6)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_expanding(data):
    modin_series, pandas_series = create_test_series(data)  # noqa: F841
    df_equals(modin_series.expanding().sum(), pandas_series.expanding().sum())


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_factorize(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with warns_that_defaulting_to_pandas():
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


@pytest.mark.parametrize("limit_area", [None, "inside", "outside"])
@pytest.mark.parametrize("method", ["ffill", "bfill"])
def test_ffill_bfill_limit_area(method, limit_area):
    modin_ser, pandas_ser = create_test_series([1, None, 2, None])
    eval_general(
        modin_ser, pandas_ser, lambda ser: getattr(ser, method)(limit_area=limit_area)
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("reindex", [None, 2, -2])
@pytest.mark.parametrize("limit", [None, 1, 2, 0.5, -1, -2, 1.5])
@pytest.mark.exclude_in_sanity
def test_fillna(data, reindex, limit):
    modin_series, pandas_series = create_test_series(data)
    index = pandas_series.index
    pandas_replace_series = index.to_series().sample(frac=1)
    modin_replace_series = pd.Series(pandas_replace_series)
    replace_dict = pandas_replace_series.to_dict()

    if reindex is not None:
        if reindex > 0:
            pandas_series = pandas_series[:reindex].reindex(index)
        else:
            pandas_series = pandas_series[reindex:].reindex(index)
        # Because of bug #3178 modin Series has to be created from pandas
        # Series instead of performing the same slice and reindex operations.
        modin_series = pd.Series(pandas_series)

    if isinstance(limit, float):
        limit = int(len(modin_series) * limit)
    if limit is not None and limit < 0:
        limit = len(modin_series) + limit

    df_equals(modin_series.fillna(0, limit=limit), pandas_series.fillna(0, limit=limit))
    df_equals(
        modin_series.fillna(method="bfill", limit=limit),
        pandas_series.fillna(method="bfill", limit=limit),
    )
    df_equals(
        modin_series.fillna(method="ffill", limit=limit),
        pandas_series.fillna(method="ffill", limit=limit),
    )
    df_equals(
        modin_series.fillna(modin_replace_series, limit=limit),
        pandas_series.fillna(pandas_replace_series, limit=limit),
    )
    df_equals(
        modin_series.fillna(replace_dict, limit=limit),
        pandas_series.fillna(replace_dict, limit=limit),
    )


@pytest.mark.xfail(reason="Using pandas Series.")
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_filter(data):
    modin_series = create_test_series(data)

    with pytest.raises(NotImplementedError):
        modin_series.filter(None, None, None)


def test_first():
    i = pd.date_range("2010-04-09", periods=400, freq="2D")
    modin_series = pd.Series(list(range(400)), index=i)
    pandas_series = pandas.Series(list(range(400)), index=i)
    with pytest.warns(FutureWarning, match="first is deprecated and will be removed"):
        modin_result = modin_series.first("3D")
    df_equals(modin_result, pandas_series.first("3D"))
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
    with warns_that_defaulting_to_pandas():
        modin_series.hist(None)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_iat(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.iat[0], pandas_series.iat[0])


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("skipna", [False, True])
def test_idxmax(data, skipna):
    modin_series, pandas_series = create_test_series(data)
    pandas_result = pandas_series.idxmax(skipna=skipna)
    modin_result = modin_series.idxmax(skipna=skipna)
    df_equals(modin_result, pandas_result)

    pandas_result = pandas_series.T.idxmax(skipna=skipna)
    modin_result = modin_series.T.idxmax(skipna=skipna)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("skipna", [False, True])
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
        # Scalar
        np.testing.assert_equal(modin_series.iloc[0], pandas_series.iloc[0])

        # Series
        df_equals(modin_series.iloc[1:], pandas_series.iloc[1:])
        df_equals(modin_series.iloc[1:2], pandas_series.iloc[1:2])
        df_equals(modin_series.iloc[[1, 2]], pandas_series.iloc[[1, 2]])

        # Write Item
        modin_series.iloc[[1, 2]] = 42
        pandas_series.iloc[[1, 2]] = 42
        df_equals(modin_series, pandas_series)
        with pytest.raises(IndexingError):
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
    with warns_that_defaulting_to_pandas():
        modin_series.interpolate()


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


def test_isin_with_series():
    modin_series1, pandas_series1 = create_test_series([1, 2, 3])
    modin_series2, pandas_series2 = create_test_series([1, 2, 3, 4, 5])

    eval_general(
        (modin_series1, modin_series2),
        (pandas_series1, pandas_series2),
        lambda srs: srs[0].isin(srs[1]),
    )

    # Verify that Series actualy behaves like Series and ignores unmatched indices on '.isin'
    modin_series1, pandas_series1 = create_test_series([1, 2, 3], index=[10, 11, 12])

    eval_general(
        (modin_series1, modin_series2),
        (pandas_series1, pandas_series2),
        lambda srs: srs[0].isin(srs[1]),
    )


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
def test_keys(data):
    modin_series, pandas_series = create_test_series(data)
    df_equals(modin_series.keys(), pandas_series.keys())


def test_kurtosis_alias():
    # It's optimization. If failed, Series.kurt should be tested explicitly
    # in tests: `test_kurt_kurtosis`, `test_kurt_kurtosis_level`.
    assert pd.Series.kurt == pd.Series.kurtosis


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("skipna", [False, True])
def test_kurtosis(axis, skipna):
    expected_exception = None
    if axis:
        expected_exception = ValueError("No axis named 1 for object type Series")
    eval_general(
        *create_test_series(test_data["float_nan_data"]),
        lambda df: df.kurtosis(axis=axis, skipna=skipna),
        expected_exception=expected_exception,
    )


@pytest.mark.parametrize("axis", ["rows", "columns"])
@pytest.mark.parametrize("numeric_only", [False, True])
def test_kurtosis_numeric_only(axis, numeric_only):
    expected_exception = None
    if axis:
        expected_exception = ValueError("No axis named columns for object type Series")
    eval_general(
        *create_test_series(test_data_diff_dtype),
        lambda df: df.kurtosis(axis=axis, numeric_only=numeric_only),
        expected_exception=expected_exception,
    )


def test_last():
    modin_index = pd.date_range("2010-04-09", periods=400, freq="2D")
    pandas_index = pandas.date_range("2010-04-09", periods=400, freq="2D")
    modin_series = pd.Series(list(range(400)), index=modin_index)
    pandas_series = pandas.Series(list(range(400)), index=pandas_index)
    with pytest.warns(FutureWarning, match="last is deprecated and will be removed"):
        modin_result = modin_series.last("3D")
    df_equals(modin_result, pandas_series.last("3D"))
    df_equals(modin_series.last("20D"), pandas_series.last("20D"))


@pytest.mark.parametrize("func", ["all", "any", "count"])
def test_index_order(func):
    # see #1708 and #1869 for details
    s_modin, s_pandas = create_test_series(test_data["float_nan_data"])
    rows_number = len(s_modin.index)
    level_0 = np.random.choice([x for x in range(10)], rows_number)
    level_1 = np.random.choice([x for x in range(10)], rows_number)
    index = pandas.MultiIndex.from_arrays([level_0, level_1])

    s_modin.index = index
    s_pandas.index = index

    # The result of the operation is not a Series, `.index` is missed
    df_equals(
        getattr(s_modin, func)(),
        getattr(s_pandas, func)(),
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
    ]  # fmt: skip
    pandas_result = pandas_series.loc[
        (slice(None), 1),
    ]  # fmt: skip
    df_equals(modin_result, pandas_result)


def test_loc_with_boolean_series():
    modin_series, pandas_series = create_test_series([1, 2, 3])
    modin_mask, pandas_mask = create_test_series([True, False, False])
    modin_result = modin_series.loc[modin_mask]
    pandas_result = pandas_series.loc[pandas_mask]
    df_equals(modin_result, pandas_result)


# This tests the bug from https://github.com/modin-project/modin/issues/3736
def test_loc_setting_categorical_series():
    modin_series = pd.Series(["a", "b", "c"], dtype="category")
    pandas_series = pandas.Series(["a", "b", "c"], dtype="category")
    modin_series.loc[1:3] = "a"
    pandas_series.loc[1:3] = "a"
    df_equals(modin_series, pandas_series)


# This tests the bug from https://github.com/modin-project/modin/issues/3736
def test_iloc_assigning_scalar_none_to_string_series():
    data = ["A"]
    modin_series, pandas_series = create_test_series(data, dtype="string")
    modin_series.iloc[0] = None
    pandas_series.iloc[0] = None
    df_equals(modin_series, pandas_series)


def test_set_ordered_categorical_column():
    data = {"a": [1, 2, 3], "b": [4, 5, 6]}
    mdf = pd.DataFrame(data)
    pdf = pandas.DataFrame(data)
    mdf["a"] = pd.Categorical(mdf["a"], ordered=True)
    pdf["a"] = pandas.Categorical(pdf["a"], ordered=True)
    df_equals(mdf, pdf)

    modin_categories = mdf["a"].dtype
    pandas_categories = pdf["a"].dtype
    assert modin_categories == pandas_categories


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_lt(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "lt")


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
        # https://github.com/modin-project/modin/issues/5967
        check_dtypes=False,
    )

    # Return list objects
    modin_series_lists = modin_series.map(lambda s: [s, s, s])
    pandas_series_lists = pandas_series.map(lambda s: [s, s, s])
    df_equals(modin_series_lists, pandas_series_lists)

    # Index into list objects
    df_equals(
        modin_series_lists.map(lambda lst: lst[0]),
        pandas_series_lists.map(lambda lst: lst[0]),
    )


def test_mask():
    modin_series = pd.Series(np.arange(10))
    m = modin_series % 3 == 0
    with warns_that_defaulting_to_pandas():
        try:
            modin_series.mask(~m, -modin_series)
        except ValueError:
            pass


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("skipna", [False, True])
def test_max(data, skipna):
    eval_general(*create_test_series(data), lambda df: df.max(skipna=skipna))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("skipna", [False, True])
def test_mean(data, skipna):
    eval_general(*create_test_series(data), lambda df: df.mean(skipna=skipna))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("skipna", [False, True])
def test_median(data, skipna):
    eval_general(*create_test_series(data), lambda df: df.median(skipna=skipna))


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
    eval_general(modin_s, pandas_s, lambda s: getattr(s, method)())


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("index", [True, False], ids=["True", "False"])
def test_memory_usage(data, index):
    modin_series, pandas_series = create_test_series(data)
    df_equals(
        modin_series.memory_usage(index=index), pandas_series.memory_usage(index=index)
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("skipna", [False, True])
def test_min(data, skipna):
    eval_general(*create_test_series(data), lambda df: df.min(skipna=skipna))


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


def test_tuple_name():
    names = [("a", 1), ("a", "b", "c"), "flat"]
    s = pd.Series(name=names[0])
    # The internal representation of the Series stores the name as a column label.
    # When it is a tuple, this label is a MultiIndex object, and this test ensures that
    # the Series's name property remains a tuple.
    assert s.name == names[0]
    assert isinstance(s.name, tuple)
    # Setting the name to a tuple of a different level or a non-tuple should not error.
    s.name = names[1]
    assert s.name == names[1]
    assert isinstance(s.name, tuple)
    s.name = names[2]
    assert s.name == names[2]
    assert isinstance(s.name, str)


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


@pytest.mark.xfail(reason="Using pandas Series.")
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_nlargest(data):
    modin_series = create_test_series(data)

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
    with warns_that_defaulting_to_pandas():
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
            x = (pd if isinstance(x, pd.Series) else pandas).concat((x, x))
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
@pytest.mark.parametrize("skipna", [False, True])
def test_prod(axis, skipna):
    expected_exception = None
    if axis:
        expected_exception = ValueError("No axis named 1 for object type Series")
    eval_general(
        *create_test_series(test_data["float_nan_data"]),
        lambda s: s.prod(axis=axis, skipna=skipna),
        expected_exception=expected_exception,
    )


@pytest.mark.parametrize("numeric_only", [False, True])
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
    except Exception as err:
        with pytest.raises(type(err)):
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
    o_data = [
        [24.3, 75.7, "high"],
        [31, 87.8, "high"],
        [22, 71.6, "medium"],
        [35, 95, "medium"],
    ]
    o_columns = ["temp_celsius", "temp_fahrenheit", "windspeed"]
    o_index = pd.date_range(start="2014-02-12", end="2014-02-15", freq="D")
    new_data = [[28, "low"], [30, "low"], [35.1, "medium"]]
    new_columns = ["temp_celsius", "windspeed"]
    new_index = pd.DatetimeIndex(["2014-02-12", "2014-02-13", "2014-02-15"])
    modin_df1 = pd.DataFrame(o_data, columns=o_columns, index=o_index)
    modin_df2 = pd.DataFrame(new_data, columns=new_columns, index=new_index)
    modin_result = modin_df2["windspeed"].reindex_like(modin_df1["windspeed"])

    pandas_df1 = pandas.DataFrame(o_data, columns=o_columns, index=o_index)
    pandas_df2 = pandas.DataFrame(new_data, columns=new_columns, index=new_index)
    pandas_result = pandas_df2["windspeed"].reindex_like(pandas_df1["windspeed"])
    df_equals(modin_result, pandas_result)


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
        0,
        2,
        [2],
        np.arange(256),
        [0] * 64 + [2] * 64 + [3] * 32 + [4] * 32 + [5] * 64,
        [2] * 257,
    ],
    ids=["0_case", "scalar", "one-elem-list", "array", "list", "wrong_list"],
)
def test_repeat_lists(data, repeats, request):
    expected_exception = None
    if "wrong_list" in request.node.callspec.id:
        expected_exception = ValueError(
            "operands could not be broadcast together with shape (256,) (257,)"
        )
    eval_general(
        *create_test_series(data),
        lambda df: df.repeat(repeats),
        expected_exception=expected_exception,
    )


def test_clip_4485():
    modin_result = pd.Series([1]).clip([3])
    pandas_result = pandas.Series([1]).clip([3])
    df_equals(modin_result, pandas_result)


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
@pytest.mark.exclude_in_sanity
def test_resample(closed, label, level):
    rule = "5min"
    freq = "h"

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
        rule, closed=closed, label=label, level=level
    )
    modin_resampler = modin_series.resample(
        rule, closed=closed, label=label, level=level
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
        df_equals(modin_resampler.nearest(), pandas_resampler.nearest())
        df_equals(modin_resampler.bfill(), pandas_resampler.bfill())
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
@pytest.mark.parametrize("name", [lib.no_default, "Custom name"])
@pytest.mark.parametrize("inplace", [True, False])
def test_reset_index(data, drop, name, inplace):
    expected_exception = None
    if inplace and not drop:
        expected_exception = TypeError(
            "Cannot reset_index inplace on a Series to create a DataFrame"
        )
    eval_general(
        *create_test_series(data),
        lambda df, *args, **kwargs: df.reset_index(*args, **kwargs),
        drop=drop,
        name=name,
        inplace=inplace,
        __inplace__=inplace,
        expected_exception=expected_exception,
    )


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
    except Exception as err:
        with pytest.raises(type(err)):
            modin_series.sample(frac=0.5, random_state=21019)
    else:
        modin_result = modin_series.sample(frac=0.5, random_state=21019)
        df_equals(pandas_result, modin_result)

    try:
        pandas_result = pandas_series.sample(n=12, random_state=21019)
    except Exception as err:
        with pytest.raises(type(err)):
            modin_series.sample(n=12, random_state=21019)
    else:
        modin_result = modin_series.sample(n=12, random_state=21019)
        df_equals(pandas_result, modin_result)

    with warns_that_defaulting_to_pandas():
        df_equals(
            modin_series.sample(n=0, random_state=21019),
            pandas_series.sample(n=0, random_state=21019),
        )
    with pytest.raises(ValueError):
        modin_series.sample(n=-3)


@pytest.mark.parametrize("single_value_data", [True, False])
@pytest.mark.parametrize("use_multiindex", [True, False])
@pytest.mark.parametrize("sorter", [True, None])
@pytest.mark.parametrize("values_number", [1, 2, 5])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.exclude_in_sanity
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


@pytest.mark.parametrize("skipna", [False, True])
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
def test_size(data):
    modin_series, pandas_series = create_test_series(data)
    assert modin_series.size == pandas_series.size


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("skipna", [False, True])
def test_skew(data, skipna):
    eval_general(*create_test_series(data), lambda df: df.skew(skipna=skipna))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("index", ["default", "ndarray", "has_duplicates"])
@pytest.mark.parametrize("periods", [0, 1, -1, 10, -10, 1000000000, -1000000000])
@pytest.mark.parametrize("name", [None, "foo"])
def test_shift(data, index, periods, name):
    modin_series, pandas_series = create_test_series(data, name=name)
    if index == "ndarray":
        data_column_length = len(data[next(iter(data))])
        modin_series.index = pandas_series.index = np.arange(2, data_column_length + 2)
    elif index == "has_duplicates":
        modin_series.index = pandas_series.index = list(modin_series.index[:-3]) + [
            0,
            1,
            2,
        ]

    df_equals(
        modin_series.shift(periods=periods),
        pandas_series.shift(periods=periods),
    )
    df_equals(
        modin_series.shift(periods=periods, fill_value=777),
        pandas_series.shift(periods=periods, fill_value=777),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("ascending", [False, True])
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
@pytest.mark.parametrize("ascending", [True, False])
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
        df_equals_with_non_stable_indices(modin_result, pandas_result)
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
        df_equals_with_non_stable_indices(modin_result, pandas_result)
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
@pytest.mark.parametrize("skipna", [False, True])
@pytest.mark.parametrize("ddof", int_arg_values, ids=arg_keys("ddof", int_arg_keys))
def test_std(request, data, skipna, ddof):
    modin_series, pandas_series = create_test_series(data)
    try:
        pandas_result = pandas_series.std(skipna=skipna, ddof=ddof)
    except Exception as err:
        with pytest.raises(type(err)):
            modin_series.std(skipna=skipna, ddof=ddof)
    else:
        modin_result = modin_series.std(skipna=skipna, ddof=ddof)
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_sub(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "sub")


def test_6782():
    datetime_scalar = datetime.datetime(1970, 1, 1, 0, 0)
    match = "Adding/subtracting object-dtype array to DatetimeArray not vectorized"
    with warnings.catch_warnings():
        warnings.filterwarnings("error", match, PerformanceWarning)
        pd.Series([datetime.datetime(2000, 1, 1)]) - datetime_scalar


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_subtract(data):
    modin_series, pandas_series = create_test_series(data)
    inter_df_math_helper(modin_series, pandas_series, "subtract")


@pytest.mark.parametrize(
    "data",
    test_data_values + test_data_small_values,
    ids=test_data_keys + test_data_small_keys,
)
@pytest.mark.parametrize("skipna", [False, True])
@pytest.mark.parametrize("numeric_only", [False, True])
@pytest.mark.parametrize(
    "min_count", int_arg_values, ids=arg_keys("min_count", int_arg_keys)
)
@pytest.mark.exclude_in_sanity
def test_sum(data, skipna, numeric_only, min_count):
    eval_general(
        *create_test_series(data),
        lambda df, *args, **kwargs: df.sum(*args, **kwargs),
        skipna=skipna,
        numeric_only=numeric_only,
        min_count=min_count,
    )


@pytest.mark.parametrize("operation", ["sum", "shift"])
def test_sum_axis_1_except(operation):
    eval_general(
        *create_test_series(test_data["int_data"]),
        lambda df, *args, **kwargs: getattr(df, operation)(*args, **kwargs),
        axis=1,
        expected_exception=ValueError("No axis named 1 for object type Series"),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis1", [0, 1, "columns", "index"])
@pytest.mark.parametrize("axis2", [0, 1, "columns", "index"])
def test_swapaxes(data, axis1, axis2):
    modin_series, pandas_series = create_test_series(data)
    try:
        pandas_result = pandas_series.swapaxes(axis1, axis2)
    except Exception as err:
        with pytest.raises(type(err)):
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
    except Exception as err:
        with pytest.raises(type(err)):
            modin_s.take([2], axis=1)


@pytest.mark.parametrize(
    "ignore_index", bool_arg_values, ids=arg_keys("ignore_index", bool_arg_keys)
)
def test_explode(ignore_index):
    # Some items in this test data are lists that explode() should expand.
    data = [[1, 2, 3], "foo", [], [3, 4]]
    modin_series, pandas_series = create_test_series(data)
    df_equals(
        modin_series.explode(ignore_index=ignore_index),
        pandas_series.explode(ignore_index=ignore_index),
    )


def test_to_period():
    idx = pd.date_range("1/1/2012", periods=5, freq="M")
    series = pd.Series(np.random.randint(0, 100, size=(len(idx))), index=idx)
    with warns_that_defaulting_to_pandas():
        series.to_period()


@pytest.mark.parametrize(
    "data",
    test_data_values + test_data_large_categorical_series_values,
    ids=test_data_keys + test_data_large_categorical_series_keys,
)
def test_to_numpy(data):
    modin_series, pandas_series = create_test_series(data)
    assert_array_equal(modin_series.to_numpy(), pandas_series.to_numpy())


def test_to_numpy_dtype():
    modin_series, pandas_series = create_test_series(test_data["float_nan_data"])
    assert_array_equal(
        modin_series.to_numpy(dtype="int64"),
        pandas_series.to_numpy(dtype="int64"),
        strict=True,
    )


@pytest.mark.parametrize(
    "data",
    test_data_values + test_data_large_categorical_series_values,
    ids=test_data_keys + test_data_large_categorical_series_keys,
)
def test_series_values(data):
    modin_series, pandas_series = create_test_series(data)
    assert_array_equal(modin_series.values, pandas_series.values)


def test_series_empty_values():
    modin_series, pandas_series = pd.Series(), pandas.Series()
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
    with warns_that_defaulting_to_pandas():
        series.to_period().to_timestamp()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_to_xarray(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with warns_that_defaulting_to_pandas():
        modin_series.to_xarray()


def test_to_xarray_mock():
    modin_series = pd.Series([])

    with mock.patch("pandas.Series.to_xarray") as to_xarray:
        modin_series.to_xarray()
    to_xarray.assert_called_once()
    assert len(to_xarray.call_args[0]) == 1
    df_equals(modin_series, to_xarray.call_args[0][0])


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_tolist(data):
    modin_series, _ = create_test_series(data)  # noqa: F841
    with warns_that_defaulting_to_pandas():
        modin_series.tolist()


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize(
    "func", [lambda x: x + 1, [np.sqrt, np.exp]], ids=["lambda", "list_udfs"]
)
def test_transform(data, func, request):
    if "list_udfs" in request.node.callspec.id:
        pytest.xfail(reason="https://github.com/modin-project/modin/issues/6998")
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
        expected_exception=ValueError("Function did not transform"),
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
    comparator = lambda *args: sort_if_range_partitioning(  # noqa: E731
        *args, comparator=assert_array_equal
    )

    modin_series, pandas_series = create_test_series(data)
    modin_result = modin_series.unique()
    pandas_result = pandas_series.unique()
    comparator(modin_result, pandas_result)
    assert modin_result.shape == pandas_result.shape
    assert type(modin_result) is type(pandas_result)

    modin_result = pd.Series([2, 1, 3, 3], name="A").unique()
    pandas_result = pandas.Series([2, 1, 3, 3], name="A").unique()
    comparator(modin_result, pandas_result)
    assert modin_result.shape == pandas_result.shape
    assert type(modin_result) is type(pandas_result)

    modin_result = pd.Series([pd.Timestamp("2016-01-01") for _ in range(3)]).unique()
    pandas_result = pandas.Series(
        [pd.Timestamp("2016-01-01") for _ in range(3)]
    ).unique()
    comparator(modin_result, pandas_result)
    assert modin_result.shape == pandas_result.shape
    assert type(modin_result) is type(pandas_result)

    modin_result = pd.Series(
        [pd.Timestamp("2016-01-01", tz="US/Eastern") for _ in range(3)]
    ).unique()
    pandas_result = pandas.Series(
        [pd.Timestamp("2016-01-01", tz="US/Eastern") for _ in range(3)]
    ).unique()
    comparator(modin_result, pandas_result)
    assert modin_result.shape == pandas_result.shape
    assert type(modin_result) is type(pandas_result)

    modin_result = pandas.Series(pd.Categorical(list("baabc"))).unique()
    pandas_result = pd.Series(pd.Categorical(list("baabc"))).unique()
    comparator(modin_result, pandas_result)
    assert modin_result.shape == pandas_result.shape
    assert type(modin_result) is type(pandas_result)

    modin_result = pd.Series(
        pd.Categorical(list("baabc"), categories=list("abc"), ordered=True)
    ).unique()
    pandas_result = pandas.Series(
        pd.Categorical(list("baabc"), categories=list("abc"), ordered=True)
    ).unique()
    comparator(modin_result, pandas_result)
    assert modin_result.shape == pandas_result.shape
    assert type(modin_result) is type(pandas_result)


def test_unique_pyarrow_dtype():
    # See #6227 for details
    modin_series, pandas_series = create_test_series(
        [1, 0, pd.NA], dtype="uint8[pyarrow]"
    )

    def comparator(df1, df2):
        # Perform our own non-strict version of dtypes equality check
        df_equals(df1, df2)
        # to be sure `unique` return `ArrowExtensionArray`
        assert type(df1) is type(df2)

    eval_general(
        modin_series, pandas_series, lambda df: df.unique(), comparator=comparator
    )


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


def test_unstack_error_no_multiindex():
    modin_series = pd.Series([0, 1, 2])
    with pytest.raises(ValueError, match="index must be a MultiIndex to unstack"):
        modin_series.unstack()


@pytest.mark.parametrize(
    "data, other_data",
    [([1, 2, 3], [4, 5, 6]), ([1, 2, 3], [4, 5, 6, 7, 8]), ([1, 2, 3], [4, np.nan, 6])],
)
def test_update(data, other_data):
    modin_series, pandas_series = pd.Series(data), pandas.Series(data)
    modin_series.update(pd.Series(other_data))
    pandas_series.update(pandas.Series(other_data))
    df_equals(modin_series, pandas_series)


@pytest.mark.parametrize("sort", bool_arg_values, ids=bool_arg_keys)
@pytest.mark.parametrize("normalize", bool_arg_values, ids=bool_arg_keys)
@pytest.mark.parametrize("bins", [3, None])
@pytest.mark.parametrize(
    "dropna",
    [
        pytest.param(None),
        pytest.param(False),
        pytest.param(True),
    ],
)
@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.exclude_in_sanity
def test_value_counts(sort, normalize, bins, dropna, ascending):
    def sort_sensitive_comparator(df1, df2):
        # We sort indices for Modin and pandas result because of issue #1650
        return (
            df_equals_with_non_stable_indices(df1, df2)
            if sort
            else df_equals(df1.sort_index(), df2.sort_index())
        )

    eval_general(
        *create_test_series(test_data_values[0]),
        lambda df: df.value_counts(
            sort=sort,
            bins=bins,
            normalize=normalize,
            dropna=dropna,
            ascending=ascending,
        ),
        comparator=sort_sensitive_comparator,
    )

    # from issue #2365
    arr = np.random.rand(2**6)
    arr[::10] = np.nan
    eval_general(
        *create_test_series(arr),
        lambda df: df.value_counts(
            sort=sort,
            bins=bins,
            normalize=normalize,
            dropna=dropna,
            ascending=ascending,
        ),
        comparator=sort_sensitive_comparator,
    )


def test_value_counts_categorical():
    # from issue #3571
    data = np.array(["a"] * 50000 + ["b"] * 10000 + ["c"] * 1000)
    random_state = np.random.RandomState(seed=42)
    random_state.shuffle(data)
    eval_general(
        *create_test_series(data, dtype="category"),
        lambda df: df.value_counts(),
        comparator=df_equals,
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_values(data):
    modin_series, pandas_series = create_test_series(data)

    np.testing.assert_equal(modin_series.values, pandas_series.values)


def test_values_non_numeric():
    data = ["str{0}".format(i) for i in range(0, 10**3)]
    modin_series, pandas_series = create_test_series(data)

    modin_series = modin_series.astype("category")
    pandas_series = pandas_series.astype("category")

    df_equals(modin_series.values, pandas_series.values)


def test_values_ea():
    data = pandas.arrays.SparseArray(np.arange(10, dtype="int64"))
    modin_series, pandas_series = create_test_series(data)
    modin_values = modin_series.values
    pandas_values = pandas_series.values

    assert modin_values.dtype == pandas_values.dtype
    df_equals(modin_values, pandas_values)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("skipna", [False, True])
@pytest.mark.parametrize("ddof", int_arg_values, ids=arg_keys("ddof", int_arg_keys))
def test_var(data, skipna, ddof):
    modin_series, pandas_series = create_test_series(data)

    try:
        pandas_result = pandas_series.var(skipna=skipna, ddof=ddof)
    except Exception as err:
        with pytest.raises(type(err)):
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

    other_data = random_state.randn(100)
    modin_other, pandas_other = pd.Series(other_data), pandas.Series(other_data)
    pandas_result = pandas_series.where(pandas_cond_series, pandas_other, axis=0)
    modin_result = modin_series.where(modin_cond_series, modin_other, axis=0)
    assert all(to_pandas(modin_result) == pandas_result)

    pandas_result = pandas_series.where(pandas_series < 2, True)
    modin_result = modin_series.where(modin_series < 2, True)
    assert all(to_pandas(modin_result) == pandas_result)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize(
    "key",
    [0, slice(0, len(test_string_data_values) / 2)],
    ids=["single_key", "slice_key"],
)
def test_str___getitem__(data, key):
    modin_series, pandas_series = create_test_series(data)
    modin_result = modin_series.str[key]
    pandas_result = pandas_series.str[key]
    df_equals(
        modin_result,
        pandas_result,
        # https://github.com/modin-project/modin/issues/5968
        check_dtypes=False,
    )


# Test str operations
@pytest.mark.parametrize(
    "others",
    [["abC|DeF,Hik", "gSaf,qWer|Gre", "asd3,4sad|", np.nan], None],
    ids=["list", "None"],
)
def test_str_cat(others):
    data = ["abC|DeF,Hik", "gSaf,qWer|Gre", "asd3,4sad|", np.nan]
    eval_general(*create_test_series(data), lambda s: s.str.cat(others=others))


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("pat", string_sep_values, ids=string_sep_keys)
@pytest.mark.parametrize("n", int_arg_values, ids=int_arg_keys)
@pytest.mark.parametrize("expand", [False, True])
def test_str_split(data, pat, n, expand):
    eval_general(
        *create_test_series(data),
        lambda series: series.str.split(pat, n=n, expand=expand),
    )


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("pat", string_sep_values, ids=string_sep_keys)
@pytest.mark.parametrize("n", int_arg_values, ids=int_arg_keys)
@pytest.mark.parametrize("expand", [False, True])
def test_str_rsplit(data, pat, n, expand):
    eval_general(
        *create_test_series(data),
        lambda series: series.str.rsplit(pat, n=n, expand=expand),
    )


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("i", int_arg_values, ids=int_arg_keys)
def test_str_get(data, i):
    modin_series, pandas_series = create_test_series(data)
    eval_general(modin_series, pandas_series, lambda series: series.str.get(i))


@pytest.mark.parametrize(
    "data", test_string_list_data_values, ids=test_string_list_data_keys
)
@pytest.mark.parametrize("sep", string_sep_values, ids=string_sep_keys)
def test_str_join(data, sep):
    modin_series, pandas_series = create_test_series(data)
    eval_general(modin_series, pandas_series, lambda series: series.str.join(sep))


@pytest.mark.parametrize(
    "data", test_string_list_data_values, ids=test_string_list_data_keys
)
@pytest.mark.parametrize("sep", string_sep_values, ids=string_sep_keys)
def test_str_get_dummies(data, sep):
    modin_series, pandas_series = create_test_series(data)

    if sep:
        with warns_that_defaulting_to_pandas():
            # We are only testing that this defaults to pandas, so we will just check for
            # the warning
            modin_series.str.get_dummies(sep)


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("pat", string_sep_values, ids=string_sep_keys)
@pytest.mark.parametrize("case", bool_arg_values, ids=bool_arg_keys)
@pytest.mark.parametrize("na", string_na_rep_values, ids=string_na_rep_keys)
def test_str_contains(data, pat, case, na):
    modin_series, pandas_series = create_test_series(data)
    eval_general(
        modin_series,
        pandas_series,
        lambda series: series.str.contains(pat, case=case, na=na, regex=False),
        # https://github.com/modin-project/modin/issues/5969
        comparator_kwargs={"check_dtypes": False},
    )

    # Test regex
    pat = ",|b"
    eval_general(
        modin_series,
        pandas_series,
        lambda series: series.str.contains(pat, case=case, na=na, regex=True),
        # https://github.com/modin-project/modin/issues/5969
        comparator_kwargs={"check_dtypes": False},
    )


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("pat", string_sep_values, ids=string_sep_keys)
@pytest.mark.parametrize("repl", string_sep_values, ids=string_sep_keys)
@pytest.mark.parametrize("n", int_arg_values, ids=int_arg_keys)
@pytest.mark.parametrize("case", bool_arg_values, ids=bool_arg_keys)
def test_str_replace(data, pat, repl, n, case):
    eval_general(
        *create_test_series(data),
        lambda series: series.str.replace(pat, repl, n=n, case=case, regex=False),
        # https://github.com/modin-project/modin/issues/5970
        comparator_kwargs={"check_dtypes": pat is not None},
    )
    # Test regex
    eval_general(
        *create_test_series(data),
        lambda series: series.str.replace(
            pat=",|b", repl=repl, n=n, case=case, regex=True
        ),
        # https://github.com/modin-project/modin/issues/5970
        comparator_kwargs={"check_dtypes": pat is not None},
    )


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("repeats", int_arg_values, ids=int_arg_keys)
def test_str_repeat(data, repeats):
    modin_series, pandas_series = create_test_series(data)
    eval_general(modin_series, pandas_series, lambda series: series.str.repeat(repeats))


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_str_removeprefix(data):
    modin_series, pandas_series = create_test_series(data)
    prefix = "test_prefix"
    eval_general(
        modin_series,
        pandas_series,
        lambda series: (prefix + series).str.removeprefix(prefix),
    )


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_str_removesuffix(data):
    modin_series, pandas_series = create_test_series(data)
    suffix = "test_suffix"
    eval_general(
        modin_series,
        pandas_series,
        lambda series: (series + suffix).str.removesuffix(suffix),
    )


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("width", [-1, 0, 5])
@pytest.mark.parametrize(
    "side", ["left", "right", "both"], ids=["left", "right", "both"]
)
@pytest.mark.parametrize("fillchar", string_sep_values, ids=string_sep_keys)
def test_str_pad(data, width, side, fillchar):
    modin_series, pandas_series = create_test_series(data)
    eval_general(
        modin_series,
        pandas_series,
        lambda series: series.str.pad(width, side=side, fillchar=fillchar),
    )


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("width", [-1, 0, 5])
@pytest.mark.parametrize("fillchar", string_sep_values, ids=string_sep_keys)
def test_str_center(data, width, fillchar):
    modin_series, pandas_series = create_test_series(data)
    eval_general(
        modin_series,
        pandas_series,
        lambda series: series.str.center(width, fillchar=fillchar),
    )


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("width", [-1, 0, 5])
@pytest.mark.parametrize("fillchar", string_sep_values, ids=string_sep_keys)
def test_str_ljust(data, width, fillchar):
    modin_series, pandas_series = create_test_series(data)
    eval_general(
        modin_series,
        pandas_series,
        lambda series: series.str.ljust(width, fillchar=fillchar),
    )


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("width", [-1, 0, 5])
@pytest.mark.parametrize("fillchar", string_sep_values, ids=string_sep_keys)
def test_str_rjust(data, width, fillchar):
    modin_series, pandas_series = create_test_series(data)
    eval_general(
        modin_series,
        pandas_series,
        lambda series: series.str.rjust(width, fillchar=fillchar),
    )


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("width", [-1, 0, 5])
def test_str_zfill(data, width):
    modin_series, pandas_series = create_test_series(data)
    eval_general(modin_series, pandas_series, lambda series: series.str.zfill(width))


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("width", [-1, 0, 5])
def test_str_wrap(data, width):
    expected_exception = None
    if width != 5:
        expected_exception = ValueError(f"invalid width {width} (must be > 0)")
    modin_series, pandas_series = create_test_series(data)
    eval_general(
        modin_series,
        pandas_series,
        lambda series: series.str.wrap(width),
        expected_exception=expected_exception,
    )


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("start", int_arg_values, ids=int_arg_keys)
@pytest.mark.parametrize("stop", int_arg_values, ids=int_arg_keys)
@pytest.mark.parametrize("step", [-2, 1, 3])
def test_str_slice(data, start, stop, step):
    modin_series, pandas_series = create_test_series(data)
    eval_general(
        modin_series,
        pandas_series,
        lambda series: series.str.slice(start=start, stop=stop, step=step),
    )


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("start", int_arg_values, ids=int_arg_keys)
@pytest.mark.parametrize("stop", int_arg_values, ids=int_arg_keys)
@pytest.mark.parametrize("repl", string_sep_values, ids=string_sep_keys)
def test_str_slice_replace(data, start, stop, repl):
    modin_series, pandas_series = create_test_series(data)
    eval_general(
        modin_series,
        pandas_series,
        lambda series: series.str.slice_replace(start=start, stop=stop, repl=repl),
    )


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("pat", string_sep_values, ids=string_sep_keys)
def test_str_count(data, pat):
    modin_series, pandas_series = create_test_series(data)
    eval_general(modin_series, pandas_series, lambda series: series.str.count(pat))


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("pat", string_sep_values, ids=string_sep_keys)
@pytest.mark.parametrize("na", string_na_rep_values, ids=string_na_rep_keys)
def test_str_startswith(data, pat, na):
    modin_series, pandas_series = create_test_series(data)
    eval_general(
        modin_series,
        pandas_series,
        lambda series: series.str.startswith(pat, na=na),
        # https://github.com/modin-project/modin/issues/5969
        comparator_kwargs={"check_dtypes": False},
    )


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("pat", string_sep_values, ids=string_sep_keys)
@pytest.mark.parametrize("na", string_na_rep_values, ids=string_na_rep_keys)
def test_str_endswith(data, pat, na):
    modin_series, pandas_series = create_test_series(data)
    eval_general(
        modin_series,
        pandas_series,
        lambda series: series.str.endswith(pat, na=na),
        # https://github.com/modin-project/modin/issues/5969
        comparator_kwargs={"check_dtypes": False},
    )


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("pat", string_sep_values, ids=string_sep_keys)
def test_str_findall(data, pat):
    modin_series, pandas_series = create_test_series(data)
    eval_general(modin_series, pandas_series, lambda series: series.str.findall(pat))


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("pat", string_sep_values, ids=string_sep_keys)
def test_str_fullmatch(data, pat):
    modin_series, pandas_series = create_test_series(data)
    eval_general(modin_series, pandas_series, lambda series: series.str.fullmatch(pat))


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("pat", string_sep_values, ids=string_sep_keys)
@pytest.mark.parametrize("case", bool_arg_values, ids=bool_arg_keys)
@pytest.mark.parametrize("na", string_na_rep_values, ids=string_na_rep_keys)
def test_str_match(data, pat, case, na):
    modin_series, pandas_series = create_test_series(data)
    eval_general(
        modin_series,
        pandas_series,
        lambda series: series.str.match(pat, case=case, na=na),
    )


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("expand", [False, True])
@pytest.mark.parametrize("pat", [r"([ab])", r"([ab])(\d)"])
def test_str_extract(data, expand, pat):
    modin_series, pandas_series = create_test_series(data)

    eval_general(
        modin_series,
        pandas_series,
        lambda series: series.str.extract(pat, expand=expand),
    )


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_str_extractall(data):
    modin_series, pandas_series = create_test_series(data)

    with warns_that_defaulting_to_pandas():
        # We are only testing that this defaults to pandas, so we will just check for
        # the warning
        modin_series.str.extractall(r"([ab])(\d)")


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_str_len(data):
    modin_series, pandas_series = create_test_series(data)
    eval_general(modin_series, pandas_series, lambda series: series.str.len())


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("to_strip", string_sep_values, ids=string_sep_keys)
def test_str_strip(data, to_strip):
    modin_series, pandas_series = create_test_series(data)
    eval_general(
        modin_series, pandas_series, lambda series: series.str.strip(to_strip=to_strip)
    )


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("to_strip", string_sep_values, ids=string_sep_keys)
def test_str_rstrip(data, to_strip):
    modin_series, pandas_series = create_test_series(data)
    eval_general(
        modin_series, pandas_series, lambda series: series.str.rstrip(to_strip=to_strip)
    )


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("to_strip", string_sep_values, ids=string_sep_keys)
def test_str_lstrip(data, to_strip):
    modin_series, pandas_series = create_test_series(data)
    eval_general(
        modin_series, pandas_series, lambda series: series.str.lstrip(to_strip=to_strip)
    )


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("sep", string_sep_values, ids=string_sep_keys)
@pytest.mark.parametrize("expand", [False, True])
def test_str_partition(data, sep, expand):
    modin_series, pandas_series = create_test_series(data)
    eval_general(
        modin_series,
        pandas_series,
        lambda series: series.str.partition(sep, expand=expand),
        # https://github.com/modin-project/modin/issues/5971
        comparator_kwargs={"check_dtypes": sep is not None},
    )


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("sep", string_sep_values, ids=string_sep_keys)
@pytest.mark.parametrize("expand", [False, True])
def test_str_rpartition(data, sep, expand):
    modin_series, pandas_series = create_test_series(data)
    eval_general(
        modin_series,
        pandas_series,
        lambda series: series.str.rpartition(sep, expand=expand),
        # https://github.com/modin-project/modin/issues/5971
        comparator_kwargs={"check_dtypes": sep is not None},
    )


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_str_lower(data):
    modin_series, pandas_series = create_test_series(data)
    eval_general(modin_series, pandas_series, lambda series: series.str.lower())


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_str_upper(data):
    modin_series, pandas_series = create_test_series(data)
    eval_general(modin_series, pandas_series, lambda series: series.str.upper())


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_str_title(data):
    modin_series, pandas_series = create_test_series(data)
    eval_general(modin_series, pandas_series, lambda series: series.str.title())


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("sub", string_sep_values, ids=string_sep_keys)
@pytest.mark.parametrize("start", int_arg_values, ids=int_arg_keys)
@pytest.mark.parametrize("end", int_arg_values, ids=int_arg_keys)
def test_str_find(data, sub, start, end):
    modin_series, pandas_series = create_test_series(data)
    eval_general(
        modin_series,
        pandas_series,
        lambda series: series.str.find(sub, start=start, end=end),
    )


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("sub", string_sep_values, ids=string_sep_keys)
@pytest.mark.parametrize("start", int_arg_values, ids=int_arg_keys)
@pytest.mark.parametrize("end", int_arg_values, ids=int_arg_keys)
def test_str_rfind(data, sub, start, end):
    modin_series, pandas_series = create_test_series(data)
    eval_general(
        modin_series,
        pandas_series,
        lambda series: series.str.rfind(sub, start=start, end=end),
    )


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("sub", string_sep_values, ids=string_sep_keys)
@pytest.mark.parametrize(
    "start, end",
    [(0, None), (1, -1), (1, 3)],
    ids=["default", "non_default_working", "exception"],
)
def test_str_index(data, sub, start, end, request):
    modin_series, pandas_series = create_test_series(data)
    expected_exception = None
    if "exception-comma sep" in request.node.callspec.id:
        expected_exception = ValueError("substring not found")
    eval_general(
        modin_series,
        pandas_series,
        lambda series: series.str.index(sub, start=start, end=end),
        expected_exception=expected_exception,
    )


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("sub", string_sep_values, ids=string_sep_keys)
@pytest.mark.parametrize(
    "start, end",
    [(0, None), (1, -1), (1, 3)],
    ids=["default", "non_default_working", "exception"],
)
def test_str_rindex(data, sub, start, end, request):
    modin_series, pandas_series = create_test_series(data)
    expected_exception = None
    if "exception-comma sep" in request.node.callspec.id:
        expected_exception = ValueError("substring not found")
    eval_general(
        modin_series,
        pandas_series,
        lambda series: series.str.rindex(sub, start=start, end=end),
        expected_exception=expected_exception,
    )


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_str_capitalize(data):
    modin_series, pandas_series = create_test_series(data)
    eval_general(modin_series, pandas_series, lambda series: series.str.capitalize())


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_str_swapcase(data):
    modin_series, pandas_series = create_test_series(data)
    eval_general(modin_series, pandas_series, lambda series: series.str.swapcase())


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize(
    "form", ["NFC", "NFKC", "NFD", "NFKD"], ids=["NFC", "NFKC", "NFD", "NFKD"]
)
def test_str_normalize(data, form):
    modin_series, pandas_series = create_test_series(data)
    eval_general(modin_series, pandas_series, lambda series: series.str.normalize(form))


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
@pytest.mark.parametrize("pat", string_sep_values, ids=string_sep_keys)
def test_str_translate(data, pat):
    modin_series, pandas_series = create_test_series(data)

    # Test none table
    eval_general(
        modin_series,
        pandas_series,
        lambda series: series.str.translate(None),
        # https://github.com/modin-project/modin/issues/5970
        comparator_kwargs={"check_dtypes": False},
    )

    # Translation dictionary
    table = {pat: "DDD"}
    eval_general(
        modin_series, pandas_series, lambda series: series.str.translate(table)
    )

    # Translation table with maketrans (python3 only)
    if pat is not None:
        table = str.maketrans(pat, "d" * len(pat))
        eval_general(
            modin_series, pandas_series, lambda series: series.str.translate(table)
        )


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_str_isalnum(data):
    modin_series, pandas_series = create_test_series(data)
    eval_general(
        modin_series,
        pandas_series,
        lambda series: series.str.isalnum(),
        # https://github.com/modin-project/modin/issues/5969
        comparator_kwargs={"check_dtypes": False},
    )


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_str_isalpha(data):
    modin_series, pandas_series = create_test_series(data)
    eval_general(
        modin_series,
        pandas_series,
        lambda series: series.str.isalpha(),
        # https://github.com/modin-project/modin/issues/5969
        comparator_kwargs={"check_dtypes": False},
    )


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_str_isdigit(data):
    modin_series, pandas_series = create_test_series(data)
    eval_general(
        modin_series,
        pandas_series,
        lambda series: series.str.isdigit(),
        # https://github.com/modin-project/modin/issues/5969
        comparator_kwargs={"check_dtypes": False},
    )


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_str_isspace(data):
    modin_series, pandas_series = create_test_series(data)
    eval_general(
        modin_series,
        pandas_series,
        lambda series: series.str.isspace(),
        # https://github.com/modin-project/modin/issues/5969
        comparator_kwargs={"check_dtypes": False},
    )


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_str_islower(data):
    modin_series, pandas_series = create_test_series(data)
    eval_general(
        modin_series,
        pandas_series,
        lambda series: series.str.islower(),
        # https://github.com/modin-project/modin/issues/5969
        comparator_kwargs={"check_dtypes": False},
    )


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_str_isupper(data):
    modin_series, pandas_series = create_test_series(data)
    eval_general(
        modin_series,
        pandas_series,
        lambda series: series.str.isupper(),
        # https://github.com/modin-project/modin/issues/5969
        comparator_kwargs={"check_dtypes": False},
    )


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_str_istitle(data):
    modin_series, pandas_series = create_test_series(data)
    eval_general(
        modin_series,
        pandas_series,
        lambda series: series.str.istitle(),
        # https://github.com/modin-project/modin/issues/5969
        comparator_kwargs={"check_dtypes": False},
    )


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_str_isnumeric(data):
    modin_series, pandas_series = create_test_series(data)
    eval_general(
        modin_series,
        pandas_series,
        lambda series: series.str.isnumeric(),
        # https://github.com/modin-project/modin/issues/5969
        comparator_kwargs={"check_dtypes": False},
    )


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_str_isdecimal(data):
    modin_series, pandas_series = create_test_series(data)
    eval_general(
        modin_series,
        pandas_series,
        lambda series: series.str.isdecimal(),
        # https://github.com/modin-project/modin/issues/5969
        comparator_kwargs={"check_dtypes": False},
    )


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_casefold(data):
    modin_series, pandas_series = create_test_series(data)
    eval_general(modin_series, pandas_series, lambda series: series.str.casefold())


@pytest.fixture
def str_encode_decode_test_data() -> list[str]:
    return [
        "abC|DeF,Hik",
        "234,3245.67",
        "gSaf,qWer|Gre",
        "asd3,4sad|",
        np.nan,
        None,
        # add a string that we can't encode in ascii, and whose utf-8 encoding
        # we cannot decode in ascii
        "ക",
    ]


@pytest.mark.parametrize("encoding", encoding_types)
@pytest.mark.parametrize("errors", ["strict", "ignore", "replace"])
def test_str_encode(encoding, errors, str_encode_decode_test_data):
    expected_exception = None
    if errors == "strict" and encoding == "ascii":
        # quite safe to check only types
        expected_exception = False
    eval_general(
        *create_test_series(str_encode_decode_test_data),
        lambda s: s.str.encode(encoding, errors=errors),
        expected_exception=expected_exception,
    )


@pytest.mark.parametrize(
    "encoding",
    encoding_types,
)
@pytest.mark.parametrize("errors", ["strict", "ignore", "replace"])
def test_str_decode(encoding, errors, str_encode_decode_test_data):
    expected_exception = None
    if errors == "strict":
        # it's quite safe here to check only types of exceptions
        expected_exception = False
    eval_general(
        *create_test_series(
            [
                s.encode("utf-8") if isinstance(s, str) else s
                for s in str_encode_decode_test_data
            ]
        ),
        lambda s: s.str.decode(encoding, errors=errors),
        expected_exception=expected_exception,
    )


def test_list_general():
    pa = pytest.importorskip("pyarrow")

    # Copied from pandas examples
    modin_series, pandas_series = create_test_series(
        [
            [1, 2, 3],
            [3],
        ],
        dtype=pd.ArrowDtype(pa.list_(pa.int64())),
    )
    eval_general(modin_series, pandas_series, lambda series: series.list.flatten())
    eval_general(modin_series, pandas_series, lambda series: series.list.len())
    eval_general(modin_series, pandas_series, lambda series: series.list[0])


def test_struct_general():
    pa = pytest.importorskip("pyarrow")

    # Copied from pandas examples
    modin_series, pandas_series = create_test_series(
        [
            {"version": 1, "project": "pandas"},
            {"version": 2, "project": "pandas"},
            {"version": 1, "project": "numpy"},
        ],
        dtype=pd.ArrowDtype(
            pa.struct([("version", pa.int64()), ("project", pa.string())])
        ),
    )
    eval_general(modin_series, pandas_series, lambda series: series.struct.dtypes)
    eval_general(
        modin_series, pandas_series, lambda series: series.struct.field("project")
    )
    eval_general(modin_series, pandas_series, lambda series: series.struct.explode())

    # nested struct types
    version_type = pa.struct(
        [
            ("major", pa.int64()),
            ("minor", pa.int64()),
        ]
    )
    modin_series, pandas_series = create_test_series(
        [
            {"version": {"major": 1, "minor": 5}, "project": "pandas"},
            {"version": {"major": 2, "minor": 1}, "project": "pandas"},
            {"version": {"major": 1, "minor": 26}, "project": "numpy"},
        ],
        dtype=pd.ArrowDtype(
            pa.struct([("version", version_type), ("project", pa.string())])
        ),
    )
    eval_general(
        modin_series,
        pandas_series,
        lambda series: series.struct.field(["version", "minor"]),
    )


def _case_when_caselists():
    def permutations(values):
        return [
            p
            for r in range(1, len(values) + 1)
            for p in itertools.permutations(values, r)
        ]

    conditions = permutations(
        [
            [True, False, False, False] * 10,
            pandas.Series([True, False, False, False] * 10),
            pandas.Series([True, False, False, False] * 10, index=range(78, -2, -2)),
            lambda df: df.gt(0),
        ]
    )
    replacements = permutations([[0, 3, 4, 5] * 10, 0, lambda df: 1])
    caselists = []
    for c in conditions:
        for r in replacements:
            if len(c) == len(r):
                caselists.append(list(zip(c, r)))
    return caselists


@pytest.mark.parametrize(
    "base",
    [
        pandas.Series(range(40)),
        pandas.Series([0, 7, 8, 9] * 10, name="c", index=range(0, 80, 2)),
    ],
)
@pytest.mark.parametrize(
    "caselist",
    _case_when_caselists(),
)
@pytest.mark.skipif(
    Engine.get() == "Dask",
    reason="https://github.com/modin-project/modin/issues/7148",
)
def test_case_when(base, caselist):
    pandas_result = base.case_when(caselist)
    modin_bases = [pd.Series(base)]

    # 'base' and serieses from 'caselist' must have equal lengths, however in this test we want
    # to verify that 'case_when' works correctly even if partitioning of 'base' and 'caselist' isn't equal.
    # BaseOnPython always uses a single partition, thus skipping this test for them.
    if f"{StorageFormat.get()}On{Engine.get()}" != "BaseOnPython":
        modin_base_repart = construct_modin_df_by_scheme(
            base.to_frame(),
            partitioning_scheme={"row_lengths": [14, 14, 12], "column_widths": [1]},
        ).squeeze(axis=1)
        assert (
            modin_bases[0]._query_compiler._modin_frame._partitions.shape
            != modin_base_repart._query_compiler._modin_frame._partitions.shape
        )
        modin_base_repart.name = base.name
        modin_bases.append(modin_base_repart)

    for modin_base in modin_bases:
        df_equals(pandas_result, modin_base.case_when(caselist))
        if any(
            isinstance(data, pandas.Series)
            for case_tuple in caselist
            for data in case_tuple
        ):
            caselist = [
                tuple(
                    pd.Series(data) if isinstance(data, pandas.Series) else data
                    for data in case_tuple
                )
                for case_tuple in caselist
            ]
            df_equals(pandas_result, modin_base.case_when(caselist))


@pytest.mark.parametrize("data", test_string_data_values, ids=test_string_data_keys)
def test_non_commutative_add_string_to_series(data):
    # This test checks that add and radd do different things when addition is
    # not commutative, e.g. for adding a string to a string. For context see
    # https://github.com/modin-project/modin/issues/4908
    eval_general(*create_test_series(data), lambda s: "string" + s)
    eval_general(*create_test_series(data), lambda s: s + "string")


def test_non_commutative_multiply_pandas():
    # The non commutative integer class implementation is tricky. Check that
    # multiplying such an integer with a pandas series is really not
    # commutative.
    pandas_series = pandas.Series(1, dtype=int)
    integer = NonCommutativeMultiplyInteger(2)
    assert not (integer * pandas_series).equals(pandas_series * integer)


def test_non_commutative_multiply():
    # This test checks that mul and rmul do different things when
    # multiplication is not commutative, e.g. for adding a string to a string.
    # For context see https://github.com/modin-project/modin/issues/5238
    modin_series, pandas_series = create_test_series(1, dtype=int)
    integer = NonCommutativeMultiplyInteger(2)
    eval_general(modin_series, pandas_series, lambda s: integer * s)
    eval_general(modin_series, pandas_series, lambda s: s * integer)


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

    def set_categories(ser):
        ser.cat.categories = list("qwert")
        return ser

    # pandas 2.0.0: Removed setting Categorical.categories directly (GH47834)
    # Just check the exception
    expected_exception = AttributeError("can't set attribute")
    if sys.version_info >= (3, 10):
        # The exception message varies across different versions of Python
        expected_exception = False
    eval_general(
        modin_series,
        pandas_series,
        set_categories,
        expected_exception=expected_exception,
    )


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
    "set_min_row_partition_size",
    [1, 2],
    ids=["four_row_partitions", "two_row_partitions"],
    indirect=True,
)
def test_cat_codes_issue5650(set_min_row_partition_size):
    data = {"name": ["abc", "def", "ghi", "jkl"]}
    pandas_df = pandas.DataFrame(data)
    pandas_df = pandas_df.astype("category")
    modin_df = pd.DataFrame(data)
    modin_df = modin_df.astype("category")
    eval_general(
        modin_df,
        pandas_df,
        lambda df: df["name"].cat.codes,
        comparator_kwargs={"check_dtypes": True},
    )


@pytest.mark.parametrize(
    "data", test_data_categorical_values, ids=test_data_categorical_keys
)
def test_cat_rename_categories(data):
    modin_series, pandas_series = create_test_series(data.copy())
    pandas_result = pandas_series.cat.rename_categories(list("qwert"))
    modin_result = modin_series.cat.rename_categories(list("qwert"))
    df_equals(modin_series, pandas_series)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "data", test_data_categorical_values, ids=test_data_categorical_keys
)
@pytest.mark.parametrize("ordered", bool_arg_values, ids=bool_arg_keys)
def test_cat_reorder_categories(data, ordered):
    modin_series, pandas_series = create_test_series(data.copy())
    pandas_result = pandas_series.cat.reorder_categories(list("tades"), ordered=ordered)
    modin_result = modin_series.cat.reorder_categories(list("tades"), ordered=ordered)
    df_equals(modin_series, pandas_series)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "data", test_data_categorical_values, ids=test_data_categorical_keys
)
def test_cat_add_categories(data):
    modin_series, pandas_series = create_test_series(data.copy())
    pandas_result = pandas_series.cat.add_categories(list("qw"))
    modin_result = modin_series.cat.add_categories(list("qw"))
    df_equals(modin_series, pandas_series)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "data", test_data_categorical_values, ids=test_data_categorical_keys
)
def test_cat_remove_categories(data):
    modin_series, pandas_series = create_test_series(data.copy())
    pandas_result = pandas_series.cat.remove_categories(list("at"))
    modin_result = modin_series.cat.remove_categories(list("at"))
    df_equals(modin_series, pandas_series)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "data", test_data_categorical_values, ids=test_data_categorical_keys
)
def test_cat_remove_unused_categories(data):
    modin_series, pandas_series = create_test_series(data.copy())
    pandas_series[1] = np.nan
    pandas_result = pandas_series.cat.remove_unused_categories()
    modin_series[1] = np.nan
    modin_result = modin_series.cat.remove_unused_categories()
    df_equals(modin_series, pandas_series)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "data", test_data_categorical_values, ids=test_data_categorical_keys
)
@pytest.mark.parametrize("ordered", bool_arg_values, ids=bool_arg_keys)
@pytest.mark.parametrize("rename", [True, False])
def test_cat_set_categories(data, ordered, rename):
    modin_series, pandas_series = create_test_series(data.copy())
    pandas_result = pandas_series.cat.set_categories(
        list("qwert"), ordered=ordered, rename=rename
    )
    modin_result = modin_series.cat.set_categories(
        list("qwert"), ordered=ordered, rename=rename
    )
    df_equals(modin_series, pandas_series)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "data", test_data_categorical_values, ids=test_data_categorical_keys
)
def test_cat_as_ordered(data):
    modin_series, pandas_series = create_test_series(data.copy())
    pandas_result = pandas_series.cat.as_ordered()
    modin_result = modin_series.cat.as_ordered()
    df_equals(modin_series, pandas_series)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize(
    "data", test_data_categorical_values, ids=test_data_categorical_keys
)
def test_cat_as_unordered(data):
    modin_series, pandas_series = create_test_series(data.copy())
    pandas_result = pandas_series.cat.as_unordered()
    modin_result = modin_series.cat.as_unordered()
    df_equals(modin_series, pandas_series)
    df_equals(modin_result, pandas_result)


def test_peculiar_callback():
    def func(val):
        if not isinstance(val, tuple):
            raise BaseException("Urgh...")
        return val

    pandas_df = pandas.DataFrame({"col": [(0, 1)]})
    pandas_series = pandas_df["col"].apply(func)

    modin_df = pd.DataFrame({"col": [(0, 1)]})
    modin_series = modin_df["col"].apply(func)

    df_equals(modin_series, pandas_series)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_apply_return_df(data):
    modin_series, pandas_series = create_test_series(data)
    eval_general(
        modin_series,
        pandas_series,
        lambda series: series.apply(
            lambda x: pandas.Series([x + i for i in range(100)])
        ),
    )


@pytest.mark.parametrize(
    "function",
    [
        np.abs,
        np.sin,
    ],
)
def test_unary_numpy_universal_function_issue_6483(function):
    eval_general(*create_test_series(test_data["float_nan_data"]), function)


def test_binary_numpy_universal_function_issue_6483():
    eval_general(
        *create_test_series(test_data["float_nan_data"]),
        lambda series: np.arctan2(series, np.sin(series)),
    )


def test__reduce__():
    # `Series.__reduce__` will be called implicitly when lambda expressions are
    # pre-processed for the distributed engine.
    series_data = ["Major League Baseball", "National Basketball Association"]
    abbr_md, abbr_pd = create_test_series(series_data, index=["MLB", "NBA"])

    dataframe_data = {
        "name": ["Mariners", "Lakers"] * 500,
        "league_abbreviation": ["MLB", "NBA"] * 500,
    }
    teams_md, teams_pd = create_test_dfs(dataframe_data)

    result_md = (
        teams_md.set_index("name")
        .league_abbreviation.apply(lambda abbr: abbr_md.loc[abbr])
        .rename("league")
    )

    result_pd = (
        teams_pd.set_index("name")
        .league_abbreviation.apply(lambda abbr: abbr_pd.loc[abbr])
        .rename("league")
    )
    df_equals(result_md, result_pd)
