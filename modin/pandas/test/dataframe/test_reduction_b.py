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
import pandas
import os
import matplotlib
import modin.pandas as pd

from modin.pandas.test.utils import (
    df_equals,
    arg_keys,
    test_data_values,
    test_data_keys,
    axis_keys,
    axis_values,
    bool_arg_keys,
    bool_arg_values,
    int_arg_keys,
    int_arg_values,
    eval_general,
    create_test_dfs,
    test_data_small_values,
    test_data_small_keys,
)

pd.DEFAULT_NPARTITIONS = 4

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")


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

    modin_result = modin_df.T.count(axis=axis, numeric_only=numeric_only)
    pandas_result = pandas_df.T.count(axis=axis, numeric_only=numeric_only)
    df_equals(modin_result, pandas_result)

    # test level
    modin_df_multi_level = modin_df.copy()
    pandas_df_multi_level = pandas_df.copy()
    axis = modin_df._get_axis_number(axis) if axis is not None else 0
    levels = 3
    axis_names_list = [["a", "b", "c"], None]
    for axis_names in axis_names_list:
        if axis == 0:
            new_idx = pandas.MultiIndex.from_tuples(
                [(i // 4, i // 2, i) for i in range(len(modin_df.index))],
                names=axis_names,
            )
            modin_df_multi_level.index = new_idx
            pandas_df_multi_level.index = new_idx
            try:  # test error
                pandas_df_multi_level.count(axis=1, numeric_only=numeric_only, level=0)
            except Exception as e:
                with pytest.raises(type(e)):
                    modin_df_multi_level.count(
                        axis=1, numeric_only=numeric_only, level=0
                    )
        else:
            new_col = pandas.MultiIndex.from_tuples(
                [(i // 4, i // 2, i) for i in range(len(modin_df.columns))],
                names=axis_names,
            )
            modin_df_multi_level.columns = new_col
            pandas_df_multi_level.columns = new_col
            try:  # test error
                pandas_df_multi_level.count(axis=0, numeric_only=numeric_only, level=0)
            except Exception as e:
                with pytest.raises(type(e)):
                    modin_df_multi_level.count(
                        axis=0, numeric_only=numeric_only, level=0
                    )

        for level in list(range(levels)) + (axis_names if axis_names else []):
            modin_multi_level_result = modin_df_multi_level.count(
                axis=axis, numeric_only=numeric_only, level=level
            )
            pandas_multi_level_result = pandas_df_multi_level.count(
                axis=axis, numeric_only=numeric_only, level=level
            )
            df_equals(modin_multi_level_result, pandas_multi_level_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_describe(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    df_equals(modin_df.describe(), pandas_df.describe())
    percentiles = [0.10, 0.11, 0.44, 0.78, 0.99]
    df_equals(
        modin_df.describe(percentiles=percentiles),
        pandas_df.describe(percentiles=percentiles),
    )

    try:
        pandas_result = pandas_df.describe(exclude=[np.float64])
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.describe(exclude=[np.float64])
    else:
        modin_result = modin_df.describe(exclude=[np.float64])
        df_equals(modin_result, pandas_result)

    try:
        pandas_result = pandas_df.describe(exclude=np.float64)
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.describe(exclude=np.float64)
    else:
        modin_result = modin_df.describe(exclude=np.float64)
        df_equals(modin_result, pandas_result)

    try:
        pandas_result = pandas_df.describe(
            include=[np.timedelta64, np.datetime64, np.object, np.bool]
        )
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.describe(
                include=[np.timedelta64, np.datetime64, np.object, np.bool]
            )
    else:
        modin_result = modin_df.describe(
            include=[np.timedelta64, np.datetime64, np.object, np.bool]
        )
        df_equals(modin_result, pandas_result)

    modin_result = modin_df.describe(include=str(modin_df.dtypes.values[0]))
    pandas_result = pandas_df.describe(include=str(pandas_df.dtypes.values[0]))
    df_equals(modin_result, pandas_result)

    modin_result = modin_df.describe(include=[np.number])
    pandas_result = pandas_df.describe(include=[np.number])
    df_equals(modin_result, pandas_result)

    df_equals(modin_df.describe(include="all"), pandas_df.describe(include="all"))

    modin_df = pd.DataFrame(data).applymap(str)
    pandas_df = pandas.DataFrame(data).applymap(str)

    try:
        df_equals(modin_df.describe(), pandas_df.describe())
    except AssertionError:
        # We have to do this because we choose the highest count slightly differently
        # than pandas. Because there is no true guarantee which one will be first,
        # If they don't match, make sure that the `freq` is the same at least.
        df_equals(
            modin_df.describe().loc[["count", "unique", "freq"]],
            pandas_df.describe().loc[["count", "unique", "freq"]],
        )


def test_describe_dtypes():
    modin_df = pd.DataFrame(
        {
            "col1": list("abc"),
            "col2": list("abc"),
            "col3": list("abc"),
            "col4": [1, 2, 3],
        }
    )
    pandas_df = pandas.DataFrame(
        {
            "col1": list("abc"),
            "col2": list("abc"),
            "col3": list("abc"),
            "col4": [1, 2, 3],
        }
    )

    modin_result = modin_df.describe()
    pandas_result = pandas_df.describe()

    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_idxmax(data, axis, skipna):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    pandas_result = pandas_df.idxmax(axis=axis, skipna=skipna)
    modin_result = modin_df.idxmax(axis=axis, skipna=skipna)
    df_equals(modin_result, pandas_result)

    pandas_result = pandas_df.T.idxmax(axis=axis, skipna=skipna)
    modin_result = modin_df.T.idxmax(axis=axis, skipna=skipna)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
def test_idxmin(data, axis, skipna):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    modin_result = modin_df.idxmin(axis=axis, skipna=skipna)
    pandas_result = pandas_df.idxmin(axis=axis, skipna=skipna)
    df_equals(modin_result, pandas_result)

    modin_result = modin_df.T.idxmin(axis=axis, skipna=skipna)
    pandas_result = pandas_df.T.idxmin(axis=axis, skipna=skipna)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_last_valid_index(data):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)

    assert modin_df.last_valid_index() == (pandas_df.last_valid_index())


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
            modin_df.max(axis=axis, skipna=skipna, numeric_only=numeric_only)
    else:
        modin_result = modin_df.max(axis=axis, skipna=skipna, numeric_only=numeric_only)
        df_equals(modin_result, pandas_result)

    try:
        pandas_result = pandas_df.T.max(
            axis=axis, skipna=skipna, numeric_only=numeric_only
        )
    except Exception:
        with pytest.raises(TypeError):
            modin_df.T.max(axis=axis, skipna=skipna, numeric_only=numeric_only)
    else:
        modin_result = modin_df.T.max(
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

    try:
        pandas_result = pandas_df.T.mean(
            axis=axis, skipna=skipna, numeric_only=numeric_only
        )
    except Exception as e:
        with pytest.raises(type(e)):
            modin_df.T.mean(axis=axis, skipna=skipna, numeric_only=numeric_only)
    else:
        modin_result = modin_df.T.mean(
            axis=axis, skipna=skipna, numeric_only=numeric_only
        )
        df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("index", bool_arg_values, ids=arg_keys("index", bool_arg_keys))
def test_memory_usage(data, index):
    modin_df = pd.DataFrame(data)
    pandas_df = pandas.DataFrame(data)  # noqa F841

    modin_result = modin_df.memory_usage(index=index)
    pandas_result = pandas_df.memory_usage(index=index)
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

    try:
        pandas_result = pandas_df.T.min(
            axis=axis, skipna=skipna, numeric_only=numeric_only
        )
    except Exception:
        with pytest.raises(TypeError):
            modin_df.T.min(axis=axis, skipna=skipna, numeric_only=numeric_only)
    else:
        modin_result = modin_df.T.min(
            axis=axis, skipna=skipna, numeric_only=numeric_only
        )
        df_equals(modin_result, pandas_result)


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows has a memory issue for large numbers on this test",
)
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
    "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
)
@pytest.mark.parametrize(
    "min_count", int_arg_values, ids=arg_keys("min_count", int_arg_keys)
)
@pytest.mark.parametrize("is_transposed", [False, True])
@pytest.mark.parametrize(
    "operation",
    [
        "prod",
        pytest.param(
            "product",
            marks=pytest.mark.skipif(
                pandas.DataFrame.product == pandas.DataFrame.prod
                and pd.DataFrame.product == pd.DataFrame.prod,
                reason="That operation was already tested.",
            ),
        ),
    ],
)
def test_prod(
    request,
    data,
    axis,
    skipna,
    numeric_only,
    min_count,
    is_transposed,
    operation,
):
    eval_general(
        *create_test_dfs(data),
        lambda df, *args, **kwargs: getattr(df.T if is_transposed else df, operation)(
            *args, **kwargs
        ),
        axis=axis,
        skipna=skipna,
        numeric_only=numeric_only,
        min_count=min_count,
    )


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
    "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
)
@pytest.mark.parametrize(
    "min_count", int_arg_values, ids=arg_keys("min_count", int_arg_keys)
)
@pytest.mark.parametrize("is_transposed", [False, True])
def test_sum(request, data, axis, skipna, numeric_only, min_count, is_transposed):
    eval_general(
        *create_test_dfs(data),
        lambda df, *args, **kwargs: (df.T if is_transposed else df).sum(
            *args, **kwargs
        ),
        axis=axis,
        skipna=skipna,
        numeric_only=numeric_only,
        min_count=min_count,
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_sum_single_column(data):
    modin_df = pd.DataFrame(data).iloc[:, [0]]
    pandas_df = pandas.DataFrame(data).iloc[:, [0]]
    df_equals(modin_df.sum(), pandas_df.sum())
    df_equals(modin_df.sum(axis=1), pandas_df.sum(axis=1))
