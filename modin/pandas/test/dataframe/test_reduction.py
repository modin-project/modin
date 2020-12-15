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
    test_data,
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
    generate_multiindex,
    test_data_diff_dtype,
)

pd.DEFAULT_NPARTITIONS = 4

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")


@pytest.mark.parametrize("method", ["all", "any"])
@pytest.mark.parametrize("is_transposed", [False, True])
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("data", [test_data["float_nan_data"]])
def test_all_any(data, axis, skipna, is_transposed, method):
    eval_general(
        *create_test_dfs(data),
        lambda df: getattr((df.T if is_transposed else df), method)(
            axis=axis, skipna=skipna, bool_only=None
        ),
    )


@pytest.mark.parametrize("method", ["all", "any"])
@pytest.mark.parametrize(
    "bool_only", bool_arg_values, ids=arg_keys("bool_only", bool_arg_keys)
)
def test_all_any_specific(bool_only, method):
    eval_general(
        *create_test_dfs(test_data_diff_dtype),
        lambda df: getattr(df, method)(bool_only=bool_only),
    )


@pytest.mark.parametrize("method", ["all", "any"])
@pytest.mark.parametrize("level", [-1, 0, 1])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("data", [test_data["int_data"]])
def test_all_any_level(data, axis, level, method):
    modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)

    if axis == 0:
        new_idx = generate_multiindex(len(modin_df.index))
        modin_df.index = new_idx
        pandas_df.index = new_idx
    else:
        new_col = generate_multiindex(len(modin_df.columns))
        modin_df.columns = new_col
        pandas_df.columns = new_col

    eval_general(
        modin_df,
        pandas_df,
        lambda df: getattr(df, method)(axis=axis, level=level),
    )


@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("data", [test_data["float_nan_data"]])
def test_count(data, axis):
    eval_general(
        *create_test_dfs(data),
        lambda df: df.count(axis=axis),
    )


@pytest.mark.parametrize("numeric_only", [True, False, None])
def test_count_specific(numeric_only):
    eval_general(
        *create_test_dfs(test_data_diff_dtype),
        lambda df: df.count(numeric_only=numeric_only),
    )


@pytest.mark.parametrize("level", [-1, 0, 1])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("data", [test_data["int_data"]])
def test_count_level(data, axis, level):
    modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)

    if axis == 0:
        new_idx = generate_multiindex(len(modin_df.index))
        modin_df.index = new_idx
        pandas_df.index = new_idx
    else:
        new_col = generate_multiindex(len(modin_df.columns))
        modin_df.columns = new_col
        pandas_df.columns = new_col

    eval_general(
        modin_df,
        pandas_df,
        lambda df: df.count(axis=axis, level=level),
    )


@pytest.mark.parametrize("percentiles", [None, 0.10, 0.11, 0.44, 0.78, 0.99])
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_describe(data, percentiles):
    eval_general(
        *create_test_dfs(data),
        lambda df: df.describe(percentiles=percentiles),
    )


@pytest.mark.parametrize("has_numeric_column", [False, True])
@pytest.mark.parametrize("datetime_is_numeric", [True, False, None])
def test_2195(datetime_is_numeric, has_numeric_column):
    data = {
        "categorical": pd.Categorical(["d"] * 10 ** 2),
        "date": [np.datetime64("2000-01-01")] * 10 ** 2,
    }

    if has_numeric_column:
        data.update({"numeric": [5] * 10 ** 2})

    modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)

    eval_general(
        modin_df,
        pandas_df,
        lambda df: df.describe(datetime_is_numeric=datetime_is_numeric),
    )


@pytest.mark.parametrize(
    "exclude,include",
    [
        ([np.float64], None),
        (np.float64, None),
        (None, [np.timedelta64, np.datetime64, np.object, np.bool]),
        (None, "all"),
        (None, np.number),
    ],
)
def test_describe_specific(exclude, include):
    eval_general(
        *create_test_dfs(test_data_diff_dtype),
        lambda df: df.drop("str_col", axis=1).describe(
            exclude=exclude, include=include
        ),
    )


@pytest.mark.parametrize("data", [test_data["int_data"]])
def test_describe_str(data):
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
    data = {
        "col1": list("abc"),
        "col2": list("abc"),
        "col3": list("abc"),
        "col4": [1, 2, 3],
    }
    eval_general(*create_test_dfs(data), lambda df: df.describe())


@pytest.mark.parametrize("method", ["idxmin", "idxmax"])
@pytest.mark.parametrize("is_transposed", [False, True])
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("data", [test_data["float_nan_data"]])
def test_idxmin_idxmax(data, axis, skipna, is_transposed, method):
    eval_general(
        *create_test_dfs(data),
        lambda df: getattr((df.T if is_transposed else df), method)(
            axis=axis, skipna=skipna
        ),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_last_valid_index(data):
    modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)
    assert modin_df.last_valid_index() == pandas_df.last_valid_index()


@pytest.mark.parametrize("index", bool_arg_values, ids=arg_keys("index", bool_arg_keys))
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_memory_usage(data, index):
    eval_general(*create_test_dfs(data), lambda df: df.memory_usage(index=index))


@pytest.mark.parametrize("method", ["min", "max", "mean"])
@pytest.mark.parametrize("is_transposed", [False, True])
@pytest.mark.parametrize(
    "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
)
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("data", [test_data["float_nan_data"]])
def test_min_max_mean(data, axis, skipna, numeric_only, is_transposed, method):
    eval_general(
        *create_test_dfs(data),
        lambda df: getattr((df.T if is_transposed else df), method)(
            axis=axis, skipna=skipna, numeric_only=numeric_only
        ),
    )


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows has a memory issue for large numbers on this test",
)
@pytest.mark.parametrize(
    "method",
    [
        "prod",
        pytest.param(
            "product",
            marks=pytest.mark.skipif(
                pandas.DataFrame.product == pandas.DataFrame.prod
                and pd.DataFrame.product == pd.DataFrame.prod,
                reason="That method was already tested.",
            ),
        ),
    ],
)
@pytest.mark.parametrize("is_transposed", [False, True])
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("data", [test_data["float_nan_data"]])
def test_prod(
    data,
    axis,
    skipna,
    is_transposed,
    method,
):
    eval_general(
        *create_test_dfs(data),
        lambda df, *args, **kwargs: getattr(df.T if is_transposed else df, method)(
            axis=axis,
            skipna=skipna,
        ),
    )

    # test for issue #1953
    arrays = [["1", "1", "2", "2"], ["1", "2", "3", "4"]]
    modin_df = pd.DataFrame(
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], index=arrays
    )
    pandas_df = pandas.DataFrame(
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], index=arrays
    )
    modin_result = modin_df.prod(level=0)
    pandas_result = pandas_df.prod(level=0)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("is_transposed", [False, True])
@pytest.mark.parametrize(
    "skipna", bool_arg_values, ids=arg_keys("skipna", bool_arg_keys)
)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("data", [test_data["float_nan_data"]])
def test_sum(data, axis, skipna, is_transposed):
    eval_general(
        *create_test_dfs(data),
        lambda df: (df.T if is_transposed else df).sum(
            axis=axis,
            skipna=skipna,
        ),
    )

    # test for issue #1953
    arrays = [["1", "1", "2", "2"], ["1", "2", "3", "4"]]
    modin_df = pd.DataFrame(
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], index=arrays
    )
    pandas_df = pandas.DataFrame(
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], index=arrays
    )
    modin_result = modin_df.sum(level=0)
    pandas_result = pandas_df.sum(level=0)
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("fn", ["prod, sum"])
@pytest.mark.parametrize(
    "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
)
@pytest.mark.parametrize(
    "min_count", int_arg_values, ids=arg_keys("min_count", int_arg_keys)
)
def test_sum_prod_specific(fn, min_count, numeric_only):
    eval_general(
        *create_test_dfs(test_data_diff_dtype),
        lambda df: getattr(df, fn)(min_count=min_count, numeric_only=numeric_only),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_sum_single_column(data):
    modin_df = pd.DataFrame(data).iloc[:, [0]]
    pandas_df = pandas.DataFrame(data).iloc[:, [0]]
    df_equals(modin_df.sum(), pandas_df.sum())
    df_equals(modin_df.sum(axis=1), pandas_df.sum(axis=1))


@pytest.mark.parametrize(
    "fn", ["max", "min", "median", "mean", "skew", "kurt", "sem", "std", "var"]
)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize(
    "numeric_only", bool_arg_values, ids=arg_keys("numeric_only", bool_arg_keys)
)
def test_reduction_specific(fn, numeric_only, axis):
    eval_general(
        *create_test_dfs(test_data_diff_dtype),
        lambda df: getattr(df, fn)(numeric_only=numeric_only, axis=axis),
    )
