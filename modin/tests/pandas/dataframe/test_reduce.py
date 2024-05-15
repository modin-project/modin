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

import matplotlib
import numpy as np
import pandas
import pytest

import modin.pandas as pd
from modin.config import NPartitions
from modin.tests.pandas.utils import (
    arg_keys,
    axis_keys,
    axis_values,
    bool_arg_keys,
    bool_arg_values,
    create_test_dfs,
    default_to_pandas_ignore_string,
    df_equals,
    df_equals_with_non_stable_indices,
    eval_general,
    int_arg_keys,
    int_arg_values,
    test_data,
    test_data_diff_dtype,
    test_data_keys,
    test_data_large_categorical_dataframe,
    test_data_values,
)

NPartitions.put(4)

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")

# Our configuration in pytest.ini requires that we explicitly catch all
# instances of defaulting to pandas, but some test modules, like this one,
# have too many such instances.
pytestmark = pytest.mark.filterwarnings(default_to_pandas_ignore_string)


@pytest.mark.parametrize("method", ["all", "any"])
@pytest.mark.parametrize("is_transposed", [False, True])
@pytest.mark.parametrize("skipna", [False, True])
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


@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize(
    "data", [test_data["float_nan_data"], test_data_large_categorical_dataframe]
)
def test_count(data, axis):
    eval_general(
        *create_test_dfs(data),
        lambda df: df.count(axis=axis),
    )


@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("dropna", [True, False])
def test_nunique(data, axis, dropna):
    eval_general(
        *create_test_dfs(data),
        lambda df: df.nunique(axis=axis, dropna=dropna),
    )


@pytest.mark.parametrize("numeric_only", [False, True])
def test_count_specific(numeric_only):
    eval_general(
        *create_test_dfs(test_data_diff_dtype),
        lambda df: df.count(numeric_only=numeric_only),
    )


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_count_dtypes(data):
    modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)

    eval_general(
        modin_df,
        pandas_df,
        lambda df: df.isna().count(axis=0),
    )


@pytest.mark.parametrize("percentiles", [None, 0.10, 0.11, 0.44, 0.78, 0.99])
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_describe(data, percentiles):
    if percentiles is not None:
        percentiles = [percentiles]
    eval_general(
        *create_test_dfs(data),
        lambda df: df.describe(percentiles=percentiles),
    )


@pytest.mark.parametrize("has_numeric_column", [False, True])
def test_2195(has_numeric_column):
    data = {
        "categorical": pd.Categorical(["d"] * 10**2),
        "date": [np.datetime64("2000-01-01")] * 10**2,
    }

    if has_numeric_column:
        data.update({"numeric": [5] * 10**2})

    modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)

    eval_general(
        modin_df,
        pandas_df,
        lambda df: df.describe(),
    )


# Issue: https://github.com/modin-project/modin/issues/4641
def test_describe_column_partition_has_different_index():
    pandas_df = pandas.DataFrame(test_data["int_data"])
    # We add a string column to test the case where partitions with mixed data
    # types have different 'describe' rows, which causes an index mismatch.
    pandas_df["string_column"] = "abc"
    modin_df = pd.DataFrame(pandas_df)
    eval_general(modin_df, pandas_df, lambda df: df.describe(include="all"))


@pytest.mark.parametrize(
    "exclude,include",
    [
        ([np.float64], None),
        (np.float64, None),
        (None, [np.timedelta64, np.datetime64, np.object_, np.bool_]),
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
@pytest.mark.parametrize("skipna", [False, True])
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("data", [test_data["float_nan_data"]])
def test_idxmin_idxmax(data, axis, skipna, is_transposed, method):
    eval_general(
        *create_test_dfs(data),
        lambda df: getattr((df.T if is_transposed else df), method)(
            axis=axis, skipna=skipna
        ),
    )


@pytest.mark.parametrize("axis", [0, 1])
def test_idxmin_idxmax_string_columns(axis):
    # https://github.com/modin-project/modin/issues/7093
    modin_df, pandas_df = create_test_dfs([["a", "b"]])
    eval_general(modin_df, pandas_df, lambda df: df.idxmax(axis=axis))
    eval_general(modin_df, pandas_df, lambda df: df.idxmin(axis=axis))


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
@pytest.mark.parametrize("numeric_only", [False, True])
@pytest.mark.parametrize("skipna", [False, True])
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("data", [test_data["float_nan_data"]])
def test_min_max_mean(data, axis, skipna, numeric_only, is_transposed, method):
    eval_general(
        *create_test_dfs(data),
        lambda df: getattr((df.T if is_transposed else df), method)(
            axis=axis, skipna=skipna, numeric_only=numeric_only
        ),
    )


@pytest.mark.parametrize("method", ["prod", "product"])
@pytest.mark.parametrize("is_transposed", [False, True])
@pytest.mark.parametrize("skipna", [False, True])
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
    modin_result = modin_df.prod()
    pandas_result = pandas_df.prod()
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("is_transposed", [False, True])
@pytest.mark.parametrize("skipna", [False, True])
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("data", [test_data["float_nan_data"]])
def test_sum(data, axis, skipna, is_transposed, request):
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
    modin_result = modin_df.sum()
    pandas_result = pandas_df.sum()
    df_equals(modin_result, pandas_result)


@pytest.mark.parametrize("dtype", ["int64", "Int64", "int64[pyarrow]"])
def test_dtype_consistency(dtype):
    # test for issue #6781
    res_dtype = pd.DataFrame([1, 2, 3, 4], dtype=dtype).sum().dtype
    assert res_dtype == pandas.api.types.pandas_dtype(dtype)


@pytest.mark.parametrize("fn", ["prod", "sum"])
@pytest.mark.parametrize("numeric_only", [False, True])
@pytest.mark.parametrize(
    "min_count", int_arg_values, ids=arg_keys("min_count", int_arg_keys)
)
def test_sum_prod_specific(fn, min_count, numeric_only):
    expected_exception = None
    if not numeric_only and fn == "prod":
        # FIXME: https://github.com/modin-project/modin/issues/7029
        expected_exception = False
    elif not numeric_only and fn == "sum":
        expected_exception = TypeError('can only concatenate str (not "int") to str')
    if numeric_only and fn == "sum":
        pytest.xfail(reason="https://github.com/modin-project/modin/issues/7029")
    if min_count == 5 and not numeric_only:
        pytest.xfail(reason="https://github.com/modin-project/modin/issues/7029")

    eval_general(
        *create_test_dfs(test_data_diff_dtype),
        lambda df: getattr(df, fn)(min_count=min_count, numeric_only=numeric_only),
        expected_exception=expected_exception,
    )


@pytest.mark.parametrize("backend", [None, "pyarrow"])
def test_sum_prod_min_count(backend):
    md_df, pd_df = create_test_dfs(test_data["float_nan_data"], backend=backend)
    eval_general(md_df, pd_df, lambda df: df.prod(min_count=len(pd_df) + 1))


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_sum_single_column(data):
    modin_df = pd.DataFrame(data).iloc[:, [0]]
    pandas_df = pandas.DataFrame(data).iloc[:, [0]]
    df_equals(modin_df.sum(), pandas_df.sum())
    df_equals(modin_df.sum(axis=1), pandas_df.sum(axis=1))


def test_sum_datetime64():
    pd_ser = pandas.date_range(start="1/1/2018", end="1/08/2018")
    modin_df, pandas_df = create_test_dfs({"A": pd_ser, "B": [1, 2, 3, 4, 5, 6, 7, 8]})
    eval_general(
        modin_df,
        pandas_df,
        lambda df: df.sum(),
        expected_exception=TypeError(
            "'DatetimeArray' with dtype datetime64[ns] does not support reduction 'sum'"
        ),
    )


def test_min_datetime64():
    pd_ser = pandas.date_range(start="1/1/2018", end="1/08/2018")
    modin_df, pandas_df = create_test_dfs({"A": pd_ser, "B": [1, 2, 3, 4, 5, 6, 7, 8]})
    eval_general(
        modin_df,
        pandas_df,
        lambda df: df.min(),
    )

    eval_general(
        modin_df,
        pandas_df,
        lambda df: df.min(axis=1),
        # pandas raises: `TypeError: '<=' not supported between instances of 'Timestamp' and 'int'`
        # while modin raises quite general: `TypeError("Cannot compare Numeric and Non-Numeric Types")`
        expected_exception=False,
    )


@pytest.mark.parametrize(
    "fn", ["max", "min", "median", "mean", "skew", "kurt", "sem", "std", "var"]
)
@pytest.mark.parametrize("axis", [0, 1, None])
@pytest.mark.parametrize("numeric_only", [False, True])
def test_reduce_specific(fn, numeric_only, axis):
    expected_exception = None
    if not numeric_only:
        if fn in ("max", "min"):
            if axis == 0:
                operator = ">=" if fn == "max" else "<="
                expected_exception = TypeError(
                    f"'{operator}' not supported between instances of 'str' and 'float'"
                )
            else:
                # FIXME: https://github.com/modin-project/modin/issues/7030
                expected_exception = False
        elif fn in ("skew", "kurt", "sem", "std", "var", "median", "mean"):
            # FIXME: https://github.com/modin-project/modin/issues/7030
            expected_exception = False

    eval_general(
        *create_test_dfs(test_data_diff_dtype),
        lambda df: getattr(df, fn)(numeric_only=numeric_only, axis=axis),
        expected_exception=expected_exception,
    )


@pytest.mark.parametrize("subset_len", [1, 2])
@pytest.mark.parametrize("sort", bool_arg_values, ids=bool_arg_keys)
@pytest.mark.parametrize("normalize", bool_arg_values, ids=bool_arg_keys)
@pytest.mark.parametrize("dropna", bool_arg_values, ids=bool_arg_keys)
@pytest.mark.parametrize("ascending", [False, True])
def test_value_counts(subset_len, sort, normalize, dropna, ascending):
    def comparator(md_res, pd_res):
        if subset_len == 1:
            # 'pandas.DataFrame.value_counts' always returns frames with MultiIndex,
            # even when 'subset_len == 1' it returns MultiIndex with 'nlevels == 1'.
            # This behavior is expensive to mimic, so Modin 'value_counts' returns frame
            # with non-multi index in that case. That's why we flatten indices here.
            assert md_res.index.nlevels == pd_res.index.nlevels == 1
            for df in [md_res, pd_res]:
                df.index = df.index.get_level_values(0)

        if sort:
            # We sort indices for the result because of:
            # https://github.com/modin-project/modin/issues/1650
            df_equals_with_non_stable_indices(md_res, pd_res)
        else:
            df_equals(md_res.sort_index(), pd_res.sort_index())

    data = test_data_values[0]
    md_df, pd_df = create_test_dfs(data)
    # We're picking columns with different index signs to involve columns from different partitions
    subset = [pd_df.columns[-i if i % 2 else i] for i in range(subset_len)]

    eval_general(
        md_df,
        pd_df,
        lambda df: df.value_counts(
            subset=subset,
            sort=sort,
            normalize=normalize,
            dropna=dropna,
            ascending=ascending,
        ),
        comparator=comparator,
    )


def test_value_counts_categorical():
    # from issue #3571
    data = np.array(["a"] * 50000 + ["b"] * 10000 + ["c"] * 1000)
    random_state = np.random.RandomState(seed=42)
    random_state.shuffle(data)
    modin_df, pandas_df = create_test_dfs(
        {"col1": data, "col2": data}, dtype="category"
    )
    eval_general(
        modin_df,
        pandas_df,
        lambda df: df.value_counts(),
        comparator=df_equals,
    )
