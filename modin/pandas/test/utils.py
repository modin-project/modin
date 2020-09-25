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
import math
import pandas
from pandas.util.testing import (
    assert_almost_equal,
    assert_frame_equal,
    assert_categorical_equal,
)
import modin.pandas as pd
from modin.utils import to_pandas
from io import BytesIO
import os

random_state = np.random.RandomState(seed=42)

DATASET_SIZE = os.environ.get("MODIN_TEST_DATASET_SIZE", "normal").lower()

DATASET_SIZE_DICT = {
    "small": (2 ** 2, 2 ** 3),
    "normal": (2 ** 6, 2 ** 8),
    "big": (2 ** 7, 2 ** 12),
}

# Size of test dataframes
NCOLS, NROWS = DATASET_SIZE_DICT.get(DATASET_SIZE, DATASET_SIZE_DICT["normal"])

# Range for values for test data
RAND_LOW = 0
RAND_HIGH = 100

# Input data and functions for the tests
# The test data that we will test our code against
test_data = {
    # "empty_data": {},
    # "columns_only": {"col1": [], "col2": [], "col3": [], "col4": [], "col5": []},
    "int_data": {
        "col{}".format(int((i - NCOLS / 2) % NCOLS + 1)): random_state.randint(
            RAND_LOW, RAND_HIGH, size=(NROWS)
        )
        for i in range(NCOLS)
    },
    "float_nan_data": {
        "col{}".format(int((i - NCOLS / 2) % NCOLS + 1)): [
            x
            if (j % 4 == 0 and i > NCOLS // 2) or (j != i and i <= NCOLS // 2)
            else np.NaN
            for j, x in enumerate(
                random_state.uniform(RAND_LOW, RAND_HIGH, size=(NROWS))
            )
        ]
        for i in range(NCOLS)
    },
    # "int_float_object_data": {
    #     "col3": [1, 2, 3, 4],
    #     "col4": [4, 5, 6, 7],
    #     "col1": [8.0, 9.4, 10.1, 11.3],
    #     "col2": ["a", "b", "c", "d"],
    # },
    # "datetime_timedelta_data": {
    #     "col3": [
    #         np.datetime64("2010"),
    #         np.datetime64("2011"),
    #         np.datetime64("2011-06-15T00:00"),
    #         np.datetime64("2009-01-01"),
    #     ],
    #     "col4": [
    #         np.datetime64("2010"),
    #         np.datetime64("2011"),
    #         np.datetime64("2011-06-15T00:00"),
    #         np.datetime64("2009-01-01"),
    #     ],
    #     "col1": [
    #         np.timedelta64(1, "M"),
    #         np.timedelta64(2, "D"),
    #         np.timedelta64(3, "Y"),
    #         np.timedelta64(20, "D"),
    #     ],
    #     "col2": [
    #         np.timedelta64(1, "M"),
    #         np.timedelta64(2, "D"),
    #         np.timedelta64(3, "Y"),
    #         np.timedelta64(20, "D"),
    #     ],
    # },
    # "all_data": {
    #     "col3": 1.0,
    #     "col4": np.datetime64("2011-06-15T00:00"),
    #     "col5": np.array([3] * 4, dtype="int32"),
    #     "col1": "foo",
    #     "col2": True,
    # },
}

# See details in #1403
test_data["int_data"]["index"] = test_data["int_data"].pop(
    "col{}".format(int(NCOLS / 2))
)

for col in test_data["float_nan_data"]:
    for row in range(NROWS // 2):
        if row % 16 == 0:
            test_data["float_nan_data"][col][row] = np.NaN

test_data_values = list(test_data.values())
test_data_keys = list(test_data.keys())

test_bool_data = {
    "col{}".format(int((i - NCOLS / 2) % NCOLS + 1)): random_state.choice(
        [True, False], size=(NROWS)
    )
    for i in range(NCOLS)
}

test_data_resample = {
    "data": {"A": range(12), "B": range(12)},
    "index": pandas.date_range("31/12/2000", periods=12, freq="H"),
}

test_data_with_duplicates = {
    "no_duplicates": {
        "col{}".format(int((i - NCOLS / 2) % NCOLS + 1)): range(NROWS)
        for i in range(NCOLS)
    },
    "all_duplicates": {
        "col{}".format(int((i - NCOLS / 2) % NCOLS + 1)): [
            float(i) for _ in range(NROWS)
        ]
        for i in range(NCOLS)
    },
    "some_duplicates": {
        "col{}".format(int((i - NCOLS / 2) % NCOLS + 1)): [
            i if j % 7 == 0 else x for j, x in enumerate(range(NROWS))
        ]
        for i in range(NCOLS)
    },
    "has_name_column": {
        "name": ["one", "two", "two", "three"],
        "col1": [1, 2, 2, 3],
        "col3": [10, 20, 20, 3],
        "col7": [100, 201, 200, 300],
    },
    "str_columns": {
        "col_str{}".format(int((i - NCOLS / 2) % NCOLS + 1)): [
            "s" + str(x % 5) for x in range(NROWS)
        ]
        for i in range(NCOLS)
    },
}

test_data_with_duplicates["float_nan"] = test_data["float_nan_data"]

test_data_small = {
    "small": {
        "col0": [1, 2, 3, 4],
        "col1": [8.0, 9.4, 10.1, 11.3],
        "col2": [4, 5, 6, 7],
    }
}

test_data_diff_dtype = {
    "int_col": [-5, 2, 7, 16],
    "float_col": [np.NaN, -9.4, 10.1, np.NaN],
    "str_col": ["a", np.NaN, "c", "d"],
    "bool_col": [False, True, True, False],
}

test_data_small_values = list(test_data_small.values())
test_data_small_keys = list(test_data_small.keys())

test_data_with_duplicates_values = list(test_data_with_duplicates.values())
test_data_with_duplicates_keys = list(test_data_with_duplicates.keys())

test_data_categorical = {
    "ordered": pandas.Categorical(list("testdata"), ordered=True),
    "unordered": pandas.Categorical(list("testdata"), ordered=False),
}

test_data_categorical_values = list(test_data_categorical.values())
test_data_categorical_keys = list(test_data_categorical.keys())

numeric_dfs = [
    "empty_data",
    "columns_only",
    "int_data",
    "float_nan_data",
    "with_index_column",
]

no_numeric_dfs = ["datetime_timedelta_data"]

# String test data
test_string_data = {
    "separator data": [
        "abC|DeF,Hik",
        "234,3245.67",
        "gSaf,qWer|Gre",
        "asd3,4sad|",
        np.NaN,
    ]
}

test_string_data_values = list(test_string_data.values())
test_string_data_keys = list(test_string_data.keys())

# List of strings test data
test_string_list_data = {"simple string": [["a"], ["CdE"], ["jDf"], ["werB"]]}

test_string_list_data_values = list(test_string_list_data.values())
test_string_list_data_keys = list(test_string_list_data.keys())

string_seperators = {"empty sep": "", "comma sep": ",", "None sep": None}

string_sep_values = list(string_seperators.values())
string_sep_keys = list(string_seperators.keys())

string_na_rep = {"None na_rep": None, "- na_rep": "-", "nan na_rep": np.NaN}

string_na_rep_values = list(string_na_rep.values())
string_na_rep_keys = list(string_na_rep.keys())

join_type = {"left": "left", "right": "right", "inner": "inner", "outer": "outer"}

join_type_keys = list(join_type.keys())
join_type_values = list(join_type.values())

# Test functions for applymap
test_func = {
    "plus one": lambda x: x + 1,
    "convert to string": lambda x: str(x),
    "square": lambda x: x * x,
    "identity": lambda x: x,
    "return false": lambda x: False,
}
test_func_keys = list(test_func.keys())
test_func_values = list(test_func.values())

numeric_test_funcs = ["plus one", "square"]

# Test functions for query
query_func = {
    "col1 < col2": "col1 < col2",
    "col3 > col4": "col3 > col4",
    "col1 == col2": "col1 == col2",
    "(col2 > col1) and (col1 < col3)": "(col2 > col1) and (col1 < col3)",
}
query_func_keys = list(query_func.keys())
query_func_values = list(query_func.values())

# Test agg functions for apply, agg, and aggregate
agg_func = {
    "sum": "sum",
    "df sum": lambda df: df.sum(),
    "str": str,
    "sum mean": ["sum", "mean"],
    "sum df sum": ["sum", lambda df: df.sum()],
    "should raise TypeError": 1,
}
agg_func_keys = list(agg_func.keys())
agg_func_values = list(agg_func.values())

# For this sort of parameters pandas throws an exception.
# See details in pandas issue 36036.
agg_func_except = {
    "sum sum": ["sum", "sum"],
}
agg_func_except_keys = list(agg_func_except.keys())
agg_func_except_values = list(agg_func_except.values())

numeric_agg_funcs = ["sum mean", "sum sum", "sum df sum"]

udf_func = {
    "return self": lambda df: lambda x, *args, **kwargs: type(x)(x.values),
    "change index": lambda df: lambda x, *args, **kwargs: pandas.Series(
        x.values, index=np.arange(-1, len(x.index) - 1)
    ),
    "return none": lambda df: lambda x, *args, **kwargs: None,
    "return empty": lambda df: lambda x, *args, **kwargs: pandas.Series(),
    "access self": lambda df: lambda x, other, *args, **kwargs: pandas.Series(
        x.values, index=other.index
    ),
}
udf_func_keys = list(udf_func.keys())
udf_func_values = list(udf_func.values())

# Test q values for quantiles
quantiles = {
    "0.25": 0.25,
    "0.5": 0.5,
    "0.75": 0.75,
    "0.66": 0.66,
    "0.01": 0.01,
    "list": [0.25, 0.5, 0.75, 0.66, 0.01],
}
quantiles_keys = list(quantiles.keys())
quantiles_values = list(quantiles.values())

# Test indices for get, set_index, __contains__, insert
indices = {
    "col1": "col1",
    "col2": "col2",
    "A": "A",
    "B": "B",
    "does not exist": "does not exist",
}
indices_keys = list(indices.keys())
indices_values = list(indices.values())

# Test functions for groupby apply
groupby_apply_func = {"sum": lambda df: df.sum(), "negate": lambda df: -df}
groupby_apply_func_keys = list(groupby_apply_func.keys())
groupby_apply_func_values = list(groupby_apply_func.values())

# Test functions for groupby agg
groupby_agg_func = {"min": "min", "max": "max"}
groupby_agg_func_keys = list(groupby_agg_func.keys())
groupby_agg_func_values = list(groupby_agg_func.values())

# Test functions for groupby transform
groupby_transform_func = {
    "add 4": lambda df: df + 4,
    "negatie and minus 10": lambda df: -df - 10,
}
groupby_transform_func_keys = list(groupby_transform_func.keys())
groupby_transform_func_values = list(groupby_transform_func.values())

# Test functions for groupby pipe
groupby_pipe_func = {"sum": lambda df: df.sum()}
groupby_pipe_func_keys = list(groupby_pipe_func.keys())
groupby_pipe_func_values = list(groupby_pipe_func.values())

# END Test input data and functions

# Parametrizations of common kwargs
axis = {
    "over_rows_int": 0,
    "over_rows_str": "rows",
    "over_columns_int": 1,
    "over_columns_str": "columns",
}
axis_keys = list(axis.keys())
axis_values = list(axis.values())

bool_arg = {"True": True, "False": False, "None": None}
bool_arg_keys = list(bool_arg.keys())
bool_arg_values = list(bool_arg.values())

int_arg = {"-5": -5, "-1": -1, "0": 0, "1": 1, "5": 5}
int_arg_keys = list(int_arg.keys())
int_arg_values = list(int_arg.values())

# END parametrizations of common kwargs

json_short_string = """[{"project": "modin"}]"""
json_long_string = """{
        "quiz": {
            "sport": {
                "q1": {
                    "question": "Which one is correct team name in NBA?",
                    "options": [
                        "New York Bulls",
                        "Los Angeles Kings",
                        "Golden State Warriros",
                        "Huston Rocket"
                    ],
                    "answer": "Huston Rocket"
                }
            },
            "maths": {
                "q1": {
                    "question": "5 + 7 = ?",
                    "options": [
                        "10",
                        "11",
                        "12",
                        "13"
                    ],
                    "answer": "12"
                },
                "q2": {
                    "question": "12 - 8 = ?",
                    "options": [
                        "1",
                        "2",
                        "3",
                        "4"
                    ],
                    "answer": "4"
                }
            }
        }
    }"""
json_long_bytes = BytesIO(json_long_string.encode(encoding="UTF-8"))
json_short_bytes = BytesIO(json_short_string.encode(encoding="UTF-8"))


# Text encoding types
encoding_types = [
    "ascii",
    "utf_32",
    "utf_32_be",
    "utf_32_le",
    "utf_16",
    "utf_16_be",
    "utf_16_le",
    "utf_7",
    "utf_8",
    "utf_8_sig",
]


def categories_equals(left, right):
    assert (left.ordered and right.ordered) or (not left.ordered and not right.ordered)
    is_category_ordered = left.ordered
    assert_categorical_equal(left, right, check_category_order=is_category_ordered)


def df_categories_equals(df1, df2):
    if not hasattr(df1, "select_dtypes"):
        if isinstance(df1, pandas.CategoricalDtype):
            return categories_equals(df1, df2)
        elif isinstance(getattr(df1, "dtype"), pandas.CategoricalDtype) and isinstance(
            getattr(df1, "dtype"), pandas.CategoricalDtype
        ):
            return categories_equals(df1.dtype, df2.dtype)
        else:
            return True

    categories_columns = df1.select_dtypes(include="category").columns
    for column in categories_columns:
        is_category_ordered = df1[column].dtype.ordered
        assert_categorical_equal(
            df1[column].values,
            df2[column].values,
            check_dtype=False,
            check_category_order=is_category_ordered,
        )


def df_equals(df1, df2):
    """Tests if df1 and df2 are equal.

    Args:
        df1: (pandas or modin DataFrame or series) dataframe to test if equal.
        df2: (pandas or modin DataFrame or series) dataframe to test if equal.

    Returns:
        True if df1 is equal to df2.
    """
    types_for_almost_equals = (
        pandas.core.indexes.range.RangeIndex,
        pandas.core.indexes.base.Index,
        np.recarray,
    )

    # Gets AttributError if modin's groupby object is not import like this
    from modin.pandas.groupby import DataFrameGroupBy

    groupby_types = (pandas.core.groupby.DataFrameGroupBy, DataFrameGroupBy)

    # The typing behavior of how pandas treats its index is not consistent when the
    # length of the DataFrame or Series is 0, so we just verify that the contents are
    # the same.
    if (
        hasattr(df1, "index")
        and hasattr(df2, "index")
        and len(df1) == 0
        and len(df2) == 0
    ):
        if type(df1).__name__ == type(df2).__name__:
            if hasattr(df1, "name") and hasattr(df2, "name") and df1.name == df2.name:
                return
            if (
                hasattr(df1, "columns")
                and hasattr(df2, "columns")
                and df1.columns.equals(df2.columns)
            ):
                return
        assert False

    if isinstance(df1, (list, tuple)) and all(
        isinstance(d, (pd.DataFrame, pd.Series, pandas.DataFrame, pandas.Series))
        for d in df1
    ):
        assert isinstance(df2, type(df1)), "Different type of collection"
        assert len(df1) == len(df2), "Different length result"
        return (df_equals(d1, d2) for d1, d2 in zip(df1, df2))

    # Convert to pandas
    if isinstance(df1, (pd.DataFrame, pd.Series)):
        df1 = to_pandas(df1)
    if isinstance(df2, (pd.DataFrame, pd.Series)):
        df2 = to_pandas(df2)

    if isinstance(df1, pandas.DataFrame) and isinstance(df2, pandas.DataFrame):
        if (df1.empty and not df2.empty) or (df2.empty and not df1.empty):
            return False
        elif df1.empty and df2.empty and type(df1) != type(df2):
            return False

    if isinstance(df1, pandas.DataFrame) and isinstance(df2, pandas.DataFrame):
        assert_frame_equal(
            df1,
            df2,
            check_dtype=False,
            check_datetimelike_compat=True,
            check_index_type=False,
            check_column_type=False,
            check_categorical=False,
        )
        df_categories_equals(df1, df2)
    elif isinstance(df1, types_for_almost_equals) and isinstance(
        df2, types_for_almost_equals
    ):
        assert_almost_equal(df1, df2, check_dtype=False)
    elif isinstance(df1, pandas.Series) and isinstance(df2, pandas.Series):
        assert_almost_equal(df1, df2, check_dtype=False, check_series_type=False)
    elif isinstance(df1, groupby_types) and isinstance(df2, groupby_types):
        for g1, g2 in zip(df1, df2):
            assert g1[0] == g2[0]
            df_equals(g1[1], g2[1])
    elif (
        isinstance(df1, pandas.Series)
        and isinstance(df2, pandas.Series)
        and df1.empty
        and df2.empty
    ):
        assert all(df1.index == df2.index)
        assert df1.dtypes == df2.dtypes
    elif isinstance(df1, pandas.core.arrays.numpy_.PandasArray):
        assert isinstance(df2, pandas.core.arrays.numpy_.PandasArray)
        assert df1 == df2
    else:
        if df1 != df2:
            np.testing.assert_almost_equal(df1, df2)


def modin_df_almost_equals_pandas(modin_df, pandas_df):
    df_categories_equals(modin_df._to_pandas(), pandas_df)

    modin_df = to_pandas(modin_df)

    if hasattr(modin_df, "select_dtypes"):
        modin_df = modin_df.select_dtypes(exclude=["category"])
    if hasattr(pandas_df, "select_dtypes"):
        pandas_df = pandas_df.select_dtypes(exclude=["category"])

    difference = modin_df - pandas_df
    diff_max = difference.max()
    if isinstance(diff_max, pandas.Series):
        diff_max = diff_max.max()
    assert (
        modin_df.equals(pandas_df)
        or diff_max < 0.0001
        or (all(modin_df.isna().all()) and all(pandas_df.isna().all()))
    )


def df_is_empty(df):
    """Tests if df is empty.

    Args:
        df: (pandas or modin DataFrame) dataframe to test if empty.

    Returns:
        True if df is empty.
    """
    assert df.size == 0 and df.empty
    assert df.shape[0] == 0 or df.shape[1] == 0


def arg_keys(arg_name, keys):
    """Appends arg_name to the front of all values in keys.

    Args:
        arg_name: (string) String containing argument name.
        keys: (list of strings) Possible inputs of argument.

    Returns:
        List of strings with arg_name append to front of keys.
    """
    return ["{0}_{1}".format(arg_name, key) for key in keys]


def name_contains(test_name, vals):
    """Determines if any string in vals is a substring of test_name.

    Args:
        test_name: (string) String to determine if contains substrings.
        vals: (list of strings) List of substrings to test for.

    Returns:
        True if a substring in vals is in test_name, else False.
    """
    return any(val in test_name for val in vals)


def check_df_columns_have_nans(df, cols):
    """Checks if there are NaN values in specified columns of a dataframe.

    :param df: Dataframe to check.
    :param cols: One column name or list of column names.
    :return:
        True if specified columns of dataframe contains NaNs.
    """
    return (
        pandas.api.types.is_list_like(cols)
        and (
            any(isinstance(x, str) and x in df.columns and df[x].hasnans for x in cols)
            or any(
                isinstance(x, pd.Series) and x._parent is df and x.hasnans for x in cols
            )
        )
    ) or (
        not pandas.api.types.is_list_like(cols)
        and cols in df.columns
        and df[cols].hasnans
    )


def eval_general(
    modin_df,
    pandas_df,
    operation,
    comparator=df_equals,
    __inplace__=False,
    check_exception_type=True,
    **kwargs,
):
    md_kwargs, pd_kwargs = {}, {}

    def execute_callable(fn, inplace=False, md_kwargs={}, pd_kwargs={}):
        try:
            pd_result = fn(pandas_df, **pd_kwargs)
        except Exception as pd_e:
            if check_exception_type is None:
                return None
            with pytest.raises(Exception) as md_e:
                # repr to force materialization
                repr(fn(modin_df, **md_kwargs))
            if check_exception_type:
                assert isinstance(md_e.value, type(pd_e))
        else:
            md_result = fn(modin_df, **md_kwargs)
            return (md_result, pd_result) if not __inplace__ else (modin_df, pandas_df)

    for key, value in kwargs.items():
        if callable(value):
            values = execute_callable(value)
            # that means, that callable raised an exception
            if values is None:
                return
            else:
                md_value, pd_value = values
        else:
            md_value, pd_value = value, value

        md_kwargs[key] = md_value
        pd_kwargs[key] = pd_value

    values = execute_callable(
        operation, md_kwargs=md_kwargs, pd_kwargs=pd_kwargs, inplace=__inplace__
    )
    if values is not None:
        comparator(*values)


def create_test_dfs(*args, **kwargs):
    return pd.DataFrame(*args, **kwargs), pandas.DataFrame(*args, **kwargs)


def generate_dfs():
    df = pandas.DataFrame(
        {
            "col1": [0, 1, 2, 3],
            "col2": [4, 5, 6, 7],
            "col3": [8, 9, 10, 11],
            "col4": [12, 13, 14, 15],
            "col5": [0, 0, 0, 0],
        }
    )

    df2 = pandas.DataFrame(
        {
            "col1": [0, 1, 2, 3],
            "col2": [4, 5, 6, 7],
            "col3": [8, 9, 10, 11],
            "col6": [12, 13, 14, 15],
            "col7": [0, 0, 0, 0],
        }
    )
    return df, df2


def generate_multiindex_dfs(axis=1):
    def generate_multiindex(index):
        return pandas.MultiIndex.from_tuples(
            [("a", x) for x in index.values], names=["name1", "name2"]
        )

    df1, df2 = generate_dfs()
    df1.axes[axis], df2.axes[axis] = map(
        generate_multiindex, [df1.axes[axis], df2.axes[axis]]
    )
    return df1, df2


def generate_multiindex(elements_number, nlevels=2, is_tree_like=False):
    def generate_level(length, nlevel):
        src = ["bar", "baz", "foo", "qux"]
        return [src[i % len(src)] + f"-{nlevel}-{i}" for i in range(length)]

    if is_tree_like:
        for penalty_level in [0, 1]:
            lvl_len_f, lvl_len_d = math.modf(
                round(elements_number ** (1 / (nlevels - penalty_level)), 12)
            )
            if lvl_len_d >= 2 and lvl_len_f == 0:
                break

        if lvl_len_d < 2 or lvl_len_f != 0:
            raise RuntimeError(
                f"Can't generate Tree-like MultiIndex with lenght: {elements_number} and number of levels: {nlevels}"
            )

        lvl_len = int(lvl_len_d)
        result = pd.MultiIndex.from_product(
            [generate_level(lvl_len, i) for i in range(nlevels - penalty_level)],
            names=[f"level-{i}" for i in range(nlevels - penalty_level)],
        )
        if penalty_level:
            result = pd.MultiIndex.from_tuples(
                [("base_level", *ml_tuple) for ml_tuple in result],
                names=[f"level-{i}" for i in range(nlevels)],
            )
        return result.sort_values()
    else:
        base_level = ["first"] * (elements_number // 2 + elements_number % 2) + [
            "second"
        ] * (elements_number // 2)
        primary_levels = [generate_level(elements_number, i) for i in range(1, nlevels)]
        arrays = [base_level] + primary_levels
        return pd.MultiIndex.from_tuples(
            list(zip(*arrays)), names=[f"level-{i}" for i in range(nlevels)]
        ).sort_values()


def generate_none_dfs():
    df = pandas.DataFrame(
        {
            "col1": [0, 1, 2, 3],
            "col2": [4, 5, None, 7],
            "col3": [8, 9, 10, 11],
            "col4": [12, 13, 14, 15],
            "col5": [None, None, None, None],
        }
    )

    df2 = pandas.DataFrame(
        {
            "col1": [0, 1, 2, 3],
            "col2": [4, 5, 6, 7],
            "col3": [8, 9, 10, 11],
            "col6": [12, 13, 14, 15],
            "col7": [0, 0, 0, 0],
        }
    )
    return df, df2
