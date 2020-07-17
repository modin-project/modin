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
import copy
import numpy as np
import pandas
from pandas.util.testing import (
    assert_almost_equal,
    assert_frame_equal,
    assert_categorical_equal,
)
import modin.pandas as pd
from modin.pandas.utils import to_pandas
from io import BytesIO

random_state = np.random.RandomState(seed=42)

# Size of test dataframes
NCOLS = 2 ** 6
NROWS = 2 ** 8

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
    "float_data": {
        "col{}".format(int((i - NCOLS / 2) % NCOLS + 1)): random_state.uniform(
            RAND_LOW, RAND_HIGH, size=(NROWS)
        )
        for i in range(NCOLS)
    },
    "sparse_nan_data": {
        "col{}".format(int((i - NCOLS / 2) % NCOLS + 1)): [
            x if j != i else np.NaN
            for j, x in enumerate(
                random_state.uniform(RAND_LOW, RAND_HIGH, size=(NROWS))
            )
        ]
        for i in range(NCOLS)
    },
    "dense_nan_data": {
        "col{}".format(int((i - NCOLS / 2) % NCOLS + 1)): [
            x if j % 4 == 0 else np.NaN
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
    "100x100": {
        "col{}".format((i - 50) % 100 + 1): random_state.randint(
            RAND_LOW, RAND_HIGH, size=(100)
        )
        for i in range(100)
    },
}

# Create a dataframe based on integer dataframe but with one column called "index". Because of bug #1481 it cannot be
# created in normal way and has to be copied from dataset that works.
# TODO(gshimansky): when bug #1481 is fixed replace this dataframe initialization with ordinary one.
test_data["with_index_column"] = copy.copy(test_data["int_data"])
test_data["with_index_column"]["index"] = test_data["with_index_column"].pop(
    "col{}".format(int(NCOLS / 2))
)

test_data_values = list(test_data.values())
test_data_keys = list(test_data.keys())

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
    "subset_duplicates": {
        "col{}".format(i): [
            i if j % 7 == 0 and i in [1, 3, 7] else x
            for j, x in enumerate(range(NROWS))
        ]
        for i in range(NCOLS)
    },
    "has_name_column": {
        "name": ["one", "two", "two", "three"],
        "col1": [1, 2, 2, 3],
        "col3": [10, 20, 20, 3],
        "col7": [100, 201, 200, 300],
    },
}

test_data_with_duplicates_values = list(test_data_with_duplicates.values())
test_data_with_duplicates_keys = list(test_data_with_duplicates.keys())

numeric_dfs = [
    "empty_data",
    "columns_only",
    "int_data",
    "float_data",
    "sparse_nan_data",
    "dense_nan_data",
    "with_index_column",
    "100x100",
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
    "sum sum": ["sum", "sum"],
    "sum df sum": ["sum", lambda df: df.sum()],
    "should raise TypeError": 1,
}
agg_func_keys = list(agg_func.keys())
agg_func_values = list(agg_func.values())

numeric_agg_funcs = ["sum mean", "sum sum", "sum df sum"]

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
        try:
            assert_frame_equal(
                df1.sort_index(axis=1),
                df2.sort_index(axis=1),
                check_dtype=False,
                check_datetimelike_compat=True,
                check_index_type=False,
                check_column_type=False,
                check_categorical=False,
            )
        except Exception:
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
    else:
        if df1 != df2:
            np.testing.assert_almost_equal(df1, df2)


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
    return ["{0} {1}".format(arg_name, key) for key in keys]


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
        and any(x in df.columns and df[x].hasnans for x in cols)
        or not pandas.api.types.is_list_like(cols)
        and cols in df.columns
        and df[cols].hasnans
    )


def eval_general(modin_df, pandas_df, operation, comparator=df_equals, **kwargs):
    md_kwargs, pd_kwargs = {}, {}

    def execute_callable(fn, md_kwargs={}, pd_kwargs={}):
        try:
            pd_result = fn(pandas_df, **pd_kwargs)
        except Exception as e:
            with pytest.raises(type(e)):
                # repr to force materialization
                repr(fn(modin_df, **md_kwargs))
        else:
            md_result = fn(modin_df, **md_kwargs)
            return md_result, pd_result

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

    values = execute_callable(operation, md_kwargs=md_kwargs, pd_kwargs=pd_kwargs)
    if values is not None:
        comparator(*values)


def create_test_dfs(*args, **kwargs):
    return pd.DataFrame(*args, **kwargs), pandas.DataFrame(*args, **kwargs)
