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

import csv
import functools
import itertools
import math
import os
import re
from io import BytesIO
from pathlib import Path
from string import ascii_letters
from typing import Union

import numpy as np
import pandas
import psutil
import pytest
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_list_like,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
    is_timedelta64_dtype,
)

import modin.pandas as pd
from modin.config import (
    Engine,
    MinColumnPartitionSize,
    MinRowPartitionSize,
    NPartitions,
    RangePartitioning,
    StorageFormat,
    TestDatasetSize,
    TrackFileLeaks,
)
from modin.pandas.io import to_pandas
from modin.pandas.testing import (
    assert_extension_array_equal,
    assert_frame_equal,
    assert_index_equal,
    assert_series_equal,
)
from modin.utils import try_cast_to_pandas

random_state = np.random.RandomState(seed=42)

DATASET_SIZE_DICT = {
    "Small": (2**6, 2**6),
    "Normal": (2**6, 2**8),
    "Big": (2**7, 2**12),
}

# Size of test dataframes
NCOLS, NROWS = DATASET_SIZE_DICT.get(TestDatasetSize.get(), DATASET_SIZE_DICT["Normal"])
NGROUPS = 10

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
            (
                x
                if (j % 4 == 0 and i > NCOLS // 2) or (j != i and i <= NCOLS // 2)
                else np.nan
            )
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
# The parse_dates param can take several different types and combinations of
# types. Use the following values to test date parsing on a CSV created for
# that purpose at `time_parsing_csv_path`
parse_dates_values_by_id = {
    "bool": False,
    "list_of_single_int": [0],
    "list_of_single_string": ["timestamp"],
    "list_of_list_of_strings": [["year", "month", "date"]],
    "list_of_string_and_list_of_strings": ["timestamp", ["year", "month", "date"]],
    "list_of_list_of_ints": [[1, 2, 3]],
    "list_of_list_of_strings_and_ints": [["year", 2, "date"]],
    "empty_list": [],
    "dict": {"year_and_month": [1, 2], "day": ["date"]},
    "nonexistent_string_column": ["z"],
    "nonexistent_int_column": [99],
}

# See details in #1403
test_data["int_data"]["index"] = test_data["int_data"].pop(
    "col{}".format(int(NCOLS / 2))
)

for col in test_data["float_nan_data"]:
    for row in range(NROWS // 2):
        if row % 16 == 0:
            test_data["float_nan_data"][col][row] = np.nan

test_data_values = list(test_data.values())
test_data_keys = list(test_data.keys())

test_bool_data = {
    "col{}".format(int((i - NCOLS / 2) % NCOLS + 1)): random_state.choice(
        [True, False], size=(NROWS)
    )
    for i in range(NCOLS)
}

test_groupby_data = {f"col{i}": np.arange(NCOLS) % NGROUPS for i in range(NROWS)}

test_data_resample = {
    "data": {
        f"col{i}": random_state.randint(RAND_LOW, RAND_HIGH, size=NROWS)
        for i in range(10)
    },
    "index": pandas.date_range("31/12/2000", periods=NROWS, freq="h"),
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
    "float_col": [np.nan, -9.4, 10.1, np.nan],
    "str_col": ["a", np.nan, "c", "d"],
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

# Fully fill all of the partitions used in tests.
test_data_large_categorical_dataframe = {
    i: pandas.Categorical(np.arange(NPartitions.get() * MinRowPartitionSize.get()))
    for i in range(NPartitions.get() * MinColumnPartitionSize.get())
}
test_data_large_categorical_series_values = [
    pandas.Categorical(np.arange(NPartitions.get() * MinRowPartitionSize.get()))
]
test_data_large_categorical_series_keys = ["categorical_series"]

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
        np.nan,
    ]
}

test_string_data_values = list(test_string_data.values())
test_string_data_keys = list(test_string_data.keys())

# List of strings test data
test_string_list_data = {"simple string": [["a"], ["CdE"], ["jDf"], ["werB"]]}

test_string_list_data_values = list(test_string_list_data.values())
test_string_list_data_keys = list(test_string_list_data.keys())

string_seperators = {"comma sep": ","}

string_sep_values = list(string_seperators.values())
string_sep_keys = list(string_seperators.keys())

string_na_rep = {"None na_rep": None, "- na_rep": "-", "nan na_rep": np.nan}

string_na_rep_values = list(string_na_rep.values())
string_na_rep_keys = list(string_na_rep.keys())

join_type = {"left": "left", "right": "right", "inner": "inner", "outer": "outer"}

join_type_keys = list(join_type.keys())
join_type_values = list(join_type.values())

# Test functions for applymap
test_func = {
    "plus one": lambda x: x + 1,
    "convert to string": str,
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
    # this is how to query for values of an unnamed index per
    # https://pandas.pydata.org/docs/user_guide/indexing.html#multiindex-query-syntax
    "ilevel_0 % 2 == 1": "ilevel_0 % 2 == 1",
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
    # The case verifies that returning a scalar that is based on a frame's data doesn't cause a problem
    "sum of certain elements": lambda axis: (
        axis.iloc[0] + axis.iloc[-1] if isinstance(axis, pandas.Series) else axis + axis
    ),
    "should raise AssertionError": 1,
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
    "return self": lambda x, *args, **kwargs: type(x)(x.values),
    "change index": lambda x, *args, **kwargs: pandas.Series(
        x.values, index=np.arange(-1, len(x.index) - 1)
    ),
    "return none": lambda x, *args, **kwargs: None,
    "return empty": lambda x, *args, **kwargs: pandas.Series(),
    "access self": lambda x, other, *args, **kwargs: pandas.Series(
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

default_to_pandas_ignore_string = "default:.*defaulting to pandas.*:UserWarning"

# Files compression to extension mapping
COMP_TO_EXT = {"gzip": "gz", "bz2": "bz2", "xz": "xz", "zip": "zip"}


time_parsing_csv_path = "modin/tests/pandas/data/test_time_parsing.csv"


class CustomIntegerForAddition:
    def __init__(self, value: int):
        self.value = value

    def __add__(self, other):
        return self.value + other

    def __radd__(self, other):
        return other + self.value


class NonCommutativeMultiplyInteger:
    """int-like class with non-commutative multiply operation.

    We need to test that rmul and mul do different things even when
    multiplication is not commutative, but almost all multiplication is
    commutative. This class' fake multiplication overloads are not commutative
    when you multiply an instance of this class with pandas.series, which
    does not know how to __mul__ with this class. e.g.

    NonCommutativeMultiplyInteger(2) * pd.Series(1, dtype=int) == pd.Series(2, dtype=int)
    pd.Series(1, dtype=int) * NonCommutativeMultiplyInteger(2) == pd.Series(3, dtype=int)
    """

    def __init__(self, value: int):
        if not isinstance(value, int):
            raise TypeError(
                f"must initialize with integer, but got {value} of type {type(value)}"
            )
        self.value = value

    def __mul__(self, other):
        # Note that we need to check other is an int, otherwise when we (left) mul
        # this with a series, we'll just multiply self.value by the series, whereas
        # we want to make the series do an rmul instead.
        if not isinstance(other, int):
            return NotImplemented
        return self.value * other

    def __rmul__(self, other):
        return self.value * other + 1


def categories_equals(left, right):
    assert (left.ordered and right.ordered) or (not left.ordered and not right.ordered)
    assert_extension_array_equal(left, right)


def df_categories_equals(df1, df2):
    if not hasattr(df1, "select_dtypes"):
        if isinstance(df1, pandas.CategoricalDtype):
            categories_equals(df1, df2)
        elif isinstance(getattr(df1, "dtype"), pandas.CategoricalDtype) and isinstance(
            getattr(df2, "dtype"), pandas.CategoricalDtype
        ):
            categories_equals(df1.dtype, df2.dtype)
        return True

    df1_categorical = df1.select_dtypes(include="category")
    df2_categorical = df2.select_dtypes(include="category")
    assert df1_categorical.columns.equals(df2_categorical.columns)
    # Use an index instead of a column name to iterate through columns. There
    # may be duplicate colum names. e.g. if two columns are named col1,
    # selecting df1_categorical["col1"] gives a dataframe of width 2 instead of a series.
    for i in range(len(df1_categorical.columns)):
        assert_extension_array_equal(
            df1_categorical.iloc[:, i].values,
            df2_categorical.iloc[:, i].values,
            check_dtype=False,
        )


def assert_empty_frame_equal(df1, df2):
    """
    Test if df1 and df2 are empty.

    Parameters
    ----------
    df1 : pandas.DataFrame or pandas.Series
    df2 : pandas.DataFrame or pandas.Series

    Raises
    ------
    AssertionError
        If check fails.
    """

    if (df1.empty and not df2.empty) or (df2.empty and not df1.empty):
        assert False, "One of the passed frames is empty, when other isn't"
    elif df1.empty and df2.empty and type(df1) is not type(df2):
        assert False, f"Empty frames have different types: {type(df1)} != {type(df2)}"


def assert_all_act_same(condition, *objs):
    """
    Assert that all of the objs give the same boolean result for the passed condition (either all True or all False).

    Parameters
    ----------
    condition : callable(obj) -> bool
        Condition to run on the passed objects.
    *objs :
        Objects to pass to the condition.

    Returns
    -------
    bool
        Result of the condition.
    """
    results = [condition(obj) for obj in objs]
    if len(results) < 2:
        return results[0] if len(results) else None

    assert all(results[0] == res for res in results[1:])
    return results[0]


def assert_dtypes_equal(df1, df2):
    """
    Assert that the two passed DataFrame/Series objects have equal dtypes.

    The function doesn't require that the dtypes are identical, it has the following reliefs:
        1. The dtypes are not required to be in the same order
           (e.g. {"col1": int, "col2": float} == {"col2": float, "col1": int})
        2. The dtypes are only required to be in the same class
           (e.g. both numerical, both categorical, etc...)

    Parameters
    ----------
    df1 : DataFrame or Series
    df2 : DataFrame or Series
    """
    if not isinstance(
        df1, (pandas.Series, pd.Series, pandas.DataFrame, pd.DataFrame)
    ) or not isinstance(
        df2, (pandas.Series, pd.Series, pandas.DataFrame, pd.DataFrame)
    ):
        return

    if isinstance(df1.dtypes, (pandas.Series, pd.Series)):
        dtypes1 = df1.dtypes
        dtypes2 = df2.dtypes
    else:
        # Case when `dtypes` is a scalar
        dtypes1 = pandas.Series({"col": df1.dtypes})
        dtypes2 = pandas.Series({"col": df2.dtypes})

    # Don't require for dtypes to be in the same order
    assert len(dtypes1.index.difference(dtypes2.index)) == 0
    assert len(dtypes1) == len(dtypes2)

    dtype_comparators = (
        is_numeric_dtype,
        lambda obj: is_object_dtype(obj) or is_string_dtype(obj),
        is_bool_dtype,
        lambda obj: isinstance(obj, pandas.CategoricalDtype),
        is_datetime64_any_dtype,
        is_timedelta64_dtype,
        lambda obj: isinstance(obj, pandas.PeriodDtype),
    )

    for idx in range(len(dtypes1)):
        for comparator in dtype_comparators:
            if assert_all_act_same(comparator, dtypes1.iloc[idx], dtypes2.iloc[idx]):
                # We met a dtype that both types satisfy, so we can stop iterating
                # over comparators and compare next dtypes
                break


def assert_set_of_rows_identical(df1, df2):
    """
    Assert that the set of rows for the passed dataframes is identical.

    Works much slower than ``df1.equals(df2)``, so it's recommended to use this
    function only in exceptional cases.
    """
    # replacing NaN with None to pass the comparison: 'NaN == NaN -> false; None == None -> True'
    df1, df2 = map(
        lambda df: (df.to_frame() if df.ndim == 1 else df).replace({np.nan: None}),
        (df1, df2),
    )
    rows1 = set((idx, *row.tolist()) for idx, row in df1.iterrows())
    rows2 = set((idx, *row.tolist()) for idx, row in df2.iterrows())
    assert rows1 == rows2


def sort_data(data):
    """Sort the passed sequence."""
    if isinstance(data, (pandas.DataFrame, pd.DataFrame)):
        return data.sort_values(data.columns.to_list(), ignore_index=True)
    elif isinstance(data, (pandas.Series, pd.Series)):
        return data.sort_values()
    else:
        return np.sort(data)


def sort_if_range_partitioning(df1, df2, comparator=None, force=False):
    """Sort the passed objects if 'RangePartitioning' is enabled and compare the sorted results."""
    if comparator is None:
        comparator = df_equals

    if force or RangePartitioning.get():
        df1, df2 = sort_data(df1), sort_data(df2)

    comparator(df1, df2)


def df_equals(df1, df2, check_dtypes=True):
    """Tests if df1 and df2 are equal.

    Args:
        df1: (pandas or modin DataFrame or series) dataframe to test if equal.
        df2: (pandas or modin DataFrame or series) dataframe to test if equal.

    Returns:
        True if df1 is equal to df2.
    """
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

    if check_dtypes:
        assert_dtypes_equal(df1, df2)

    # Convert to pandas
    if isinstance(df1, (pd.DataFrame, pd.Series)):
        df1 = to_pandas(df1)
    if isinstance(df2, (pd.DataFrame, pd.Series)):
        df2 = to_pandas(df2)

    if isinstance(df1, pandas.DataFrame) and isinstance(df2, pandas.DataFrame):
        assert_empty_frame_equal(df1, df2)

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
    elif isinstance(df1, pandas.Index) and isinstance(df2, pandas.Index):
        assert_index_equal(df1, df2)
    elif isinstance(df1, pandas.Series) and isinstance(df2, pandas.Series):
        assert_series_equal(df1, df2, check_dtype=False, check_series_type=False)
    elif (
        hasattr(df1, "dtype")
        and hasattr(df2, "dtype")
        and isinstance(df1.dtype, pandas.core.dtypes.dtypes.ExtensionDtype)
        and isinstance(df2.dtype, pandas.core.dtypes.dtypes.ExtensionDtype)
    ):
        assert_extension_array_equal(df1, df2)
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
    elif isinstance(df1, pandas.core.arrays.NumpyExtensionArray):
        assert isinstance(df2, pandas.core.arrays.NumpyExtensionArray)
        assert df1 == df2
    elif isinstance(df1, np.recarray) and isinstance(df2, np.recarray):
        np.testing.assert_array_equal(df1, df2)
    else:
        res = df1 != df2
        if res.any() if isinstance(res, np.ndarray) else res:
            np.testing.assert_almost_equal(df1, df2)


def modin_df_almost_equals_pandas(modin_df, pandas_df, max_diff=0.0001):
    df_categories_equals(modin_df._to_pandas(), pandas_df)

    modin_df = to_pandas(modin_df)

    if hasattr(modin_df, "select_dtypes"):
        modin_df = modin_df.select_dtypes(exclude=["category"])
    if hasattr(pandas_df, "select_dtypes"):
        pandas_df = pandas_df.select_dtypes(exclude=["category"])

    if modin_df.equals(pandas_df):
        return

    isna = modin_df.isna().all()
    if isinstance(isna, bool):
        if isna:
            assert pandas_df.isna().all()
            return
    elif isna.all():
        assert pandas_df.isna().all().all()
        return

    diff = (modin_df - pandas_df).abs()
    diff /= pandas_df.abs()
    diff_max = diff.max() if isinstance(diff, pandas.Series) else diff.max().max()
    assert diff_max < max_diff, f"{diff_max} >= {max_diff}"


def try_modin_df_almost_equals_compare(df1, df2):
    """Compare two dataframes as nearly equal if possible, otherwise compare as completely equal."""
    # `modin_df_almost_equals_pandas` is numeric-only comparator
    dtypes1, dtypes2 = [
        dtype if is_list_like(dtype := df.dtypes) else [dtype] for df in (df1, df2)
    ]
    if all(map(is_numeric_dtype, dtypes1)) and all(map(is_numeric_dtype, dtypes2)):
        modin_df_almost_equals_pandas(df1, df2)
    else:
        df_equals(df1, df2)


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


class NoModinException(Exception):
    pass


def eval_general(
    modin_df,
    pandas_df,
    operation,
    comparator=df_equals,
    __inplace__=False,
    expected_exception=None,
    check_kwargs_callable=True,
    md_extra_kwargs=None,
    comparator_kwargs=None,
    check_for_execution_propagation=True,
    no_check_for_execution_propagation_reason=None,
    **kwargs,
):
    md_kwargs, pd_kwargs = {}, {}

    if isinstance(modin_df, (pd.DataFrame, pd.Series)):
        original_engine = modin_df._query_compiler.engine
        original_storage_format = modin_df._query_compiler.storage_format
    else:
        original_engine = None
        original_storage_format = None

    def execute_callable(fn, inplace=False, md_kwargs={}, pd_kwargs={}):
        try:
            pd_result = fn(pandas_df, **pd_kwargs)
        except Exception as pd_e:
            try:
                if inplace:
                    _ = fn(modin_df, **md_kwargs)
                    try_cast_to_pandas(modin_df)  # force materialization
                else:
                    try_cast_to_pandas(
                        fn(modin_df, **md_kwargs)
                    )  # force materialization
            except Exception as md_e:
                assert isinstance(
                    md_e, type(pd_e)
                ), "Got Modin Exception type {}, but pandas Exception type {} was expected".format(
                    type(md_e), type(pd_e)
                )
                if expected_exception:
                    if Engine.get() == "Ray":
                        from ray.exceptions import RayTaskError

                        # unwrap ray exceptions from remote worker
                        if isinstance(md_e, RayTaskError):
                            md_e = md_e.args[0]
                    assert (
                        type(md_e) is type(expected_exception)
                        and md_e.args == expected_exception.args
                    ), f"not acceptable Modin's exception: [{repr(md_e)}]"
                    assert (
                        pd_e.args == expected_exception.args
                    ), f"not acceptable Pandas' exception: [{repr(pd_e)}]"
                elif expected_exception is False:
                    # The only way to disable exception message checking.
                    pass
                else:
                    # It’s not enough that Modin and pandas have the same types of exceptions;
                    # we need to explicitly specify the instance of an exception
                    # (using `expected_exception`) in tests so that we can check exception messages.
                    # This allows us to eliminate situations where exceptions are thrown
                    # that we don't expect, which could hide different bugs.
                    raise pd_e
            else:
                raise NoModinException(
                    f"Modin doesn't throw an exception, while pandas does: [{repr(pd_e)}]"
                )
        else:
            md_result = fn(modin_df, **md_kwargs)
            return (md_result, pd_result) if not inplace else (modin_df, pandas_df)

    for key, value in kwargs.items():
        if check_kwargs_callable and callable(value):
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

        if md_extra_kwargs:
            assert isinstance(md_extra_kwargs, dict)
            md_kwargs.update(md_extra_kwargs)

    values = execute_callable(
        operation, md_kwargs=md_kwargs, pd_kwargs=pd_kwargs, inplace=__inplace__
    )
    if values is not None:
        assert isinstance(values, tuple) and len(values) == 2
        modin_result, pandas_result = values
        if (
            isinstance(modin_result, (pd.DataFrame, pd.Series))
            and original_engine is not None
            and original_storage_format is not None
        ):
            if check_for_execution_propagation:
                assert modin_result._query_compiler.engine == original_engine, (
                    f"Result engine {modin_result._query_compiler.engine} does "
                    + f"not match expected engine {original_engine}"
                )
                assert (
                    modin_result._query_compiler.storage_format
                    == original_storage_format
                ), (
                    "Result storage format "
                    + f"{modin_result._query_compiler.storage_format} does "
                    + f"not match expected storage format {original_storage_format}"
                )
            else:
                assert (
                    isinstance(no_check_for_execution_propagation_reason, str)
                    and len(no_check_for_execution_propagation_reason) > 0
                ), (
                    "Must provide a reason for not expecting the operation to "
                    + "propagate dataframe/series engine."
                )
        comparator(modin_result, pandas_result, **(comparator_kwargs or {}))


def eval_io(
    fn_name,
    comparator=df_equals,
    cast_to_str=False,
    expected_exception=None,
    check_kwargs_callable=True,
    modin_warning=None,
    modin_warning_str_match=None,
    md_extra_kwargs=None,
    *args,
    **kwargs,
):
    """Evaluate I/O operation outputs equality check.

    Parameters
    ----------
    fn_name: str
        I/O operation name ("read_csv" for example).
    comparator: obj
        Function to perform comparison.
    cast_to_str: bool
        There could be some mismatches in dtypes, so we're
        casting the whole frame to `str` before comparison.
        See issue #1931 for details.
    expected_exception: Exception
        Exception that should be raised even if it is raised
        both by Pandas and Modin.
    modin_warning: obj
        Warning that should be raised by Modin.
    modin_warning_str_match: str
        If `modin_warning` is set, checks that the raised warning matches this string.
    md_extra_kwargs: dict
        Modin operation specific kwargs.
    """

    def applyier(module, *args, **kwargs):
        result = getattr(module, fn_name)(*args, **kwargs)
        if cast_to_str:
            result = result.astype(str)
        if isinstance(result, (pd.DataFrame, pd.Series)):
            # Input methods that return a dataframe, e.g. read_csv, should
            # return a dataframe with engine and storage_format that match
            # the default Engine and StorageFormat, respectively.
            assert result._query_compiler.engine == Engine.get()
            assert result._query_compiler.storage_format == StorageFormat.get()
        return result

    def call_eval_general():
        eval_general(
            pd,
            pandas,
            applyier,
            comparator=comparator,
            expected_exception=expected_exception,
            check_kwargs_callable=check_kwargs_callable,
            md_extra_kwargs=md_extra_kwargs,
            *args,
            **kwargs,
        )

    warn_match = modin_warning_str_match if modin_warning is not None else None
    if modin_warning:
        with pytest.warns(modin_warning, match=warn_match):
            call_eval_general()
    else:
        call_eval_general()


def eval_io_from_str(csv_str: str, unique_filename: str, **kwargs):
    """Evaluate I/O operation outputs equality check by using `csv_str`
    data passed as python str (csv test file will be created from `csv_str`).

    Parameters
    ----------
    csv_str: str
        Test data for storing to csv file.
    unique_filename: str
        csv file name.
    """
    with open(unique_filename, "w") as f:
        f.write(csv_str)

    eval_io(
        filepath_or_buffer=unique_filename,
        fn_name="read_csv",
        **kwargs,
    )


def create_test_dfs(
    *args, post_fn=None, backend=None, **kwargs
) -> tuple[pd.DataFrame, pandas.DataFrame]:
    if post_fn is None:
        post_fn = lambda df: (  # noqa: E731
            df.convert_dtypes(dtype_backend=backend) if backend is not None else df
        )
    elif backend is not None:
        post_fn = lambda df: post_fn(df).convert_dtypes(  # noqa: E731
            dtype_backend=backend
        )
    return tuple(
        map(post_fn, [pd.DataFrame(*args, **kwargs), pandas.DataFrame(*args, **kwargs)])
    )


def create_test_series(
    vals, sort=False, backend=None, **kwargs
) -> tuple[pd.Series, pandas.Series]:
    if isinstance(vals, dict):
        modin_series = pd.Series(vals[next(iter(vals.keys()))], **kwargs)
        pandas_series = pandas.Series(vals[next(iter(vals.keys()))], **kwargs)
    else:
        modin_series = pd.Series(vals, **kwargs)
        pandas_series = pandas.Series(vals, **kwargs)
    if sort:
        modin_series = modin_series.sort_values().reset_index(drop=True)
        pandas_series = pandas_series.sort_values().reset_index(drop=True)

    if backend is not None:
        modin_series = modin_series.convert_dtypes(dtype_backend=backend)
        pandas_series = pandas_series.convert_dtypes(dtype_backend=backend)
    return modin_series, pandas_series


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


def get_unique_filename(
    test_name: str = "test",
    kwargs: dict = {},
    extension: str = "csv",
    data_dir: Union[str, Path] = "",
    suffix: str = "",
    debug_mode=False,
):
    """Returns unique file name with specified parameters.

    Parameters
    ----------
    test_name: str
        name of the test for which the unique file name is needed.
    kwargs: list of ints
        Unique combiantion of test parameters for creation of unique name.
    extension: str, default: "csv"
        Extension of unique file.
    data_dir: Union[str, Path]
        Data directory where test files will be created.
    suffix: str
        String to append to the resulted name.
    debug_mode: bool, default: False
        Get unique filename containing kwargs values.
        Otherwise kwargs values will be replaced with hash equivalent.

    Returns
    -------
        Unique file name.
    """
    suffix_part = f"_{suffix}" if suffix else ""
    extension_part = f".{extension}" if extension else ""
    if debug_mode:
        # shortcut if kwargs parameter are not provided
        if len(kwargs) == 0 and extension == "csv" and suffix == "":
            return os.path.join(data_dir, (test_name + suffix_part + f".{extension}"))

        assert "." not in extension, "please provide pure extension name without '.'"
        prohibited_chars = ['"', "\n"]
        non_prohibited_char = "np_char"
        char_counter = 0
        kwargs_name = dict(kwargs)
        for key, value in kwargs_name.items():
            for char in prohibited_chars:
                if isinstance(value, str) and char in value or callable(value):
                    kwargs_name[key] = non_prohibited_char + str(char_counter)
                    char_counter += 1
        parameters_values = "_".join(
            [
                (
                    str(value)
                    if not isinstance(value, (list, tuple))
                    else "_".join([str(x) for x in value])
                )
                for value in kwargs_name.values()
            ]
        )
        return os.path.join(
            data_dir, test_name + parameters_values + suffix_part + extension_part
        )
    else:
        import uuid

        return os.path.join(data_dir, uuid.uuid1().hex + suffix_part + extension_part)


def get_random_string():
    random_string = "".join(
        random_state.choice([x for x in ascii_letters], size=10).tolist()
    )
    return random_string


def insert_lines_to_csv(
    csv_name: str,
    lines_positions: list,
    lines_type: str = "blank",
    encoding: str = None,
    **csv_reader_writer_params,
):
    """Insert lines to ".csv" file.

    Parameters
    ----------
    csv_name: str
        ".csv" file that should be modified.
    lines_positions: list of ints
        Lines postions that sghould be modified (serial number
        of line - begins from 0, ends in <rows_number> - 1).
    lines_type: str
        Lines types that should be inserted to ".csv" file. Possible types:
        "blank" - empty line without any delimiters/separators,
        "bad" - lines with len(lines_data) > cols_number
    encoding: str
        Encoding type that should be used during file reading and writing.
    """
    if lines_type == "blank":
        lines_data = []
    elif lines_type == "bad":
        cols_number = len(pandas.read_csv(csv_name, nrows=1).columns)
        lines_data = [x for x in range(cols_number + 1)]
    else:
        raise ValueError(
            f"acceptable values for  parameter are ['blank', 'bad'], actually passed {lines_type}"
        )
    lines = []
    with open(csv_name, "r", encoding=encoding, newline="") as read_file:
        try:
            dialect = csv.Sniffer().sniff(read_file.read())
            read_file.seek(0)
        except Exception:
            dialect = None

        reader = csv.reader(
            read_file,
            dialect=dialect if dialect is not None else "excel",
            **csv_reader_writer_params,
        )
        counter = 0
        for row in reader:
            if counter in lines_positions:
                lines.append(lines_data)
            else:
                lines.append(row)
            counter += 1
    with open(csv_name, "w", encoding=encoding, newline="") as write_file:
        writer = csv.writer(
            write_file,
            dialect=dialect if dialect is not None else "excel",
            **csv_reader_writer_params,
        )
        writer.writerows(lines)


def _get_open_files():
    """
    psutil open_files() can return a lot of extra information that we can allow to
    be different, like file position; for simplicity we care about path and fd only.
    """
    return sorted((info.path, info.fd) for info in psutil.Process().open_files())


def check_file_leaks(func):
    """
    A decorator that ensures that no *newly* opened file handles are left
    after decorated function is finished.
    """
    if not TrackFileLeaks.get():
        return func

    @functools.wraps(func)
    def check(*a, **kw):
        fstart = _get_open_files()
        try:
            return func(*a, **kw)
        finally:
            leaks = []
            for item in _get_open_files():
                try:
                    fstart.remove(item)
                except ValueError:
                    # Ignore files in /proc/, as they have nothing to do with
                    # modin reading any data (and this is what we care about).
                    if item[0].startswith("/proc/"):
                        continue
                    # Ignore files in /tmp/ray/session_*/logs (ray session logs)
                    # because Ray intends to keep these logs open even after
                    # work has been done.
                    if re.search(r"/tmp/ray/session_.*/logs", item[0]):
                        continue
                    leaks.append(item)

            assert (
                not leaks
            ), f"Unexpected open handles left for: {', '.join(item[0] for item in leaks)}"

    return check


def dummy_decorator():
    """A problematic decorator that does not use `functools.wraps`. This introduces unwanted local variables for
    inspect.currentframe. This decorator is used in test_io to test `read_csv` and `read_table`
    """

    def wrapper(method):
        def wrapped_function(self, *args, **kwargs):
            result = method(self, *args, **kwargs)
            return result

        return wrapped_function

    return wrapper


def generate_dataframe(row_size=NROWS, additional_col_values=None, idx_name=None):
    dates = pandas.date_range("2000", freq="h", periods=row_size)
    data = {
        "col1": np.arange(row_size) * 10,
        "col2": [str(x.date()) for x in dates],
        "col3": np.arange(row_size) * 10,
        "col4": [str(x.time()) for x in dates],
        "col5": [get_random_string() for _ in range(row_size)],
        "col6": random_state.uniform(low=0.0, high=10000.0, size=row_size),
    }
    index = None if idx_name is None else pd.RangeIndex(0, row_size, name=idx_name)

    if additional_col_values is not None:
        assert isinstance(additional_col_values, (list, tuple))
        data.update({"col7": random_state.choice(additional_col_values, size=row_size)})
    return pandas.DataFrame(data, index=index)


def _make_csv_file(data_dir):
    def _csv_file_maker(
        filename=None,
        row_size=NROWS,
        force=True,
        delimiter=",",
        encoding=None,
        compression="infer",
        additional_col_values=None,
        remove_randomness=False,
        add_blank_lines=False,
        add_bad_lines=False,
        add_nan_lines=False,
        thousands_separator=None,
        decimal_separator=None,
        comment_col_char=None,
        quoting=csv.QUOTE_MINIMAL,
        quotechar='"',
        doublequote=True,
        escapechar=None,
        lineterminator=None,
    ):
        if filename is None:
            filename = get_unique_filename(data_dir=data_dir)
        if os.path.exists(filename) and not force:
            return None
        else:
            df = generate_dataframe(row_size, additional_col_values)
            if remove_randomness:
                df = df[["col1", "col2", "col3", "col4"]]
            if add_nan_lines:
                for i in range(0, row_size, row_size // (row_size // 10)):
                    df.loc[i] = pandas.Series()
            if comment_col_char:
                char = comment_col_char if isinstance(comment_col_char, str) else "#"
                df.insert(
                    loc=0,
                    column="col_with_comments",
                    value=[char if (x + 2) == 0 else x for x in range(row_size)],
                )

            if thousands_separator is not None:
                for col_id in ["col1", "col3"]:
                    df[col_id] = df[col_id].apply(
                        lambda x: f"{x:,d}".replace(",", thousands_separator)
                    )
                df["col6"] = df["col6"].apply(
                    lambda x: f"{x:,f}".replace(",", thousands_separator)
                )
            filename = (
                f"{filename}.{COMP_TO_EXT[compression]}"
                if compression != "infer"
                else filename
            )
            df.to_csv(
                filename,
                sep=delimiter,
                encoding=encoding,
                compression=compression,
                index=False,
                decimal=decimal_separator if decimal_separator else ".",
                lineterminator=lineterminator,
                quoting=quoting,
                quotechar=quotechar,
                doublequote=doublequote,
                escapechar=escapechar,
            )
            csv_reader_writer_params = {
                "delimiter": delimiter,
                "doublequote": doublequote,
                "escapechar": escapechar,
                "lineterminator": lineterminator if lineterminator else os.linesep,
                "quotechar": quotechar,
                "quoting": quoting,
            }
            if add_blank_lines:
                insert_lines_to_csv(
                    csv_name=filename,
                    lines_positions=[
                        x for x in range(5, row_size, row_size // (row_size // 10))
                    ],
                    lines_type="blank",
                    encoding=encoding,
                    **csv_reader_writer_params,
                )
            if add_bad_lines:
                insert_lines_to_csv(
                    csv_name=filename,
                    lines_positions=[
                        x for x in range(6, row_size, row_size // (row_size // 10))
                    ],
                    lines_type="bad",
                    encoding=encoding,
                    **csv_reader_writer_params,
                )
            return filename

    return _csv_file_maker


def sort_index_for_equal_values(df, ascending=True):
    """Sort `df` indices of equal rows."""
    if df.index.dtype == np.float64:
        # HACK: workaround for pandas bug:
        # https://github.com/pandas-dev/pandas/issues/34455
        df.index = df.index.astype("str")
    res = df.groupby(by=df if df.ndim == 1 else df.columns, sort=False).apply(
        lambda df: df.sort_index(ascending=ascending)
    )
    if res.index.nlevels > df.index.nlevels:
        # Sometimes GroupBy adds an extra level with 'by' to the result index.
        # GroupBy is very inconsistent about when it's doing this, so that's
        # why this clumsy if-statement is used.
        res.index = res.index.droplevel(0)
    # GroupBy overwrites original index names with 'by', so the following line restores original names
    res.index.names = df.index.names
    return res


def df_equals_with_non_stable_indices(df1, df2):
    """Assert equality of two frames regardless of the index order for equal values."""
    df1, df2 = map(try_cast_to_pandas, (df1, df2))
    np.testing.assert_array_equal(df1.values, df2.values)
    sorted1, sorted2 = map(sort_index_for_equal_values, (df1, df2))
    df_equals(sorted1, sorted2)


def rotate_decimal_digits_or_symbols(value):
    if value.dtype == object:
        # When dtype is object, we assume that it is actually strings from MultiIndex level names
        return [x[-1] + x[:-1] for x in value]
    else:
        tens = value // 10
        ones = value % 10
        return tens + ones * 10


def make_default_file(file_type: str, data_dir: str):
    """Helper function for pytest fixtures."""

    def _create_file(filename, force, nrows, ncols, func: str, func_kw=None):
        """
        Helper function that creates a dataframe before writing it to a file.

        Eliminates the duplicate code that is needed before of output functions calls.

        Notes
        -----
        Importantly, names of created files are added to `filenames` variable for
        their further automatic deletion. Without this step, files created by
        `pytest` fixtures will not be deleted.
        """
        if force or not os.path.exists(filename):
            df = pandas.DataFrame(
                {f"col{x + 1}": np.arange(nrows) for x in range(ncols)}
            )
            getattr(df, func)(filename, **func_kw if func_kw else {})

    file_type_to_extension = {
        "excel": "xlsx",
        "fwf": "txt",
        "pickle": "pkl",
    }
    extension = file_type_to_extension.get(file_type, file_type)

    def _make_default_file(nrows=NROWS, ncols=2, force=True, **kwargs):
        filename = get_unique_filename(extension=extension, data_dir=data_dir)

        if file_type == "json":
            lines = kwargs.get("lines")
            func_kw = {"lines": lines, "orient": "records"} if lines else {}
            _create_file(filename, force, nrows, ncols, "to_json", func_kw)
        elif file_type in ("html", "excel", "feather", "stata", "pickle"):
            _create_file(filename, force, nrows, ncols, f"to_{file_type}")
        elif file_type == "hdf":
            func_kw = {"key": "df", "format": kwargs.get("format")}
            _create_file(filename, force, nrows, ncols, "to_hdf", func_kw)
        elif file_type == "fwf":
            if force or not os.path.exists(filename):
                fwf_data = kwargs.get("fwf_data")
                if fwf_data is None:
                    with open("modin/tests/pandas/data/test_data.fwf", "r") as fwf_file:
                        fwf_data = fwf_file.read()
                with open(filename, "w") as f:
                    f.write(fwf_data)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        return filename

    return _make_default_file


def value_equals(obj1, obj2):
    """Check wherher two scalar or list-like values are equal and raise an ``AssertionError`` if they aren't."""
    if is_list_like(obj1):
        np.testing.assert_array_equal(obj1, obj2)
    else:
        assert (obj1 == obj2) or (np.isnan(obj1) and np.isnan(obj2))


def dict_equals(dict1, dict2):
    """Check whether two dictionaries are equal and raise an ``AssertionError`` if they aren't."""
    for key1, key2 in itertools.zip_longest(sorted(dict1), sorted(dict2)):
        value_equals(key1, key2)
        value_equals(dict1[key1], dict2[key2])
