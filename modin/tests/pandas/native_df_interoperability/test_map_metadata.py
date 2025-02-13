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
from modin.tests.pandas.native_df_interoperability.utils import (
    create_test_df_in_defined_mode,
    create_test_series_in_defined_mode,
)
from modin.tests.pandas.utils import (
    RAND_HIGH,
    RAND_LOW,
    axis_keys,
    axis_values,
    default_to_pandas_ignore_string,
    df_equals,
    eval_general,
    name_contains,
    numeric_dfs,
    random_state,
    test_data,
    test_data_keys,
    test_data_values,
)

NPartitions.put(4)

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")

# Our configuration in pytest.ini requires that we explicitly catch all
# instances of defaulting to pandas, but some test modules, like this one,
# have too many such instances.
pytestmark = pytest.mark.filterwarnings(default_to_pandas_ignore_string)


def eval_insert(modin_df, pandas_df, **kwargs):
    if "col" in kwargs and "column" not in kwargs:
        kwargs["column"] = kwargs.pop("col")
    _kwargs = {"loc": 0, "column": "New column"}
    _kwargs.update(kwargs)

    eval_general(
        modin_df,
        pandas_df,
        operation=lambda df, **kwargs: df.insert(**kwargs),
        __inplace__=True,
        **_kwargs,
    )


def test_empty_df(df_mode_pair):
    modin_df, pd_df = create_test_df_in_defined_mode(None, native=df_mode_pair[0])
    md_series, pd_series = create_test_series_in_defined_mode(
        [1, 2, 3, 4, 5], native=df_mode_pair[1]
    )
    modin_df["a"] = md_series
    pd_df["a"] = pd_series
    df_equals(modin_df, pd_df)


def test_astype(df_mode_pair):
    td = pandas.DataFrame(test_data["int_data"])[["col1", "index", "col3", "col4"]]
    modin_df, pandas_df = create_test_df_in_defined_mode(
        td.values,
        index=td.index,
        columns=td.columns,
        native=df_mode_pair[0],
    )

    def astype_func(df):
        md_ser, pd_ser = create_test_series_in_defined_mode(
            [str, str], index=["col1", "col1"], native=df_mode_pair[1]
        )
        if isinstance(df, pd.DataFrame):
            return df.astype(md_ser)
        else:
            return df.astype(pd_ser)

    # The dtypes series must have a unique index.
    eval_general(
        modin_df,
        pandas_df,
        astype_func,
        expected_exception=ValueError(
            "cannot reindex on an axis with duplicate labels"
        ),
    )


###########################################################################


def test_convert_dtypes_5653(df_mode_pair):
    modin_part1, _ = create_test_df_in_defined_mode(
        {"col1": ["a", "b", "c", "d"]}, native=df_mode_pair[0]
    )
    modin_part2, _ = create_test_df_in_defined_mode(
        {"col1": [None, None, None, None]}, native=df_mode_pair[1]
    )
    modin_df = pd.concat([modin_part1, modin_part2])
    if modin_df._query_compiler.storage_format == "Pandas":
        assert modin_df._query_compiler._modin_frame._partitions.shape == (2, 1)
    modin_df = modin_df.convert_dtypes()
    assert len(modin_df.dtypes) == 1
    assert modin_df.dtypes.iloc[0] == "string"


@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
@pytest.mark.parametrize("axis", axis_values, ids=axis_keys)
@pytest.mark.parametrize("bound_type", ["list", "series"], ids=["list", "series"])
@pytest.mark.exclude_in_sanity
def test_clip(request, data, axis, bound_type, df_mode_pair):
    modin_df, pandas_df = create_test_df_in_defined_mode(data, native=df_mode_pair[0])

    if name_contains(request.node.name, numeric_dfs):
        ind_len = (
            len(modin_df.index)
            if not pandas.DataFrame()._get_axis_number(axis)
            else len(modin_df.columns)
        )

        lower = random_state.randint(RAND_LOW, RAND_HIGH, ind_len)
        upper = random_state.randint(RAND_LOW, RAND_HIGH, ind_len)

        if bound_type == "series":
            modin_lower, pandas_lower = create_test_series_in_defined_mode(
                lower, native=df_mode_pair[1]
            )
            modin_upper, pandas_upper = create_test_series_in_defined_mode(
                upper, native=df_mode_pair[0]
            )
        else:
            modin_lower = pandas_lower = lower
            modin_upper = pandas_upper = upper

        # test lower and upper list bound on each column
        modin_result = modin_df.clip(modin_lower, modin_upper, axis=axis)
        pandas_result = pandas_df.clip(pandas_lower, pandas_upper, axis=axis)
        df_equals(modin_result, pandas_result)

        # test only upper list bound on each column
        modin_result = modin_df.clip(np.nan, modin_upper, axis=axis)
        pandas_result = pandas_df.clip(np.nan, pandas_upper, axis=axis)
        df_equals(modin_result, pandas_result)

        with pytest.raises(ValueError):
            modin_df.clip(lower=[1, 2, 3], axis=None)


@pytest.mark.parametrize(
    "data, other_data",
    [
        ({"A": [1, 2, 3], "B": [400, 500, 600]}, {"B": [4, 5, 6], "C": [7, 8, 9]}),
        ({"C": [1, 2, 3], "B": [400, 500, 600]}, {"B": [4, 5, 6], "A": [7, 8, 9]}),
        (
            {"A": ["a", "b", "c"], "B": ["x", "y", "z"]},
            {"B": ["d", "e", "f", "g", "h", "i"]},
        ),
        ({"A": [1, 2, 3], "B": [400, 500, 600]}, {"B": [4, np.nan, 6]}),
    ],
)
@pytest.mark.parametrize("errors", ["raise", "ignore"])
def test_update(data, other_data, errors, df_mode_pair):
    modin_df, pandas_df = create_test_df_in_defined_mode(data, native=df_mode_pair[0])
    other_modin_df, other_pandas_df = create_test_df_in_defined_mode(
        other_data, native=df_mode_pair[1]
    )
    expected_exception = None
    if errors == "raise":
        expected_exception = ValueError("Data overlaps.")
    eval_general(
        modin_df,
        pandas_df,
        lambda df: (
            df.update(other_modin_df, errors=errors)
            if isinstance(df, pd.DataFrame)
            else df.update(other_pandas_df, errors=errors)
        ),
        __inplace__=True,
        expected_exception=expected_exception,
    )


@pytest.mark.parametrize(
    "get_index",
    [
        pytest.param(lambda idx: None, id="None_idx"),
        pytest.param(lambda idx: ["a", "b", "c"], id="No_intersection_idx"),
        pytest.param(lambda idx: idx, id="Equal_idx"),
        pytest.param(lambda idx: idx[::-1], id="Reversed_idx"),
    ],
)
@pytest.mark.parametrize(
    "get_columns",
    [
        pytest.param(lambda idx: None, id="None_idx"),
        pytest.param(lambda idx: ["a", "b", "c"], id="No_intersection_idx"),
        pytest.param(lambda idx: idx, id="Equal_idx"),
        pytest.param(lambda idx: idx[::-1], id="Reversed_idx"),
    ],
)
@pytest.mark.parametrize("dtype", [None, "str"])
@pytest.mark.exclude_in_sanity
def test_constructor_from_modin_series(get_index, get_columns, dtype, df_mode_pair):
    modin_df, pandas_df = create_test_df_in_defined_mode(
        test_data_values[0], native=df_mode_pair[0]
    )

    modin_data = {f"new_col{i}": modin_df.iloc[:, i] for i in range(modin_df.shape[1])}
    pandas_data = {
        f"new_col{i}": pandas_df.iloc[:, i] for i in range(pandas_df.shape[1])
    }

    index = get_index(modin_df.index)
    columns = get_columns(list(modin_data.keys()))

    new_modin = pd.DataFrame(modin_data, index=index, columns=columns, dtype=dtype)
    new_pandas = pandas.DataFrame(
        pandas_data, index=index, columns=columns, dtype=dtype
    )
    df_equals(new_modin, new_pandas)
