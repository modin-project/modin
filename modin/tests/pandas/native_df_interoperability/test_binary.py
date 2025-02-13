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
import pytest

from modin.config import NPartitions
from modin.tests.pandas.native_df_interoperability.utils import (
    create_test_df_in_defined_mode,
    eval_general_interop,
)
from modin.tests.pandas.utils import (
    default_to_pandas_ignore_string,
    df_equals,
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


@pytest.mark.parametrize(
    "other",
    [
        lambda df, axis: 4,
        lambda df, axis: df.iloc[0] if axis == "columns" else list(df[df.columns[0]]),
        lambda df, axis: {
            label: idx + 1
            for idx, label in enumerate(df.axes[0 if axis == "rows" else 1])
        },
        lambda df, axis: {
            label if idx % 2 else f"random_key{idx}": idx + 1
            for idx, label in enumerate(df.axes[0 if axis == "rows" else 1][::-1])
        },
    ],
    ids=[
        "scalar",
        "series_or_list",
        "dictionary_keys_equal_columns",
        "dictionary_keys_unequal_columns",
    ],
)
@pytest.mark.parametrize("axis", ["rows", "columns"])
@pytest.mark.parametrize(
    "op",
    [
        *("add", "radd", "sub", "rsub", "mod", "rmod", "pow", "rpow"),
        *("truediv", "rtruediv", "mul", "rmul", "floordiv", "rfloordiv"),
    ],
)
@pytest.mark.parametrize("backend", [None, "pyarrow"])
def test_math_functions(other, axis, op, backend, df_mode_pair):
    data = test_data["float_nan_data"]
    if (op == "floordiv" or op == "rfloordiv") and axis == "rows":
        # lambda == "series_or_list"
        pytest.xfail(reason="different behavior")

    if op == "rmod" and axis == "rows":
        # lambda == "series_or_list"
        pytest.xfail(reason="different behavior")

    if op in ("mod", "rmod") and backend == "pyarrow":
        pytest.skip(reason="These functions are not implemented in pandas itself")

    eval_general_interop(
        data,
        backend,
        lambda df1, df2: getattr(df1, op)(other(df2, axis), axis=axis),
        df_mode_pair,
    )


@pytest.mark.parametrize("other", [lambda df: 2, lambda df: df])
def test___divmod__(other, df_mode_pair):
    data = test_data["float_nan_data"]
    eval_general_interop(
        data, None, lambda df1, df2: divmod(df1, other(df2)), df_mode_pair
    )


@pytest.mark.parametrize("other", ["as_left", 4])
@pytest.mark.parametrize("op", ["eq", "ge", "gt", "le", "lt", "ne"])
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_comparison(data, op, other, request, df_mode_pair):
    def operation(df1, df2):
        return getattr(df1, op)(df2 if other == "as_left" else other)

    expected_exception = None
    if "int_data" in request.node.callspec.id and other == "a":
        pytest.xfail(reason="https://github.com/modin-project/modin/issues/7019")
    elif "float_nan_data" in request.node.callspec.id and other == "a":
        expected_exception = TypeError(
            "Invalid comparison between dtype=float64 and str"
        )
    eval_general_interop(
        data,
        None,
        operation,
        df_mode_pair,
        expected_exception=expected_exception,
    )


@pytest.mark.parametrize(
    "frame1_data,frame2_data,expected_pandas_equals",
    [
        pytest.param({}, {}, True, id="two_empty_dataframes"),
        pytest.param([[1]], [[0]], False, id="single_unequal_values"),
        pytest.param([[None]], [[None]], True, id="single_none_values"),
        pytest.param(
            [[1, 2], [3, 4]],
            [[1, 2], [3, 4]],
            True,
            id="equal_two_by_two_dataframes",
        ),
        pytest.param(
            [[1, 2], [3, 4]],
            [[5, 2], [3, 4]],
            False,
            id="unequal_two_by_two_dataframes",
        ),
    ],
)
def test_equals(frame1_data, frame2_data, expected_pandas_equals, df_mode_pair):
    modin_df1, pandas_df1 = create_test_df_in_defined_mode(
        frame1_data, native=df_mode_pair[0]
    )
    modin_df2, pandas_df2 = create_test_df_in_defined_mode(
        frame2_data, native=df_mode_pair[1]
    )

    pandas_equals = pandas_df1.equals(pandas_df2)
    assert pandas_equals == expected_pandas_equals, (
        "Test expected pandas to say the dataframes were"
        + f"{'' if expected_pandas_equals else ' not'} equal, but they were"
        + f"{' not' if expected_pandas_equals else ''} equal."
    )

    assert modin_df1.equals(modin_df2) == pandas_equals
    assert modin_df1.equals(pandas_df2) == pandas_equals


@pytest.mark.parametrize("empty_operand", ["right", "left", "both"])
def test_empty_df(empty_operand, df_mode_pair):
    modin_df, pandas_df = create_test_df_in_defined_mode(
        [0, 1, 2, 0, 1, 2], native=df_mode_pair[0]
    )
    modin_df_empty, pandas_df_empty = create_test_df_in_defined_mode(
        native=df_mode_pair[1]
    )

    if empty_operand == "right":
        modin_res = modin_df + modin_df_empty
        pandas_res = pandas_df + pandas_df_empty
    elif empty_operand == "left":
        modin_res = modin_df_empty + modin_df
        pandas_res = pandas_df_empty + pandas_df
    else:
        modin_res = modin_df_empty + modin_df_empty
        pandas_res = pandas_df_empty + pandas_df_empty

    df_equals(modin_res, pandas_res)
