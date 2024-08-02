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
from modin.config import NativeDataframeMode, NPartitions, StorageFormat
from modin.core.dataframe.pandas.partitioning.axis_partition import (
    PandasDataframeAxisPartition,
)
from modin.tests.pandas.utils import (
    CustomIntegerForAddition,
    NonCommutativeMultiplyInteger,
    create_test_dfs,
    default_to_pandas_ignore_string,
    df_equals,
    eval_general,
    test_data,
    test_data_keys,
    test_data_values,
)
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution

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
def test_math_functions(other, axis, op, backend):
    data = test_data["float_nan_data"]
    if (op == "floordiv" or op == "rfloordiv") and axis == "rows":
        # lambda == "series_or_list"
        pytest.xfail(reason="different behavior")

    if op == "rmod" and axis == "rows":
        # lambda == "series_or_list"
        pytest.xfail(reason="different behavior")

    if op in ("mod", "rmod") and backend == "pyarrow":
        pytest.skip(reason="These functions are not implemented in pandas itself")
    eval_general(
        *create_test_dfs(data, backend=backend),
        lambda df: getattr(df, op)(other(df, axis), axis=axis),
    )


@pytest.mark.parametrize("other", [lambda df: 2, lambda df: df])
def test___divmod__(other):
    data = test_data["float_nan_data"]
    eval_general(*create_test_dfs(data), lambda df: divmod(df, other(df)))


def test___rdivmod__():
    data = test_data["float_nan_data"]
    eval_general(*create_test_dfs(data), lambda df: divmod(2, df))


@pytest.mark.parametrize(
    "other",
    [lambda df: df[: -(2**4)], lambda df: df[df.columns[0]].reset_index(drop=True)],
    ids=["check_missing_value", "check_different_index"],
)
@pytest.mark.parametrize("fill_value", [None, 3.0])
@pytest.mark.parametrize(
    "op",
    [
        *("add", "radd", "sub", "rsub", "mod", "rmod", "pow", "rpow"),
        *("truediv", "rtruediv", "mul", "rmul", "floordiv", "rfloordiv"),
    ],
)
def test_math_functions_fill_value(other, fill_value, op, request):
    data = test_data["int_data"]
    modin_df, pandas_df = pd.DataFrame(data), pandas.DataFrame(data)

    expected_exception = None
    if "check_different_index" in request.node.callspec.id and fill_value == 3.0:
        expected_exception = NotImplementedError("fill_value 3.0 not supported.")

    eval_general(
        modin_df,
        pandas_df,
        lambda df: getattr(df, op)(other(df), axis=0, fill_value=fill_value),
        expected_exception=expected_exception,
        # This test causes an empty slice to be generated thus triggering:
        # https://github.com/modin-project/modin/issues/5974
        comparator_kwargs={"check_dtypes": get_current_execution() != "BaseOnPython"},
    )


@pytest.mark.parametrize(
    "op",
    [
        *("add", "radd", "sub", "rsub", "mod", "rmod", "pow", "rpow"),
        *("truediv", "rtruediv", "mul", "rmul", "floordiv", "rfloordiv"),
    ],
)
def test_math_functions_level(op):
    modin_df = pd.DataFrame(test_data["int_data"])
    modin_df.index = pandas.MultiIndex.from_tuples(
        [(i // 4, i // 2, i) for i in modin_df.index]
    )

    # Defaults to pandas
    with warns_that_defaulting_to_pandas():
        # Operation against self for sanity check
        getattr(modin_df, op)(modin_df, axis=0, level=1)


@pytest.mark.parametrize(
    "math_op, alias",
    [
        ("truediv", "divide"),
        ("truediv", "div"),
        ("rtruediv", "rdiv"),
        ("mul", "multiply"),
        ("sub", "subtract"),
        ("add", "__add__"),
        ("radd", "__radd__"),
        ("truediv", "__truediv__"),
        ("rtruediv", "__rtruediv__"),
        ("floordiv", "__floordiv__"),
        ("rfloordiv", "__rfloordiv__"),
        ("mod", "__mod__"),
        ("rmod", "__rmod__"),
        ("mul", "__mul__"),
        ("rmul", "__rmul__"),
        ("pow", "__pow__"),
        ("rpow", "__rpow__"),
        ("sub", "__sub__"),
        ("rsub", "__rsub__"),
    ],
)
def test_math_alias(math_op, alias):
    assert getattr(pd.DataFrame, math_op) == getattr(pd.DataFrame, alias)


@pytest.mark.parametrize("other", ["as_left", 4, 4.0, "a"])
@pytest.mark.parametrize("op", ["eq", "ge", "gt", "le", "lt", "ne"])
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_comparison(data, op, other, request):
    def operation(df):
        return getattr(df, op)(df if other == "as_left" else other)

    expected_exception = None
    if "int_data" in request.node.callspec.id and other == "a":
        pytest.xfail(reason="https://github.com/modin-project/modin/issues/7019")
    elif "float_nan_data" in request.node.callspec.id and other == "a":
        expected_exception = TypeError(
            "Invalid comparison between dtype=float64 and str"
        )

    eval_general(
        *create_test_dfs(data),
        operation=operation,
        expected_exception=expected_exception,
    )


@pytest.mark.skipif(
    StorageFormat.get() != "Pandas",
    reason="Modin on this engine doesn't create virtual partitions.",
)
@pytest.mark.skipif(
    NativeDataframeMode.get() == "Pandas",
    reason="NativeQueryCompiler does not contain partitions.",
)
@pytest.mark.parametrize(
    "left_virtual,right_virtual", [(True, False), (False, True), (True, True)]
)
def test_virtual_partitions(left_virtual: bool, right_virtual: bool):
    # This test covers https://github.com/modin-project/modin/issues/4691
    n: int = 1000
    pd_df = pandas.DataFrame(list(range(n)))

    def modin_df(is_virtual):
        if not is_virtual:
            return pd.DataFrame(pd_df)
        result = pd.concat([pd.DataFrame([i]) for i in range(n)], ignore_index=True)
        # Modin should rebalance the partitions after the concat, producing virtual partitions.
        assert isinstance(
            result._query_compiler._modin_frame._partitions[0][0],
            PandasDataframeAxisPartition,
        )
        return result

    df_equals(modin_df(left_virtual) + modin_df(right_virtual), pd_df + pd_df)


@pytest.mark.parametrize("op", ["eq", "ge", "gt", "le", "lt", "ne"])
@pytest.mark.parametrize("data", test_data_values, ids=test_data_keys)
def test_multi_level_comparison(data, op):
    modin_df_multi_level = pd.DataFrame(data)

    new_idx = pandas.MultiIndex.from_tuples(
        [(i // 4, i // 2, i) for i in modin_df_multi_level.index]
    )
    modin_df_multi_level.index = new_idx

    # Defaults to pandas
    with warns_that_defaulting_to_pandas():
        # Operation against self for sanity check
        getattr(modin_df_multi_level, op)(modin_df_multi_level, axis=0, level=1)


@pytest.mark.parametrize(
    "frame1_data,frame2_data,expected_pandas_equals",
    [
        pytest.param({}, {}, True, id="two_empty_dataframes"),
        pytest.param([[1]], [[0]], False, id="single_unequal_values"),
        pytest.param([[None]], [[None]], True, id="single_none_values"),
        pytest.param([[np.nan]], [[np.nan]], True, id="single_nan_values"),
        pytest.param({1: [10]}, {1.0: [10]}, True, id="different_column_types"),
        pytest.param({1: [10]}, {2: [10]}, False, id="different_columns"),
        pytest.param(
            pandas.DataFrame({1: [10]}, index=[1]),
            pandas.DataFrame({1: [10]}, index=[1.0]),
            True,
            id="different_index_types",
        ),
        pytest.param(
            pandas.DataFrame({1: [10]}, index=[1]),
            pandas.DataFrame({1: [10]}, index=[2]),
            False,
            id="different_indexes",
        ),
        pytest.param({1: [10]}, {1: [10.0]}, False, id="different_value_types"),
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
        pytest.param(
            [[1, 1]],
            [[1]],
            False,
            id="different_row_lengths",
        ),
        pytest.param(
            [[1], [1]],
            [[1]],
            False,
            id="different_column_lengths",
        ),
    ],
)
def test_equals(frame1_data, frame2_data, expected_pandas_equals):
    modin_df1 = pd.DataFrame(frame1_data)
    pandas_df1 = pandas.DataFrame(frame1_data)
    modin_df2 = pd.DataFrame(frame2_data)
    pandas_df2 = pandas.DataFrame(frame2_data)

    pandas_equals = pandas_df1.equals(pandas_df2)
    assert pandas_equals == expected_pandas_equals, (
        "Test expected pandas to say the dataframes were"
        + f"{'' if expected_pandas_equals else ' not'} equal, but they were"
        + f"{' not' if expected_pandas_equals else ''} equal."
    )

    assert modin_df1.equals(modin_df2) == pandas_equals
    assert modin_df1.equals(pandas_df2) == pandas_equals


def test_equals_several_partitions():
    modin_series1 = pd.concat([pd.DataFrame([0, 1]), pd.DataFrame([None, 1])])
    modin_series2 = pd.concat([pd.DataFrame([0, 1]), pd.DataFrame([1, None])])
    assert not modin_series1.equals(modin_series2)


def test_equals_with_nans():
    df1 = pd.DataFrame([0, 1, None], dtype="uint8[pyarrow]")
    df2 = pd.DataFrame([None, None, None], dtype="uint8[pyarrow]")
    assert not df1.equals(df2)


@pytest.mark.parametrize("is_more_other_partitions", [True, False])
@pytest.mark.parametrize(
    "op_type", ["df_ser", "df_df", "ser_ser_same_name", "ser_ser_different_name"]
)
@pytest.mark.parametrize(
    "is_idx_aligned", [True, False], ids=["idx_aligned", "idx_not_aligned"]
)
def test_mismatched_row_partitions(is_idx_aligned, op_type, is_more_other_partitions):
    data = [0, 1, 2, 3, 4, 5]
    modin_df1, pandas_df1 = create_test_dfs({"a": data, "b": data})
    modin_df, pandas_df = modin_df1.loc[:2], pandas_df1.loc[:2]

    modin_df2 = pd.concat((modin_df, modin_df))
    pandas_df2 = pandas.concat((pandas_df, pandas_df))
    if is_more_other_partitions:
        modin_df2, modin_df1 = modin_df1, modin_df2
        pandas_df2, pandas_df1 = pandas_df1, pandas_df2

    if is_idx_aligned:
        if is_more_other_partitions:
            modin_df1.index = pandas_df1.index = pandas_df2.index
        else:
            modin_df2.index = pandas_df2.index = pandas_df1.index

    # Pandas don't support this case because result will contain duplicate values by col axis.
    if op_type == "df_ser" and not is_idx_aligned and is_more_other_partitions:
        eval_general(
            modin_df2,
            pandas_df2,
            lambda df: (
                df / modin_df1.a if isinstance(df, pd.DataFrame) else df / pandas_df1.a
            ),
            expected_exception=ValueError(
                "cannot reindex on an axis with duplicate labels"
            ),
        )
        return

    if op_type == "df_ser":
        modin_res = modin_df2 / modin_df1.a
        pandas_res = pandas_df2 / pandas_df1.a
    elif op_type == "df_df":
        modin_res = modin_df2 / modin_df1
        pandas_res = pandas_df2 / pandas_df1
    elif op_type == "ser_ser_same_name":
        modin_res = modin_df2.a / modin_df1.a
        pandas_res = pandas_df2.a / pandas_df1.a
    elif op_type == "ser_ser_different_name":
        modin_res = modin_df2.a / modin_df1.b
        pandas_res = pandas_df2.a / pandas_df1.b
    else:
        raise Exception(f"op_type: {op_type} not supported in test")
    df_equals(modin_res, pandas_res)


def test_duplicate_indexes():
    data = [0, 1, 2, 3, 4, 5]
    modin_df1, pandas_df1 = create_test_dfs(
        {"a": data, "b": data}, index=[0, 1, 2, 0, 1, 2]
    )
    modin_df2, pandas_df2 = create_test_dfs({"a": data, "b": data})
    df_equals(modin_df1 / modin_df2, pandas_df1 / pandas_df2)
    df_equals(modin_df1 / modin_df1, pandas_df1 / pandas_df1)


@pytest.mark.parametrize("subset_operand", ["left", "right"])
def test_mismatched_col_partitions(subset_operand):
    data = [0, 1, 2, 3]
    modin_df1, pandas_df1 = create_test_dfs({"a": data, "b": data})
    modin_df_tmp, pandas_df_tmp = create_test_dfs({"c": data})

    modin_df2 = pd.concat([modin_df1, modin_df_tmp], axis=1)
    pandas_df2 = pandas.concat([pandas_df1, pandas_df_tmp], axis=1)

    if subset_operand == "right":
        modin_res = modin_df2 + modin_df1
        pandas_res = pandas_df2 + pandas_df1
    else:
        modin_res = modin_df1 + modin_df2
        pandas_res = pandas_df1 + pandas_df2

    df_equals(modin_res, pandas_res)


@pytest.mark.parametrize("empty_operand", ["right", "left", "both"])
def test_empty_df(empty_operand):
    modin_df, pandas_df = create_test_dfs([0, 1, 2, 0, 1, 2])
    modin_df_empty, pandas_df_empty = create_test_dfs()

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


def test_add_string_to_df():
    modin_df, pandas_df = create_test_dfs(["a", "b"])
    eval_general(modin_df, pandas_df, lambda df: "string" + df)
    eval_general(modin_df, pandas_df, lambda df: df + "string")


def test_add_custom_class():
    # see https://github.com/modin-project/modin/issues/5236
    # Test that we can add any object that is addable to pandas object data
    # via "+".
    eval_general(
        *create_test_dfs(test_data["int_data"]),
        lambda df: df + CustomIntegerForAddition(4),
    )


def test_non_commutative_multiply_pandas():
    # The non commutative integer class implementation is tricky. Check that
    # multiplying such an integer with a pandas dataframe is really not
    # commutative.
    pandas_df = pandas.DataFrame([[1]], dtype=int)
    integer = NonCommutativeMultiplyInteger(2)
    assert not (integer * pandas_df).equals(pandas_df * integer)


def test_non_commutative_multiply():
    # This test checks that mul and rmul do different things when
    # multiplication is not commutative, e.g. for adding a string to a string.
    # For context see https://github.com/modin-project/modin/issues/5238
    modin_df, pandas_df = create_test_dfs([1], dtype=int)
    integer = NonCommutativeMultiplyInteger(2)
    eval_general(modin_df, pandas_df, lambda s: integer * s)
    eval_general(modin_df, pandas_df, lambda s: s * integer)


@pytest.mark.parametrize(
    "op",
    [
        *("add", "radd", "sub", "rsub", "mod", "rmod", "pow", "rpow"),
        *("truediv", "rtruediv", "mul", "rmul", "floordiv", "rfloordiv"),
    ],
)
@pytest.mark.parametrize(
    "val1",
    [
        pytest.param([10, 20], id="int"),
        pytest.param([10, True], id="obj"),
        pytest.param([True, True], id="bool"),
        pytest.param([3.5, 4.5], id="float"),
    ],
)
@pytest.mark.parametrize(
    "val2",
    [
        pytest.param([10, 20], id="int"),
        pytest.param([10, True], id="obj"),
        pytest.param([True, True], id="bool"),
        pytest.param([3.5, 4.5], id="float"),
        pytest.param(2, id="int scalar"),
        pytest.param(True, id="bool scalar"),
        pytest.param(3.5, id="float scalar"),
    ],
)
def test_arithmetic_with_tricky_dtypes(val1, val2, op, request):
    modin_df1, pandas_df1 = create_test_dfs(val1)
    modin_df2, pandas_df2 = (
        create_test_dfs(val2) if isinstance(val2, list) else (val2, val2)
    )

    expected_exception = None
    if (
        "bool-bool" in request.node.callspec.id
        or "bool scalar-bool" in request.node.callspec.id
    ) and op in [
        "pow",
        "rpow",
        "truediv",
        "rtruediv",
        "floordiv",
        "rfloordiv",
    ]:
        op_name = op[1:] if op.startswith("r") else op
        expected_exception = NotImplementedError(
            f"operator '{op_name}' not implemented for bool dtypes"
        )
    elif (
        "bool-bool" in request.node.callspec.id
        or "bool scalar-bool" in request.node.callspec.id
    ) and op in ["sub", "rsub"]:
        expected_exception = TypeError(
            "numpy boolean subtract, the `-` operator, is not supported, "
            + "use the bitwise_xor, the `^` operator, or the logical_xor function instead."
        )

    eval_general(
        (modin_df1, modin_df2),
        (pandas_df1, pandas_df2),
        lambda dfs: getattr(dfs[0], op)(dfs[1]),
        expected_exception=expected_exception,
    )


@pytest.mark.parametrize(
    "data, other_data",
    [
        ({"A": [1, 2, 3], "B": [400, 500, 600]}, {"B": [4, 5, 6], "C": [7, 8, 9]}),
        ({"C": [1, 2, 3], "B": [400, 500, 600]}, {"B": [4, 5, 6], "A": [7, 8, 9]}),
    ],
)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("match_index", [True, False])
def test_bin_op_mismatched_columns(data, other_data, axis, match_index):
    modin_df, pandas_df = create_test_dfs(data)
    other_modin_df, other_pandas_df = create_test_dfs(other_data)
    if axis == 0:
        if not match_index:
            modin_df.index = pandas_df.index = ["1", "2", "3"]
            other_modin_df.index = other_pandas_df.index = ["2", "1", "3"]
    eval_general(
        modin_df,
        pandas_df,
        lambda df: (
            df.add(other_modin_df, axis=axis)
            if isinstance(df, pd.DataFrame)
            else df.add(other_pandas_df, axis=axis)
        ),
    )
