from __future__ import annotations

from typing import Any

import pytest

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_column_with_reference,
    integer_dataframe_1,
    integer_dataframe_7,
)


@pytest.mark.parametrize(
    ("comparison", "expected_data", "expected_dtype"),
    [
        ("__eq__", [True, True, False], "Bool"),
        ("__ne__", [False, False, True], "Bool"),
        ("__ge__", [True, True, False], "Bool"),
        ("__gt__", [False, False, False], "Bool"),
        ("__le__", [True, True, True], "Bool"),
        ("__lt__", [False, False, True], "Bool"),
        ("__add__", [2, 4, 7], "Int64"),
        ("__sub__", [0, 0, -1], "Int64"),
        ("__mul__", [1, 4, 12], "Int64"),
        ("__truediv__", [1, 1, 0.75], "Float64"),
        ("__floordiv__", [1, 1, 0], "Int64"),
        ("__pow__", [1, 4, 81], "Int64"),
        ("__mod__", [0, 0, 3], "Int64"),
    ],
)
def test_column_comparisons(
    library: BaseHandler,
    comparison: str,
    expected_data: list[object],
    expected_dtype: str,
) -> None:
    ser: Any
    df = integer_dataframe_7(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    other = df.col("b")
    result = df.assign(getattr(ser, comparison)(other).rename("result"))
    expected_ns_dtype = getattr(ns, expected_dtype)
    compare_column_with_reference(
        result.col("result"), expected_data, expected_ns_dtype
    )


@pytest.mark.parametrize(
    ("comparison", "expected_data", "expected_dtype"),
    [
        ("__eq__", [False, False, True], "Bool"),
        ("__ne__", [True, True, False], "Bool"),
        ("__ge__", [False, False, True], "Bool"),
        ("__gt__", [False, False, False], "Bool"),
        ("__le__", [True, True, True], "Bool"),
        ("__lt__", [True, True, False], "Bool"),
        ("__add__", [4, 5, 6], "Int64"),
        ("__sub__", [-2, -1, 0], "Int64"),
        ("__mul__", [3, 6, 9], "Int64"),
        ("__truediv__", [1 / 3, 2 / 3, 1], "Float64"),
        ("__floordiv__", [0, 0, 1], "Int64"),
        ("__pow__", [1, 8, 27], "Int64"),
        ("__mod__", [1, 2, 0], "Int64"),
    ],
)
def test_column_comparisons_scalar(
    library: BaseHandler,
    comparison: str,
    expected_data: list[object],
    expected_dtype: str,
) -> None:
    ser: Any
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    other = 3
    result = df.assign(getattr(ser, comparison)(other).rename("result"))
    expected_ns_dtype = getattr(ns, expected_dtype)
    compare_column_with_reference(
        result.col("result"), expected_data, expected_ns_dtype
    )


@pytest.mark.parametrize(
    ("comparison", "expected_data"),
    [
        ("__radd__", [3, 4, 5]),
        ("__rsub__", [1, 0, -1]),
        ("__rmul__", [2, 4, 6]),
    ],
)
def test_right_column_comparisons(
    library: BaseHandler,
    comparison: str,
    expected_data: list[object],
) -> None:
    # 1,2,3
    ser: Any
    df = integer_dataframe_7(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    other = 2
    result = df.assign(getattr(ser, comparison)(other).rename("result"))
    compare_column_with_reference(result.col("result"), expected_data, dtype=ns.Int64)
