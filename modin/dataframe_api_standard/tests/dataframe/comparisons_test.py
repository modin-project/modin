from __future__ import annotations

import pytest

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_dataframe_with_reference,
    integer_dataframe_1,
)


@pytest.mark.parametrize(
    ("comparison", "expected_data", "expected_dtype"),
    [
        ("__eq__", {"a": [False, True, False], "b": [False, False, False]}, "Bool"),
        ("__ne__", {"a": [True, False, True], "b": [True, True, True]}, "Bool"),
        ("__ge__", {"a": [False, True, True], "b": [True, True, True]}, "Bool"),
        ("__gt__", {"a": [False, False, True], "b": [True, True, True]}, "Bool"),
        ("__le__", {"a": [True, True, False], "b": [False, False, False]}, "Bool"),
        ("__lt__", {"a": [True, False, False], "b": [False, False, False]}, "Bool"),
        ("__add__", {"a": [3, 4, 5], "b": [6, 7, 8]}, "Int64"),
        ("__sub__", {"a": [-1, 0, 1], "b": [2, 3, 4]}, "Int64"),
        ("__mul__", {"a": [2, 4, 6], "b": [8, 10, 12]}, "Int64"),
        ("__truediv__", {"a": [0.5, 1, 1.5], "b": [2, 2.5, 3]}, "Float64"),
        ("__floordiv__", {"a": [0, 1, 1], "b": [2, 2, 3]}, "Int64"),
        ("__pow__", {"a": [1, 4, 9], "b": [16, 25, 36]}, "Int64"),
        ("__mod__", {"a": [1, 0, 1], "b": [0, 1, 0]}, "Int64"),
    ],
)
def test_comparisons_with_scalar(
    library: BaseHandler,
    comparison: str,
    expected_data: dict[str, object],
    expected_dtype: str,
) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    other = 2
    result = getattr(df, comparison)(other)
    expected_ns_dtype = getattr(ns, expected_dtype)
    compare_dataframe_with_reference(result, expected_data, dtype=expected_ns_dtype)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("comparison", "expected_data"),
    [
        ("__radd__", {"a": [3, 4, 5], "b": [6, 7, 8]}),
        ("__rsub__", {"a": [1, 0, -1], "b": [-2, -3, -4]}),
        ("__rmul__", {"a": [2, 4, 6], "b": [8, 10, 12]}),
    ],
)
def test_rcomparisons_with_scalar(
    library: BaseHandler,
    comparison: str,
    expected_data: dict[str, object],
) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    other = 2
    result = getattr(df, comparison)(other)
    compare_dataframe_with_reference(result, expected_data, dtype=ns.Int64)  # type: ignore[arg-type]
