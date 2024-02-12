from __future__ import annotations

from typing import Any

import pytest

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_dataframe_with_reference,
    integer_dataframe_1,
)


@pytest.mark.parametrize(
    ("reduction", "expected", "expected_dtype"),
    [
        ("min", {"a": [1], "b": [4]}, "Int64"),
        ("max", {"a": [3], "b": [6]}, "Int64"),
        ("sum", {"a": [6], "b": [15]}, "Int64"),
        ("prod", {"a": [6], "b": [120]}, "Int64"),
        ("median", {"a": [2.0], "b": [5.0]}, "Float64"),
        ("mean", {"a": [2.0], "b": [5.0]}, "Float64"),
        ("std", {"a": [1.0], "b": [1.0]}, "Float64"),
        ("var", {"a": [1.0], "b": [1.0]}, "Float64"),
    ],
)
def test_dataframe_reductions(
    library: BaseHandler,
    reduction: str,
    expected: dict[str, Any],
    expected_dtype: str,
) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    result = getattr(df, reduction)()
    expected_ns_dtype = getattr(ns, expected_dtype)
    compare_dataframe_with_reference(result, expected, dtype=expected_ns_dtype)
