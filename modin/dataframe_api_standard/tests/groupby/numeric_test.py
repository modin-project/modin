from __future__ import annotations

import pytest

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_dataframe_with_reference,
    integer_dataframe_4,
)


@pytest.mark.parametrize(
    ("aggregation", "expected_b", "expected_c", "expected_dtype"),
    [
        ("min", [1, 3], [4, 6], "Int64"),
        ("max", [2, 4], [5, 7], "Int64"),
        ("sum", [3, 7], [9, 13], "Int64"),
        ("prod", [2, 12], [20, 42], "Int64"),
        ("median", [1.5, 3.5], [4.5, 6.5], "Float64"),
        ("mean", [1.5, 3.5], [4.5, 6.5], "Float64"),
        (
            "std",
            [0.7071067811865476, 0.7071067811865476],
            [0.7071067811865476, 0.7071067811865476],
            "Float64",
        ),
        ("var", [0.5, 0.5], [0.5, 0.5], "Float64"),
    ],
)
def test_group_by_numeric(
    library: BaseHandler,
    aggregation: str,
    expected_b: list[float],
    expected_c: list[float],
    expected_dtype: str,
) -> None:
    df = integer_dataframe_4(library)
    ns = df.__dataframe_namespace__()
    result = getattr(df.group_by("key"), aggregation)()
    result = result.sort("key")
    expected = {"key": [1, 2], "b": expected_b, "c": expected_c}
    dtype = getattr(ns, expected_dtype)
    expected_ns_dtype = {"key": ns.Int64, "b": dtype, "c": dtype}
    compare_dataframe_with_reference(result, expected, dtype=expected_ns_dtype)  # type: ignore[arg-type]
