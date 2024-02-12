from __future__ import annotations

import pytest

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_column_with_reference,
    integer_dataframe_1,
)


@pytest.mark.parametrize(
    ("reduction", "expected", "expected_dtype"),
    [
        ("min", 1, "Int64"),
        ("max", 3, "Int64"),
        ("sum", 6, "Int64"),
        ("prod", 6, "Int64"),
        ("median", 2.0, "Float64"),
        ("mean", 2.0, "Float64"),
        ("std", 1.0, "Float64"),
        ("var", 1.0, "Float64"),
    ],
)
def test_expression_reductions(
    library: BaseHandler,
    reduction: str,
    expected: float,
    expected_dtype: str,
) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    ser = ser - getattr(ser, reduction)()
    result = df.assign(ser.rename("result"))
    reference = list((df.col("a") - expected).persist().to_array())
    expected_ns_dtype = getattr(ns, expected_dtype)
    compare_column_with_reference(result.col("result"), reference, expected_ns_dtype)
