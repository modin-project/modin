from __future__ import annotations

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_dataframe_with_reference,
    integer_dataframe_1,
)


def test_float_powers_column(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    other = df.col("b") * 1.0
    result = df.assign(ser.__pow__(other).rename("result"))
    expected = {"a": [1, 2, 3], "b": [4, 5, 6], "result": [1.0, 32.0, 729.0]}
    expected_dtype = {"a": ns.Int64, "b": ns.Int64, "result": ns.Float64}
    compare_dataframe_with_reference(result, expected, expected_dtype)  # type: ignore[arg-type]


def test_float_powers_scalar_column(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    other = 1.0
    result = df.assign(ser.__pow__(other).rename("result"))
    expected = {"a": [1, 2, 3], "b": [4, 5, 6], "result": [1.0, 2.0, 3.0]}
    expected_dtype = {"a": ns.Int64, "b": ns.Int64, "result": ns.Float64}
    compare_dataframe_with_reference(result, expected, expected_dtype)  # type: ignore[arg-type]


def test_int_powers_column(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    other = df.col("b") * 1
    result = df.assign(ser.__pow__(other).rename("result"))
    expected = {"a": [1, 2, 3], "b": [4, 5, 6], "result": [1, 32, 729]}
    expected_dtype = {name: ns.Int64 for name in ("a", "b", "result")}
    compare_dataframe_with_reference(result, expected, expected_dtype)


def test_int_powers_scalar_column(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    other = 1
    result = df.assign(ser.__pow__(other).rename("result"))
    expected = {"a": [1, 2, 3], "b": [4, 5, 6], "result": [1, 2, 3]}
    expected_dtype = {name: ns.Int64 for name in ("a", "b", "result")}
    compare_dataframe_with_reference(result, expected, expected_dtype)
