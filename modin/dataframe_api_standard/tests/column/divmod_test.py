from __future__ import annotations

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_column_with_reference,
    integer_dataframe_1,
)


def test_expression_divmod(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    other = df.col("b")
    result_quotient, result_remainder = ser.__divmod__(other)
    # quotient
    result = df.assign(result_quotient.rename("result"))
    compare_column_with_reference(result.col("result"), [0, 0, 0], dtype=ns.Int64)
    # remainder
    result = df.assign(result_remainder.rename("result"))
    compare_column_with_reference(result.col("result"), [1, 2, 3], dtype=ns.Int64)


def test_expression_divmod_with_scalar(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    result_quotient, result_remainder = ser.__divmod__(2)
    # quotient
    result = df.assign(result_quotient.rename("result"))
    compare_column_with_reference(result.col("result"), [0, 1, 1], dtype=ns.Int64)
    # remainder
    result = df.assign(result_remainder.rename("result"))
    compare_column_with_reference(result.col("result"), [1, 0, 1], dtype=ns.Int64)
