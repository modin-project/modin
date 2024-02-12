from __future__ import annotations

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    bool_dataframe_1,
    compare_column_with_reference,
)


def test_expression_invert(library: BaseHandler) -> None:
    df = bool_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    result = df.assign((~ser).rename("result"))
    expected = [False, False, True]
    compare_column_with_reference(result.col("result"), expected, dtype=ns.Bool)


def test_column_invert(library: BaseHandler) -> None:
    df = bool_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    result = df.assign((~ser).rename("result"))
    expected = [False, False, True]
    compare_column_with_reference(result.col("result"), expected, dtype=ns.Bool)
