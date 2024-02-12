from __future__ import annotations

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    bool_dataframe_1,
    compare_column_with_reference,
)


def test_column_and(library: BaseHandler) -> None:
    df = bool_dataframe_1(library, api_version="2023.09-beta")
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    other = df.col("b")
    result = df.assign((ser & other).rename("result"))
    expected = [True, True, False]
    compare_column_with_reference(result.col("result"), expected, dtype=ns.Bool)


def test_column_or(library: BaseHandler) -> None:
    df = bool_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    other = df.col("b")
    result = df.assign((ser | other).rename("result"))
    expected = [True, True, True]
    compare_column_with_reference(result.col("result"), expected, dtype=ns.Bool)


def test_column_and_with_scalar(library: BaseHandler) -> None:
    df = bool_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    other = True
    result = df.assign((other & ser).rename("result"))
    expected = [True, True, False]
    compare_column_with_reference(result.col("result"), expected, dtype=ns.Bool)


def test_column_or_with_scalar(library: BaseHandler) -> None:
    df = bool_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    other = True
    result = df.assign((other | ser).rename("result"))
    expected = [True, True, True]
    compare_column_with_reference(result.col("result"), expected, dtype=ns.Bool)
