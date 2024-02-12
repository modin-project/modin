from __future__ import annotations

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_dataframe_with_reference,
    integer_dataframe_6,
)


def test_expression_sort_ascending(library: BaseHandler) -> None:
    df = integer_dataframe_6(library, api_version="2023.09-beta")
    ns = df.__dataframe_namespace__()
    s_sorted = df.col("b").sort().rename("c")
    result = df.assign(s_sorted)
    expected = {
        "a": [1, 1, 1, 2, 2],
        "b": [4, 4, 3, 1, 2],
        "c": [1, 2, 3, 4, 4],
    }
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)


def test_expression_sort_descending(library: BaseHandler) -> None:
    df = integer_dataframe_6(library, api_version="2023.09-beta")
    ns = df.__dataframe_namespace__()
    s_sorted = df.col("b").sort(ascending=False).rename("c")
    result = df.assign(s_sorted)
    expected = {
        "a": [1, 1, 1, 2, 2],
        "b": [4, 4, 3, 1, 2],
        "c": [4, 4, 3, 2, 1],
    }
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)


def test_column_sort_ascending(library: BaseHandler) -> None:
    df = integer_dataframe_6(library, api_version="2023.09-beta")
    ns = df.__dataframe_namespace__()
    s_sorted = df.col("b").sort().rename("c")
    result = df.assign(s_sorted)
    expected = {
        "a": [1, 1, 1, 2, 2],
        "b": [4, 4, 3, 1, 2],
        "c": [1, 2, 3, 4, 4],
    }
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)


def test_column_sort_descending(library: BaseHandler) -> None:
    df = integer_dataframe_6(library, api_version="2023.09-beta")
    ns = df.__dataframe_namespace__()
    s_sorted = df.col("b").sort(ascending=False).rename("c")
    result = df.assign(s_sorted)
    expected = {
        "a": [1, 1, 1, 2, 2],
        "b": [4, 4, 3, 1, 2],
        "c": [4, 4, 3, 2, 1],
    }
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)
