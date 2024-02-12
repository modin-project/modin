from __future__ import annotations

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_dataframe_with_reference,
    integer_dataframe_6,
)


def test_expression_sorted_indices_ascending(library: BaseHandler) -> None:
    df = integer_dataframe_6(library)
    ns = df.__dataframe_namespace__()
    col = df.col
    sorted_indices = col("b").sorted_indices()
    result = df.take(sorted_indices)
    expected = {"a": [2, 2, 1, 1, 1], "b": [1, 2, 3, 4, 4]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)


def test_expression_sorted_indices_descending(library: BaseHandler) -> None:
    df = integer_dataframe_6(library)
    ns = df.__dataframe_namespace__()
    col = df.col
    sorted_indices = col("b").sorted_indices(ascending=False)
    result = df.take(sorted_indices)
    expected = {"a": [1, 1, 1, 2, 2], "b": [4, 4, 3, 2, 1]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)


def test_column_sorted_indices_ascending(library: BaseHandler) -> None:
    df = integer_dataframe_6(library)
    ns = df.__dataframe_namespace__()
    sorted_indices = df.col("b").sorted_indices()
    result = df.take(sorted_indices)
    expected = {"a": [2, 2, 1, 1, 1], "b": [1, 2, 3, 4, 4]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)


def test_column_sorted_indices_descending(library: BaseHandler) -> None:
    df = integer_dataframe_6(library)
    ns = df.__dataframe_namespace__()
    sorted_indices = df.col("b").sorted_indices(ascending=False)
    result = df.take(sorted_indices)
    expected = {"a": [1, 1, 1, 2, 2], "b": [4, 4, 3, 2, 1]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)
