from __future__ import annotations

import pytest

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_dataframe_with_reference,
    integer_dataframe_1,
)


def test_insert_columns(library: BaseHandler) -> None:
    df = integer_dataframe_1(library, api_version="2023.09-beta")
    ns = df.__dataframe_namespace__()
    new_col = (df.col("b") + 3).rename("result")
    result = df.assign(new_col.rename("c"))
    expected = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)
    # check original df didn't change
    expected = {"a": [1, 2, 3], "b": [4, 5, 6]}
    compare_dataframe_with_reference(df, expected, dtype=ns.Int64)


def test_insert_multiple_columns(library: BaseHandler) -> None:
    df = integer_dataframe_1(library, api_version="2023.09-beta")
    ns = df.__dataframe_namespace__()
    new_col = (df.col("b") + 3).rename("result")
    result = df.assign(new_col.rename("c"), new_col.rename("d"))
    expected = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [7, 8, 9]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)
    # check original df didn't change
    expected = {"a": [1, 2, 3], "b": [4, 5, 6]}
    compare_dataframe_with_reference(df, expected, dtype=ns.Int64)


def test_insert_multiple_columns_invalid(library: BaseHandler) -> None:
    df = integer_dataframe_1(library, api_version="2023.09-beta")
    df.__dataframe_namespace__()
    new_col = (df.col("b") + 3).rename("result")
    with pytest.raises(TypeError):
        _ = df.assign([new_col.rename("c"), new_col.rename("d")])  # type: ignore[arg-type]


def test_insert_eager_columns(library: BaseHandler) -> None:
    df = integer_dataframe_1(library, api_version="2023.09-beta")
    ns = df.__dataframe_namespace__()
    new_col = (df.col("b") + 3).rename("result")
    result = df.assign(new_col.rename("c"), new_col.rename("d"))
    expected = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9], "d": [7, 8, 9]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)
    # check original df didn't change
    expected = {"a": [1, 2, 3], "b": [4, 5, 6]}
    compare_dataframe_with_reference(df, expected, dtype=ns.Int64)
