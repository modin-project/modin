from __future__ import annotations

import pytest

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_dataframe_with_reference,
    integer_dataframe_5,
)


@pytest.mark.parametrize("keys", [["a", "b"], []])
def test_sort(library: BaseHandler, keys: list[str]) -> None:
    df = integer_dataframe_5(library, api_version="2023.09-beta")
    ns = df.__dataframe_namespace__()
    result = df.sort(*keys)
    expected = {"a": [1, 1], "b": [3, 4]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)


@pytest.mark.parametrize("keys", [["a", "b"], []])
def test_sort_descending(
    library: BaseHandler,
    keys: list[str],
) -> None:
    df = integer_dataframe_5(library, api_version="2023.09-beta")
    ns = df.__dataframe_namespace__()
    result = df.sort(*keys, ascending=False)
    expected = {"a": [1, 1], "b": [4, 3]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)
