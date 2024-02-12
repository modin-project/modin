from __future__ import annotations

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_dataframe_with_reference,
    nan_dataframe_2,
    null_dataframe_1,
)


def test_is_null_1(library: BaseHandler) -> None:
    df = nan_dataframe_2(library)
    ns = df.__dataframe_namespace__()
    result = df.is_null()
    expected = {"a": [False, False, False]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Bool)


def test_is_null_2(library: BaseHandler) -> None:
    df = null_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    result = df.is_null()
    expected = {"a": [False, False, True]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Bool)
