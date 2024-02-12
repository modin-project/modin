from __future__ import annotations

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_dataframe_with_reference,
    nan_dataframe_1,
)


def test_dataframe_is_nan(library: BaseHandler) -> None:
    df = nan_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    result = df.is_nan()
    expected = {"a": [False, False, True]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Bool)
