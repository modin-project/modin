from __future__ import annotations

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_dataframe_with_reference,
    integer_dataframe_4,
)


def test_group_by_size(library: BaseHandler) -> None:
    df = integer_dataframe_4(library)
    ns = df.__dataframe_namespace__()
    result = df.group_by("key").size()
    result = result.sort("key")
    expected = {"key": [1, 2], "size": [2, 2]}
    result = result.cast({"size": ns.Int64()})
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)
