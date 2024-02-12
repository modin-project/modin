from __future__ import annotations

from modin.dataframe_api_standard.tests.utils import BaseHandler, mixed_dataframe_1


def test_schema(library: BaseHandler) -> None:
    df = mixed_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    result = df.col("a").dtype
    assert isinstance(result, namespace.Int64)
