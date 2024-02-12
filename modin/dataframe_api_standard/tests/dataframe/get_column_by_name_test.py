from __future__ import annotations

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_dataframe_with_reference,
    integer_dataframe_1,
)


def test_get_column(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    col = df.col
    result = df.assign(col("a").rename("_tmp")).drop("a").rename({"_tmp": "a"})
    expected = {"b": [4, 5, 6], "a": [1, 2, 3]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)
