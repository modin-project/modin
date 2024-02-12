from __future__ import annotations

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_column_with_reference,
    integer_dataframe_1,
)


def test_mean(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    result = df.assign((df.col("a") - df.col("a").mean()).rename("result"))
    compare_column_with_reference(result.col("result"), [-1, 0, 1.0], dtype=ns.Float64)
