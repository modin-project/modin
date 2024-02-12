from __future__ import annotations

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_dataframe_with_reference,
    integer_dataframe_1,
)


def test_float_scalar_powers(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    other = 1.0
    result = df.__pow__(other)
    result = result.cast({"a": ns.Int64(), "b": ns.Int64()})
    expected = {"a": [1, 2, 3], "b": [4, 5, 6]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)
