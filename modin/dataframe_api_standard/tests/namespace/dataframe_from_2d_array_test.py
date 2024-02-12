from __future__ import annotations

import numpy as np

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_dataframe_with_reference,
    integer_dataframe_1,
)


def test_dataframe_from_2d_array(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    arr = np.array([[1, 4], [2, 5], [3, 6]])
    result = ns.dataframe_from_2d_array(
        arr,
        names=["a", "b"],
    )
    # TODO: consistent return type, for windows compat?
    result = result.cast({"a": ns.Int64(), "b": ns.Int64()})
    expected = {"a": [1, 2, 3], "b": [4, 5, 6]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)
