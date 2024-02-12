from __future__ import annotations

import pytest

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_dataframe_with_reference,
    integer_dataframe_1,
    integer_dataframe_2,
    integer_dataframe_4,
)


def test_concat(library: BaseHandler) -> None:
    df1 = integer_dataframe_1(library)
    df2 = integer_dataframe_2(library)
    ns = df1.__dataframe_namespace__()
    result = ns.concat([df1, df2])
    expected = {"a": [1, 2, 3, 1, 2, 4], "b": [4, 5, 6, 4, 2, 6]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)


def test_concat_mismatch(library: BaseHandler) -> None:
    df1 = integer_dataframe_1(library).persist()
    df2 = integer_dataframe_4(library).persist()
    ns = df1.__dataframe_namespace__()
    # TODO check the error
    with pytest.raises(ValueError):
        _ = ns.concat([df1, df2]).persist()
