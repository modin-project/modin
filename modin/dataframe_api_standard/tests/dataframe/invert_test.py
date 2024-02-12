from __future__ import annotations

import pytest

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    bool_dataframe_1,
    compare_dataframe_with_reference,
    integer_dataframe_1,
)


def test_invert(library: BaseHandler) -> None:
    df = bool_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    result = ~df
    expected = {"a": [False, False, True], "b": [False, False, False]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Bool)


def test_invert_invalid(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    with pytest.raises(TypeError):
        _ = ~df
