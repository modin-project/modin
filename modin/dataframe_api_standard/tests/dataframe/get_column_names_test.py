from __future__ import annotations

from modin.dataframe_api_standard.tests.utils import BaseHandler, integer_dataframe_1


def test_get_column_names(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    result = df.column_names
    assert list(result) == ["a", "b"]
