from __future__ import annotations

from modin.dataframe_api_standard.tests.utils import BaseHandler, integer_dataframe_1


def test_column_len(library: BaseHandler) -> None:
    result = integer_dataframe_1(library).col("a").n_unique().persist().scalar
    assert result == 3
