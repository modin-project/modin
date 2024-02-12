from __future__ import annotations

import pytest

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    integer_dataframe_1,
    integer_dataframe_2,
)


def test_invalid_comparisons(library: BaseHandler) -> None:
    df1 = integer_dataframe_1(library)
    df2 = integer_dataframe_2(library)
    mask = df2.col("a") > 1
    with pytest.raises(ValueError):
        _ = df1.filter(mask)
