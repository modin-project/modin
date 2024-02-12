from __future__ import annotations

import pytest

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    integer_dataframe_1,
    integer_dataframe_2,
)


def test_invalid_comparisons(library: BaseHandler) -> None:
    with pytest.raises(ValueError):
        _ = integer_dataframe_1(library).col("a") > integer_dataframe_2(library).col(
            "a"
        )


def test_invalid_comparisons_scalar(library: BaseHandler) -> None:
    with pytest.raises(ValueError):
        _ = (
            integer_dataframe_1(library).col("a")
            > integer_dataframe_2(library).col("a").mean()
        )
