from __future__ import annotations

from modin.dataframe_api_standard.tests.utils import BaseHandler, integer_dataframe_1


def test_get_value(library: BaseHandler) -> None:
    result = integer_dataframe_1(library).persist().col("a").get_value(0)
    assert int(result) == 1  # type: ignore[call-overload]


def test_mean_scalar(library: BaseHandler) -> None:
    result = integer_dataframe_1(library).persist().col("a").max()
    assert int(result) == 3  # type: ignore[call-overload]
