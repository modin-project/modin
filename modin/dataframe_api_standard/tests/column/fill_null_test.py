from __future__ import annotations

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    nan_dataframe_1,
    null_dataframe_2,
)


def test_fill_null_column(library: BaseHandler) -> None:
    df = null_dataframe_2(library)
    ser = df.col("a")
    result = df.assign(ser.fill_null(0).rename("result")).col("result")
    assert float(result.get_value(2).persist()) == 0.0  # type: ignore[arg-type]
    assert float(result.get_value(1).persist()) != 0.0  # type: ignore[arg-type]
    assert float(result.get_value(0).persist()) != 0.0  # type: ignore[arg-type]


def test_fill_null_noop_column(library: BaseHandler) -> None:
    df = nan_dataframe_1(library)
    ser = df.col("a")
    result = df.assign(ser.fill_null(0).rename("result")).persist().col("result")
    # nan was filled with 0
    assert float(result.get_value(2)) == 0  # type: ignore[arg-type]
    assert float(result.get_value(1)) != 0.0  # type: ignore[arg-type]
    assert float(result.get_value(0)) != 0.0  # type: ignore[arg-type]
