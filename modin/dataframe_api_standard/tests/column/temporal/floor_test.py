from __future__ import annotations

from datetime import datetime

import pytest

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_column_with_reference,
    temporal_dataframe_1,
)


@pytest.mark.parametrize(
    ("freq", "expected"),
    [
        ("1day", [datetime(2020, 1, 1), datetime(2020, 1, 2), datetime(2020, 1, 3)]),
    ],
)
def test_floor(library: BaseHandler, freq: str, expected: list[datetime]) -> None:
    df = temporal_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    col = df.col
    result = df.assign(col("a").floor(freq).rename("result")).select("result")  # type: ignore[attr-defined]
    # TODO check the resolution
    compare_column_with_reference(result.col("result"), expected, dtype=ns.Datetime)
