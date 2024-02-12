from __future__ import annotations

import pytest

import modin.pandas as pd
from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_column_with_reference,
    integer_dataframe_1,
)


@pytest.mark.parametrize(
    ("func", "expected_data"),
    [
        ("cumulative_sum", [1, 3, 6]),
        ("cumulative_prod", [1, 2, 6]),
        ("cumulative_max", [1, 2, 3]),
        ("cumulative_min", [1, 1, 1]),
    ],
)
def test_cumulative_functions_column(
    library: BaseHandler,
    func: str,
    expected_data: list[float],
) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    expected = pd.Series(expected_data, name="result")
    result = df.assign(getattr(ser, func)().rename("result"))
    compare_column_with_reference(result.col("result"), expected, dtype=ns.Int64)
