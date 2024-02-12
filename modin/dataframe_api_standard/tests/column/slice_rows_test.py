from __future__ import annotations

from typing import Any

import pytest

import modin.pandas as pd
from modin.dataframe_api_standard.tests.utils import BaseHandler, integer_dataframe_3
from modin.pandas.test.utils import df_equals


@pytest.mark.parametrize(
    ("start", "stop", "step", "expected"),
    [
        (2, 7, 2, pd.Series([3, 5, 7], name="result")),
        (None, 7, 2, pd.Series([1, 3, 5, 7], name="result")),
        (2, None, 2, pd.Series([3, 5, 7], name="result")),
        (2, None, None, pd.Series([3, 4, 5, 6, 7], name="result")),
    ],
)
def test_column_slice_rows(
    library: BaseHandler,
    start: int | None,
    stop: int | None,
    step: int | None,
    expected: pd.Series[Any],
) -> None:
    ser = integer_dataframe_3(library).col("a")
    result = ser.slice_rows(start, stop, step).persist()
    result_pd = pd.Series(result.to_array(), name="result")
    df_equals(result_pd, expected)
