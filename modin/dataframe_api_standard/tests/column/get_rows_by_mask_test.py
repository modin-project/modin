from __future__ import annotations

import modin.pandas as pd
from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_column_with_reference,
    integer_dataframe_1,
)
from modin.pandas.test.utils import df_equals


def test_column_filter(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ser = df.col("a")
    mask = ser > 1
    ser = ser.filter(mask).persist()
    result_pd = pd.Series(ser.to_array(), name="result")
    expected = pd.Series([2, 3], name="result")
    df_equals(result_pd, expected)


def test_column_take_by_mask_noop(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    mask = ser > 0
    ser = ser.filter(mask)
    result = df.assign(ser.rename("result"))
    compare_column_with_reference(result.col("result"), [1, 2, 3], dtype=ns.Int64)
