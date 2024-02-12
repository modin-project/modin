from __future__ import annotations

import pytest

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_column_with_reference,
    nan_dataframe_1,
)
from modin.pandas.test.utils import default_to_pandas_ignore_string

pytestmark = pytest.mark.filterwarnings(default_to_pandas_ignore_string)


def test_column_fill_nan(library: BaseHandler) -> None:
    # TODO: test with nullable pandas, check null isn't filled
    df = nan_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    result = df.assign(ser.fill_nan(-1.0).rename("result"))
    expected = [1.0, 2.0, -1.0]
    compare_column_with_reference(result.col("result"), expected, dtype=ns.Float64)


def test_column_fill_nan_with_null(library: BaseHandler) -> None:
    # TODO: test with nullable pandas, check null isn't filled
    df = nan_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    result = df.assign(ser.fill_nan(ns.null).is_null().rename("result"))
    expected = [False, False, True]
    compare_column_with_reference(result.col("result"), expected, dtype=ns.Bool)
