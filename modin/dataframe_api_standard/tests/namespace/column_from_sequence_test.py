from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pytest

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_column_with_reference,
    integer_dataframe_1,
)


@pytest.mark.parametrize(
    ("values", "dtype", "kwargs"),
    [
        ([1, 2, 3], "Int64", {}),
        ([1, 2, 3], "Int32", {}),
        ([1, 2, 3], "Int16", {}),
        ([1, 2, 3], "Int8", {}),
        ([1, 2, 3], "UInt64", {}),
        ([1, 2, 3], "UInt32", {}),
        ([1, 2, 3], "UInt16", {}),
        ([1, 2, 3], "UInt8", {}),
        ([1.0, 2.0, 3.0], "Float64", {}),
        ([1.0, 2.0, 3.0], "Float32", {}),
        ([True, False, True], "Bool", {}),
        (["express", "yourself"], "String", {}),
        ([datetime(2020, 1, 1), datetime(2020, 1, 2)], "Datetime", {"time_unit": "us"}),
        ([timedelta(1), timedelta(2)], "Duration", {"time_unit": "us"}),
    ],
)
def test_column_from_sequence(
    library: BaseHandler,
    values: list[Any],
    dtype: str,
    kwargs: dict[str, Any],
) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    ns = ser.__column_namespace__()
    expected_dtype = getattr(ns, dtype)
    result = ns.dataframe_from_columns(
        ns.column_from_sequence(
            values,
            dtype=expected_dtype(**kwargs),
            name="result",
        ),
    )
    compare_column_with_reference(result.col("result"), values, dtype=expected_dtype)


def test_column_from_sequence_no_dtype(
    library: BaseHandler,
) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    result = ns.dataframe_from_columns(ns.column_from_sequence([1, 2, 3], name="result"))  # type: ignore[call-arg]
    expected = [1, 2, 3]
    compare_column_with_reference(result.col("result"), expected, dtype=ns.Int64)
