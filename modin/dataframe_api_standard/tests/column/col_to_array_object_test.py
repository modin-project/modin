from __future__ import annotations

import numpy as np
import pytest

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    bool_dataframe_1,
    integer_dataframe_1,
)


@pytest.mark.parametrize(
    "dtype",
    [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float32",
        "float64",
    ],
)
def test_column_to_array_object(
    library: BaseHandler, dtype: str
) -> None:  # noqa: ARG001
    ser = integer_dataframe_1(library).col("a").persist()
    result = np.asarray(ser.to_array())
    expected = np.array([1, 2, 3], dtype=np.int64)
    np.testing.assert_array_equal(result, expected)


def test_column_to_array_object_bool(library: BaseHandler) -> None:
    df = bool_dataframe_1(library).persist().col("a")
    result = np.asarray(df.to_array())
    expected = np.array([True, True, False], dtype="bool")
    np.testing.assert_array_equal(result, expected)


def test_column_to_array_object_invalid(library: BaseHandler) -> None:
    df = bool_dataframe_1(library).col("a")
    with pytest.raises(RuntimeError):
        _ = np.asarray(df.to_array())
