from __future__ import annotations

from modin.dataframe_api_standard.tests.utils import BaseHandler, integer_dataframe_1


def test_rename(library: BaseHandler) -> None:
    df = integer_dataframe_1(library).persist()
    ser = df.col("a")
    result = ser.rename("new_name")
    assert result.name == "new_name"
