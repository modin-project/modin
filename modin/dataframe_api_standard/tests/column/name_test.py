from __future__ import annotations

from modin.dataframe_api_standard.tests.utils import BaseHandler, integer_dataframe_1


def test_name(library: BaseHandler) -> None:
    df = integer_dataframe_1(library).persist()
    name = df.col("a").name
    assert name == "a"


def test_pandas_name_if_0_named_column(library) -> None:
    df = library.dataframe({0: [1, 2, 3]})
    assert df.column_names == [0]  # type: ignore[comparison-overlap]
    assert [col.name for col in df.iter_columns()] == [0]  # type: ignore[comparison-overlap]
