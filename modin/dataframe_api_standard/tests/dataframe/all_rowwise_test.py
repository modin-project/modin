from __future__ import annotations

import pytest

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    bool_dataframe_1,
    compare_dataframe_with_reference,
)


def test_all_horizontal(library: BaseHandler) -> None:
    df = bool_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    mask = ns.all_horizontal(*[df.col(col_name) for col_name in df.column_names])
    result = df.filter(mask)
    expected = {"a": [True, True], "b": [True, True]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Bool)


def test_all_horizontal_invalid(library: BaseHandler) -> None:
    df = bool_dataframe_1(library)
    namespace = df.__dataframe_namespace__()
    with pytest.raises(ValueError):
        _ = namespace.all_horizontal(df.col("a"), (df + 1).col("b"))
