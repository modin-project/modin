from __future__ import annotations

import pytest

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    bool_dataframe_1,
    bool_dataframe_3,
    compare_dataframe_with_reference,
)


@pytest.mark.parametrize(
    ("reduction", "expected_data"),
    [
        ("any", {"a": [True], "b": [True]}),
        ("all", {"a": [False], "b": [True]}),
    ],
)
def test_reductions(
    library: BaseHandler,
    reduction: str,
    expected_data: dict[str, object],
) -> None:
    df = bool_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    result = getattr(df, reduction)()
    compare_dataframe_with_reference(result, expected_data, dtype=ns.Bool)  # type: ignore[arg-type]


def test_any(library: BaseHandler) -> None:
    df = bool_dataframe_3(library)
    ns = df.__dataframe_namespace__()
    result = df.any()
    expected = {"a": [False], "b": [True], "c": [True]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Bool)


def test_all(library: BaseHandler) -> None:
    df = bool_dataframe_3(library)
    ns = df.__dataframe_namespace__()
    result = df.all()
    expected = {"a": [False], "b": [False], "c": [True]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Bool)
