from __future__ import annotations

import pytest

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_dataframe_with_reference,
    integer_dataframe_1,
)


def test_rename(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    result = df.rename({"a": "c", "b": "e"})
    expected = {"c": [1, 2, 3], "e": [4, 5, 6]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)


def test_rename_invalid(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    with pytest.raises(
        TypeError,
        match="Expected Mapping, got: <class 'function'>",
    ):  # pragma: no cover
        # why is this not covered? bug in coverage?
        df.rename(lambda x: x.upper())  # type: ignore  # noqa: PGH003
