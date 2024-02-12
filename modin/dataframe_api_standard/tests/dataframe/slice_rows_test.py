from __future__ import annotations

from typing import Any

import pytest

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_dataframe_with_reference,
    integer_dataframe_3,
)


@pytest.mark.parametrize(
    ("start", "stop", "step", "expected"),
    [
        (2, 7, 2, {"a": [3, 5, 7], "b": [5, 3, 1]}),
        (None, 7, 2, {"a": [1, 3, 5, 7], "b": [7, 5, 3, 1]}),
        (2, None, 2, {"a": [3, 5, 7], "b": [5, 3, 1]}),
        (2, None, None, {"a": [3, 4, 5, 6, 7], "b": [5, 4, 3, 2, 1]}),
    ],
)
def test_slice_rows(
    library: BaseHandler,
    start: int | None,
    stop: int | None,
    step: int | None,
    expected: dict[str, Any],
) -> None:
    df = integer_dataframe_3(library)
    ns = df.__dataframe_namespace__()
    result = df.slice_rows(start, stop, step)
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)
