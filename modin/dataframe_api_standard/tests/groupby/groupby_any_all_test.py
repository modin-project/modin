from __future__ import annotations

import pytest

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    bool_dataframe_2,
    compare_dataframe_with_reference,
)


@pytest.mark.parametrize(
    ("aggregation", "expected_b", "expected_c"),
    [
        ("any", [True, True], [True, False]),
        ("all", [False, True], [False, False]),
    ],
)
def test_groupby_boolean(
    library: BaseHandler,
    aggregation: str,
    expected_b: list[bool],
    expected_c: list[bool],
) -> None:
    df = bool_dataframe_2(library)
    ns = df.__dataframe_namespace__()
    result = getattr(df.group_by("key"), aggregation)()
    # need to sort
    result = result.sort("key")
    expected = {"key": [1, 2], "b": expected_b, "c": expected_c}
    expected_dtype = {"key": ns.Int64, "b": ns.Bool, "c": ns.Bool}
    compare_dataframe_with_reference(result, expected, dtype=expected_dtype)  # type: ignore[arg-type]
