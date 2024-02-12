from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_column_with_reference,
    float_dataframe_1,
    float_dataframe_2,
    float_dataframe_3,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.mark.parametrize(
    ("df_factory", "expected_values"),
    [
        (float_dataframe_1, [False, True]),
        (float_dataframe_2, [True, False]),
        (float_dataframe_3, [True, False]),
    ],
)
@pytest.mark.filterwarnings("ignore:np.find_common_type is deprecated")
def test_is_in(
    library: BaseHandler,
    df_factory: Callable[[BaseHandler], Any],
    expected_values: list[bool],
) -> None:
    df = df_factory(library)
    ns = df.__dataframe_namespace__()
    ser = df.col("a")
    other = ser + 1
    result = df.assign(ser.is_in(other).rename("result"))
    compare_column_with_reference(result.col("result"), expected_values, dtype=ns.Bool)


@pytest.mark.parametrize(
    ("df_factory", "expected_values"),
    [
        (float_dataframe_1, [False, True]),
        (float_dataframe_2, [True, False]),
        (float_dataframe_3, [True, False]),
    ],
)
@pytest.mark.filterwarnings("ignore:np.find_common_type is deprecated")
def test_expr_is_in(
    library: BaseHandler,
    df_factory: Callable[[BaseHandler], Any],
    expected_values: list[bool],
) -> None:
    df = df_factory(library)
    ns = df.__dataframe_namespace__()
    col = df.col
    ser = col("a")
    other = ser + 1
    result = df.assign(ser.is_in(other).rename("result"))
    compare_column_with_reference(result.col("result"), expected_values, dtype=ns.Bool)
