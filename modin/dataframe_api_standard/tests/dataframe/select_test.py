from __future__ import annotations

import pytest

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_dataframe_with_reference,
    integer_dataframe_1,
)
from modin.pandas.test.utils import default_to_pandas_ignore_string

pytestmark = pytest.mark.filterwarnings(default_to_pandas_ignore_string)


def test_select(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    result = df.select("b")
    expected = {"b": [4, 5, 6]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)


def test_select_list_of_str(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    result = df.select("a", "b")
    expected = {"a": [1, 2, 3], "b": [4, 5, 6]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)


def test_select_list_of_str_invalid(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    with pytest.raises(TypeError):
        _ = df.select(["a", "b"])  # type: ignore[arg-type]


@pytest.mark.filterwarnings("ignore:np.find_common_type is deprecated")
def test_select_empty(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    result = df.select()
    assert result.column_names == []
