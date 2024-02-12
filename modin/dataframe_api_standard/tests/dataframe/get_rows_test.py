from __future__ import annotations

import pytest

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_dataframe_with_reference,
    integer_dataframe_1,
)
from modin.pandas.test.utils import default_to_pandas_ignore_string

pytestmark = pytest.mark.filterwarnings(default_to_pandas_ignore_string)


def test_take(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    df = df.assign((df.col("a") - 1).sort(ascending=False).rename("result"))
    result = df.take(df.col("result"))
    expected = {"a": [3, 2, 1], "b": [6, 5, 4], "result": [0, 1, 2]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Int64)
