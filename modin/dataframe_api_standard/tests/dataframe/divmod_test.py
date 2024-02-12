from __future__ import annotations

import pytest

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_dataframe_with_reference,
    integer_dataframe_1,
)
from modin.pandas.test.utils import default_to_pandas_ignore_string

pytestmark = pytest.mark.filterwarnings(default_to_pandas_ignore_string)


def test_divmod_with_scalar(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    other = 2
    result_quotient, result_remainder = df.__divmod__(other)
    expected_quotient = {"a": [0, 1, 1], "b": [2, 2, 3]}
    expected_remainder = {"a": [1, 0, 1], "b": [0, 1, 0]}
    compare_dataframe_with_reference(result_quotient, expected_quotient, dtype=ns.Int64)
    compare_dataframe_with_reference(
        result_remainder, expected_remainder, dtype=ns.Int64
    )
