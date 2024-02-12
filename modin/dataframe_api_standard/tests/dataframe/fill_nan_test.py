from __future__ import annotations

import pytest

from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_dataframe_with_reference,
    nan_dataframe_1,
)
from modin.pandas.test.utils import default_to_pandas_ignore_string

pytestmark = pytest.mark.filterwarnings(default_to_pandas_ignore_string)


def test_fill_nan(library: BaseHandler) -> None:
    df = nan_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    result = df.fill_nan(-1)
    result = result.cast({"a": ns.Float64()})
    expected = {"a": [1.0, 2.0, -1.0]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Float64)


def test_fill_nan_with_scalar(library: BaseHandler) -> None:
    df = nan_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    result = df.fill_nan(df.col("a").get_value(0))
    result = result.cast({"a": ns.Float64()})
    expected = {"a": [1.0, 2.0, 1.0]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Float64)


def test_fill_nan_with_scalar_invalid(library: BaseHandler) -> None:
    df = nan_dataframe_1(library)
    other = df + 1
    with pytest.raises(ValueError):
        _ = df.fill_nan(other.col("a").get_value(0))


def test_fill_nan_with_null(library: BaseHandler) -> None:
    df = nan_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    result = df.fill_nan(ns.null)
    n_nans = result.is_nan().sum()
    result = n_nans.col("a").persist().get_value(0).scalar
    # null is nan for pandas-numpy
    assert result == 1
