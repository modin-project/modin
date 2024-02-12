from __future__ import annotations

import pytest

from modin.dataframe_api_standard.tests.utils import BaseHandler, integer_dataframe_1


def test_shape(library: BaseHandler) -> None:
    df = integer_dataframe_1(library).persist()
    assert df.shape() == (3, 2)

    df = integer_dataframe_1(library)
    with pytest.raises(ValueError):
        df.shape()
