from __future__ import annotations

import pytest

from modin.dataframe_api_standard.tests.utils import BaseHandler, integer_dataframe_1


def test_groupby_invalid(library: BaseHandler) -> None:
    df = integer_dataframe_1(library).select("a")
    with pytest.raises((KeyError, TypeError)):
        df.group_by(0)  # type: ignore[arg-type]
    with pytest.raises((KeyError, TypeError)):
        df.group_by("0")
    with pytest.raises((KeyError, TypeError)):
        df.group_by("b")
