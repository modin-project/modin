from __future__ import annotations

import modin.pandas as pd


def test_convert_to_std_column() -> None:
    s = pd.Series([1, 2, 3]).__column_consortium_standard__()
    assert float(s.mean()) == 2
    s = pd.Series([1, 2, 3], name="alice").__column_consortium_standard__()
    assert float(s.mean()) == 2
