from __future__ import annotations

import modin.pandas as pd
from modin.pandas.test.utils import df_equals


def test_scale_column_modin() -> None:
    s = pd.Series([1, 2, 3], name="a")
    ser = s.__column_consortium_standard__()
    ser = ser - ser.mean()
    result = ser.column
    df_equals(result, pd.Series([-1, 0, 1.0], name="a"))
