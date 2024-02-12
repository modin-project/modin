import numpy as np

from modin.dataframe_api_standard.tests.utils import BaseHandler, integer_dataframe_1


def test_free_vs_w_parent(library: BaseHandler) -> None:
    df1 = integer_dataframe_1(library)
    namespace = df1.__dataframe_namespace__()
    free_ser1 = namespace.column_from_1d_array(  # type: ignore[call-arg]
        np.array([1, 2, 3], dtype="int64"),
        name="preds",
    )
    free_ser2 = namespace.column_from_1d_array(  # type: ignore[call-arg]
        np.array([4, 5, 6], dtype="int64"),
        name="preds",
    )

    result = free_ser1 + free_ser2
    assert namespace.is_dtype(result.dtype, "integral")
