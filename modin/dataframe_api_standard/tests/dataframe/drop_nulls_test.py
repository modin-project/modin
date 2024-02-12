from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_dataframe_with_reference,
    null_dataframe_1,
)


def test_drop_nulls(library: BaseHandler) -> None:
    df = null_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    result = df.drop_nulls()
    expected = {"a": [1.0, 2.0]}
    compare_dataframe_with_reference(result, expected, dtype=ns.Float64)
