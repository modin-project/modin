from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_dataframe_with_reference,
    integer_dataframe_1,
)


def test_cast_integers(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    result = df.cast({"a": ns.Int32()})
    expected = {"a": [1, 2, 3], "b": [4, 5, 6]}
    expected_dtype = {"a": ns.Int32, "b": ns.Int64}
    compare_dataframe_with_reference(result, expected, dtype=expected_dtype)
