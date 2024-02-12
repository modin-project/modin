from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_dataframe_with_reference,
    integer_dataframe_1,
)


def test_iter_columns(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    ns = df.__dataframe_namespace__()
    result = df.assign(
        *[col / col.mean() for col in df.iter_columns()],
    )
    expected = {
        "a": [0.5, 1.0, 1.5],
        "b": [0.8, 1.0, 1.2],
    }
    compare_dataframe_with_reference(result, expected, dtype=ns.Float64)
