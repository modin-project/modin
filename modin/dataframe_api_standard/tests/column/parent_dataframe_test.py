from modin.dataframe_api_standard.tests.utils import BaseHandler, integer_dataframe_1


def test_parent_dataframe(library: BaseHandler) -> None:
    df = integer_dataframe_1(library)
    assert df.col("a").parent_dataframe is df
