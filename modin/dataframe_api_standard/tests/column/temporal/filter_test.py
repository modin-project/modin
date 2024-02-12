from modin.dataframe_api_standard.tests.utils import (
    BaseHandler,
    compare_dataframe_with_reference,
    temporal_dataframe_1,
)


def test_filter_w_date(library: BaseHandler) -> None:
    df = temporal_dataframe_1(library).select("a", "index")
    ns = df.__dataframe_namespace__()
    result = df.filter(df.col("a") > ns.date(2020, 1, 2)).select("index")
    compare_dataframe_with_reference(result, {"index": [1, 2]}, dtype=ns.Int64)
