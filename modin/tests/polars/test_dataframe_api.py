import polars
import modin.polars as pl


def test_init_roundtrip():
    data = {"a": [1, 2, 3], "b": [4, 5, 6]}
    df = pl.DataFrame(data)
    polars_df = polars.DataFrame(data)
    to_polars = polars.from_pandas(df._query_compiler.to_pandas())
    assert polars_df.frame_equal(to_polars)
