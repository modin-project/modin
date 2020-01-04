import os
import pandas
import pytest
import modin.experimental.pandas as pd
from modin.pandas.test.test_io import (  # noqa: F401
    modin_df_equals_pandas,
    make_sql_connection,
)


@pytest.mark.skipif(
    os.environ.get("MODIN_ENGINE", "Ray").title() == "Dask",
    reason="Dask does not have experimental API",
)
def test_from_sql_distributed(make_sql_connection):  # noqa: F811
    if os.environ.get("MODIN_ENGINE", "") == "Ray":
        filename = "test_from_sql_distributed.db"
        table = "test_from_sql_distributed"
        conn = make_sql_connection(filename, table)
        query = "select * from {0}".format(table)

        pandas_df = pandas.read_sql(query, conn)
        modin_df_from_query = pd.read_sql(
            query, conn, partition_column="col1", lower_bound=0, upper_bound=6
        )
        modin_df_from_table = pd.read_sql(
            table, conn, partition_column="col1", lower_bound=0, upper_bound=6
        )

        assert modin_df_equals_pandas(modin_df_from_query, pandas_df)
        assert modin_df_equals_pandas(modin_df_from_table, pandas_df)


@pytest.mark.skipif(
    os.environ.get("MODIN_ENGINE", "Ray").title() == "Dask",
    reason="Dask does not have experimental API",
)
def test_from_sql_defaults(make_sql_connection):  # noqa: F811
    filename = "test_from_sql_distributed.db"
    table = "test_from_sql_distributed"
    conn = make_sql_connection(filename, table)
    query = "select * from {0}".format(table)

    pandas_df = pandas.read_sql(query, conn)
    with pytest.warns(UserWarning):
        modin_df_from_query = pd.read_sql(query, conn)
    with pytest.warns(UserWarning):
        modin_df_from_table = pd.read_sql(table, conn)

    assert modin_df_equals_pandas(modin_df_from_query, pandas_df)
    assert modin_df_equals_pandas(modin_df_from_table, pandas_df)
