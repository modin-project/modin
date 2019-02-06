import os
import pandas
import pytest
import modin.experimental.pandas as pd


from modin.pandas.test.test_io import (
    setup_sql_file,
    teardown_sql_file,
    modin_df_equals_pandas,
)


def test_from_sql_distributed():
    if os.environ.get("MODIN_ENGINE", "") == "Ray":
        filename = "test_from_sql_distributed.db"
        teardown_sql_file(filename)
        table = "test_from_sql_distributed"
        db_uri = "sqlite:///" + filename
        setup_sql_file(db_uri, filename, table, True)
        query = "select * from {0}".format(table)

        pandas_df = pandas.read_sql(query, db_uri)
        modin_df_from_query = pd.read_sql(
            query, db_uri, partition_column="col1", lower_bound=0, upper_bound=6
        )
        modin_df_from_table = pd.read_sql(
            table, db_uri, partition_column="col1", lower_bound=0, upper_bound=6
        )

        assert modin_df_equals_pandas(modin_df_from_query, pandas_df)
        assert modin_df_equals_pandas(modin_df_from_table, pandas_df)

        teardown_sql_file(filename)


def test_from_sql_defaults():
    filename = "test_from_sql_distributed.db"
    teardown_sql_file(filename)
    table = "test_from_sql_distributed"
    db_uri = "sqlite:///" + filename
    setup_sql_file(db_uri, filename, table, True)
    query = "select * from {0}".format(table)

    pandas_df = pandas.read_sql(query, db_uri)
    with pytest.warns(UserWarning):
        modin_df_from_query = pd.read_sql(query, db_uri)
    with pytest.warns(UserWarning):
        modin_df_from_table = pd.read_sql(table, db_uri)

    assert modin_df_equals_pandas(modin_df_from_query, pandas_df)
    assert modin_df_equals_pandas(modin_df_from_table, pandas_df)

    teardown_sql_file(filename)
