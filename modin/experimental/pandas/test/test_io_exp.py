# Licensed to Modin Development Team under one or more contributor license agreements.
# See the NOTICE file distributed with this work for additional information regarding
# copyright ownership.  The Modin Development Team licenses this file to you under the
# Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.

import pandas
import pytest
import modin.experimental.pandas as pd
from modin.config import Engine
from modin.pandas.test.test_io import (  # noqa: F401
    df_equals,
    make_sql_connection,
)


@pytest.mark.skipif(
    Engine.get() == "Dask",
    reason="Dask does not have experimental API",
)
def test_from_sql_distributed(make_sql_connection):  # noqa: F811
    if Engine.get() == "Ray":
        pytest.xfail("Distributed read_sql is broken, see GH#2194")
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

        df_equals(modin_df_from_query, pandas_df)
        df_equals(modin_df_from_table, pandas_df)


@pytest.mark.skipif(
    Engine.get() == "Dask",
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

    df_equals(modin_df_from_query, pandas_df)
    df_equals(modin_df_from_table, pandas_df)
