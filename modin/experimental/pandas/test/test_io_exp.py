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

from platform import java_ver
import pandas
import pytest
from typing import Callable

import modin.experimental.pandas as pd
from modin.config import Engine
from modin.pandas.test.test_io import (  # noqa: F401
    df_equals,
    make_sql_connection,
    teardown_test_files,
)
from modin.pandas.test.utils import get_unique_filename


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


def test_read_json_row_partitions():
    fake_json_input = "fake_json"

    def split_json_string(json_string):
        assert json_string == fake_json_input, "can only split fake json"
        return ["fake_json_part1", "fake_json_part2"]

    def json_to_dataframes(json_string):
        return [
            pandas.DataFrame({0: [json_string]}),
            pandas.DataFrame({1: [json_string]}),
        ]

    actual_df1, actual_df2 = pd.read_json_row_partitions(
        fake_json_input, split_json_string, json_to_dataframes
    )

    df_equals(actual_df1, pandas.DataFrame({0: ["fake_json_part1", "fake_json_part2"]}))
    df_equals(actual_df2, pandas.DataFrame({1: ["fake_json_part1", "fake_json_part2"]}))
