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

import os
import glob
import pandas
import pytest
import modin.experimental.pandas as pd
from modin.config import Engine
from modin.pandas.test.test_io import (  # noqa: F401
    df_equals,
    eval_io,
    make_sql_connection,
    _make_csv_file,
    teardown_test_files,
)
from modin.pandas.test.utils import get_unique_filename, test_data


@pytest.mark.skipif(
    Engine.get() == "Dask",
    reason="Dask does not have experimental API",
)
def test_from_sql_distributed(make_sql_connection):  # noqa: F811
    if Engine.get() == "Ray":
        filename = "test_from_sql_distributed.db"
        table = "test_from_sql_distributed"
        conn = make_sql_connection(filename, table)
        query = "select * from {0}".format(table)

        pandas_df = pandas.read_sql(query, conn)
        modin_df_from_query = pd.read_sql(
            query,
            conn,
            partition_column="col1",
            lower_bound=0,
            upper_bound=6,
            max_sessions=2,
        )
        modin_df_from_table = pd.read_sql(
            table,
            conn,
            partition_column="col1",
            lower_bound=0,
            upper_bound=6,
            max_sessions=2,
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


@pytest.fixture(scope="class")
def TestReadGlobCSVFixture():
    filenames = []

    base_name = get_unique_filename(extension="")
    pytest.glob_path = "{}_*.csv".format(base_name)
    pytest.files = ["{}_{}.csv".format(base_name, i) for i in range(11)]
    for fname in pytest.files:
        # Glob does not guarantee ordering so we have to remove the randomness in the generated csvs.
        _make_csv_file(filenames)(fname, row_size=11, remove_randomness=True)

    yield

    teardown_test_files(filenames)


@pytest.mark.usefixtures("TestReadGlobCSVFixture")
@pytest.mark.skipif(
    Engine.get() != "Ray", reason="Currently only support Ray engine for glob paths."
)
class TestCsvGlob:
    def test_read_multiple_small_csv(self):  # noqa: F811
        pandas_df = pandas.concat([pandas.read_csv(fname) for fname in pytest.files])
        modin_df = pd.read_csv_glob(pytest.glob_path)

        # Indexes get messed up when concatting so we reset both.
        pandas_df = pandas_df.reset_index(drop=True)
        modin_df = modin_df.reset_index(drop=True)

        df_equals(modin_df, pandas_df)

    @pytest.mark.parametrize("nrows", [35, 100])
    def test_read_multiple_csv_nrows(self, request, nrows):  # noqa: F811
        pandas_df = pandas.concat([pandas.read_csv(fname) for fname in pytest.files])
        pandas_df = pandas_df.iloc[:nrows, :]

        modin_df = pd.read_csv_glob(pytest.glob_path, nrows=nrows)

        # Indexes get messed up when concatting so we reset both.
        pandas_df = pandas_df.reset_index(drop=True)
        modin_df = modin_df.reset_index(drop=True)

        df_equals(modin_df, pandas_df)

    def test_read_csv_empty_frame(self):
        kwargs = {
            "usecols": [0],
            "index_col": 0,
        }

        modin_df = pd.read_csv_glob(pytest.files[0], **kwargs)
        pandas_df = pandas.read_csv(pytest.files[0], **kwargs)

        df_equals(modin_df, pandas_df)


@pytest.mark.skipif(
    Engine.get() != "Ray", reason="Currently only support Ray engine for glob paths."
)
def test_read_multiple_csv_s3():
    modin_df = pd.read_csv_glob("S3://noaa-ghcn-pds/csv/178*.csv")

    # We have to specify the columns because the column names are not identical. Since we specified the column names, we also have to skip the original column names.
    pandas_dfs = [
        pandas.read_csv(
            "s3://noaa-ghcn-pds/csv/178{}.csv".format(i),
            names=modin_df.columns,
            skiprows=[0],
        )
        for i in range(10)
    ]
    pandas_df = pd.concat(pandas_dfs)

    # Indexes get messed up when concatting so we reset both.
    pandas_df = pandas_df.reset_index(drop=True)
    modin_df = modin_df.reset_index(drop=True)

    df_equals(modin_df, pandas_df)


@pytest.mark.skipif(
    not Engine.get() == "Ray",
    reason=f"{Engine.get()} does not have experimental API",
)
@pytest.mark.parametrize("compression", [None, "gzip"])
def test_distributed_pickling(compression):
    data = test_data["int_data"]
    df = pd.DataFrame(data)

    if compression:
        filename_pattern = "test_to_pickle*.pkl.gz"
    else:
        filename_pattern = "test_to_pickle*.pkl"

    df.to_pickle(filename_pattern, compression=compression)

    pickle_files = glob.glob(filename_pattern)
    pickled_df = pd.read_pickle(pickle_files, compression=compression)
    df_equals(pickled_df, df)

    # clean up
    for pickle_file in pickle_files:
        os.remove(pickle_file)
