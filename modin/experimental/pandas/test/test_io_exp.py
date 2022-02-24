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

from contextlib import nullcontext
import glob
import pandas
import pytest
import modin.experimental.pandas as pd
from modin.config import Engine
from modin.utils import get_current_execution
from modin.pandas.test.utils import df_equals, teardown_test_files, test_data
from modin.test.test_utils import warns_that_defaulting_to_pandas


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

    def test_read_csv_without_glob(self):
        with pytest.warns(UserWarning, match=r"Shell-style wildcard"):
            with pytest.raises(FileNotFoundError):
                pd.read_csv_glob("s3://nyc-tlc/trip data/yellow_tripdata_2020-")


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


test_default_to_pickle_filename = "test_default_to_pickle.pkl"


@pytest.mark.skipif(
    get_current_execution() != "ExperimentalPandasOnRay",
    reason=f"Execution {get_current_execution()} isn't supported.",
)
@pytest.mark.parametrize(
    "storage_options",
    [{"anon": False}, {"anon": True}, {"key": "123", "secret": "123"}, None],
)
def test_read_multiple_csv_s3_storage_opts(storage_options):
    path = "s3://modin-datasets/testing/multiple_csv/"
    # Test the fact of handling of `storage_options`
    modin_df = pd.read_csv_glob(path, storage_options=storage_options)
    pandas_df = pd.concat(
        [
            pandas.read_csv(
                f"{path}test_data{i}.csv",
                storage_options=storage_options,
            )
            for i in range(2)
        ],
    ).reset_index(drop=True)

    df_equals(modin_df, pandas_df)


@pytest.mark.skipif(
    not Engine.get() == "Ray",
    reason=f"{Engine.get()} does not have experimental API",
)
@pytest.mark.parametrize("compression", [None, "gzip"])
@pytest.mark.parametrize(
    "filename", [test_default_to_pickle_filename, "test_to_pickle*.pkl"]
)
def test_distributed_pickling(filename, compression):
    data = test_data["int_data"]
    df = pd.DataFrame(data)

    filename_param = filename
    if compression:
        filename = f"{filename}.gz"

    with (
        warns_that_defaulting_to_pandas()
        if filename_param == test_default_to_pickle_filename
        else nullcontext()
    ):
        df.to_pickle_distributed(filename, compression=compression)
        pickled_df = pd.read_pickle_distributed(filename, compression=compression)
    df_equals(pickled_df, df)

    pickle_files = glob.glob(filename)
    teardown_test_files(pickle_files)
