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
import json
import numpy as np
import pandas
import pytest
import modin.experimental.pandas as pd
from modin.config import Engine
from modin.utils import get_current_execution
from modin.pandas.test.utils import (
    df_equals,
    get_unique_filename,
    teardown_test_files,
    test_data,
)
from modin.test.test_utils import warns_that_defaulting_to_pandas
from modin.pandas.test.utils import parse_dates_values_by_id, time_parsing_csv_path


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
                pd.read_csv_glob("s3://dask-data/nyc-taxi/2015/yellow_tripdata_2015-")

    def test_read_csv_glob_4373(self):
        columns, filename = ["col0"], "1x1.csv"
        pd.DataFrame([[1]], columns=columns).to_csv(filename)

        kwargs = {"filepath_or_buffer": filename, "usecols": columns}
        modin_df = pd.read_csv_glob(**kwargs)
        pandas_df = pandas.read_csv(**kwargs)
        df_equals(modin_df, pandas_df)

    @pytest.mark.parametrize(
        "parse_dates",
        [pytest.param(value, id=id) for id, value in parse_dates_values_by_id.items()],
    )
    def test_read_single_csv_with_parse_dates(self, parse_dates):
        try:
            pandas_df = pandas.read_csv(time_parsing_csv_path, parse_dates=parse_dates)
        except Exception as pandas_exception:
            with pytest.raises(Exception) as modin_exception:
                modin_df = pd.read_csv_glob(
                    time_parsing_csv_path, parse_dates=parse_dates
                )
                # Call __repr__ on the modin df to force it to materialize.
                repr(modin_df)
            assert isinstance(
                modin_exception.value, type(pandas_exception)
            ), "Got Modin Exception type {}, but pandas Exception type {} was expected".format(
                type(modin_exception.value), type(pandas_exception)
            )
        else:
            modin_df = pd.read_csv_glob(time_parsing_csv_path, parse_dates=parse_dates)
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


@pytest.mark.skipif(
    not Engine.get() == "Ray",
    reason=f"{Engine.get()} does not have experimental read_custom_text API",
)
def test_read_custom_json_text():
    filename = get_unique_filename(extension="json")

    def _generate_json(file_name, nrows, ncols):
        data = np.random.rand(nrows, ncols)
        df = pandas.DataFrame(data, columns=[f"col{x}" for x in range(ncols)])
        df.to_json(file_name, lines=True, orient="records")

    _generate_json(filename, 64, 8)

    # Custom parser allows us to add some specifics to reading files,
    # which is not available through the ready-made API.
    # For example, the parser allows us to reduce the amount of RAM
    # required for reading by selecting a subset of columns.
    def _custom_parser(io_input, **kwargs):
        result = {"col0": [], "col1": [], "col3": []}
        for line in io_input:
            # for example, simjson can be used here
            obj = json.loads(line)
            for key in result:
                result[key].append(obj[key])
        return pandas.DataFrame(result).rename(columns={"col0": "testID"})

    df1 = pd.read_custom_text(
        filename,
        columns=["testID", "col1", "col3"],
        custom_parser=_custom_parser,
        is_quoting=False,
    )
    df2 = pd.read_json(filename, lines=True)[["col0", "col1", "col3"]].rename(
        columns={"col0": "testID"}
    )
    df_equals(df1, df2)


@pytest.mark.skipif(
    not Engine.get() == "Ray",
    reason=f"{Engine.get()} does not have experimental API",
)
def test_read_evaluated_dict():
    filename = get_unique_filename(extension="json")

    def _generate_evaluated_dict(file_name, nrows, ncols):
        result = {}
        keys = [f"col{x}" for x in range(ncols)]

        with open(file_name, mode="w") as _file:
            for i in range(nrows):
                data = np.random.rand(ncols)
                for idx, key in enumerate(keys):
                    result[key] = data[idx]
                _file.write(str(result))
                _file.write("\n")

    _generate_evaluated_dict(filename, 64, 8)

    # This parser allows us to read a format not supported by other reading functions
    def _custom_parser(io_input, **kwargs):
        cat_list = []
        asin_list = []
        for line in io_input:
            obj = eval(line)
            cat_list.append(obj["col1"])
            asin_list.append(obj["col2"])
        return pandas.DataFrame({"col1": asin_list, "col2": cat_list})

    df1 = pd.read_custom_text(
        filename,
        columns=["col1", "col2"],
        custom_parser=_custom_parser,
    )
    assert df1.shape == (64, 2)

    def columns_callback(io_input, **kwargs):
        columns = None
        for line in io_input:
            columns = list(eval(line).keys())[1:3]
            break
        return columns

    df2 = pd.read_custom_text(
        filename, columns=columns_callback, custom_parser=_custom_parser
    )
    df_equals(df1, df2)
