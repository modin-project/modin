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
from pandas._testing import ensure_clean
import pytest

import modin.experimental.pandas as pd
from modin.config import Engine, AsyncReadMode
from modin.pandas.test.utils import (
    df_equals,
    teardown_test_files,
    test_data,
    eval_general,
)
from modin.test.test_utils import warns_that_defaulting_to_pandas
from modin.pandas.test.utils import parse_dates_values_by_id, time_parsing_csv_path


@pytest.mark.skipif(
    Engine.get() not in ("Ray", "Unidist", "Dask"),
    reason=f"{Engine.get()} does not have experimental API",
)
def test_from_sql_distributed(tmp_path, make_sql_connection):
    filename = "test_from_sql_distributed.db"
    table = "test_from_sql_distributed"
    conn = make_sql_connection(tmp_path / filename, table)
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
    Engine.get() not in ("Ray", "Unidist", "Dask"),
    reason=f"{Engine.get()} does not have experimental API",
)
def test_from_sql_defaults(tmp_path, make_sql_connection):
    filename = "test_from_sql_distributed.db"
    table = "test_from_sql_distributed"
    conn = make_sql_connection(tmp_path / filename, table)
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
    Engine.get() not in ("Ray", "Unidist", "Dask"),
    reason=f"{Engine.get()} does not have experimental glob API",
)
class TestCsvGlob:
    def test_read_multiple_small_csv(self):
        pandas_df = pandas.concat([pandas.read_csv(fname) for fname in pytest.files])
        modin_df = pd.read_csv_glob(pytest.glob_path)

        # Indexes get messed up when concatting so we reset both.
        pandas_df = pandas_df.reset_index(drop=True)
        modin_df = modin_df.reset_index(drop=True)

        df_equals(modin_df, pandas_df)

    @pytest.mark.parametrize("nrows", [35, 100])
    def test_read_multiple_csv_nrows(self, request, nrows):
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
                pd.read_csv_glob(
                    "s3://dask-data/nyc-taxi/2015/yellow_tripdata_2015-",
                    storage_options={"anon": True},
                )

    def test_read_csv_glob_4373(self):
        columns, filename = ["col0"], "1x1.csv"
        df = pd.DataFrame([[1]], columns=columns)
        with (
            warns_that_defaulting_to_pandas()
            if Engine.get() == "Dask"
            else nullcontext()
        ):
            df.to_csv(filename)

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
    Engine.get() not in ("Ray", "Unidist", "Dask"),
    reason=f"{Engine.get()} does not have experimental glob API",
)
@pytest.mark.parametrize(
    "path",
    [
        "s3://modin-datasets/testing/multiple_csv/test_data*.csv",
        "gs://modin-testing/testing/multiple_csv/test_data*.csv",
    ],
)
def test_read_multiple_csv_cloud_store(path):
    def _pandas_read_csv_glob(path, storage_options):
        pandas_dfs = [
            pandas.read_csv(
                f"{path.lower().split('*')[0]}{i}.csv", storage_options=storage_options
            )
            for i in range(2)
        ]
        return pandas.concat(pandas_dfs).reset_index(drop=True)

    eval_general(
        pd,
        pandas,
        lambda module, **kwargs: pd.read_csv_glob(path, **kwargs).reset_index(drop=True)
        if hasattr(module, "read_csv_glob")
        else _pandas_read_csv_glob(path, **kwargs),
        storage_options={"anon": True},
    )


test_default_to_pickle_filename = "test_default_to_pickle.pkl"


@pytest.mark.skipif(
    Engine.get() not in ("Ray", "Unidist", "Dask"),
    reason=f"{Engine.get()} does not have experimental API",
)
@pytest.mark.parametrize(
    "storage_options",
    [{"anon": False}, {"anon": True}, {"key": "123", "secret": "123"}, None],
)
def test_read_multiple_csv_s3_storage_opts(storage_options):
    path = "s3://modin-datasets/testing/multiple_csv/"

    def _pandas_read_csv_glob(path, storage_options):
        pandas_df = pandas.concat(
            [
                pandas.read_csv(
                    f"{path}test_data{i}.csv",
                    storage_options=storage_options,
                )
                for i in range(2)
            ],
        ).reset_index(drop=True)
        return pandas_df

    eval_general(
        pd,
        pandas,
        lambda module, **kwargs: pd.read_csv_glob(path, **kwargs)
        if hasattr(module, "read_csv_glob")
        else _pandas_read_csv_glob(path, **kwargs),
        storage_options=storage_options,
    )


@pytest.mark.skipif(
    Engine.get() not in ("Ray", "Unidist", "Dask"),
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
    Engine.get() not in ("Ray", "Unidist", "Dask"),
    reason=f"{Engine.get()} does not have experimental read_custom_text API",
)
@pytest.mark.parametrize("set_async_read_mode", [False, True], indirect=True)
def test_read_custom_json_text(set_async_read_mode):
    def _generate_json(file_name, nrows, ncols):
        data = np.random.rand(nrows, ncols)
        df = pandas.DataFrame(data, columns=[f"col{x}" for x in range(ncols)])
        df.to_json(file_name, lines=True, orient="records")

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

    with ensure_clean() as filename:
        _generate_json(filename, 64, 8)

        df1 = pd.read_custom_text(
            filename,
            columns=["testID", "col1", "col3"],
            custom_parser=_custom_parser,
            is_quoting=False,
        )
        df2 = pd.read_json(filename, lines=True)[["col0", "col1", "col3"]].rename(
            columns={"col0": "testID"}
        )
        if AsyncReadMode.get():
            # If read operations are asynchronous, then the dataframes
            # check should be inside `ensure_clean` context
            # because the file may be deleted before actual reading starts
            df_equals(df1, df2)
    if not AsyncReadMode.get():
        df_equals(df1, df2)


@pytest.mark.skipif(
    Engine.get() not in ("Ray", "Unidist", "Dask"),
    reason=f"{Engine.get()} does not have experimental API",
)
@pytest.mark.parametrize("set_async_read_mode", [False, True], indirect=True)
def test_read_evaluated_dict(set_async_read_mode):
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

    # This parser allows us to read a format not supported by other reading functions
    def _custom_parser(io_input, **kwargs):
        cat_list = []
        asin_list = []
        for line in io_input:
            obj = eval(line)
            cat_list.append(obj["col1"])
            asin_list.append(obj["col2"])
        return pandas.DataFrame({"col1": asin_list, "col2": cat_list})

    def columns_callback(io_input, **kwargs):
        columns = None
        for line in io_input:
            columns = list(eval(line).keys())[1:3]
            break
        return columns

    with ensure_clean() as filename:
        _generate_evaluated_dict(filename, 64, 8)

        df1 = pd.read_custom_text(
            filename,
            columns=["col1", "col2"],
            custom_parser=_custom_parser,
        )
        assert df1.shape == (64, 2)

        df2 = pd.read_custom_text(
            filename, columns=columns_callback, custom_parser=_custom_parser
        )
        if AsyncReadMode.get():
            # If read operations are asynchronous, then the dataframes
            # check should be inside `ensure_clean` context
            # because the file may be deleted before actual reading starts
            df_equals(df1, df2)
    if not AsyncReadMode.get():
        df_equals(df1, df2)
