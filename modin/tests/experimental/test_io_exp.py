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

import contextlib
import json
from pathlib import Path

import numpy as np
import pandas
import pytest
from pandas._testing import ensure_clean

import modin.experimental.pandas as pd
from modin.config import AsyncReadMode, Engine
from modin.tests.pandas.utils import (
    df_equals,
    eval_general,
    parse_dates_values_by_id,
    test_data,
    time_parsing_csv_path,
)
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import try_cast_to_pandas


@pytest.mark.skipif(
    Engine.get() not in ("Ray", "Unidist", "Dask"),
    reason=f"{Engine.get()} does not have experimental API",
)
def test_from_sql_distributed(tmp_path, make_sql_connection):
    filename = "test_from_sql_distributed.db"
    table = "test_from_sql_distributed"
    conn = make_sql_connection(str(tmp_path / filename), table)
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
    conn = make_sql_connection(str(tmp_path / filename), table)
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
        with pytest.raises(FileNotFoundError):
            with warns_that_defaulting_to_pandas():
                pd.read_csv_glob(
                    "s3://dask-data/nyc-taxi/2015/yellow_tripdata_2015-",
                    storage_options={"anon": True},
                )

    def test_read_csv_glob_4373(self, tmp_path):
        columns, filename = ["col0"], str(tmp_path / "1x1.csv")
        df = pd.DataFrame([[1]], columns=columns)
        with (
            warns_that_defaulting_to_pandas()
            if Engine.get() == "Dask"
            else contextlib.nullcontext()
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
                try_cast_to_pandas(modin_df)  # force materialization
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
        "s3://modin-test/modin-bugs/multiple_csv/test_data*.csv",
    ],
)
def test_read_multiple_csv_cloud_store(path, s3_resource, s3_storage_options):
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
        lambda module, **kwargs: (
            pd.read_csv_glob(path, **kwargs).reset_index(drop=True)
            if hasattr(module, "read_csv_glob")
            else _pandas_read_csv_glob(path, **kwargs)
        ),
        storage_options=s3_storage_options,
    )


@pytest.mark.skipif(
    Engine.get() not in ("Ray", "Unidist", "Dask"),
    reason=f"{Engine.get()} does not have experimental API",
)
@pytest.mark.parametrize(
    "storage_options_extra",
    [{"anon": False}, {"anon": True}, {"key": "123", "secret": "123"}],
)
def test_read_multiple_csv_s3_storage_opts(
    s3_resource, s3_storage_options, storage_options_extra
):
    s3_path = "s3://modin-test/modin-bugs/multiple_csv/"

    def _pandas_read_csv_glob(path, storage_options):
        pandas_df = pandas.concat(
            [
                pandas.read_csv(
                    f"{s3_path}test_data{i}.csv",
                    storage_options=storage_options,
                )
                for i in range(2)
            ],
        ).reset_index(drop=True)
        return pandas_df

    expected_exception = None
    if "anon" in storage_options_extra:
        expected_exception = PermissionError("Forbidden")
    eval_general(
        pd,
        pandas,
        lambda module, **kwargs: (
            pd.read_csv_glob(s3_path, **kwargs)
            if hasattr(module, "read_csv_glob")
            else _pandas_read_csv_glob(s3_path, **kwargs)
        ),
        storage_options=s3_storage_options | storage_options_extra,
        expected_exception=expected_exception,
    )


@pytest.mark.skipif(
    Engine.get() not in ("Ray", "Unidist", "Dask"),
    reason=f"{Engine.get()} does not have experimental API",
)
@pytest.mark.parametrize("pathlike", [False, True])
@pytest.mark.parametrize("compression", [None, "gzip"])
@pytest.mark.parametrize(
    "filename", ["test_default_to_pickle.pkl", "test_to_pickle*.pkl"]
)
@pytest.mark.parametrize("read_func", ["read_pickle_glob"])
@pytest.mark.parametrize("to_func", ["to_pickle_glob"])
def test_distributed_pickling(
    tmp_path, filename, compression, pathlike, read_func, to_func
):
    data = test_data["int_data"]
    df = pd.DataFrame(data)

    filename_param = filename
    if compression:
        filename = f"{filename}.gz"

    filename = Path(filename) if pathlike else filename

    with (
        warns_that_defaulting_to_pandas()
        if filename_param == "test_default_to_pickle.pkl"
        else contextlib.nullcontext()
    ):
        getattr(df.modin, to_func)(str(tmp_path / filename), compression=compression)
        pickled_df = getattr(pd, read_func)(
            str(tmp_path / filename), compression=compression
        )
    df_equals(pickled_df, df)


@pytest.mark.skipif(
    Engine.get() not in ("Ray", "Unidist", "Dask"),
    reason=f"{Engine.get()} does not have experimental API",
)
@pytest.mark.parametrize(
    "filename",
    ["test_parquet_glob.parquet", "test_parquet_glob*.parquet"],
)
def test_parquet_glob(tmp_path, filename):
    data = test_data["int_data"]
    df = pd.DataFrame(data)

    filename_param = filename

    with (
        warns_that_defaulting_to_pandas()
        if filename_param == "test_parquet_glob.parquet"
        else contextlib.nullcontext()
    ):
        df.modin.to_parquet_glob(str(tmp_path / filename))
        read_df = pd.read_parquet_glob(str(tmp_path / filename))
    df_equals(read_df, df)


@pytest.mark.skipif(
    Engine.get() not in ("Ray", "Unidist", "Dask"),
    reason=f"{Engine.get()} does not have experimental API",
)
@pytest.mark.parametrize(
    "filename",
    ["test_json_glob.json", "test_json_glob*.json"],
)
def test_json_glob(tmp_path, filename):
    data = test_data["int_data"]
    df = pd.DataFrame(data)

    filename_param = filename

    with (
        warns_that_defaulting_to_pandas()
        if filename_param == "test_json_glob.json"
        else contextlib.nullcontext()
    ):
        df.modin.to_json_glob(str(tmp_path / filename))
        read_df = pd.read_json_glob(str(tmp_path / filename))
    df_equals(read_df, df)


@pytest.mark.skipif(
    Engine.get() not in ("Ray", "Unidist", "Dask"),
    reason=f"{Engine.get()} does not have experimental API",
)
@pytest.mark.parametrize(
    "filename",
    ["test_xml_glob.xml", "test_xml_glob*.xml"],
)
def test_xml_glob(tmp_path, filename):
    data = test_data["int_data"]
    df = pd.DataFrame(data)

    filename_param = filename

    with (
        warns_that_defaulting_to_pandas()
        if filename_param == "test_xml_glob.xml"
        else contextlib.nullcontext()
    ):
        df.modin.to_xml_glob(str(tmp_path / filename), index=False)
        read_df = pd.read_xml_glob(str(tmp_path / filename))

    # Index get messed up when concatting so we reset it.
    read_df = read_df.reset_index(drop=True)
    df_equals(read_df, df)


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
