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

import unittest.mock as mock
import inspect
import contextlib
import pytest
import numpy as np
from packaging import version
import pandas
from pandas.errors import ParserWarning
import pandas._libs.lib as lib
from pandas._testing import ensure_clean
from pathlib import Path
from collections import OrderedDict, defaultdict
from modin.config.envvars import MinPartitionSize
from modin.db_conn import (
    ModinDatabaseConnection,
    UnsupportedDatabaseException,
)
from modin.config import (
    TestDatasetSize,
    Engine,
    StorageFormat,
    IsExperimental,
    TestReadFromPostgres,
    TestReadFromSqlServer,
    ReadSqlEngine,
    AsyncReadMode,
)
from modin.utils import to_pandas
from modin.pandas.utils import from_arrow
from modin.test.test_utils import warns_that_defaulting_to_pandas
import pyarrow as pa
import os
from scipy import sparse
import sys
import sqlalchemy as sa
import csv
from typing import Dict

from .utils import (
    check_file_leaks,
    df_equals,
    json_short_string,
    json_short_bytes,
    json_long_string,
    json_long_bytes,
    get_unique_filename,
    io_ops_bad_exc,
    eval_io_from_str,
    dummy_decorator,
    create_test_dfs,
    COMP_TO_EXT,
    generate_dataframe,
    default_to_pandas_ignore_string,
    parse_dates_values_by_id,
    time_parsing_csv_path,
    test_data as utils_test_data,
    eval_general,
)

if StorageFormat.get() == "Hdk":
    from modin.experimental.core.execution.native.implementations.hdk_on_native.test.utils import (
        eval_io,
        align_datetime_dtypes,
    )
else:
    from .utils import eval_io

if StorageFormat.get() == "Pandas":
    import modin.pandas as pd
else:
    import modin.experimental.pandas as pd

try:
    import ray

    EXCEPTIONS = (ray.exceptions.WorkerCrashedError,)
except ImportError:
    EXCEPTIONS = ()


from modin.config import NPartitions

NPartitions.put(4)

DATASET_SIZE_DICT = {
    "Small": 64,
    "Normal": 2000,
    "Big": 20000,
}

# Number of rows in the test file
NROWS = DATASET_SIZE_DICT.get(TestDatasetSize.get(), DATASET_SIZE_DICT["Small"])

TEST_DATA = {
    "col1": [0, 1, 2, 3],
    "col2": [4, 5, 6, 7],
    "col3": [8, 9, 10, 11],
    "col4": [12, 13, 14, 15],
    "col5": [0, 0, 0, 0],
}


@contextlib.contextmanager
def _nullcontext():
    """Replacement for contextlib.nullcontext missing in older Python."""
    yield


def assert_files_eq(path1, path2):
    with open(path1, "rb") as file1, open(path2, "rb") as file2:
        file1_content = file1.read()
        file2_content = file2.read()

        if file1_content == file2_content:
            return True
        else:
            return False


def setup_clipboard(row_size=NROWS):
    df = pandas.DataFrame({"col1": np.arange(row_size), "col2": np.arange(row_size)})
    df.to_clipboard()


def parquet_eval_to_file(tmp_dir, modin_obj, pandas_obj, fn, extension, **fn_kwargs):
    """
    Helper function to test `to_parquet` method.

    Parameters
    ----------
    tmp_dir : Union[str, Path]
        Temporary directory.
    modin_obj : pd.DataFrame
        A Modin DataFrame or a Series to test `to_parquet` method.
    pandas_obj: pandas.DataFrame
        A pandas DataFrame or a Series to test `to_parquet` method.
    fn : str
        Name of the method, that should be tested.
    extension : str
        Extension of the test file.
    """
    unique_filename_modin = get_unique_filename(extension=extension, data_dir=tmp_dir)
    unique_filename_pandas = get_unique_filename(extension=extension, data_dir=tmp_dir)

    engine = fn_kwargs.get("engine", "auto")

    getattr(modin_obj, fn)(unique_filename_modin, **fn_kwargs)
    getattr(pandas_obj, fn)(unique_filename_pandas, **fn_kwargs)

    pandas_df = pandas.read_parquet(unique_filename_pandas, engine=engine)
    modin_df = pd.read_parquet(unique_filename_modin, engine=engine)
    df_equals(pandas_df, modin_df)


def eval_to_file(tmp_dir, modin_obj, pandas_obj, fn, extension, **fn_kwargs):
    """
    Test `fn` method of `modin_obj` and `pandas_obj`.

    Parameters
    ----------
    tmp_dir : Union[str, Path]
        Temporary directory.
    modin_obj: Modin DataFrame or Series
        Object to test.
    pandas_obj: Pandas DataFrame or Series
        Object to test.
    fn: str
        Name of the method, that should be tested.
    extension: str
        Extension of the test file.
    """
    unique_filename_modin = get_unique_filename(extension=extension, data_dir=tmp_dir)
    unique_filename_pandas = get_unique_filename(extension=extension, data_dir=tmp_dir)

    # parameter `max_retries=0` is set for `to_csv` function on Ray engine,
    # in order to increase the stability of tests, we repeat the call of
    # the entire function manually
    last_exception = None
    for _ in range(3):
        try:
            getattr(modin_obj, fn)(unique_filename_modin, **fn_kwargs)
        except EXCEPTIONS as err:
            last_exception = err
            continue
        break
    # If we do have an exception that's valid let's raise it
    if last_exception:
        raise last_exception

    getattr(pandas_obj, fn)(unique_filename_pandas, **fn_kwargs)

    assert assert_files_eq(unique_filename_modin, unique_filename_pandas)


def eval_to_csv_file(tmp_dir, modin_obj, pandas_obj, extension, **kwargs):
    if extension is None:
        kwargs["mode"] = "t"
        kwargs["compression"] = "infer"
        modin_csv = modin_obj.to_csv(**kwargs)
        pandas_csv = pandas_obj.to_csv(**kwargs)
        if modin_csv == pandas_csv:
            return

        force_read = True
        modin_file = get_unique_filename(extension="csv", data_dir=tmp_dir)
        pandas_file = get_unique_filename(extension="csv", data_dir=tmp_dir)
        with open(modin_file, "w") as file:
            file.write(modin_csv)
        with open(pandas_file, "w") as file:
            file.write(pandas_csv)
    else:
        force_read = extension != "csv" or kwargs.get("compression", None)
        modin_file = get_unique_filename(extension=extension, data_dir=tmp_dir)
        pandas_file = get_unique_filename(extension=extension, data_dir=tmp_dir)
        modin_obj.to_csv(modin_file, **kwargs)
        pandas_obj.to_csv(pandas_file, **kwargs)

    if force_read or not assert_files_eq(modin_file, pandas_file):
        # If the files are not identical, make sure they can
        # be read by pandas and contains identical data.
        read_kwargs = {}
        if kwargs.get("index", None) is not False:
            read_kwargs["index_col"] = 0
        if (value := kwargs.get("sep", None)) is not None:
            read_kwargs["sep"] = value
        if (value := kwargs.get("compression", None)) is not None:
            read_kwargs["compression"] = value
        modin_obj = pandas.read_csv(modin_file, **read_kwargs)
        pandas_obj = pandas.read_csv(pandas_file, **read_kwargs)
        df_equals(pandas_obj, modin_obj)


@pytest.fixture
def make_parquet_dir(tmp_path):
    def _make_parquet_dir(
        dfs_by_filename: Dict[str, pandas.DataFrame], row_group_size: int
    ):
        for filename, df in dfs_by_filename.items():
            df.to_parquet(
                os.path.join(tmp_path, filename), row_group_size=row_group_size
            )
        return tmp_path

    yield _make_parquet_dir


@pytest.mark.usefixtures("TestReadCSVFixture")
@pytest.mark.skipif(
    IsExperimental.get() and StorageFormat.get() == "Pyarrow",
    reason="Segmentation fault; see PR #2347 ffor details",
)
@pytest.mark.filterwarnings(default_to_pandas_ignore_string)
class TestCsv:
    # delimiter tests
    @pytest.mark.parametrize("sep", [None, "_", ",", ".", "\n"])
    @pytest.mark.parametrize("delimiter", ["_", ",", ".", "\n"])
    @pytest.mark.parametrize("decimal", [".", "_"])
    @pytest.mark.parametrize("thousands", [None, ",", "_", " "])
    def test_read_csv_delimiters(
        self, make_csv_file, sep, delimiter, decimal, thousands
    ):
        with ensure_clean(".csv") as unique_filename:
            make_csv_file(
                filename=unique_filename,
                delimiter=delimiter,
                thousands_separator=thousands,
                decimal_separator=decimal,
            )

            eval_io(
                fn_name="read_csv",
                # read_csv kwargs
                filepath_or_buffer=unique_filename,
                delimiter=delimiter,
                sep=sep,
                decimal=decimal,
                thousands=thousands,
            )

    @pytest.mark.parametrize(
        "dtype_backend", [lib.no_default, "numpy_nullable", "pyarrow"]
    )
    def test_read_csv_dtype_backend(self, make_csv_file, dtype_backend):
        with ensure_clean(".csv") as unique_filename:
            make_csv_file(filename=unique_filename)

            def comparator(df1, df2):
                df_equals(df1, df2)
                df_equals(df1.dtypes, df2.dtypes)

            eval_io(
                fn_name="read_csv",
                # read_csv kwargs
                filepath_or_buffer=unique_filename,
                dtype_backend=dtype_backend,
                comparator=comparator,
            )

    # Column and Index Locations and Names tests
    @pytest.mark.parametrize("header", ["infer", None, 0])
    @pytest.mark.parametrize("index_col", [None, "col1"])
    @pytest.mark.parametrize(
        "names", [lib.no_default, ["col1"], ["c1", "c2", "c3", "c4", "c5", "c6", "c7"]]
    )
    @pytest.mark.parametrize(
        "usecols", [None, ["col1"], ["col1", "col2", "col6"], [0, 1, 5]]
    )
    @pytest.mark.parametrize("skip_blank_lines", [True, False])
    def test_read_csv_col_handling(
        self,
        header,
        index_col,
        names,
        usecols,
        skip_blank_lines,
    ):
        if names is lib.no_default:
            pytest.skip("some parameters combiantions fails: issue #2312")
        if header in ["infer", None] and names is not lib.no_default:
            pytest.skip(
                "Heterogeneous data in a column is not cast to a common type: issue #3346"
            )
        eval_io(
            fn_name="read_csv",
            # read_csv kwargs
            filepath_or_buffer=pytest.csvs_names["test_read_csv_blank_lines"],
            header=header,
            index_col=index_col,
            names=names,
            usecols=usecols,
            skip_blank_lines=skip_blank_lines,
        )

    @pytest.mark.parametrize("usecols", [lambda col_name: col_name in ["a", "b", "e"]])
    def test_from_csv_with_callable_usecols(self, usecols):
        fname = "modin/pandas/test/data/test_usecols.csv"
        pandas_df = pandas.read_csv(fname, usecols=usecols)
        modin_df = pd.read_csv(fname, usecols=usecols)
        df_equals(modin_df, pandas_df)

    # General Parsing Configuration
    @pytest.mark.parametrize("dtype", [None, True])
    @pytest.mark.parametrize("engine", [None, "python", "c"])
    @pytest.mark.parametrize(
        "converters",
        [
            None,
            {
                "col1": lambda x: np.int64(x) * 10,
                "col2": pandas.to_datetime,
                "col4": lambda x: x.replace(":", ";"),
            },
        ],
    )
    @pytest.mark.parametrize("skipfooter", [0, 10])
    def test_read_csv_parsing_1(
        self,
        dtype,
        engine,
        converters,
        skipfooter,
    ):
        if dtype:
            dtype = {
                col: "object"
                for col in pandas.read_csv(
                    pytest.csvs_names["test_read_csv_regular"], nrows=1
                ).columns
            }

        eval_io(
            fn_name="read_csv",
            raising_exceptions=None,
            check_kwargs_callable=not callable(converters),
            # read_csv kwargs
            filepath_or_buffer=pytest.csvs_names["test_read_csv_regular"],
            dtype=dtype,
            engine=engine,
            converters=converters,
            skipfooter=skipfooter,
        )

    @pytest.mark.parametrize("header", ["infer", None, 0])
    @pytest.mark.parametrize(
        "skiprows",
        [
            2,
            lambda x: x % 2,
            lambda x: x > 25,
            lambda x: x > 128,
            np.arange(10, 50),
            np.arange(10, 50, 2),
        ],
    )
    @pytest.mark.parametrize("nrows", [35, None])
    @pytest.mark.parametrize(
        "names",
        [
            [f"c{col_number}" for col_number in range(4)],
            [f"c{col_number}" for col_number in range(6)],
            None,
        ],
    )
    @pytest.mark.parametrize("encoding", ["latin1", "windows-1251", None])
    def test_read_csv_parsing_2(
        self,
        make_csv_file,
        request,
        header,
        skiprows,
        nrows,
        names,
        encoding,
    ):
        with ensure_clean(".csv") as unique_filename:
            if encoding:
                make_csv_file(
                    filename=unique_filename,
                    encoding=encoding,
                )
            kwargs = {
                "filepath_or_buffer": unique_filename
                if encoding
                else pytest.csvs_names["test_read_csv_regular"],
                "header": header,
                "skiprows": skiprows,
                "nrows": nrows,
                "names": names,
                "encoding": encoding,
            }

            if Engine.get() != "Python":
                df = pandas.read_csv(**dict(kwargs, nrows=1))
                # in that case first partition will contain str
                if df[df.columns[0]][df.index[0]] in ["c1", "col1", "c3", "col3"]:
                    pytest.xfail(
                        "read_csv incorrect output with float data - issue #2634"
                    )

            eval_io(
                fn_name="read_csv",
                raising_exceptions=None,
                check_kwargs_callable=not callable(skiprows),
                # read_csv kwargs
                **kwargs,
            )

    @pytest.mark.parametrize("true_values", [["Yes"], ["Yes", "true"], None])
    @pytest.mark.parametrize("false_values", [["No"], ["No", "false"], None])
    @pytest.mark.parametrize("skipfooter", [0, 10])
    @pytest.mark.parametrize("nrows", [35, None])
    def test_read_csv_parsing_3(
        self,
        true_values,
        false_values,
        skipfooter,
        nrows,
    ):
        xfail_case = (
            (false_values or true_values)
            and Engine.get() != "Python"
            and StorageFormat.get() != "Hdk"
        )
        if xfail_case:
            pytest.xfail("modin and pandas dataframes differs - issue #2446")

        eval_io(
            fn_name="read_csv",
            raising_exceptions=None,
            # read_csv kwargs
            filepath_or_buffer=pytest.csvs_names["test_read_csv_yes_no"],
            true_values=true_values,
            false_values=false_values,
            skipfooter=skipfooter,
            nrows=nrows,
        )

    def test_read_csv_skipinitialspace(self):
        with ensure_clean(".csv") as unique_filename:
            str_initial_spaces = (
                "col1,col2,col3,col4\n"
                + "five,  six,  seven,  eight\n"
                + "    five,    six,    seven,    eight\n"
                + "five, six,  seven,   eight\n"
            )

            eval_io_from_str(str_initial_spaces, unique_filename, skipinitialspace=True)

    # NA and Missing Data Handling tests
    @pytest.mark.parametrize("na_values", ["custom_nan", "73"])
    @pytest.mark.parametrize("keep_default_na", [True, False])
    @pytest.mark.parametrize("na_filter", [True, False])
    @pytest.mark.parametrize("verbose", [True, False])
    @pytest.mark.parametrize("skip_blank_lines", [True, False])
    def test_read_csv_nans_handling(
        self,
        na_values,
        keep_default_na,
        na_filter,
        verbose,
        skip_blank_lines,
    ):
        eval_io(
            fn_name="read_csv",
            # read_csv kwargs
            filepath_or_buffer=pytest.csvs_names["test_read_csv_nans"],
            na_values=na_values,
            keep_default_na=keep_default_na,
            na_filter=na_filter,
            verbose=verbose,
            skip_blank_lines=skip_blank_lines,
        )

    # Datetime Handling tests
    @pytest.mark.parametrize(
        "parse_dates", [True, False, ["col2"], ["col2", "col4"], [1, 3]]
    )
    @pytest.mark.parametrize("infer_datetime_format", [True, False])
    @pytest.mark.parametrize("keep_date_col", [True, False])
    @pytest.mark.parametrize(
        "date_parser",
        [lib.no_default, lambda x: pandas.to_datetime(x, format="%Y-%m-%d")],
    )
    @pytest.mark.parametrize("dayfirst", [True, False])
    @pytest.mark.parametrize("cache_dates", [True, False])
    def test_read_csv_datetime(
        self,
        parse_dates,
        infer_datetime_format,
        keep_date_col,
        date_parser,
        dayfirst,
        cache_dates,
    ):
        raising_exceptions = io_ops_bad_exc  # default value
        if isinstance(parse_dates, dict) and callable(date_parser):
            # In this case raised TypeError: <lambda>() takes 1 positional argument but 2 were given
            raising_exceptions = list(io_ops_bad_exc)
            raising_exceptions.remove(TypeError)

        eval_io(
            fn_name="read_csv",
            check_kwargs_callable=not callable(date_parser),
            raising_exceptions=raising_exceptions,
            # read_csv kwargs
            filepath_or_buffer=pytest.csvs_names["test_read_csv_regular"],
            parse_dates=parse_dates,
            infer_datetime_format=infer_datetime_format,
            keep_date_col=keep_date_col,
            date_parser=date_parser,
            dayfirst=dayfirst,
            cache_dates=cache_dates,
        )

    @pytest.mark.parametrize("date", ["2023-01-01 00:00:01.000000000", "2023"])
    @pytest.mark.parametrize("dtype", [None, "str", {"id": "int64"}])
    @pytest.mark.parametrize("parse_dates", [None, [], ["date"], [1]])
    def test_read_csv_dtype_parse_dates(self, date, dtype, parse_dates):
        with ensure_clean(".csv") as filename:
            with open(filename, "w") as file:
                file.write(f"id,date\n1,{date}")
            eval_io(
                fn_name="read_csv",
                # read_csv kwargs
                filepath_or_buffer=filename,
                dtype=dtype,
                parse_dates=parse_dates,
            )

    # Iteration tests
    @pytest.mark.parametrize("iterator", [True, False])
    def test_read_csv_iteration(self, iterator):
        filename = pytest.csvs_names["test_read_csv_regular"]

        # Tests __next__ and correctness of reader as an iterator
        # Use larger chunksize to read through file quicker
        rdf_reader = pd.read_csv(filename, chunksize=500, iterator=iterator)
        pd_reader = pandas.read_csv(filename, chunksize=500, iterator=iterator)

        for modin_df, pd_df in zip(rdf_reader, pd_reader):
            df_equals(modin_df, pd_df)

        # Tests that get_chunk works correctly
        rdf_reader = pd.read_csv(filename, chunksize=1, iterator=iterator)
        pd_reader = pandas.read_csv(filename, chunksize=1, iterator=iterator)

        modin_df = rdf_reader.get_chunk(1)
        pd_df = pd_reader.get_chunk(1)

        df_equals(modin_df, pd_df)

        # Tests that read works correctly
        rdf_reader = pd.read_csv(filename, chunksize=1, iterator=iterator)
        pd_reader = pandas.read_csv(filename, chunksize=1, iterator=iterator)

        modin_df = rdf_reader.read()
        pd_df = pd_reader.read()

        df_equals(modin_df, pd_df)

    def test_read_csv_encoding_976(self):
        file_name = "modin/pandas/test/data/issue_976.csv"
        names = [str(i) for i in range(11)]

        kwargs = {
            "sep": ";",
            "names": names,
            "encoding": "windows-1251",
        }
        df1 = pd.read_csv(file_name, **kwargs)
        df2 = pandas.read_csv(file_name, **kwargs)
        # these columns contain data of various types in partitions
        # see #1931 for details;
        df1 = df1.drop(["4", "5"], axis=1)
        df2 = df2.drop(["4", "5"], axis=1)

        df_equals(df1, df2)

    # Quoting, Compression parameters tests
    @pytest.mark.parametrize("compression", ["infer", "gzip", "bz2", "xz", "zip"])
    @pytest.mark.parametrize("encoding", [None, "latin8", "utf16"])
    @pytest.mark.parametrize("engine", [None, "python", "c"])
    def test_read_csv_compression(self, make_csv_file, compression, encoding, engine):
        with ensure_clean(".csv") as unique_filename:
            make_csv_file(
                filename=unique_filename, encoding=encoding, compression=compression
            )
            compressed_file_path = (
                f"{unique_filename}.{COMP_TO_EXT[compression]}"
                if compression != "infer"
                else unique_filename
            )

            eval_io(
                fn_name="read_csv",
                # read_csv kwargs
                filepath_or_buffer=compressed_file_path,
                compression=compression,
                encoding=encoding,
                engine=engine,
            )

    @pytest.mark.parametrize(
        "encoding",
        [
            None,
            "ISO-8859-1",
            "latin1",
            "iso-8859-1",
            "cp1252",
            "utf8",
            pytest.param(
                "unicode_escape",
                marks=pytest.mark.skipif(
                    condition=sys.version_info < (3, 9),
                    reason="https://bugs.python.org/issue45461",
                ),
            ),
            "raw_unicode_escape",
            "utf_16_le",
            "utf_16_be",
            "utf32",
            "utf_32_le",
            "utf_32_be",
            "utf-8-sig",
        ],
    )
    def test_read_csv_encoding(self, make_csv_file, encoding):
        with ensure_clean(".csv") as unique_filename:
            make_csv_file(filename=unique_filename, encoding=encoding)

            eval_io(
                fn_name="read_csv",
                # read_csv kwargs
                filepath_or_buffer=unique_filename,
                encoding=encoding,
            )

    @pytest.mark.parametrize("thousands", [None, ",", "_", " "])
    @pytest.mark.parametrize("decimal", [".", "_"])
    @pytest.mark.parametrize("lineterminator", [None, "x", "\n"])
    @pytest.mark.parametrize("escapechar", [None, "d", "x"])
    @pytest.mark.parametrize("dialect", ["test_csv_dialect", "use_dialect_name", None])
    def test_read_csv_file_format(
        self,
        make_csv_file,
        thousands,
        decimal,
        lineterminator,
        escapechar,
        dialect,
    ):
        with ensure_clean(".csv") as unique_filename:
            if dialect:
                test_csv_dialect_params = {
                    "delimiter": "_",
                    "doublequote": False,
                    "escapechar": "\\",
                    "quotechar": "d",
                    "quoting": csv.QUOTE_ALL,
                }
                csv.register_dialect(dialect, **test_csv_dialect_params)
                if dialect != "use_dialect_name":
                    # otherwise try with dialect name instead of `_csv.Dialect` object
                    dialect = csv.get_dialect(dialect)
                make_csv_file(filename=unique_filename, **test_csv_dialect_params)
            else:
                make_csv_file(
                    filename=unique_filename,
                    thousands_separator=thousands,
                    decimal_separator=decimal,
                    escapechar=escapechar,
                    lineterminator=lineterminator,
                )

            if (
                (StorageFormat.get() == "Hdk")
                and (escapechar is not None)
                and (lineterminator is None)
                and (thousands is None)
                and (decimal == ".")
            ):
                with open(unique_filename, "r") as f:
                    if any(
                        line.find(f',"{escapechar}') != -1 for _, line in enumerate(f)
                    ):
                        pytest.xfail(
                            "Tests with this character sequence fail due to #5649"
                        )

            eval_io(
                raising_exceptions=None,
                fn_name="read_csv",
                # read_csv kwargs
                filepath_or_buffer=unique_filename,
                thousands=thousands,
                decimal=decimal,
                lineterminator=lineterminator,
                escapechar=escapechar,
                dialect=dialect,
            )

    @pytest.mark.parametrize(
        "quoting",
        [csv.QUOTE_ALL, csv.QUOTE_MINIMAL, csv.QUOTE_NONNUMERIC, csv.QUOTE_NONE],
    )
    @pytest.mark.parametrize("quotechar", ['"', "_", "d"])
    @pytest.mark.parametrize("doublequote", [True, False])
    @pytest.mark.parametrize("comment", [None, "#", "x"])
    def test_read_csv_quoting(
        self,
        make_csv_file,
        quoting,
        quotechar,
        doublequote,
        comment,
    ):
        # in these cases escapechar should be set, otherwise error occures
        # _csv.Error: need to escape, but no escapechar set"
        use_escapechar = (
            not doublequote and quotechar != '"' and quoting != csv.QUOTE_NONE
        )
        escapechar = "\\" if use_escapechar else None
        with ensure_clean(".csv") as unique_filename:
            make_csv_file(
                filename=unique_filename,
                quoting=quoting,
                quotechar=quotechar,
                doublequote=doublequote,
                escapechar=escapechar,
                comment_col_char=comment,
            )

            eval_io(
                fn_name="read_csv",
                # read_csv kwargs
                filepath_or_buffer=unique_filename,
                quoting=quoting,
                quotechar=quotechar,
                doublequote=doublequote,
                escapechar=escapechar,
                comment=comment,
            )

    # Error Handling parameters tests
    @pytest.mark.skip(reason="https://github.com/modin-project/modin/issues/6239")
    @pytest.mark.parametrize("on_bad_lines", ["error", "warn", "skip", None])
    def test_read_csv_error_handling(self, on_bad_lines):
        # in that case exceptions are raised both by Modin and pandas
        # and tests pass
        raise_exception_case = on_bad_lines is not None
        if (
            not raise_exception_case
            and Engine.get() not in ["Python"]
            and StorageFormat.get() != "Hdk"
        ):
            pytest.xfail("read_csv doesn't raise `bad lines` exceptions - issue #2500")
        eval_io(
            fn_name="read_csv",
            # read_csv kwargs
            filepath_or_buffer=pytest.csvs_names["test_read_csv_bad_lines"],
            on_bad_lines=on_bad_lines,
        )

    # Internal parameters tests
    @pytest.mark.parametrize("use_str_data", [True, False])
    @pytest.mark.parametrize("engine", [None, "python", "c"])
    @pytest.mark.parametrize("delimiter", [",", " "])
    @pytest.mark.parametrize("delim_whitespace", [True, False])
    @pytest.mark.parametrize("low_memory", [True, False])
    @pytest.mark.parametrize("memory_map", [True, False])
    @pytest.mark.parametrize("float_precision", [None, "high", "round_trip"])
    def test_read_csv_internal(
        self,
        make_csv_file,
        use_str_data,
        engine,
        delimiter,
        delim_whitespace,
        low_memory,
        memory_map,
        float_precision,
    ):
        # In this case raised TypeError: cannot use a string pattern on a bytes-like object,
        # so TypeError should be excluded from raising_exceptions list in order to check, that
        # the same exceptions are raised by Pandas and Modin
        case_with_TypeError_exc = (
            engine == "python"
            and delimiter == ","
            and delim_whitespace
            and low_memory
            and memory_map
            and float_precision is None
        )

        raising_exceptions = io_ops_bad_exc  # default value
        if case_with_TypeError_exc:
            raising_exceptions = list(io_ops_bad_exc)
            raising_exceptions.remove(TypeError)

        kwargs = {
            "engine": engine,
            "delimiter": delimiter,
            "delim_whitespace": delim_whitespace,
            "low_memory": low_memory,
            "memory_map": memory_map,
            "float_precision": float_precision,
        }

        with ensure_clean(".csv") as unique_filename:
            if use_str_data:
                str_delim_whitespaces = (
                    "col1 col2  col3   col4\n5 6   7  8\n9  10    11 12\n"
                )
                eval_io_from_str(
                    str_delim_whitespaces,
                    unique_filename,
                    raising_exceptions=raising_exceptions,
                    **kwargs,
                )
            else:
                make_csv_file(
                    filename=unique_filename,
                    delimiter=delimiter,
                )

                eval_io(
                    filepath_or_buffer=unique_filename,
                    fn_name="read_csv",
                    raising_exceptions=raising_exceptions,
                    **kwargs,
                )

    # Issue related, specific or corner cases
    @pytest.mark.parametrize("nrows", [2, None])
    def test_read_csv_bad_quotes(self, nrows):
        csv_bad_quotes = (
            '1, 2, 3, 4\none, two, three, four\nfive, "six", seven, "eight\n'
        )

        with ensure_clean(".csv") as unique_filename:
            eval_io_from_str(csv_bad_quotes, unique_filename, nrows=nrows)

    def test_read_csv_categories(self):
        eval_io(
            fn_name="read_csv",
            # read_csv kwargs
            filepath_or_buffer="modin/pandas/test/data/test_categories.csv",
            names=["one", "two"],
            dtype={"one": "int64", "two": "category"},
        )

    def test_read_csv_google_cloud_storage(self):
        eval_io(
            fn_name="read_csv",
            # read_csv kwargs
            filepath_or_buffer="gs://modin-testing/testing/multiple_csv/test_data0.csv",
        )

    @pytest.mark.parametrize("encoding", [None, "utf-8"])
    @pytest.mark.parametrize("encoding_errors", ["strict", "ignore"])
    @pytest.mark.parametrize(
        "parse_dates",
        [pytest.param(value, id=id) for id, value in parse_dates_values_by_id.items()],
    )
    @pytest.mark.parametrize("index_col", [None, 0, 5])
    @pytest.mark.parametrize("header", ["infer", 0])
    @pytest.mark.parametrize(
        "names",
        [
            None,
            [
                "timestamp",
                "year",
                "month",
                "date",
                "symbol",
                "high",
                "low",
                "open",
                "close",
                "spread",
                "volume",
            ],
        ],
    )
    @pytest.mark.exclude_in_sanity
    def test_read_csv_parse_dates(
        self, names, header, index_col, parse_dates, encoding, encoding_errors
    ):
        if names is not None and header == "infer":
            pytest.xfail(
                "read_csv with Ray engine works incorrectly with date data and names parameter provided - issue #2509"
            )

        eval_io(
            fn_name="read_csv",
            # read_csv kwargs
            filepath_or_buffer=time_parsing_csv_path,
            names=names,
            header=header,
            index_col=index_col,
            parse_dates=parse_dates,
            encoding=encoding,
            encoding_errors=encoding_errors,
        )

    @pytest.mark.parametrize(
        "storage_options",
        [{"anon": False}, {"anon": True}, {"key": "123", "secret": "123"}, None],
    )
    @pytest.mark.xfail(
        reason="S3 file gone missing, see https://github.com/modin-project/modin/issues/4875"
    )
    def test_read_csv_s3(self, storage_options):
        eval_io(
            fn_name="read_csv",
            # read_csv kwargs
            filepath_or_buffer="s3://noaa-ghcn-pds/csv/1788.csv",
            storage_options=storage_options,
        )

    def test_read_csv_s3_issue4658(self):
        eval_io(
            fn_name="read_csv",
            # read_csv kwargs
            filepath_or_buffer="s3://dask-data/nyc-taxi/2015/yellow_tripdata_2015-01.csv",
            nrows=10,
            storage_options={"anon": True},
        )

    @pytest.mark.parametrize("names", [list("XYZ"), None])
    @pytest.mark.parametrize("skiprows", [1, 2, 3, 4, None])
    def test_read_csv_skiprows_names(self, names, skiprows):
        eval_io(
            fn_name="read_csv",
            # read_csv kwargs
            filepath_or_buffer="modin/pandas/test/data/issue_2239.csv",
            names=names,
            skiprows=skiprows,
        )

    def _has_pandas_fallback_reason(self):
        # The Python engine does not use custom IO dispatchers, so specialized error messages
        # won't appear
        return Engine.get() != "Python" and StorageFormat.get() != "Hdk"

    def test_read_csv_default_to_pandas(self):
        if self._has_pandas_fallback_reason():
            warning_suffix = "buffers"
        else:
            warning_suffix = ""
        with warns_that_defaulting_to_pandas(suffix=warning_suffix):
            # This tests that we default to pandas on a buffer
            from io import StringIO

            with open(pytest.csvs_names["test_read_csv_regular"], "r") as _f:
                pd.read_csv(StringIO(_f.read()))

    def test_read_csv_url(self):
        eval_io(
            fn_name="read_csv",
            # read_csv kwargs
            filepath_or_buffer="https://raw.githubusercontent.com/modin-project/modin/master/modin/pandas/test/data/blah.csv",
            # It takes about ~17Gb of RAM for HDK to import the whole table from this test
            # because of too many (~1000) string columns in it. Taking a subset of columns
            # to be able to run this test on low-RAM machines.
            usecols=[0, 1, 2, 3] if StorageFormat.get() == "Hdk" else None,
        )

    @pytest.mark.parametrize("nrows", [21, 5, None])
    @pytest.mark.parametrize("skiprows", [4, 1, 500, None])
    def test_read_csv_newlines_in_quotes(self, nrows, skiprows):
        eval_io(
            fn_name="read_csv",
            # read_csv kwargs
            filepath_or_buffer="modin/pandas/test/data/newlines.csv",
            nrows=nrows,
            skiprows=skiprows,
            cast_to_str=StorageFormat.get() != "Hdk",
        )

    @pytest.mark.parametrize("skiprows", [None, 0, [], [1, 2], np.arange(0, 2)])
    def test_read_csv_skiprows_with_usecols(self, skiprows):
        usecols = {"float_data": "float64"}
        eval_io(
            fn_name="read_csv",
            # read_csv kwargs
            filepath_or_buffer="modin/pandas/test/data/issue_4543.csv",
            skiprows=skiprows,
            usecols=usecols.keys(),
            dtype=usecols,
        )

    def test_read_csv_sep_none(self):
        eval_io(
            fn_name="read_csv",
            modin_warning=ParserWarning,
            # read_csv kwargs
            filepath_or_buffer=pytest.csvs_names["test_read_csv_regular"],
            sep=None,
        )

    def test_read_csv_incorrect_data(self):
        eval_io(
            fn_name="read_csv",
            # read_csv kwargs
            filepath_or_buffer="modin/pandas/test/data/test_categories.json",
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"names": [5, 1, 3, 4, 2, 6]},
            {"names": [0]},
            {"names": None, "usecols": [1, 0, 2]},
            {"names": [3, 1, 2, 5], "usecols": [4, 1, 3, 2]},
        ],
    )
    def test_read_csv_names_neq_num_cols(self, kwargs):
        eval_io(
            fn_name="read_csv",
            # read_csv kwargs
            filepath_or_buffer="modin/pandas/test/data/issue_2074.csv",
            **kwargs,
        )

    def test_read_csv_wrong_path(self):
        raising_exceptions = [e for e in io_ops_bad_exc if e != FileNotFoundError]

        eval_io(
            fn_name="read_csv",
            raising_exceptions=raising_exceptions,
            # read_csv kwargs
            filepath_or_buffer="/some/wrong/path.csv",
        )

    @pytest.mark.parametrize("extension", [None, "csv", "csv.gz"])
    @pytest.mark.parametrize("sep", [" "])
    @pytest.mark.parametrize("header", [False, True, "sfx-"])
    @pytest.mark.parametrize("mode", ["w", "wb+"])
    @pytest.mark.parametrize("idx_name", [None, "Index"])
    @pytest.mark.parametrize("index", [True, False, "New index"])
    @pytest.mark.parametrize("index_label", [None, False, "New index"])
    @pytest.mark.parametrize("columns", [None, ["col1", "col3", "col5"]])
    @pytest.mark.exclude_in_sanity
    def test_to_csv(
        self,
        tmp_path,
        extension,
        sep,
        header,
        mode,
        idx_name,
        index,
        index_label,
        columns,
    ):
        pandas_df = generate_dataframe(idx_name=idx_name)
        modin_df = pd.DataFrame(pandas_df)

        if isinstance(header, str):
            if columns is None:
                header = [f"{header}{c}" for c in modin_df.columns]
            else:
                header = [f"{header}{c}" for c in columns]

        eval_to_csv_file(
            tmp_path,
            modin_obj=modin_df,
            pandas_obj=pandas_df,
            extension=extension,
            sep=sep,
            header=header,
            mode=mode,
            index=index,
            index_label=index_label,
            columns=columns,
        )

    def test_dataframe_to_csv(self, tmp_path):
        pandas_df = pandas.read_csv(pytest.csvs_names["test_read_csv_regular"])
        modin_df = pd.DataFrame(pandas_df)
        eval_to_csv_file(
            tmp_path,
            modin_obj=modin_df,
            pandas_obj=pandas_df,
            extension="csv",
        )

    def test_series_to_csv(self, tmp_path):
        pandas_s = pandas.read_csv(
            pytest.csvs_names["test_read_csv_regular"], usecols=["col1"]
        ).squeeze()
        modin_s = pd.Series(pandas_s)
        eval_to_csv_file(
            tmp_path,
            modin_obj=modin_s,
            pandas_obj=pandas_s,
            extension="csv",
        )

    def test_read_csv_within_decorator(self):
        @dummy_decorator()
        def wrapped_read_csv(file, method):
            if method == "pandas":
                return pandas.read_csv(file)

            if method == "modin":
                return pd.read_csv(file)

        pandas_df = wrapped_read_csv(
            pytest.csvs_names["test_read_csv_regular"], method="pandas"
        )
        modin_df = wrapped_read_csv(
            pytest.csvs_names["test_read_csv_regular"], method="modin"
        )

        if StorageFormat.get() == "Hdk":
            # Aligning DateTime dtypes because of the bug related to the `parse_dates` parameter:
            # https://github.com/modin-project/modin/issues/3485
            modin_df, pandas_df = align_datetime_dtypes(modin_df, pandas_df)

        df_equals(modin_df, pandas_df)

    @pytest.mark.parametrize(
        "read_mode",
        [
            "r",
            "rb",
        ],
    )
    @pytest.mark.parametrize("buffer_start_pos", [0, 10])
    @pytest.mark.parametrize("set_async_read_mode", [False, True], indirect=True)
    def test_read_csv_file_handle(
        self, read_mode, make_csv_file, buffer_start_pos, set_async_read_mode
    ):
        with ensure_clean() as unique_filename:
            make_csv_file(filename=unique_filename)

            with open(unique_filename, mode=read_mode) as buffer:
                buffer.seek(buffer_start_pos)
                pandas_df = pandas.read_csv(buffer)
                buffer.seek(buffer_start_pos)
                modin_df = pd.read_csv(buffer)
            if AsyncReadMode.get():
                # If read operations are asynchronous, then the dataframes
                # check should be inside `ensure_clean` context
                # because the file may be deleted before actual reading starts
                df_equals(modin_df, pandas_df)
        if not AsyncReadMode.get():
            df_equals(modin_df, pandas_df)

    def test_unnamed_index(self):
        def get_internal_df(df):
            partition = read_df._query_compiler._modin_frame._partitions[0][0]
            return partition.to_pandas()

        path = "modin/pandas/test/data/issue_3119.csv"
        read_df = pd.read_csv(path, index_col=0)
        assert get_internal_df(read_df).index.name is None
        read_df = pd.read_csv(path, index_col=[0, 1])
        for name1, name2 in zip(get_internal_df(read_df).index.names, [None, "a"]):
            assert name1 == name2

    def test_read_csv_empty_frame(self):
        eval_io(
            fn_name="read_csv",
            # read_csv kwargs
            filepath_or_buffer=pytest.csvs_names["test_read_csv_regular"],
            usecols=["col1"],
            index_col="col1",
        )

    @pytest.mark.parametrize(
        "skiprows",
        [
            [x for x in range(10)],
            [x + 5 for x in range(15)],
            [x for x in range(10) if x % 2 == 0],
            [x + 5 for x in range(15) if x % 2 == 0],
            lambda x: x % 2,
            lambda x: x > 20,
            lambda x: x < 20,
            lambda x: True,
            lambda x: x in [10, 20],
            lambda x: x << 10,
        ],
    )
    @pytest.mark.parametrize("header", ["infer", None, 0, 1, 150])
    def test_read_csv_skiprows_corner_cases(self, skiprows, header):
        eval_io(
            fn_name="read_csv",
            check_kwargs_callable=not callable(skiprows),
            # read_csv kwargs
            filepath_or_buffer=pytest.csvs_names["test_read_csv_regular"],
            skiprows=skiprows,
            header=header,
            dtype="str",  # to avoid issues with heterogeneous data
        )

    def test_to_csv_with_index(self, tmp_path):
        cols = 100
        arows = 20000
        keyrange = 100
        values = np.vstack(
            [
                np.random.choice(keyrange, size=(arows)),
                np.random.normal(size=(cols, arows)),
            ]
        ).transpose()
        modin_df = pd.DataFrame(
            values,
            columns=["key"] + ["avalue" + str(i) for i in range(1, 1 + cols)],
        ).set_index("key")
        pandas_df = pandas.DataFrame(
            values,
            columns=["key"] + ["avalue" + str(i) for i in range(1, 1 + cols)],
        ).set_index("key")
        eval_to_csv_file(tmp_path, modin_df, pandas_df, "csv")

    @pytest.mark.parametrize("set_async_read_mode", [False, True], indirect=True)
    def test_read_csv_issue_5150(self, set_async_read_mode):
        with ensure_clean(".csv") as unique_filename:
            pandas_df = pandas.DataFrame(
                np.random.randint(0, 100, size=(2**6, 2**6))
            )
            pandas_df.to_csv(unique_filename, index=False)
            expected_pandas_df = pandas.read_csv(unique_filename, index_col=False)
            modin_df = pd.read_csv(unique_filename, index_col=False)
            actual_pandas_df = modin_df._to_pandas()
            if AsyncReadMode.get():
                # If read operations are asynchronous, then the dataframes
                # check should be inside `ensure_clean` context
                # because the file may be deleted before actual reading starts
                df_equals(expected_pandas_df, actual_pandas_df)
        if not AsyncReadMode.get():
            df_equals(expected_pandas_df, actual_pandas_df)

    @pytest.mark.parametrize("usecols", [None, [0, 1, 2, 3, 4]])
    def test_read_csv_1930(self, usecols):
        eval_io(
            fn_name="read_csv",
            # read_csv kwargs
            filepath_or_buffer="modin/pandas/test/data/issue_1930.csv",
            names=["c1", "c2", "c3", "c4", "c5"],
            usecols=usecols,
        )


def _check_relative_io(fn_name, unique_filename, path_arg, storage_default=()):
    # Windows can be funny at where it searches for ~; besides, Python >= 3.8 no longer honors %HOME%
    dirname, basename = os.path.split(unique_filename)
    pinned_home = {envvar: dirname for envvar in ("HOME", "USERPROFILE", "HOMEPATH")}
    should_default = Engine.get() == "Python" or StorageFormat.get() in storage_default
    with mock.patch.dict(os.environ, pinned_home):
        with warns_that_defaulting_to_pandas() if should_default else _nullcontext():
            eval_io(
                fn_name=fn_name,
                **{path_arg: f"~/{basename}"},
            )
        # check that when read without $HOME patched we have equivalent results
        eval_general(
            f"~/{basename}",
            unique_filename,
            lambda fname: getattr(pandas, fn_name)(**{path_arg: fname}),
        )


# Leave this test apart from the test classes, which skip the default to pandas
# warning check. We want to make sure we are NOT defaulting to pandas for a
# path relative to user home.
# TODO(https://github.com/modin-project/modin/issues/3655): Get rid of this
# commment once we turn all default to pandas messages into errors.
def test_read_csv_relative_to_user_home(make_csv_file):
    with ensure_clean(".csv") as unique_filename:
        make_csv_file(filename=unique_filename)
        _check_relative_io("read_csv", unique_filename, "filepath_or_buffer")


@pytest.mark.filterwarnings(default_to_pandas_ignore_string)
class TestTable:
    def test_read_table(self, make_csv_file):
        with ensure_clean() as unique_filename:
            make_csv_file(filename=unique_filename, delimiter="\t")
            eval_io(
                fn_name="read_table",
                # read_table kwargs
                filepath_or_buffer=unique_filename,
            )

    @pytest.mark.parametrize("set_async_read_mode", [False, True], indirect=True)
    def test_read_table_within_decorator(self, make_csv_file, set_async_read_mode):
        @dummy_decorator()
        def wrapped_read_table(file, method):
            if method == "pandas":
                return pandas.read_table(file)

            if method == "modin":
                return pd.read_table(file)

        with ensure_clean() as unique_filename:
            make_csv_file(filename=unique_filename, delimiter="\t")

            pandas_df = wrapped_read_table(unique_filename, method="pandas")
            modin_df = wrapped_read_table(unique_filename, method="modin")

        if StorageFormat.get() == "Hdk":
            modin_df, pandas_df = align_datetime_dtypes(modin_df, pandas_df)

            if AsyncReadMode.get():
                # If read operations are asynchronous, then the dataframes
                # check should be inside `ensure_clean` context
                # because the file may be deleted before actual reading starts
                df_equals(modin_df, pandas_df)
        if not AsyncReadMode.get():
            df_equals(modin_df, pandas_df)

    def test_read_table_empty_frame(self, make_csv_file):
        with ensure_clean() as unique_filename:
            make_csv_file(filename=unique_filename, delimiter="\t")

            eval_io(
                fn_name="read_table",
                # read_table kwargs
                filepath_or_buffer=unique_filename,
                usecols=["col1"],
                index_col="col1",
            )


@pytest.mark.parametrize("engine", ["pyarrow", "fastparquet"])
@pytest.mark.filterwarnings(default_to_pandas_ignore_string)
class TestParquet:
    @pytest.mark.parametrize("columns", [None, ["col1"]])
    @pytest.mark.parametrize("row_group_size", [None, 100, 1000, 10_000])
    @pytest.mark.parametrize("path_type", [Path, str])
    def test_read_parquet(
        self, engine, make_parquet_file, columns, row_group_size, path_type
    ):
        with ensure_clean(".parquet") as unique_filename:
            unique_filename = path_type(unique_filename)
            make_parquet_file(filename=unique_filename, row_group_size=row_group_size)

            eval_io(
                fn_name="read_parquet",
                # read_parquet kwargs
                engine=engine,
                path=unique_filename,
                columns=columns,
            )

    @pytest.mark.parametrize(
        "dtype_backend", [lib.no_default, "numpy_nullable", "pyarrow"]
    )
    def test_read_parquet_dtype_backend(self, engine, make_parquet_file, dtype_backend):
        with ensure_clean(".parquet") as unique_filename:
            make_parquet_file(filename=unique_filename, row_group_size=100)

            def comparator(df1, df2):
                df_equals(df1, df2)
                df_equals(df1.dtypes, df2.dtypes)

            eval_io(
                fn_name="read_parquet",
                # read_parquet kwargs
                engine=engine,
                path=unique_filename,
                dtype_backend=dtype_backend,
                comparator=comparator,
            )

    def test_read_parquet_list_of_files_5698(self, engine, make_parquet_file):
        if engine == "fastparquet" and os.name == "nt":
            pytest.xfail(reason="https://github.com/pandas-dev/pandas/issues/51720")
        with ensure_clean(".parquet") as f1, ensure_clean(
            ".parquet"
        ) as f2, ensure_clean(".parquet") as f3:
            for f in [f1, f2, f3]:
                make_parquet_file(filename=f)
            eval_io(fn_name="read_parquet", path=[f1, f2, f3], engine=engine)

    def test_read_parquet_indexing_by_column(self, tmp_path, engine, make_parquet_file):
        # Test indexing into a column of Modin with various parquet file row lengths.
        # Specifically, tests for https://github.com/modin-project/modin/issues/3527
        # which fails when min_partition_size < nrows < min_partition_size * (num_partitions - 1)

        nrows = (
            MinPartitionSize.get() + 1
        )  # Use the minimal guaranteed failing value for nrows.
        unique_filename = get_unique_filename(extension="parquet", data_dir=tmp_path)
        make_parquet_file(filename=unique_filename, nrows=nrows)

        parquet_df = pd.read_parquet(unique_filename, engine=engine)
        for col in parquet_df.columns:
            parquet_df[col]

    @pytest.mark.parametrize("columns", [None, ["col1"]])
    @pytest.mark.parametrize("row_group_size", [None, 100, 1000, 10_000])
    @pytest.mark.parametrize(
        "rows_per_file", [[1000] * 40, [0, 0, 40_000], [10_000, 10_000] + [100] * 200]
    )
    @pytest.mark.exclude_in_sanity
    def test_read_parquet_directory(
        self, engine, make_parquet_dir, columns, row_group_size, rows_per_file
    ):
        num_cols = DATASET_SIZE_DICT.get(
            TestDatasetSize.get(), DATASET_SIZE_DICT["Small"]
        )
        dfs_by_filename = {}
        start_row = 0
        for i, length in enumerate(rows_per_file):
            end_row = start_row + length
            dfs_by_filename[f"{i}.parquet"] = pandas.DataFrame(
                {f"col{x + 1}": np.arange(start_row, end_row) for x in range(num_cols)}
            )
            start_row = end_row
        path = make_parquet_dir(dfs_by_filename, row_group_size)

        # There are specific files that PyArrow will try to ignore by default
        # in a parquet directory. One example are files that start with '_'. Our
        # previous implementation tried to read all files in a parquet directory,
        # but we now make use of PyArrow to ensure the directory is valid.
        with open(os.path.join(path, "_committed_file"), "w+") as f:
            f.write("testingtesting")

        eval_io(
            fn_name="read_parquet",
            # read_parquet kwargs
            engine=engine,
            path=path,
            columns=columns,
        )

    @pytest.mark.parametrize("columns", [None, ["col1"]])
    def test_read_parquet_partitioned_directory(
        self, tmp_path, make_parquet_file, columns, engine
    ):
        unique_filename = get_unique_filename(extension=None, data_dir=tmp_path)
        make_parquet_file(filename=unique_filename, partitioned_columns=["col1"])

        eval_io(
            fn_name="read_parquet",
            # read_parquet kwargs
            engine=engine,
            path=unique_filename,
            columns=columns,
        )

    def test_read_parquet_pandas_index(self, engine):
        if (
            version.parse(pa.__version__) >= version.parse("12.0.0")
            and version.parse(pd.__version__) < version.parse("2.0.0")
            and engine == "pyarrow"
        ):
            pytest.xfail("incompatible versions; see #6072")
        # Ensure modin can read parquet files written by pandas with a non-RangeIndex object
        pandas_df = pandas.DataFrame(
            {
                "idx": np.random.randint(0, 100_000, size=2000),
                "idx_categorical": pandas.Categorical(["y", "z"] * 1000),
                # Can't do interval index right now because of this bug fix that is planned
                # to be apart of the pandas 1.5.0 release: https://github.com/pandas-dev/pandas/pull/46034
                # "idx_interval": pandas.interval_range(start=0, end=2000),
                "idx_periodrange": pandas.period_range(
                    start="2017-01-01", periods=2000
                ),
                "A": np.random.randint(0, 100_000, size=2000),
                "B": ["a", "b"] * 1000,
                "C": ["c"] * 2000,
            }
        )
        # Older versions of pyarrow do not support Arrow to Parquet
        # schema conversion for duration[ns]
        # https://issues.apache.org/jira/browse/ARROW-6780
        if version.parse(pa.__version__) >= version.parse("8.0.0"):
            pandas_df["idx_timedelta"] = pandas.timedelta_range(
                start="1 day", periods=2000
            )

        # There is a non-deterministic bug in the fastparquet engine when we
        # try to set the index to the datetime column. Please see:
        # https://github.com/dask/fastparquet/issues/796
        if engine == "pyarrow":
            pandas_df["idx_datetime"] = pandas.date_range(
                start="1/1/2018", periods=2000
            )

        for col in pandas_df.columns:
            if col.startswith("idx"):
                with ensure_clean(".parquet") as unique_filename:
                    pandas_df.set_index(col).to_parquet(unique_filename)
                    # read the same parquet using modin.pandas
                    eval_io(
                        "read_parquet",
                        # read_parquet kwargs
                        path=unique_filename,
                        engine=engine,
                    )

        with ensure_clean(".parquet") as unique_filename:
            pandas_df.set_index(["idx", "A"]).to_parquet(unique_filename)
            eval_io(
                "read_parquet",
                # read_parquet kwargs
                path=unique_filename,
                engine=engine,
            )

    def test_read_parquet_pandas_index_partitioned(self, tmp_path, engine):
        # Ensure modin can read parquet files written by pandas with a non-RangeIndex object
        pandas_df = pandas.DataFrame(
            {
                "idx": np.random.randint(0, 100_000, size=2000),
                "A": np.random.randint(0, 10, size=2000),
                "B": ["a", "b"] * 1000,
                "C": ["c"] * 2000,
            }
        )
        unique_filename = get_unique_filename(extension="parquet", data_dir=tmp_path)
        pandas_df.set_index("idx").to_parquet(unique_filename, partition_cols=["A"])
        # read the same parquet using modin.pandas
        eval_io(
            "read_parquet",
            # read_parquet kwargs
            path=unique_filename,
            engine=engine,
        )

    def test_read_parquet_hdfs(self, engine):
        eval_io(
            fn_name="read_parquet",
            # read_parquet kwargs
            path="modin/pandas/test/data/hdfs.parquet",
            engine=engine,
        )

    @pytest.mark.parametrize(
        "path_type",
        ["object", "directory", "url"],
    )
    def test_read_parquet_s3(self, path_type, engine):
        dataset_url = "s3://modin-datasets/testing/test_data.parquet"
        if path_type == "object":
            import s3fs

            fs = s3fs.S3FileSystem(anon=True)
            with fs.open(dataset_url, "rb") as file_obj:
                eval_io("read_parquet", path=file_obj, engine=engine)
        elif path_type == "directory":
            eval_io(
                "read_parquet",
                path="s3://modin-datasets/test_data_dir.parquet",
                storage_options={"anon": True},
                engine=engine,
            )
        else:
            eval_io(
                "read_parquet",
                path=dataset_url,
                storage_options={"anon": True},
                engine=engine,
            )

    def test_read_parquet_without_metadata(self, tmp_path, engine):
        """Test that Modin can read parquet files not written by pandas."""
        from pyarrow import csv
        from pyarrow import parquet

        parquet_fname = get_unique_filename(extension="parquet", data_dir=tmp_path)
        csv_fname = get_unique_filename(extension="parquet", data_dir=tmp_path)
        pandas_df = pandas.DataFrame(
            {
                "idx": np.random.randint(0, 100_000, size=2000),
                "A": np.random.randint(0, 10, size=2000),
                "B": ["a", "b"] * 1000,
                "C": ["c"] * 2000,
            }
        )
        pandas_df.to_csv(csv_fname, index=False)
        # read into pyarrow table and write it to a parquet file
        t = csv.read_csv(csv_fname)
        parquet.write_table(t, parquet_fname)

        eval_io(
            "read_parquet",
            # read_parquet kwargs
            path=parquet_fname,
            engine=engine,
        )

    def test_read_empty_parquet_file(self, tmp_path, engine):
        test_df = pandas.DataFrame()
        path = tmp_path / "data"
        path.mkdir()
        test_df.to_parquet(path / "part-00000.parquet", engine=engine)
        eval_io(fn_name="read_parquet", path=path, engine=engine)

    @pytest.mark.parametrize(
        "compression_kwargs",
        [
            pytest.param({}, id="no_compression_kwargs"),
            pytest.param({"compression": None}, id="compression=None"),
            pytest.param({"compression": "gzip"}, id="compression=gzip"),
            pytest.param({"compression": "snappy"}, id="compression=snappy"),
            pytest.param({"compression": "brotli"}, id="compression=brotli"),
        ],
    )
    @pytest.mark.parametrize("extension", ["parquet", ".gz", ".bz2", ".zip", ".xz"])
    def test_to_parquet(self, tmp_path, engine, compression_kwargs, extension):
        modin_df, pandas_df = create_test_dfs(TEST_DATA)
        parquet_eval_to_file(
            tmp_path,
            modin_obj=modin_df,
            pandas_obj=pandas_df,
            fn="to_parquet",
            extension="parquet",
            engine=engine,
        )

    def test_to_parquet_keep_index(self, tmp_path, engine):
        data = {"c0": [0, 1] * 1000, "c1": [2, 3] * 1000}
        modin_df, pandas_df = create_test_dfs(data)
        modin_df.index.name = "foo"
        pandas_df.index.name = "foo"

        parquet_eval_to_file(
            tmp_path,
            modin_obj=modin_df,
            pandas_obj=pandas_df,
            fn="to_parquet",
            extension="parquet",
            index=True,
            engine=engine,
        )

    def test_to_parquet_s3(self, s3_resource, engine, s3_storage_options):
        # use utils_test_data because it spans multiple partitions
        modin_path = "s3://modin-test/modin-dir/modin_df.parquet"
        mdf, pdf = create_test_dfs(utils_test_data["int_data"])
        pdf.to_parquet(
            "s3://modin-test/pandas-dir/pandas_df.parquet",
            engine=engine,
            storage_options=s3_storage_options,
        )
        mdf.to_parquet(modin_path, engine=engine, storage_options=s3_storage_options)
        df_equals(
            pandas.read_parquet(
                "s3://modin-test/pandas-dir/pandas_df.parquet",
                storage_options=s3_storage_options,
            ),
            pd.read_parquet(modin_path, storage_options=s3_storage_options),
        )
        # check we're not creating local file:
        # https://github.com/modin-project/modin/issues/5888
        assert not os.path.isdir(modin_path)

    def test_read_parquet_2462(self, tmp_path, engine):
        test_df = pandas.DataFrame({"col1": [["ad_1", "ad_2"], ["ad_3"]]})
        path = tmp_path / "data"
        path.mkdir()
        test_df.to_parquet(path / "part-00000.parquet", engine=engine)
        read_df = pd.read_parquet(path, engine=engine)
        df_equals(test_df, read_df)

    def test_read_parquet_5767(self, tmp_path, engine):
        test_df = pandas.DataFrame({"a": [1, 2, 3, 4], "b": [1, 1, 2, 2]})
        path = tmp_path / "data"
        path.mkdir()
        file_name = "modin_issue#0000.parquet"
        test_df.to_parquet(path / file_name, engine=engine, partition_cols=["b"])
        read_df = pd.read_parquet(path / file_name)
        # both Modin and pandas read column "b" as a category
        df_equals(test_df, read_df.astype("int64"))

    def test_read_parquet_5509(self, tmp_path, engine):
        test_df = pandas.DataFrame({"col_a": [1, 2, 3], "col_b": ["a", "b", "c"]})

        path = tmp_path / "data"
        path.mkdir()
        file_name = "5509.parquet"
        test_df.to_parquet(path / file_name)
        with warns_that_defaulting_to_pandas():
            eval_io(
                fn_name="read_parquet",
                path=str(path / file_name),
                columns=["col_b"],
                engine=engine,
                filters=[[("col_a", "==", 1)]],
            )

    def test_read_parquet_s3_with_column_partitioning(self, engine):
        # This test case comes from
        # https://github.com/modin-project/modin/issues/4636
        dataset_url = "s3://modin-datasets/modin-bugs/modin_bug_5159_parquet/df.parquet"
        eval_io(
            fn_name="read_parquet",
            path=dataset_url,
            engine=engine,
            storage_options={"anon": True},
        )


# Leave this test apart from the test classes, which skip the default to pandas
# warning check. We want to make sure we are NOT defaulting to pandas for a
# path relative to user home.
# TODO(https://github.com/modin-project/modin/issues/3655): Get rid of this
# commment once we turn all default to pandas messages into errors.
def test_read_parquet_relative_to_user_home(make_parquet_file):
    with ensure_clean(".parquet") as unique_filename:
        make_parquet_file(filename=unique_filename)
        _check_relative_io(
            "read_parquet", unique_filename, "path", storage_default=("Hdk",)
        )


@pytest.mark.filterwarnings(default_to_pandas_ignore_string)
class TestJson:
    @pytest.mark.parametrize("lines", [False, True])
    def test_read_json(self, make_json_file, lines):
        eval_io(
            fn_name="read_json",
            # read_json kwargs
            path_or_buf=make_json_file(lines=lines),
            lines=lines,
        )

    @pytest.mark.parametrize(
        "dtype_backend", [lib.no_default, "numpy_nullable", "pyarrow"]
    )
    def test_read_json_dtype_backend(self, make_json_file, dtype_backend):
        def comparator(df1, df2):
            df_equals(df1, df2)
            df_equals(df1.dtypes, df2.dtypes)

        eval_io(
            fn_name="read_json",
            # read_json kwargs
            path_or_buf=make_json_file(lines=True),
            lines=True,
            dtype_backend=dtype_backend,
            comparator=comparator,
        )

    @pytest.mark.parametrize(
        "storage_options",
        [{"anon": False}, {"anon": True}, {"key": "123", "secret": "123"}, None],
    )
    def test_read_json_s3(self, storage_options):
        eval_io(
            fn_name="read_json",
            path_or_buf="s3://modin-datasets/testing/test_data.json",
            lines=True,
            orient="records",
            storage_options=storage_options,
        )

    def test_read_json_categories(self):
        eval_io(
            fn_name="read_json",
            # read_json kwargs
            path_or_buf="modin/pandas/test/data/test_categories.json",
            dtype={"one": "int64", "two": "category"},
        )

    def test_read_json_different_columns(self):
        with warns_that_defaulting_to_pandas():
            eval_io(
                fn_name="read_json",
                # read_json kwargs
                path_or_buf="modin/pandas/test/data/test_different_columns_in_rows.json",
                lines=True,
            )

    @pytest.mark.parametrize(
        "data",
        [json_short_string, json_short_bytes, json_long_string, json_long_bytes],
    )
    def test_read_json_string_bytes(self, data):
        with warns_that_defaulting_to_pandas():
            modin_df = pd.read_json(data)
        # For I/O objects we need to rewind to reuse the same object.
        if hasattr(data, "seek"):
            data.seek(0)
        df_equals(modin_df, pandas.read_json(data))

    def test_to_json(self, tmp_path):
        modin_df, pandas_df = create_test_dfs(TEST_DATA)
        eval_to_file(
            tmp_path,
            modin_obj=modin_df,
            pandas_obj=pandas_df,
            fn="to_json",
            extension="json",
        )

    @pytest.mark.parametrize(
        "read_mode",
        [
            "r",
            "rb",
        ],
    )
    def test_read_json_file_handle(self, make_json_file, read_mode):
        with open(make_json_file(), mode=read_mode) as buf:
            df_pandas = pandas.read_json(buf)
            buf.seek(0)
            df_modin = pd.read_json(buf)
            df_equals(df_pandas, df_modin)

    def test_read_json_metadata(self, make_json_file):
        # `lines=True` is for triggering Modin implementation,
        # `orient="records"` should be set if `lines=True`
        df = pd.read_json(
            make_json_file(ncols=80, lines=True), lines=True, orient="records"
        )
        parts_width_cached = df._query_compiler._modin_frame._column_widths_cache
        num_splits = len(df._query_compiler._modin_frame._partitions[0])
        parts_width_actual = [
            len(df._query_compiler._modin_frame._partitions[0][i].get().columns)
            for i in range(num_splits)
        ]

        assert parts_width_cached == parts_width_actual


@pytest.mark.filterwarnings(default_to_pandas_ignore_string)
class TestExcel:
    @check_file_leaks
    def test_read_excel(self, make_excel_file):
        eval_io(
            fn_name="read_excel",
            # read_excel kwargs
            io=make_excel_file(),
        )

    @check_file_leaks
    @pytest.mark.parametrize(
        "dtype_backend", [lib.no_default, "numpy_nullable", "pyarrow"]
    )
    def test_read_excel_dtype_backend(self, make_excel_file, dtype_backend):
        def comparator(df1, df2):
            df_equals(df1, df2)
            df_equals(df1.dtypes, df2.dtypes)

        eval_io(
            fn_name="read_excel",
            # read_csv kwargs
            io=make_excel_file(),
            dtype_backend=dtype_backend,
            comparator=comparator,
        )

    @check_file_leaks
    def test_read_excel_engine(self, make_excel_file):
        eval_io(
            fn_name="read_excel",
            modin_warning=UserWarning,
            # read_excel kwargs
            io=make_excel_file(),
            engine="openpyxl",
        )

    @check_file_leaks
    def test_read_excel_index_col(self, make_excel_file):
        eval_io(
            fn_name="read_excel",
            modin_warning=UserWarning,
            # read_excel kwargs
            io=make_excel_file(),
            index_col=0,
        )

    @check_file_leaks
    def test_read_excel_all_sheets(self, make_excel_file):
        unique_filename = make_excel_file()

        pandas_df = pandas.read_excel(unique_filename, sheet_name=None)
        modin_df = pd.read_excel(unique_filename, sheet_name=None)

        assert isinstance(pandas_df, (OrderedDict, dict))
        assert isinstance(modin_df, type(pandas_df))
        assert pandas_df.keys() == modin_df.keys()

        for key in pandas_df.keys():
            df_equals(modin_df.get(key), pandas_df.get(key))

    @pytest.mark.xfail(
        Engine.get() != "Python" and StorageFormat.get() != "Hdk",
        reason="pandas throws the exception. See pandas issue #39250 for more info",
    )
    @check_file_leaks
    def test_read_excel_sheetname_title(self):
        eval_io(
            fn_name="read_excel",
            # read_excel kwargs
            io="modin/pandas/test/data/excel_sheetname_title.xlsx",
        )

    @check_file_leaks
    def test_excel_empty_line(self):
        path = "modin/pandas/test/data/test_emptyline.xlsx"
        modin_df = pd.read_excel(path)
        assert str(modin_df)

    @check_file_leaks
    def test_read_excel_empty_rows(self):
        # Test parsing empty rows in middle of excel dataframe as NaN values
        eval_io(
            fn_name="read_excel",
            io="modin/pandas/test/data/test_empty_rows.xlsx",
        )

    @check_file_leaks
    def test_read_excel_border_rows(self):
        # Test parsing border rows as NaN values in excel dataframe
        eval_io(
            fn_name="read_excel",
            io="modin/pandas/test/data/test_border_rows.xlsx",
        )

    @check_file_leaks
    def test_read_excel_every_other_nan(self):
        # Test for reading excel dataframe with every other row as a NaN value
        eval_io(
            fn_name="read_excel",
            io="modin/pandas/test/data/every_other_row_nan.xlsx",
        )

    @check_file_leaks
    def test_read_excel_header_none(self):
        eval_io(
            fn_name="read_excel",
            io="modin/pandas/test/data/every_other_row_nan.xlsx",
            header=None,
        )

    @pytest.mark.parametrize(
        "sheet_name",
        [
            "Sheet1",
            "AnotherSpecialName",
            "SpecialName",
            "SecondSpecialName",
            0,
            1,
            2,
            3,
        ],
    )
    @check_file_leaks
    def test_read_excel_sheet_name(self, sheet_name):
        eval_io(
            fn_name="read_excel",
            # read_excel kwargs
            io="modin/pandas/test/data/modin_error_book.xlsx",
            sheet_name=sheet_name,
            # https://github.com/modin-project/modin/issues/5965
            comparator_kwargs={"check_dtypes": False},
        )

    def test_ExcelFile(self, make_excel_file):
        unique_filename = make_excel_file()

        modin_excel_file = pd.ExcelFile(unique_filename)
        pandas_excel_file = pandas.ExcelFile(unique_filename)

        try:
            df_equals(modin_excel_file.parse(), pandas_excel_file.parse())
            assert modin_excel_file.io == unique_filename
            assert isinstance(modin_excel_file, pd.ExcelFile)
        finally:
            modin_excel_file.close()
            pandas_excel_file.close()

    @pytest.mark.xfail(strict=False, reason="Flaky test, defaults to pandas")
    def test_to_excel(self, tmp_path):
        modin_df, pandas_df = create_test_dfs(TEST_DATA)

        unique_filename_modin = get_unique_filename(extension="xlsx", data_dir=tmp_path)
        unique_filename_pandas = get_unique_filename(
            extension="xlsx", data_dir=tmp_path
        )

        modin_writer = pandas.ExcelWriter(unique_filename_modin)
        pandas_writer = pandas.ExcelWriter(unique_filename_pandas)

        modin_df.to_excel(modin_writer)
        pandas_df.to_excel(pandas_writer)

        modin_writer.save()
        pandas_writer.save()

        assert assert_files_eq(unique_filename_modin, unique_filename_pandas)

    @check_file_leaks
    def test_read_excel_empty_frame(self, make_excel_file):
        eval_io(
            fn_name="read_excel",
            modin_warning=UserWarning,
            # read_excel kwargs
            io=make_excel_file(),
            usecols=[0],
            index_col=0,
        )


@pytest.mark.filterwarnings(default_to_pandas_ignore_string)
class TestHdf:
    @pytest.mark.parametrize("format", [None, "table"])
    def test_read_hdf(self, make_hdf_file, format):
        eval_io(
            fn_name="read_hdf",
            # read_hdf kwargs
            path_or_buf=make_hdf_file(format=format),
            key="df",
        )

    def test_HDFStore(self, tmp_path):
        unique_filename_modin = get_unique_filename(extension="hdf", data_dir=tmp_path)
        unique_filename_pandas = get_unique_filename(extension="hdf", data_dir=tmp_path)

        modin_store = pd.HDFStore(unique_filename_modin)
        pandas_store = pandas.HDFStore(unique_filename_pandas)

        modin_df, pandas_df = create_test_dfs(TEST_DATA)

        modin_store["foo"] = modin_df
        pandas_store["foo"] = pandas_df

        modin_df = modin_store.get("foo")
        pandas_df = pandas_store.get("foo")
        df_equals(modin_df, pandas_df)

        modin_store.close()
        pandas_store.close()
        modin_df = pandas.read_hdf(unique_filename_modin, key="foo", mode="r")
        pandas_df = pandas.read_hdf(unique_filename_pandas, key="foo", mode="r")
        df_equals(modin_df, pandas_df)
        assert isinstance(modin_store, pd.HDFStore)

        with ensure_clean(".hdf5") as hdf_file:
            with pd.HDFStore(hdf_file, mode="w") as store:
                store.append("data/df1", pd.DataFrame(np.random.randn(5, 5)))
                store.append("data/df2", pd.DataFrame(np.random.randn(4, 4)))

            modin_df = pd.read_hdf(hdf_file, key="data/df1", mode="r")
            pandas_df = pandas.read_hdf(hdf_file, key="data/df1", mode="r")
        df_equals(modin_df, pandas_df)

    def test_HDFStore_in_read_hdf(self):
        with ensure_clean(".hdf") as filename:
            dfin = pd.DataFrame(np.random.rand(8, 8))
            dfin.to_hdf(filename, "/key")

            with pd.HDFStore(filename) as h:
                modin_df = pd.read_hdf(h, "/key")
            with pandas.HDFStore(filename) as h:
                pandas_df = pandas.read_hdf(h, "/key")
        df_equals(modin_df, pandas_df)


@pytest.mark.filterwarnings(default_to_pandas_ignore_string)
class TestSql:
    @pytest.mark.parametrize("read_sql_engine", ["Pandas", "Connectorx"])
    def test_read_sql(self, tmp_path, make_sql_connection, read_sql_engine):
        filename = get_unique_filename(".db")
        table = "test_read_sql"
        conn = make_sql_connection(tmp_path / filename, table)
        query = f"select * from {table}"

        eval_io(
            fn_name="read_sql",
            # read_sql kwargs
            sql=query,
            con=conn,
        )

        eval_io(
            fn_name="read_sql",
            # read_sql kwargs
            sql=query,
            con=conn,
            index_col="index",
        )

        with warns_that_defaulting_to_pandas():
            pd.read_sql_query(query, conn)

        with warns_that_defaulting_to_pandas():
            pd.read_sql_table(table, conn)

        # Test SQLAlchemy engine
        sqlalchemy_engine = sa.create_engine(conn)
        eval_io(
            fn_name="read_sql",
            # read_sql kwargs
            sql=query,
            con=sqlalchemy_engine,
        )

        # Test SQLAlchemy Connection
        sqlalchemy_connection = sqlalchemy_engine.connect()
        eval_io(
            fn_name="read_sql",
            # read_sql kwargs
            sql=query,
            con=sqlalchemy_connection,
        )

        old_sql_engine = ReadSqlEngine.get()
        ReadSqlEngine.put(read_sql_engine)
        if ReadSqlEngine.get() == "Connectorx":
            modin_df = pd.read_sql(sql=query, con=conn)
        else:
            modin_df = pd.read_sql(
                sql=query, con=ModinDatabaseConnection("sqlalchemy", conn)
            )
        ReadSqlEngine.put(old_sql_engine)
        pandas_df = pandas.read_sql(sql=query, con=sqlalchemy_connection)
        df_equals(modin_df, pandas_df)

    @pytest.mark.parametrize(
        "dtype_backend", [lib.no_default, "numpy_nullable", "pyarrow"]
    )
    def test_read_sql_dtype_backend(self, tmp_path, make_sql_connection, dtype_backend):
        filename = get_unique_filename(extension="db")

        table = "test_read_sql_dtype_backend"
        conn = make_sql_connection(tmp_path / filename, table)
        query = f"select * from {table}"

        def comparator(df1, df2):
            df_equals(df1, df2)
            df_equals(df1.dtypes, df2.dtypes)

        eval_io(
            fn_name="read_sql",
            # read_sql kwargs
            sql=query,
            con=conn,
            dtype_backend=dtype_backend,
            comparator=comparator,
        )

    @pytest.mark.skipif(
        not TestReadFromSqlServer.get(),
        reason="Skip the test when the test SQL server is not set up.",
    )
    def test_read_sql_from_sql_server(self):
        table_name = "test_1000x256"
        query = f"SELECT * FROM {table_name}"
        sqlalchemy_connection_string = (
            "mssql+pymssql://sa:Strong.Pwd-123@0.0.0.0:1433/master"
        )
        pandas_df_to_read = pandas.DataFrame(
            np.arange(
                1000 * 256,
            ).reshape(1000, 256)
        ).add_prefix("col")
        pandas_df_to_read.to_sql(
            table_name, sqlalchemy_connection_string, if_exists="replace"
        )
        modin_df = pd.read_sql(
            query,
            ModinDatabaseConnection("sqlalchemy", sqlalchemy_connection_string),
        )
        pandas_df = pandas.read_sql(query, sqlalchemy_connection_string)
        df_equals(modin_df, pandas_df)

    @pytest.mark.skipif(
        not TestReadFromPostgres.get(),
        reason="Skip the test when the postgres server is not set up.",
    )
    def test_read_sql_from_postgres(self):
        table_name = "test_1000x256"
        query = f"SELECT * FROM {table_name}"
        connection = "postgresql://sa:Strong.Pwd-123@localhost:2345/postgres"
        pandas_df_to_read = pandas.DataFrame(
            np.arange(
                1000 * 256,
            ).reshape(1000, 256)
        ).add_prefix("col")
        pandas_df_to_read.to_sql(table_name, connection, if_exists="replace")
        modin_df = pd.read_sql(
            query,
            ModinDatabaseConnection("psycopg2", connection),
        )
        pandas_df = pandas.read_sql(query, connection)
        df_equals(modin_df, pandas_df)

    def test_invalid_modin_database_connections(self):
        with pytest.raises(UnsupportedDatabaseException):
            ModinDatabaseConnection("unsupported_database")

    def test_read_sql_with_chunksize(self, make_sql_connection):
        filename = get_unique_filename(extension="db")
        table = "test_read_sql_with_chunksize"
        conn = make_sql_connection(filename, table)
        query = f"select * from {table}"

        pandas_gen = pandas.read_sql(query, conn, chunksize=10)
        modin_gen = pd.read_sql(query, conn, chunksize=10)
        for modin_df, pandas_df in zip(modin_gen, pandas_gen):
            df_equals(modin_df, pandas_df)

    @pytest.mark.parametrize("index", [False, True])
    @pytest.mark.parametrize("conn_type", ["str", "sqlalchemy", "sqlalchemy+connect"])
    def test_to_sql(self, tmp_path, make_sql_connection, index, conn_type):
        table_name = f"test_to_sql_{str(index)}"
        modin_df, pandas_df = create_test_dfs(TEST_DATA)

        # We do not pass the table name so the fixture won't generate a table
        conn = make_sql_connection(tmp_path / f"{table_name}_modin.db")
        if conn_type.startswith("sqlalchemy"):
            conn = sa.create_engine(conn)
            if conn_type == "sqlalchemy+connect":
                conn = conn.connect()
        modin_df.to_sql(table_name, conn, index=index)
        df_modin_sql = pandas.read_sql(
            table_name, con=conn, index_col="index" if index else None
        )

        # We do not pass the table name so the fixture won't generate a table
        conn = make_sql_connection(tmp_path / f"{table_name}_pandas.db")
        if conn_type.startswith("sqlalchemy"):
            conn = sa.create_engine(conn)
            if conn_type == "sqlalchemy+connect":
                conn = conn.connect()
        pandas_df.to_sql(table_name, conn, index=index)
        df_pandas_sql = pandas.read_sql(
            table_name, con=conn, index_col="index" if index else None
        )

        assert df_modin_sql.sort_index().equals(df_pandas_sql.sort_index())


@pytest.mark.filterwarnings(default_to_pandas_ignore_string)
class TestHtml:
    def test_read_html(self, make_html_file):
        eval_io(fn_name="read_html", io=make_html_file())

    def test_to_html(self, tmp_path):
        modin_df, pandas_df = create_test_dfs(TEST_DATA)

        eval_to_file(
            tmp_path,
            modin_obj=modin_df,
            pandas_obj=pandas_df,
            fn="to_html",
            extension="html",
        )


@pytest.mark.filterwarnings(default_to_pandas_ignore_string)
class TestFwf:
    def test_fwf_file(self, make_fwf_file):
        fwf_data = (
            "id8141  360.242940  149.910199 11950.7\n"
            + "id1594  444.953632  166.985655 11788.4\n"
            + "id1849  364.136849  183.628767 11806.2\n"
            + "id1230  413.836124  184.375703 11916.8\n"
            + "id1948  502.953953  173.237159 12468.3\n"
        )
        unique_filename = make_fwf_file(fwf_data=fwf_data)

        colspecs = [(0, 6), (8, 20), (21, 33), (34, 43)]
        df = pd.read_fwf(unique_filename, colspecs=colspecs, header=None, index_col=0)
        assert isinstance(df, pd.DataFrame)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {
                "colspecs": [
                    (0, 11),
                    (11, 15),
                    (19, 24),
                    (27, 32),
                    (35, 40),
                    (43, 48),
                    (51, 56),
                    (59, 64),
                    (67, 72),
                    (75, 80),
                    (83, 88),
                    (91, 96),
                    (99, 104),
                    (107, 112),
                ],
                "names": ["stationID", "year", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                "na_values": ["-9999"],
                "index_col": ["stationID", "year"],
            },
            {
                "widths": [20, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8],
                "names": ["id", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                "index_col": [0],
            },
        ],
    )
    def test_fwf_file_colspecs_widths(self, make_fwf_file, kwargs):
        unique_filename = make_fwf_file()

        modin_df = pd.read_fwf(unique_filename, **kwargs)
        pandas_df = pd.read_fwf(unique_filename, **kwargs)

        df_equals(modin_df, pandas_df)

    @pytest.mark.parametrize("usecols", [["a"], ["a", "b", "d"], [0, 1, 3]])
    def test_fwf_file_usecols(self, make_fwf_file, usecols):
        fwf_data = (
            "a       b           c          d\n"
            + "id8141  360.242940  149.910199 11950.7\n"
            + "id1594  444.953632  166.985655 11788.4\n"
            + "id1849  364.136849  183.628767 11806.2\n"
            + "id1230  413.836124  184.375703 11916.8\n"
            + "id1948  502.953953  173.237159 12468.3\n"
        )
        eval_io(
            fn_name="read_fwf",
            # read_fwf kwargs
            filepath_or_buffer=make_fwf_file(fwf_data=fwf_data),
            usecols=usecols,
        )

    @pytest.mark.parametrize(
        "dtype_backend", [lib.no_default, "numpy_nullable", "pyarrow"]
    )
    def test_read_fwf_dtype_backend(self, make_fwf_file, dtype_backend):
        with ensure_clean(".fwf") as unique_filename:
            make_fwf_file(filename=unique_filename)

            def comparator(df1, df2):
                df_equals(df1, df2)
                df_equals(df1.dtypes, df2.dtypes)

            eval_io(
                fn_name="read_fwf",
                # read_csv kwargs
                filepath_or_buffer=unique_filename,
                dtype_backend=dtype_backend,
                comparator=comparator,
            )

    def test_fwf_file_chunksize(self, make_fwf_file):
        unique_filename = make_fwf_file()

        # Tests __next__ and correctness of reader as an iterator
        rdf_reader = pd.read_fwf(unique_filename, chunksize=5)
        pd_reader = pandas.read_fwf(unique_filename, chunksize=5)

        for modin_df, pd_df in zip(rdf_reader, pd_reader):
            df_equals(modin_df, pd_df)

        # Tests that get_chunk works correctly
        rdf_reader = pd.read_fwf(unique_filename, chunksize=1)
        pd_reader = pandas.read_fwf(unique_filename, chunksize=1)

        modin_df = rdf_reader.get_chunk(1)
        pd_df = pd_reader.get_chunk(1)

        df_equals(modin_df, pd_df)

        # Tests that read works correctly
        rdf_reader = pd.read_fwf(unique_filename, chunksize=1)
        pd_reader = pandas.read_fwf(unique_filename, chunksize=1)

        modin_df = rdf_reader.read()
        pd_df = pd_reader.read()

        df_equals(modin_df, pd_df)

    @pytest.mark.parametrize("nrows", [13, None])
    def test_fwf_file_skiprows(self, make_fwf_file, nrows):
        unique_filename = make_fwf_file()

        eval_io(
            fn_name="read_fwf",
            # read_fwf kwargs
            filepath_or_buffer=unique_filename,
            skiprows=2,
            nrows=nrows,
        )

        eval_io(
            fn_name="read_fwf",
            # read_fwf kwargs
            filepath_or_buffer=unique_filename,
            usecols=[0, 4, 7],
            skiprows=[2, 5],
            nrows=nrows,
        )

    def test_fwf_file_index_col(self, make_fwf_file):
        fwf_data = (
            "a       b           c          d\n"
            + "id8141  360.242940  149.910199 11950.7\n"
            + "id1594  444.953632  166.985655 11788.4\n"
            + "id1849  364.136849  183.628767 11806.2\n"
            + "id1230  413.836124  184.375703 11916.8\n"
            + "id1948  502.953953  173.237159 12468.3\n"
        )
        eval_io(
            fn_name="read_fwf",
            # read_fwf kwargs
            filepath_or_buffer=make_fwf_file(fwf_data=fwf_data),
            index_col="c",
        )

    def test_fwf_file_skipfooter(self, make_fwf_file):
        eval_io(
            fn_name="read_fwf",
            # read_fwf kwargs
            filepath_or_buffer=make_fwf_file(),
            skipfooter=2,
        )

    def test_fwf_file_parse_dates(self, make_fwf_file):
        dates = pandas.date_range("2000", freq="h", periods=10)
        fwf_data = "col1 col2        col3 col4"
        for i in range(10, 20):
            fwf_data = fwf_data + "\n{col1}   {col2}  {col3}   {col4}".format(
                col1=str(i),
                col2=str(dates[i - 10].date()),
                col3=str(i),
                col4=str(dates[i - 10].time()),
            )
        unique_filename = make_fwf_file(fwf_data=fwf_data)

        eval_io(
            fn_name="read_fwf",
            # read_fwf kwargs
            filepath_or_buffer=unique_filename,
            parse_dates=[["col2", "col4"]],
        )

        eval_io(
            fn_name="read_fwf",
            # read_fwf kwargs
            filepath_or_buffer=unique_filename,
            parse_dates={"time": ["col2", "col4"]},
        )

    @pytest.mark.parametrize(
        "read_mode",
        [
            "r",
            "rb",
        ],
    )
    def test_read_fwf_file_handle(self, make_fwf_file, read_mode):
        with open(make_fwf_file(), mode=read_mode) as buffer:
            df_pandas = pandas.read_fwf(buffer)
            buffer.seek(0)
            df_modin = pd.read_fwf(buffer)
            df_equals(df_modin, df_pandas)

    def test_read_fwf_empty_frame(self, make_fwf_file):
        kwargs = {
            "usecols": [0],
            "index_col": 0,
        }
        unique_filename = make_fwf_file()

        modin_df = pd.read_fwf(unique_filename, **kwargs)
        pandas_df = pandas.read_fwf(unique_filename, **kwargs)

        df_equals(modin_df, pandas_df)

    @pytest.mark.parametrize(
        "storage_options",
        [{"anon": False}, {"anon": True}, {"key": "123", "secret": "123"}, None],
    )
    def test_read_fwf_s3(self, storage_options):
        eval_io(
            fn_name="read_fwf",
            filepath_or_buffer="s3://modin-datasets/testing/test_data.fwf",
            storage_options=storage_options,
        )


@pytest.mark.filterwarnings(default_to_pandas_ignore_string)
class TestGbq:
    @pytest.mark.skip(reason="Can not pass without GBQ access")
    def test_read_gbq(self):
        # Test API, but do not supply credentials until credits can be secured.
        with pytest.raises(
            ValueError, match="Could not determine project ID and one was not supplied."
        ):
            pd.read_gbq("SELECT 1")

    @pytest.mark.skip(reason="Can not pass without GBQ access")
    def test_to_gbq(self):
        modin_df, _ = create_test_dfs(TEST_DATA)
        # Test API, but do not supply credentials until credits can be secured.
        with pytest.raises(
            ValueError, match="Could not determine project ID and one was not supplied."
        ):
            modin_df.to_gbq("modin.table")

    def test_read_gbq_mock(self):
        test_args = ("fake_query",)
        test_kwargs = inspect.signature(pd.read_gbq).parameters.copy()
        test_kwargs.update(project_id="test_id", dialect="standart")
        test_kwargs.pop("query", None)
        with mock.patch(
            "pandas.read_gbq", return_value=pandas.DataFrame([])
        ) as read_gbq:
            pd.read_gbq(*test_args, **test_kwargs)
        read_gbq.assert_called_once_with(*test_args, **test_kwargs)


@pytest.mark.filterwarnings(default_to_pandas_ignore_string)
class TestStata:
    def test_read_stata(self, make_stata_file):
        eval_io(
            fn_name="read_stata",
            # read_stata kwargs
            filepath_or_buffer=make_stata_file(),
        )

    def test_to_stata(self, tmp_path):
        modin_df, pandas_df = create_test_dfs(TEST_DATA)
        eval_to_file(
            tmp_path,
            modin_obj=modin_df,
            pandas_obj=pandas_df,
            fn="to_stata",
            extension="stata",
        )


@pytest.mark.filterwarnings(default_to_pandas_ignore_string)
class TestSas:
    def test_read_sas(self):
        eval_io(
            fn_name="read_sas",
            # read_sas kwargs
            filepath_or_buffer="modin/pandas/test/data/airline.sas7bdat",
        )


@pytest.mark.filterwarnings(default_to_pandas_ignore_string)
class TestFeather:
    def test_read_feather(self, make_feather_file):
        eval_io(
            fn_name="read_feather",
            # read_feather kwargs
            path=make_feather_file(),
        )

    @pytest.mark.parametrize(
        "dtype_backend", [lib.no_default, "numpy_nullable", "pyarrow"]
    )
    def test_read_feather_dtype_backend(self, make_feather_file, dtype_backend):
        def comparator(df1, df2):
            df_equals(df1, df2)
            df_equals(df1.dtypes, df2.dtypes)

        eval_io(
            fn_name="read_feather",
            # read_feather kwargs
            path=make_feather_file(),
            dtype_backend=dtype_backend,
            comparator=comparator,
        )

    @pytest.mark.parametrize(
        "storage_options",
        [{"anon": False}, {"anon": True}, {"key": "123", "secret": "123"}, None],
    )
    def test_read_feather_s3(self, storage_options):
        eval_io(
            fn_name="read_feather",
            path="s3://modin-datasets/testing/test_data.feather",
            storage_options=storage_options,
        )

    def test_read_feather_path_object(self, make_feather_file):
        eval_io(
            fn_name="read_feather",
            path=Path(make_feather_file()),
        )

    def test_to_feather(self, tmp_path):
        modin_df, pandas_df = create_test_dfs(TEST_DATA)
        eval_to_file(
            tmp_path,
            modin_obj=modin_df,
            pandas_obj=pandas_df,
            fn="to_feather",
            extension="feather",
        )

    def test_read_feather_with_index_metadata(self, tmp_path):
        # see: https://github.com/modin-project/modin/issues/6212
        df = pandas.DataFrame({"a": [1, 2, 3]}, index=[0, 1, 2])
        assert not isinstance(df.index, pandas.RangeIndex)

        path = get_unique_filename(extension=".feather", data_dir=tmp_path)
        df.to_feather(path)
        eval_io(
            fn_name="read_feather",
            path=path,
        )


@pytest.mark.filterwarnings(default_to_pandas_ignore_string)
class TestClipboard:
    @pytest.mark.skip(reason="No clipboard in CI")
    def test_read_clipboard(self):
        setup_clipboard()

        eval_io(fn_name="read_clipboard")

    @pytest.mark.skip(reason="No clipboard in CI")
    def test_to_clipboard(self):
        modin_df, pandas_df = create_test_dfs(TEST_DATA)

        modin_df.to_clipboard()
        modin_as_clip = pandas.read_clipboard()

        pandas_df.to_clipboard()
        pandas_as_clip = pandas.read_clipboard()

        assert modin_as_clip.equals(pandas_as_clip)


@pytest.mark.filterwarnings(default_to_pandas_ignore_string)
class TestPickle:
    def test_read_pickle(self, make_pickle_file):
        eval_io(
            fn_name="read_pickle",
            # read_pickle kwargs
            filepath_or_buffer=make_pickle_file(),
        )

    def test_to_pickle(self, tmp_path):
        modin_df, pandas_df = create_test_dfs(TEST_DATA)
        eval_to_file(
            tmp_path,
            modin_obj=modin_df,
            pandas_obj=pandas_df,
            fn="to_pickle",
            extension="pkl",
        )


@pytest.mark.filterwarnings(default_to_pandas_ignore_string)
class TestXml:
    def test_read_xml(self):
        # example from pandas
        data = """<?xml version='1.0' encoding='utf-8'?>
<data xmlns="http://example.com">
 <row>
   <shape>square</shape>
   <degrees>360</degrees>
   <sides>4.0</sides>
 </row>
 <row>
   <shape>circle</shape>
   <degrees>360</degrees>
   <sides/>
 </row>
"""
        eval_io("read_xml", path_or_buffer=data)


@pytest.mark.filterwarnings(default_to_pandas_ignore_string)
class TestOrc:
    # It's not easy to add infrastructure for `orc` format.
    # In case of defaulting to pandas, it's enough
    # to check that the parameters are passed to pandas.
    def test_read_orc(self):
        test_args = ("fake_path",)
        test_kwargs = dict(
            columns=["A"],
            dtype_backend=lib.no_default,
            fake_kwarg="some_pyarrow_parameter",
        )
        with mock.patch(
            "pandas.read_orc", return_value=pandas.DataFrame([])
        ) as read_orc:
            pd.read_orc(*test_args, **test_kwargs)
        read_orc.assert_called_once_with(*test_args, **test_kwargs)


@pytest.mark.filterwarnings(default_to_pandas_ignore_string)
class TestSpss:
    # It's not easy to add infrastructure for `spss` format.
    # In case of defaulting to pandas, it's enough
    # to check that the parameters are passed to pandas.
    def test_read_spss(self):
        test_args = ("fake_path",)
        test_kwargs = dict(
            usecols=["A"], convert_categoricals=False, dtype_backend=lib.no_default
        )
        with mock.patch(
            "pandas.read_spss", return_value=pandas.DataFrame([])
        ) as read_spss:
            pd.read_spss(*test_args, **test_kwargs)
        read_spss.assert_called_once_with(*test_args, **test_kwargs)


@pytest.mark.filterwarnings(default_to_pandas_ignore_string)
def test_json_normalize():
    # example from pandas
    data = [
        {"id": 1, "name": {"first": "Coleen", "last": "Volk"}},
        {"name": {"given": "Mark", "family": "Regner"}},
        {"id": 2, "name": "Faye Raker"},
    ]
    eval_io("json_normalize", data=data)


@pytest.mark.filterwarnings(default_to_pandas_ignore_string)
def test_from_arrow():
    _, pandas_df = create_test_dfs(TEST_DATA)
    modin_df = from_arrow(pa.Table.from_pandas(pandas_df))
    df_equals(modin_df, pandas_df)


@pytest.mark.filterwarnings(default_to_pandas_ignore_string)
def test_from_spmatrix():
    data = sparse.eye(3)
    with pytest.warns(UserWarning, match="defaulting to pandas.*"):
        modin_df = pd.DataFrame.sparse.from_spmatrix(data)
    pandas_df = pandas.DataFrame.sparse.from_spmatrix(data)
    df_equals(modin_df, pandas_df)


@pytest.mark.filterwarnings(default_to_pandas_ignore_string)
def test_to_dense():
    data = {"col1": pandas.arrays.SparseArray([0, 1, 0])}
    modin_df, pandas_df = create_test_dfs(data)
    df_equals(modin_df.sparse.to_dense(), pandas_df.sparse.to_dense())


@pytest.mark.filterwarnings(default_to_pandas_ignore_string)
def test_to_dict_dataframe():
    modin_df, _ = create_test_dfs(TEST_DATA)
    assert modin_df.to_dict() == to_pandas(modin_df).to_dict()


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({}, id="no_kwargs"),
        pytest.param({"into": dict}, id="into_dict"),
        pytest.param({"into": OrderedDict}, id="into_ordered_dict"),
        pytest.param({"into": defaultdict(list)}, id="into_defaultdict"),
    ],
)
@pytest.mark.filterwarnings(default_to_pandas_ignore_string)
def test_to_dict_series(kwargs):
    eval_general(
        *[df.iloc[:, 0] for df in create_test_dfs(utils_test_data["int_data"])],
        lambda df: df.to_dict(**kwargs),
        # TODO(https://github.com/modin-project/modin/issues/6016): fix eval_general
        # and remove this raising_exceptions
        raising_exceptions=(Exception,),
    )


@pytest.mark.filterwarnings(default_to_pandas_ignore_string)
def test_to_latex():
    modin_df, _ = create_test_dfs(TEST_DATA)
    assert modin_df.to_latex() == to_pandas(modin_df).to_latex()


@pytest.mark.filterwarnings(default_to_pandas_ignore_string)
def test_to_period():
    index = pandas.DatetimeIndex(
        pandas.date_range("2000", freq="h", periods=len(TEST_DATA["col1"]))
    )
    modin_df, pandas_df = create_test_dfs(TEST_DATA, index=index)
    df_equals(modin_df.to_period(), pandas_df.to_period())
