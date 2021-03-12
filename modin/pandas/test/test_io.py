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

import pytest
import numpy as np
import pandas
from pandas.errors import ParserWarning
from collections import OrderedDict
from modin.config import TestDatasetSize
from modin.utils import to_pandas
from modin.pandas.utils import from_arrow
import pyarrow as pa
import os
import shutil
import sqlalchemy as sa
import csv
import tempfile

from .utils import (
    check_file_leaks,
    df_equals,
    json_short_string,
    json_short_bytes,
    json_long_string,
    json_long_bytes,
    eval_io,
    get_unique_filename,
    io_ops_bad_exc,
    eval_io_from_str,
    dummy_decorator,
    create_test_dfs,
    COMP_TO_EXT,
    teardown_test_files,
    generate_dataframe,
)

from modin.config import Engine, Backend, IsExperimental

if Backend.get() == "Pandas":
    import modin.pandas as pd
else:
    import modin.experimental.pandas as pd
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


def assert_files_eq(path1, path2):
    with open(path1, "rb") as file1, open(path2, "rb") as file2:
        file1_content = file1.read()
        file2_content = file2.read()

        if file1_content == file2_content:
            return True
        else:
            return False


def setup_json_file(filename, row_size=NROWS, force=True):
    if os.path.exists(filename) and not force:
        pass
    else:
        df = pandas.DataFrame(
            {"col1": np.arange(row_size), "col2": np.arange(row_size)}
        )
        df.to_json(filename)


def setup_json_lines_file(filename, row_size=NROWS, force=True):
    if os.path.exists(filename) and not force:
        pass
    else:
        df = pandas.DataFrame(
            {"col1": np.arange(row_size), "col2": np.arange(row_size)}
        )
        df.to_json(filename, lines=True, orient="records")


def setup_html_file(filename, row_size=NROWS, force=True):
    if os.path.exists(filename) and not force:
        pass
    else:
        df = pandas.DataFrame(
            {"col1": np.arange(row_size), "col2": np.arange(row_size)}
        )
        df.to_html(filename)


def setup_clipboard(row_size=NROWS):
    df = pandas.DataFrame({"col1": np.arange(row_size), "col2": np.arange(row_size)})
    df.to_clipboard()


def setup_excel_file(filename, row_size=NROWS, force=True):
    if os.path.exists(filename) and not force:
        pass
    else:
        df = pandas.DataFrame(
            {"col1": np.arange(row_size), "col2": np.arange(row_size)}
        )
        df.to_excel(filename)


def setup_feather_file(filename, row_size=NROWS, force=True):
    if os.path.exists(filename) and not force:
        pass
    else:
        df = pandas.DataFrame(
            {"col1": np.arange(row_size), "col2": np.arange(row_size)}
        )
        df.to_feather(filename)


def setup_hdf_file(filename, row_size=NROWS, force=True, format=None):
    if os.path.exists(filename) and not force:
        pass
    else:
        df = pandas.DataFrame(
            {"col1": np.arange(row_size), "col2": np.arange(row_size)}
        )
        df.to_hdf(filename, key="df", format=format)


def setup_stata_file(filename, row_size=NROWS, force=True):
    if os.path.exists(filename) and not force:
        pass
    else:
        df = pandas.DataFrame(
            {"col1": np.arange(row_size), "col2": np.arange(row_size)}
        )
        df.to_stata(filename)


def setup_pickle_file(filename, row_size=NROWS, force=True):
    if os.path.exists(filename) and not force:
        pass
    else:
        df = pandas.DataFrame(
            {"col1": np.arange(row_size), "col2": np.arange(row_size)}
        )
        df.to_pickle(filename)


def setup_fwf_file(filename, force=True, fwf_data=None):
    if not force and os.path.exists(filename):
        return

    if fwf_data is None:
        fwf_data = """ACW000116041961TAVG -142  k  183  k  419  k  720  k 1075  k 1546  k 1517  k 1428  k 1360  k 1121  k  457  k  -92  k
ACW000116041962TAVG   60  k   32  k -207  k  582  k  855  k 1328  k 1457  k 1340  k 1110  k  941  k  270  k -179  k
ACW000116041963TAVG -766  k -606  k -152  k  488  k 1171  k 1574  k 1567  k 1543  k 1279  k  887  k  513  k -161  k
ACW000116041964TAVG    9  k -138  k    2  k  685  k 1166  k 1389  k 1453  k 1504  k 1168  k  735  k  493  k   59  k
ACW000116041965TAVG   -9  k -158  k  -15  k  537  k  934  k 1447  k 1434  k 1424  k 1324  k  921  k  -22  k -231  k
ACW000116041966TAVG -490  k -614  k  108  k  246  k 1082  k 1642  k 1620  k 1471  k 1195  k  803  k  329  k    2  k
ACW000116041967TAVG -270  k   36  k  397  k  481  k 1052  k 1373  k 1655  k 1598  k 1318  k  997  k  559  k  -96  k
ACW000116041968TAVG -306  k -183  k  220  k  714  k  935  k 1635  k 1572  k 1718  k 1331  k  781  k  180  k  -56  k
ACW000116041969TAVG -134  k -494  k -185  k  497  k  962  k 1634  k 1687  k 1773  k 1379  k  932  k  321  k -275  k
ACW000116041970TAVG -483  k -704  k  -75  k  261  k 1093  k 1724  k 1470  k 1609  k 1163  k  836  k  300  k   73  k
ACW000116041971TAVG   -6  k   83  k  -40  k  472  k 1180  k 1411  k 1700  k 1600  k 1165  k  908  k  361  k  383  k
ACW000116041972TAVG -377  k   -4  k  250  k  556  k 1117  k 1444  k 1778  k 1545  k 1073  k  797  k  481  k  404  k
ACW000116041973TAVG   61  k  169  k  453  k  472  k 1075  k 1545  k 1866  k 1579  k 1199  k  563  k  154  k   11  k
ACW000116041974TAVG  191  k  209  k  339  k  748  k 1094  k 1463  k 1498  k 1541  k 1319  k  585  k  428  k  335  k
ACW000116041975TAVG  346  k   88  k  198  k  488  k 1165  k 1483  k 1756  k 1906  k 1374  k  845  k  406  k  387  k
ACW000116041976TAVG -163  k  -62  k -135  k  502  k 1128  k 1461  k 1822  k 1759  k 1136  k  715  k  458  k -205  k
ACW000116041977TAVG -192  k -279  k  234  k  332  k 1128  k 1566  k 1565  k 1556  k 1126  k  949  k  421  k  162  k
ACW000116041978TAVG   55  k -354  k   66  k  493  k 1155  k 1552  k 1564  k 1555  k 1061  k  932  k  688  k -464  k
ACW000116041979TAVG -618  k -632  k   35  k  474  k  993  k 1566  k 1484  k 1483  k 1229  k  647  k  412  k  -40  k
ACW000116041980TAVG -340  k -500  k  -35  k  524  k 1071  k 1534  k 1655  k 1502  k 1269  k  660  k  138  k  125  k"""

    with open(filename, "w") as f:
        f.write(fwf_data)


def eval_to_file(modin_obj, pandas_obj, fn, extension, **fn_kwargs):
    """Helper function to test `to_<extension>` methods.

    Args:
        modin_obj: Modin DataFrame or Series to test `to_<extension>` method.
        pandas_obj: Pandas DataFrame or Series to test `to_<extension>` method.
        fn: name of the method, that should be tested.
        extension: Extension of the test file.
    """
    unique_filename_modin = get_unique_filename(extension=extension)
    unique_filename_pandas = get_unique_filename(extension=extension)

    try:
        getattr(modin_obj, fn)(unique_filename_modin, **fn_kwargs)
        getattr(pandas_obj, fn)(unique_filename_pandas, **fn_kwargs)

        assert assert_files_eq(unique_filename_modin, unique_filename_pandas)
    finally:
        teardown_test_files([unique_filename_modin, unique_filename_pandas])


@pytest.mark.usefixtures("TestReadCSVFixture")
@pytest.mark.skipif(
    IsExperimental.get() and Backend.get() == "Pyarrow",
    reason="Segmentation fault; see PR #2347 ffor details",
)
class TestCsv:
    # delimiter tests
    @pytest.mark.parametrize("sep", [None, "_", ",", ".", "\n"])
    @pytest.mark.parametrize("delimiter", ["_", ",", ".", "\n"])
    @pytest.mark.parametrize("decimal", [".", "_"])
    @pytest.mark.parametrize("thousands", [None, ",", "_", " "])
    def test_read_csv_delimiters(
        self, make_csv_file, sep, delimiter, decimal, thousands
    ):
        unique_filename = get_unique_filename()
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

    # Column and Index Locations and Names tests
    @pytest.mark.skipif(
        Engine.get() != "Python",
        reason="many parameters combiantions fails: issue #2312, #2307",
    )
    @pytest.mark.parametrize("header", ["infer", None, 0])
    @pytest.mark.parametrize("index_col", [None, "col1"])
    @pytest.mark.parametrize("prefix", [None, "_", "col"])
    @pytest.mark.parametrize(
        "names", [None, ["col1"], ["c1", "c2", "c3", "c4", "c5", "c6", "c7"]]
    )
    @pytest.mark.parametrize(
        "usecols", [None, ["col1"], ["col1", "col2", "col6"], [0, 1, 5]]
    )
    @pytest.mark.parametrize("skip_blank_lines", [True, False])
    def test_read_csv_col_handling(
        self,
        request,
        header,
        index_col,
        prefix,
        names,
        usecols,
        skip_blank_lines,
    ):
        if request.config.getoption("--simulate-cloud").lower() != "off":
            pytest.skip(
                "The reason of tests fail in `cloud` mode is unknown for now - issue #2340"
            )
        eval_io(
            fn_name="read_csv",
            # read_csv kwargs
            filepath_or_buffer=pytest.csvs_names["test_read_csv_blank_lines"],
            header=header,
            index_col=index_col,
            prefix=prefix,
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
        request,
        dtype,
        engine,
        converters,
        skipfooter,
    ):
        if request.config.getoption("--simulate-cloud").lower() != "off":
            pytest.skip(
                "The reason of tests fail in `cloud` mode is unknown for now - issue #2340"
            )

        if dtype:
            dtype = {
                col: "object"
                for col in pandas.read_csv(
                    pytest.csvs_names["test_read_csv_regular"], nrows=1
                ).columns
            }

        eval_io(
            fn_name="read_csv",
            check_exception_type=None,  # issue #2320
            raising_exceptions=None,
            check_kwargs_callable=not callable(converters),
            # read_csv kwargs
            filepath_or_buffer=pytest.csvs_names["test_read_csv_regular"],
            dtype=dtype,
            engine=engine,
            converters=converters,
            skipfooter=skipfooter,
        )

    @pytest.mark.parametrize("true_values", [["Yes"], ["Yes", "true"], None])
    @pytest.mark.parametrize("false_values", [["No"], ["No", "false"], None])
    @pytest.mark.parametrize("skiprows", [2, lambda x: x % 2])
    @pytest.mark.parametrize("skipfooter", [0, 10])
    @pytest.mark.parametrize("nrows", [35, None])
    @pytest.mark.parametrize(
        "names",
        [
            pytest.param(
                ["c1", "c2", "c3", "c4"],
                marks=pytest.mark.xfail(reason="Excluded because of the issue #2845"),
            ),
            None,
        ],
    )
    def test_read_csv_parsing_2(
        self,
        request,
        true_values,
        false_values,
        skiprows,
        skipfooter,
        nrows,
        names,
    ):
        if false_values or true_values and Engine.get() != "Python":
            pytest.xfail("modin and pandas dataframes differs - issue #2446")
        if request.config.getoption("--simulate-cloud").lower() != "off":
            pytest.skip(
                "The reason of tests fail in `cloud` mode is unknown for now - issue #2340"
            )

        eval_io(
            fn_name="read_csv",
            check_exception_type=None,  # issue #2320
            raising_exceptions=None,
            check_kwargs_callable=not callable(skiprows),
            # read_csv kwargs
            filepath_or_buffer=pytest.csvs_names["test_read_csv_yes_no"],
            true_values=true_values,
            false_values=false_values,
            skiprows=skiprows,
            skipfooter=skipfooter,
            nrows=nrows,
            names=names,
        )

    def test_read_csv_skipinitialspace(self):
        unique_filename = get_unique_filename()
        str_initial_spaces = (
            "col1,col2,col3,col4\n"
            "five,  six,  seven,  eight\n"
            "    five,    six,    seven,    eight\n"
            "five, six,  seven,   eight\n"
        )

        eval_io_from_str(str_initial_spaces, unique_filename, skipinitialspace=True)

    @pytest.mark.xfail(reason="infinite recursion error - issue #2032")
    @pytest.mark.parametrize(
        "test_case", ["single_element", "single_column", "multiple_columns"]
    )
    def test_read_csv_squeeze(self, test_case):
        unique_filename = get_unique_filename()

        str_single_element = "1"
        str_single_col = "1\n2\n3\n"
        str_four_cols = "1, 2, 3, 4\n5, 6, 7, 8\n9, 10, 11, 12\n"
        case_to_data = {
            "single_element": str_single_element,
            "single_column": str_single_col,
            "multiple_columns": str_four_cols,
        }

        eval_io_from_str(case_to_data[test_case], unique_filename, squeeze=True)
        eval_io_from_str(
            case_to_data[test_case], unique_filename, header=None, squeeze=True
        )

    def test_read_csv_mangle_dupe_cols(self):
        unique_filename = get_unique_filename()
        str_non_unique_cols = "col,col,col,col\n5, 6, 7, 8\n9, 10, 11, 12\n"
        eval_io_from_str(str_non_unique_cols, unique_filename, mangle_dupe_cols=True)

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
        "parse_dates",
        [
            True,
            False,
            ["col2"],
            ["col2", "col4"],
            [1, 3],
            pytest.param(
                {"foo": ["col2", "col4"]},
                marks=pytest.mark.xfail(
                    Engine.get() != "Python",
                    reason="Exception: Internal Error - issue #2073",
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("infer_datetime_format", [True, False])
    @pytest.mark.parametrize("keep_date_col", [True, False])
    @pytest.mark.parametrize(
        "date_parser", [None, lambda x: pandas.datetime.strptime(x, "%Y-%m-%d")]
    )
    @pytest.mark.parametrize("dayfirst", [True, False])
    @pytest.mark.parametrize("cache_dates", [True, False])
    def test_read_csv_datetime(
        self,
        request,
        parse_dates,
        infer_datetime_format,
        keep_date_col,
        date_parser,
        dayfirst,
        cache_dates,
    ):
        if request.config.getoption("--simulate-cloud").lower() != "off":
            pytest.skip(
                "The reason of tests fail in `cloud` mode is unknown for now - issue #2340"
            )

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

    # Quoting, Compression, and File Format parameters tests
    @pytest.mark.parametrize("compression", ["infer", "gzip", "bz2", "xz", "zip"])
    @pytest.mark.parametrize(
        "encoding",
        [None, "latin8", "ISO-8859-1", "latin1", "iso-8859-1", "cp1252", "utf8"],
    )
    @pytest.mark.parametrize("engine", [None, "python", "c"])
    def test_read_csv_compression(self, make_csv_file, compression, encoding, engine):
        unique_filename = get_unique_filename()
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

    @pytest.mark.parametrize("thousands", [None, ",", "_", " "])
    @pytest.mark.parametrize("decimal", [".", "_"])
    @pytest.mark.parametrize("lineterminator", [None, "x", "\n"])
    @pytest.mark.parametrize("escapechar", [None, "d", "x"])
    @pytest.mark.parametrize("dialect", ["test_csv_dialect", None])
    def test_read_csv_file_format(
        self,
        request,
        make_csv_file,
        thousands,
        decimal,
        lineterminator,
        escapechar,
        dialect,
    ):
        if request.config.getoption("--simulate-cloud").lower() != "off":
            pytest.skip(
                "The reason of tests fail in `cloud` mode is unknown for now - issue #2340"
            )
        elif Engine.get() != "Python" and lineterminator == "x":
            pytest.xfail("read_csv with Ray engine outputs empty frame - issue #2493")
        elif Engine.get() != "Python" and escapechar:
            pytest.xfail(
                "read_csv with Ray engine fails with some 'escapechar' parameters - issue #2494"
            )
        elif Engine.get() != "Python" and dialect:
            pytest.xfail(
                "read_csv with Ray engine fails with `dialect` parameter - issue #2508"
            )

        unique_filename = get_unique_filename()
        if dialect:
            test_csv_dialect_params = {
                "delimiter": "_",
                "doublequote": False,
                "escapechar": "\\",
                "quotechar": "d",
                "quoting": csv.QUOTE_ALL,
            }
            csv.register_dialect(dialect, **test_csv_dialect_params)
            dialect = csv.get_dialect(dialect)
            make_csv_file(filename=unique_filename, **test_csv_dialect_params)
        else:
            make_csv_file(
                filename=unique_filename,
                thousands_separator=thousands,
                decimal_separator=decimal,
                escapechar=escapechar,
                line_terminator=lineterminator,
            )

        eval_io(
            check_exception_type=None,  # issue #2320
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
        unique_filename = get_unique_filename()

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
    @pytest.mark.xfail(
        Engine.get() != "Python",
        reason="read_csv with Ray engine doen't raise `bad lines` exceptions - issue #2500",
    )
    @pytest.mark.parametrize("warn_bad_lines", [True, False])
    @pytest.mark.parametrize("error_bad_lines", [True, False])
    def test_read_csv_error_handling(
        self,
        warn_bad_lines,
        error_bad_lines,
    ):
        eval_io(
            fn_name="read_csv",
            # read_csv kwargs
            filepath_or_buffer=pytest.csvs_names["test_read_csv_bad_lines"],
            warn_bad_lines=warn_bad_lines,
            error_bad_lines=error_bad_lines,
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
        if Engine.get() != "Python" and delimiter == " ":
            pytest.xfail(
                "read_csv with Ray engine doesn't \
                raise exceptions while Pandas raises - issue #2320"
            )

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

        unique_filename = get_unique_filename()

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

        unique_filename = get_unique_filename()

        eval_io_from_str(csv_bad_quotes, unique_filename, nrows=nrows)

    def test_read_csv_categories(self, request):
        if request.config.getoption("--simulate-cloud").lower() != "off":
            pytest.skip(
                "The reason of tests fail in `cloud` mode is unknown for now - issue #2340"
            )
        eval_io(
            fn_name="read_csv",
            # read_csv kwargs
            filepath_or_buffer="modin/pandas/test/data/test_categories.csv",
            names=["one", "two"],
            dtype={"one": "int64", "two": "category"},
        )

    @pytest.mark.parametrize("encoding", [None, "utf-8"])
    @pytest.mark.parametrize("parse_dates", [False, ["timestamp"]])
    @pytest.mark.parametrize("index_col", [None, 0, 2])
    @pytest.mark.parametrize("header", ["infer", 0])
    @pytest.mark.parametrize(
        "names",
        [
            None,
            ["timestamp", "symbol", "high", "low", "open", "close", "spread", "volume"],
        ],
    )
    def test_read_csv_parse_dates(
        self, request, names, header, index_col, parse_dates, encoding
    ):
        if (
            parse_dates
            and request.config.getoption("--simulate-cloud").lower() != "off"
        ):
            pytest.skip(
                "The reason of tests fail in `cloud` mode is unknown for now - issue #2340"
            )

        if names is not None and header == "infer":
            pytest.xfail(
                "read_csv with Ray engine works incorrectly with date data and names parameter provided - issue #2509"
            )

        eval_io(
            fn_name="read_csv",
            # read_csv kwargs
            filepath_or_buffer="modin/pandas/test/data/test_time_parsing.csv",
            names=names,
            header=header,
            index_col=index_col,
            parse_dates=parse_dates,
            encoding=encoding,
        )

    @pytest.mark.skipif(Engine.get() == "Python", reason="Using pandas implementation")
    def test_read_csv_s3(self):
        eval_io(
            fn_name="read_csv",
            # read_csv kwargs
            filepath_or_buffer="s3://noaa-ghcn-pds/csv/1788.csv",
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

    def test_read_csv_default_to_pandas(self, request):
        if request.config.getoption("--simulate-cloud").lower() != "off":
            pytest.skip(
                "The reason of tests fail in `cloud` mode is unknown for now - issue #2340"
            )
        with pytest.warns(UserWarning):
            # This tests that we default to pandas on a buffer
            from io import StringIO

            pd.read_csv(
                StringIO(open(pytest.csvs_names["test_read_csv_regular"], "r").read())
            )

        with pytest.warns(UserWarning):
            pd.read_csv(
                pytest.csvs_names["test_read_csv_regular"],
                skiprows=lambda x: x in [0, 2],
            )

    def test_read_csv_default_to_pandas_url(self, request):
        if request.config.getoption("--simulate-cloud").lower() != "off":
            pytest.skip(
                "The reason of tests fail in `cloud` mode is unknown for now - issue #2340"
            )
        # We haven't implemented read_csv from https, but if it's implemented, then this needs to change
        eval_io(
            fn_name="read_csv",
            modin_warning=UserWarning,
            # read_csv kwargs
            filepath_or_buffer="https://raw.githubusercontent.com/modin-project/modin/master/modin/pandas/test/data/blah.csv",
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
            cast_to_str=True,
        )

    def test_read_csv_sep_none(self, request):
        if request.config.getoption("--simulate-cloud").lower() != "off":
            pytest.skip(
                "The reason of tests fail in `cloud` mode is unknown for now - issue #2340"
            )
        eval_io(
            fn_name="read_csv",
            modin_warning=ParserWarning,
            # read_csv kwargs
            filepath_or_buffer=pytest.csvs_names["test_read_csv_regular"],
            sep=None,
        )

    def test_read_csv_incorrect_data(self, request):
        if request.config.getoption("--simulate-cloud").lower() != "off":
            pytest.skip(
                "The reason of tests fail in `cloud` mode is unknown for now - issue #2340"
            )
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
    def test_read_csv_names_neq_num_cols(self, request, kwargs):
        if request.config.getoption("--simulate-cloud").lower() != "off":
            pytest.skip(
                "The reason of tests fail in `cloud` mode is unknown for now - issue #2340"
            )
        eval_io(
            fn_name="read_csv",
            # read_csv kwargs
            filepath_or_buffer="modin/pandas/test/data/issue_2074.csv",
            **kwargs,
        )

    @pytest.mark.parametrize("header", [False, True])
    @pytest.mark.parametrize("mode", ["w", "wb+"])
    def test_to_csv(self, request, header, mode):
        if request.config.getoption("--simulate-cloud").lower() != "off":
            pytest.skip(
                "The reason of tests fail in `cloud` mode is unknown for now - issue #2340"
            )

        pandas_df = generate_dataframe()
        modin_df = pd.DataFrame(pandas_df)

        eval_to_file(
            modin_obj=modin_df,
            pandas_obj=pandas_df,
            fn="to_csv",
            extension="csv",
            header=header,
            mode=mode,
        )

    def test_dataframe_to_csv(self, request):
        if request.config.getoption("--simulate-cloud").lower() != "off":
            pytest.skip(
                "The reason of tests fail in `cloud` mode is unknown for now - issue #2340"
            )
        pandas_df = pandas.read_csv(pytest.csvs_names["test_read_csv_regular"])
        modin_df = pd.DataFrame(pandas_df)
        eval_to_file(
            modin_obj=modin_df, pandas_obj=pandas_df, fn="to_csv", extension="csv"
        )

    def test_series_to_csv(self, request):
        if request.config.getoption("--simulate-cloud").lower() != "off":
            pytest.skip(
                "The reason of tests fail in `cloud` mode is unknown for now - issue #2340"
            )
        pandas_s = pandas.read_csv(
            pytest.csvs_names["test_read_csv_regular"], usecols=["col1"]
        ).squeeze()
        modin_s = pd.Series(pandas_s)
        eval_to_file(
            modin_obj=modin_s, pandas_obj=pandas_s, fn="to_csv", extension="csv"
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

        df_equals(modin_df, pandas_df)

    @pytest.mark.parametrize("read_mode", ["r", "rb"])
    def test_read_csv_file_handle(self, request, read_mode, make_csv_file):
        if request.config.getoption("--simulate-cloud").lower() != "off":
            pytest.skip("Cannot pickle file handles. See comments in PR #2625")

        unique_filename = get_unique_filename()
        make_csv_file(filename=unique_filename)

        try:
            with open(unique_filename, mode=read_mode) as buffer:
                df_pandas = pandas.read_csv(buffer)
                buffer.seek(0)
                df_modin = pd.read_csv(buffer)
                df_equals(df_modin, df_pandas)
        finally:
            teardown_test_files([unique_filename])


class TestTable:
    def test_read_table(self, make_csv_file):
        unique_filename = get_unique_filename()
        make_csv_file(filename=unique_filename, delimiter="\t")
        eval_io(
            fn_name="read_table",
            # read_table kwargs
            filepath_or_buffer=unique_filename,
        )

    def test_read_table_within_decorator(self, make_csv_file):
        unique_filename = get_unique_filename()
        make_csv_file(filename=unique_filename, delimiter="\t")

        @dummy_decorator()
        def wrapped_read_table(file, method):
            if method == "pandas":
                return pandas.read_table(file)

            if method == "modin":
                return pd.read_table(file)

        pandas_df = wrapped_read_table(unique_filename, method="pandas")
        modin_df = wrapped_read_table(unique_filename, method="modin")

        df_equals(modin_df, pandas_df)


class TestParquet:
    @pytest.mark.parametrize("columns", [None, ["col1"]])
    def test_read_parquet(self, make_parquet_file, columns):
        unique_filename = get_unique_filename(extension="parquet")
        make_parquet_file(filename=unique_filename)

        eval_io(
            fn_name="read_parquet",
            # read_parquet kwargs
            path=unique_filename,
            columns=columns,
        )

    @pytest.mark.parametrize("columns", [None, ["col1"]])
    def test_read_parquet_directory(self, make_parquet_file, columns):  #

        unique_filename = get_unique_filename(extension=None)
        make_parquet_file(filename=unique_filename, directory=True)
        eval_io(
            fn_name="read_parquet",
            # read_parquet kwargs
            path=unique_filename,
            columns=columns,
        )

    @pytest.mark.parametrize("columns", [None, ["col1"]])
    def test_read_parquet_partitioned_directory(self, make_parquet_file, columns):
        unique_filename = get_unique_filename(extension=None)
        make_parquet_file(filename=unique_filename, partitioned_columns=["col1"])

        eval_io(
            fn_name="read_parquet",
            # read_parquet kwargs
            path=unique_filename,
            columns=columns,
        )

    def test_read_parquet_pandas_index(self):
        # Ensure modin can read parquet files written by pandas with a non-RangeIndex object
        unique_filename = get_unique_filename(extension="parquet")
        pandas_df = pandas.DataFrame(
            {
                "idx": np.random.randint(0, 100_000, size=2000),
                "A": np.random.randint(0, 100_000, size=2000),
                "B": ["a", "b"] * 1000,
                "C": ["c"] * 2000,
            }
        )
        try:
            pandas_df.set_index("idx").to_parquet(unique_filename)
            # read the same parquet using modin.pandas
            df_equals(
                pd.read_parquet(unique_filename), pandas.read_parquet(unique_filename)
            )

            pandas_df.set_index(["idx", "A"]).to_parquet(unique_filename)
            df_equals(
                pd.read_parquet(unique_filename), pandas.read_parquet(unique_filename)
            )
        finally:
            os.remove(unique_filename)

    def test_read_parquet_pandas_index_partitioned(self):
        # Ensure modin can read parquet files written by pandas with a non-RangeIndex object
        unique_filename = get_unique_filename(extension="parquet")
        pandas_df = pandas.DataFrame(
            {
                "idx": np.random.randint(0, 100_000, size=2000),
                "A": np.random.randint(0, 10, size=2000),
                "B": ["a", "b"] * 1000,
                "C": ["c"] * 2000,
            }
        )
        try:
            pandas_df.set_index("idx").to_parquet(unique_filename, partition_cols=["A"])
            # read the same parquet using modin.pandas
            df_equals(
                pd.read_parquet(unique_filename), pandas.read_parquet(unique_filename)
            )
        finally:
            shutil.rmtree(unique_filename)

    def test_read_parquet_hdfs(self):
        eval_io(
            fn_name="read_parquet",
            # read_parquet kwargs
            path="modin/pandas/test/data/hdfs.parquet",
        )

    @pytest.mark.skipif(
        Engine.get() == "Python",
        reason="S3-like path doesn't support in pandas with anonymous credentials. See issue #2301.",
    )
    def test_read_parquet_s3(self):
        import s3fs

        # Pandas currently supports only default credentials for boto therefore
        # we use S3FileSystem with `anon=True` for  to make testing possible.
        dataset_url = "s3://aws-roda-hcls-datalake/chembl_27/chembl_27_public_tissue_dictionary/part-00000-66508102-96fa-4fd9-a0fd-5bc072a74293-c000.snappy.parquet"
        fs = s3fs.S3FileSystem(anon=True)
        pandas_df = pandas.read_parquet(fs.open(dataset_url, "rb"))
        modin_df_s3fs = pd.read_parquet(fs.open(dataset_url, "rb"))
        df_equals(pandas_df, modin_df_s3fs)

        # Modin supports default and anonymous credentials and resolves this internally.
        modin_df_s3 = pd.read_parquet(dataset_url)
        df_equals(pandas_df, modin_df_s3)

    def test_read_parquet_without_metadata(self):
        """Test that Modin can read parquet files not written by pandas."""
        from pyarrow import csv
        from pyarrow import parquet

        parquet_fname = get_unique_filename(extension="parquet")
        csv_fname = get_unique_filename(extension="parquet")
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

        df_equals(pd.read_parquet(parquet_fname), pandas.read_parquet(parquet_fname))

    def test_to_parquet(self):
        modin_df, pandas_df = create_test_dfs(TEST_DATA)
        eval_to_file(
            modin_obj=modin_df,
            pandas_obj=pandas_df,
            fn="to_parquet",
            extension="parquet",
        )

    def test_read_parquet_2462(self):
        test_df = pd.DataFrame(
            {
                "col1": [["ad_1", "ad_2"], ["ad_3"]],
            }
        )

        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/data"
            os.makedirs(path)
            test_df.to_parquet(path + "/part-00000.parquet")
            read_df = pd.read_parquet(path)

            df_equals(test_df, read_df)


class TestJson:
    @pytest.mark.parametrize("lines", [False, True])
    def test_read_json(self, lines):
        unique_filename = get_unique_filename(extension="json")
        try:
            setup_json_file(filename=unique_filename)
            eval_io(
                fn_name="read_json",
                # read_json kwargs
                path_or_buf=unique_filename,
                lines=lines,
            )
        finally:
            teardown_test_files([unique_filename])

    def test_read_json_categories(self):
        eval_io(
            fn_name="read_json",
            # read_json kwargs
            path_or_buf="modin/pandas/test/data/test_categories.json",
            dtype={"one": "int64", "two": "category"},
        )

    @pytest.mark.parametrize(
        "data",
        [json_short_string, json_short_bytes, json_long_string, json_long_bytes],
    )
    def test_read_json_string_bytes(self, data):
        with pytest.warns(UserWarning):
            modin_df = pd.read_json(data)
        # For I/O objects we need to rewind to reuse the same object.
        if hasattr(data, "seek"):
            data.seek(0)
        df_equals(modin_df, pandas.read_json(data))

    def test_to_json(self):
        modin_df, pandas_df = create_test_dfs(TEST_DATA)
        eval_to_file(
            modin_obj=modin_df, pandas_obj=pandas_df, fn="to_json", extension="json"
        )

    @pytest.mark.parametrize("read_mode", ["r", "rb"])
    def test_read_json_file_handle(self, request, read_mode):
        if request.config.getoption("--simulate-cloud").lower() != "off":
            pytest.skip("Cannot pickle file handles. See comments in PR #2625")
        unique_filename = get_unique_filename(extension="json")
        try:
            setup_json_file(filename=unique_filename)
            with open(unique_filename, mode=read_mode) as buf:
                df_pandas = pandas.read_json(buf)
                buf.seek(0)
                df_modin = pd.read_json(buf)
                df_equals(df_pandas, df_modin)
        finally:
            teardown_test_files([unique_filename])


class TestExcel:
    @check_file_leaks
    def test_read_excel(self):
        unique_filename = get_unique_filename(extension="xlsx")
        try:
            setup_excel_file(filename=unique_filename)
            eval_io(
                fn_name="read_excel",
                # read_excel kwargs
                io=unique_filename,
            )
        finally:
            teardown_test_files([unique_filename])

    @check_file_leaks
    def test_read_excel_engine(self):
        unique_filename = get_unique_filename(extension="xlsx")
        try:
            setup_excel_file(filename=unique_filename)
            eval_io(
                fn_name="read_excel",
                modin_warning=UserWarning,
                # read_excel kwargs
                io=unique_filename,
                engine="openpyxl",
            )
        finally:
            teardown_test_files([unique_filename])

    @check_file_leaks
    def test_read_excel_index_col(self):
        unique_filename = get_unique_filename(extension="xlsx")
        try:
            setup_excel_file(filename=unique_filename)

            eval_io(
                fn_name="read_excel",
                modin_warning=UserWarning,
                # read_excel kwargs
                io=unique_filename,
                index_col=0,
            )
        finally:
            teardown_test_files([unique_filename])

    @check_file_leaks
    def test_read_excel_all_sheets(self):
        unique_filename = get_unique_filename(extension="xlsx")
        try:
            setup_excel_file(filename=unique_filename)

            pandas_df = pandas.read_excel(unique_filename, sheet_name=None)
            modin_df = pd.read_excel(unique_filename, sheet_name=None)

            assert isinstance(pandas_df, (OrderedDict, dict))
            assert isinstance(modin_df, type(pandas_df))

            assert pandas_df.keys() == modin_df.keys()

            for key in pandas_df.keys():
                df_equals(modin_df.get(key), pandas_df.get(key))
        finally:
            teardown_test_files([unique_filename])

    @pytest.mark.xfail(
        reason="pandas throws the exception. See pandas issue #39250 for more info"
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
        )

    def test_ExcelFile(self):
        unique_filename = get_unique_filename(extension="xlsx")
        try:
            setup_excel_file(filename=unique_filename)

            modin_excel_file = pd.ExcelFile(unique_filename)
            pandas_excel_file = pandas.ExcelFile(unique_filename)

            df_equals(modin_excel_file.parse(), pandas_excel_file.parse())

            assert modin_excel_file.io == unique_filename
            assert isinstance(modin_excel_file, pd.ExcelFile)
            modin_excel_file.close()
            pandas_excel_file.close()
        finally:
            teardown_test_files([unique_filename])

    @pytest.mark.xfail(strict=False, reason="Flaky test, defaults to pandas")
    def test_to_excel(self):
        modin_df, pandas_df = create_test_dfs(TEST_DATA)

        unique_filename_modin = get_unique_filename(extension="xlsx")
        unique_filename_pandas = get_unique_filename(extension="xlsx")

        modin_writer = pandas.ExcelWriter(unique_filename_modin)
        pandas_writer = pandas.ExcelWriter(unique_filename_pandas)
        try:
            modin_df.to_excel(modin_writer)
            pandas_df.to_excel(pandas_writer)

            modin_writer.save()
            pandas_writer.save()

            assert assert_files_eq(unique_filename_modin, unique_filename_pandas)
        finally:
            teardown_test_files([unique_filename_modin, unique_filename_pandas])


class TestHdf:
    @pytest.mark.skipif(os.name == "nt", reason="Windows not supported")
    @pytest.mark.parametrize("format", [None, "table"])
    def test_read_hdf(self, format):
        unique_filename = get_unique_filename(extension="hdf")
        try:
            setup_hdf_file(filename=unique_filename, format=format)
            eval_io(
                fn_name="read_hdf",
                # read_hdf kwargs
                path_or_buf=unique_filename,
                key="df",
            )
        finally:
            teardown_test_files([unique_filename])

    @pytest.mark.skipif(os.name == "nt", reason="Windows not supported")
    def test_HDFStore(self):
        try:
            unique_filename_modin = get_unique_filename(extension="hdf")
            unique_filename_pandas = get_unique_filename(extension="hdf")
            modin_store = pd.HDFStore(unique_filename_modin)
            pandas_store = pandas.HDFStore(unique_filename_pandas)

            modin_df, pandas_df = create_test_dfs(TEST_DATA)

            modin_store["foo"] = modin_df
            pandas_store["foo"] = pandas_df

            assert assert_files_eq(unique_filename_modin, unique_filename_pandas)
            modin_df = modin_store.get("foo")
            pandas_df = pandas_store.get("foo")
            df_equals(modin_df, pandas_df)

            assert isinstance(modin_store, pd.HDFStore)

            handle, hdf_file = tempfile.mkstemp(suffix=".hdf5", prefix="test_read")
            os.close(handle)
            with pd.HDFStore(hdf_file, mode="w") as store:
                store.append("data/df1", pd.DataFrame(np.random.randn(5, 5)))
                store.append("data/df2", pd.DataFrame(np.random.randn(4, 4)))

            modin_df = pd.read_hdf(hdf_file, key="data/df1", mode="r")
            pandas_df = pandas.read_hdf(hdf_file, key="data/df1", mode="r")
            df_equals(modin_df, pandas_df)
        finally:
            os.unlink(hdf_file)
            teardown_test_files([unique_filename_modin, unique_filename_pandas])


class TestSql:
    def test_read_sql(self, make_sql_connection):
        filename = get_unique_filename(extension="db")
        table = "test_read_sql"
        conn = make_sql_connection(filename, table)
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

        with pytest.warns(UserWarning):
            pd.read_sql_query(query, conn)

        with pytest.warns(UserWarning):
            pd.read_sql_table(table, conn)

        # Test SQLAlchemy engine
        conn = sa.create_engine(conn)
        eval_io(
            fn_name="read_sql",
            # read_sql kwargs
            sql=query,
            con=conn,
        )

        # Test SQLAlchemy Connection
        conn = conn.connect()
        eval_io(
            fn_name="read_sql",
            # read_sql kwargs
            sql=query,
            con=conn,
        )

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
    def test_to_sql(self, make_sql_connection, index):
        table_name = f"test_to_sql_{str(index)}"
        modin_df, pandas_df = create_test_dfs(TEST_DATA)

        # We do not pass the table name so the fixture won't generate a table
        conn = make_sql_connection(f"{table_name}_modin.db")
        modin_df.to_sql(table_name, conn, index=index)
        df_modin_sql = pandas.read_sql(
            table_name, con=conn, index_col="index" if index else None
        )

        # We do not pass the table name so the fixture won't generate a table
        conn = make_sql_connection(f"{table_name}_pandas.db")
        pandas_df.to_sql(table_name, conn, index=index)
        df_pandas_sql = pandas.read_sql(
            table_name, con=conn, index_col="index" if index else None
        )

        assert df_modin_sql.sort_index().equals(df_pandas_sql.sort_index())


class TestHtml:
    @pytest.mark.xfail(reason="read_html is not yet implemented properly - issue #1296")
    def test_read_html(self):
        unique_filename = get_unique_filename(extension="html")
        try:
            setup_html_file(filename=unique_filename)
            eval_io(fn_name="read_html", io=unique_filename)
        finally:
            teardown_test_files([unique_filename])

    def test_to_html(self):
        modin_df, pandas_df = create_test_dfs(TEST_DATA)

        eval_to_file(
            modin_obj=modin_df, pandas_obj=pandas_df, fn="to_html", extension="html"
        )


class TestFwf:
    def test_fwf_file(self):
        fwf_data = (
            "id8141  360.242940  149.910199 11950.7\n"
            "id1594  444.953632  166.985655 11788.4\n"
            "id1849  364.136849  183.628767 11806.2\n"
            "id1230  413.836124  184.375703 11916.8\n"
            "id1948  502.953953  173.237159 12468.3\n"
        )

        unique_filename = get_unique_filename(extension="txt")
        try:
            setup_fwf_file(filename=unique_filename, fwf_data=fwf_data)

            colspecs = [(0, 6), (8, 20), (21, 33), (34, 43)]
            df = pd.read_fwf(
                unique_filename, colspecs=colspecs, header=None, index_col=0
            )
            assert isinstance(df, pd.DataFrame)
        finally:
            teardown_test_files([unique_filename])

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
    def test_fwf_file_colspecs_widths(self, kwargs):
        unique_filename = get_unique_filename(extension="txt")
        try:
            setup_fwf_file(filename=unique_filename)

            modin_df = pd.read_fwf(unique_filename, **kwargs)
            pandas_df = pd.read_fwf(unique_filename, **kwargs)

            df_equals(modin_df, pandas_df)
        finally:
            teardown_test_files([unique_filename])

    @pytest.mark.parametrize("usecols", [["a"], ["a", "b", "d"], [0, 1, 3]])
    def test_fwf_file_usecols(self, usecols):
        fwf_data = (
            "a       b           c          d\n"
            "id8141  360.242940  149.910199 11950.7\n"
            "id1594  444.953632  166.985655 11788.4\n"
            "id1849  364.136849  183.628767 11806.2\n"
            "id1230  413.836124  184.375703 11916.8\n"
            "id1948  502.953953  173.237159 12468.3\n"
        )

        unique_filename = get_unique_filename(extension="txt")
        try:
            setup_fwf_file(filename=unique_filename, fwf_data=fwf_data)

            eval_io(
                fn_name="read_fwf",
                # read_fwf kwargs
                filepath_or_buffer=unique_filename,
                usecols=usecols,
            )
        finally:
            teardown_test_files([unique_filename])

    def test_fwf_file_chunksize(self):
        unique_filename = get_unique_filename(extension="txt")
        try:
            setup_fwf_file(filename=unique_filename)

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
        finally:
            teardown_test_files([unique_filename])

    @pytest.mark.parametrize("nrows", [13, None])
    def test_fwf_file_skiprows(self, nrows):
        unique_filename = get_unique_filename(extension="txt")
        try:
            setup_fwf_file(filename=unique_filename)

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
        finally:
            teardown_test_files([unique_filename])

    def test_fwf_file_index_col(self):
        fwf_data = (
            "a       b           c          d\n"
            "id8141  360.242940  149.910199 11950.7\n"
            "id1594  444.953632  166.985655 11788.4\n"
            "id1849  364.136849  183.628767 11806.2\n"
            "id1230  413.836124  184.375703 11916.8\n"
            "id1948  502.953953  173.237159 12468.3\n"
        )

        unique_filename = get_unique_filename(extension="txt")
        try:
            setup_fwf_file(filename=unique_filename, fwf_data=fwf_data)
            eval_io(
                fn_name="read_fwf",
                # read_fwf kwargs
                filepath_or_buffer=unique_filename,
                index_col="c",
            )
        finally:
            teardown_test_files([unique_filename])

    def test_fwf_file_skipfooter(self):
        unique_filename = get_unique_filename(extension="txt")
        try:
            setup_fwf_file(filename=unique_filename)

            eval_io(
                fn_name="read_fwf",
                # read_fwf kwargs
                filepath_or_buffer=unique_filename,
                skipfooter=2,
            )
        finally:
            teardown_test_files([unique_filename])

    def test_fwf_file_parse_dates(self):
        dates = pandas.date_range("2000", freq="h", periods=10)
        fwf_data = "col1 col2        col3 col4"
        for i in range(10, 20):
            fwf_data = fwf_data + "\n{col1}   {col2}  {col3}   {col4}".format(
                col1=str(i),
                col2=str(dates[i - 10].date()),
                col3=str(i),
                col4=str(dates[i - 10].time()),
            )
        unique_filename = get_unique_filename(extension="txt")
        try:
            setup_fwf_file(filename=unique_filename, fwf_data=fwf_data)

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
        finally:
            teardown_test_files([unique_filename])

    @pytest.mark.parametrize("read_mode", ["r", "rb"])
    def test_read_fwf_file_handle(self, request, read_mode):
        if request.config.getoption("--simulate-cloud").lower() != "off":
            pytest.skip("Cannot pickle file handles. See comments in PR #2625")
        unique_filename = get_unique_filename(extension="txt")
        try:
            setup_fwf_file(filename=unique_filename)

            with open(unique_filename, mode=read_mode) as buffer:
                df_pandas = pandas.read_fwf(buffer)
                buffer.seek(0)
                df_modin = pd.read_fwf(buffer)
                df_equals(df_modin, df_pandas)
        finally:
            teardown_test_files([unique_filename])


class TestGbq:
    @pytest.mark.skip(reason="Need to verify GBQ access")
    def test_read_gbq(self):
        # Test API, but do not supply credentials until credits can be secured.
        with pytest.raises(
            ValueError, match="Could not determine project ID and one was not supplied."
        ):
            pd.read_gbq("SELECT 1")

    @pytest.mark.skip(reason="Need to verify GBQ access")
    def test_to_gbq(self):
        modin_df, _ = create_test_dfs(TEST_DATA)
        # Test API, but do not supply credentials until credits can be secured.
        with pytest.raises(
            ValueError, match="Could not determine project ID and one was not supplied."
        ):
            modin_df.to_gbq("modin.table")


class TestStata:
    def test_read_stata(self):
        unique_filename = get_unique_filename(extension="stata")
        try:
            setup_stata_file(filename=unique_filename)
            eval_io(
                fn_name="read_stata",
                # read_stata kwargs
                filepath_or_buffer=unique_filename,
            )
        finally:
            teardown_test_files([unique_filename])

    def test_to_stata(self):
        modin_df, pandas_df = create_test_dfs(TEST_DATA)
        eval_to_file(
            modin_obj=modin_df, pandas_obj=pandas_df, fn="to_stata", extension="stata"
        )


class TestFeather:
    @pytest.mark.xfail(
        Engine.get() != "Python",
        reason="Excluded because of the issue #2845",
    )
    def test_read_feather(self):
        unique_filename = get_unique_filename(extension="feather")
        try:
            setup_feather_file(filename=unique_filename)

            eval_io(
                fn_name="read_feather",
                # read_feather kwargs
                path=unique_filename,
            )
        finally:
            teardown_test_files([unique_filename])

    def test_to_feather(self):
        modin_df, pandas_df = create_test_dfs(TEST_DATA)
        eval_to_file(
            modin_obj=modin_df,
            pandas_obj=pandas_df,
            fn="to_feather",
            extension="feather",
        )


class TestClipboard:
    @pytest.mark.skip(reason="No clipboard on Travis")
    def test_read_clipboard(self):
        setup_clipboard()

        eval_io(fn_name="read_clipboard")

    @pytest.mark.skip(reason="No clipboard on Travis")
    def test_to_clipboard(self):
        modin_df, pandas_df = create_test_dfs(TEST_DATA)

        modin_df.to_clipboard()
        modin_as_clip = pandas.read_clipboard()

        pandas_df.to_clipboard()
        pandas_as_clip = pandas.read_clipboard()

        assert modin_as_clip.equals(pandas_as_clip)


class TestPickle:
    def test_read_pickle(self):
        unique_filename = get_unique_filename(extension="pkl")
        try:
            setup_pickle_file(filename=unique_filename)

            eval_io(
                fn_name="read_pickle",
                # read_pickle kwargs
                filepath_or_buffer=unique_filename,
            )
        finally:
            teardown_test_files([unique_filename])

    def test_to_pickle(self):
        modin_df, pandas_df = create_test_dfs(TEST_DATA)
        eval_to_file(
            modin_obj=modin_df, pandas_obj=pandas_df, fn="to_pickle", extension="pkl"
        )

        unique_filename_modin = get_unique_filename(extension="pkl")
        unique_filename_pandas = get_unique_filename(extension="pkl")
        try:
            pd.to_pickle(modin_df, unique_filename_modin)
            pandas.to_pickle(pandas_df, unique_filename_pandas)

            assert assert_files_eq(unique_filename_modin, unique_filename_pandas)
        finally:
            teardown_test_files([unique_filename_modin, unique_filename_pandas])


def test_from_arrow():
    _, pandas_df = create_test_dfs(TEST_DATA)
    modin_df = from_arrow(pa.Table.from_pandas(pandas_df))
    df_equals(modin_df, pandas_df)


def test_to_dense():
    modin_df, pandas_df = create_test_dfs({"col1": pandas.SparseArray([0, 1, 0])})
    df_equals(modin_df.sparse.to_dense(), pandas_df.sparse.to_dense())


def test_to_dict():
    modin_df, _ = create_test_dfs(TEST_DATA)
    assert modin_df.to_dict() == to_pandas(modin_df).to_dict()


def test_to_latex():
    modin_df, _ = create_test_dfs(TEST_DATA)
    assert modin_df.to_latex() == to_pandas(modin_df).to_latex()


def test_to_period():
    index = pandas.DatetimeIndex(
        pandas.date_range("2000", freq="h", periods=len(TEST_DATA["col1"]))
    )
    modin_df, pandas_df = create_test_dfs(TEST_DATA, index=index)
    df_equals(modin_df.to_period(), pandas_df.to_period())
