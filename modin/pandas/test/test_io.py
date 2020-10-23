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
from modin.utils import to_pandas
from modin.pandas.utils import from_arrow
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import os
import shutil
import sqlalchemy as sa
import csv
from string import ascii_uppercase, ascii_lowercase, digits

from .utils import (
    df_equals,
    json_short_string,
    json_short_bytes,
    json_long_string,
    json_long_bytes,
    eval_general,
    random_state,
)

from modin.config import Engine, Backend

if Backend.get() == "Pandas":
    import modin.pandas as pd
else:
    import modin.experimental.pandas as pd

pd.DEFAULT_NPARTITIONS = 4

TEST_PARQUET_FILENAME = "test.parquet"
TEST_CSV_FILENAME = "test.csv"
TEST_JSON_FILENAME = "test.json"
TEST_HTML_FILENAME = "test.html"
TEST_EXCEL_FILENAME = "test.xlsx"
TEST_FEATHER_FILENAME = "test.feather"
TEST_READ_HDF_FILENAME = "test.hdf"
TEST_WRITE_HDF_FILENAME_MODIN = "test_write_modin.hdf"
TEST_WRITE_HDF_FILENAME_PANDAS = "test_write_pandas.hdf"
TEST_STATA_FILENAME = "test.dta"
TEST_PICKLE_FILENAME = "test.pkl"
TEST_SAS_FILENAME = os.getcwd() + "/data/test1.sas7bdat"
TEST_FWF_FILENAME = "test_fwf.txt"
TEST_GBQ_FILENAME = "test_gbq."
SMALL_ROW_SIZE = 2000

test_data_dir = os.path.join(os.path.dirname(__file__), "data")

str_single_element = "1"
str_single_col = """1
2
3
"""
str_four_cols = """1, 2, 3, 4
5, 6, 7, 8
9, 10, 11, 12
"""
str_non_unique_cols = """col,col,col,col
5,6,7,8
9,10,11,12
"""
str_initial_spaces = """col1,col2,col3,col4
five,  six,  seven,  eight
    five,    six,    seven,    eight
five, six,  seven,   eight
"""
str_delim_whitespaces = """col1 col2  col3   col4
5 6   7  8
9  10    11 12
"""
str_bad_none_quotes = """1, 2, 3, 4
one, two, three, four
five, "six", seven, "eight
"""

read_csv_non_acc_exc = [TypeError, FileNotFoundError]
read_csv_non_acc_exc_without_TypeError = list(read_csv_non_acc_exc).remove(TypeError)

test_csv_dialect_params = {
    "delimiter": "_",
    "doublequote": False,
    "escapechar": "d",
    "quotechar": "d",
    "quoting": csv.QUOTE_ALL,
}
csv.register_dialect("test_csv_dialect", **test_csv_dialect_params)


def eval_io(
    fn_name,
    comparator=df_equals,
    cast_to_str=False,
    check_exception_type=True,
    nonacceptable_exception_types=read_csv_non_acc_exc,
    *args,
    **kwargs,
):
    def applyier(module, *args, **kwargs):
        result = getattr(module, fn_name)(*args, **kwargs)
        # There could be some missmatches in dtypes, so we're
        # casting the whole frame to `str` before comparison.
        # See issue #1931 for details.
        if cast_to_str:
            result = result.astype(str)
        return result

    eval_general(
        pd,
        pandas,
        applyier,
        check_exception_type=check_exception_type,
        nonacceptable_exception_types=nonacceptable_exception_types,
        *args,
        **kwargs,
    )


def eval_io_from_str(csv_str, unique_filename, **kwargs):
    try:
        with open(unique_filename, "w") as f:
            f.write(csv_str)

        eval_io(
            filepath_or_buffer=unique_filename,
            fn_name="read_csv",
            **kwargs,
        )

    finally:
        os.remove(unique_filename)


@pytest.fixture
def make_parquet_file():
    """Pytest fixture factory that makes a parquet file/dir for testing.

    Yields:
        Function that generates a parquet file/dir
    """

    def _make_parquet_file(
        row_size=SMALL_ROW_SIZE, force=False, directory=False, partitioned_columns=[]
    ):
        """Helper function to generate parquet files/directories.

        Args:
            row_size: Number of rows for the dataframe.
            force: Create a new file/directory even if one already exists.
            directory: Create a partitioned directory using pyarrow.
            partitioned_columns: Create a partitioned directory using pandas.
            Will be ignored if directory=True.
        """
        df = pandas.DataFrame(
            {"col1": np.arange(row_size), "col2": np.arange(row_size)}
        )
        if os.path.exists(TEST_PARQUET_FILENAME) and not force:
            pass
        elif directory:
            if os.path.exists(TEST_PARQUET_FILENAME):
                shutil.rmtree(TEST_PARQUET_FILENAME)
            else:
                os.mkdir(TEST_PARQUET_FILENAME)
            table = pa.Table.from_pandas(df)
            pq.write_to_dataset(table, root_path=TEST_PARQUET_FILENAME)
        elif len(partitioned_columns) > 0:
            df.to_parquet(TEST_PARQUET_FILENAME, partition_cols=partitioned_columns)
        else:
            df.to_parquet(TEST_PARQUET_FILENAME)

    # Return function that generates csv files
    yield _make_parquet_file

    # Delete parquet file that was created
    if os.path.exists(TEST_PARQUET_FILENAME):
        if os.path.isdir(TEST_PARQUET_FILENAME):
            shutil.rmtree(TEST_PARQUET_FILENAME)
        else:
            os.remove(TEST_PARQUET_FILENAME)


def create_test_modin_dataframe():
    df = pd.DataFrame(
        {
            "col1": [0, 1, 2, 3],
            "col2": [4, 5, 6, 7],
            "col3": [8, 9, 10, 11],
            "col4": [12, 13, 14, 15],
            "col5": [0, 0, 0, 0],
        }
    )

    return df


def create_test_pandas_dataframe():
    df = pandas.DataFrame(
        {
            "col1": [0, 1, 2, 3],
            "col2": [4, 5, 6, 7],
            "col3": [8, 9, 10, 11],
            "col4": [12, 13, 14, 15],
            "col5": [0, 0, 0, 0],
        }
    )

    return df


def assert_files_eq(path1, path2):
    with open(path1, "rb") as file1, open(path2, "rb") as file2:
        file1_content = file1.read()
        file2_content = file2.read()

        if file1_content == file2_content:
            return True
        else:
            return False


def teardown_test_file(test_path):
    if os.path.exists(test_path):
        os.remove(test_path)


def get_random_string():
    random_string = "".join(
        random_state.choice(
            [x for x in ascii_uppercase + ascii_lowercase + digits], size=10
        ).tolist()
    )
    return random_string


def insert_lines_to_csv(
    csv_name: str,
    lines_positions: list,
    lines_type: str = "blank",
    encoding: str = None,
    **csv_reader_writer_params,
):
    cols_number = len(pandas.read_csv(csv_name, nrows=1).columns)
    if lines_type == "blank":
        # empty lines
        lines_data = []
    elif lines_type == "bad":
        # lines with len(lines_data) > cols_number
        cols_number = len(pandas.read_csv(csv_name, nrows=1).columns)
        lines_data = [x for x in range(cols_number + 1)]
    else:
        raise ValueError(
            f"acceptable values for  parameter are ['blank', 'bad'], actually passed {lines_type}"
        )
    lines = []
    dialect = "excel"
    with open(csv_name, "r", encoding=encoding, newline="") as read_file:
        try:
            dialect = csv.Sniffer().sniff(read_file.read())
            read_file.seek(0)
        except Exception:
            dialect = None

        reader = csv.reader(
            read_file,
            dialect=dialect if dialect is not None else "excel",
            **csv_reader_writer_params,
        )
        counter = 0
        for row in reader:
            if counter in lines_positions:
                lines.append(lines_data)
            else:
                lines.append(row)
            counter += 1
    with open(csv_name, "w", encoding=encoding, newline="") as write_file:
        writer = csv.writer(
            write_file,
            dialect=dialect if dialect is not None else "excel",
            **csv_reader_writer_params,
        )
        writer.writerows(lines)


def _make_csv_file(filenames_list, caching=False):
    filenames = filenames_list

    def _csv_file_maker(
        filename=TEST_CSV_FILENAME,
        row_size=SMALL_ROW_SIZE,
        force=True,
        delimiter=",",
        encoding=None,
        compression="infer",
        additional_col_values=None,
        add_blank_lines=False,
        add_bad_lines=False,
        add_nan_lines=False,
        thousands_separator=None,
        decimal_separator=None,
        lineterminator=None,
        comment_col_char=None,
        quoting=csv.QUOTE_MINIMAL,
        quotechar='"',
        doublequote=True,
        escapechar=None,
        line_terminator=os.linesep,
    ):
        if os.path.exists(filename) and not force:
            pass
        else:
            dates = pandas.date_range("2000", freq="h", periods=row_size)
            data = {
                "col1": np.arange(row_size) * 10,
                "col2": [str(x.date()) for x in dates],
                "col3": np.arange(row_size) * 10,
                "col4": [str(x.time()) for x in dates],
                "col5": [get_random_string() for _ in range(row_size)],
                "col6": random_state.uniform(low=0.0, high=10000.0, size=row_size),
            }

            if additional_col_values is not None:
                assert isinstance(additional_col_values, (list, tuple))
                data.update(
                    {
                        "col7": random_state.choice(
                            additional_col_values, size=row_size
                        ),
                    }
                )
            df = pandas.DataFrame(data)
            if add_nan_lines:
                for i in range(0, row_size, row_size // (row_size // 10)):
                    df.loc[i] = pandas.Series()
            if comment_col_char:
                char = comment_col_char if isinstance(comment_col_char, str) else "#"
                df.insert(
                    loc=0,
                    column="col_with_comments",
                    value=[char if (x + 2) == 0 else x for x in range(row_size)],
                )

            if thousands_separator:
                df["col1"] = df["col1"].apply(
                    lambda x: f"{x:,d}".replace(",", thousands_separator)
                )
                df["col3"] = df["col3"].apply(
                    lambda x: f"{x:,d}".replace(",", thousands_separator)
                )
                df["col6"] = df["col6"].apply(
                    lambda x: f"{x:,f}".replace(",", thousands_separator)
                )

            if compression == "gzip":
                filename = "{}.gz".format(filename)
            elif compression == "zip" or compression == "xz" or compression == "bz2":
                filename = "{fname}.{comp}".format(fname=filename, comp=compression)
            df.to_csv(
                filename,
                sep=delimiter,
                encoding=encoding,
                compression=compression,
                index=False,
                decimal=decimal_separator if decimal_separator else ".",
                line_terminator=line_terminator,
                quoting=quoting,
                quotechar=quotechar,
                doublequote=doublequote,
                escapechar=escapechar,
            )
            csv_reader_writer_params = {
                "delimiter": delimiter,
                "doublequote": doublequote,
                "escapechar": escapechar,
                "lineterminator": line_terminator,
                "quotechar": quotechar,
                "quoting": quoting,
            }
            if add_blank_lines:
                insert_lines_to_csv(
                    csv_name=filename,
                    lines_positions=[
                        x for x in range(5, row_size, row_size // (row_size // 10))
                    ],
                    lines_type="blank",
                    encoding=encoding,
                    **csv_reader_writer_params,
                )
            if add_bad_lines:
                insert_lines_to_csv(
                    csv_name=filename,
                    lines_positions=[
                        x for x in range(6, row_size, row_size // (row_size // 10))
                    ],
                    lines_type="bad",
                    encoding=encoding,
                    **csv_reader_writer_params,
                )
            filenames.append(filename)
            return df

    return _csv_file_maker


@pytest.fixture
def make_csv_file():
    """Pytest fixture factory that makes temp csv files for testing.

    Yields:
        Function that generates csv files
    """
    filenames = []

    yield _make_csv_file(filenames)

    # Delete csv files that were created
    for filename in filenames:
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except PermissionError:
                pass


@pytest.fixture(scope="function")
def TestReadCSVFixture():
    filenames = []
    files_ids = [
        "test_from_csv_regular",
        "test_from_csv_blank_lines",
        "test_from_csv_yes_no",
        "test_from_csv_nas",
        "test_from_csv_bad_lines",
    ]
    pytest.csvs_names = {file_id: get_unique_filename(file_id) for file_id in files_ids}
    # test_from_csv_col_handling, test_from_csv_parsing
    _make_csv_file(filenames)(
        filename=pytest.csvs_names["test_from_csv_regular"], force=False
    )
    _make_csv_file(filenames)(
        filename=pytest.csvs_names["test_from_csv_yes_no"],
        force=False,
        additional_col_values=["Yes", "true", "No", "false"],
    )
    # test_from_csv_col_handling
    _make_csv_file(filenames)(
        filename=pytest.csvs_names["test_from_csv_blank_lines"],
        force=False,
        add_blank_lines=True,
    )
    # test_from_csv_na_handling
    _make_csv_file(filenames)(
        filename=pytest.csvs_names["test_from_csv_nas"],
        force=False,
        add_blank_lines=True,
        additional_col_values=["<NA>", "N/A", "NA", "NULL", "custom_nan", "73"],
    )
    _make_csv_file(filenames)(
        filename=pytest.csvs_names["test_from_csv_bad_lines"],
        force=False,
        add_bad_lines=True,
    )

    yield

    # Delete csv files that were created
    for filename in filenames:
        if os.path.exists(filename):
            try:
                os.remove(filename)
            except PermissionError:
                pass


def setup_json_file(row_size, force=False):
    if os.path.exists(TEST_JSON_FILENAME) and not force:
        pass
    else:
        df = pandas.DataFrame(
            {"col1": np.arange(row_size), "col2": np.arange(row_size)}
        )
        df.to_json(TEST_JSON_FILENAME)


def setup_json_lines_file(row_size, force=False):
    if os.path.exists(TEST_JSON_FILENAME) and not force:
        pass
    else:
        df = pandas.DataFrame(
            {"col1": np.arange(row_size), "col2": np.arange(row_size)}
        )
        df.to_json(TEST_JSON_FILENAME, lines=True, orient="records")


def teardown_json_file():
    if os.path.exists(TEST_JSON_FILENAME):
        os.remove(TEST_JSON_FILENAME)


def setup_html_file(row_size, force=False):
    if os.path.exists(TEST_HTML_FILENAME) and not force:
        pass
    else:
        df = pandas.DataFrame(
            {"col1": np.arange(row_size), "col2": np.arange(row_size)}
        )
        df.to_html(TEST_HTML_FILENAME)


def teardown_html_file():
    if os.path.exists(TEST_HTML_FILENAME):
        os.remove(TEST_HTML_FILENAME)


def setup_clipboard(row_size, force=False):
    df = pandas.DataFrame({"col1": np.arange(row_size), "col2": np.arange(row_size)})
    df.to_clipboard()


def setup_excel_file(row_size, force=False):
    if os.path.exists(TEST_EXCEL_FILENAME) and not force:
        pass
    else:
        df = pandas.DataFrame(
            {"col1": np.arange(row_size), "col2": np.arange(row_size)}
        )
        df.to_excel(TEST_EXCEL_FILENAME)


def teardown_excel_file():
    if os.path.exists(TEST_EXCEL_FILENAME):
        try:
            os.remove(TEST_EXCEL_FILENAME)
        except PermissionError:
            pass


def setup_feather_file(row_size, force=False):
    if os.path.exists(TEST_FEATHER_FILENAME) and not force:
        pass
    else:
        df = pandas.DataFrame(
            {"col1": np.arange(row_size), "col2": np.arange(row_size)}
        )
        df.to_feather(TEST_FEATHER_FILENAME)


def teardown_feather_file():
    if os.path.exists(TEST_FEATHER_FILENAME):
        os.remove(TEST_FEATHER_FILENAME)


def setup_hdf_file(row_size, force=False, format=None):
    if os.path.exists(TEST_READ_HDF_FILENAME) and not force:
        pass
    else:
        df = pandas.DataFrame(
            {"col1": np.arange(row_size), "col2": np.arange(row_size)}
        )
        df.to_hdf(TEST_READ_HDF_FILENAME, key="df", format=format)


def teardown_hdf_file():
    if os.path.exists(TEST_READ_HDF_FILENAME):
        os.remove(TEST_READ_HDF_FILENAME)


def setup_stata_file(row_size, force=False):
    if os.path.exists(TEST_STATA_FILENAME) and not force:
        pass
    else:
        df = pandas.DataFrame(
            {"col1": np.arange(row_size), "col2": np.arange(row_size)}
        )
        df.to_stata(TEST_STATA_FILENAME)


def teardown_stata_file():
    if os.path.exists(TEST_STATA_FILENAME):
        os.remove(TEST_STATA_FILENAME)


def setup_pickle_file(row_size, force=False):
    if os.path.exists(TEST_PICKLE_FILENAME) and not force:
        pass
    else:
        df = pandas.DataFrame(
            {"col1": np.arange(row_size), "col2": np.arange(row_size)}
        )
        df.to_pickle(TEST_PICKLE_FILENAME)


def teardown_pickle_file():
    if os.path.exists(TEST_PICKLE_FILENAME):
        os.remove(TEST_PICKLE_FILENAME)


@pytest.fixture
def make_sql_connection():
    """Sets up sql connections and takes them down after the caller is done.

    Yields:
        Factory that generates sql connection objects
    """
    filenames = []

    def _sql_connection(filename, table=""):
        # Remove file if exists
        if os.path.exists(filename):
            os.remove(filename)
        filenames.append(filename)
        # Create connection and, if needed, table
        conn = "sqlite:///{}".format(filename)
        if table:
            df = pandas.DataFrame(
                {
                    "col1": [0, 1, 2, 3, 4, 5, 6],
                    "col2": [7, 8, 9, 10, 11, 12, 13],
                    "col3": [14, 15, 16, 17, 18, 19, 20],
                    "col4": [21, 22, 23, 24, 25, 26, 27],
                    "col5": [0, 0, 0, 0, 0, 0, 0],
                }
            )
            df.to_sql(table, conn)
        return conn

    yield _sql_connection

    # Takedown the fixture
    for filename in filenames:
        if os.path.exists(filename):
            os.remove(filename)


def setup_fwf_file(overwrite=False, fwf_data=None):
    if not overwrite and os.path.exists(TEST_FWF_FILENAME):
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

    with open(TEST_FWF_FILENAME, "w") as f:
        f.write(fwf_data)


def teardown_fwf_file():
    if os.path.exists(TEST_FWF_FILENAME):
        try:
            os.remove(TEST_FWF_FILENAME)
        except PermissionError:
            pass


def get_unique_filename(
    test_name: str,
    kwargs: dict = {},
    extension: str = "csv",
    data_dir=test_data_dir,
    suffix: str = "",
):
    if len(kwargs) == 0 and extension == "csv" and suffix == "":
        return os.path.join(data_dir, (test_name + f"_{suffix}" + f".{extension}"))

    assert "." not in extension, "please provide pure extenxion name without '.'"
    prohibited_chars = ['"', "\n"]
    non_prohibited_char = "np_char"
    char_counter = 0
    kwargs_name = dict(kwargs)
    for key, value in kwargs_name.items():
        for char in prohibited_chars:
            if isinstance(value, str) and char in value or callable(value):
                kwargs_name[key] = non_prohibited_char + str(char_counter)
                char_counter += 1
    parameters_values = "_".join(
        [
            str(value)
            if not isinstance(value, (list, tuple))
            else "_".join([str(x) for x in value])
            for value in kwargs_name.values()
        ]
    )
    return os.path.join(data_dir, parameters_values + f"_{suffix}" + f".{extension}")


@pytest.mark.usefixtures("TestReadCSVFixture")
class TestReadCSV:
    # delimiter tests
    @pytest.mark.parametrize("sep", ["_", ",", ".", "\n"])
    @pytest.mark.parametrize("delimiter", ["_", ",", ".", "\n"])
    @pytest.mark.parametrize("decimal", [".", "_"])
    @pytest.mark.parametrize("thousands", [None, ",", "_", " "])
    def test_from_csv_delimiters(
        self, make_csv_file, sep, delimiter, decimal, thousands
    ):
        kwargs = {
            "delimiter": delimiter,
            "sep": sep,
            "decimal": decimal,
            "thousands": thousands,
        }
        unique_filename = get_unique_filename("test_from_csv_delimiter", kwargs)
        make_csv_file(
            filename=unique_filename,
            delimiter=delimiter,
            thousands_separator=thousands,
            decimal_separator=decimal,
        )

        eval_io(
            filepath_or_buffer=unique_filename,
            fn_name="read_csv",
            **kwargs,
        )

    # Column and Index Locations and Names tests
    @pytest.mark.parametrize("header", ["infer", None, 0])
    @pytest.mark.parametrize("index_col", [None, "col1"])
    @pytest.mark.parametrize("prefix", [None, "_", "col"])
    @pytest.mark.parametrize(
        "names", [None, ["col1"], ["c1", "c2", "c3", "c4", "c5", "c6", "c7"]]
    )
    @pytest.mark.parametrize("usecols", [["col1"], ["col1", "col2", "col6"], [0, 1, 5]])
    @pytest.mark.parametrize("skip_blank_lines", [True, False])
    def test_from_csv_col_handling(
        self,
        header,
        index_col,
        prefix,
        names,
        usecols,
        skip_blank_lines,
    ):
        kwargs = {
            "header": header,
            "index_col": index_col,
            "prefix": prefix,
            "names": names,
            "usecols": usecols,
            "skip_blank_lines": skip_blank_lines,
        }

        eval_io(
            filepath_or_buffer=pytest.csvs_names["test_from_csv_blank_lines"],
            fn_name="read_csv",
            **kwargs,
        )

    @pytest.mark.xfail(reason="infinite recursion error - issue #2032")
    def test_from_csv_squeeze(self, make_csv_file):
        unique_filename = get_unique_filename("test_from_csv_squeeze")

        for csv_str in [str_single_element, str_single_col, str_four_cols]:
            eval_io_from_str(csv_str, unique_filename, squeeze=True)
            eval_io_from_str(csv_str, unique_filename, header=None, squeeze=True)

    def test_from_csv_mangle_dupe_cols(self):
        unique_filename = get_unique_filename("test_from_csv_mangle_dupe_cols")

        eval_io_from_str(str_non_unique_cols, unique_filename, mangle_dupe_cols=True)

    # General Parsing Configuration
    @pytest.mark.parametrize("dtype", [None, True])
    @pytest.mark.parametrize("engine", [None, "python", "c"])
    @pytest.mark.parametrize(
        "converters",
        [
            None,
            {
                "col1": lambda x: np.int64(x) * 10,
                "col2": pd.to_datetime,
                "col4": lambda x: x.replace(":", ";"),
            },
        ],
    )
    @pytest.mark.parametrize("true_values", [["Yes"], ["Yes", "true"], None])
    @pytest.mark.parametrize("false_values", [["No"], ["No", "false"], None])
    @pytest.mark.parametrize("skiprows", [2, lambda x: x % 2])
    @pytest.mark.parametrize("skipfooter", [0, 10])
    @pytest.mark.parametrize("nrows", [123, None])
    @pytest.mark.parametrize("names", [["c1", "c2", "c3", "c4"], None])
    def test_from_csv_parsing(
        self,
        dtype,
        engine,
        converters,
        true_values,
        false_values,
        skiprows,
        skipfooter,
        nrows,
        names,
    ):
        kwargs = {
            "dtype": dtype,
            "engine": engine,
            "converters": converters,
            "true_values": true_values,
            "false_values": false_values,
            "skiprows": skiprows,
            "skipfooter": skipfooter,
            "nrows": nrows,
            "names": names,
        }

        filename = (
            pytest.csvs_names["test_from_csv_yes_no"]
            if true_values or false_values
            else pytest.csvs_names["test_from_csv_regular"]
        )

        if kwargs["dtype"]:
            kwargs["dtype"] = {
                col: "object" for col in pandas.read_csv(filename, nrows=1).columns
            }

        eval_io(
            filepath_or_buffer=filename,
            fn_name="read_csv",
            check_kwargs_callable=not (callable(skiprows) or callable(converters)),
            **kwargs,
        )

    def test_from_csv_skipinitialspace(self, make_csv_file):
        unique_filename = get_unique_filename("test_from_csv_skipinitialspace")

        eval_io_from_str(str_initial_spaces, unique_filename, skipinitialspace=True)

    # NA and Missing Data Handling tests
    @pytest.mark.parametrize("na_values", ["custom_nan", "73"])
    @pytest.mark.parametrize("keep_default_na", [True, False])
    @pytest.mark.parametrize("na_filter", [True, False])
    @pytest.mark.parametrize("verbose", [True, False])
    @pytest.mark.parametrize("skip_blank_lines", [True, False])
    def test_from_csv_na_handling(
        self,
        na_values,
        keep_default_na,
        na_filter,
        verbose,
        skip_blank_lines,
    ):
        kwargs = {
            "na_values": na_values,
            "keep_default_na": keep_default_na,
            "na_filter": na_filter,
            "verbose": verbose,
            "skip_blank_lines": skip_blank_lines,
        }
        filename = pytest.csvs_names["test_from_csv_nas"]
        eval_io(
            filepath_or_buffer=filename,
            fn_name="read_csv",
            **kwargs,
        )

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
                    reason="Exception: Internal Error - issue #2073"
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("infer_datetime_format", [True, False])
    @pytest.mark.parametrize("keep_date_col", [True, False])
    @pytest.mark.parametrize(
        "date_parser", [None, lambda x: pd.datetime.strptime(x, "%Y-%m-%d")]
    )
    @pytest.mark.parametrize("dayfirst", [True, False])
    @pytest.mark.parametrize("cache_dates", [True, False])
    def test_from_csv_datetime(
        self,
        parse_dates,
        infer_datetime_format,
        keep_date_col,
        date_parser,
        dayfirst,
        cache_dates,
    ):
        case_with_TypeError_exc = isinstance(parse_dates, dict) and callable(
            date_parser
        )
        kwargs = {
            "parse_dates": parse_dates,
            "infer_datetime_format": infer_datetime_format,
            "keep_date_col": keep_date_col,
            "date_parser": date_parser,
            "dayfirst": dayfirst,
            "cache_dates": cache_dates,
        }

        eval_io(
            filepath_or_buffer=pytest.csvs_names["test_from_csv_regular"],
            fn_name="read_csv",
            check_kwargs_callable=not callable(date_parser),
            nonacceptable_exception_types=read_csv_non_acc_exc_without_TypeError
            if case_with_TypeError_exc
            else read_csv_non_acc_exc,
            **kwargs,
        )

    # Iteration tests
    @pytest.mark.parametrize("iterator", [True, False])
    def test_from_csv_iteration(self, make_csv_file, iterator):
        filename = pytest.csvs_names["test_from_csv_regular"]

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

    # Quoting, Compression, and File Format parameters tests
    @pytest.mark.parametrize("compression", ["infer", "gzip", "bz2", "xz", "zip"])
    @pytest.mark.parametrize(
        "encoding",
        [None, "latin8", "ISO-8859-1", "latin1", "iso-8859-1", "cp1252", "utf8"],
    )
    @pytest.mark.parametrize("lineterminator", [None, "x", "\n"])
    @pytest.mark.parametrize("engine", [None, "python", "c"])
    def test_from_csv_compressed(
        self, make_csv_file, compression, encoding, lineterminator, engine
    ):
        kwargs = {
            "compression": compression,
            "encoding": encoding,
            "lineterminator": lineterminator,
            "engine": engine,
        }
        unique_filename = get_unique_filename("test_from_csv_compressed", kwargs)
        make_csv_file(
            filename=unique_filename, encoding=encoding, compression=compression
        )
        compressed_file_path = unique_filename
        if compression == "gzip":
            compressed_file_path += ".gz"
        elif compression != "infer":
            compressed_file_path += f".{compression}"

        eval_io(
            filepath_or_buffer=compressed_file_path,
            fn_name="read_csv",
            **kwargs,
        )

    @pytest.mark.parametrize("thousands", [None, ",", "_", " "])
    @pytest.mark.parametrize("decimal", [".", "_"])
    @pytest.mark.parametrize(
        "quoting",
        [csv.QUOTE_ALL, csv.QUOTE_MINIMAL, csv.QUOTE_NONNUMERIC, csv.QUOTE_NONE],
    )
    @pytest.mark.parametrize("quotechar", ['"', "_", "d"])
    @pytest.mark.parametrize("doublequote", [True, False])
    @pytest.mark.parametrize("escapechar", [None, "d", "x"])
    @pytest.mark.parametrize("comment", [None, "#", "x"])
    @pytest.mark.parametrize("dialect", ["test_csv_dialect", None])
    def test_from_csv_quotes(
        self,
        make_csv_file,
        thousands,
        decimal,
        quoting,
        quotechar,
        doublequote,
        escapechar,
        comment,
        dialect,
    ):
        if escapechar is None and (quotechar == "d" or comment is not None):
            pytest.skip(
                "can't create such file because of error: _csv.Error: need to escape, but no escapechar set"
            )
        kwargs = {
            "thousands": thousands,
            "decimal": decimal,
            "quoting": quoting,
            "quotechar": quotechar,
            "doublequote": doublequote,
            "escapechar": escapechar,
            "comment": comment,
            "dialect": dialect,
        }
        unique_filename = get_unique_filename("test_from_csv_quotes", kwargs)
        if dialect:
            make_csv_file(filename=unique_filename, **test_csv_dialect_params)
        else:
            make_csv_file(
                filename=unique_filename,
                quoting=quoting,
                quotechar=quotechar,
                doublequote=doublequote,
                escapechar=escapechar,
                comment_col_char=comment,
            )

        eval_io(
            filepath_or_buffer=unique_filename,
            fn_name="read_csv",
            **kwargs,
        )

    # Error Handling parameters tests
    @pytest.mark.parametrize("warn_bad_lines", [True, False])
    @pytest.mark.parametrize("error_bad_lines", [True, False])
    def test_from_csv_errors(
        self,
        make_csv_file,
        warn_bad_lines,
        error_bad_lines,
    ):
        kwargs = {
            "warn_bad_lines": warn_bad_lines,
            "error_bad_lines": error_bad_lines,
        }

        eval_io(
            filepath_or_buffer=pytest.csvs_names["test_from_csv_bad_lines"],
            fn_name="read_csv",
            **kwargs,
        )

    # Internal parameters tests
    @pytest.mark.parametrize("engine", [None, "python", "c"])
    @pytest.mark.parametrize("delimiter", [",", " "])
    @pytest.mark.parametrize("delim_whitespace", [True, False])
    @pytest.mark.parametrize("low_memory", [True, False])
    @pytest.mark.parametrize("memory_map", [True, False])
    @pytest.mark.parametrize("float_precision", [None, "high", "round_trip"])
    def test_from_csv_internal(
        self,
        make_csv_file,
        engine,
        delimiter,
        delim_whitespace,
        low_memory,
        memory_map,
        float_precision,
    ):
        case_with_TypeError_exc = (
            engine == "python"
            and delimiter == ","
            and delim_whitespace
            and low_memory
            and memory_map
            and float_precision is None
        )

        kwargs = {
            "engine": engine,
            "delimiter": delimiter,
            "delim_whitespace": delim_whitespace,
            "low_memory": low_memory,
            "memory_map": memory_map,
            "float_precision": float_precision,
        }

        unique_filename_1 = get_unique_filename("test_from_csv_internal", kwargs)
        unique_filename_2 = get_unique_filename(
            "test_from_csv_internal", kwargs, suffix="from_csv"
        )

        make_csv_file(
            filename=unique_filename_1,
            delimiter=delimiter,
        )

        eval_io(
            filepath_or_buffer=unique_filename_1,
            fn_name="read_csv",
            nonacceptable_exception_types=read_csv_non_acc_exc_without_TypeError
            if case_with_TypeError_exc
            else read_csv_non_acc_exc,
            **kwargs,
        )

        eval_io_from_str(
            str_delim_whitespaces,
            unique_filename_2,
            nonacceptable_exception_types=read_csv_non_acc_exc_without_TypeError
            if case_with_TypeError_exc
            else read_csv_non_acc_exc,
            **kwargs,
        )

    def test_from_csv_default_to_pandas_behavior(self, make_csv_file):
        unique_filename = get_unique_filename(
            "test_from_csv_default_to_pandas_behavior"
        )
        make_csv_file(filename=unique_filename)

        with pytest.warns(UserWarning):
            # This tests that we default to pandas on a buffer
            from io import StringIO

            pd.read_csv(StringIO(open(unique_filename, "r").read()))

        with pytest.warns(UserWarning):
            pd.read_csv(unique_filename, skiprows=lambda x: x in [0, 2])

    def test_from_csv_parse_dates(self, make_csv_file):
        unique_filename = get_unique_filename("test_from_csv_parse_dates")
        make_csv_file(filename=unique_filename)

        eval_io(
            filepath_or_buffer=unique_filename,
            fn_name="read_csv",
            parse_dates=[["col2", "col4"]],
        )

        eval_io(
            filepath_or_buffer=unique_filename,
            fn_name="read_csv",
            parse_dates={"time": ["col2", "col4"]},
        )

    @pytest.mark.parametrize("nrows", [21, 5, None])
    @pytest.mark.parametrize("skiprows", [4, 1, 500, None])
    def test_from_csv_newlines_in_quotes(self, nrows, skiprows):
        eval_io(
            filepath_or_buffer="modin/pandas/test/data/newlines.csv",
            fn_name="read_csv",
            nrows=nrows,
            skiprows=skiprows,
            cast_to_str=True,
        )

    @pytest.mark.parametrize("nrows", [123, None])
    def test_from_csv_sep_none(self, make_csv_file, nrows):
        unique_filename = get_unique_filename(
            "test_from_csv_sep_none", {"nrows": nrows}
        )
        make_csv_file(filename=unique_filename)

        with pytest.warns(ParserWarning):
            pandas_df = pandas.read_csv(unique_filename, sep=None, nrows=nrows)
        with pytest.warns(ParserWarning):
            modin_df = pd.read_csv(unique_filename, sep=None, nrows=nrows)
        df_equals(modin_df, pandas_df)

    @pytest.mark.parametrize("nrows", [2, None])
    def test_from_csv_bad_quotes(self, nrows):
        unique_filename = get_unique_filename(
            "test_from_csv_bad_quotes", {"nrows": nrows}
        )

        eval_io_from_str(str_bad_none_quotes, unique_filename, nrows=nrows)

    @pytest.mark.parametrize("nrows", [2, None])
    def test_from_csv_quote_none(self, nrows):
        unique_filename = get_unique_filename(
            "test_from_csv_quote_none", {"nrows": nrows}
        )

        eval_io_from_str(
            str_bad_none_quotes, unique_filename, quoting=csv.QUOTE_NONE, nrows=nrows
        )

    def test_from_csv_categories(self):

        eval_io(
            filepath_or_buffer="modin/pandas/test/data/test_categories.csv",
            fn_name="read_csv",
            names=["one", "two"],
            dtype={"one": "int64", "two": "category"},
        )

    def test_parse_dates_read_csv(self):

        eval_io(
            filepath_or_buffer="modin/pandas/test/data/test_time_parsing.csv",
            fn_name="read_csv",
        )

        eval_io(
            filepath_or_buffer="modin/pandas/test/data/test_time_parsing.csv",
            fn_name="read_csv",
            names=[
                "timestamp",
                "symbol",
                "high",
                "low",
                "open",
                "close",
                "spread",
                "volume",
            ],
            header=0,
            index_col=0,
            encoding="utf-8",
        )

        eval_io(
            filepath_or_buffer="modin/pandas/test/data/test_time_parsing.csv",
            fn_name="read_csv",
            names=[
                "timestamp",
                "symbol",
                "high",
                "low",
                "open",
                "close",
                "spread",
                "volume",
            ],
            header=0,
            index_col=0,
            parse_dates=["timestamp"],
            encoding="utf-8",
        )

        eval_io(
            filepath_or_buffer="modin/pandas/test/data/test_time_parsing.csv",
            fn_name="read_csv",
            names=[
                "timestamp",
                "symbol",
                "high",
                "low",
                "open",
                "close",
                "spread",
                "volume",
            ],
            header=0,
            index_col=2,
            parse_dates=["timestamp"],
            encoding="utf-8",
        )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"header": None, "usecols": [0, 7]},
            {"usecols": [0, 7]},
            {"names": [0, 7], "usecols": [0, 7]},
        ],
    )
    def test_from_csv_with_args(self, kwargs):
        eval_io(
            filepath_or_buffer="modin/pandas/test/data/issue_621.csv",
            fn_name="read_csv",
            **kwargs,
        )

    @pytest.mark.skipif(Engine.get() == "Python", reason="Using pandas implementation")
    def test_from_csv_s3(self, make_csv_file):
        dataset_url = "s3://noaa-ghcn-pds/csv/1788.csv"
        pandas_df = pandas.read_csv(dataset_url)

        # This first load is to trigger all the import deprecation warnings
        modin_df = pd.read_csv(dataset_url)

        # This will warn if it defaults to pandas behavior, but it shouldn't
        with pytest.warns(None) as record:
            modin_df = pd.read_csv(dataset_url)

        assert not any(
            "defaulting to pandas implementation" in str(err) for err in record.list
        )

        df_equals(modin_df, pandas_df)

    def test_from_csv_default(self, make_csv_file):
        # We haven't implemented read_csv from https, but if it's implemented, then this needs to change
        dataset_url = "https://raw.githubusercontent.com/modin-project/modin/master/modin/pandas/test/data/blah.csv"
        pandas_df = pandas.read_csv(dataset_url)

        with pytest.warns(UserWarning):
            modin_df = pd.read_csv(dataset_url)

        df_equals(modin_df, pandas_df)

    def test_read_csv_incorrect_data(self):

        eval_io(
            filepath_or_buffer="modin/pandas/test/data/test_categories.json",
            fn_name="read_csv",
        )


def test_from_table(make_csv_file):
    make_csv_file(delimiter="\t")

    pandas_df = pandas.read_table(TEST_CSV_FILENAME)
    modin_df = pd.read_table(TEST_CSV_FILENAME)

    df_equals(modin_df, pandas_df)

    pandas_df = pandas.read_table(Path(TEST_CSV_FILENAME))
    modin_df = pd.read_table(Path(TEST_CSV_FILENAME))

    df_equals(modin_df, pandas_df)


def test_from_parquet(make_parquet_file):
    make_parquet_file(SMALL_ROW_SIZE)

    pandas_df = pandas.read_parquet(TEST_PARQUET_FILENAME)
    modin_df = pd.read_parquet(TEST_PARQUET_FILENAME)
    df_equals(modin_df, pandas_df)


def test_from_parquet_with_columns(make_parquet_file):
    make_parquet_file(SMALL_ROW_SIZE)

    pandas_df = pandas.read_parquet(TEST_PARQUET_FILENAME, columns=["col1"])
    modin_df = pd.read_parquet(TEST_PARQUET_FILENAME, columns=["col1"])
    df_equals(modin_df, pandas_df)


def test_from_parquet_partition(make_parquet_file):
    make_parquet_file(SMALL_ROW_SIZE, directory=True)

    pandas_df = pandas.read_parquet(TEST_PARQUET_FILENAME)
    modin_df = pd.read_parquet(TEST_PARQUET_FILENAME)
    df_equals(modin_df, pandas_df)


def test_from_parquet_partition_with_columns(make_parquet_file):
    make_parquet_file(SMALL_ROW_SIZE, directory=True)

    pandas_df = pandas.read_parquet(TEST_PARQUET_FILENAME, columns=["col1"])
    modin_df = pd.read_parquet(TEST_PARQUET_FILENAME, columns=["col1"])
    df_equals(modin_df, pandas_df)


def test_from_parquet_partitioned_columns(make_parquet_file):
    make_parquet_file(SMALL_ROW_SIZE, partitioned_columns=["col1"])

    pandas_df = pandas.read_parquet(TEST_PARQUET_FILENAME)
    modin_df = pd.read_parquet(TEST_PARQUET_FILENAME)
    df_equals(modin_df, pandas_df)


def test_from_parquet_partitioned_columns_with_columns(make_parquet_file):
    make_parquet_file(SMALL_ROW_SIZE, partitioned_columns=["col1"])

    pandas_df = pandas.read_parquet(TEST_PARQUET_FILENAME, columns=["col1"])
    modin_df = pd.read_parquet(TEST_PARQUET_FILENAME, columns=["col1"])
    df_equals(modin_df, pandas_df)


def test_from_parquet_pandas_index():
    # Ensure modin can read parquet files written by pandas with a non-RangeIndex object
    pandas_df = pandas.DataFrame(
        {
            "idx": np.random.randint(0, 100_000, size=2000),
            "A": np.random.randint(0, 100_000, size=2000),
            "B": ["a", "b"] * 1000,
            "C": ["c"] * 2000,
        }
    )
    filepath = "tmp.parquet"
    pandas_df.set_index("idx").to_parquet(filepath)
    # read the same parquet using modin.pandas
    df_equals(pd.read_parquet(filepath), pandas.read_parquet(filepath))

    pandas_df.set_index(["idx", "A"]).to_parquet(filepath)
    df_equals(pd.read_parquet(filepath), pandas.read_parquet(filepath))
    os.remove(filepath)


def test_from_parquet_pandas_index_partitioned():
    # Ensure modin can read parquet files written by pandas with a non-RangeIndex object
    pandas_df = pandas.DataFrame(
        {
            "idx": np.random.randint(0, 100_000, size=2000),
            "A": np.random.randint(0, 10, size=2000),
            "B": ["a", "b"] * 1000,
            "C": ["c"] * 2000,
        }
    )
    filepath = "tmp_folder.parquet"
    pandas_df.set_index("idx").to_parquet(filepath, partition_cols=["A"])
    # read the same parquet using modin.pandas
    df_equals(pd.read_parquet(filepath), pandas.read_parquet(filepath))
    shutil.rmtree(filepath)


def test_from_parquet_hdfs():
    path = "modin/pandas/test/data/hdfs.parquet"
    pandas_df = pandas.read_parquet(path)
    modin_df = pd.read_parquet(path)
    df_equals(modin_df, pandas_df)


def test_from_json():
    setup_json_file(SMALL_ROW_SIZE)

    pandas_df = pandas.read_json(TEST_JSON_FILENAME)
    modin_df = pd.read_json(TEST_JSON_FILENAME)

    df_equals(modin_df, pandas_df)

    teardown_json_file()


def test_from_json_categories():
    pandas_df = pandas.read_json(
        "modin/pandas/test/data/test_categories.json",
        dtype={"one": "int64", "two": "category"},
    )
    modin_df = pd.read_json(
        "modin/pandas/test/data/test_categories.json",
        dtype={"one": "int64", "two": "category"},
    )
    df_equals(modin_df, pandas_df)


def test_from_json_lines():
    setup_json_lines_file(SMALL_ROW_SIZE)

    pandas_df = pandas.read_json(TEST_JSON_FILENAME, lines=True)
    modin_df = pd.read_json(TEST_JSON_FILENAME, lines=True)
    df_equals(modin_df, pandas_df)

    teardown_json_file()


@pytest.mark.parametrize(
    "data",
    [json_short_string, json_short_bytes, json_long_string, json_long_bytes],
)
def test_read_json_string_bytes(data):
    with pytest.warns(UserWarning):
        modin_df = pd.read_json(data)
    # For I/O objects we need to rewind to reuse the same object.
    if hasattr(data, "seek"):
        data.seek(0)
    df_equals(modin_df, pandas.read_json(data))


def test_from_html():
    setup_html_file(SMALL_ROW_SIZE)

    pandas_df = pandas.read_html(TEST_HTML_FILENAME)[0]
    modin_df = pd.read_html(TEST_HTML_FILENAME)

    df_equals(modin_df, pandas_df)

    teardown_html_file()


@pytest.mark.skip(reason="No clipboard on Travis")
def test_from_clipboard():
    setup_clipboard(SMALL_ROW_SIZE)

    pandas_df = pandas.read_clipboard()
    modin_df = pd.read_clipboard()

    df_equals(modin_df, pandas_df)


@pytest.mark.xfail(reason="read_excel is broken for now, see #1733 for details")
def test_from_excel():
    setup_excel_file(SMALL_ROW_SIZE)

    pandas_df = pandas.read_excel(TEST_EXCEL_FILENAME)
    modin_df = pd.read_excel(TEST_EXCEL_FILENAME)

    df_equals(modin_df, pandas_df)

    teardown_excel_file()


def test_from_excel_engine():
    setup_excel_file(SMALL_ROW_SIZE)

    pandas_df = pandas.read_excel(TEST_EXCEL_FILENAME, engine="xlrd")
    with pytest.warns(UserWarning):
        modin_df = pd.read_excel(TEST_EXCEL_FILENAME, engine="xlrd")

    df_equals(modin_df, pandas_df)

    teardown_excel_file()


def test_from_excel_index_col():
    setup_excel_file(SMALL_ROW_SIZE)

    pandas_df = pandas.read_excel(TEST_EXCEL_FILENAME, index_col=0)
    with pytest.warns(UserWarning):
        modin_df = pd.read_excel(TEST_EXCEL_FILENAME, index_col=0)

    df_equals(modin_df, pandas_df)

    teardown_excel_file()


def test_from_excel_all_sheets():
    setup_excel_file(SMALL_ROW_SIZE)

    pandas_df = pandas.read_excel(TEST_EXCEL_FILENAME, sheet_name=None)
    modin_df = pd.read_excel(TEST_EXCEL_FILENAME, sheet_name=None)

    assert isinstance(pandas_df, (OrderedDict, dict))
    assert isinstance(modin_df, type(pandas_df))

    assert pandas_df.keys() == modin_df.keys()

    for key in pandas_df.keys():
        df_equals(modin_df.get(key), pandas_df.get(key))

    teardown_excel_file()


@pytest.mark.parametrize(
    "sheet_name",
    ["Sheet1", "AnotherSpecialName", "SpecialName", "SecondSpecialName", 0, 1, 2, 3],
)
def test_from_excel_sheet_name(sheet_name):
    fname = "modin/pandas/test/data/modin_error_book.xlsx"
    modin_df = pd.read_excel(fname, sheet_name=sheet_name)
    pandas_df = pandas.read_excel(fname, sheet_name=sheet_name)
    df_equals(modin_df, pandas_df)


# @pytest.mark.skip(reason="Arrow version mismatch between Pandas and Feather")
def test_from_feather():
    setup_feather_file(SMALL_ROW_SIZE)

    pandas_df = pandas.read_feather(TEST_FEATHER_FILENAME)
    modin_df = pd.read_feather(TEST_FEATHER_FILENAME)

    df_equals(modin_df, pandas_df)

    teardown_feather_file()


@pytest.mark.skipif(os.name == "nt", reason="Windows not supported")
def test_from_hdf():
    setup_hdf_file(SMALL_ROW_SIZE, format=None)

    pandas_df = pandas.read_hdf(TEST_READ_HDF_FILENAME, key="df")
    modin_df = pd.read_hdf(TEST_READ_HDF_FILENAME, key="df")

    df_equals(modin_df, pandas_df)

    teardown_hdf_file()


@pytest.mark.skipif(os.name == "nt", reason="Windows not supported")
def test_from_hdf_format():
    setup_hdf_file(SMALL_ROW_SIZE, format="table")

    pandas_df = pandas.read_hdf(TEST_READ_HDF_FILENAME, key="df")
    modin_df = pd.read_hdf(TEST_READ_HDF_FILENAME, key="df")

    df_equals(modin_df, pandas_df)

    teardown_hdf_file()


def test_from_stata():
    setup_stata_file(SMALL_ROW_SIZE)

    pandas_df = pandas.read_stata(TEST_STATA_FILENAME)
    modin_df = pd.read_stata(TEST_STATA_FILENAME)

    df_equals(modin_df, pandas_df)

    teardown_stata_file()


def test_from_pickle():
    setup_pickle_file(SMALL_ROW_SIZE)

    pandas_df = pandas.read_pickle(TEST_PICKLE_FILENAME)
    modin_df = pd.read_pickle(TEST_PICKLE_FILENAME)

    df_equals(modin_df, pandas_df)

    teardown_pickle_file()


def test_from_sql(make_sql_connection):
    filename = "test_from_sql.db"
    table = "test_from_sql"
    conn = make_sql_connection(filename, table)
    query = "select * from {0}".format(table)

    pandas_df = pandas.read_sql(query, conn)
    modin_df = pd.read_sql(query, conn)

    df_equals(modin_df, pandas_df)

    pandas_df = pandas.read_sql(query, conn, index_col="index")
    modin_df = pd.read_sql(query, conn, index_col="index")

    df_equals(modin_df, pandas_df)

    with pytest.warns(UserWarning):
        pd.read_sql_query(query, conn)

    with pytest.warns(UserWarning):
        pd.read_sql_table(table, conn)

    # Test SQLAlchemy engine
    conn = sa.create_engine(conn)
    pandas_df = pandas.read_sql(query, conn)
    modin_df = pd.read_sql(query, conn)

    df_equals(modin_df, pandas_df)

    # Test SQLAlchemy Connection
    conn = conn.connect()
    pandas_df = pandas.read_sql(query, conn)
    modin_df = pd.read_sql(query, conn)

    df_equals(modin_df, pandas_df)


def test_from_sql_with_chunksize(make_sql_connection):
    filename = "test_from_sql.db"
    table = "test_from_sql"
    conn = make_sql_connection(filename, table)
    query = "select * from {0}".format(table)

    pandas_gen = pandas.read_sql(query, conn, chunksize=10)
    modin_gen = pd.read_sql(query, conn, chunksize=10)
    for modin_df, pandas_df in zip(modin_gen, pandas_gen):
        df_equals(modin_df, pandas_df)


@pytest.mark.skip(reason="No SAS write methods in Pandas")
def test_from_sas():
    pandas_df = pandas.read_sas(TEST_SAS_FILENAME)
    modin_df = pd.read_sas(TEST_SAS_FILENAME)

    df_equals(modin_df, pandas_df)


@pytest.mark.skipif(
    Engine.get() == "Python",
    reason="S3-like path doesn't support in pandas with anonymous credentials. See issue #2301.",
)
def test_read_parquet_s3():
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


@pytest.mark.skip(reason="No clipboard on Travis")
def test_to_clipboard():
    modin_df = create_test_modin_dataframe()
    pandas_df = create_test_pandas_dataframe()

    modin_df.to_clipboard()
    modin_as_clip = pandas.read_clipboard()

    pandas_df.to_clipboard()
    pandas_as_clip = pandas.read_clipboard()

    assert modin_as_clip.equals(pandas_as_clip)


def test_dataframe_to_csv():
    modin_df = create_test_modin_dataframe()
    pandas_df = create_test_pandas_dataframe()

    TEST_CSV_DF_FILENAME = "test_df.csv"
    TEST_CSV_pandas_FILENAME = "test_pandas.csv"

    modin_df.to_csv(TEST_CSV_DF_FILENAME)
    pandas_df.to_csv(TEST_CSV_pandas_FILENAME)

    assert assert_files_eq(TEST_CSV_DF_FILENAME, TEST_CSV_pandas_FILENAME)

    teardown_test_file(TEST_CSV_pandas_FILENAME)
    teardown_test_file(TEST_CSV_DF_FILENAME)


def test_series_to_csv():
    modin_df = create_test_modin_dataframe()
    pandas_df = create_test_pandas_dataframe()

    TEST_CSV_DF_FILENAME = "test_df.csv"
    TEST_CSV_pandas_FILENAME = "test_pandas.csv"

    modin_s = modin_df["col1"]
    pandas_s = pandas_df["col1"]
    modin_s.to_csv(TEST_CSV_DF_FILENAME)
    pandas_s.to_csv(TEST_CSV_pandas_FILENAME)

    df_equals(modin_s, pandas_s)
    assert modin_s.name == pandas_s.name
    assert assert_files_eq(TEST_CSV_DF_FILENAME, TEST_CSV_pandas_FILENAME)

    teardown_test_file(TEST_CSV_pandas_FILENAME)
    teardown_test_file(TEST_CSV_DF_FILENAME)


@pytest.mark.skip(reason="Defaulting to Pandas")
def test_to_dense():
    modin_df = create_test_modin_dataframe()

    with pytest.raises(NotImplementedError):
        modin_df.to_dense()


def test_to_dict():
    modin_df = create_test_modin_dataframe()
    assert modin_df.to_dict() == to_pandas(modin_df).to_dict()


@pytest.mark.xfail(strict=False, reason="Flaky test, defaults to pandas")
def test_to_excel():
    modin_df = create_test_modin_dataframe()
    pandas_df = create_test_pandas_dataframe()

    TEST_EXCEL_DF_FILENAME = "test_df.xlsx"
    TEST_EXCEL_pandas_FILENAME = "test_pandas.xlsx"

    modin_writer = pandas.ExcelWriter(TEST_EXCEL_DF_FILENAME)
    pandas_writer = pandas.ExcelWriter(TEST_EXCEL_pandas_FILENAME)

    modin_df.to_excel(modin_writer)
    pandas_df.to_excel(pandas_writer)

    modin_writer.save()
    pandas_writer.save()

    assert assert_files_eq(TEST_EXCEL_DF_FILENAME, TEST_EXCEL_pandas_FILENAME)

    teardown_test_file(TEST_EXCEL_DF_FILENAME)
    teardown_test_file(TEST_EXCEL_pandas_FILENAME)


def test_to_feather():
    modin_df = create_test_modin_dataframe()
    pandas_df = create_test_pandas_dataframe()

    TEST_FEATHER_DF_FILENAME = "test_df.feather"
    TEST_FEATHER_pandas_FILENAME = "test_pandas.feather"

    modin_df.to_feather(TEST_FEATHER_DF_FILENAME)
    pandas_df.to_feather(TEST_FEATHER_pandas_FILENAME)

    assert assert_files_eq(TEST_FEATHER_DF_FILENAME, TEST_FEATHER_pandas_FILENAME)

    teardown_test_file(TEST_FEATHER_pandas_FILENAME)
    teardown_test_file(TEST_FEATHER_DF_FILENAME)


def test_to_html():
    modin_df = create_test_modin_dataframe()
    pandas_df = create_test_pandas_dataframe()

    TEST_HTML_DF_FILENAME = "test_df.html"
    TEST_HTML_pandas_FILENAME = "test_pandas.html"

    modin_df.to_html(TEST_HTML_DF_FILENAME)
    pandas_df.to_html(TEST_HTML_pandas_FILENAME)

    assert assert_files_eq(TEST_HTML_DF_FILENAME, TEST_HTML_pandas_FILENAME)

    teardown_test_file(TEST_HTML_pandas_FILENAME)
    teardown_test_file(TEST_HTML_DF_FILENAME)


def test_to_json():
    modin_df = create_test_modin_dataframe()
    pandas_df = create_test_pandas_dataframe()

    TEST_JSON_DF_FILENAME = "test_df.json"
    TEST_JSON_pandas_FILENAME = "test_pandas.json"

    modin_df.to_json(TEST_JSON_DF_FILENAME)
    pandas_df.to_json(TEST_JSON_pandas_FILENAME)

    assert assert_files_eq(TEST_JSON_DF_FILENAME, TEST_JSON_pandas_FILENAME)

    teardown_test_file(TEST_JSON_pandas_FILENAME)
    teardown_test_file(TEST_JSON_DF_FILENAME)


def test_to_latex():
    modin_df = create_test_modin_dataframe()
    assert modin_df.to_latex() == to_pandas(modin_df).to_latex()


def test_to_parquet():
    modin_df = create_test_modin_dataframe()
    pandas_df = create_test_pandas_dataframe()

    TEST_PARQUET_DF_FILENAME = "test_df.parquet"
    TEST_PARQUET_pandas_FILENAME = "test_pandas.parquet"

    modin_df.to_parquet(TEST_PARQUET_DF_FILENAME)
    pandas_df.to_parquet(TEST_PARQUET_pandas_FILENAME)

    assert assert_files_eq(TEST_PARQUET_DF_FILENAME, TEST_PARQUET_pandas_FILENAME)

    teardown_test_file(TEST_PARQUET_pandas_FILENAME)
    teardown_test_file(TEST_PARQUET_DF_FILENAME)


@pytest.mark.skip(reason="Defaulting to Pandas")
def test_to_period():
    modin_df = create_test_modin_dataframe()

    with pytest.raises(NotImplementedError):
        modin_df.to_period()


def test_to_pickle():
    modin_df = create_test_modin_dataframe()
    pandas_df = create_test_pandas_dataframe()

    TEST_PICKLE_DF_FILENAME = "test_df.pkl"
    TEST_PICKLE_pandas_FILENAME = "test_pandas.pkl"

    modin_df.to_pickle(TEST_PICKLE_DF_FILENAME)
    pandas_df.to_pickle(TEST_PICKLE_pandas_FILENAME)

    assert assert_files_eq(TEST_PICKLE_DF_FILENAME, TEST_PICKLE_pandas_FILENAME)

    teardown_test_file(TEST_PICKLE_pandas_FILENAME)
    teardown_test_file(TEST_PICKLE_DF_FILENAME)

    pd.to_pickle(modin_df, TEST_PICKLE_DF_FILENAME)
    pandas.to_pickle(pandas_df, TEST_PICKLE_pandas_FILENAME)

    assert assert_files_eq(TEST_PICKLE_DF_FILENAME, TEST_PICKLE_pandas_FILENAME)

    teardown_test_file(TEST_PICKLE_pandas_FILENAME)
    teardown_test_file(TEST_PICKLE_DF_FILENAME)


def test_to_sql_without_index(make_sql_connection):
    table_name = "tbl_without_index"
    modin_df = create_test_modin_dataframe()
    pandas_df = create_test_pandas_dataframe()

    # We do not pass the table name so the fixture won't generate a table
    conn = make_sql_connection("test_to_sql.db")
    modin_df.to_sql(table_name, conn, index=False)
    df_modin_sql = pandas.read_sql(table_name, con=conn)

    # We do not pass the table name so the fixture won't generate a table
    conn = make_sql_connection("test_to_sql_pandas.db")
    pandas_df.to_sql(table_name, conn, index=False)
    df_pandas_sql = pandas.read_sql(table_name, con=conn)

    assert df_modin_sql.sort_index().equals(df_pandas_sql.sort_index())


def test_to_sql_with_index(make_sql_connection):
    table_name = "tbl_with_index"
    modin_df = create_test_modin_dataframe()
    pandas_df = create_test_pandas_dataframe()

    # We do not pass the table name so the fixture won't generate a table
    conn = make_sql_connection("test_to_sql.db")
    modin_df.to_sql(table_name, conn)
    df_modin_sql = pandas.read_sql(table_name, con=conn, index_col="index")

    # We do not pass the table name so the fixture won't generate a table
    conn = make_sql_connection("test_to_sql_pandas.db")
    pandas_df.to_sql(table_name, conn)
    df_pandas_sql = pandas.read_sql(table_name, con=conn, index_col="index")

    assert df_modin_sql.sort_index().equals(df_pandas_sql.sort_index())


def test_to_stata():
    modin_df = create_test_modin_dataframe()
    pandas_df = create_test_pandas_dataframe()

    TEST_STATA_DF_FILENAME = "test_df.stata"
    TEST_STATA_pandas_FILENAME = "test_pandas.stata"

    modin_df.to_stata(TEST_STATA_DF_FILENAME)
    pandas_df.to_stata(TEST_STATA_pandas_FILENAME)

    assert assert_files_eq(TEST_STATA_DF_FILENAME, TEST_STATA_pandas_FILENAME)

    teardown_test_file(TEST_STATA_pandas_FILENAME)
    teardown_test_file(TEST_STATA_DF_FILENAME)


@pytest.mark.skipif(os.name == "nt", reason="Windows not supported")
def test_HDFStore():
    modin_store = pd.HDFStore(TEST_WRITE_HDF_FILENAME_MODIN)
    pandas_store = pandas.HDFStore(TEST_WRITE_HDF_FILENAME_PANDAS)

    modin_df = create_test_modin_dataframe()
    pandas_df = create_test_pandas_dataframe()

    modin_store["foo"] = modin_df
    pandas_store["foo"] = pandas_df

    assert assert_files_eq(
        TEST_WRITE_HDF_FILENAME_MODIN, TEST_WRITE_HDF_FILENAME_PANDAS
    )
    modin_df = modin_store.get("foo")
    pandas_df = pandas_store.get("foo")
    df_equals(modin_df, pandas_df)

    assert isinstance(modin_store, pd.HDFStore)

    hdf_file = "/tmp/test_read_hdf.hdf5"
    with pd.HDFStore(hdf_file, mode="w") as store:
        store.append("data/df1", pd.DataFrame(np.random.randn(5, 5)))
        store.append("data/df2", pd.DataFrame(np.random.randn(4, 4)))

    modin_df = pd.read_hdf(hdf_file, key="data/df1", mode="r")
    pandas_df = pandas.read_hdf(hdf_file, key="data/df1", mode="r")
    df_equals(modin_df, pandas_df)


def test_ExcelFile():
    setup_excel_file(SMALL_ROW_SIZE)

    modin_excel_file = pd.ExcelFile(TEST_EXCEL_FILENAME)
    pandas_excel_file = pandas.ExcelFile(TEST_EXCEL_FILENAME)

    df_equals(modin_excel_file.parse(), pandas_excel_file.parse())

    assert modin_excel_file.io == TEST_EXCEL_FILENAME
    assert isinstance(modin_excel_file, pd.ExcelFile)
    modin_excel_file.close()
    pandas_excel_file.close()

    teardown_excel_file()


def test_fwf_file():
    fwf_data = """id8141  360.242940  149.910199 11950.7
id1594  444.953632  166.985655 11788.4
id1849  364.136849  183.628767 11806.2
id1230  413.836124  184.375703 11916.8
id1948  502.953953  173.237159 12468.3"""

    setup_fwf_file(True, fwf_data=fwf_data)

    colspecs = [(0, 6), (8, 20), (21, 33), (34, 43)]
    df = pd.read_fwf(TEST_FWF_FILENAME, colspecs=colspecs, header=None, index_col=0)
    assert isinstance(df, pd.DataFrame)

    teardown_fwf_file()


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
def test_fwf_file_colspecs_widths(kwargs):
    setup_fwf_file(overwrite=True)

    modin_df = pd.read_fwf(TEST_FWF_FILENAME, **kwargs)
    pandas_df = pd.read_fwf(TEST_FWF_FILENAME, **kwargs)

    df_equals(modin_df, pandas_df)


@pytest.mark.parametrize("usecols", [["a"], ["a", "b", "d"], [0, 1, 3]])
def test_fwf_file_usecols(usecols):
    fwf_data = """a       b           c          d
id8141  360.242940  149.910199 11950.7
id1594  444.953632  166.985655 11788.4
id1849  364.136849  183.628767 11806.2
id1230  413.836124  184.375703 11916.8
id1948  502.953953  173.237159 12468.3"""

    setup_fwf_file(overwrite=True, fwf_data=fwf_data)

    pandas_df = pandas.read_fwf(TEST_FWF_FILENAME, usecols=usecols)
    modin_df = pd.read_fwf(TEST_FWF_FILENAME, usecols=usecols)

    df_equals(modin_df, pandas_df)

    teardown_fwf_file()


def test_fwf_file_chunksize():
    setup_fwf_file(overwrite=True)

    # Tests __next__ and correctness of reader as an iterator
    rdf_reader = pd.read_fwf(TEST_FWF_FILENAME, chunksize=5)
    pd_reader = pandas.read_fwf(TEST_FWF_FILENAME, chunksize=5)

    for modin_df, pd_df in zip(rdf_reader, pd_reader):
        df_equals(modin_df, pd_df)

    # Tests that get_chunk works correctly
    rdf_reader = pd.read_fwf(TEST_FWF_FILENAME, chunksize=1)
    pd_reader = pandas.read_fwf(TEST_FWF_FILENAME, chunksize=1)

    modin_df = rdf_reader.get_chunk(1)
    pd_df = pd_reader.get_chunk(1)

    df_equals(modin_df, pd_df)

    # Tests that read works correctly
    rdf_reader = pd.read_fwf(TEST_FWF_FILENAME, chunksize=1)
    pd_reader = pandas.read_fwf(TEST_FWF_FILENAME, chunksize=1)

    modin_df = rdf_reader.read()
    pd_df = pd_reader.read()

    df_equals(modin_df, pd_df)


@pytest.mark.parametrize("nrows", [13, None])
def test_fwf_file_skiprows(nrows):
    setup_fwf_file(overwrite=True)

    pandas_df = pandas.read_fwf(TEST_FWF_FILENAME, skiprows=2, nrows=nrows)
    modin_df = pd.read_fwf(TEST_FWF_FILENAME, skiprows=2, nrows=nrows)
    df_equals(modin_df, pandas_df)

    pandas_df = pandas.read_fwf(
        TEST_FWF_FILENAME, usecols=[0, 4, 7], skiprows=[2, 5], nrows=nrows
    )
    modin_df = pd.read_fwf(
        TEST_FWF_FILENAME, usecols=[0, 4, 7], skiprows=[2, 5], nrows=nrows
    )
    df_equals(modin_df, pandas_df)


def test_fwf_file_index_col():
    fwf_data = """a       b           c          d
id8141  360.242940  149.910199 11950.7
id1594  444.953632  166.985655 11788.4
id1849  364.136849  183.628767 11806.2
id1230  413.836124  184.375703 11916.8
id1948  502.953953  173.237159 12468.3"""

    setup_fwf_file(overwrite=True, fwf_data=fwf_data)

    pandas_df = pandas.read_fwf(TEST_FWF_FILENAME, index_col="c")
    modin_df = pd.read_fwf(TEST_FWF_FILENAME, index_col="c")
    df_equals(modin_df, pandas_df)

    teardown_fwf_file()


def test_fwf_file_skipfooter():
    setup_fwf_file(overwrite=True)

    pandas_df = pandas.read_fwf(TEST_FWF_FILENAME, skipfooter=2)
    modin_df = pd.read_fwf(TEST_FWF_FILENAME, skipfooter=2)

    df_equals(modin_df, pandas_df)


def test_fwf_file_parse_dates():
    dates = pandas.date_range("2000", freq="h", periods=10)
    fwf_data = "col1 col2        col3 col4"
    for i in range(10, 20):
        fwf_data = fwf_data + "\n{col1}   {col2}  {col3}   {col4}".format(
            col1=str(i),
            col2=str(dates[i - 10].date()),
            col3=str(i),
            col4=str(dates[i - 10].time()),
        )

    setup_fwf_file(overwrite=True, fwf_data=fwf_data)

    pandas_df = pandas.read_fwf(TEST_FWF_FILENAME, parse_dates=[["col2", "col4"]])
    modin_df = pd.read_fwf(TEST_FWF_FILENAME, parse_dates=[["col2", "col4"]])
    df_equals(modin_df, pandas_df)

    pandas_df = pandas.read_fwf(
        TEST_FWF_FILENAME, parse_dates={"time": ["col2", "col4"]}
    )
    modin_df = pd.read_fwf(TEST_FWF_FILENAME, parse_dates={"time": ["col2", "col4"]})
    df_equals(modin_df, pandas_df)

    teardown_fwf_file()


@pytest.mark.skip(reason="Need to verify GBQ access")
def test_from_gbq():
    # Test API, but do not supply credentials until credits can be secured.
    with pytest.raises(
        ValueError, match="Could not determine project ID and one was not supplied."
    ):
        pd.read_gbq("SELECT 1")


@pytest.mark.skip(reason="Need to verify GBQ access")
def test_to_gbq():
    modin_df = create_test_modin_dataframe()
    # Test API, but do not supply credentials until credits can be secured.
    with pytest.raises(
        ValueError, match="Could not determine project ID and one was not supplied."
    ):
        modin_df.to_gbq("modin.table")


def test_cleanup():
    filenames = [
        TEST_PARQUET_FILENAME,
        TEST_CSV_FILENAME,
        TEST_JSON_FILENAME,
        TEST_HTML_FILENAME,
        TEST_EXCEL_FILENAME,
        TEST_FEATHER_FILENAME,
        TEST_READ_HDF_FILENAME,
        TEST_WRITE_HDF_FILENAME_MODIN,
        TEST_WRITE_HDF_FILENAME_PANDAS,
        TEST_STATA_FILENAME,
        TEST_PICKLE_FILENAME,
        TEST_SAS_FILENAME,
        TEST_FWF_FILENAME,
        TEST_GBQ_FILENAME,
    ]
    for f in filenames:
        if os.path.exists(f):
            # Need try..except for Windows
            try:
                os.remove(f)
            except PermissionError:
                pass


def test_from_arrow():
    pandas_df = create_test_pandas_dataframe()
    modin_df = from_arrow(pa.Table.from_pandas(pandas_df))
    df_equals(modin_df, pandas_df)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"names": [5, 1, 3, 4, 2, 6]},
        {"names": [0]},
        {"names": None, "usecols": [1, 0, 2]},
        {"names": [3, 1, 2, 5], "usecols": [4, 1, 3, 2]},
    ],
)
def test_csv_names_neq_num_cols(kwargs):
    file_name = "modin/pandas/test/data/issue_2074.csv"
    pandas_df = pandas.read_csv(file_name, **kwargs)
    modin_df = pd.read_csv(file_name, **kwargs)
    df_equals(modin_df, pandas_df)
