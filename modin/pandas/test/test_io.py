from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pytest
import numpy as np
import pandas
from collections import OrderedDict
from modin.pandas.utils import to_pandas
from pathlib import Path
import pyarrow as pa
import os
import sys

from .utils import df_equals

from modin import __execution_engine__

if os.environ.get("MODIN_BACKEND", "Pandas").lower() == "pandas":
    import modin.pandas as pd
else:
    import modin.experimental.pandas as pd

# needed to resolve ray-project/ray#3744
pa.__version__ = "0.11.0"
pd.DEFAULT_NPARTITIONS = 4

PY2 = sys.version_info[0] == 2
TEST_PARQUET_FILENAME = "test.parquet"
TEST_CSV_FILENAME = "test.csv"
TEST_JSON_FILENAME = "test.json"
TEST_HTML_FILENAME = "test.html"
TEST_EXCEL_FILENAME = "test.xlsx"
TEST_FEATHER_FILENAME = "test.feather"
TEST_READ_HDF_FILENAME = "test.hdf"
TEST_WRITE_HDF_FILENAME_MODIN = "test_write_modin.hdf"
TEST_WRITE_HDF_FILENAME_PANDAS = "test_write_pandas.hdf"
TEST_MSGPACK_FILENAME = "test.msg"
TEST_STATA_FILENAME = "test.dta"
TEST_PICKLE_FILENAME = "test.pkl"
TEST_SAS_FILENAME = os.getcwd() + "/data/test1.sas7bdat"
TEST_FWF_FILENAME = "test_fwf.txt"
TEST_GBQ_FILENAME = "test_gbq."
SMALL_ROW_SIZE = 2000


def modin_df_equals_pandas(modin_df, pandas_df):
    df1 = to_pandas(modin_df).sort_index()
    df2 = pandas_df.sort_index()
    if os.environ.get("MODIN_BACKEND", "Pandas").lower() == "pyarrow":
        if not df1.dtypes.equals(df2.dtypes):
            return df2.astype(df1.dtypes).equals(df1)
    return df1.equals(df2)


def setup_parquet_file(row_size, force=False):
    if os.path.exists(TEST_PARQUET_FILENAME) and not force:
        pass
    else:
        pandas.DataFrame(
            {"col1": np.arange(row_size), "col2": np.arange(row_size)}
        ).to_parquet(TEST_PARQUET_FILENAME)


def create_test_ray_dataframe():
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


@pytest.fixture
def test_files_eq(path1, path2):
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


def teardown_parquet_file():
    if os.path.exists(TEST_PARQUET_FILENAME):
        os.remove(TEST_PARQUET_FILENAME)


@pytest.fixture
def make_csv_file(delimiter=","):
    """Pytest fixture factory that makes temp csv files for testing.

    Yields:
        Function that generates csv files
    """
    filenames = []

    def _make_csv_file(
        filename=TEST_CSV_FILENAME,
        row_size=SMALL_ROW_SIZE,
        force=False,
        delimiter=delimiter,
        encoding=None,
    ):
        if os.path.exists(filename) and not force:
            pass
        else:
            dates = pandas.date_range("2000", freq="h", periods=row_size)
            df = pandas.DataFrame(
                {
                    "col1": np.arange(row_size),
                    "col2": [str(x.date()) for x in dates],
                    "col3": np.arange(row_size),
                    "col4": [str(x.time()) for x in dates],
                }
            )
            df.to_csv(filename, sep=delimiter, encoding=encoding)
            filenames.append(filename)
            return df

    # Return function that generates csv files
    yield _make_csv_file

    # Delete csv files that were created
    for filename in filenames:
        if os.path.exists(filename):
            os.remove(filename)


def setup_json_file(row_size, force=False):
    if os.path.exists(TEST_JSON_FILENAME) and not force:
        pass
    else:
        df = pandas.DataFrame(
            {"col1": np.arange(row_size), "col2": np.arange(row_size)}
        )
        df.to_json(TEST_JSON_FILENAME)


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
        os.remove(TEST_EXCEL_FILENAME)


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


def setup_msgpack_file(row_size, force=False):
    if os.path.exists(TEST_MSGPACK_FILENAME) and not force:
        pass
    else:
        df = pandas.DataFrame(
            {"col1": np.arange(row_size), "col2": np.arange(row_size)}
        )
        df.to_msgpack(TEST_MSGPACK_FILENAME)


def teardown_msgpack_file():
    if os.path.exists(TEST_MSGPACK_FILENAME):
        os.remove(TEST_MSGPACK_FILENAME)


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


def setup_fwf_file():
    if os.path.exists(TEST_FWF_FILENAME):
        return

    fwf_data = """id8141    360.242940  149.910199  11950.7
    id1594  444.953632  166.985655 11788.4
    id1849  364.136849  183.628767 11806.2
    id1230  413.836124  184.375703 11916.8
    id1948  502.953953  173.237159 12468.3"""

    with open(TEST_FWF_FILENAME, "w") as f:
        f.write(fwf_data)


def teardown_fwf_file():
    if os.path.exists(TEST_FWF_FILENAME):
        os.remove(TEST_FWF_FILENAME)


def test_from_parquet():
    setup_parquet_file(SMALL_ROW_SIZE)

    pandas_df = pandas.read_parquet(TEST_PARQUET_FILENAME)
    modin_df = pd.read_parquet(TEST_PARQUET_FILENAME)
    assert modin_df_equals_pandas(modin_df, pandas_df)

    teardown_parquet_file()


def test_from_parquet_with_columns():
    setup_parquet_file(SMALL_ROW_SIZE)

    pandas_df = pandas.read_parquet(TEST_PARQUET_FILENAME, columns=["col1"])
    modin_df = pd.read_parquet(TEST_PARQUET_FILENAME, columns=["col1"])
    assert modin_df_equals_pandas(modin_df, pandas_df)

    teardown_parquet_file()


def test_from_json():
    setup_json_file(SMALL_ROW_SIZE)

    pandas_df = pandas.read_json(TEST_JSON_FILENAME)
    modin_df = pd.read_json(TEST_JSON_FILENAME)

    assert modin_df_equals_pandas(modin_df, pandas_df)

    teardown_json_file()


def test_from_html():
    setup_html_file(SMALL_ROW_SIZE)

    pandas_df = pandas.read_html(TEST_HTML_FILENAME)[0]
    modin_df = pd.read_html(TEST_HTML_FILENAME)

    assert modin_df_equals_pandas(modin_df, pandas_df)

    teardown_html_file()


@pytest.mark.skip(reason="No clipboard on Travis")
def test_from_clipboard():
    setup_clipboard(SMALL_ROW_SIZE)

    pandas_df = pandas.read_clipboard()
    modin_df = pd.read_clipboard()

    assert modin_df_equals_pandas(modin_df, pandas_df)


def test_from_excel():
    setup_excel_file(SMALL_ROW_SIZE)

    pandas_df = pandas.read_excel(TEST_EXCEL_FILENAME)
    modin_df = pd.read_excel(TEST_EXCEL_FILENAME)

    assert modin_df_equals_pandas(modin_df, pandas_df)

    teardown_excel_file()


def test_from_excel_all_sheets():
    setup_excel_file(SMALL_ROW_SIZE)

    pandas_df = pandas.read_excel(TEST_EXCEL_FILENAME, sheet_name=None)
    modin_df = pd.read_excel(TEST_EXCEL_FILENAME, sheet_name=None)

    assert isinstance(pandas_df, OrderedDict)
    assert isinstance(modin_df, OrderedDict)

    assert pandas_df.keys() == modin_df.keys()

    for key in pandas_df.keys():
        assert modin_df_equals_pandas(modin_df.get(key), pandas_df.get(key))

    teardown_excel_file()


# @pytest.mark.skip(reason="Arrow version mismatch between Pandas and Feather")
def test_from_feather():
    setup_feather_file(SMALL_ROW_SIZE)

    pandas_df = pandas.read_feather(TEST_FEATHER_FILENAME)
    modin_df = pd.read_feather(TEST_FEATHER_FILENAME)

    assert modin_df_equals_pandas(modin_df, pandas_df)

    teardown_feather_file()


def test_from_hdf():
    setup_hdf_file(SMALL_ROW_SIZE, format=None)

    pandas_df = pandas.read_hdf(TEST_READ_HDF_FILENAME, key="df")
    modin_df = pd.read_hdf(TEST_READ_HDF_FILENAME, key="df")

    assert modin_df_equals_pandas(modin_df, pandas_df)

    teardown_hdf_file()


def test_from_hdf_format():
    setup_hdf_file(SMALL_ROW_SIZE, format="table")

    pandas_df = pandas.read_hdf(TEST_READ_HDF_FILENAME, key="df")
    modin_df = pd.read_hdf(TEST_READ_HDF_FILENAME, key="df")

    assert modin_df_equals_pandas(modin_df, pandas_df)

    teardown_hdf_file()


def test_from_msgpack():
    setup_msgpack_file(SMALL_ROW_SIZE)

    pandas_df = pandas.read_msgpack(TEST_MSGPACK_FILENAME)
    modin_df = pd.read_msgpack(TEST_MSGPACK_FILENAME)

    assert modin_df_equals_pandas(modin_df, pandas_df)

    teardown_msgpack_file()


def test_from_stata():
    setup_stata_file(SMALL_ROW_SIZE)

    pandas_df = pandas.read_stata(TEST_STATA_FILENAME)
    modin_df = pd.read_stata(TEST_STATA_FILENAME)

    assert modin_df_equals_pandas(modin_df, pandas_df)

    teardown_stata_file()


def test_from_pickle():
    setup_pickle_file(SMALL_ROW_SIZE)

    pandas_df = pandas.read_pickle(TEST_PICKLE_FILENAME)
    modin_df = pd.read_pickle(TEST_PICKLE_FILENAME)

    assert modin_df_equals_pandas(modin_df, pandas_df)

    teardown_pickle_file()


def test_from_sql(make_sql_connection):
    filename = "test_from_sql.db"
    table = "test_from_sql"
    conn = make_sql_connection(filename, table)
    query = "select * from {0}".format(table)

    pandas_df = pandas.read_sql(query, conn)
    modin_df = pd.read_sql(query, conn)

    assert modin_df_equals_pandas(modin_df, pandas_df)

    pandas_df = pandas.read_sql(query, conn, index_col="index")
    modin_df = pd.read_sql(query, conn, index_col="index")

    assert modin_df_equals_pandas(modin_df, pandas_df)

    with pytest.warns(UserWarning):
        pd.read_sql_query(query, conn)

    with pytest.warns(UserWarning):
        pd.read_sql_table(table, conn)


@pytest.mark.skip(reason="No SAS write methods in Pandas")
def test_from_sas():
    pandas_df = pandas.read_sas(TEST_SAS_FILENAME)
    modin_df = pd.read_sas(TEST_SAS_FILENAME)

    assert modin_df_equals_pandas(modin_df, pandas_df)


def test_from_csv(make_csv_file):
    make_csv_file()

    pandas_df = pandas.read_csv(TEST_CSV_FILENAME)
    modin_df = pd.read_csv(TEST_CSV_FILENAME)

    assert modin_df_equals_pandas(modin_df, pandas_df)

    if not PY2:
        pandas_df = pandas.read_csv(Path(TEST_CSV_FILENAME))
        modin_df = pd.read_csv(Path(TEST_CSV_FILENAME))

        assert modin_df_equals_pandas(modin_df, pandas_df)


def test_parse_dates_read_csv():
    pandas_df = pandas.read_csv("modin/pandas/test/data/test_time_parsing.csv")
    modin_df = pd.read_csv("modin/pandas/test/data/test_time_parsing.csv")
    modin_df_equals_pandas(modin_df, pandas_df)

    pandas_df = pandas.read_csv(
        "modin/pandas/test/data/test_time_parsing.csv",
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
    modin_df = pd.read_csv(
        "modin/pandas/test/data/test_time_parsing.csv",
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
    modin_df_equals_pandas(modin_df, pandas_df)

    pandas_df = pandas.read_csv(
        "modin/pandas/test/data/test_time_parsing.csv",
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
    modin_df = pd.read_csv(
        "modin/pandas/test/data/test_time_parsing.csv",
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
    modin_df_equals_pandas(modin_df, pandas_df)

    pandas_df = pandas.read_csv(
        "modin/pandas/test/data/test_time_parsing.csv",
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
    modin_df = pd.read_csv(
        "modin/pandas/test/data/test_time_parsing.csv",
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
    modin_df_equals_pandas(modin_df, pandas_df)


def test_from_table(make_csv_file):
    make_csv_file(delimiter="\t")

    pandas_df = pandas.read_table(TEST_CSV_FILENAME)
    modin_df = pd.read_table(TEST_CSV_FILENAME)

    assert modin_df_equals_pandas(modin_df, pandas_df)

    if not PY2:
        pandas_df = pandas.read_table(Path(TEST_CSV_FILENAME))
        modin_df = pd.read_table(Path(TEST_CSV_FILENAME))

        assert modin_df_equals_pandas(modin_df, pandas_df)


class FakeS3FS:
    def exists(self, path):
        return "s3://bucket/path.csv" == path

    def open(self, path, mode="rb"):
        if "s3://bucket/path.csv" == path:
            return open(TEST_CSV_FILENAME, mode=mode)
        else:
            raise Exception("You shouldn't access that! (%s)" % path)


@pytest.mark.skipif(
    __execution_engine__.lower() == "python", reason="Using pandas implementation"
)
def test_from_csv_s3(make_csv_file):
    from modin.engines.ray.generic import io

    io.s3fs = FakeS3FS()

    make_csv_file()

    pandas_df = pandas.read_csv(TEST_CSV_FILENAME)
    modin_df = pd.read_csv("s3://bucket/path.csv")

    assert modin_df_equals_pandas(modin_df, pandas_df)


def test_from_csv_chunksize(make_csv_file):
    make_csv_file()

    # Tests __next__ and correctness of reader as an iterator
    # Use larger chunksize to read through file quicker
    rdf_reader = pd.read_csv(TEST_CSV_FILENAME, chunksize=500)
    pd_reader = pandas.read_csv(TEST_CSV_FILENAME, chunksize=500)

    for modin_df, pd_df in zip(rdf_reader, pd_reader):
        assert modin_df_equals_pandas(modin_df, pd_df)

    # Tests that get_chunk works correctly
    rdf_reader = pd.read_csv(TEST_CSV_FILENAME, chunksize=1)
    pd_reader = pandas.read_csv(TEST_CSV_FILENAME, chunksize=1)

    modin_df = rdf_reader.get_chunk(1)
    pd_df = pd_reader.get_chunk(1)

    assert modin_df_equals_pandas(modin_df, pd_df)

    # Tests that read works correctly
    rdf_reader = pd.read_csv(TEST_CSV_FILENAME, chunksize=1)
    pd_reader = pandas.read_csv(TEST_CSV_FILENAME, chunksize=1)

    modin_df = rdf_reader.read()
    pd_df = pd_reader.read()

    assert modin_df_equals_pandas(modin_df, pd_df)


def test_from_csv_delimiter(make_csv_file):
    make_csv_file(delimiter="|")

    pandas_df = pandas.read_csv(TEST_CSV_FILENAME, sep="|")
    modin_df = pd.read_csv(TEST_CSV_FILENAME, sep="|")

    assert modin_df_equals_pandas(modin_df, pandas_df)

    modin_df = pd.DataFrame.from_csv(
        TEST_CSV_FILENAME, sep="|", parse_dates=False, header="infer", index_col=None
    )
    pandas_df = pandas.DataFrame.from_csv(
        TEST_CSV_FILENAME, sep="|", parse_dates=False, header="infer", index_col=None
    )
    assert modin_df_equals_pandas(modin_df, pandas_df)


def test_from_csv_skiprows(make_csv_file):
    make_csv_file()

    pandas_df = pandas.read_csv(TEST_CSV_FILENAME, skiprows=2)
    modin_df = pd.read_csv(TEST_CSV_FILENAME, skiprows=2)

    assert modin_df_equals_pandas(modin_df, pandas_df)


def test_from_csv_encoding(make_csv_file):
    make_csv_file(encoding="latin8")

    pandas_df = pandas.read_csv(TEST_CSV_FILENAME, encoding="latin8")
    modin_df = pd.read_csv(TEST_CSV_FILENAME, encoding="latin8")

    assert modin_df_equals_pandas(modin_df, pandas_df)


def test_from_csv_default_to_pandas_behavior(make_csv_file):
    make_csv_file()

    with pytest.warns(UserWarning):
        # Test nrows
        pd.read_csv(TEST_CSV_FILENAME, nrows=10)

    if not PY2:
        with pytest.warns(UserWarning):
            # This tests that we default to pandas on a buffer
            from io import StringIO

            pd.read_csv(StringIO(open(TEST_CSV_FILENAME, "r").read()))

    with pytest.warns(UserWarning):
        pd.read_csv(TEST_CSV_FILENAME, skiprows=lambda x: x in [0, 2])


def test_from_csv_index_col(make_csv_file):
    make_csv_file()

    pandas_df = pandas.read_csv(TEST_CSV_FILENAME, index_col="col1")
    modin_df = pd.read_csv(TEST_CSV_FILENAME, index_col="col1")

    assert modin_df_equals_pandas(modin_df, pandas_df)


def test_from_csv_skipfooter(make_csv_file):
    make_csv_file()

    pandas_df = pandas.read_csv(TEST_CSV_FILENAME, skipfooter=13)
    modin_df = pd.read_csv(TEST_CSV_FILENAME, skipfooter=13)

    assert modin_df_equals_pandas(modin_df, pandas_df)


def test_from_csv_parse_dates(make_csv_file):
    make_csv_file(force=True)

    pandas_df = pandas.read_csv(TEST_CSV_FILENAME, parse_dates=[["col2", "col4"]])
    modin_df = pd.read_csv(TEST_CSV_FILENAME, parse_dates=[["col2", "col4"]])
    assert modin_df_equals_pandas(modin_df, pandas_df)

    pandas_df = pandas.read_csv(
        TEST_CSV_FILENAME, parse_dates={"time": ["col2", "col4"]}
    )
    modin_df = pd.read_csv(TEST_CSV_FILENAME, parse_dates={"time": ["col2", "col4"]})
    assert modin_df_equals_pandas(modin_df, pandas_df)


@pytest.mark.skip(reason="No clipboard on Travis")
def test_to_clipboard():
    modin_df = create_test_ray_dataframe()
    pandas_df = create_test_pandas_dataframe()

    modin_df.to_clipboard()
    modin_as_clip = pandas.read_clipboard()

    pandas_df.to_clipboard()
    pandas_as_clip = pandas.read_clipboard()

    assert modin_as_clip.equals(pandas_as_clip)


def test_to_csv():
    modin_df = create_test_ray_dataframe()
    pandas_df = create_test_pandas_dataframe()

    TEST_CSV_DF_FILENAME = "test_df.csv"
    TEST_CSV_pandas_FILENAME = "test_pandas.csv"

    modin_df.to_csv(TEST_CSV_DF_FILENAME)
    pandas_df.to_csv(TEST_CSV_pandas_FILENAME)

    assert test_files_eq(TEST_CSV_DF_FILENAME, TEST_CSV_pandas_FILENAME)

    teardown_test_file(TEST_CSV_pandas_FILENAME)
    teardown_test_file(TEST_CSV_DF_FILENAME)


@pytest.mark.skip(reason="Defaulting to Pandas")
def test_to_dense():
    modin_df = create_test_ray_dataframe()

    with pytest.raises(NotImplementedError):
        modin_df.to_dense()


def test_to_dict():
    modin_df = create_test_ray_dataframe()
    assert modin_df.to_dict() == to_pandas(modin_df).to_dict()


@pytest.mark.xfail(strict=False, reason="Flaky test, defaults to pandas")
def test_to_excel():
    modin_df = create_test_ray_dataframe()
    pandas_df = create_test_pandas_dataframe()

    TEST_EXCEL_DF_FILENAME = "test_df.xlsx"
    TEST_EXCEL_pandas_FILENAME = "test_pandas.xlsx"

    modin_writer = pandas.ExcelWriter(TEST_EXCEL_DF_FILENAME)
    pandas_writer = pandas.ExcelWriter(TEST_EXCEL_pandas_FILENAME)

    modin_df.to_excel(modin_writer)
    pandas_df.to_excel(pandas_writer)

    modin_writer.save()
    pandas_writer.save()

    assert test_files_eq(TEST_EXCEL_DF_FILENAME, TEST_EXCEL_pandas_FILENAME)

    teardown_test_file(TEST_EXCEL_DF_FILENAME)
    teardown_test_file(TEST_EXCEL_pandas_FILENAME)


def test_to_feather():
    modin_df = create_test_ray_dataframe()
    pandas_df = create_test_pandas_dataframe()

    TEST_FEATHER_DF_FILENAME = "test_df.feather"
    TEST_FEATHER_pandas_FILENAME = "test_pandas.feather"

    modin_df.to_feather(TEST_FEATHER_DF_FILENAME)
    pandas_df.to_feather(TEST_FEATHER_pandas_FILENAME)

    assert test_files_eq(TEST_FEATHER_DF_FILENAME, TEST_FEATHER_pandas_FILENAME)

    teardown_test_file(TEST_FEATHER_pandas_FILENAME)
    teardown_test_file(TEST_FEATHER_DF_FILENAME)


def test_to_gbq():
    modin_df = create_test_ray_dataframe()
    pandas_df = create_test_pandas_dataframe()
    # Because we default to pandas, we can just test the equality of the two frames.
    assert to_pandas(modin_df).equals(pandas_df)


def test_to_html():
    modin_df = create_test_ray_dataframe()
    pandas_df = create_test_pandas_dataframe()

    TEST_HTML_DF_FILENAME = "test_df.html"
    TEST_HTML_pandas_FILENAME = "test_pandas.html"

    modin_df.to_html(TEST_HTML_DF_FILENAME)
    pandas_df.to_html(TEST_HTML_pandas_FILENAME)

    assert test_files_eq(TEST_HTML_DF_FILENAME, TEST_HTML_pandas_FILENAME)

    teardown_test_file(TEST_HTML_pandas_FILENAME)
    teardown_test_file(TEST_HTML_DF_FILENAME)


def test_to_json():
    modin_df = create_test_ray_dataframe()
    pandas_df = create_test_pandas_dataframe()

    TEST_JSON_DF_FILENAME = "test_df.json"
    TEST_JSON_pandas_FILENAME = "test_pandas.json"

    modin_df.to_json(TEST_JSON_DF_FILENAME)
    pandas_df.to_json(TEST_JSON_pandas_FILENAME)

    assert test_files_eq(TEST_JSON_DF_FILENAME, TEST_JSON_pandas_FILENAME)

    teardown_test_file(TEST_JSON_pandas_FILENAME)
    teardown_test_file(TEST_JSON_DF_FILENAME)


def test_to_latex():
    modin_df = create_test_ray_dataframe()
    assert modin_df.to_latex() == to_pandas(modin_df).to_latex()


def test_to_msgpack():
    modin_df = create_test_ray_dataframe()
    pandas_df = create_test_pandas_dataframe()

    TEST_MSGPACK_DF_FILENAME = "test_df.msgpack"
    TEST_MSGPACK_pandas_FILENAME = "test_pandas.msgpack"

    modin_df.to_msgpack(TEST_MSGPACK_DF_FILENAME)
    pandas_df.to_msgpack(TEST_MSGPACK_pandas_FILENAME)

    assert test_files_eq(TEST_MSGPACK_DF_FILENAME, TEST_MSGPACK_pandas_FILENAME)

    teardown_test_file(TEST_MSGPACK_pandas_FILENAME)
    teardown_test_file(TEST_MSGPACK_DF_FILENAME)


def test_to_panel():
    modin_df = create_test_ray_dataframe()

    with pytest.raises(NotImplementedError):
        modin_df.to_panel()


def test_to_parquet():
    modin_df = create_test_ray_dataframe()
    pandas_df = create_test_pandas_dataframe()

    TEST_PARQUET_DF_FILENAME = "test_df.parquet"
    TEST_PARQUET_pandas_FILENAME = "test_pandas.parquet"

    modin_df.to_parquet(TEST_PARQUET_DF_FILENAME)
    pandas_df.to_parquet(TEST_PARQUET_pandas_FILENAME)

    assert test_files_eq(TEST_PARQUET_DF_FILENAME, TEST_PARQUET_pandas_FILENAME)

    teardown_test_file(TEST_PARQUET_pandas_FILENAME)
    teardown_test_file(TEST_PARQUET_DF_FILENAME)


@pytest.mark.skip(reason="Defaulting to Pandas")
def test_to_period():
    modin_df = create_test_ray_dataframe()

    with pytest.raises(NotImplementedError):
        modin_df.to_period()


def test_to_pickle():
    modin_df = create_test_ray_dataframe()
    pandas_df = create_test_pandas_dataframe()

    TEST_PICKLE_DF_FILENAME = "test_df.pkl"
    TEST_PICKLE_pandas_FILENAME = "test_pandas.pkl"

    modin_df.to_pickle(TEST_PICKLE_DF_FILENAME)
    pandas_df.to_pickle(TEST_PICKLE_pandas_FILENAME)

    assert test_files_eq(TEST_PICKLE_DF_FILENAME, TEST_PICKLE_pandas_FILENAME)

    teardown_test_file(TEST_PICKLE_pandas_FILENAME)
    teardown_test_file(TEST_PICKLE_DF_FILENAME)

    pd.to_pickle(modin_df, TEST_PICKLE_DF_FILENAME)
    pandas.to_pickle(pandas_df, TEST_PICKLE_pandas_FILENAME)

    assert test_files_eq(TEST_PICKLE_DF_FILENAME, TEST_PICKLE_pandas_FILENAME)

    teardown_test_file(TEST_PICKLE_pandas_FILENAME)
    teardown_test_file(TEST_PICKLE_DF_FILENAME)


def test_to_sql_without_index(make_sql_connection):
    table_name = "tbl_without_index"
    modin_df = create_test_ray_dataframe()
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
    modin_df = create_test_ray_dataframe()
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
    modin_df = create_test_ray_dataframe()
    pandas_df = create_test_pandas_dataframe()

    TEST_STATA_DF_FILENAME = "test_df.stata"
    TEST_STATA_pandas_FILENAME = "test_pandas.stata"

    modin_df.to_stata(TEST_STATA_DF_FILENAME)
    pandas_df.to_stata(TEST_STATA_pandas_FILENAME)

    assert test_files_eq(TEST_STATA_DF_FILENAME, TEST_STATA_pandas_FILENAME)

    teardown_test_file(TEST_STATA_pandas_FILENAME)
    teardown_test_file(TEST_STATA_DF_FILENAME)


def test_HDFStore():
    modin_store = pd.HDFStore(TEST_WRITE_HDF_FILENAME_MODIN)
    pandas_store = pandas.HDFStore(TEST_WRITE_HDF_FILENAME_PANDAS)

    modin_df = create_test_ray_dataframe()
    pandas_df = create_test_pandas_dataframe()

    modin_store["foo"] = modin_df
    pandas_store["foo"] = pandas_df

    assert test_files_eq(TEST_WRITE_HDF_FILENAME_MODIN, TEST_WRITE_HDF_FILENAME_PANDAS)
    modin_df = modin_store.get("foo")
    pandas_df = pandas_store.get("foo")
    df_equals(modin_df, pandas_df)

    assert isinstance(modin_store, pd.HDFStore)


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
    setup_fwf_file()

    colspecs = [(0, 6), (8, 20), (21, 33), (34, 43)]
    with pytest.warns(UserWarning):
        df = pd.read_fwf(TEST_FWF_FILENAME, colspecs=colspecs, header=None, index_col=0)
        assert isinstance(df, pd.DataFrame)

    teardown_fwf_file()
