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


"""
Module houses Modin parser classes, that are used for data parsing on the workers.

Notes
-----
Data parsing mechanism differs depending on the data format type:

* text format type (CSV, EXCEL, FWF, JSON):
  File parsing begins from retrieving `start` and `end` parameters from `parse`
  kwargs - these parameters define start and end bytes of data file, that should
  be read in the concrete partition. Using this data and file handle got from
  `fname`, binary data is read by python `read` function. Then resulting data is passed
  into `pandas.read_*` function as `io.BytesIO` object to get corresponding
  `pandas.DataFrame` (we need to do this because Modin partitions internally stores data
  as `pandas.DataFrame`).

* columnar store type (FEATHER, HDF, PARQUET):
  In this case data chunk to be read is defined by columns names passed as `columns`
  parameter as part of `parse` kwargs, so no additional action is needed and `fname`
  and `kwargs` are just passed into `pandas.read_*` function (in some corner cases
  `pyarrow.read_*` function can be used).

* SQL type:
  Chunking is incorporated in the `sql` parameter as part of query, so `parse`
  parameters are passed into `pandas.read_sql` function without modification.
"""

from collections import OrderedDict
from io import BytesIO, TextIOWrapper
import numpy as np
import pandas
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.concat import union_categoricals
from pandas.io.common import infer_compression
from pandas.util._decorators import doc
import warnings

from modin.core.io.file_dispatcher import OpenFile
from modin.db_conn import ModinDatabaseConnection
from modin.core.storage_formats.pandas.utils import split_result_of_axis_func_pandas
from modin.error_message import ErrorMessage
from modin.logging import LoggerMetaClass

_doc_pandas_parser_class = """
Class for handling {data_type} on the workers using pandas storage format.

Inherits common functions from `PandasParser` class.
"""

_doc_parse_func = """
Parse data on the workers.

Parameters
----------
{parameters}
**kwargs : dict
    Keywords arguments to be used by `parse` function or
    passed into `read_*` function.

Returns
-------
list
    List with split parse results and it's metadata
    (index, dtypes, etc.).
"""

_doc_parse_parameters_common = """fname : str or path object
    Name of the file or path to read."""


def _split_result_for_readers(axis, num_splits, df):  # pragma: no cover
    """
    Split the read DataFrame into smaller DataFrames and handle all edge cases.

    Parameters
    ----------
    axis : int
        The axis to split across (0 - index, 1 - columns).
    num_splits : int
        The number of splits to create.
    df : pandas.DataFrame
        `pandas.DataFrame` to split.

    Returns
    -------
    list
        A list of pandas DataFrames.
    """
    splits = split_result_of_axis_func_pandas(axis, num_splits, df)
    if not isinstance(splits, list):
        splits = [splits]
    return splits


def find_common_type_cat(types):
    """
    Find a common data type among the given dtypes.

    Parameters
    ----------
    types : array-like
        Array of dtypes.

    Returns
    -------
    pandas.core.dtypes.dtypes.ExtensionDtype or
    np.dtype or
    None
        `dtype` that is common for all passed `types`.
    """
    if all(isinstance(t, pandas.CategoricalDtype) for t in types):
        if all(t.ordered for t in types):
            categories = np.sort(np.unique([c for t in types for c in t.categories]))
            return pandas.CategoricalDtype(
                categories,
                ordered=True,
            )
        return union_categoricals(
            [pandas.Categorical([], dtype=t) for t in types],
            sort_categories=all(t.ordered for t in types),
        ).dtype
    else:
        return find_common_type(list(types))


class PandasParser(object, metaclass=LoggerMetaClass):
    """Base class for parser classes with pandas storage format."""

    @staticmethod
    @doc(_doc_parse_func, parameters=_doc_parse_parameters_common)
    def generic_parse(fname, **kwargs):
        warnings.filterwarnings("ignore")
        num_splits = kwargs.pop("num_splits", None)
        start = kwargs.pop("start", None)
        end = kwargs.pop("end", None)
        header_size = kwargs.pop("header_size", 0)
        encoding = kwargs.get("encoding", None)
        callback = kwargs.pop("callback")
        if start is None or end is None:
            # This only happens when we are reading with only one worker (Default)
            return callback(fname, **kwargs)

        # pop "compression" from kwargs because bio is uncompressed
        with OpenFile(
            fname,
            "rb",
            kwargs.pop("compression", "infer"),
            **(kwargs.pop("storage_options", None) or {}),
        ) as bio:
            header = b""
            # In this case we beware that first line can contain BOM, so
            # adding this line to the `header` for reading and then skip it
            if encoding and (
                "utf" in encoding
                and "8" not in encoding
                or encoding == "unicode_escape"
                or encoding.replace("-", "_") == "utf_8_sig"
            ):
                # do not 'close' the wrapper - underlying buffer is managed by `bio` handle
                fio = TextIOWrapper(bio, encoding=encoding, newline="")
                if header_size == 0:
                    header = fio.readline().encode(encoding)
                    kwargs["skiprows"] = 1
                for _ in range(header_size):
                    header += fio.readline().encode(encoding)
            elif encoding is not None:
                if header_size == 0:
                    header = bio.readline()
                    # `skiprows` can be only None here, so don't check it's type
                    # and just set to 1
                    kwargs["skiprows"] = 1
                for _ in range(header_size):
                    header += bio.readline()
            else:
                for _ in range(header_size):
                    header += bio.readline()

            bio.seek(start)
            to_read = header + bio.read(end - start)
        if "memory_map" in kwargs:
            kwargs = kwargs.copy()
            del kwargs["memory_map"]
        pandas_df = callback(BytesIO(to_read), **kwargs)
        index = (
            pandas_df.index
            if not isinstance(pandas_df.index, pandas.RangeIndex)
            else len(pandas_df)
        )
        return _split_result_for_readers(1, num_splits, pandas_df) + [
            index,
            pandas_df.dtypes,
        ]

    @classmethod
    def get_dtypes(cls, dtypes_ids):
        """
        Get common for all partitions dtype for each of the columns.

        Parameters
        ----------
        dtypes_ids : list
            Array with references to the partitions dtypes objects.

        Returns
        -------
        frame_dtypes : pandas.Series or dtype
            Resulting dtype or pandas.Series where column names are used as
            index and types of columns are used as values for full resulting
            frame.
        """
        # each element in `partitions_dtypes` is a Series, where column names are
        # used as index and types of columns for different partitions are used as values
        partitions_dtypes = cls.materialize(dtypes_ids)
        if all([len(dtype) == 0 for dtype in partitions_dtypes]):
            return None

        combined_part_dtypes = pandas.concat(partitions_dtypes, axis=1)
        frame_dtypes = combined_part_dtypes.iloc[:, 0]

        if not combined_part_dtypes.eq(frame_dtypes, axis=0).all(axis=None):
            ErrorMessage.missmatch_with_pandas(
                operation="read_*",
                message="Data types of partitions are different! "
                + "Please refer to the troubleshooting section of the Modin documentation "
                + "to fix this issue",
            )

            # concat all elements of `partitions_dtypes` and find common dtype
            # for each of the column among all partitions
            frame_dtypes = combined_part_dtypes.apply(
                lambda row: find_common_type_cat(row.values),
                axis=1,
            ).squeeze(axis=0)

        return frame_dtypes

    @classmethod
    def single_worker_read(cls, fname, **kwargs):
        """
        Perform reading by single worker (default-to-pandas implementation).

        Parameters
        ----------
        fname : str, path object or file-like object
            Name of the file or file-like object to read.
        **kwargs : dict
            Keywords arguments to be passed into `read_*` function.

        Returns
        -------
        BaseQueryCompiler or
        dict or
        pandas.io.parsers.TextFileReader
            Object with imported data (or with reference to data) for further
            processing, object type depends on the child class `parse` function
            result type.
        """
        ErrorMessage.default_to_pandas("Parameters provided")
        # Use default args for everything
        pandas_frame = cls.parse(fname, **kwargs)
        if isinstance(pandas_frame, pandas.io.parsers.TextFileReader):
            pd_read = pandas_frame.read
            pandas_frame.read = (
                lambda *args, **kwargs: cls.query_compiler_cls.from_pandas(
                    pd_read(*args, **kwargs), cls.frame_cls
                )
            )
            return pandas_frame
        elif isinstance(pandas_frame, (OrderedDict, dict)):
            return {
                i: cls.query_compiler_cls.from_pandas(frame, cls.frame_cls)
                for i, frame in pandas_frame.items()
            }
        return cls.query_compiler_cls.from_pandas(pandas_frame, cls.frame_cls)

    infer_compression = infer_compression


@doc(_doc_pandas_parser_class, data_type="CSV files")
class PandasCSVParser(PandasParser):
    @staticmethod
    @doc(_doc_parse_func, parameters=_doc_parse_parameters_common)
    def parse(fname, **kwargs):
        return PandasParser.generic_parse(fname, **kwargs)


@doc(_doc_pandas_parser_class, data_type="multiple CSV files simultaneously")
class PandasCSVGlobParser(PandasCSVParser):
    @staticmethod
    @doc(
        _doc_parse_func,
        parameters="""chunks : list
    List, where each element of the list is a list of tuples. The inner lists
    of tuples contains the data file name of the chunk, chunk start offset, and
    chunk end offsets for its corresponding file.""",
    )
    def parse(chunks, **kwargs):
        warnings.filterwarnings("ignore")
        num_splits = kwargs.pop("num_splits", None)
        index_col = kwargs.get("index_col", None)

        # `single_worker_read` just pass filename via chunks; need check
        if isinstance(chunks, str):
            return pandas.read_csv(chunks, **kwargs)

        # pop `compression` from kwargs because `bio` below is uncompressed
        compression = kwargs.pop("compression", "infer")
        storage_options = kwargs.pop("storage_options", None) or {}
        pandas_dfs = []
        for fname, start, end in chunks:
            if start is not None and end is not None:
                with OpenFile(fname, "rb", compression, **storage_options) as bio:
                    if kwargs.get("encoding", None) is not None:
                        header = b"" + bio.readline()
                    else:
                        header = b""
                    bio.seek(start)
                    to_read = header + bio.read(end - start)
                pandas_dfs.append(pandas.read_csv(BytesIO(to_read), **kwargs))
            else:
                # This only happens when we are reading with only one worker (Default)
                return pandas.read_csv(
                    fname,
                    compression=compression,
                    storage_options=storage_options,
                    **kwargs,
                )

        # Combine read in data.
        if len(pandas_dfs) > 1:
            pandas_df = pandas.concat(pandas_dfs)
        elif len(pandas_dfs) > 0:
            pandas_df = pandas_dfs[0]
        else:
            pandas_df = pandas.DataFrame()

        # Set internal index.
        if index_col is not None:
            index = pandas_df.index
        else:
            # The lengths will become the RangeIndex
            index = len(pandas_df)
        return _split_result_for_readers(1, num_splits, pandas_df) + [
            index,
            pandas_df.dtypes,
        ]


@doc(_doc_pandas_parser_class, data_type="pickled pandas objects")
class PandasPickleExperimentalParser(PandasParser):
    @staticmethod
    @doc(_doc_parse_func, parameters=_doc_parse_parameters_common)
    def parse(fname, **kwargs):
        warnings.filterwarnings("ignore")
        num_splits = 1
        single_worker_read = kwargs.pop("single_worker_read", None)
        df = pandas.read_pickle(fname, **kwargs)
        if single_worker_read:
            return df
        assert isinstance(
            df, pandas.DataFrame
        ), f"Pickled obj type: [{type(df)}] in [{fname}]; works only with pandas.DataFrame"

        length = len(df)
        width = len(df.columns)

        return _split_result_for_readers(1, num_splits, df) + [length, width]


@doc(_doc_pandas_parser_class, data_type="custom text")
class CustomTextExperimentalParser(PandasParser):
    @staticmethod
    @doc(_doc_parse_func, parameters=_doc_parse_parameters_common)
    def parse(fname, **kwargs):
        return PandasParser.generic_parse(fname, **kwargs)


@doc(_doc_pandas_parser_class, data_type="tables with fixed-width formatted lines")
class PandasFWFParser(PandasParser):
    @staticmethod
    @doc(_doc_parse_func, parameters=_doc_parse_parameters_common)
    def parse(fname, **kwargs):
        return PandasParser.generic_parse(fname, **kwargs)


@doc(_doc_pandas_parser_class, data_type="excel files")
class PandasExcelParser(PandasParser):
    @classmethod
    def get_sheet_data(cls, sheet, convert_float):
        """
        Get raw data from the excel sheet.

        Parameters
        ----------
        sheet : openpyxl.worksheet.worksheet.Worksheet
            Sheet to get data from.
        convert_float : bool
            Whether to convert floats to ints or not.

        Returns
        -------
        list
            List with sheet data.
        """
        return [
            [cls._convert_cell(cell, convert_float) for cell in row]
            for row in sheet.rows
        ]

    @classmethod
    def _convert_cell(cls, cell, convert_float):
        """
        Convert excel cell to value.

        Parameters
        ----------
        cell : openpyxl.cell.cell.Cell
            Excel cell to convert.
        convert_float : bool
            Whether to convert floats to ints or not.

        Returns
        -------
        list
            Value that was converted from the excel cell.
        """
        if cell.is_date:
            return cell.value
        elif cell.data_type == "e":
            return np.nan
        elif cell.data_type == "b":
            return bool(cell.value)
        elif cell.value is None:
            return ""
        elif cell.data_type == "n":
            if convert_float:
                val = int(cell.value)
                if val == cell.value:
                    return val
            else:
                return float(cell.value)

        return cell.value

    @staticmethod
    @doc(_doc_parse_func, parameters=_doc_parse_parameters_common)
    def parse(fname, **kwargs):
        num_splits = kwargs.pop("num_splits", None)
        start = kwargs.pop("start", None)
        end = kwargs.pop("end", None)
        _skiprows = kwargs.pop("skiprows")
        excel_header = kwargs.get("_header")
        sheet_name = kwargs.get("sheet_name", 0)
        footer = b"</sheetData></worksheet>"

        # Default to pandas case, where we are not splitting or partitioning
        if start is None or end is None:
            return pandas.read_excel(fname, **kwargs)

        from zipfile import ZipFile
        from openpyxl import load_workbook
        from openpyxl.worksheet._reader import WorksheetReader
        from openpyxl.reader.excel import ExcelReader
        from openpyxl.worksheet.worksheet import Worksheet
        from pandas.core.dtypes.common import is_list_like
        from pandas.io.excel._util import (
            fill_mi_header,
            maybe_convert_usecols,
        )
        from pandas.io.parsers import TextParser
        import re

        wb = load_workbook(filename=fname, read_only=True)
        # Get shared strings
        ex = ExcelReader(fname, read_only=True)
        ex.read_manifest()
        ex.read_strings()
        # Convert string name 0 to string
        if sheet_name == 0:
            sheet_name = wb.sheetnames[sheet_name]
        # get the worksheet to use with the worksheet reader
        ws = Worksheet(wb)
        # Read the raw data
        with ZipFile(fname) as z:
            with z.open("xl/worksheets/{}.xml".format(sheet_name)) as file:
                file.seek(start)
                bytes_data = file.read(end - start)

        def update_row_nums(match):
            """
            Update the row numbers to start at 1.

            Parameters
            ----------
            match : re.Match object
                The match from the origin `re.sub` looking for row number tags.

            Returns
            -------
            str
                The updated string with new row numbers.

            Notes
            -----
            This is needed because the parser we are using does not scale well if
            the row numbers remain because empty rows are inserted for all "missing"
            rows.
            """
            b = match.group(0)
            return re.sub(
                rb"\d+",
                lambda c: str(int(c.group(0).decode("utf-8")) - _skiprows).encode(
                    "utf-8"
                ),
                b,
            )

        bytes_data = re.sub(rb'r="[A-Z]*\d+"', update_row_nums, bytes_data)
        bytesio = BytesIO(excel_header + bytes_data + footer)
        # Use openpyxl to read/parse sheet data
        reader = WorksheetReader(ws, bytesio, ex.shared_strings, False)
        # Attach cells to worksheet object
        reader.bind_cells()
        data = PandasExcelParser.get_sheet_data(ws, kwargs.pop("convert_float", True))
        usecols = maybe_convert_usecols(kwargs.pop("usecols", None))
        header = kwargs.pop("header", 0)
        index_col = kwargs.pop("index_col", None)
        # skiprows is handled externally
        skiprows = None

        # Handle header and create MultiIndex for columns if necessary
        if is_list_like(header) and len(header) == 1:
            header = header[0]
        if header is not None and is_list_like(header):
            control_row = [True] * len(data[0])

            for row in header:
                data[row], control_row = fill_mi_header(data[row], control_row)
        # Handle MultiIndex for row Index if necessary
        if is_list_like(index_col):
            # Forward fill values for MultiIndex index.
            if not is_list_like(header):
                offset = 1 + header
            else:
                offset = 1 + max(header)

            # Check if dataset is empty
            if offset < len(data):
                for col in index_col:
                    last = data[offset][col]
                    for row in range(offset + 1, len(data)):
                        if data[row][col] == "" or data[row][col] is None:
                            data[row][col] = last
                        else:
                            last = data[row][col]
        parser = TextParser(
            data,
            header=header,
            index_col=index_col,
            has_index_names=is_list_like(header) and len(header) > 1,
            skiprows=skiprows,
            usecols=usecols,
            skip_blank_lines=False,
            **kwargs,
        )
        pandas_df = parser.read()
        if len(pandas_df) > 1 and pandas_df.isnull().all().all():
            # Drop NaN rows at the end of the DataFrame
            pandas_df = pandas.DataFrame(columns=pandas_df.columns)

        # Since we know the number of rows that occur before this partition, we can
        # correctly assign the index in cases of RangeIndex. If it is not a RangeIndex,
        # the index is already correct because it came from the data.
        if isinstance(pandas_df.index, pandas.RangeIndex):
            pandas_df.index = pandas.RangeIndex(
                start=_skiprows, stop=len(pandas_df.index) + _skiprows
            )
        # We return the length if it is a RangeIndex (common case) to reduce
        # serialization cost.
        if index_col is not None:
            index = pandas_df.index
        else:
            # The lengths will become the RangeIndex
            index = len(pandas_df)
        return _split_result_for_readers(1, num_splits, pandas_df) + [
            index,
            pandas_df.dtypes,
        ]


@doc(_doc_pandas_parser_class, data_type="JSON files")
class PandasJSONParser(PandasParser):
    @staticmethod
    @doc(_doc_parse_func, parameters=_doc_parse_parameters_common)
    def parse(fname, **kwargs):
        num_splits = kwargs.pop("num_splits", None)
        start = kwargs.pop("start", None)
        end = kwargs.pop("end", None)
        if start is not None and end is not None:
            # pop "compression" from kwargs because bio is uncompressed
            with OpenFile(
                fname,
                "rb",
                kwargs.pop("compression", "infer"),
                **(kwargs.pop("storage_options", None) or {}),
            ) as bio:
                bio.seek(start)
                to_read = b"" + bio.read(end - start)
            columns = kwargs.pop("columns")
            pandas_df = pandas.read_json(BytesIO(to_read), **kwargs)
        else:
            # This only happens when we are reading with only one worker (Default)
            return pandas.read_json(fname, **kwargs)
        if not pandas_df.columns.equals(columns):
            raise ValueError("Columns must be the same across all rows.")
        partition_columns = pandas_df.columns
        return _split_result_for_readers(1, num_splits, pandas_df) + [
            len(pandas_df),
            pandas_df.dtypes,
            partition_columns,
        ]


@doc(_doc_pandas_parser_class, data_type="PARQUET data")
class PandasParquetParser(PandasParser):
    @staticmethod
    @doc(_doc_parse_func, parameters=_doc_parse_parameters_common)
    def parse(fname, **kwargs):
        num_splits = kwargs.pop("num_splits", None)
        columns = kwargs.get("columns", None)
        if num_splits is None:
            return pandas.read_parquet(fname, **kwargs)
        kwargs["use_pandas_metadata"] = True
        df = pandas.read_parquet(fname, **kwargs)
        if isinstance(df.index, pandas.RangeIndex):
            idx = len(df.index)
        else:
            idx = df.index
        columns = [c for c in columns if c not in df.index.names and c in df.columns]
        if columns is not None:
            df = df[columns]
        # Append the length of the index here to build it externally
        return _split_result_for_readers(0, num_splits, df) + [idx, df.dtypes]


@doc(_doc_pandas_parser_class, data_type="HDF data")
class PandasHDFParser(PandasParser):  # pragma: no cover
    @staticmethod
    @doc(
        _doc_parse_func,
        parameters="""fname : str, path object, pandas.HDFStore or file-like object
    Name of the file, path pandas.HDFStore or file-like object to read.""",
    )
    def parse(fname, **kwargs):
        kwargs["key"] = kwargs.pop("_key", None)
        num_splits = kwargs.pop("num_splits", None)
        if num_splits is None:
            return pandas.read_hdf(fname, **kwargs)
        df = pandas.read_hdf(fname, **kwargs)
        # Append the length of the index here to build it externally
        return _split_result_for_readers(0, num_splits, df) + [len(df.index), df.dtypes]


@doc(_doc_pandas_parser_class, data_type="FEATHER files")
class PandasFeatherParser(PandasParser):
    @staticmethod
    @doc(
        _doc_parse_func,
        parameters="""fname : str, path object or file-like object
    Name of the file, path or file-like object to read.""",
    )
    def parse(fname, **kwargs):
        from pyarrow import feather

        num_splits = kwargs.pop("num_splits", None)
        if num_splits is None:
            return pandas.read_feather(fname, **kwargs)

        with OpenFile(
            fname,
            **(kwargs.pop("storage_options", None) or {}),
        ) as file:
            df = feather.read_feather(file, **kwargs)
        # Append the length of the index here to build it externally
        return _split_result_for_readers(0, num_splits, df) + [len(df.index), df.dtypes]


@doc(_doc_pandas_parser_class, data_type="SQL queries or tables")
class PandasSQLParser(PandasParser):
    @staticmethod
    @doc(
        _doc_parse_func,
        parameters="""sql : str or SQLAlchemy Selectable (select or text object)
    SQL query to be executed or a table name.
con : SQLAlchemy connectable, str, or sqlite3 connection
    Connection object to database.
index_col : str or list of str
    Column(s) to set as index(MultiIndex).
read_sql_engine : str
    Underlying engine ('pandas' or 'connectorx') used for fetching query result.""",
    )
    def parse(sql, con, index_col, read_sql_engine, **kwargs):

        enable_cx = False
        if read_sql_engine == "Connectorx":
            try:
                import connectorx as cx

                enable_cx = True
            except ImportError:
                warnings.warn(
                    "Switch to 'pandas.read_sql' since 'connectorx' is not installed, please run 'pip install connectorx'."
                )

        num_splits = kwargs.pop("num_splits", None)
        if isinstance(con, ModinDatabaseConnection):
            con = con.get_string() if enable_cx else con.get_connection()

        if num_splits is None:
            if enable_cx:
                return cx.read_sql(con, sql, index_col=index_col)
            return pandas.read_sql(sql, con, index_col=index_col, **kwargs)

        if enable_cx:
            df = cx.read_sql(con, sql, index_col=index_col)
        else:
            df = pandas.read_sql(sql, con, index_col=index_col, **kwargs)
        if index_col is None:
            index = len(df)
        else:
            index = df.index
        return _split_result_for_readers(1, num_splits, df) + [index, df.dtypes]
