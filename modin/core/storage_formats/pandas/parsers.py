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

import contextlib
import json
import os
import warnings
from io import BytesIO, IOBase, TextIOWrapper
from typing import Any, NamedTuple

import fsspec
import numpy as np
import pandas
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.concat import union_categoricals
from pandas.io.common import infer_compression
from pandas.util._decorators import doc

from modin.config import MinColumnPartitionSize, MinRowPartitionSize
from modin.core.io.file_dispatcher import OpenFile
from modin.core.storage_formats.pandas.utils import split_result_of_axis_func_pandas
from modin.db_conn import ModinDatabaseConnection
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
from modin.logging.config import LogLevel
from modin.utils import ModinAssumptionError

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

_doc_common_read_kwargs = """common_read_kwargs : dict
    Common keyword parameters for read functions.
"""
_doc_parse_parameters_common2 = "\n".join(
    (_doc_parse_parameters_common, _doc_common_read_kwargs)
)


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
    splits = split_result_of_axis_func_pandas(
        axis,
        num_splits,
        df,
        min_block_size=(
            MinRowPartitionSize.get() if axis == 0 else MinColumnPartitionSize.get()
        ),
    )
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


class PandasParser(ClassLogger, modin_layer="PARSER", log_level=LogLevel.DEBUG):
    """Base class for parser classes with pandas storage format."""

    @staticmethod
    @doc(_doc_parse_func, parameters=_doc_parse_parameters_common)
    def generic_parse(fname, **kwargs):
        warnings.filterwarnings("ignore")
        num_splits = kwargs.pop("num_splits", None)
        start = kwargs.pop("start", None)
        end = kwargs.pop("end", None)
        header_size = kwargs.pop("header_size", 0)
        common_dtypes = kwargs.pop("common_dtypes", None)
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
        if common_dtypes is not None:
            kwargs["dtype"] = common_dtypes
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
    def get_dtypes(cls, dtypes_ids, columns):
        """
        Get common for all partitions dtype for each of the columns.

        Parameters
        ----------
        dtypes_ids : list
            Array with references to the partitions dtypes objects.
        columns : array-like or Index (1d)
            The names of the columns in this variable will be used
            for dtypes creation.

        Returns
        -------
        frame_dtypes : pandas.Series, dtype or None
            Resulting dtype or pandas.Series where column names are used as
            index and types of columns are used as values for full resulting
            frame.
        """
        if len(dtypes_ids) == 0:
            return None
        # each element in `partitions_dtypes` is a Series, where column names are
        # used as index and types of columns for different partitions are used as values
        partitions_dtypes = cls.materialize(dtypes_ids)
        if all([len(dtype) == 0 for dtype in partitions_dtypes]):
            return None

        combined_part_dtypes = pandas.concat(partitions_dtypes, axis=1)
        frame_dtypes = combined_part_dtypes.iloc[:, 0]
        frame_dtypes.name = None

        if not combined_part_dtypes.eq(frame_dtypes, axis=0).all(axis=None):
            ErrorMessage.mismatch_with_pandas(
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

        # Set the index for the dtypes to the column names
        if isinstance(frame_dtypes, pandas.Series):
            frame_dtypes.index = columns
        else:
            frame_dtypes = pandas.Series(frame_dtypes, index=columns)

        return frame_dtypes

    @classmethod
    def single_worker_read(cls, fname, *args, reason: str, **kwargs):
        """
        Perform reading by single worker (default-to-pandas implementation).

        Parameters
        ----------
        fname : str, path object or file-like object
            Name of the file or file-like object to read.
        *args : tuple
            Positional arguments to be passed into `read_*` function.
        reason : str
            Message describing the reason for falling back to pandas.
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
        ErrorMessage.default_to_pandas(reason=reason)
        # Use default args for everything
        pandas_frame = cls.parse(fname, *args, **kwargs)
        if isinstance(pandas_frame, pandas.io.parsers.TextFileReader):
            pd_read = pandas_frame.read
            pandas_frame.read = (
                lambda *args, **kwargs: cls.query_compiler_cls.from_pandas(
                    pd_read(*args, **kwargs), cls.frame_cls
                )
            )
            return pandas_frame
        elif isinstance(pandas_frame, dict):
            return {
                i: cls.query_compiler_cls.from_pandas(frame, cls.frame_cls)
                for i, frame in pandas_frame.items()
            }
        return cls.query_compiler_cls.from_pandas(pandas_frame, cls.frame_cls)

    @staticmethod
    def get_types_mapper(dtype_backend):
        """
        Get types mapper that would be used in read_parquet/read_feather.

        Parameters
        ----------
        dtype_backend : {"numpy_nullable", "pyarrow", lib.no_default}

        Returns
        -------
        dict
        """
        to_pandas_kwargs = {}
        if dtype_backend == "numpy_nullable":
            from pandas.io._util import _arrow_dtype_mapping

            mapping = _arrow_dtype_mapping()
            to_pandas_kwargs["types_mapper"] = mapping.get
        elif dtype_backend == "pyarrow":
            to_pandas_kwargs["types_mapper"] = pandas.ArrowDtype
        return to_pandas_kwargs

    infer_compression = infer_compression


@doc(_doc_pandas_parser_class, data_type="CSV files")
class PandasCSVParser(PandasParser):
    @staticmethod
    @doc(_doc_parse_func, parameters=_doc_parse_parameters_common2)
    def parse(fname, common_read_kwargs, **kwargs):
        return PandasParser.generic_parse(
            fname,
            callback=PandasCSVParser.read_callback,
            **common_read_kwargs,
            **kwargs,
        )

    @staticmethod
    def read_callback(*args, **kwargs):
        """
        Parse data on each partition.

        Parameters
        ----------
        *args : list
            Positional arguments to be passed to the callback function.
        **kwargs : dict
            Keyword arguments to be passed to the callback function.

        Returns
        -------
        pandas.DataFrame or pandas.io.parsers.TextParser
            Function call result.
        """
        return pandas.read_csv(*args, **kwargs)


@doc(_doc_pandas_parser_class, data_type="tables with fixed-width formatted lines")
class PandasFWFParser(PandasParser):
    @staticmethod
    @doc(_doc_parse_func, parameters=_doc_parse_parameters_common2)
    def parse(fname, common_read_kwargs, **kwargs):
        return PandasParser.generic_parse(
            fname,
            callback=PandasFWFParser.read_callback,
            **common_read_kwargs,
            **kwargs,
        )

    @staticmethod
    def read_callback(*args, **kwargs):
        """
        Parse data on each partition.

        Parameters
        ----------
        *args : list
            Positional arguments to be passed to the callback function.
        **kwargs : dict
            Keyword arguments to be passed to the callback function.

        Returns
        -------
        pandas.DataFrame or pandas.io.parsers.TextFileReader
            Function call result.
        """
        return pandas.read_fwf(*args, **kwargs)


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
    def need_rich_text_param():
        """
        Determine whether a required `rich_text` parameter should be specified for the ``WorksheetReader`` constructor.

        Returns
        -------
        bool
        """
        import openpyxl
        from packaging import version

        return version.parse(openpyxl.__version__) >= version.parse("3.1.0")

    @staticmethod
    @doc(_doc_parse_func, parameters=_doc_parse_parameters_common)
    def parse(fname, **kwargs):
        num_splits = kwargs.pop("num_splits", None)
        start = kwargs.pop("start", None)
        end = kwargs.pop("end", None)
        excel_header = kwargs.get("_header")
        sheet_name = kwargs.get("sheet_name", 0)
        footer = b"</sheetData></worksheet>"

        # Default to pandas case, where we are not splitting or partitioning
        if start is None or end is None:
            return pandas.read_excel(fname, **kwargs)

        _skiprows = kwargs.pop("skiprows")

        import re
        from zipfile import ZipFile

        import openpyxl
        from openpyxl.reader.excel import ExcelReader
        from openpyxl.worksheet._reader import WorksheetReader
        from openpyxl.worksheet.worksheet import Worksheet
        from pandas.core.dtypes.common import is_list_like
        from pandas.io.excel._util import fill_mi_header, maybe_convert_usecols
        from pandas.io.parsers import TextParser

        wb = openpyxl.load_workbook(filename=fname, read_only=True)
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
        common_args = (ws, bytesio, ex.shared_strings, False)
        if PandasExcelParser.need_rich_text_param():
            reader = WorksheetReader(*common_args, rich_text=False)
        else:
            reader = WorksheetReader(*common_args)
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
        if (
            len(pandas_df) > 1
            and len(pandas_df.columns) != 0
            and pandas_df.isnull().all().all()
        ):
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
            raise ModinAssumptionError("Columns must be the same across all rows.")
        partition_columns = pandas_df.columns
        return _split_result_for_readers(1, num_splits, pandas_df) + [
            len(pandas_df),
            pandas_df.dtypes,
            partition_columns,
        ]


class ParquetFileToRead(NamedTuple):
    """
    Class to store path and row group information for parquet reads.

    Parameters
    ----------
    path : str, path object or file-like object
        Name of the file to read.
    row_group_start : int
        Row group to start read from.
    row_group_end : int
        Row group to stop read.
    """

    path: Any
    row_group_start: int
    row_group_end: int


@doc(_doc_pandas_parser_class, data_type="PARQUET data")
class PandasParquetParser(PandasParser):
    @staticmethod
    def _read_row_group_chunk(
        f, row_group_start, row_group_end, columns, filters, engine, to_pandas_kwargs
    ):  # noqa: GL08
        if engine == "pyarrow":
            if filters is not None:
                import pyarrow.dataset as ds
                from pyarrow.parquet import filters_to_expression

                parquet_format = ds.ParquetFileFormat()
                fragment = parquet_format.make_fragment(
                    f,
                    row_groups=range(
                        row_group_start,
                        row_group_end,
                    ),
                )
                dataset = ds.FileSystemDataset(
                    [fragment],
                    schema=fragment.physical_schema,
                    format=parquet_format,
                    filesystem=fragment.filesystem,
                )

                # This lower-level API doesn't have the ability to automatically handle pandas metadata
                # The following code is based on
                # https://github.com/apache/arrow/blob/f44e28fa03a64ae5b3d9352d21aee2cc84f9af6c/python/pyarrow/parquet/core.py#L2619-L2628

                # if use_pandas_metadata, we need to include index columns in the
                # column selection, to be able to restore those in the pandas DataFrame
                metadata = dataset.schema.metadata or {}

                if b"pandas" in metadata and columns is not None:
                    index_columns = json.loads(metadata[b"pandas"].decode("utf8"))[
                        "index_columns"
                    ]
                    # In the pandas metadata, the index columns can either be string column names,
                    # or a dictionary that describes a RangeIndex.
                    # Here, we are finding the real data columns that need to be read to become part
                    # of the pandas Index, so we can skip the RangeIndex.
                    # Not only can a RangeIndex be trivially reconstructed later, but we actually
                    # ignore partition-level range indices, because we want to have a single Modin
                    # RangeIndex that spans all partitions.
                    index_columns = [
                        col for col in index_columns if not isinstance(col, dict)
                    ]
                    columns = list(columns) + list(set(index_columns) - set(columns))

                return dataset.to_table(
                    columns=columns,
                    filter=filters_to_expression(filters),
                ).to_pandas(**to_pandas_kwargs)
            else:
                from pyarrow.parquet import ParquetFile

                return (
                    ParquetFile(f)
                    .read_row_groups(
                        range(
                            row_group_start,
                            row_group_end,
                        ),
                        columns=columns,
                        use_pandas_metadata=True,
                    )
                    .to_pandas(**to_pandas_kwargs)
                )
        elif engine == "fastparquet":
            from fastparquet import ParquetFile

            return ParquetFile(f)[row_group_start:row_group_end].to_pandas(
                columns=columns,
                filters=filters,
                # Setting row_filter=True would perform filtering at the row level, which is more correct
                # (in line with pyarrow)
                # However, it doesn't work: https://github.com/dask/fastparquet/issues/873
                # Also, this would create incompatibility with pandas
            )
        else:
            # We shouldn't ever come to this case, so something went wrong
            raise ValueError(
                f"engine must be one of 'pyarrow', 'fastparquet', got: {engine}"
            )

    @staticmethod
    @doc(
        _doc_parse_func,
        parameters="""files_for_parser : list
    List of files to be read.
engine : str
    Parquet library to use (either PyArrow or fastparquet).
""",
    )
    def parse(files_for_parser, engine, **kwargs):
        columns = kwargs.get("columns", None)
        filters = kwargs.get("filters", None)
        storage_options = kwargs.get("storage_options", {})
        chunks = []
        # `single_worker_read` just passes in a string path or path-like object
        if isinstance(files_for_parser, (str, os.PathLike)):
            return pandas.read_parquet(files_for_parser, engine=engine, **kwargs)

        to_pandas_kwargs = PandasParser.get_types_mapper(kwargs["dtype_backend"])

        for file_for_parser in files_for_parser:
            if isinstance(file_for_parser.path, IOBase):
                context = contextlib.nullcontext(file_for_parser.path)
            else:
                context = fsspec.open(file_for_parser.path, **storage_options)
            with context as f:
                chunk = PandasParquetParser._read_row_group_chunk(
                    f,
                    file_for_parser.row_group_start,
                    file_for_parser.row_group_end,
                    columns,
                    filters,
                    engine,
                    to_pandas_kwargs,
                )
            chunks.append(chunk)
        df = pandas.concat(chunks)
        return df, df.index, len(df)


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

        to_pandas_kwargs = PandasParser.get_types_mapper(kwargs["dtype_backend"])
        del kwargs["dtype_backend"]

        with OpenFile(
            fname,
            **(kwargs.pop("storage_options", None) or {}),
        ) as file:
            # The implementation is as close as possible to the one in pandas.
            # For reference see `read_feather` in pandas/io/feather_format.py.
            if not to_pandas_kwargs:
                df = feather.read_feather(file, **kwargs)
            else:
                # `read_feather` doesn't accept `types_mapper` if pyarrow<11.0
                pa_table = feather.read_table(file, **kwargs)
                df = pa_table.to_pandas(**to_pandas_kwargs)
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
