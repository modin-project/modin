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

"""Module houses `ExcelDispatcher` class, that is used for reading excel files."""

import os
import re
import warnings
from io import BytesIO

import pandas
from pandas.io.common import stringify_path

from modin.config import NPartitions
from modin.core.io.text.text_file_dispatcher import TextFileDispatcher
from modin.pandas.io import ExcelFile

EXCEL_READ_BLOCK_SIZE = 4096


class ExcelDispatcher(TextFileDispatcher):
    """Class handles utils for reading excel files."""

    @classmethod
    def _read(cls, io, **kwargs):
        """
        Read data from `io` according to the passed `read_excel` `kwargs` parameters.

        Parameters
        ----------
        io : str, bytes, ExcelFile, xlrd.Book, path object, or file-like object
            `io` parameter of `read_excel` function.
        **kwargs : dict
            Parameters of `read_excel` function.

        Returns
        -------
        new_query_compiler : BaseQueryCompiler
            Query compiler with imported data for further processing.
        """
        io = stringify_path(io)
        if (
            kwargs.get("engine", None) is not None
            and kwargs.get("engine") != "openpyxl"
        ):
            return cls.single_worker_read(
                io,
                reason="Modin only implements parallel `read_excel` with `openpyxl` engine, "
                + 'please specify `engine=None` or `engine="openpyxl"` to '
                + "use Modin's parallel implementation.",
                **kwargs
            )

        if kwargs.get("skiprows") is not None:
            return cls.single_worker_read(
                io,
                reason="Modin doesn't support 'skiprows' parameter of `read_excel`",
                **kwargs
            )

        if isinstance(io, bytes):
            io = BytesIO(io)

        # isinstance(ExcelFile, os.PathLike) == True
        if not isinstance(io, (str, os.PathLike, BytesIO)) or isinstance(
            io, (ExcelFile, pandas.ExcelFile)
        ):
            if isinstance(io, ExcelFile):
                io._set_pandas_mode()
            return cls.single_worker_read(
                io,
                reason="Modin only implements parallel `read_excel` the following types of `io`: "
                + "str, os.PathLike, io.BytesIO.",
                **kwargs
            )

        from zipfile import ZipFile

        from openpyxl.reader.excel import ExcelReader
        from openpyxl.worksheet._reader import WorksheetReader
        from openpyxl.worksheet.worksheet import Worksheet

        from modin.core.storage_formats.pandas.parsers import PandasExcelParser

        sheet_name = kwargs.get("sheet_name", 0)
        if sheet_name is None or isinstance(sheet_name, list):
            return cls.single_worker_read(
                io,
                reason="`read_excel` functionality is only implemented for a single sheet at a "
                + "time. Multiple sheet reading coming soon!",
                **kwargs
            )

        warnings.warn(
            "Parallel `read_excel` is a new feature! If you run into any "
            + "problems, please visit https://github.com/modin-project/modin/issues. "
            + "If you find a new issue and can't file it on GitHub, please "
            + "email bug_reports@modin.org."
        )

        # NOTE: ExcelReader() in read-only mode does not close file handle by itself
        # work around that by passing file object if we received some path
        io_file = open(io, "rb") if isinstance(io, (str, os.PathLike)) else io
        try:
            ex = ExcelReader(io_file, read_only=True)
            ex.read()
            wb = ex.wb

            # Get shared strings
            ex.read_manifest()
            ex.read_strings()
            ws = Worksheet(wb)
        finally:
            if isinstance(io, (str, os.PathLike)):
                # close only if it were us who opened the object
                io_file.close()

        pandas_kw = dict(kwargs)  # preserve original kwargs
        with ZipFile(io) as z:
            # Convert index to sheet name in file
            if isinstance(sheet_name, int):
                sheet_name = "sheet{}".format(sheet_name + 1)
            else:
                sheet_name = "sheet{}".format(wb.sheetnames.index(sheet_name) + 1)
            if any(sheet_name.lower() in name for name in z.namelist()):
                sheet_name = sheet_name.lower()
            elif any(sheet_name.title() in name for name in z.namelist()):
                sheet_name = sheet_name.title()
            else:
                raise ValueError("Sheet {} not found".format(sheet_name.lower()))
            # Pass this value to the workers
            kwargs["sheet_name"] = sheet_name

            f = z.open("xl/worksheets/{}.xml".format(sheet_name))
            f = BytesIO(f.read())
            total_bytes = cls.file_size(f)

            # Read some bytes from the sheet so we can extract the XML header and first
            # line. We need to make sure we get the first line of the data as well
            # because that is where the column names are. The header information will
            # be extracted and sent to all of the nodes.
            sheet_block = f.read(EXCEL_READ_BLOCK_SIZE)
            end_of_row_tag = b"</row>"
            while end_of_row_tag not in sheet_block:
                sheet_block += f.read(EXCEL_READ_BLOCK_SIZE)
            idx_of_header_end = sheet_block.index(end_of_row_tag) + len(end_of_row_tag)
            sheet_header_with_first_row = sheet_block[:idx_of_header_end]

            if kwargs["header"] is not None:
                # Reset the file pointer to begin at the end of the header information.
                f.seek(idx_of_header_end)
                sheet_header = sheet_header_with_first_row
            else:
                start_of_row_tag = b"<row"
                idx_of_header_start = sheet_block.index(start_of_row_tag)
                sheet_header = sheet_block[:idx_of_header_start]
                # Reset the file pointer to begin at the end of the header information.
                f.seek(idx_of_header_start)

            kwargs["_header"] = sheet_header
            footer = b"</sheetData></worksheet>"
            # Use openpyxml to parse the data
            common_args = (
                ws,
                BytesIO(sheet_header_with_first_row + footer),
                ex.shared_strings,
                False,
            )
            if cls.need_rich_text_param():
                reader = WorksheetReader(*common_args, rich_text=False)
            else:
                reader = WorksheetReader(*common_args)
            # Attach cells to the worksheet
            reader.bind_cells()
            data = PandasExcelParser.get_sheet_data(
                ws, kwargs.get("convert_float", True)
            )
            # Extract column names from parsed data.
            if kwargs["header"] is None:
                column_names = pandas.RangeIndex(len(data[0]))
            else:
                column_names = pandas.Index(data[0])
            index_col = kwargs.get("index_col", None)
            # Remove column names that are specified as `index_col`
            if index_col is not None:
                column_names = column_names.drop(column_names[index_col])

            if not all(column_names) or kwargs.get("usecols"):
                # some column names are empty, use pandas reader to take the names from it
                pandas_kw["nrows"] = 1
                df = pandas.read_excel(io, **pandas_kw)
                column_names = df.columns

            # Compute partition metadata upfront so it is uniform for all partitions
            chunk_size = max(1, (total_bytes - f.tell()) // NPartitions.get())
            column_widths, num_splits = cls._define_metadata(
                pandas.DataFrame(columns=column_names), column_names
            )
            kwargs["fname"] = io
            # Skiprows will be used to inform a partition how many rows come before it.
            kwargs["skiprows"] = 0
            row_count = 0
            data_ids = []
            index_ids = []
            dtypes_ids = []

            kwargs["num_splits"] = num_splits

            while f.tell() < total_bytes:
                args = kwargs
                args["skiprows"] = row_count + args["skiprows"]
                args["start"] = f.tell()
                chunk = f.read(chunk_size)
                # This edge case can happen when we have reached the end of the data
                # but not the end of the file.
                if b"<row" not in chunk:
                    break
                row_close_tag = b"</row>"
                row_count = re.subn(row_close_tag, b"", chunk)[1]

                # Make sure we are reading at least one row.
                while row_count == 0:
                    chunk += f.read(chunk_size)
                    row_count += re.subn(row_close_tag, b"", chunk)[1]

                last_index = chunk.rindex(row_close_tag)
                f.seek(-(len(chunk) - last_index) + len(row_close_tag), 1)
                args["end"] = f.tell()

                # If there is no data, exit before triggering computation.
                if b"</row>" not in chunk and b"</sheetData>" in chunk:
                    break
                remote_results_list = cls.deploy(
                    func=cls.parse,
                    f_kwargs=args,
                    num_returns=num_splits + 2,
                )
                data_ids.append(remote_results_list[:-2])
                index_ids.append(remote_results_list[-2])
                dtypes_ids.append(remote_results_list[-1])

                # The end of the spreadsheet
                if b"</sheetData>" in chunk:
                    break

        # Compute the index based on a sum of the lengths of each partition (by default)
        # or based on the column(s) that were requested.
        if index_col is None:
            row_lengths = cls.materialize(index_ids)
            new_index = pandas.RangeIndex(sum(row_lengths))
        else:
            index_objs = cls.materialize(index_ids)
            row_lengths = [len(o) for o in index_objs]
            new_index = index_objs[0].append(index_objs[1:])

        data_ids = cls.build_partition(data_ids, row_lengths, column_widths)

        # Compute dtypes by getting collecting and combining all of the partitions. The
        # reported dtypes from differing rows can be different based on the inference in
        # the limited data seen by each worker. We use pandas to compute the exact dtype
        # over the whole column for each column. The index is set below.
        dtypes = cls.get_dtypes(dtypes_ids, column_names)

        new_frame = cls.frame_cls(
            data_ids,
            new_index,
            column_names,
            row_lengths,
            column_widths,
            dtypes=dtypes,
        )
        new_query_compiler = cls.query_compiler_cls(new_frame)
        if index_col is None:
            new_query_compiler._modin_frame.synchronize_labels(axis=0)
        return new_query_compiler
