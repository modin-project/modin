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

import pandas
import re
import sys
import warnings

from modin.data_management.utils import compute_chunksize
from modin.engines.base.io.text.text_file_dispatcher import TextFileDispatcher


EXCEL_READ_BLOCK_SIZE = 4096


class ExcelDispatcher(TextFileDispatcher):
    @classmethod
    def _read(cls, io, **kwargs):
        if (
            kwargs.get("engine", None) is not None
            and kwargs.get("engine") != "openpyxl"
        ):
            warnings.warn(
                "Modin only implements parallel `read_excel` with `openpyxl` engine, "
                'please specify `engine=None` or `engine="openpyxl"` to '
                "use Modin's parallel implementation."
            )
            return cls.single_worker_read(io, **kwargs)
        if sys.version_info < (3, 7):
            warnings.warn("Python 3.7 or higher required for parallel `read_excel`.")
            return cls.single_worker_read(io, **kwargs)

        from zipfile import ZipFile
        from openpyxl.worksheet.worksheet import Worksheet
        from openpyxl.worksheet._reader import WorksheetReader
        from openpyxl.reader.excel import ExcelReader
        from modin.backends.pandas.parsers import PandasExcelParser

        sheet_name = kwargs.get("sheet_name", 0)
        if sheet_name is None or isinstance(sheet_name, list):
            warnings.warn(
                "`read_excel` functionality is only implemented for a single sheet at a "
                "time. Multiple sheet reading coming soon!"
            )
            return cls.single_worker_read(io, **kwargs)

        warnings.warn(
            "Parallel `read_excel` is a new feature! Please email "
            "bug_reports@modin.org if you run into any problems."
        )

        # NOTE: ExcelReader() in read-only mode does not close file handle by itself
        # work around that by passing file object if we received some path
        io_file = open(io, "rb") if isinstance(io, str) else io
        try:
            ex = ExcelReader(io_file, read_only=True)
            ex.read()
            wb = ex.wb

            # Get shared strings
            ex.read_manifest()
            ex.read_strings()
            ws = Worksheet(wb)
        finally:
            if isinstance(io, str):
                # close only if it were us who opened the object
                io_file.close()

        pandas_kw = dict(kwargs)  # preserve original kwargs
        with ZipFile(io) as z:
            from io import BytesIO

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

            from modin.pandas import DEFAULT_NPARTITIONS

            num_partitions = DEFAULT_NPARTITIONS
            # Read some bytes from the sheet so we can extract the XML header and first
            # line. We need to make sure we get the first line of the data as well
            # because that is where the column names are. The header information will
            # be extracted and sent to all of the nodes.
            sheet_block = f.read(EXCEL_READ_BLOCK_SIZE)
            end_of_row_tag = b"</row>"
            while end_of_row_tag not in sheet_block:
                sheet_block += f.read(EXCEL_READ_BLOCK_SIZE)
            idx_of_header_end = sheet_block.index(end_of_row_tag) + len(end_of_row_tag)
            sheet_header = sheet_block[:idx_of_header_end]
            # Reset the file pointer to begin at the end of the header information.
            f.seek(idx_of_header_end)
            kwargs["_header"] = sheet_header
            footer = b"</sheetData></worksheet>"
            # Use openpyxml to parse the data
            reader = WorksheetReader(
                ws, BytesIO(sheet_header + footer), ex.shared_strings, False
            )
            # Attach cells to the worksheet
            reader.bind_cells()
            data = PandasExcelParser.get_sheet_data(
                ws, kwargs.get("convert_float", True)
            )
            # Extract column names from parsed data.
            column_names = pandas.Index(data[0])
            index_col = kwargs.get("index_col", None)
            # Remove column names that are specified as `index_col`
            if index_col is not None:
                column_names = column_names.drop(column_names[index_col])

            if not all(column_names):
                # some column names are empty, use pandas reader to take the names from it
                pandas_kw["nrows"] = 1
                df = pandas.read_excel(io, **pandas_kw)
                column_names = df.columns

            # Compute partition metadata upfront so it is uniform for all partitions
            chunk_size = max(1, (total_bytes - f.tell()) // num_partitions)
            num_splits = min(len(column_names), num_partitions)
            kwargs["fname"] = io
            # Skiprows will be used to inform a partition how many rows come before it.
            kwargs["skiprows"] = 0
            row_count = 0
            data_ids = []
            index_ids = []
            dtypes_ids = []

            # Compute column metadata
            column_chunksize = compute_chunksize(
                pandas.DataFrame(columns=column_names), num_splits, axis=1
            )
            if column_chunksize > len(column_names):
                column_widths = [len(column_names)]
                # This prevents us from unnecessarily serializing a bunch of empty
                # objects.
                num_splits = 1
            else:
                column_widths = [
                    column_chunksize
                    if len(column_names) > (column_chunksize * (i + 1))
                    else 0
                    if len(column_names) < (column_chunksize * i)
                    else len(column_names) - (column_chunksize * i)
                    for i in range(num_splits)
                ]
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
                remote_results_list = cls.deploy(cls.parse, num_splits + 2, args)
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

        # Compute dtypes by getting collecting and combining all of the partitions. The
        # reported dtypes from differing rows can be different based on the inference in
        # the limited data seen by each worker. We use pandas to compute the exact dtype
        # over the whole column for each column. The index is set below.
        dtypes = cls.get_dtypes(dtypes_ids)

        data_ids = cls.build_partition(data_ids, row_lengths, column_widths)
        # Set the index for the dtypes to the column names
        if isinstance(dtypes, pandas.Series):
            dtypes.index = column_names
        else:
            dtypes = pandas.Series(dtypes, index=column_names)
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
            new_query_compiler._modin_frame._apply_index_objs(axis=0)
        return new_query_compiler
