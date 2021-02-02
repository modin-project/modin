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

from modin.engines.base.io.file_dispatcher import FileDispatcher
import numpy as np
import warnings
import io
import os
from typing import Tuple, List

from modin.config import NPartitions


class TextFileDispatcher(FileDispatcher):
    @classmethod
    def get_path_or_buffer(cls, filepath_or_buffer):
        """Given a buffer, try and extract the filepath from it so that we can
        use it without having to fall back to Pandas and share file objects between
        workers. Given a filepath, return it immediately.
        """
        if isinstance(filepath_or_buffer, (io.BufferedReader, io.TextIOWrapper)):
            buffer_filepath = filepath_or_buffer.name
            if cls.file_exists(buffer_filepath):
                warnings.warn(
                    "For performance reasons, the filepath will be "
                    "used in place of the file handle passed in "
                    "to load the data"
                )
                return cls.get_path(buffer_filepath)
        return filepath_or_buffer

    @classmethod
    def build_partition(cls, partition_ids, row_lengths, column_widths):
        return np.array(
            [
                [
                    cls.frame_partition_cls(
                        partition_ids[i][j],
                        length=row_lengths[i],
                        width=column_widths[j],
                    )
                    for j in range(len(partition_ids[i]))
                ]
                for i in range(len(partition_ids))
            ]
        )

    @classmethod
    def pathlib_or_pypath(cls, filepath_or_buffer):
        try:
            import py

            if isinstance(filepath_or_buffer, py.path.local):
                return True
        except ImportError:  # pragma: no cover
            pass
        try:
            import pathlib

            if isinstance(filepath_or_buffer, pathlib.Path):
                return True
        except ImportError:  # pragma: no cover
            pass
        return False

    @classmethod
    def offset(
        cls,
        f,
        offset_size: int,
        quotechar: bytes = b'"',
        is_quoting: bool = True,
    ):
        """
        Moves the file offset at the specified amount of bytes.

        Parameters
        ----------
        f: file object
        offset_size: int
            Number of bytes to read and ignore.
        quotechar: bytes, default b'"'
            Indicate quote in a file.
        is_quoting: bool, default True
            Whether or not to consider quotes.

        Returns
        -------
        bool
            If file pointer reached the end of the file, but did not find
            closing quote returns `False`. `True` in any other case.
        """

        if is_quoting:
            chunk = f.read(offset_size)
            outside_quotes = not chunk.count(quotechar) % 2
        else:
            f.seek(offset_size, os.SEEK_CUR)
            outside_quotes = True

        # after we read `offset_size` bytes, we most likely break the line but
        # the modin implementation doesn't work correctly in the case, so we must
        # make sure that the line is read completely to the lineterminator,
        # which is what the `_read_rows` does
        outside_quotes, _ = cls._read_rows(
            f,
            nrows=1,
            quotechar=quotechar,
            is_quoting=is_quoting,
            outside_quotes=outside_quotes,
        )

        return outside_quotes

    @classmethod
    def partitioned_file(
        cls,
        f,
        fname: str,
        num_partitions: int = None,
        partition_size: int = None,
        nrows: int = None,
        skiprows: int = None,
        skip_header: int = None,
        quotechar: bytes = b'"',
        is_quoting: bool = True,
    ) -> Tuple[List[Tuple[str, int, int]], int]:
        """
        Compute chunk sizes in bytes for every partition.

        Parameters
        ----------
        f: file
            File(s) to be partitioned.
        fname: str
            File name(s) to be partitioned.
        num_partitions: int, optional
            For what number of partitions split a file.
            If not specified grabs the value from `modin.config.NPartitions.get()`.
        partition_size: int, optional
            Specifies the partition size. Overrides num_partitions argument.
        nrows: int, optional
            Number of rows of file to read.
        skiprows: int, optional
            Specifies rows to skip.
        quotechar: bytes, default b'"'
            Indicate quote in a file.
        is_quoting: bool, default True
            Whether or not to consider quotes.

        Returns
        -------
        (list, int)
            First is a list of tuples consisting of the file name, start offset, and end offset of each partition split. Second is the number of rows read including skipped rows.
        """
        file_size = cls.file_size(f)
        if partition_size is None:
            if num_partitions is None:
                num_partitions = NPartitions.get()
            partition_size = max(
                1, num_partitions, (nrows if nrows else file_size) // num_partitions
            )

        skip_read_rows = 0
        if skiprows:
            # TODO(williamma12): Handle when skiprows > number of rows in file. Currently returns empty df.
            outside_quotes, skip_read_rows = cls._read_rows(
                f,
                nrows=skiprows,
                quotechar=quotechar,
                is_quoting=is_quoting,
            )

        start = f.tell()

        result = []
        read_rows_counter = 0
        while f.tell() < file_size:

            if nrows:
                if read_rows_counter >= nrows:
                    # Finish when we have read enough rows.
                    break
                elif read_rows_counter + partition_size > nrows:
                    # Ensure that we will not read more than nrows.
                    partition_size = nrows - read_rows_counter

                outside_quotes, read_rows = cls._read_rows(
                    f,
                    nrows=partition_size,
                    quotechar=quotechar,
                    is_quoting=is_quoting,
                )
                read_rows_counter += read_rows
            else:
                outside_quotes = cls.offset(
                    f,
                    offset_size=partition_size,
                    quotechar=quotechar,
                    is_quoting=is_quoting,
                )

            result.append((fname, start, f.tell()))
            start = f.tell()

            # Add outside_quotes.
            if is_quoting and not outside_quotes:
                warnings.warn("File has mismatched quotes")

        total_rows_read = read_rows_counter + skip_read_rows
        return result, total_rows_read

    @classmethod
    def _read_rows(
        cls,
        f,
        nrows: int,
        quotechar: bytes = b'"',
        is_quoting: bool = True,
        outside_quotes: bool = True,
    ):
        """
        Move the file offset at the specified amount of rows.

        Parameters
        ----------
        f: file object
        nrows: int
            Number of rows to read.
        quotechar: bytes, default b'"'
            Indicate quote in a file.
        is_quoting: bool, default True
            Whether or not to consider quotes.
        outside_quotes: bool, default True
            Whether the file pointer is within quotes or not at the time this function is called.

        Returns
        -------
        tuple of bool and int,
            bool: If file pointer reached the end of the file, but did not find
                closing quote returns `False`. `True` in any other case.
            int: Number of rows that was read.
        """
        if nrows is not None and nrows <= 0:
            return True, 0

        rows_read = 0

        for line in f:
            if is_quoting and line.count(quotechar) % 2:
                outside_quotes = not outside_quotes
            if outside_quotes:
                rows_read += 1
                if rows_read >= nrows:
                    break

        # case when EOF
        if not outside_quotes:
            rows_read += 1

        return outside_quotes, rows_read
