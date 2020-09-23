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

from modin.engines.base.io.file_reader import FileReader
import numpy as np
import warnings
import os


class TextFileReader(FileReader):
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
        nrows=None,
        skiprows=None,
        chunk_size_bytes=None,
        quotechar=b'"',
        is_quoting=True,
    ):
        """
        Moves the file offset at the specified amount of bytes/rows.

        Parameters
        ----------
            f: file object
            nrows: int, number of rows to read. Optional, if not specified will only
                consider `chunk_size_bytes` parameter.
            chunk_size_bytes: int, Will read new rows while file pointer
                is less than `chunk_size_bytes`. Optional, if not specified will only
                consider `nrows` parameter.
            skiprows: array or callable (optional), specifies rows to skip
            quotechar: char that indicates quote in a file
                (optional, by default it's '\"')
            is_quoting: bool, Whether or not to consider quotes
                (optional, by default it's `True`)

        Returns
        -------
            bool: If file pointer reached the end of the file, but did not find
            closing quote returns `False`. `True` in any other case.
        """
        assert (
            nrows is not None or chunk_size_bytes is not None
        ), "`nrows` and `chunk_size_bytes` can't be None at the same time"

        if nrows is not None or skiprows is not None:
            return cls._read_rows(
                f,
                nrows=nrows,
                skiprows=skiprows,
                quotechar=quotechar,
                is_quoting=is_quoting,
                max_bytes=chunk_size_bytes,
            )[0]

        outside_quotes = True

        if is_quoting:
            chunk = f.read(chunk_size_bytes)
            line = f.readline()  # Ensure we read up to a newline
            # We need to ensure that one row isn't split across different partitions
            outside_quotes = not ((chunk.count(quotechar) + line.count(quotechar)) % 2)
            while not outside_quotes:
                line = f.readline()
                outside_quotes = line.count(quotechar) % 2
                if not line:
                    break
        else:
            f.seek(chunk_size_bytes, os.SEEK_CUR)
            f.readline()
        return outside_quotes

    @classmethod
    def partitioned_file(
        cls,
        f,
        nrows=None,
        skiprows=None,
        num_partitions=None,
        quotechar=b'"',
        is_quoting=True,
        from_begin=False,
    ):
        """Computes chunk sizes in bytes for every partition.

        Parameters
        ----------
            f: file to be partitioned
            nrows: int (optional), number of rows of file to read
            skiprows: array or callable (optional), specifies rows to skip
            num_partitions: int, for what number of partitions split a file.
                Optional, if not specified grabs the value from `modin.pandas.DEFAULT_NPARTITIONS`
            quotechar: char that indicates quote in a file
                (optional, by default it's '\"')
            is_quoting: bool, Whether or not to consider quotes
                (optional, by default it's `True`)
            from_begin: bool, Whether or not to set the file pointer to the begining of the file
                (optional, by default it's `False`)

        Returns
        -------
            An array, where each element of array is a tuple of two ints:
            beginning and the end offsets of the current chunk.
        """
        if num_partitions is None:
            from modin.pandas import DEFAULT_NPARTITIONS

            num_partitions = DEFAULT_NPARTITIONS

        result = []

        old_position = f.tell()
        if from_begin:
            f.seek(0, os.SEEK_SET)

        current_start = f.tell()
        total_bytes = cls.file_size(f)

        # if `nrows` are specified we want to use rows as a part measure
        if nrows is not None:
            chunk_size_bytes = None
            rows_per_part = max(1, num_partitions, nrows // num_partitions)
        else:
            chunk_size_bytes = max(1, num_partitions, total_bytes // num_partitions)
            rows_per_part = None
            nrows = float("inf")

        rows_readed = 0
        while f.tell() < total_bytes and rows_readed < nrows:
            if rows_per_part is not None and rows_readed + rows_per_part > nrows:
                rows_per_part = nrows - rows_readed

            outside_quotes = cls.offset(
                f,
                nrows=rows_per_part,
                skiprows=skiprows,
                chunk_size_bytes=chunk_size_bytes,
                quotechar=quotechar,
                is_quoting=is_quoting,
            )

            result.append((current_start, f.tell()))
            current_start = f.tell()
            if rows_per_part is not None:
                rows_readed += rows_per_part

            if is_quoting and not outside_quotes:
                warnings.warn("File has mismatched quotes")

        f.seek(old_position, os.SEEK_SET)

        return result

    @classmethod
    def _read_rows(
        cls,
        f,
        nrows=None,
        skiprows=None,
        quotechar=b'"',
        is_quoting=True,
        max_bytes=None,
    ):
        """
        Moves the file offset at the specified amount of rows
        Note: the difference between `offset` is that `_read_rows` is more
            specific version of `offset` which is focused of reading **rows**.
            In common case it's better to use `offset`.

        Parameters
        ----------
            f: file object
            nrows: int, number of rows to read. Optional, if not specified will only
                consider `max_bytes` parameter.
            skiprows: int, array or callable (optional), specifies rows to skip
            quotechar: char that indicates quote in a file
                (optional, by default it's '\"')
            is_quoting: bool, Whether or not to consider quotes
                (optional, by default it's `True`)
            max_bytes: int, Will read new rows while file pointer
                is less than `max_bytes`. Optional, if not specified will only
                consider `nrows` parameter, if both not specified will read till
                the end of the file.

        Returns
        -------
            tuple of bool and int,
                bool: If file pointer reached the end of the file, but did not find
                closing quote returns `False`. `True` in any other case.
                int: Number of rows that was readed.
        """
        assert skiprows is None or isinstance(
            skiprows, int
        ), f"Skiprows as a {type(skiprows)} is not supported yet."

        if nrows is None and max_bytes is None:
            max_bytes = float("inf")

        if nrows is not None and nrows <= 0:
            return True, 0

        # we need this condition to avoid unnecessary checks in `stop_condition`
        # which executes in a huge for loop
        if nrows is not None and max_bytes is None:
            stop_condition = lambda rows_readed: rows_readed >= nrows  # noqa (E731)
        elif nrows is not None and max_bytes is not None:
            stop_condition = (
                lambda rows_readed: f.tell() >= max_bytes or rows_readed >= nrows
            )  # noqa (E731)
        else:
            stop_condition = lambda rows_readed: f.tell() >= max_bytes  # noqa (E731)

        if max_bytes is not None:
            max_bytes = max_bytes + f.tell()

        rows_readed = 0
        outside_quotes = True
        for line in f:
            if is_quoting and line.count(quotechar) % 2:
                outside_quotes = not outside_quotes
            if outside_quotes:
                rows_readed += 1
                if stop_condition(rows_readed):
                    break

        if not outside_quotes:
            rows_readed += 1

        return outside_quotes, rows_readed
