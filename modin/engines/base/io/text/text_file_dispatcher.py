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
from modin.data_management.utils import compute_chunksize
import numpy as np
import warnings
import io
import os
from typing import Union, Sequence, Optional, Tuple
import pandas

from modin.config import NPartitions

ColumnNamesTypes = Tuple[Union[pandas.Index, pandas.MultiIndex, pandas.Int64Index]]


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
        num_partitions: int = None,
        nrows: int = None,
        skiprows: int = None,
        quotechar: bytes = b'"',
        is_quoting: bool = True,
    ):
        """
        Compute chunk sizes in bytes for every partition.

        Parameters
        ----------
        f: file to be partitioned
        num_partitions: int, optional
            For what number of partitions split a file.
            If not specified grabs the value from `modin.config.NPartitions.get()`
        nrows: int, optional
            Number of rows of file to read.
        skiprows: array or callable, optional
            Specifies rows to skip.
        quotechar: bytes, default b'"'
            Indicate quote in a file.
        is_quoting: bool, default True
            Whether or not to consider quotes.

        Returns
        -------
        An array, where each element of array is a tuple of two ints:
        beginning and the end offsets of the current chunk.
        """
        if num_partitions is None:
            num_partitions = NPartitions.get()

        rows_skipper = cls.rows_skipper_builder(f, quotechar, is_quoting=is_quoting)
        result = []

        file_size = cls.file_size(f)

        rows_skipper(skiprows)

        start = f.tell()

        if nrows:
            read_rows_counter = 0
            partition_size = max(1, num_partitions, nrows // num_partitions)
            while f.tell() < file_size and read_rows_counter < nrows:
                if read_rows_counter + partition_size > nrows:
                    # it's possible only if is_quoting==True
                    partition_size = nrows - read_rows_counter
                outside_quotes, read_rows = cls._read_rows(
                    f,
                    nrows=partition_size,
                    quotechar=quotechar,
                    is_quoting=is_quoting,
                )
                result.append((start, f.tell()))
                start = f.tell()
                read_rows_counter += read_rows

                # add outside_quotes
                if is_quoting and not outside_quotes:
                    warnings.warn("File has mismatched quotes")
        else:
            partition_size = max(1, num_partitions, file_size // num_partitions)
            while f.tell() < file_size:
                outside_quotes = cls.offset(
                    f,
                    offset_size=partition_size,
                    quotechar=quotechar,
                    is_quoting=is_quoting,
                )

                result.append((start, f.tell()))
                start = f.tell()

                # add outside_quotes
                if is_quoting and not outside_quotes:
                    warnings.warn("File has mismatched quotes")

        return result

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

    # _read helper functions
    @classmethod
    def rows_skipper_builder(cls, f, quotechar, is_quoting):
        def skipper(n):
            if n == 0 or n is None:
                return 0
            else:
                return cls._read_rows(
                    f,
                    quotechar=quotechar,
                    is_quoting=is_quoting,
                    nrows=n,
                )[1]

        return skipper

    @classmethod
    def _define_header_size(
        cls,
        header: Union[int, Sequence[int], str, None] = "infer",
        names: Optional[Sequence] = None,
    ) -> int:
        """
        Defines the number of rows that are used by header.
        Parameters
        ----------
        header: int, list of int or str ("infer")
                Original header parameter of read_csv function.
        names:  array
                Original names parameter of read_csv function.
        Returns
        -------
        header_size: int
                The number of rows that are used by header.
        """
        header_size = 0
        if header == "infer" and names is None:
            header_size += 1
        elif isinstance(header, int):
            header_size += header + 1
        elif hasattr(header, "__iter__") and not isinstance(header, str):
            header_size += max(header) + 1

        return header_size

    @classmethod
    def _define_metadata(
        cls,
        df: pandas.DataFrame,
        num_splits: int,
        column_names: ColumnNamesTypes,
    ) -> Tuple[list, int]:
        """
        Define partitioning metadata.
        ----------
        df: pandas.DataFrame
            The DataFrame to split.
        num_splits: int
            The maximum number of splits to separate the DataFrame into.
        column_names: ColumnNamesTypes
            column names of df.

        Returns
        -------
        column_widths: list
                Column width to use during new frame creation (number of
                columns for each partition).
        num_splits: int
                Updated `num_splits` parameter.
        """
        column_chunksize = compute_chunksize(df, num_splits, axis=1)
        if column_chunksize > len(column_names):
            column_widths = [len(column_names)]
            # This prevents us from unnecessarily serializing a bunch of empty
            # objects.
            num_splits = 1
        else:
            # split columns into chunks with maximal size column_chunksize, for example
            # if num_splits == 4, len(column_names) == 80 and column_chunksize == 32,
            # column_widths will be [32, 32, 16, 0]
            column_widths = []
            for i in range(num_splits):
                if len(column_names) > (column_chunksize * i):
                    if len(column_names) > (column_chunksize * (i + 1)):
                        column_widths.append(column_chunksize)
                    else:
                        column_widths.append(len(column_names) - (column_chunksize * i))
                else:
                    column_widths.append(0)

        return column_widths, num_splits

    @classmethod
    def _launch_tasks(cls, splits: list, **partition_kwargs) -> Tuple[list, list, list]:
        """
        Launch tasks to read partitions.
        ----------
        splits: list
            list of tuples with partitions data, which defines
            parser task (start/end read bytes and etc.)
        partition_kwargs:
            kwargs that should be passed to the parser function.

        Returns
        -------
        partition_ids: list
                array with references to the partitions data.
        index_ids: list
                array with references to the partitions index objects.
        dtypes_ids: list
                array with references to the partitions dtypes objects.
        """
        partition_ids = []
        index_ids = []
        dtypes_ids = []
        for start, end in splits:
            partition_kwargs.update({"start": start, "end": end})
            partition_id = cls.deploy(
                cls.parse, partition_kwargs.get("num_splits") + 2, partition_kwargs
            )
            partition_ids.append(partition_id[:-2])
            index_ids.append(partition_id[-2])
            dtypes_ids.append(partition_id[-1])

        return partition_ids, index_ids, dtypes_ids
