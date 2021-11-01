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
Module houses `TextFileDispatcher` class.

`TextFileDispatcher` contains utils for text formats files, inherits util functions for
files from `FileDispatcher` class and can be used as base class for dipatchers of SQL queries.
"""

from modin.core.io.file_dispatcher import FileDispatcher
from modin.core.storage_formats.pandas.utils import compute_chunksize
from modin.utils import _inherit_docstrings
from modin.core.io.text.utils import CustomNewlineIterator
import numpy as np
import warnings
import os
import codecs
from typing import Union, Sequence, Optional, Tuple
import pandas
import pandas._libs.lib as lib
from pandas.core.dtypes.common import is_list_like

from modin.config import NPartitions

ColumnNamesTypes = Tuple[Union[pandas.Index, pandas.MultiIndex, pandas.Int64Index]]


class TextFileDispatcher(FileDispatcher):
    """
    Class handles utils for reading text formats files.

    Inherits some util functions for processing files from `FileDispatcher` class.
    """

    @classmethod
    def get_path_or_buffer(cls, filepath_or_buffer):
        """
        Extract path from `filepath_or_buffer`.

        Parameters
        ----------
        filepath_or_buffer : str, path object or file-like object
            `filepath_or_buffer` parameter of `read_csv` function.

        Returns
        -------
        str or path object
            verified `filepath_or_buffer` parameter.

        Notes
        -----
        Given a buffer, try and extract the filepath from it so that we can
        use it without having to fall back to pandas and share file objects between
        workers. Given a filepath, return it immediately.
        """
        if hasattr(filepath_or_buffer, "name"):
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
        """
        Build array with partitions of `cls.frame_partition_cls` class.

        Parameters
        ----------
        partition_ids : list
                Array with references to the partitions data.
        row_lengths : list
                Partitions rows lengths.
        column_widths : list
                Number of columns in each partition.

        Returns
        -------
        np.ndarray
            array with shape equals to the shape of `partition_ids` and
            filed with partitions objects.
        """
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
        """
        Check if `filepath_or_buffer` is instance of `py.path.local` or `pathlib.Path`.

        Parameters
        ----------
        filepath_or_buffer : str, path object or file-like object
            `filepath_or_buffer` parameter of `read_csv` function.

        Returns
        -------
        bool
            Whether or not `filepath_or_buffer` is instance of `py.path.local`
            or `pathlib.Path`.
        """
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
        encoding: str = None,
        newline: bytes = None,
    ):
        """
        Move the file offset at the specified amount of bytes.

        Parameters
        ----------
        f : file-like object
            File handle that should be used for offset movement.
        offset_size : int
            Number of bytes to read and ignore.
        quotechar : bytes, default: b'"'
            Indicate quote in a file.
        is_quoting : bool, default: True
            Whether or not to consider quotes.
        encoding : str, optional
            Encoding of `f`.
        newline : bytes, optional
            Byte or sequence of bytes indicating line endings.

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
            encoding=encoding,
            newline=newline,
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
        encoding: str = None,
        newline: bytes = None,
        header_size: int = 0,
        pre_reading: int = 0,
    ):
        """
        Compute chunk sizes in bytes for every partition.

        Parameters
        ----------
        f : file-like object
            File handle of file to be partitioned.
        num_partitions : int, optional
            For what number of partitions split a file.
            If not specified grabs the value from `modin.config.NPartitions.get()`.
        nrows : int, optional
            Number of rows of file to read.
        skiprows : int, optional
            Specifies rows to skip.
        quotechar : bytes, default: b'"'
            Indicate quote in a file.
        is_quoting : bool, default: True
            Whether or not to consider quotes.
        encoding : str, optional
            Encoding of `f`.
        newline : bytes, optional
            Byte or sequence of bytes indicating line endings.
        header_size : int, default: 0
            Number of rows, that occupied by header.
        pre_reading : int, default: 0
            Number of rows between header and skipped rows, that should be read.

        Returns
        -------
        list
            List with the next elements:
                int : partition start read byte
                int : partition end read byte
        """
        read_rows_counter = 0
        outside_quotes = True

        if num_partitions is None:
            num_partitions = NPartitions.get() - 1 if pre_reading else NPartitions.get()

        rows_skipper = cls.rows_skipper_builder(
            f, quotechar, is_quoting=is_quoting, encoding=encoding, newline=newline
        )
        result = []

        file_size = cls.file_size(f)

        rows_skipper(header_size)

        if pre_reading:
            pre_reading_start = f.tell()
            outside_quotes, read_rows = cls._read_rows(
                f,
                nrows=pre_reading,
                quotechar=quotechar,
                is_quoting=is_quoting,
                outside_quotes=outside_quotes,
                encoding=encoding,
                newline=newline,
            )
            read_rows_counter += read_rows

            result.append((pre_reading_start, f.tell()))

            # add outside_quotes
            if is_quoting and not outside_quotes:
                warnings.warn("File has mismatched quotes")

        rows_skipper(skiprows)

        start = f.tell()

        if nrows:
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
                    encoding=encoding,
                    newline=newline,
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
                    encoding=encoding,
                    newline=newline,
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
        encoding: str = None,
        newline: bytes = None,
    ):
        """
        Move the file offset at the specified amount of rows.

        Parameters
        ----------
        f : file-like object
            File handle that should be used for offset movement.
        nrows : int
            Number of rows to read.
        quotechar : bytes, default: b'"'
            Indicate quote in a file.
        is_quoting : bool, default: True
            Whether or not to consider quotes.
        outside_quotes : bool, default: True
            Whether the file pointer is within quotes or not at the time this function is called.
        encoding : str, optional
            Encoding of `f`.
        newline : bytes, optional
            Byte or sequence of bytes indicating line endings.

        Returns
        -------
        bool
            If file pointer reached the end of the file, but did not find closing quote
            returns `False`. `True` in any other case.
        int
            Number of rows that were read.
        """
        if nrows is not None and nrows <= 0:
            return True, 0

        rows_read = 0

        if encoding and (
            "utf" in encoding
            and "8" not in encoding
            or encoding == "unicode_escape"
            or encoding.replace("-", "_") == "utf_8_sig"
        ):
            iterator = CustomNewlineIterator(f, newline)
        else:
            iterator = f

        for line in iterator:
            if is_quoting and line.count(quotechar) % 2:
                outside_quotes = not outside_quotes
            if outside_quotes:
                rows_read += 1
                if rows_read >= nrows:
                    break

        if isinstance(iterator, CustomNewlineIterator):
            iterator.seek()

        # case when EOF
        if not outside_quotes:
            rows_read += 1

        return outside_quotes, rows_read

    @classmethod
    def compute_newline(cls, file_like, encoding, quotechar):
        """
        Compute byte or sequence of bytes indicating line endings.

        Parameters
        ----------
        file_like : file-like object
            File handle that should be used for line endings computing.
        encoding : str
            Encoding of `file_like`.
        quotechar : str
            Quotechar used for parsing `file-like`.

        Returns
        -------
        bytes
            line endings
        """
        newline = None

        if encoding is None:
            return newline, quotechar.encode("UTF-8")

        quotechar = quotechar.encode(encoding)
        encoding = encoding.replace("-", "_")

        if (
            "utf" in encoding
            and "8" not in encoding
            or encoding == "unicode_escape"
            or encoding == "utf_8_sig"
        ):
            # trigger for computing f.newlines
            file_like.readline()
            # in bytes
            newline = file_like.newlines.encode(encoding)
            boms = ()
            if encoding == "utf_8_sig":
                boms = (codecs.BOM_UTF8,)
            elif "16" in encoding:
                boms = (codecs.BOM_UTF16_BE, codecs.BOM_UTF16_LE)
            elif "32" in encoding:
                boms = (codecs.BOM_UTF32_BE, codecs.BOM_UTF32_LE)

            for bom in boms:
                if newline.startswith(bom):
                    bom_len = len(bom)
                    newline = newline[bom_len:]
                    quotechar = quotechar[bom_len:]
                    break

        return newline, quotechar

    # _read helper functions
    @classmethod
    def rows_skipper_builder(
        cls, f, quotechar, is_quoting, encoding=None, newline=None
    ):
        """
        Build object for skipping passed number of lines.

        Parameters
        ----------
        f : file-like object
            File handle that should be used for offset movement.
        quotechar : bytes
            Indicate quote in a file.
        is_quoting : bool
            Whether or not to consider quotes.
        encoding : str, optional
            Encoding of `f`.
        newline : bytes, optional
            Byte or sequence of bytes indicating line endings.

        Returns
        -------
        object
            skipper object.
        """

        def skipper(n):
            if n == 0 or n is None:
                return 0
            else:
                return cls._read_rows(
                    f,
                    quotechar=quotechar,
                    is_quoting=is_quoting,
                    nrows=n,
                    encoding=encoding,
                    newline=newline,
                )[1]

        return skipper

    @classmethod
    def _define_header_size(
        cls,
        header: Union[int, Sequence[int], str, None] = "infer",
        names: Optional[Sequence] = lib.no_default,
    ) -> int:
        """
        Define the number of rows that are used by header.

        Parameters
        ----------
        header : int, list of int or str, default: "infer"
            Original `header` parameter of `read_csv` function.
        names :  array-like, optional
            Original names parameter of `read_csv` function.

        Returns
        -------
        header_size : int
            The number of rows that are used by header.
        """
        header_size = 0
        if header == "infer" and names in [lib.no_default, None]:
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
        column_names: ColumnNamesTypes,
    ) -> Tuple[list, int]:
        """
        Define partitioning metadata.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame to split.
        column_names : ColumnNamesTypes
            Column names of df.

        Returns
        -------
        column_widths : list
            Column width to use during new frame creation (number of
            columns for each partition).
        num_splits : int
            The maximum number of splits to separate the DataFrame into.
        """
        # This is the number of splits for the columns
        num_splits = min(len(column_names) or 1, NPartitions.get())
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
            column_widths = [
                column_chunksize
                if len(column_names) > (column_chunksize * (i + 1))
                else 0
                if len(column_names) < (column_chunksize * i)
                else len(column_names) - (column_chunksize * i)
                for i in range(num_splits)
            ]

        return column_widths, num_splits

    @classmethod
    def _launch_tasks(cls, splits: list, **partition_kwargs) -> Tuple[list, list, list]:
        """
        Launch tasks to read partitions.

        Parameters
        ----------
        splits : list
            List of tuples with partitions data, which defines
            parser task (start/end read bytes and etc.).
        **partition_kwargs : dict
            `kwargs` that should be passed to the parser function.

        Returns
        -------
        partition_ids : list
            array with references to the partitions data.
        index_ids : list
            array with references to the partitions index objects.
        dtypes_ids : list
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

    @classmethod
    @_inherit_docstrings(pandas.io.parsers.base_parser.ParserBase._validate_usecols_arg)
    def _validate_usecols_arg(cls, usecols):
        msg = (
            "'usecols' must either be list-like of all strings, all unicode, "
            "all integers or a callable."
        )
        if usecols is not None:
            if callable(usecols):
                return usecols, None

            if not is_list_like(usecols):
                raise ValueError(msg)

            usecols_dtype = lib.infer_dtype(usecols, skipna=False)

            if usecols_dtype not in ("empty", "integer", "string"):
                raise ValueError(msg)

            usecols = set(usecols)

            return usecols, usecols_dtype
        return usecols, None
