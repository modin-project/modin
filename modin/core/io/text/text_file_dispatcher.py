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
import warnings
import os
import io
import codecs
from typing import Union, Sequence, Optional, Tuple, Callable
from csv import QUOTE_NONE

import numpy as np
import pandas
import pandas._libs.lib as lib
from pandas.core.dtypes.common import is_list_like

from modin.core.io.file_dispatcher import FileDispatcher, OpenFile
from modin.core.storage_formats.pandas.utils import compute_chunksize
from modin.utils import _inherit_docstrings
from modin.core.io.text.utils import CustomNewlineIterator
from modin.config import NPartitions
from modin.error_message import ErrorMessage
from modin.logging import logger_decorator

ColumnNamesTypes = Tuple[Union[pandas.Index, pandas.MultiIndex]]
IndexColType = Union[int, str, bool, Sequence[int], Sequence[str], None]


class TextFileDispatcher(FileDispatcher):
    """Class handles utils for reading text formats files."""

    # The variable allows to set a function with which one partition will be read;
    # Used in dispatchers and parsers
    read_callback = None

    @classmethod
    @logger_decorator("PANDAS-API", "TextFileDispatcher.get_path_or_buffer", "INFO")
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
                    + "used in place of the file handle passed in "
                    + "to load the data"
                )
                return cls.get_path(buffer_filepath)
        return filepath_or_buffer

    @classmethod
    @logger_decorator("PANDAS-API", "TextFileDispatcher.build_partition", "INFO")
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
    @logger_decorator("PANDAS-API", "TextFileDispatcher.pathlib_or_pypath", "INFO")
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
    @logger_decorator("PANDAS-API", "TextFileDispatcher.offset", "INFO")
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
    @logger_decorator("PANDAS-API", "TextFileDispatcher.partitioned_file", "INFO")
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
    @logger_decorator("PANDAS-API", "TextFileDispatcher._read_rows", "INFO")
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
    @logger_decorator("PANDAS-API", "TextFileDispatcher.compute_newline", "INFO")
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
    @logger_decorator("PANDAS-API", "TextFileDispatcher.rows_skipper_builder", "INFO")
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
    @logger_decorator("PANDAS-API", "TextFileDispatcher._define_header_size", "INFO")
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
    @logger_decorator("PANDAS-API", "TextFileDispatcher._define_metadata", "INFO")
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
        column_chunksize = compute_chunksize(df.shape[1], num_splits)
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
    @logger_decorator("PANDAS-API", "TextFileDispatcher._launch_tasks", "INFO")
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
        partition_ids = [None] * len(splits)
        index_ids = [None] * len(splits)
        dtypes_ids = [None] * len(splits)
        for idx, (start, end) in enumerate(splits):
            partition_kwargs.update({"start": start, "end": end})
            *partition_ids[idx], index_ids[idx], dtypes_ids[idx] = cls.deploy(
                cls.parse,
                num_returns=partition_kwargs.get("num_splits") + 2,
                **partition_kwargs,
            )
        return partition_ids, index_ids, dtypes_ids

    @classmethod
    @logger_decorator(
        "PANDAS-API", "TextFileDispatcher.check_parameters_support", "INFO"
    )
    def check_parameters_support(
        cls,
        filepath_or_buffer,
        read_kwargs: dict,
        skiprows_md: Union[Sequence, callable, int],
        header_size: int,
    ) -> bool:
        """
        Check support of only general parameters of `read_*` function.

        Parameters
        ----------
        filepath_or_buffer : str, path object or file-like object
            `filepath_or_buffer` parameter of `read_*` function.
        read_kwargs : dict
            Parameters of `read_*` function.
        skiprows_md : int, array or callable
            `skiprows` parameter modified for easier handling by Modin.
        header_size : int
            Number of rows that are used by header.

        Returns
        -------
        bool
            Whether passed parameters are supported or not.
        """
        skiprows = read_kwargs.get("skiprows")
        if isinstance(filepath_or_buffer, str):
            if not cls.file_exists(filepath_or_buffer):
                return False
        elif not cls.pathlib_or_pypath(filepath_or_buffer):
            return False

        if read_kwargs["chunksize"] is not None:
            return False

        skiprows_supported = True
        if is_list_like(skiprows_md) and skiprows_md[0] < header_size:
            skiprows_supported = False
        elif callable(skiprows):
            # check if `skiprows` callable gives True for any of header indices
            is_intersection = any(
                cls._get_skip_mask(pandas.RangeIndex(header_size), skiprows)
            )
            if is_intersection:
                skiprows_supported = False

        if not skiprows_supported:
            ErrorMessage.single_warning(
                "Values of `header` and `skiprows` parameters have intersections. "
                + "This case is unsupported by Modin, so pandas implementation will be used"
            )
            return False

        return True

    @classmethod
    @_inherit_docstrings(pandas.io.parsers.base_parser.ParserBase._validate_usecols_arg)
    @logger_decorator("PANDAS-API", "TextFileDispatcher._validate_usecols_arg", "INFO")
    def _validate_usecols_arg(cls, usecols):
        msg = (
            "'usecols' must either be list-like of all strings, all unicode, "
            + "all integers or a callable."
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

    @classmethod
    @logger_decorator(
        "PANDAS-API", "TextFileDispatcher._manage_skiprows_parameter", "INFO"
    )
    def _manage_skiprows_parameter(
        cls,
        skiprows: Union[int, Sequence[int], Callable, None] = None,
        header_size: int = 0,
    ) -> Tuple[Union[int, Sequence, Callable], bool, int]:
        """
        Manage `skiprows` parameter of read_csv and read_fwf functions.

        Change `skiprows` parameter in the way Modin could more optimally
        process it. `csv_dispatcher` and `fwf_dispatcher` have two mechanisms of rows skipping:

        1) During file partitioning (setting of file limits that should be read
        by each partition) exact rows can be excluded from partitioning scope,
        thus they won't be read at all and can be considered as skipped. This is
        the most effective way of rows skipping (since it doesn't require any
        actual data reading and postprocessing), but in this case `skiprows`
        parameter can be an integer only. When it possible Modin always uses
        this approach by setting of `skiprows_partitioning` return value.

        2) Rows for skipping can be dropped after full dataset import. This is
        more expensive way since it requires extra IO work and postprocessing
        afterwards, but `skiprows` parameter can be of any non-integer type
        supported by any pandas read function. These rows is
        specified by setting of `skiprows_md` return value.

        In some cases, if `skiprows` is uniformly distributed array (e.g. [1,2,3]),
        `skiprows` can be "squashed" and represented as integer to make a fastpath.
        If there is a gap between the first row for skipping and the last line of
        the header (that will be skipped too), then assign to read this gap first
        (assign the first partition to read these rows be setting of `pre_reading`
        return value). See `Examples` section for details.

        Parameters
        ----------
        skiprows : int, array or callable, optional
            Original `skiprows` parameter of any pandas read function.
        header_size : int, default: 0
            Number of rows that are used by header.

        Returns
        -------
        skiprows_md : int, array or callable
            Updated skiprows parameter. If `skiprows` is an array, this
            array will be sorted. Also parameter will be aligned to
            actual data in the `query_compiler` (which, for example,
            doesn't contain header rows)
        pre_reading : int
            The number of rows that should be read before data file
            splitting for further reading (the number of rows for
            the first partition).
        skiprows_partitioning : int
            The number of rows that should be skipped virtually (skipped during
            data file partitioning).

        Examples
        --------
        Let's consider case when `header`="infer" and `skiprows`=[3,4,5]. In
        this specific case fastpath can be done since `skiprows` is uniformly
        distributed array, so we can "squash" it to integer and set
        `skiprows_partitioning`=3. But if no additional action will be done,
        these three rows will be skipped right after header line, that corresponds
        to `skiprows`=[1,2,3]. Now, to avoid this discrepancy, we need to assign
        the first partition to read data between header line and the first
        row for skipping by setting of `pre_reading` parameter, so setting
        `pre_reading`=2. During data file partitiong, these lines will be assigned
        for reading for the first partition, and then file position will be set at
        the beginning of rows that should be skipped by `skiprows_partitioning`.
        After skipping of these rows, the rest data will be divided between the
        rest of partitions, see rows assignement below:

        0 - header line (skip during partitioning)
        1 - pre_reading (assign to read by the first partition)
        2 - pre_reading (assign to read by the first partition)
        3 - skiprows_partitioning (skip during partitioning)
        4 - skiprows_partitioning (skip during partitioning)
        5 - skiprows_partitioning (skip during partitioning)
        6 - data to partition (divide between the rest of partitions)
        7 - data to partition (divide between the rest of partitions)
        """
        pre_reading = skiprows_partitioning = skiprows_md = 0
        if isinstance(skiprows, int):
            skiprows_partitioning = skiprows
        elif is_list_like(skiprows):
            skiprows_md = np.sort(skiprows)
            if np.all(np.diff(skiprows_md) == 1):
                # `skiprows` is uniformly distributed array.
                pre_reading = (
                    skiprows_md[0] - header_size if skiprows_md[0] > header_size else 0
                )
                skiprows_partitioning = len(skiprows_md)
                skiprows_md = 0
            elif skiprows_md[0] > header_size:
                skiprows_md = skiprows_md - header_size
        elif callable(skiprows):

            def skiprows_func(x):
                return skiprows(x + header_size)

            skiprows_md = skiprows_func

        return skiprows_md, pre_reading, skiprows_partitioning

    @classmethod
    @logger_decorator("PANDAS-API", "TextFileDispatcher._define_index", "INFO")
    def _define_index(
        cls,
        index_ids: list,
        index_name: str,
    ) -> Tuple[IndexColType, list]:
        """
        Compute the resulting DataFrame index and index lengths for each of partitions.

        Parameters
        ----------
        index_ids : list
            Array with references to the partitions index objects.
        index_name : str
            Name that should be assigned to the index if `index_col`
            is not provided.

        Returns
        -------
        new_index : IndexColType
            Index that should be passed to the new_frame constructor.
        row_lengths : list
            Partitions rows lengths.
        """
        index_objs = cls.materialize(index_ids)
        if len(index_objs) == 0 or isinstance(index_objs[0], int):
            row_lengths = index_objs
            new_index = pandas.RangeIndex(sum(index_objs))
        else:
            row_lengths = [len(o) for o in index_objs]
            new_index = index_objs[0].append(index_objs[1:])
            new_index.name = index_name

        return new_index, row_lengths

    @classmethod
    @logger_decorator("PANDAS-API", "TextFileDispatcher._get_new_qc", "INFO")
    def _get_new_qc(
        cls,
        partition_ids: list,
        index_ids: list,
        dtypes_ids: list,
        index_col: IndexColType,
        index_name: str,
        column_widths: list,
        column_names: ColumnNamesTypes,
        skiprows_md: Union[Sequence, callable, None] = None,
        header_size: int = None,
        **kwargs,
    ):
        """
        Get new query compiler from data received from workers.

        Parameters
        ----------
        partition_ids : list
            Array with references to the partitions data.
        index_ids : list
            Array with references to the partitions index objects.
        dtypes_ids : list
            Array with references to the partitions dtypes objects.
        index_col : IndexColType
            `index_col` parameter of `read_csv` function.
        index_name : str
            Name that should be assigned to the index if `index_col`
            is not provided.
        column_widths : list
            Number of columns in each partition.
        column_names : ColumnNamesTypes
            Array with columns names.
        skiprows_md : array-like or callable, optional
            Specifies rows to skip.
        header_size : int, default: 0
            Number of rows, that occupied by header.
        **kwargs : dict
            Parameters of `read_csv` function needed for postprocessing.

        Returns
        -------
        new_query_compiler : BaseQueryCompiler
            New query compiler, created from `new_frame`.
        """
        new_index, row_lengths = cls._define_index(index_ids, index_name)
        # Compute dtypes by collecting and combining all of the partition dtypes. The
        # reported dtypes from differing rows can be different based on the inference in
        # the limited data seen by each worker. We use pandas to compute the exact dtype
        # over the whole column for each column. The index is set below.
        dtypes = cls.get_dtypes(dtypes_ids) if len(dtypes_ids) > 0 else None
        # Compose modin partitions from `partition_ids`
        partition_ids = cls.build_partition(partition_ids, row_lengths, column_widths)

        # Set the index for the dtypes to the column names
        if isinstance(dtypes, pandas.Series):
            dtypes.index = column_names
        else:
            dtypes = pandas.Series(dtypes, index=column_names)

        new_frame = cls.frame_cls(
            partition_ids,
            new_index,
            column_names,
            row_lengths,
            column_widths,
            dtypes=dtypes,
        )
        new_query_compiler = cls.query_compiler_cls(new_frame)
        skipfooter = kwargs.get("skipfooter", None)
        if skipfooter:
            new_query_compiler = new_query_compiler.drop(
                new_query_compiler.index[-skipfooter:]
            )
        if skiprows_md is not None:
            # skip rows that passed as array or callable
            nrows = kwargs.get("nrows", None)
            index_range = pandas.RangeIndex(len(new_query_compiler.index))
            if is_list_like(skiprows_md):
                new_query_compiler = new_query_compiler.view(
                    index=index_range.delete(skiprows_md)
                )
            elif callable(skiprows_md):
                skip_mask = cls._get_skip_mask(index_range, skiprows_md)
                if not isinstance(skip_mask, np.ndarray):
                    skip_mask = skip_mask.to_numpy("bool")
                view_idx = index_range[~skip_mask]
                new_query_compiler = new_query_compiler.view(index=view_idx)
            else:
                raise TypeError(
                    f"Not acceptable type of `skiprows` parameter: {type(skiprows_md)}"
                )

            if not isinstance(new_query_compiler.index, pandas.MultiIndex):
                new_query_compiler = new_query_compiler.reset_index(drop=True)

            if nrows:
                new_query_compiler = new_query_compiler.view(
                    pandas.RangeIndex(len(new_query_compiler.index))[:nrows]
                )
        if index_col is None:
            new_query_compiler._modin_frame.synchronize_labels(axis=0)

        return new_query_compiler

    @classmethod
    @logger_decorator("PANDAS-API", "TextFileDispatcher._read", "INFO")
    def _read(cls, filepath_or_buffer, **kwargs):
        """
        Read data from `filepath_or_buffer` according to `kwargs` parameters.

        Used in `read_csv` and `read_fwf` Modin implementations.

        Parameters
        ----------
        filepath_or_buffer : str, path object or file-like object
            `filepath_or_buffer` parameter of read functions.
        **kwargs : dict
            Parameters of read functions.

        Returns
        -------
        new_query_compiler : BaseQueryCompiler
            Query compiler with imported data for further processing.
        """
        filepath_or_buffer_md = (
            cls.get_path(filepath_or_buffer)
            if isinstance(filepath_or_buffer, str)
            else cls.get_path_or_buffer(filepath_or_buffer)
        )
        compression_infered = cls.infer_compression(
            filepath_or_buffer, kwargs["compression"]
        )
        # Getting frequently used kwargs;
        # They should be defined in higher level
        names = kwargs["names"]
        index_col = kwargs["index_col"]
        encoding = kwargs["encoding"]
        skiprows = kwargs["skiprows"]
        header = kwargs["header"]
        # Define header size for further skipping (Header can be skipped because header
        # information will be obtained further from empty_df, so no need to handle it
        # by workers)
        header_size = cls._define_header_size(
            header,
            names,
        )
        (
            skiprows_md,
            pre_reading,
            skiprows_partitioning,
        ) = cls._manage_skiprows_parameter(skiprows, header_size)
        should_handle_skiprows = skiprows_md is not None and not isinstance(
            skiprows_md, int
        )

        use_modin_impl = cls.check_parameters_support(
            filepath_or_buffer,
            kwargs,
            skiprows_md,
            header_size,
        )
        if not use_modin_impl:
            return cls.single_worker_read(
                filepath_or_buffer, callback=cls.read_callback, **kwargs
            )

        is_quoting = kwargs["quoting"] != QUOTE_NONE
        # In these cases we should pass additional metadata
        # to the workers to match pandas output
        pass_names = names in [None, lib.no_default] and (
            skiprows is not None or kwargs["skipfooter"] != 0
        )

        pd_df_metadata = cls.read_callback(
            filepath_or_buffer,
            **dict(kwargs, nrows=1, skipfooter=0, index_col=index_col),
        )
        column_names = pd_df_metadata.columns
        column_widths, num_splits = cls._define_metadata(pd_df_metadata, column_names)

        # kwargs that will be passed to the workers
        partition_kwargs = dict(
            kwargs,
            fname=filepath_or_buffer_md,
            num_splits=num_splits,
            header_size=header_size if not pass_names else 0,
            names=names if not pass_names else column_names,
            header=header if not pass_names else "infer",
            skipfooter=0,
            skiprows=None,
            nrows=None,
            compression=compression_infered,
        )

        with OpenFile(filepath_or_buffer_md, "rb", compression_infered) as f:
            old_pos = f.tell()
            fio = io.TextIOWrapper(f, encoding=encoding, newline="")
            newline, quotechar = cls.compute_newline(
                fio, encoding, kwargs.get("quotechar", '"')
            )
            f.seek(old_pos)
            splits = cls.partitioned_file(
                f,
                num_partitions=NPartitions.get(),
                nrows=kwargs["nrows"] if not should_handle_skiprows else None,
                skiprows=skiprows_partitioning,
                quotechar=quotechar,
                is_quoting=is_quoting,
                encoding=encoding,
                newline=newline,
                header_size=header_size,
                pre_reading=pre_reading,
            )

        partition_ids, index_ids, dtypes_ids = cls._launch_tasks(
            splits, callback=cls.read_callback, **partition_kwargs
        )

        new_query_compiler = cls._get_new_qc(
            partition_ids=partition_ids,
            index_ids=index_ids,
            dtypes_ids=dtypes_ids,
            index_col=index_col,
            index_name=pd_df_metadata.index.name,
            column_widths=column_widths,
            column_names=column_names,
            skiprows_md=skiprows_md if should_handle_skiprows else None,
            header_size=header_size,
            skipfooter=kwargs["skipfooter"],
            parse_dates=kwargs["parse_dates"],
            nrows=kwargs["nrows"] if should_handle_skiprows else None,
        )
        return new_query_compiler

    @classmethod
    @logger_decorator("PANDAS-API", "TextFileDispatcher._get_skip_mask", "INFO")
    def _get_skip_mask(cls, rows_index: pandas.Index, skiprows: Callable):
        """
        Get mask of skipped by callable `skiprows` rows.

        Parameters
        ----------
        rows_index : pandas.Index
            Rows index to get mask for.
        skiprows : Callable
            Callable to check whether row index should be skipped.

        Returns
        -------
        pandas.Index
        """
        try:
            # direct `skiprows` call is more efficient than using of
            # map method, but in some cases it can work incorrectly, e.g.
            # when `skiprows` contains `in` operator
            mask = skiprows(rows_index)
            assert is_list_like(mask)
        except (ValueError, TypeError, AssertionError):
            # ValueError can be raised if `skiprows` callable contains membership operator
            # TypeError is raised if `skiprows` callable contains bitwise operator
            # AssertionError is raised if unexpected behavior was detected
            mask = rows_index.map(skiprows)

        return mask
