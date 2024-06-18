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
import codecs
import io
import os
import warnings
from csv import QUOTE_NONE
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas
import pandas._libs.lib as lib
from pandas.core.dtypes.common import is_list_like
from pandas.io.common import stringify_path

from modin.config import MinColumnPartitionSize, NPartitions
from modin.core.io.file_dispatcher import FileDispatcher, OpenFile
from modin.core.io.text.utils import CustomNewlineIterator
from modin.core.storage_formats.pandas.utils import compute_chunksize
from modin.utils import _inherit_docstrings

ColumnNamesTypes = Tuple[Union[pandas.Index, pandas.MultiIndex]]
IndexColType = Union[int, str, bool, Sequence[int], Sequence[str], None]


class TextFileDispatcher(FileDispatcher):
    """Class handles utils for reading text formats files."""

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
        if (
            hasattr(filepath_or_buffer, "name")
            and hasattr(filepath_or_buffer, "seekable")
            and filepath_or_buffer.seekable()
            and filepath_or_buffer.tell() == 0
        ):
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
        get_metadata_kw: dict = None,
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
        get_metadata_kw : dict, optional
            Keyword arguments for `cls.read_callback` to compute metadata if needed.
            This option is not compatible with `pre_reading!=0`.

        Returns
        -------
        list
            List with the next elements:
                int : partition start read byte
                int : partition end read byte
        pandas.DataFrame or None
            Dataframe from which metadata can be retrieved. Can be None if `get_metadata_kw=None`.
        """
        if get_metadata_kw is not None and pre_reading != 0:
            raise ValueError(
                f"Incompatible combination of parameters: {get_metadata_kw=}, {pre_reading=}"
            )
        read_rows_counter = 0
        outside_quotes = True

        if num_partitions is None:
            num_partitions = NPartitions.get() - 1 if pre_reading else NPartitions.get()

        rows_skipper = cls.rows_skipper_builder(
            f, quotechar, is_quoting=is_quoting, encoding=encoding, newline=newline
        )
        result = []

        file_size = cls.file_size(f)

        pd_df_metadata = None
        if pre_reading:
            rows_skipper(header_size)
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
        else:
            rows_skipper(skiprows)
            if get_metadata_kw:
                start = f.tell()
                # For correct behavior, if we want to avoid double skipping rows,
                # we need to get metadata after skipping.
                pd_df_metadata = cls.read_callback(f, **get_metadata_kw)
                f.seek(start)
            rows_skipper(header_size)

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
        return result, pd_df_metadata

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
        min_block_size = MinColumnPartitionSize.get()
        column_chunksize = compute_chunksize(df.shape[1], num_splits, min_block_size)
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
                (
                    column_chunksize
                    if len(column_names) > (column_chunksize * (i + 1))
                    else (
                        0
                        if len(column_names) < (column_chunksize * i)
                        else len(column_names) - (column_chunksize * i)
                    )
                )
                for i in range(num_splits)
            ]

        return column_widths, num_splits

    _parse_func = None

    @classmethod
    def preprocess_func(cls):  # noqa: RT01
        """Prepare a function for transmission to remote workers."""
        if cls._parse_func is None:
            cls._parse_func = cls.put(cls.parse)
        return cls._parse_func

    @classmethod
    def _launch_tasks(
        cls, splits: list, *partition_args, **partition_kwargs
    ) -> Tuple[list, list, list]:
        """
        Launch tasks to read partitions.

        Parameters
        ----------
        splits : list
            List of tuples with partitions data, which defines
            parser task (start/end read bytes and etc.).
        *partition_args : tuple
            Positional arguments to be passed to the parser function.
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
        # this is done mostly for performance; see PR#5678 for details
        func = cls.preprocess_func()
        for idx, (start, end) in enumerate(splits):
            partition_kwargs.update({"start": start, "end": end})
            *partition_ids[idx], index_ids[idx], dtypes_ids[idx] = cls.deploy(
                func=func,
                f_args=partition_args,
                f_kwargs=partition_kwargs,
                num_returns=partition_kwargs.get("num_splits") + 2,
            )
        return partition_ids, index_ids, dtypes_ids

    @classmethod
    def check_parameters_support(
        cls,
        filepath_or_buffer,
        read_kwargs: dict,
        skiprows_md: Union[Sequence, callable, int],
        header_size: int,
    ) -> Tuple[bool, Optional[str]]:
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
        Optional[str]
            `None` if parameters are supported, otherwise an error
            message describing why parameters are not supported.
        """
        skiprows = read_kwargs.get("skiprows")
        if isinstance(filepath_or_buffer, str):
            if not cls.file_exists(
                filepath_or_buffer, read_kwargs.get("storage_options")
            ):
                return (False, cls._file_not_found_msg(filepath_or_buffer))
        elif not cls.pathlib_or_pypath(filepath_or_buffer):
            return (False, cls.BUFFER_UNSUPPORTED_MSG)

        if read_kwargs["chunksize"] is not None:
            return (False, "`chunksize` parameter is not supported")

        if read_kwargs.get("iterator"):
            return (False, "`iterator==True` parameter is not supported")

        if read_kwargs.get("dialect") is not None:
            return (False, "`dialect` parameter is not supported")

        if read_kwargs["lineterminator"] is not None:
            return (False, "`lineterminator` parameter is not supported")

        if read_kwargs["escapechar"] is not None:
            return (False, "`escapechar` parameter is not supported")

        if read_kwargs.get("skipfooter"):
            if read_kwargs.get("nrows") or read_kwargs.get("engine") == "c":
                return (False, "Exception is raised by pandas itself")

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
            return (
                False,
                "Values of `header` and `skiprows` parameters have intersections; "
                + "this case is unsupported by Modin",
            )

        return (True, None)

    @classmethod
    @_inherit_docstrings(pandas.io.parsers.base_parser.ParserBase._validate_usecols_arg)
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
        elif is_list_like(skiprows) and len(skiprows) > 0:
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
        partition_ids = cls.build_partition(
            partition_ids, [None] * len(index_ids), column_widths
        )

        new_frame = cls.frame_cls(
            partition_ids,
            lambda: cls._define_index(index_ids, index_name),
            column_names,
            None,
            column_widths,
            dtypes=lambda: cls.get_dtypes(dtypes_ids, column_names),
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
                new_query_compiler = new_query_compiler.take_2d_positional(
                    index=index_range.delete(skiprows_md)
                )
            elif callable(skiprows_md):
                skip_mask = cls._get_skip_mask(index_range, skiprows_md)
                if not isinstance(skip_mask, np.ndarray):
                    skip_mask = skip_mask.to_numpy("bool")
                view_idx = index_range[~skip_mask]
                new_query_compiler = new_query_compiler.take_2d_positional(
                    index=view_idx
                )
            else:
                raise TypeError(
                    f"Not acceptable type of `skiprows` parameter: {type(skiprows_md)}"
                )

            if not isinstance(new_query_compiler.index, pandas.MultiIndex):
                new_query_compiler = new_query_compiler.reset_index(drop=True)

            if nrows:
                new_query_compiler = new_query_compiler.take_2d_positional(
                    pandas.RangeIndex(len(new_query_compiler.index))[:nrows]
                )
        if index_col is None or index_col is False:
            new_query_compiler._modin_frame.synchronize_labels(axis=0)

        return new_query_compiler

    @classmethod
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
        filepath_or_buffer = stringify_path(filepath_or_buffer)
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

        (use_modin_impl, fallback_reason) = cls.check_parameters_support(
            filepath_or_buffer_md,
            kwargs,
            skiprows_md,
            header_size,
        )
        if not use_modin_impl:
            return cls.single_worker_read(
                filepath_or_buffer,
                kwargs,
                reason=fallback_reason,
            )

        is_quoting = kwargs["quoting"] != QUOTE_NONE
        usecols = kwargs["usecols"]
        use_inferred_column_names = cls._uses_inferred_column_names(
            names, skiprows, kwargs["skipfooter"], usecols
        )

        # Computing metadata simultaneously with skipping rows allows us to not
        # do extra work and improve performance for certain cases, as otherwise,
        # it would require double re-reading of skipped rows in order to retrieve metadata.
        can_compute_metadata_while_skipping_rows = (
            # basic supported case: isinstance(skiprows, int) without any additional params
            isinstance(skiprows, int)
            and (usecols is None or skiprows is None)
            and pre_reading == 0
        )
        get_metadata_kw = dict(kwargs, nrows=1, skipfooter=0, index_col=index_col)
        if get_metadata_kw.get("engine", None) == "pyarrow":
            # pyarrow engine doesn't support `nrows` option;
            # https://github.com/pandas-dev/pandas/issues/38872 can be used to track pyarrow engine features
            get_metadata_kw["engine"] = "c"
        if not can_compute_metadata_while_skipping_rows:
            pd_df_metadata = cls.read_callback(
                filepath_or_buffer_md,
                **get_metadata_kw,
            )
            column_names = pd_df_metadata.columns
            column_widths, num_splits = cls._define_metadata(
                pd_df_metadata, column_names
            )
            get_metadata_kw = None
        else:
            get_metadata_kw = dict(get_metadata_kw, skiprows=None)
            # `memory_map` doesn't work with file-like object so we can't use it here.
            # We can definitely skip it without violating the reading logic
            # since this parameter is intended to optimize reading.
            # For reading a couple of lines, this is not essential.
            get_metadata_kw.pop("memory_map", None)
            # These parameters are already used when opening file `f`,
            # they do not need to be used again.
            get_metadata_kw.pop("storage_options", None)
            get_metadata_kw.pop("compression", None)

        with OpenFile(
            filepath_or_buffer_md,
            "rb",
            compression_infered,
            **(kwargs.get("storage_options", None) or {}),
        ) as f:
            old_pos = f.tell()
            fio = io.TextIOWrapper(f, encoding=encoding, newline="")
            newline, quotechar = cls.compute_newline(
                fio, encoding, kwargs.get("quotechar", '"')
            )
            f.seek(old_pos)

            splits, pd_df_metadata_temp = cls.partitioned_file(
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
                get_metadata_kw=get_metadata_kw,
            )
            if can_compute_metadata_while_skipping_rows:
                pd_df_metadata = pd_df_metadata_temp

        # compute dtypes if possible
        common_dtypes = None
        if kwargs["dtype"] is None:
            most_common_dtype = (object,)
            common_dtypes = {}
            for col, dtype in pd_df_metadata.dtypes.to_dict().items():
                if dtype in most_common_dtype:
                    common_dtypes[col] = dtype
        column_names = pd_df_metadata.columns
        column_widths, num_splits = cls._define_metadata(pd_df_metadata, column_names)
        # kwargs that will be passed to the workers
        partition_kwargs = dict(
            kwargs,
            header_size=0 if use_inferred_column_names else header_size,
            names=column_names if use_inferred_column_names else names,
            header="infer" if use_inferred_column_names else header,
            skipfooter=0,
            skiprows=None,
            nrows=None,
            compression=compression_infered,
            common_dtypes=common_dtypes,
        )
        # this is done mostly for performance; see PR#5678 for details
        filepath_or_buffer_md_ref = cls.put(filepath_or_buffer_md)
        kwargs_ref = cls.put(partition_kwargs)
        partition_ids, index_ids, dtypes_ids = cls._launch_tasks(
            splits,
            filepath_or_buffer_md_ref,
            kwargs_ref,
            num_splits=num_splits,
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

    @staticmethod
    def _uses_inferred_column_names(names, skiprows, skipfooter, usecols):
        """
        Tell whether need to use inferred column names in workers or not.

        1) ``False`` is returned in 2 cases and means next:
            1.a) `names` parameter was provided from the API layer. In this case parameter
            `names` must be provided as `names` parameter for ``read_csv`` in the workers.
            1.b) `names` parameter wasn't provided from the API layer. In this case column names
            inference must happen in each partition.
        2) ``True`` is returned in case when inferred column names from pre-reading stage must be
            provided as `names` parameter for ``read_csv`` in the workers.

        In case `names` was provided, the other parameters aren't checked. Otherwise, inferred column
        names should be used in a case of not full data reading which is defined by `skipfooter` parameter,
        when need to skip lines at the bottom of file or by `skiprows` parameter, when need to skip lines at
        the top of file (but if `usecols` was provided, column names inference must happen in the workers).

        Parameters
        ----------
        names : array-like
            List of column names to use.
        skiprows : list-like, int or callable
            Line numbers to skip (0-indexed) or number of lines to skip (int) at
            the start of the file. If callable, the callable function will be
            evaluated against the row indices, returning ``True`` if the row should
            be skipped and ``False`` otherwise.
        skipfooter : int
            Number of lines at bottom of the file to skip.
        usecols : list-like or callable
            Subset of the columns.

        Returns
        -------
        bool
            Whether to use inferred column names in ``read_csv`` of the workers or not.
        """
        if names not in [None, lib.no_default]:
            return False
        if skipfooter != 0:
            return True
        if isinstance(skiprows, int) and skiprows == 0:
            return False
        if is_list_like(skiprows):
            return usecols is None
        return skiprows is not None
