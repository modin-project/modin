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

"""Module houses `CSVGlobDispatcher` class, that is used for reading multiple `.csv` files simultaneously."""

from contextlib import ExitStack
import csv
import glob
import os
import sys
from typing import List, Tuple
import warnings

import pandas
import pandas._libs.lib as lib

from modin.config import NPartitions
from modin.core.io.file_dispatcher import OpenFile
from modin.core.io.file_dispatcher import S3_ADDRESS_REGEX
from modin.core.io.text.csv_dispatcher import CSVDispatcher
from modin.utils import import_optional_dependency


class CSVGlobDispatcher(CSVDispatcher):
    """Class contains utils for reading multiple `.csv` files simultaneously."""

    @classmethod
    def _read(cls, filepath_or_buffer, **kwargs):
        """
        Read data from multiple `.csv` files passed with `filepath_or_buffer` simultaneously.

        Parameters
        ----------
        filepath_or_buffer : str, path object or file-like object
            `filepath_or_buffer` parameter of ``read_csv`` function.
        **kwargs : dict
            Parameters of ``read_csv`` function.

        Returns
        -------
        new_query_compiler : BaseQueryCompiler
            Query compiler with imported data for further processing.
        """
        # Ensures that the file is a string file path. Otherwise, default to pandas.
        filepath_or_buffer = cls.get_path_or_buffer(filepath_or_buffer)
        if isinstance(filepath_or_buffer, str):
            # os.altsep == None on Linux
            is_folder = any(
                filepath_or_buffer.endswith(sep) for sep in (os.sep, os.altsep) if sep
            )
            if "*" not in filepath_or_buffer and not is_folder:
                warnings.warn(
                    "Shell-style wildcard '*' must be in the filename pattern in order to read multiple "
                    + f"files at once. Did you forget it? Passed filename: '{filepath_or_buffer}'"
                )
            if not cls.file_exists(filepath_or_buffer):
                return cls.single_worker_read(filepath_or_buffer, **kwargs)
            filepath_or_buffer = cls.get_path(filepath_or_buffer)
        elif not cls.pathlib_or_pypath(filepath_or_buffer):
            return cls.single_worker_read(filepath_or_buffer, **kwargs)

        # We read multiple csv files when the file path is a list of absolute file paths. We assume that all of the files will be essentially replicas of the
        # first file but with different data values.
        glob_filepaths = filepath_or_buffer
        filepath_or_buffer = filepath_or_buffer[0]

        compression_type = cls.infer_compression(
            filepath_or_buffer, kwargs.get("compression")
        )
        if compression_type is not None:
            if (
                compression_type == "gzip"
                or compression_type == "bz2"
                or compression_type == "xz"
            ):
                kwargs["compression"] = compression_type
            elif (
                compression_type == "zip"
                and sys.version_info[0] == 3
                and sys.version_info[1] >= 7
            ):
                # need python3.7 to .seek and .tell ZipExtFile
                kwargs["compression"] = compression_type
            else:
                return cls.single_worker_read(filepath_or_buffer, **kwargs)

        chunksize = kwargs.get("chunksize")
        if chunksize is not None:
            return cls.single_worker_read(filepath_or_buffer, **kwargs)

        skiprows = kwargs.get("skiprows")
        if skiprows is not None and not isinstance(skiprows, int):
            return cls.single_worker_read(filepath_or_buffer, **kwargs)

        nrows = kwargs.pop("nrows", None)
        names = kwargs.get("names", lib.no_default)
        index_col = kwargs.get("index_col", None)
        usecols = kwargs.get("usecols", None)
        encoding = kwargs.get("encoding", None)
        if names in [lib.no_default, None]:
            # For the sake of the empty df, we assume no `index_col` to get the correct
            # column names before we build the index. Because we pass `names` in, this
            # step has to happen without removing the `index_col` otherwise it will not
            # be assigned correctly.
            names = pandas.read_csv(
                filepath_or_buffer,
                **dict(kwargs, usecols=None, nrows=0, skipfooter=0, index_col=None),
            ).columns
        elif index_col is None and not usecols:
            # When names is set to some list that is smaller than the number of columns
            # in the file, the first columns are built as a hierarchical index.
            empty_pd_df = pandas.read_csv(
                filepath_or_buffer, nrows=0, encoding=encoding
            )
            num_cols = len(empty_pd_df.columns)
            if num_cols > len(names):
                index_col = list(range(num_cols - len(names)))
                if len(index_col) == 1:
                    index_col = index_col[0]
                kwargs["index_col"] = index_col
        empty_pd_df = pandas.read_csv(
            filepath_or_buffer, **dict(kwargs, nrows=0, skipfooter=0)
        )
        column_names = empty_pd_df.columns
        skipfooter = kwargs.get("skipfooter", None)
        skiprows = kwargs.pop("skiprows", None)
        usecols_md = cls._validate_usecols_arg(usecols)
        if usecols is not None and usecols_md[1] != "integer":
            del kwargs["usecols"]
            all_cols = pandas.read_csv(
                filepath_or_buffer,
                **dict(kwargs, nrows=0, skipfooter=0),
            ).columns
            usecols = all_cols.get_indexer_for(list(usecols_md[0]))
        parse_dates = kwargs.pop("parse_dates", False)
        partition_kwargs = dict(
            kwargs,
            header=None,
            names=names,
            skipfooter=0,
            skiprows=None,
            parse_dates=parse_dates,
            usecols=usecols,
        )
        encoding = kwargs.get("encoding", None)
        quotechar = kwargs.get("quotechar", '"').encode(
            encoding if encoding is not None else "UTF-8"
        )
        is_quoting = kwargs.get("quoting", "") != csv.QUOTE_NONE

        with ExitStack() as stack:
            files = [
                stack.enter_context(OpenFile(fname, "rb", compression_type))
                for fname in glob_filepaths
            ]

            # Skip the header since we already have the header information and skip the
            # rows we are told to skip.
            if isinstance(skiprows, int) or skiprows is None:
                if skiprows is None:
                    skiprows = 0
                header = kwargs.get("header", "infer")
                if header == "infer" and kwargs.get("names", lib.no_default) in [
                    lib.no_default,
                    None,
                ]:
                    skip_header = 1
                elif isinstance(header, int):
                    skip_header = header + 1
                elif hasattr(header, "__iter__") and not isinstance(header, str):
                    skip_header = max(header) + 1
                else:
                    skip_header = 0
            if kwargs.get("encoding", None) is not None:
                partition_kwargs["skiprows"] = 1
            # Launch tasks to read partitions
            column_widths, num_splits = cls._define_metadata(empty_pd_df, column_names)

            args = {
                "num_splits": num_splits,
                **partition_kwargs,
            }

            splits = cls.partitioned_file(
                files,
                glob_filepaths,
                num_partitions=NPartitions.get(),
                nrows=nrows,
                skiprows=skiprows,
                skip_header=skip_header,
                quotechar=quotechar,
                is_quoting=is_quoting,
            )
            partition_ids = [None] * len(splits)
            index_ids = [None] * len(splits)
            dtypes_ids = [None] * len(splits)
            for idx, chunks in enumerate(splits):
                args.update({"chunks": chunks})
                *partition_ids[idx], index_ids[idx], dtypes_ids[idx] = cls.deploy(
                    cls.parse, num_returns=num_splits + 2, **args
                )

        # Compute the index based on a sum of the lengths of each partition (by default)
        # or based on the column(s) that were requested.
        if index_col is None:
            row_lengths = cls.materialize(index_ids)
            new_index = pandas.RangeIndex(sum(row_lengths))
        else:
            index_objs = cls.materialize(index_ids)
            row_lengths = [len(o) for o in index_objs]
            new_index = index_objs[0].append(index_objs[1:])
            new_index.name = empty_pd_df.index.name

        # Compute dtypes by getting collecting and combining all of the partitions. The
        # reported dtypes from differing rows can be different based on the inference in
        # the limited data seen by each worker. We use pandas to compute the exact dtype
        # over the whole column for each column. The index is set below.
        dtypes = cls.get_dtypes(dtypes_ids) if len(dtypes_ids) > 0 else None

        partition_ids = cls.build_partition(partition_ids, row_lengths, column_widths)
        # If parse_dates is present, the column names that we have might not be
        # the same length as the returned column names. If we do need to modify
        # the column names, we remove the old names from the column names and
        # insert the new one at the front of the Index.
        if parse_dates is not None:
            # We have to recompute the column widths if `parse_dates` is set because
            # we are not guaranteed to have the correct information regarding how many
            # columns are on each partition.
            column_widths = None
            if isinstance(parse_dates, list):
                for date in parse_dates:
                    # Lists within the parse_dates list are sequences of
                    # CSV columns that are parsed together as a single date
                    # column. They can be a list of either string column names
                    # or integer column indices. e.g. if parse_dates is
                    # [[1, 2]] and columns at indices 1 and 2 are "b" and "c",
                    # the output dataframe has the single date column "b_c". If
                    # parse_dates is [["a", 1]] and the column at index 1 is
                    # named "b", the output dataframe has the single date
                    # column "a_b".
                    if isinstance(date, list):
                        for i, part in enumerate(date):
                            if isinstance(part, int):
                                date[i] = column_names[part]
                        new_col_name = "_".join(date)
                        column_names = column_names.drop(date).insert(0, new_col_name)
            elif isinstance(parse_dates, dict):
                for new_col_name, group in parse_dates.items():
                    column_names = column_names.drop(group).insert(0, new_col_name)
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

        if skipfooter:
            new_query_compiler = new_query_compiler.drop(
                new_query_compiler.index[-skipfooter:]
            )
        if kwargs.get("squeeze", False) and len(new_query_compiler.columns) == 1:
            return new_query_compiler[new_query_compiler.columns[0]]
        if index_col is None:
            new_query_compiler._modin_frame.synchronize_labels(axis=0)
        return new_query_compiler

    @classmethod
    def file_exists(cls, file_path: str) -> bool:
        """
        Check if the `file_path` is valid.

        Parameters
        ----------
        file_path : str
            String representing a path.

        Returns
        -------
        bool
            True if the path is valid.
        """
        if isinstance(file_path, str):
            match = S3_ADDRESS_REGEX.search(file_path)
            if match is not None:
                if file_path[0] == "S":
                    file_path = "{}{}".format("s", file_path[1:])
                S3FS = import_optional_dependency(
                    "s3fs", "Module s3fs is required to read S3FS files."
                )
                from botocore.exceptions import NoCredentialsError

                s3fs = S3FS.S3FileSystem(anon=False)
                exists = False
                try:
                    exists = len(s3fs.glob(file_path)) > 0 or exists
                except NoCredentialsError:
                    pass
                s3fs = S3FS.S3FileSystem(anon=True)
                return exists or len(s3fs.glob(file_path)) > 0
        return len(glob.glob(file_path)) > 0

    @classmethod
    def get_path(cls, file_path: str) -> list:
        """
        Return the path of the file(s).

        Parameters
        ----------
        file_path : str
            String representing a path.

        Returns
        -------
        list
            List of strings of absolute file paths.
        """
        if S3_ADDRESS_REGEX.search(file_path):
            # S3FS does not allow captial S in s3 addresses.
            if file_path[0] == "S":
                file_path = "{}{}".format("s", file_path[1:])

            S3FS = import_optional_dependency(
                "s3fs", "Module s3fs is required to read S3FS files."
            )
            from botocore.exceptions import NoCredentialsError

            def get_file_path(fs_handle) -> List[str]:
                file_paths = fs_handle.glob(file_path)
                s3_addresses = ["{}{}".format("s3://", path) for path in file_paths]
                return s3_addresses

            s3fs = S3FS.S3FileSystem(anon=False)
            try:
                return get_file_path(s3fs)
            except NoCredentialsError:
                pass
            s3fs = S3FS.S3FileSystem(anon=True)
            return get_file_path(s3fs)
        else:
            relative_paths = glob.glob(file_path)
            abs_paths = [os.path.abspath(path) for path in relative_paths]
            return abs_paths

    @classmethod
    def partitioned_file(
        cls,
        files,
        fnames: List[str],
        num_partitions: int = None,
        nrows: int = None,
        skiprows: int = None,
        skip_header: int = None,
        quotechar: bytes = b'"',
        is_quoting: bool = True,
    ) -> List[List[Tuple[str, int, int]]]:
        """
        Compute chunk sizes in bytes for every partition.

        Parameters
        ----------
        files : file or list of files
            File(s) to be partitioned.
        fnames : str or list of str
            File name(s) to be partitioned.
        num_partitions : int, optional
            For what number of partitions split a file.
            If not specified grabs the value from `modin.config.NPartitions.get()`.
        nrows : int, optional
            Number of rows of file to read.
        skiprows : int, optional
            Specifies rows to skip.
        skip_header : int, optional
            Specifies header rows to skip.
        quotechar : bytes, default: b'"'
            Indicate quote in a file.
        is_quoting : bool, default: True
            Whether or not to consider quotes.

        Returns
        -------
        list
            List, where each element of the list is a list of tuples. The inner lists
            of tuples contains the data file name of the chunk, chunk start offset, and
            chunk end offsets for its corresponding file.

        Notes
        -----
        The logic gets really complicated if we try to use the `TextFileDispatcher.partitioned_file`.
        """
        if type(files) != list:
            files = [files]

        if num_partitions is None:
            num_partitions = NPartitions.get()

        file_sizes = [cls.file_size(f) for f in files]
        partition_size = max(
            1, num_partitions, (nrows if nrows else sum(file_sizes)) // num_partitions
        )

        result = []
        split_result = []
        split_size = 0
        read_rows_counter = 0
        for f, fname, f_size in zip(files, fnames, file_sizes):
            if skiprows or skip_header:
                skip_amount = (skiprows if skiprows else 0) + (
                    skip_header if skip_header else 0
                )

                # TODO(williamma12): Handle when skiprows > number of rows in file. Currently returns empty df.
                outside_quotes, read_rows = cls._read_rows(
                    f,
                    nrows=skip_amount,
                    quotechar=quotechar,
                    is_quoting=is_quoting,
                )
                if skiprows:
                    skiprows -= read_rows
                    if skiprows > 0:
                        # We have more rows to skip than the amount read in the file.
                        continue

            start = f.tell()

            while f.tell() < f_size:
                if split_size >= partition_size:
                    # Create a new split when the split has reached partition_size.
                    # This is mainly used when we are reading row-wise partitioned files.
                    result.append(split_result)
                    split_result = []
                    split_size = 0

                # We calculate the amount that we need to read based off of how much of the split we have already read.
                read_size = partition_size - split_size

                if nrows:
                    if read_rows_counter >= nrows:
                        # # Finish when we have read enough rows.
                        if len(split_result) > 0:
                            # Add last split into the result.
                            result.append(split_result)
                        return result
                    elif read_rows_counter + read_size > nrows:
                        # Ensure that we will not read more than nrows.
                        read_size = nrows - read_rows_counter

                    outside_quotes, read_rows = cls._read_rows(
                        f,
                        nrows=read_size,
                        quotechar=quotechar,
                        is_quoting=is_quoting,
                    )
                    split_size += read_rows
                    read_rows_counter += read_rows
                else:
                    outside_quotes = cls.offset(
                        f,
                        offset_size=read_size,
                        quotechar=quotechar,
                        is_quoting=is_quoting,
                    )

                split_result.append((fname, start, f.tell()))
                split_size += f.tell() - start
                start = f.tell()

                # Add outside_quotes.
                if is_quoting and not outside_quotes:
                    warnings.warn("File has mismatched quotes")

        # Add last split into the result.
        if len(split_result) > 0:
            result.append(split_result)

        return result
