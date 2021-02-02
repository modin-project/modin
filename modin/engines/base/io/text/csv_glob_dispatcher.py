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

from contextlib import ExitStack
import csv
import glob
import os
import sys
from typing import List

import pandas
from pandas.io.parsers import _validate_usecols_arg

from modin.config import NPartitions
from modin.data_management.utils import compute_chunksize
from modin.engines.base.io.file_dispatcher import S3_ADDRESS_REGEX
from modin.engines.base.io.text.csv_dispatcher import CSVDispatcher


class CSVGlobDispatcher(CSVDispatcher):
    @classmethod
    def _read(cls, filepath_or_buffer, **kwargs):
        # Ensures that the file is a string file path. Otherwise, default to pandas.
        filepath_or_buffer = cls.get_path_or_buffer(filepath_or_buffer)
        if isinstance(filepath_or_buffer, str):
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
        names = kwargs.get("names", None)
        index_col = kwargs.get("index_col", None)
        usecols = kwargs.get("usecols", None)
        encoding = kwargs.get("encoding", None)
        if names is None:
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
        usecols_md = _validate_usecols_arg(usecols)
        if usecols is not None and usecols_md[1] != "integer":
            del kwargs["usecols"]
            all_cols = pandas.read_csv(
                cls.file_open(filepath_or_buffer, "rb"),
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
                stack.enter_context(cls.file_open(fname, "rb", compression_type))
                for fname in glob_filepaths
            ]

            # Skip the header since we already have the header information and skip the
            # rows we are told to skip.
            if isinstance(skiprows, int) or skiprows is None:
                if skiprows is None:
                    skiprows = 0
                header = kwargs.get("header", "infer")
                if header == "infer" and kwargs.get("names", None) is None:
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
            partition_ids = []
            index_ids = []
            dtypes_ids = []
            # Max number of partitions available
            num_partitions = NPartitions.get()
            # This is the number of splits for the columns
            num_splits = min(len(column_names), num_partitions)
            # Metadata
            column_chunksize = compute_chunksize(empty_pd_df, num_splits, axis=1)
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

            args = {
                "num_splits": num_splits,
                **partition_kwargs,
            }

            splits = cls.partitioned_multiple_files(
                files,
                glob_filepaths,
                num_partitions=num_partitions,
                nrows=nrows,
                skiprows=skiprows,
                skip_header=skip_header,
                quotechar=quotechar,
                is_quoting=is_quoting,
            )

            for chunks in splits:
                args.update({"chunks": chunks})
                partition_id = cls.deploy(cls.parse, num_splits + 2, args)
                partition_ids.append(partition_id[:-2])
                index_ids.append(partition_id[-2])
                dtypes_ids.append(partition_id[-1])

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
            # Check if is list of lists
            if isinstance(parse_dates, list) and isinstance(parse_dates[0], list):
                for group in parse_dates:
                    new_col_name = "_".join(group)
                    column_names = column_names.drop(group).insert(0, new_col_name)
            # Check if it is a dictionary
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
            new_query_compiler._modin_frame._apply_index_objs(axis=0)
        return new_query_compiler

    @classmethod
    def file_exists(cls, file_path: str) -> bool:
        """
        Checks if the file_path is valid.

        Parameters
        ----------
        file_path: str
            String representing a path.

        Returns
        -------
        bool
            True if the glob path is valid.
        """
        if isinstance(file_path, str):
            match = S3_ADDRESS_REGEX.search(file_path)
            if match is not None:
                if file_path[0] == "S":
                    file_path = "{}{}".format("s", file_path[1:])
                import s3fs as S3FS
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
        Returns the path of the file(s).

        Parameters
        ----------
        file_path: str
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

            import s3fs as S3FS
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
    def partitioned_multiple_files(
        cls,
        files,
        fnames: List[str],
        num_partitions: int = None,
        nrows: int = None,
        skiprows: int = None,
        skip_header: int = None,
        quotechar: bytes = b'"',
        is_quoting: bool = True,
    ):
        """
        Compute chunk sizes in bytes for every partition.

        Parameters
        ----------
        files: file
            Files to be partitioned.
        fnames: str
            File names to be partitioned.
        num_partitions: int, optional
            For what number of partitions split a file.
            If not specified grabs the value from `modin.config.NPartitions.get()`.
        nrows: int, optional
            Number of rows of file to read.
        skiprows: int, optional
            Specifies rows to skip.
        skip_header: int, optional
            Specifies header rows to skip.
        quotechar: bytes, default b'"'
            Indicate quote in a file.
        is_quoting: bool, default True
            Whether or not to consider quotes.

        Returns
        -------
        list
            List of list consisting of tuples, which contain data of which files make up each corresponding partition. The tuples consisting of the file name, start offset, and end offset of each partition split.
        """
        if num_partitions is None:
            num_partitions = NPartitions.get()

        file_sizes = [cls.file_size(f) for f in files]
        partition_size = max(
            1, num_partitions, (nrows if nrows else sum(file_sizes)) // num_partitions
        )

        final_result = []
        partial_partition = []
        partial_partition_size = 0
        for f, fname, fsize in zip(files, fnames, file_sizes):
            # We skip the headers of every file before trying to read from them.
            if skip_header:
                cls._read_rows(
                    f,
                    nrows=skip_header,
                    quotechar=quotechar,
                    is_quoting=is_quoting,
                )

            # Fill up the rest of the partition before partitioning the rest of the file.
            if partial_partition_size > 0:
                start = f.tell()
                if nrows:
                    remainder_size = min(partition_size, nrows) - partial_partition_size
                    _, read_size = cls._read_rows(
                        f,
                        nrows=remainder_size,
                        quotechar=quotechar,
                        is_quoting=is_quoting,
                    )
                    end = f.tell()
                else:
                    remainder_size = partition_size - partial_partition_size
                    cls.offset(
                        f,
                        offset_size=remainder_size,
                        quotechar=quotechar,
                        is_quoting=is_quoting,
                    )
                    end = f.tell()
                    read_size = end - start

                partial_partition.append((fname, start, end))
                partial_partition_size += read_size
                if read_size < remainder_size:
                    # The file that we were reading was too small to fill the carried over partiton (partial_partition).
                    continue
                else:
                    if nrows:
                        nrows -= partial_partition_size
                    final_result.append(partial_partition)
                    partial_partition = []
                    partial_partition_size = 0

            if nrows == 0:
                # We stop reading here having completed, if necessary, the partial partition.
                break

            if f.tell() == fsize:
                # Don't bother reading an empty file.
                continue

            file_splits, rows_read = cls.partitioned_file(
                f,
                fname,
                partition_size=partition_size,
                nrows=nrows,
                skiprows=skiprows,
                quotechar=quotechar,
                is_quoting=is_quoting,
            )

            if skiprows:
                # Update bookkeeping on skipped rows.
                if skiprows > rows_read:
                    # Wanted to skip more rows than what was available in the file.
                    skiprows -= rows_read
                    continue
                else:
                    rows_read -= skiprows

            # Calculate if the last split needs to be carried over to the next file.
            if nrows:
                last_size = rows_read % partition_size
                full_last_partition = last_size == 0
                nrows -= rows_read
            else:
                _, last_start, last_end = file_splits[-1]
                last_size = last_end - last_start
                full_last_partition = last_size >= partition_size

            if full_last_partition:
                final_result.append(file_splits)
            else:
                if len(file_splits) > 1:
                    # Don't append anything if the file was too small for one partition.
                    final_result.append(file_splits[:-1])
                partial_partition = [file_splits[-1]]
                partial_partition_size = last_size
                if nrows:
                    # We add the carried over partition because we need it to calculate the how much to read for the partial partition.
                    nrows += partial_partition_size

        # Add straggler splits into the final result.
        if partial_partition_size > 0:
            final_result.append(partial_partition)

        return final_result
