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

"""Module houses `ParquetDispatcher` class, that is used for reading `.parquet` files."""

import os

import fsspec
import fsspec.core
from fsspec.core import split_protocol
import io
from fsspec.registry import get_filesystem_class
import numpy as np
import pandas
from pandas.io.common import is_fsspec_url
from pyarrow.parquet import ParquetFile, ParquetDataset, read_table

from modin.core.storage_formats.pandas.utils import compute_chunksize
from modin.config import NPartitions


from modin.core.io.column_stores.column_store_dispatcher import ColumnStoreDispatcher
from modin.utils import import_optional_dependency


class ParquetDispatcher(ColumnStoreDispatcher):
    """Class handles utils for reading `.parquet` files."""

    @classmethod
    def get_fsspec_files(cls, path, storage_options):
        """
        Retrieve filesystem interface and list of files from path.

        Parameters
        ----------
        path : str, path object or file-like object
            Path.
        storage_options : dict
            Parameters for specific storage engine

        Returns
        -------
        filesystem: Any
            Protocol implementation of registry
        files: list
            List of files from path
        """
        if isinstance(path, io.IOBase):
            return path.fs, [path]
        protocol, path = split_protocol(path)
        filesystem = get_filesystem_class(protocol)(**storage_options)
        if filesystem.stat(path)["type"] == "directory":
            return filesystem, [
                filesystem.unstrip_protocol(path)
                for path in sorted(filesystem.find(path))
            ]
        return filesystem, [filesystem.unstrip_protocol(path)]

    @classmethod
    def call_deploy(cls, fname, col_partitions, storage_options, **kwargs):
        """
        Deploy remote tasks to the workers with passed parameters.

        Parameters
        ----------
        fname : str, path object or file-like object
            Name of the file to read.
        col_partitions : list
            List of arrays with columns names that should be read
            by each partition.
        storage_options: dict
            Paramters for specific storage engine
        **kwargs : dict
            Parameters of deploying read_* function.

        Returns
        -------
        np.ndarray
            Array with references to the task deploy result for each partition.
        """
        from modin.core.storage_formats.pandas.parsers import ParquetFileToRead

        storage_options = storage_options or {}

        filesystem, parquet_files = cls.get_fsspec_files(fname, storage_options)

        row_groups_per_file = []
        num_row_groups = 0
        # Count up the total number of row groups across all files and
        # keep track of row groups per file to use later.
        for file in parquet_files:
            with filesystem.open(file) as f:
                row_groups = ParquetFile(f).num_row_groups
                row_groups_per_file.append(row_groups)
                num_row_groups += row_groups

        # step determines how many row groups are going to be in a partition
        step = compute_chunksize(
            num_row_groups,
            NPartitions.get(),
            min_block_size=1,
        )
        current_partition_size = 0
        file_index = 0
        partition_files = []  # 2D array - each element contains list of chunks to read
        row_groups_used_in_current_file = 0
        total_row_groups_added = 0
        # On each iteration, we add a chunk of one file. That will
        # take us either to the end of a partition, or to the end
        # of a file.
        while total_row_groups_added < num_row_groups:
            if current_partition_size == 0:
                partition_files.append([])
            partition_file = partition_files[-1]
            file_path = parquet_files[file_index]
            row_group_start = row_groups_used_in_current_file
            row_groups_left_in_file = (
                row_groups_per_file[file_index] - row_groups_used_in_current_file
            )
            row_groups_left_for_this_partition = step - current_partition_size
            if row_groups_left_for_this_partition <= row_groups_left_in_file:
                # File has at least what we need to finish partition
                # So finish this partition and start a new one.
                num_row_groups_to_add = row_groups_left_for_this_partition
                current_partition_size = 0
            else:
                # File doesn't have enough to complete this partition. Add
                # it into current partition and go to next file.
                num_row_groups_to_add = row_groups_left_in_file
                current_partition_size += num_row_groups_to_add
            if num_row_groups_to_add == row_groups_left_in_file:
                file_index += 1
                row_groups_used_in_current_file = 0
            else:
                row_groups_used_in_current_file += num_row_groups_to_add
            partition_file.append(
                ParquetFileToRead(
                    file_path, row_group_start, row_group_start + num_row_groups_to_add
                )
            )
            total_row_groups_added += num_row_groups_to_add

        assert (
            total_row_groups_added == num_row_groups
        ), "row groups added does not match total num of row groups across parquet files"

        all_partitions = []
        for files_to_read in partition_files:
            all_partitions.append([])
            for cols in col_partitions:
                all_partitions[-1].append(
                    cls.deploy(
                        cls.parse_fsspec_files,
                        files_for_parser=files_to_read,
                        columns=cols,
                        num_returns=3,
                        storage_options=storage_options,
                        **kwargs,
                    )
                )
        return all_partitions

    @classmethod
    def build_partition(cls, partition_ids, column_widths):
        """
        Build array with partitions of `cls.frame_partition_cls` class.

        Parameters
        ----------
        partition_ids : list
            Array with references to the partitions data.
        column_widths : list
            Number of columns in each partition.

        Returns
        -------
        np.ndarray
            array with shape equals to the shape of `partition_ids` and
            filed with partition objects.
        """
        return np.array(
            [
                [
                    cls.frame_partition_cls(
                        partition_ids[i][j][0],
                        length=partition_ids[i][j][2],
                        width=column_widths[j],
                    )
                    for j in range(len(partition_ids[i]))
                ]
                for i in range(len(partition_ids))
            ]
        )

    @classmethod
    def build_index(cls, path, storage_options):
        """
        Compute index and its split sizes of resulting Modin DataFrame.

        Parameters
        ----------
        path : Pathlike
            Path to dataset

        Returns
        -------
        index : pandas.Index
            Index of resulting Modin DataFrame.
        needs_index_sync : bool
            Whether the partition indices need to be synced with frame
            index because there's no index column, or at least one
            index column is a RangeIndex.
        """

        fs, path_ = (
            fsspec.core.url_to_fs(path, **storage_options)
            if is_fsspec_url(path)
            else (None, path)
        )
        pandas_metadata = ParquetDataset(
            path_, filesystem=fs, use_legacy_dataset=False
        ).schema.pandas_metadata
        index_columns = []
        if pandas_metadata is not None:
            index_columns = pandas_metadata.get("index_columns", index_columns)
        column_names_to_read = []
        for column in index_columns:
            if isinstance(column, str):
                column_names_to_read.append(column)
            elif column["name"] is not None:
                column_names_to_read.append(column["name"])
        complete_index = (
            read_table(path, columns=column_names_to_read).to_pandas().index
        )
        return complete_index, len(index_columns) == 0 or any(
            not isinstance(c, str) for c in index_columns
        )

    @classmethod
    def build_columns(cls, columns):
        """
        Split columns into chunks that should be read be workers.

        Parameters
        ----------
        columns : list
            List of columns that should be read from file.

        Returns
        -------
        col_partitions : list
            List of lists with columns for reading by workers.
        column_widths : list
            List with lengths of `col_partitions` subarrays
            (number of columns that should be read by workers).
        """
        columns_length = len(columns)
        if columns_length == 0:
            return [], []
        num_partitions = NPartitions.get()
        column_splits = (
            columns_length // num_partitions
            if columns_length % num_partitions == 0
            else columns_length // num_partitions + 1
        )
        col_partitions = [
            columns[i : i + column_splits]
            for i in range(0, columns_length, column_splits)
        ]
        column_widths = [len(c) for c in col_partitions]
        return col_partitions, column_widths

    @classmethod
    def build_dtypes(cls, partition_ids, columns):
        """
        Compute common for all partitions `dtypes` for each of the DataFrame column.

        Parameters
        ----------
        partition_ids : list
            Array with references to the partitions data.
        columns : list
            List of columns that should be read from file.

        Returns
        -------
        dtypes : pandas.Series
            Series with dtypes for columns.
        """
        dtypes = pandas.concat(cls.materialize(list(partition_ids)), axis=0)
        dtypes.index = columns
        return dtypes

    @classmethod
    def build_query_compiler(cls, path, columns, **kwargs):
        """
        Build query compiler from deployed tasks outputs.

        Parameters
        ----------
        path : str, path object or file-like object
            Path to the file to read.
        columns : list
            List of columns that should be read from file.
        **kwargs : dict
            Parameters of deploying read_* function.

        Returns
        -------
        new_query_compiler : BaseQueryCompiler
            Query compiler with imported data for further processing.
        """
        col_partitions, column_widths = cls.build_columns(columns)
        partition_ids = cls.call_deploy(path, col_partitions, **kwargs)
        index, needs_index_sync = cls.build_index(path, kwargs["storage_options"])
        remote_parts = cls.build_partition(partition_ids, column_widths)
        if len(partition_ids) > 0 and len(partition_ids[0]) > 0:
            first_row = partition_ids[0]
            dtypes = cls.build_dtypes(
                [dtype_and_partition[1] for dtype_and_partition in first_row], columns
            )
            row_lengths = [part.length() for part in remote_parts.T[0]]
        else:
            dtypes = None
            row_lengths = None
        frame = cls.frame_cls(
            remote_parts,
            index,
            columns,
            row_lengths=row_lengths,
            column_widths=column_widths,
            dtypes=dtypes,
        )
        new_query_compiler = cls.query_compiler_cls(frame)
        if needs_index_sync:
            frame.synchronize_labels(axis=0)
        return new_query_compiler

    @classmethod
    def _read(cls, path, engine, columns, **kwargs):
        """
        Load a parquet object from the file path, returning a query compiler.

        Parameters
        ----------
        path : str, path object or file-like object
            The filepath of the parquet file in local filesystem or hdfs.
        engine : str
            Parquet library to use (only 'PyArrow' is supported for now).
        columns : list
            If not None, only these columns will be read from the file.
        **kwargs : dict
            Keyword arguments.

        Returns
        -------
        BaseQueryCompiler
            A new Query Compiler.

        Notes
        -----
        ParquetFile API is used. Please refer to the documentation here
        https://arrow.apache.org/docs/python/parquet.html
        """
        import_optional_dependency(
            "pyarrow",
            "pyarrow is required to read parquet files.",
        )
        from pyarrow.parquet import ParquetDataset
        from modin.pandas.io import PQ_INDEX_REGEX

        if isinstance(path, str) and os.path.isdir(path):
            partitioned_columns = set()
            # We do a tree walk of the path directory because partitioned
            # parquet directories have a unique column at each directory level.
            # Thus, we can use os.walk(), which does a dfs search, to walk
            # through the different columns that the data is partitioned on
            for (root, dir_names, files) in os.walk(path):
                if dir_names:
                    partitioned_columns.add(dir_names[0].split("=")[0])
                if files:
                    # Metadata files, git files, .DSStore
                    if files[0][0] == ".":
                        continue
                    break
            partitioned_columns = list(partitioned_columns)
            if len(partitioned_columns):
                return cls.single_worker_read(
                    path,
                    engine=engine,
                    columns=columns,
                    reason="Mixed partitioning columns in Parquet",
                    **kwargs
                )

        if not columns:

            fs, path_ = (
                fsspec.core.url_to_fs(path, **(kwargs.get("storage_options") or {}))
                if is_fsspec_url(path)
                else (None, path)
            )

            dataset = ParquetDataset(path_, filesystem=fs, use_legacy_dataset=False)
            column_names = dataset.schema.names

            if dataset.schema.pandas_metadata is not None:
                index_columns = dataset.schema.pandas_metadata.get("index_columns", [])
                column_names = [c for c in column_names if c not in index_columns]
            columns = [name for name in column_names if not PQ_INDEX_REGEX.match(name)]
        return cls.build_query_compiler(path, columns, **kwargs)
