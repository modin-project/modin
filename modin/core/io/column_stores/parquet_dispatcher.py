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
from fsspec.core import split_protocol, url_to_fs
from fsspec.registry import get_filesystem_class
from fsspec.spec import AbstractBufferedFile
import numpy as np
from pandas.io.common import is_fsspec_url
from packaging import version

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
            Path to dataset.
        storage_options : dict
            Parameters for specific storage engine.

        Returns
        -------
        filesystem: Any
            Protocol implementation of registry.
        files: list
            List of files from path.
        """
        # Older versions of fsspec doesn't support unstrip_protocol(). It
        # was only added relatively recently:
        # https://github.com/fsspec/filesystem_spec/pull/828
        def _unstrip_protocol(protocol, path):
            protos = (protocol,) if isinstance(protocol, str) else protocol
            for protocol in protos:
                if path.startswith(f"{protocol}://"):
                    return path
            return f"{protos[0]}://{path}"

        if isinstance(path, AbstractBufferedFile):
            return path.fs, [path]
        protocol, path = split_protocol(path)
        filesystem = get_filesystem_class(protocol)(**storage_options)
        if filesystem.stat(path)["type"] == "directory":
            files = []
            for path in sorted(filesystem.find(path)):
                if version.parse(fsspec.__version__) < version.parse("2022.5.0"):
                    files.append(_unstrip_protocol(filesystem.protocol, path))
                else:
                    files.append(filesystem.unstrip_protocol(path))
            return filesystem, files

        return filesystem, [
            _unstrip_protocol(filesystem.protocol, path)
            if version.parse(fsspec.__version__) < version.parse("2022.5.0")
            else filesystem.unstrip_protocol(path)
        ]

    @classmethod
    def _get_fs_and_fs_path(cls, path, storage_options):
        """
        Retrieve filesystem interface and filesystem-specific path.

        Parameters
        ----------
        path : str, path object or file-like object
            Path to dataset.
        storage_options : dict
            Parameters for specific storage engine.

        Returns
        -------
        filesystem : Any
            Protocol implementation of registry.
        fs_path : list
            Filesystem's specific path.
        """
        return (
            url_to_fs(path, **storage_options) if is_fsspec_url(path) else (None, path)
        )

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
        storage_options : dict
            Parameters for specific storage engine.
        **kwargs : dict
            Parameters of deploying read_* function.

        Returns
        -------
        List
            Array with references to the task deploy result for each partition.
        """
        from pyarrow.parquet import ParquetFile
        from modin.core.storage_formats.pandas.parsers import ParquetFileToRead

        # If we don't have any columns to read, we should just return an empty
        # set of references.
        if len(col_partitions) == 0:
            return []

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
            all_partitions.append(
                [
                    cls.deploy(
                        cls.parse,
                        files_for_parser=files_to_read,
                        columns=cols,
                        num_returns=3,
                        storage_options=storage_options,
                        **kwargs,
                    )
                    for cols in col_partitions
                ]
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

        Notes
        -----
        The second level of partitions_ids contains a list of object references
        for each read call:
        partition_ids[i][j] -> [ObjectRef(df), ObjectRef(df.index), ObjectRef(len(df))].
        """
        return np.array(
            [
                [
                    cls.frame_partition_cls(
                        part_id[0],
                        length=part_id[2],
                        width=col_width,
                    )
                    for part_id, col_width in zip(part_ids, column_widths)
                ]
                for part_ids in partition_ids
            ]
        )

    @classmethod
    def build_index(cls, path, partition_ids, index_columns, storage_options):
        """
        Compute index and its split sizes of resulting Modin DataFrame.

        Parameters
        ----------
        path : Pathlike
            Path to dataset.
        partition_ids : list
            Array with references to the partitions data.
        index_columns : list
            List of index columns specified by pandas metadata.
        storage_options : dict
            Parameters for specific storage engine.

        Returns
        -------
        index : pandas.Index
            Index of resulting Modin DataFrame.
        needs_index_sync : bool
            Whether the partition indices need to be synced with frame
            index because there's no index column, or at least one
            index column is a RangeIndex.

        Notes
        -----
        See `build_partition` for more detail on the contents of partitions_ids.
        """
        from pyarrow.parquet import read_table

        range_index = True
        column_names_to_read = []
        for column in index_columns:
            # According to https://arrow.apache.org/docs/python/generated/pyarrow.Schema.html,
            # only RangeIndex will be stored as metadata. Otherwise, the default behavior is
            # to store the index as a column.
            if isinstance(column, str):
                column_names_to_read.append(column)
                range_index = False
            elif column["name"] is not None:
                column_names_to_read.append(column["name"])

        # For the second check, let us consider the case where we have an empty dataframe,
        # that has a valid index.
        if range_index or (len(partition_ids) == 0 and len(column_names_to_read) != 0):
            fs, fs_path = cls._get_fs_and_fs_path(path, storage_options)
            complete_index = (
                read_table(fs_path, columns=column_names_to_read, filesystem=fs)
                .to_pandas()
                .index
            )
        # Empty DataFrame case
        elif len(partition_ids) == 0:
            return [], False
        else:
            index_ids = [part_id[0][1] for part_id in partition_ids if len(part_id) > 0]
            index_objs = cls.materialize(index_ids)
            complete_index = index_objs[0].append(index_objs[1:])
        return complete_index, range_index or (len(index_columns) == 0)

    @classmethod
    def build_query_compiler(cls, path, columns, index_columns, **kwargs):
        """
        Build query compiler from deployed tasks outputs.

        Parameters
        ----------
        path : str, path object or file-like object
            Path to the file to read.
        columns : list
            List of columns that should be read from file.
        index_columns : list
            List of index columns specified by pandas metadata.
        **kwargs : dict
            Parameters of deploying read_* function.

        Returns
        -------
        new_query_compiler : BaseQueryCompiler
            Query compiler with imported data for further processing.
        """
        storage_options = kwargs.pop("storage_options", {}) or {}
        col_partitions, column_widths = cls.build_columns(columns)
        partition_ids = cls.call_deploy(path, col_partitions, storage_options, **kwargs)
        index, sync_index = cls.build_index(
            path, partition_ids, index_columns, storage_options
        )
        remote_parts = cls.build_partition(partition_ids, column_widths)
        if len(partition_ids) > 0:
            row_lengths = [part.length() for part in remote_parts.T[0]]
        else:
            row_lengths = None
        frame = cls.frame_cls(
            remote_parts,
            index,
            columns,
            row_lengths=row_lengths,
            column_widths=column_widths,
            dtypes=None,
        )
        if sync_index:
            frame.synchronize_labels(axis=0)
        return cls.query_compiler_cls(frame)

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
            for (_, dir_names, files) in os.walk(path):
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
                    **kwargs,
                )
        # url_to_fs returns the fs and path formatted for the specific fs
        fs, fs_path = cls._get_fs_and_fs_path(path, kwargs.get("storage_options") or {})
        dataset = ParquetDataset(fs_path, filesystem=fs, use_legacy_dataset=False)
        index_columns = (
            dataset.schema.pandas_metadata.get("index_columns", [])
            if dataset.schema.pandas_metadata
            else []
        )
        # If we have columns as None, then we default to reading in all the columns
        column_names = dataset.schema.names if not columns else columns
        columns = [
            c
            for c in column_names
            if c not in index_columns and not PQ_INDEX_REGEX.match(c)
        ]

        return cls.build_query_compiler(path, columns, index_columns, **kwargs)
