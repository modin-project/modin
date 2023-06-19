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
import re
import json

import fsspec
from fsspec.core import url_to_fs
from fsspec.spec import AbstractBufferedFile
import numpy as np
from pandas.io.common import stringify_path
import pandas
import pandas._libs.lib as lib
from packaging import version

from modin.core.storage_formats.pandas.utils import compute_chunksize
from modin.config import NPartitions


from modin.core.io.column_stores.column_store_dispatcher import ColumnStoreDispatcher
from modin.utils import _inherit_docstrings


class ColumnStoreDataset:
    """
    Base class that encapsulates Parquet engine-specific details.

    This class exposes a set of functions that are commonly used in the
    `read_parquet` implementation.

    Attributes
    ----------
    path : str, path object or file-like object
        The filepath of the parquet file in local filesystem or hdfs.
    storage_options : dict
        Parameters for specific storage engine.
    _fs_path : str, path object or file-like object
        The filepath or handle of the parquet dataset specific to the
        filesystem implementation. E.g. for `s3://test/example`, _fs
        would be set to S3FileSystem and _fs_path would be `test/example`.
    _fs : Filesystem
        Filesystem object specific to the given parquet file/dataset.
    dataset : ParquetDataset or ParquetFile
        Underlying dataset implementation for PyArrow and fastparquet
        respectively.
    _row_groups_per_file : list
        List that contains the number of row groups for each file in the
        given parquet dataset.
    _files : list
        List that contains the full paths of the parquet files in the dataset.
    """

    def __init__(self, path, storage_options):  # noqa : PR01
        self.path = path.__fspath__() if isinstance(path, os.PathLike) else path
        self.storage_options = storage_options
        self._fs_path = None
        self._fs = None
        self.dataset = self._init_dataset()
        self._row_groups_per_file = None
        self._files = None

    @property
    def pandas_metadata(self):
        """Return the pandas metadata of the dataset."""
        raise NotImplementedError

    @property
    def columns(self):
        """Return the list of columns in the dataset."""
        raise NotImplementedError

    @property
    def engine(self):
        """Return string representing what engine is being used."""
        raise NotImplementedError

    # TODO: make this cache_readonly after docstring inheritance is fixed.
    @property
    def files(self):
        """Return the list of formatted file paths of the dataset."""
        raise NotImplementedError

    # TODO: make this cache_readonly after docstring inheritance is fixed.
    @property
    def row_groups_per_file(self):
        """Return a list with the number of row groups per file."""
        raise NotImplementedError

    @property
    def fs(self):
        """
        Return the filesystem object associated with the dataset path.

        Returns
        -------
        filesystem
            Filesystem object.
        """
        if self._fs is None:
            if isinstance(self.path, AbstractBufferedFile):
                self._fs = self.path.fs
            else:
                self._fs, self._fs_path = url_to_fs(self.path, **self.storage_options)
        return self._fs

    @property
    def fs_path(self):
        """
        Return the filesystem-specific path or file handle.

        Returns
        -------
        fs_path : str, path object or file-like object
            String path specific to filesystem or a file handle.
        """
        if self._fs_path is None:
            if isinstance(self.path, AbstractBufferedFile):
                self._fs_path = self.path
            else:
                self._fs, self._fs_path = url_to_fs(self.path, **self.storage_options)
        return self._fs_path

    def to_pandas_dataframe(self, columns):
        """
        Read the given columns as a pandas dataframe.

        Parameters
        ----------
        columns : list
            List of columns that should be read from file.
        """
        raise NotImplementedError

    def _get_files(self, files):
        """
        Retrieve list of formatted file names in dataset path.

        Parameters
        ----------
        files : list
            List of files from path.

        Returns
        -------
        fs_files : list
            List of files from path with fs-protocol prepended.
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

        if isinstance(self.path, AbstractBufferedFile):
            return [self.path]
        # version.parse() is expensive, so we can split this into two separate loops
        if version.parse(fsspec.__version__) < version.parse("2022.5.0"):
            fs_files = [_unstrip_protocol(self.fs.protocol, fpath) for fpath in files]
        else:
            fs_files = [self.fs.unstrip_protocol(fpath) for fpath in files]

        return fs_files


@_inherit_docstrings(ColumnStoreDataset)
class PyArrowDataset(ColumnStoreDataset):
    def _init_dataset(self):  # noqa: GL08
        from pyarrow.parquet import ParquetDataset

        return ParquetDataset(
            self.fs_path, filesystem=self.fs, use_legacy_dataset=False
        )

    @property
    def pandas_metadata(self):
        return self.dataset.schema.pandas_metadata

    @property
    def columns(self):
        return self.dataset.schema.names

    @property
    def engine(self):
        return "pyarrow"

    @property
    def row_groups_per_file(self):
        from pyarrow.parquet import ParquetFile

        if self._row_groups_per_file is None:
            row_groups_per_file = []
            # Count up the total number of row groups across all files and
            # keep track of row groups per file to use later.
            for file in self.files:
                with self.fs.open(file) as f:
                    row_groups = ParquetFile(f).num_row_groups
                    row_groups_per_file.append(row_groups)
            self._row_groups_per_file = row_groups_per_file
        return self._row_groups_per_file

    @property
    def files(self):
        if self._files is None:
            try:
                files = self.dataset.files
            except AttributeError:
                # compatibility at least with 3.0.0 <= pyarrow < 8.0.0
                files = self.dataset._dataset.files
            self._files = self._get_files(files)
        return self._files

    def to_pandas_dataframe(
        self,
        columns,
    ):
        from pyarrow.parquet import read_table

        return read_table(
            self._fs_path, columns=columns, filesystem=self.fs
        ).to_pandas()


@_inherit_docstrings(ColumnStoreDataset)
class FastParquetDataset(ColumnStoreDataset):
    def _init_dataset(self):  # noqa: GL08
        from fastparquet import ParquetFile

        return ParquetFile(self.fs_path, fs=self.fs)

    @property
    def pandas_metadata(self):
        if "pandas" not in self.dataset.key_value_metadata:
            return {}
        return json.loads(self.dataset.key_value_metadata["pandas"])

    @property
    def columns(self):
        return self.dataset.columns

    @property
    def engine(self):
        return "fastparquet"

    @property
    def row_groups_per_file(self):
        from fastparquet import ParquetFile

        if self._row_groups_per_file is None:
            row_groups_per_file = []
            # Count up the total number of row groups across all files and
            # keep track of row groups per file to use later.
            for file in self.files:
                with self.fs.open(file) as f:
                    row_groups = ParquetFile(f).info["row_groups"]
                    row_groups_per_file.append(row_groups)
            self._row_groups_per_file = row_groups_per_file
        return self._row_groups_per_file

    @property
    def files(self):
        if self._files is None:
            self._files = self._get_files(self._get_fastparquet_files())
        return self._files

    def to_pandas_dataframe(self, columns):
        return self.dataset.to_pandas(columns=columns)

    def _get_fastparquet_files(self):  # noqa: GL08
        # fastparquet doesn't have a nice method like PyArrow, so we
        # have to copy some of their logic here while we work on getting
        # an easier method to get a list of valid files.
        # See: https://github.com/dask/fastparquet/issues/795
        if "*" in self.path:
            files = self.fs.glob(self.path)
        else:
            files = [
                f
                for f in self.fs.find(self.path)
                if f.endswith(".parquet") or f.endswith(".parq")
            ]
        return files


class ParquetDispatcher(ColumnStoreDispatcher):
    """Class handles utils for reading `.parquet` files."""

    index_regex = re.compile(r"__index_level_\d+__")

    @classmethod
    def get_dataset(cls, path, engine, storage_options):
        """
        Retrieve Parquet engine specific Dataset implementation.

        Parameters
        ----------
        path : str, path object or file-like object
            The filepath of the parquet file in local filesystem or hdfs.
        engine : str
            Parquet library to use (only 'PyArrow' is supported for now).
        storage_options : dict
            Parameters for specific storage engine.

        Returns
        -------
        Dataset
            Either a PyArrowDataset or FastParquetDataset object.
        """
        if engine == "auto":
            # We follow in concordance with pandas
            engine_classes = [PyArrowDataset, FastParquetDataset]

            error_msgs = ""
            for engine_class in engine_classes:
                try:
                    return engine_class(path, storage_options)
                except ImportError as err:
                    error_msgs += "\n - " + str(err)

            raise ImportError(
                "Unable to find a usable engine; "
                + "tried using: 'pyarrow', 'fastparquet'.\n"
                + "A suitable version of "
                + "pyarrow or fastparquet is required for parquet "
                + "support.\n"
                + "Trying to import the above resulted in these errors:"
                + f"{error_msgs}"
            )
        elif engine == "pyarrow":
            return PyArrowDataset(path, storage_options)
        elif engine == "fastparquet":
            return FastParquetDataset(path, storage_options)
        else:
            raise ValueError("engine must be one of 'pyarrow', 'fastparquet'")

    @classmethod
    def call_deploy(cls, dataset, col_partitions, storage_options, **kwargs):
        """
        Deploy remote tasks to the workers with passed parameters.

        Parameters
        ----------
        dataset : Dataset
            Dataset object of Parquet file/files.
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
        from modin.core.storage_formats.pandas.parsers import ParquetFileToRead

        # If we don't have any columns to read, we should just return an empty
        # set of references.
        if len(col_partitions) == 0:
            return []

        row_groups_per_file = dataset.row_groups_per_file
        num_row_groups = sum(row_groups_per_file)
        parquet_files = dataset.files

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
                        func=cls.parse,
                        f_kwargs={
                            "files_for_parser": files_to_read,
                            "columns": cols,
                            "engine": dataset.engine,
                            "storage_options": storage_options,
                            **kwargs,
                        },
                        num_returns=3,
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
    def build_index(cls, dataset, partition_ids, index_columns):
        """
        Compute index and its split sizes of resulting Modin DataFrame.

        Parameters
        ----------
        dataset : Dataset
            Dataset object of Parquet file/files.
        partition_ids : list
            Array with references to the partitions data.
        index_columns : list
            List of index columns specified by pandas metadata.

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
            complete_index = dataset.to_pandas_dataframe(
                columns=column_names_to_read
            ).index
        # Empty DataFrame case
        elif len(partition_ids) == 0:
            return [], False
        else:
            index_ids = [part_id[0][1] for part_id in partition_ids if len(part_id) > 0]
            index_objs = cls.materialize(index_ids)
            complete_index = index_objs[0].append(index_objs[1:])
        return complete_index, range_index or (len(index_columns) == 0)

    @classmethod
    def build_query_compiler(cls, dataset, columns, index_columns, **kwargs):
        """
        Build query compiler from deployed tasks outputs.

        Parameters
        ----------
        dataset : Dataset
            Dataset object of Parquet file/files.
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
        partition_ids = cls.call_deploy(
            dataset, col_partitions, storage_options, **kwargs
        )
        index, sync_index = cls.build_index(dataset, partition_ids, index_columns)
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
    def _read(cls, path, engine, columns, use_nullable_dtypes, dtype_backend, **kwargs):
        """
        Load a parquet object from the file path, returning a query compiler.

        Parameters
        ----------
        path : str, path object or file-like object
            The filepath of the parquet file in local filesystem or hdfs.
        engine : {"auto", "pyarrow", "fastparquet"}
            Parquet library to use.
        columns : list
            If not None, only these columns will be read from the file.
        use_nullable_dtypes : Union[bool, lib.NoDefault]
        dtype_backend : {"numpy_nullable", "pyarrow", lib.no_default}
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
        if (
            any(arg not in ("storage_options",) for arg in kwargs)
            or use_nullable_dtypes != lib.no_default
        ):
            return cls.single_worker_read(
                path,
                engine=engine,
                columns=columns,
                use_nullable_dtypes=use_nullable_dtypes,
                dtype_backend=dtype_backend,
                reason="Parquet options that are not currently supported",
                **kwargs,
            )
        path = stringify_path(path)
        if isinstance(path, list):
            # TODO(https://github.com/modin-project/modin/issues/5723): read all
            # files in parallel.
            compilers: list[cls.query_compiler_cls] = [
                cls._read(
                    p, engine, columns, use_nullable_dtypes, dtype_backend, **kwargs
                )
                for p in path
            ]
            return compilers[0].concat(axis=0, other=compilers[1:], ignore_index=True)
        if isinstance(path, str):
            if os.path.isdir(path):
                path_generator = os.walk(path)
            else:
                storage_options = kwargs.get("storage_options")
                if storage_options is not None:
                    fs, fs_path = url_to_fs(path, **storage_options)
                else:
                    fs, fs_path = url_to_fs(path)
                path_generator = fs.walk(fs_path)
            partitioned_columns = set()
            # We do a tree walk of the path directory because partitioned
            # parquet directories have a unique column at each directory level.
            # Thus, we can use os.walk(), which does a dfs search, to walk
            # through the different columns that the data is partitioned on
            for _, dir_names, files in path_generator:
                if dir_names:
                    partitioned_columns.add(dir_names[0].split("=")[0])
                if files:
                    # Metadata files, git files, .DSStore
                    # TODO: fix conditional for column partitioning, see issue #4637
                    if len(files[0]) > 0 and files[0][0] == ".":
                        continue
                    break
            partitioned_columns = list(partitioned_columns)
            if len(partitioned_columns):
                return cls.single_worker_read(
                    path,
                    engine=engine,
                    columns=columns,
                    use_nullable_dtypes=use_nullable_dtypes,
                    dtype_backend=dtype_backend,
                    reason="Mixed partitioning columns in Parquet",
                    **kwargs,
                )

        dataset = cls.get_dataset(path, engine, kwargs.get("storage_options") or {})
        index_columns = (
            dataset.pandas_metadata.get("index_columns", [])
            if dataset.pandas_metadata
            else []
        )
        # If we have columns as None, then we default to reading in all the columns
        column_names = columns if columns else dataset.columns
        columns = [
            c
            for c in column_names
            if c not in index_columns and not cls.index_regex.match(c)
        ]

        return cls.build_query_compiler(
            dataset, columns, index_columns, dtype_backend=dtype_backend, **kwargs
        )

    @staticmethod
    def _to_parquet_check_support(kwargs):
        """
        Check if parallel version of `to_parquet` could be used.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed to `.to_parquet()`.

        Returns
        -------
        bool
            Whether parallel version of `to_parquet` is applicable.
        """
        path = kwargs["path"]
        compression = kwargs["compression"]
        if not isinstance(path, str):
            return False
        if any((path.endswith(ext) for ext in [".gz", ".bz2", ".zip", ".xz"])):
            return False
        if compression is None or not compression == "snappy":
            return False
        return True

    @classmethod
    def write(cls, qc, **kwargs):
        """
        Write a ``DataFrame`` to the binary parquet format.

        Parameters
        ----------
        qc : BaseQueryCompiler
            The query compiler of the Modin dataframe that we want to run `to_parquet` on.
        **kwargs : dict
            Parameters for `pandas.to_parquet(**kwargs)`.
        """
        if not cls._to_parquet_check_support(kwargs):
            return cls.base_io.to_parquet(qc, **kwargs)

        output_path = kwargs["path"]
        client_kwargs = (kwargs.get("storage_options") or {}).get("client_kwargs", {})
        fs, url = fsspec.core.url_to_fs(output_path, client_kwargs=client_kwargs)
        fs.mkdirs(url, exist_ok=True)

        def func(df, **kw):  # pragma: no cover
            """
            Dump a chunk of rows as parquet, then save them to target maintaining order.

            Parameters
            ----------
            df : pandas.DataFrame
                A chunk of rows to write to a parquet file.
            **kw : dict
                Arguments to pass to ``pandas.to_parquet(**kwargs)`` plus an extra argument
                `partition_idx` serving as chunk index to maintain rows order.
            """
            compression = kwargs["compression"]
            partition_idx = kw["partition_idx"]
            kwargs[
                "path"
            ] = f"{output_path}/part-{partition_idx:04d}.{compression}.parquet"
            df.to_parquet(**kwargs)
            return pandas.DataFrame()

        # Ensure that the metadata is synchronized
        qc._modin_frame._propagate_index_objs(axis=None)
        result = qc._modin_frame._partition_mgr_cls.map_axis_partitions(
            axis=1,
            partitions=qc._modin_frame._partitions,
            map_func=func,
            keep_partitioning=True,
            lengths=None,
            enumerate_partitions=True,
        )
        # pending completion
        cls.materialize([part.list_of_blocks[0] for row in result for part in row])
