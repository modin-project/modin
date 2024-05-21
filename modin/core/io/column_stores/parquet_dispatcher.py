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

from __future__ import annotations

import functools
import json
import os
import re
from typing import TYPE_CHECKING

import fsspec
import numpy as np
import pandas
import pandas._libs.lib as lib
from fsspec.core import url_to_fs
from fsspec.spec import AbstractBufferedFile
from packaging import version
from pandas.io.common import stringify_path

from modin.config import MinColumnPartitionSize, MinRowPartitionSize, NPartitions
from modin.core.io.column_stores.column_store_dispatcher import ColumnStoreDispatcher
from modin.error_message import ErrorMessage
from modin.utils import _inherit_docstrings

if TYPE_CHECKING:
    from modin.core.storage_formats.pandas.parsers import ParquetFileToRead


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
    """

    def __init__(self, path, storage_options):  # noqa : PR01
        self.path = path.__fspath__() if isinstance(path, os.PathLike) else path
        self.storage_options = storage_options
        self._fs_path = None
        self._fs = None
        self.dataset = self._init_dataset()

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

    @functools.cached_property
    def files(self):
        """Return the list of formatted file paths of the dataset."""
        raise NotImplementedError

    @functools.cached_property
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

        return ParquetDataset(self.fs_path, filesystem=self.fs)

    @property
    def pandas_metadata(self):
        return self.dataset.schema.pandas_metadata

    @property
    def columns(self):
        return self.dataset.schema.names

    @property
    def engine(self):
        return "pyarrow"

    @functools.cached_property
    def row_groups_per_file(self):
        from pyarrow.parquet import ParquetFile

        row_groups_per_file = []
        # Count up the total number of row groups across all files and
        # keep track of row groups per file to use later.
        for file in self.files:
            with self.fs.open(file) as f:
                row_groups = ParquetFile(f).num_row_groups
                row_groups_per_file.append(row_groups)
        return row_groups_per_file

    @functools.cached_property
    def files(self):
        files = self.dataset.files
        return self._get_files(files)

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

    @functools.cached_property
    def row_groups_per_file(self):
        from fastparquet import ParquetFile

        row_groups_per_file = []
        # Count up the total number of row groups across all files and
        # keep track of row groups per file to use later.
        for file in self.files:
            with self.fs.open(file) as f:
                row_groups = ParquetFile(f).info["row_groups"]
                row_groups_per_file.append(row_groups)
        return row_groups_per_file

    @functools.cached_property
    def files(self):
        return self._get_files(self._get_fastparquet_files())

    def to_pandas_dataframe(self, columns):
        return self.dataset.to_pandas(columns=columns)

    # Karthik Velayutham writes:
    #
    # fastparquet doesn't have a nice method like PyArrow, so we
    # have to copy some of their logic here while we work on getting
    # an easier method to get a list of valid files.
    # See: https://github.com/dask/fastparquet/issues/795
    def _get_fastparquet_files(self):  # noqa: GL08
        if "*" in self.path:
            files = self.fs.glob(self.path)
        else:
            # (Resolving issue #6778)
            #
            # Users will pass in a directory to a delta table, which stores parquet
            # files in various directories along with other, non-parquet files. We
            # need to identify those parquet files and not the non-parquet files.
            #
            # However, we also need to support users passing in explicit files that
            # don't necessarily have the `.parq` or `.parquet` extension -- if a user
            # says that a file is parquet, then we should probably give it a shot.
            if self.fs.isfile(self.path):
                files = self.fs.find(self.path)
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
    def _determine_partitioning(
        cls, dataset: ColumnStoreDataset
    ) -> "list[list[ParquetFileToRead]]":
        """
        Determine which partition will read certain files/row groups of the dataset.

        Parameters
        ----------
        dataset : ColumnStoreDataset

        Returns
        -------
        list[list[ParquetFileToRead]]
            Each element in the returned list describes a list of files that a partition has to read.
        """
        from modin.core.storage_formats.pandas.parsers import ParquetFileToRead

        parquet_files = dataset.files
        row_groups_per_file = dataset.row_groups_per_file
        num_row_groups = sum(row_groups_per_file)

        if num_row_groups == 0:
            return []

        num_splits = min(NPartitions.get(), num_row_groups)
        part_size = num_row_groups // num_splits
        # If 'num_splits' does not divide 'num_row_groups' then we can't cover all of
        # the row groups using the original 'part_size'. According to the 'reminder'
        # there has to be that number of partitions that should read 'part_size + 1'
        # number of row groups.
        reminder = num_row_groups % num_splits
        part_sizes = [part_size] * (num_splits - reminder) + [part_size + 1] * reminder

        partition_files = []
        file_idx = 0
        row_group_idx = 0
        row_groups_left_in_current_file = row_groups_per_file[file_idx]
        # this is used for sanity check at the end, verifying that we indeed added all of the row groups
        total_row_groups_added = 0
        for size in part_sizes:
            row_groups_taken = 0
            part_files = []
            while row_groups_taken != size:
                if row_groups_left_in_current_file < 1:
                    file_idx += 1
                    row_group_idx = 0
                    row_groups_left_in_current_file = row_groups_per_file[file_idx]

                to_take = min(size - row_groups_taken, row_groups_left_in_current_file)
                part_files.append(
                    ParquetFileToRead(
                        parquet_files[file_idx],
                        row_group_start=row_group_idx,
                        row_group_end=row_group_idx + to_take,
                    )
                )
                row_groups_left_in_current_file -= to_take
                row_groups_taken += to_take
                row_group_idx += to_take

            total_row_groups_added += row_groups_taken
            partition_files.append(part_files)

        sanity_check = (
            len(partition_files) == num_splits
            and total_row_groups_added == num_row_groups
        )
        ErrorMessage.catch_bugs_and_request_email(
            failure_condition=not sanity_check,
            extra_log="row groups added does not match total num of row groups across parquet files",
        )
        return partition_files

    @classmethod
    def call_deploy(
        cls,
        partition_files: "list[list[ParquetFileToRead]]",
        col_partitions: "list[list[str]]",
        storage_options: dict,
        engine: str,
        **kwargs,
    ):
        """
        Deploy remote tasks to the workers with passed parameters.

        Parameters
        ----------
        partition_files : list[list[ParquetFileToRead]]
            List of arrays with files that should be read by each partition.
        col_partitions : list[list[str]]
            List of arrays with columns names that should be read
            by each partition.
        storage_options : dict
            Parameters for specific storage engine.
        engine : {"auto", "pyarrow", "fastparquet"}
            Parquet library to use for reading.
        **kwargs : dict
            Parameters of deploying read_* function.

        Returns
        -------
        List
            Array with references to the task deploy result for each partition.
        """
        # If we don't have any columns to read, we should just return an empty
        # set of references.
        if len(col_partitions) == 0:
            return []

        all_partitions = []
        for files_to_read in partition_files:
            all_partitions.append(
                [
                    cls.deploy(
                        func=cls.parse,
                        f_kwargs={
                            "files_for_parser": files_to_read,
                            "columns": cols,
                            "engine": engine,
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
    def build_index(cls, dataset, partition_ids, index_columns, filters):
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
        filters : list
            List of filters to be used in reading the Parquet file/files.

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
        range_index_metadata = None
        column_names_to_read = []
        for column in index_columns:
            # https://pandas.pydata.org/docs/development/developer.html#storing-pandas-dataframe-objects-in-apache-parquet-format
            # describes the format of the index column metadata.
            # It is a list, where each entry is either a string or a dictionary.
            # A string means that a column stored in the dataset is (part of) the index.
            # A dictionary is metadata about a RangeIndex, which is metadata-only and not stored
            # in the dataset as a column.
            # There cannot be both for a single dataframe, because a MultiIndex can only contain
            # "actual data" columns and not RangeIndex objects.
            # See similar code in pyarrow: https://github.com/apache/arrow/blob/44811ba18477560711d512939535c8389dd7787b/python/pyarrow/pandas_compat.py#L912-L926
            # and in fastparquet, here is where RangeIndex is handled: https://github.com/dask/fastparquet/blob/df1219300a96bc1baf9ebad85f4f5676a130c9e8/fastparquet/api.py#L809-L815
            if isinstance(column, str):
                column_names_to_read.append(column)
                range_index = False
            elif column["kind"] == "range":
                range_index_metadata = column

        # When the index has meaningful values, stored in a column, we will replicate those
        # exactly in the Modin dataframe's index. This index may have repeated values, be unsorted,
        # etc. This is all fine.
        # A range index is the special case: we want the Modin dataframe to have a single range,
        # not a range that keeps restarting. i.e. if the partitions have index 0-9, 0-19, 0-29,
        # we want our Modin dataframe to have 0-59.
        # When there are no filters, it is relatively cheap to construct the index by
        # actually reading in the necessary data, here in the main process.
        # When there are filters, we let the workers materialize the indices before combining to
        # get a single range.

        # For the second check, let us consider the case where we have an empty dataframe,
        # that has a valid index.
        if (range_index and filters is None) or (
            len(partition_ids) == 0 and len(column_names_to_read) != 0
        ):
            complete_index = dataset.to_pandas_dataframe(
                columns=column_names_to_read
            ).index
        # Empty DataFrame case
        elif len(partition_ids) == 0:
            return [], False
        else:
            index_ids = [part_id[0][1] for part_id in partition_ids if len(part_id) > 0]
            index_objs = cls.materialize(index_ids)
            if range_index:
                # There are filters, so we had to materialize in order to
                # determine how many items there actually are
                total_filtered_length = sum(
                    len(index_part) for index_part in index_objs
                )

                metadata_length_mismatch = False
                if range_index_metadata is not None:
                    metadata_implied_length = (
                        range_index_metadata["stop"] - range_index_metadata["start"]
                    ) / range_index_metadata["step"]
                    metadata_length_mismatch = (
                        total_filtered_length != metadata_implied_length
                    )

                # pyarrow ignores the RangeIndex metadata if it is not consistent with data length.
                # https://github.com/apache/arrow/blob/44811ba18477560711d512939535c8389dd7787b/python/pyarrow/pandas_compat.py#L924-L926
                # fastparquet keeps the start and step from the metadata and just adjusts to the length.
                # https://github.com/dask/fastparquet/blob/df1219300a96bc1baf9ebad85f4f5676a130c9e8/fastparquet/api.py#L815
                if range_index_metadata is None or (
                    isinstance(dataset, PyArrowDataset) and metadata_length_mismatch
                ):
                    complete_index = pandas.RangeIndex(total_filtered_length)
                else:
                    complete_index = pandas.RangeIndex(
                        start=range_index_metadata["start"],
                        step=range_index_metadata["step"],
                        stop=(
                            range_index_metadata["start"]
                            + (total_filtered_length * range_index_metadata["step"])
                        ),
                        name=range_index_metadata["name"],
                    )
            else:
                complete_index = index_objs[0].append(index_objs[1:])
        return complete_index, range_index or (len(index_columns) == 0)

    @classmethod
    def _normalize_partitioning(cls, remote_parts, row_lengths, column_widths):
        """
        Normalize partitioning according to the default partitioning scheme in Modin.

        The result of 'read_parquet()' is often under partitioned over rows and over partitioned
        over columns, so this method expands the number of row splits and shrink the number of column splits.

        Parameters
        ----------
        remote_parts : np.ndarray
        row_lengths : list of ints or None
            Row lengths, if 'None', won't repartition across rows.
        column_widths : list of ints

        Returns
        -------
        remote_parts : np.ndarray
        row_lengths : list of ints or None
        column_widths : list of ints
        """
        if len(remote_parts) == 0:
            return remote_parts, row_lengths, column_widths

        from modin.core.storage_formats.pandas.utils import get_length_list

        # The code in this function is actually a duplication of what 'BaseQueryCompiler.repartition()' does,
        # however this implementation works much faster for some reason

        actual_row_nparts = remote_parts.shape[0]

        if row_lengths is not None:
            desired_row_nparts = max(
                1, min(sum(row_lengths) // MinRowPartitionSize.get(), NPartitions.get())
            )
        else:
            desired_row_nparts = actual_row_nparts

        # only repartition along rows if the actual number of row splits 1.5 times SMALLER than desired
        if 1.5 * actual_row_nparts < desired_row_nparts:
            # assuming that the sizes of parquet's row groups are more or less equal,
            # so trying to use the same number of splits for each partition
            splits_per_partition = desired_row_nparts // actual_row_nparts
            remainder = desired_row_nparts % actual_row_nparts

            new_parts = []
            new_row_lengths = []

            for row_idx, (part_len, row_parts) in enumerate(
                zip(row_lengths, remote_parts)
            ):
                num_splits = splits_per_partition
                # 'remainder' indicates how many partitions have to be split into 'num_splits + 1' splits
                # to have exactly 'desired_row_nparts' in the end
                if row_idx < remainder:
                    num_splits += 1

                if num_splits == 1:
                    new_parts.append(row_parts)
                    new_row_lengths.append(part_len)
                    continue

                offset = len(new_parts)
                # adding empty row parts according to the number of splits
                new_parts.extend([[] for _ in range(num_splits)])
                for part in row_parts:
                    split = cls.frame_cls._partition_mgr_cls._column_partitions_class(
                        [part]
                    ).apply(
                        lambda df: df,
                        num_splits=num_splits,
                        maintain_partitioning=False,
                    )
                    for i in range(num_splits):
                        new_parts[offset + i].append(split[i])

                new_row_lengths.extend(
                    get_length_list(part_len, num_splits, MinRowPartitionSize.get())
                )

            remote_parts = np.array(new_parts)
            row_lengths = new_row_lengths

        desired_col_nparts = max(
            1,
            min(sum(column_widths) // MinColumnPartitionSize.get(), NPartitions.get()),
        )
        # only repartition along cols if the actual number of col splits 1.5 times BIGGER than desired
        if 1.5 * desired_col_nparts < remote_parts.shape[1]:
            remote_parts = np.array(
                [
                    (
                        cls.frame_cls._partition_mgr_cls._row_partition_class(
                            row_parts
                        ).apply(
                            lambda df: df,
                            num_splits=desired_col_nparts,
                            maintain_partitioning=False,
                        )
                    )
                    for row_parts in remote_parts
                ]
            )
            column_widths = get_length_list(
                sum(column_widths), desired_col_nparts, MinColumnPartitionSize.get()
            )

        return remote_parts, row_lengths, column_widths

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
        filters = kwargs.get("filters", None)

        partition_files = cls._determine_partitioning(dataset)
        col_partitions, column_widths = cls.build_columns(
            columns,
            num_row_parts=len(partition_files),
        )
        partition_ids = cls.call_deploy(
            partition_files, col_partitions, storage_options, dataset.engine, **kwargs
        )
        index, sync_index = cls.build_index(
            dataset, partition_ids, index_columns, filters
        )
        remote_parts = cls.build_partition(partition_ids, column_widths)
        if len(partition_ids) > 0:
            row_lengths = [part.length() for part in remote_parts.T[0]]
        else:
            row_lengths = None

        remote_parts, row_lengths, column_widths = cls._normalize_partitioning(
            remote_parts, row_lengths, column_widths
        )

        if (
            dataset.pandas_metadata
            and "column_indexes" in dataset.pandas_metadata
            and len(dataset.pandas_metadata["column_indexes"]) == 1
            and dataset.pandas_metadata["column_indexes"][0]["numpy_type"] == "int64"
        ):
            columns = pandas.Index(columns).astype("int64").to_list()

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
            (set(kwargs) - {"storage_options", "filters", "filesystem"})
            or use_nullable_dtypes != lib.no_default
            or kwargs.get("filesystem") is not None
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
        kwargs["path"] = stringify_path(kwargs["path"])
        output_path = kwargs["path"]
        if not isinstance(output_path, str):
            return cls.base_io.to_parquet(qc, **kwargs)
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
            kwargs["path"] = (
                f"{output_path}/part-{partition_idx:04d}.{compression}.parquet"
            )
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
