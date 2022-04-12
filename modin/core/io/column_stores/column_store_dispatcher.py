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
Module houses `ColumnStoreDispatcher` class.

`ColumnStoreDispatcher` contains utils for handling columnar store format files,
inherits util functions for handling files from `FileDispatcher` class and can be
used as base class for dipatchers of specific columnar store formats.
"""

import numpy as np
import os
import pandas
import warnings

from modin.core.storage_formats.pandas.utils import compute_chunksize
from modin.core.io.file_dispatcher import FileDispatcher
from modin.config import NPartitions


class ColumnStoreDispatcher(FileDispatcher):
    """
    Class handles utils for reading columnar store format files.

    Inherits some util functions for processing files from `FileDispatcher` class.
    """

    @classmethod
    def call_deploy(cls, fname, col_partitions, **kwargs):
        """
        Deploy remote tasks to the workers with passed parameters.

        Parameters
        ----------
        fname : str, path object or file-like object
            Name of the file to read.
        col_partitions : list
            List of arrays with columns names that should be read
            by each partition.
        **kwargs : dict
            Parameters of deploying read_* function.

        Returns
        -------
        np.ndarray
            Array with references to the task deploy result for each partition.
        """
        from pyarrow.parquet import ParquetFile

        num_row_groups = ParquetFile(fname).metadata.num_row_groups
        step = num_row_groups // NPartitions.get()
        if num_row_groups % NPartitions.get() != 0:
            step += 1
        import time

        start = time.time()
        return np.array(
            [
                [
                    cls.deploy(
                        cls.parse,
                        fname=fname,
                        columns=cols,
                        row_group_start=row_start,
                        row_group_end=row_start + step,
                        row_group_end=row_start + step,
                        start=start,
                        **kwargs,
                    )
                    for cols in col_partitions
                ]
                for row_start in range(0, num_row_groups, step)
            ]
        )

    @classmethod
    def build_partition(cls, partition_ids, column_widths):
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
            filed with partition objects.
        """
        return np.array(
            [
                [
                    cls.frame_partition_cls(
                        partition_ids[i][j],
                        width=column_widths[j],
                    )
                    for j in range(len(partition_ids[i]))
                ]
                for i in range(len(partition_ids))
            ]
        )

    @classmethod
    def build_index(cls, path):
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
        # num_partitions = NPartitions.get()
        # index_len = (
        #     0 if len(partition_ids) == 0 else cls.materialize(partition_ids[-2][0])
        # )
        # if isinstance(index_len, int):
        #     index = pandas.RangeIndex(index_len)
        # else:
        #     index = index_len
        #     index_len = len(index)
        # index_chunksize = compute_chunksize(index_len, num_partitions)
        # if index_chunksize > index_len:
        #     row_lengths = [index_len] + [0 for _ in range(num_partitions - 1)]
        # else:
        #     row_lengths = [
        #         index_chunksize
        #         if (i + 1) * index_chunksize < index_len
        #         else max(0, index_len - (index_chunksize * i))
        #         for i in range(num_partitions)
        #     ]
        # return index, row_lengths

        from pyarrow.parquet import ParquetFile, ParquetDataset

        index_columns = []
        pandas_metadata = ParquetDataset(
            path, use_legacy_dataset=False
        ).schema.pandas_metadata
        if pandas_metadata is not None:
            index_columns = pandas_metadata.get("index_columns", index_columns)
        file = ParquetFile(path)
        index = file.read(columns=[], use_pandas_metadata=True).to_pandas().index
        return index, len(index_columns) == 0 or any(
            not isinstance(c, str) for c in index_columns
        )

    @classmethod
    def build_columns(cls, columns):
        """
        Split columns into chunks, that should be read be workers.

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
        dtypes = pandas.concat(
            [df.dtypes for df in cls.materialize(list(partition_ids))], axis=0
        )
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
        index, needs_index_sync = cls.build_index(path)
        remote_parts = cls.build_partition(partition_ids, column_widths)
        dtypes = (
            cls.build_dtypes(partition_ids[0], columns)
            if len(partition_ids) > 0
            else None
        )
        frame = cls.frame_cls(
            remote_parts,
            index,
            columns,
            # TODO: see if there's a way to get row lengths without reading partition.
            row_lengths=None,
            column_widths=column_widths,
            dtypes=dtypes,
        )
        new_query_compiler = cls.query_compiler_cls(frame)
        new_query_compiler = cls.query_compiler_cls(frame)
        new_query_compiler = cls.query_compiler_cls(frame)
        if needs_index_sync:
            frame.synchronize_labels(axis=0)
        return new_query_compiler
