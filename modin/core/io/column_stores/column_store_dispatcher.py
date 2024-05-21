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
import pandas

from modin.config import MinColumnPartitionSize, MinRowPartitionSize, NPartitions
from modin.core.io.file_dispatcher import FileDispatcher
from modin.core.storage_formats.pandas.utils import compute_chunksize


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
        return np.array(
            [
                cls.deploy(
                    func=cls.parse,
                    f_kwargs={
                        "fname": fname,
                        "columns": cols,
                        "num_splits": NPartitions.get(),
                        **kwargs,
                    },
                    num_returns=NPartitions.get() + 2,
                )
                for cols in col_partitions
            ]
        ).T

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
            filed with partition objects.
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
    def build_index(cls, partition_ids):
        """
        Compute index and its split sizes of resulting Modin DataFrame.

        Parameters
        ----------
        partition_ids : list
            Array with references to the partitions data.

        Returns
        -------
        index : pandas.Index
            Index of resulting Modin DataFrame.
        row_lengths : list
            List with lengths of index chunks.
        """
        index_len = (
            0 if len(partition_ids) == 0 else cls.materialize(partition_ids[-2][0])
        )
        if isinstance(index_len, int):
            index = pandas.RangeIndex(index_len)
        else:
            index = index_len
            index_len = len(index)
        num_partitions = NPartitions.get()
        min_block_size = MinRowPartitionSize.get()
        index_chunksize = compute_chunksize(index_len, num_partitions, min_block_size)
        if index_chunksize > index_len:
            row_lengths = [index_len] + [0 for _ in range(num_partitions - 1)]
        else:
            row_lengths = [
                (
                    index_chunksize
                    if (i + 1) * index_chunksize < index_len
                    else max(0, index_len - (index_chunksize * i))
                )
                for i in range(num_partitions)
            ]
        return index, row_lengths

    @classmethod
    def build_columns(cls, columns, num_row_parts=None):
        """
        Split columns into chunks that should be read by workers.

        Parameters
        ----------
        columns : list
            List of columns that should be read from file.
        num_row_parts : int, optional
            Number of parts the dataset is split into. This parameter is used
            to align the column partitioning with it so we won't end up with an
            over partitioned frame.

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
        if num_row_parts is None:
            # in column formats we mostly read columns in parallel rather than rows,
            # so we try to chunk columns as much as possible
            min_block_size = 1
        else:
            num_remaining_parts = round(NPartitions.get() / num_row_parts)
            min_block_size = min(
                columns_length // num_remaining_parts, MinColumnPartitionSize.get()
            )
        column_splits = compute_chunksize(
            columns_length, NPartitions.get(), max(1, min_block_size)
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
        index, row_lens = cls.build_index(partition_ids)
        remote_parts = cls.build_partition(partition_ids[:-2], row_lens, column_widths)
        dtypes = (
            cls.build_dtypes(partition_ids[-1], columns)
            if len(partition_ids) > 0
            else None
        )
        new_query_compiler = cls.query_compiler_cls(
            cls.frame_cls(
                remote_parts,
                index,
                columns,
                row_lens,
                column_widths,
                dtypes=dtypes,
            )
        )
        return new_query_compiler
