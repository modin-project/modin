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

import numpy as np
import pandas

from modin.data_management.utils import compute_chunksize
from modin.engines.base.io.file_dispatcher import FileDispatcher


class ColumnStoreDispatcher(FileDispatcher):
    @classmethod
    def call_deploy(cls, fname, col_partitions, **kwargs):
        from modin.pandas import DEFAULT_NPARTITIONS

        return np.array(
            [
                cls.deploy(
                    cls.parse,
                    DEFAULT_NPARTITIONS + 2,
                    dict(
                        fname=fname,
                        columns=cols,
                        num_splits=DEFAULT_NPARTITIONS,
                        **kwargs,
                    ),
                )
                for cols in col_partitions
            ]
        ).T

    @classmethod
    def build_partition(cls, partition_ids, row_lengths, column_widths):
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
        from modin.pandas import DEFAULT_NPARTITIONS

        index_len = cls.materialize(partition_ids[-2][0])
        if isinstance(index_len, int):
            index = pandas.RangeIndex(index_len)
        else:
            index = index_len
            index_len = len(index)
        index_chunksize = compute_chunksize(
            pandas.DataFrame(index=index), DEFAULT_NPARTITIONS, axis=0
        )
        if index_chunksize > index_len:
            row_lengths = [index_len] + [0 for _ in range(DEFAULT_NPARTITIONS - 1)]
        else:
            row_lengths = [
                index_chunksize
                if i != DEFAULT_NPARTITIONS - 1
                else index_len - (index_chunksize * (DEFAULT_NPARTITIONS - 1))
                for i in range(DEFAULT_NPARTITIONS)
            ]
        return index, row_lengths

    @classmethod
    def build_columns(cls, columns):
        from modin.pandas import DEFAULT_NPARTITIONS

        column_splits = (
            len(columns) // DEFAULT_NPARTITIONS
            if len(columns) % DEFAULT_NPARTITIONS == 0
            else len(columns) // DEFAULT_NPARTITIONS + 1
        )
        col_partitions = [
            columns[i : i + column_splits]
            for i in range(0, len(columns), column_splits)
        ]
        column_widths = [len(c) for c in col_partitions]
        return col_partitions, column_widths

    @classmethod
    def build_dtypes(cls, partition_ids, columns):
        # Compute dtypes concatenating the results from each of the columns splits
        # determined above. This creates a pandas Series that contains a dtype for every
        # column.
        dtypes = pandas.concat(cls.materialize(list(partition_ids)), axis=0)
        dtypes.index = columns
        return dtypes

    @classmethod
    def build_query_compiler(cls, path, columns, **kwargs):
        col_partitions, column_widths = cls.build_columns(columns)
        partition_ids = cls.call_deploy(path, col_partitions, **kwargs)
        index, row_lens = cls.build_index(partition_ids)
        remote_parts = cls.build_partition(partition_ids[:-2], row_lens, column_widths)
        dtypes = cls.build_dtypes(partition_ids[-1], columns)
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
