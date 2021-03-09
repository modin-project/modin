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

from modin.engines.base.io.text.text_file_dispatcher import (
    TextFileDispatcher,
    ColumnNamesTypes,
)
import pandas
from csv import QUOTE_NONE, Dialect
import sys
from typing import Union, Sequence, Callable, Dict, Tuple
from pandas._typing import FilePathOrBuffer

from modin.config import NPartitions

ReadCsvKwargsType = Dict[
    str, Union[str, int, bool, dict, object, Sequence, Callable, Dialect, None]
]
IndexColType = Union[int, str, bool, Sequence[int], Sequence[str], None]


class CSVDispatcher(TextFileDispatcher):
    @classmethod
    def _read(cls, filepath_or_buffer, **kwargs):
        filepath_or_buffer_md = (
            cls.get_path(filepath_or_buffer)
            if isinstance(filepath_or_buffer, str)
            else cls.get_path_or_buffer(filepath_or_buffer)
        )
        compression_infered = cls.infer_compression(
            filepath_or_buffer, kwargs.get("compression")
        )
        use_modin_impl = cls._read_csv_check_support(
            filepath_or_buffer, kwargs, compression_infered
        )
        if not use_modin_impl:
            return cls.single_worker_read(filepath_or_buffer, **kwargs)

        # Getting frequently used read_csv kwargs
        names = kwargs.get("names", None)
        index_col = kwargs.get("index_col", None)
        encoding = kwargs.get("encoding", None)
        skiprows = kwargs.get("skiprows")

        is_quoting = kwargs.get("quoting", "") != QUOTE_NONE
        quotechar = kwargs.get("quotechar", '"').encode(
            encoding if encoding is not None else "UTF-8"
        )

        # Define header size for further skipping (Header can be skipped because header
        # information will be obtained further from empty_df, so no need to handle it
        # by workers)
        header_size = cls._define_header_size(
            kwargs.get("header", "infer"),
            names,
        )
        # Since skiprows can be only integer here (non-integer skiprows trigger fallback
        # to pandas implementation for now) we can process header_size and skiprows
        # simultaneously
        skiprows = skiprows + header_size if skiprows else header_size

        # Now we need to define parameters, which are common for all partitions. These
        # parameters can be `sniffed` from empty dataframes created further
        if names is None:
            # For the sake of the empty df, we assume no `index_col` to get the correct
            # column names before we build the index. Because we pass `names` in, this
            # step has to happen without removing the `index_col` otherwise it will not
            # be assigned correctly
            names = pandas.read_csv(
                filepath_or_buffer,
                **dict(kwargs, usecols=None, nrows=0, skipfooter=0, index_col=None),
            ).columns
        elif index_col is None and not kwargs.get("usecols", None):
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
        empty_pd_df = pandas.read_csv(
            filepath_or_buffer,
            **dict(kwargs, nrows=0, skipfooter=0, index_col=index_col),
        )
        column_names = empty_pd_df.columns

        # Max number of partitions available
        num_partitions = NPartitions.get()
        # This is the number of splits for the columns
        num_splits = min(len(column_names), num_partitions)
        # Metadata definition
        column_widths, num_splits = cls._define_metadata(
            empty_pd_df, num_splits, column_names
        )

        # kwargs that will be passed to the workers
        partition_kwargs = dict(
            kwargs,
            fname=filepath_or_buffer_md,
            num_splits=num_splits,
            header=None,
            names=names,
            skipfooter=0,
            skiprows=1 if encoding is not None else None,
            nrows=None,
            compression=compression_infered,
            index_col=index_col,
        )

        with cls.file_open(filepath_or_buffer_md, "rb", compression_infered) as f:
            splits = cls.partitioned_file(
                f,
                num_partitions=num_partitions,
                nrows=kwargs.get("nrows", None),
                skiprows=skiprows,
                quotechar=quotechar,
                is_quoting=is_quoting,
            )

        partition_ids, index_ids, dtypes_ids = cls._launch_tasks(
            splits, **partition_kwargs
        )

        new_query_compiler = cls._get_new_qc(
            partition_ids=partition_ids,
            index_ids=index_ids,
            dtypes_ids=dtypes_ids,
            index_col_md=index_col,
            index_name=empty_pd_df.index.name,
            column_widths=column_widths,
            column_names=column_names,
            squeeze=kwargs.get("squeeze", False),
            skipfooter=kwargs.get("skipfooter", None),
            parse_dates=kwargs.get("parse_dates", False),
        )
        return new_query_compiler

    # _read helper functions
    @classmethod
    def _read_csv_check_support(
        cls,
        filepath_or_buffer: FilePathOrBuffer,
        read_csv_kwargs: ReadCsvKwargsType,
        compression_infered: str,
    ) -> bool:
        """
        Check whatever or not passed parameters are supported by current modin.read_csv
        implementation.
        ----------
        filepath_or_buffer: str, path object or file-like object
                `filepath_or_buffer` parameter of read_csv function.
        read_csv_kwargs: ReadCsvKwargsType
                Parameters of read_csv function.
        compression_infered: str
                Infered `compression` parameter of read_csv function.

        Returns
        -------
        bool:
            Whatever passed parameters are supported or not.
        """
        if isinstance(filepath_or_buffer, str):
            if not cls.file_exists(filepath_or_buffer):
                return False
        elif not cls.pathlib_or_pypath(filepath_or_buffer):
            return False

        if compression_infered is not None:
            use_modin_impl = compression_infered in ["gzip", "bz2", "xz"] or (
                compression_infered == "zip"
                # need python3.7 to .seek and .tell ZipExtFile
                and sys.version_info[0] == 3
                and sys.version_info[1] >= 7
            )
            if not use_modin_impl:
                return False

        if read_csv_kwargs.get("chunksize") is not None:
            return False

        skiprows = read_csv_kwargs.get("skiprows")
        if skiprows is not None and not isinstance(skiprows, int):
            return False

        return True

    @classmethod
    def _define_index(
        cls,
        index_ids: list,
        index_col: IndexColType,
        index_name: str,
    ) -> Tuple[IndexColType, list]:
        """
        Compute the index based on a sum of the lengths of each partition
        (by default) or based on the column(s) that were requested.
        ----------
        index_ids: list
                Array with references to the partitions index objects.
        index_col: IndexColType
                index_col parameter of read_csv function.
        index_name: str
                Name that should be assigned to the index if `index_col`
                is not provided.

        Returns
        -------
        new_index: IndexColType
                Index that should be passed to the new_frame constructor.
        row_lengths: list
                Partitions rows lengths.
        """
        if index_col is None:
            row_lengths = cls.materialize(index_ids)
            new_index = pandas.RangeIndex(sum(row_lengths))
        else:
            index_objs = cls.materialize(index_ids)
            row_lengths = [len(o) for o in index_objs]
            new_index = index_objs[0].append(index_objs[1:])
            new_index.name = index_name

        return new_index, row_lengths

    @classmethod
    def _define_column_names(
        cls,
        column_names: ColumnNamesTypes,
        parse_dates: Union[bool, dict, Sequence],
        column_widths: list,
    ) -> Tuple[ColumnNamesTypes, list]:
        """
        Redefine columns names in accordance to the parse_dates parameter.
        ----------
        column_names: ColumnNamesTypes
                Array with columns names.
        parse_dates: array, bool or dict
                `parse_dates` parameter of read_csv function.
        column_widths: list
                Number of columns in each partition.

        Returns
        -------
        column_names: ColumnNamesTypes
                Array with redefined columns names.
        column_widths: list
                Updated `column_widths` parameter.
        """
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

        return column_names, column_widths

    @classmethod
    def _get_new_qc(
        cls,
        partition_ids: list,
        index_ids: list,
        dtypes_ids: list,
        index_col_md: IndexColType,
        index_name: str,
        column_widths: list,
        column_names: ColumnNamesTypes,
        **kwargs,
    ):
        """
        Get new query compiler from data received from workers.
        ----------
        partition_ids: list
                array with references to the partitions data.
        index_ids: list
                array with references to the partitions index objects.
        dtypes_ids: list
                array with references to the partitions dtypes objects.
        index_col_md: IndexColType
                `index_col` parameter passed to the workers.
        index_name: str
                Name that should be assigned to the index if `index_col`
                is not provided.
        column_widths: list
                Number of columns in each partition.
        column_names: ColumnNamesTypes
                Array with columns names.

        Returns
        -------
        new_query_compiler:
                New query compiler, created from `new_frame`.
        """
        new_index, row_lengths = cls._define_index(index_ids, index_col_md, index_name)
        # Compute dtypes by getting collecting and combining all of the partitions. The
        # reported dtypes from differing rows can be different based on the inference in
        # the limited data seen by each worker. We use pandas to compute the exact dtype
        # over the whole column for each column. The index is set below.
        dtypes = cls.get_dtypes(dtypes_ids) if len(dtypes_ids) > 0 else None
        # Compose modin partitions from `partition_ids`
        partition_ids = cls.build_partition(partition_ids, row_lengths, column_widths)
        column_names, column_widths = cls._define_column_names(
            column_names, kwargs.get("parse_dates", False), column_widths
        )

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
        skipfooter = kwargs.get("skipfooter", None)
        if skipfooter:
            new_query_compiler = new_query_compiler.drop(
                new_query_compiler.index[-skipfooter:]
            )
        if kwargs.get("squeeze", False) and len(new_query_compiler.columns) == 1:
            return new_query_compiler[new_query_compiler.columns[0]]
        if index_col_md is None:
            new_query_compiler._modin_frame._apply_index_objs(axis=0)

        return new_query_compiler
