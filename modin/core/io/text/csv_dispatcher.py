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

"""Module houses `CSVDispatcher` class, that is used for reading `.csv` files."""

from modin.core.io.file_dispatcher import OpenFile
from modin.core.io.text.text_file_dispatcher import (
    TextFileDispatcher,
    ColumnNamesTypes,
)
import pandas
from pandas.core.dtypes.common import is_list_like
from csv import QUOTE_NONE, Dialect
import sys
import io
from typing import Union, Sequence, Callable, Dict, Tuple
from pandas._typing import FilePathOrBuffer
import pandas._libs.lib as lib
import numpy as np

from modin.config import NPartitions

ReadCsvKwargsType = Dict[
    str, Union[str, int, bool, dict, object, Sequence, Callable, Dialect, None]
]
IndexColType = Union[int, str, bool, Sequence[int], Sequence[str], None]


class CSVDispatcher(TextFileDispatcher):
    """
    Class handles utils for reading `.csv` files.

    Inherits some common for text files util functions from `TextFileDispatcher` class.
    """

    @classmethod
    def _read(cls, filepath_or_buffer, **kwargs):
        """
        Read data from `filepath_or_buffer` according to `kwargs` parameters.

        Parameters
        ----------
        filepath_or_buffer : str, path object or file-like object
            `filepath_or_buffer` parameter of `read_csv` function.
        **kwargs : dict
            Parameters of `read_csv` function.

        Returns
        -------
        new_query_compiler : BaseQueryCompiler
            Query compiler with imported data for further processing.

        Notes
        -----
        `skiprows` is handled diferently based on the parameter type because of
        performance reasons. If `skiprows` is integer - rows will be skipped during
        data file partitioning and wouldn't be actually read. If `skiprows` is array
        or callable - full data file will be read and only then rows will be dropped.
        """
        filepath_or_buffer_md = (
            cls.get_path(filepath_or_buffer)
            if isinstance(filepath_or_buffer, str)
            else cls.get_path_or_buffer(filepath_or_buffer)
        )
        compression_infered = cls.infer_compression(
            filepath_or_buffer, kwargs.get("compression")
        )
        # Getting frequently used read_csv kwargs
        names = kwargs.get("names", lib.no_default)
        index_col = kwargs.get("index_col", None)
        encoding = kwargs.get("encoding", None)
        skiprows = kwargs.get("skiprows", None)
        header = kwargs.get("header", "infer")
        # Define header size for further skipping (Header can be skipped because header
        # information will be obtained further from empty_df, so no need to handle it
        # by workers)
        header_size = cls._define_header_size(
            header,
            names,
        )
        (
            skiprows_md,
            pre_reading,
            skiprows_partitioning,
        ) = cls._manage_skiprows_parameter(skiprows, header_size)
        should_handle_skiprows = skiprows_md is not None and not isinstance(
            skiprows_md, int
        )

        use_modin_impl = cls._read_csv_check_support(
            filepath_or_buffer, kwargs, compression_infered
        )
        if not use_modin_impl:
            return cls.single_worker_read(filepath_or_buffer, **kwargs)

        is_quoting = kwargs.get("quoting", "") != QUOTE_NONE
        # In these cases we should pass additional metadata
        # to the workers to match pandas output
        pass_names = names in [None, lib.no_default] and (
            skiprows is not None or kwargs.get("skipfooter", 0) != 0
        )

        pd_df_metadata = pandas.read_csv(
            filepath_or_buffer,
            **dict(kwargs, nrows=1, skipfooter=0, index_col=index_col),
        )
        column_names = pd_df_metadata.columns
        column_widths, num_splits = cls._define_metadata(pd_df_metadata, column_names)

        # kwargs that will be passed to the workers
        partition_kwargs = dict(
            kwargs,
            fname=filepath_or_buffer_md,
            num_splits=num_splits,
            header_size=header_size if not pass_names else 0,
            names=names if not pass_names else column_names,
            header=header if not pass_names else "infer",
            skipfooter=0,
            skiprows=None,
            nrows=None,
            compression=compression_infered,
        )

        with OpenFile(filepath_or_buffer_md, "rb", compression_infered) as f:
            old_pos = f.tell()
            fio = io.TextIOWrapper(f, encoding=encoding, newline="")
            newline, quotechar = cls.compute_newline(
                fio, encoding, kwargs.get("quotechar", '"')
            )
            f.seek(old_pos)
            splits = cls.partitioned_file(
                f,
                num_partitions=NPartitions.get(),
                nrows=kwargs.get("nrows", None) if not should_handle_skiprows else None,
                skiprows=skiprows_partitioning,
                quotechar=quotechar,
                is_quoting=is_quoting,
                encoding=encoding,
                newline=newline,
                header_size=header_size,
                pre_reading=pre_reading,
            )

        partition_ids, index_ids, dtypes_ids = cls._launch_tasks(
            splits, **partition_kwargs
        )

        new_query_compiler = cls._get_new_qc(
            partition_ids=partition_ids,
            index_ids=index_ids,
            dtypes_ids=dtypes_ids,
            index_col=index_col,
            index_name=pd_df_metadata.index.name,
            column_widths=column_widths,
            column_names=column_names,
            skiprows_md=skiprows_md if should_handle_skiprows else None,
            header_size=header_size,
            skipfooter=kwargs.get("skipfooter", None),
            parse_dates=kwargs.get("parse_dates", False),
            nrows=kwargs.get("nrows", None) if should_handle_skiprows else None,
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
        Check if passed parameters are supported by current `read_csv` implementation.

        Parameters
        ----------
        filepath_or_buffer : str, path object or file-like object
            `filepath_or_buffer` parameter of read_csv function.
        read_csv_kwargs : dict
            Parameters of read_csv function.
        compression_infered : str
            Inferred `compression` parameter of read_csv function.

        Returns
        -------
        bool
            Whether passed parameters are supported or not.
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

        return True

    @classmethod
    def _define_index(
        cls,
        index_ids: list,
        index_name: str,
    ) -> Tuple[IndexColType, list]:
        """
        Compute the resulting DataFrame index and index lengths for each of partitions.

        Parameters
        ----------
        index_ids : list
            Array with references to the partitions index objects.
        index_name : str
            Name that should be assigned to the index if `index_col`
            is not provided.

        Returns
        -------
        new_index : IndexColType
            Index that should be passed to the new_frame constructor.
        row_lengths : list
            Partitions rows lengths.
        """
        index_objs = cls.materialize(index_ids)
        if len(index_objs) == 0 or isinstance(index_objs[0], int):
            row_lengths = index_objs
            new_index = pandas.RangeIndex(sum(index_objs))
        else:
            row_lengths = [len(o) for o in index_objs]
            new_index = index_objs[0].append(index_objs[1:])
            new_index.name = index_name

        return new_index, row_lengths

    @classmethod
    def _get_new_qc(
        cls,
        partition_ids: list,
        index_ids: list,
        dtypes_ids: list,
        index_col: IndexColType,
        index_name: str,
        column_widths: list,
        column_names: ColumnNamesTypes,
        skiprows_md: Union[Sequence, callable, None] = None,
        header_size: int = None,
        **kwargs,
    ):
        """
        Get new query compiler from data received from workers.

        Parameters
        ----------
        partition_ids : list
            Array with references to the partitions data.
        index_ids : list
            Array with references to the partitions index objects.
        dtypes_ids : list
            Array with references to the partitions dtypes objects.
        index_col : IndexColType
            `index_col` parameter of `read_csv` function.
        index_name : str
            Name that should be assigned to the index if `index_col`
            is not provided.
        column_widths : list
            Number of columns in each partition.
        column_names : ColumnNamesTypes
            Array with columns names.
        skiprows_md : array-like or callable, optional
            Specifies rows to skip.
        header_size : int, default: 0
            Number of rows, that occupied by header.
        **kwargs : dict
            Parameters of `read_csv` function needed for postprocessing.

        Returns
        -------
        new_query_compiler : BaseQueryCompiler
            New query compiler, created from `new_frame`.
        """
        new_index, row_lengths = cls._define_index(index_ids, index_name)
        # Compute dtypes by getting collecting and combining all of the partitions. The
        # reported dtypes from differing rows can be different based on the inference in
        # the limited data seen by each worker. We use pandas to compute the exact dtype
        # over the whole column for each column. The index is set below.
        dtypes = cls.get_dtypes(dtypes_ids) if len(dtypes_ids) > 0 else None
        # Compose modin partitions from `partition_ids`
        partition_ids = cls.build_partition(partition_ids, row_lengths, column_widths)

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
        if skiprows_md is not None:
            # skip rows that passed as array or callable
            nrows = kwargs.get("nrows", None)
            index_range = pandas.RangeIndex(len(new_query_compiler.index))
            if is_list_like(skiprows_md):
                new_query_compiler = new_query_compiler.view(
                    index=index_range.delete(skiprows_md)
                )
            elif callable(skiprows_md):
                mod_index = skiprows_md(index_range)
                mod_index = (
                    mod_index
                    if isinstance(mod_index, np.ndarray)
                    else mod_index.to_numpy("bool")
                )
                view_idx = index_range[~mod_index]
                new_query_compiler = new_query_compiler.view(index=view_idx)
            else:
                raise TypeError(
                    f"Not acceptable type of `skiprows` parameter: {type(skiprows_md)}"
                )

            if not isinstance(new_query_compiler.index, pandas.MultiIndex):
                new_query_compiler = new_query_compiler.reset_index(drop=True)

            if nrows:
                new_query_compiler = new_query_compiler.view(
                    pandas.RangeIndex(len(new_query_compiler.index))[:nrows]
                )
        if index_col is None:
            new_query_compiler._modin_frame.synchronize_labels(axis=0)

        return new_query_compiler

    @classmethod
    def _manage_skiprows_parameter(
        cls,
        skiprows: Union[int, Sequence[int], Callable, None] = None,
        header_size: int = 0,
    ) -> Tuple[Union[int, Sequence, Callable], bool, int]:
        """
        Manage read_csv `skiprows` parameter.

        Change `skiprows` parameter in the way Modin could more optimally
        process it. `csv_dispatcher` have two mechanisms of rows skipping:

        1) During file partitioning (setting of file limits that should be read
        by each partition) exact rows can be excluded from partitioning scope,
        thus they won't be read at all and can be considered as skipped. This is
        the most effective way of rows skipping (since it doesn't require any
        actual data reading and postprocessing), but in this case `skiprows`
        parameter can be an integer only. When it possible Modin always uses
        this approach by setting of `skiprows_partitioning` return value.

        2) Rows for skipping can be dropped after full dataset import. This is
        more expensive way since it requires extra IO work and postprocessing
        afterwards, but `skiprows` parameter can be of any non-integer type
        supported by `pandas.read_csv` `skiprows` parameter. These rows is
        specified by setting of `skiprows_md` return value.

        In some cases, if `skiprows` is uniformly distributed array (e.g. [1,2,3]),
        `skiprows` can be "squashed" and represented as integer to make a fastpath.
        If there is a gap between the first row for skipping and the last line of
        the header (that will be skipped too), then assign to read this gap first
        (assign the first partition to read these rows be setting of `pre_reading`
        return value). See `Examples` section for details.

        Parameters
        ----------
        skiprows : int, array or callable, optional
            Original `skiprows` parameter of read_csv function.
        header_size : int, default: 0
            Number of rows that are used by header.

        Returns
        -------
        skiprows_md : int, array or callable
            Updated skiprows parameter. If `skiprows` is an array, this
            array will be sorted and. Also parameter will be aligned to
            actual data in the `query_compiler` (which, for example,
            doesn't contain header rows)
        pre_reading : int
            The number of rows that should be read before data file
            splitting for further reading (the number of rows for
            the first partition).
        skiprows_partitioning : int
            The number of rows that should be skipped virtually (skipped during
            data file partitioning).

        Examples
        --------
        Let's consider case when `header`="infer" and `skiprows`=[3,4,5]. In
        this specific case fastpath can be done since `skiprows` is uniformly
        distributed array, so we can "squash" it to integer and set
        `skiprows_partitioning`=3. But if no additional action will be done,
        these three rows will be skipped right after header line, that corresponds
        to `skiprows`=[1,2,3]. Now, to avoid this discrepancy, we need to assign
        the first partition to read data between header line and the first
        row for skipping by setting of `pre_reading` parameter, so setting
        `pre_reading`=2. During data file partitiong, these lines will be assigned
        for reading for the first partition, and then file position will be set at
        the beginning of rows that should be skipped by `skiprows_partitioning`.
        After skipping of these rows, the rest data will be divided between the
        rest of partitions, see rows assignement below:

        0 - header line (skip during partitioning)
        1 - pre_reading (assign to read by the first partition)
        2 - pre_reading (assign to read by the first partition)
        3 - skiprows_partitioning (skip during partitioning)
        4 - skiprows_partitioning (skip during partitioning)
        5 - skiprows_partitioning (skip during partitioning)
        6 - data to partition (divide between the rest of partitions)
        7 - data to partition (divide between the rest of partitions)
        """
        pre_reading = 0
        skiprows_partitioning = 0
        skiprows_md = 0
        if isinstance(skiprows, int):
            skiprows_partitioning = skiprows
        elif is_list_like(skiprows):
            skiprows_md = np.sort(skiprows)
            if np.all(np.diff(skiprows_md) == 1):
                # `skiprows` is uniformly distributed array.
                pre_reading = (
                    skiprows_md[0] - header_size if skiprows_md[0] > header_size else 0
                )
                skiprows_partitioning = len(skiprows_md)
                skiprows_md = 0
            else:
                skiprows_md = skiprows_md - header_size
        elif callable(skiprows):

            def skiprows_func(x):
                return skiprows(x + header_size)

            skiprows_md = skiprows_func

        return skiprows_md, pre_reading, skiprows_partitioning
