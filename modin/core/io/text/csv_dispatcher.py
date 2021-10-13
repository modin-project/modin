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

from csv import QUOTE_NONE

import pandas
import pandas._libs.lib as lib

from modin.config import NPartitions
from modin.core.io.text.text_file_dispatcher import TextFileDispatcher


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

        use_modin_impl = cls._read_csv_check_support(filepath_or_buffer, kwargs)
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

        with OpenFile(
            filepath_or_buffer_md,
            "rb",
            compression_infered,
        ) as f:
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
