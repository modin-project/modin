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

"""Module houses `ExperimentalCustomTextDispatcher` class, that is used for reading custom text files."""

import pandas
from pandas.io.common import stringify_path

from modin.config import NPartitions
from modin.core.io.file_dispatcher import OpenFile
from modin.core.io.text.text_file_dispatcher import TextFileDispatcher


class ExperimentalCustomTextDispatcher(TextFileDispatcher):
    """Class handles utils for reading custom text files."""

    @classmethod
    def _read(cls, filepath_or_buffer, columns, custom_parser, **kwargs):
        r"""
        Read data from `filepath_or_buffer` according to the passed `read_custom_text` `kwargs` parameters.

        Parameters
        ----------
        filepath_or_buffer : str, path object or file-like object
            `filepath_or_buffer` parameter of `read_custom_text` function.
        columns : list or callable(file-like object, \*\*kwargs -> list
            Column names of list type or callable that create column names from opened file
            and passed `kwargs`.
        custom_parser : callable(file-like object, \*\*kwargs -> pandas.DataFrame
            Function that takes as input a part of the `filepath_or_buffer` file loaded into
            memory in file-like object form.
        **kwargs : dict
            Parameters of `read_custom_text` function.

        Returns
        -------
        BaseQueryCompiler
            Query compiler with imported data for further processing.
        """
        filepath_or_buffer = stringify_path(filepath_or_buffer)
        filepath_or_buffer_md = (
            cls.get_path(filepath_or_buffer)
            if isinstance(filepath_or_buffer, str)
            else cls.get_path_or_buffer(filepath_or_buffer)
        )
        compression_infered = cls.infer_compression(
            filepath_or_buffer, kwargs["compression"]
        )

        with OpenFile(filepath_or_buffer_md, "rb", compression_infered) as f:
            splits, _ = cls.partitioned_file(
                f,
                num_partitions=NPartitions.get(),
                is_quoting=kwargs.pop("is_quoting"),
                nrows=kwargs["nrows"],
            )

        if callable(columns):
            with OpenFile(filepath_or_buffer_md, "rb", compression_infered) as f:
                columns = columns(f, **kwargs)
        if not isinstance(columns, pandas.Index):
            columns = pandas.Index(columns)

        empty_pd_df = pandas.DataFrame(columns=columns)
        index_name = empty_pd_df.index.name
        column_widths, num_splits = cls._define_metadata(empty_pd_df, columns)

        # kwargs that will be passed to the workers
        partition_kwargs = dict(
            kwargs,
            fname=filepath_or_buffer_md,
            num_splits=num_splits,
            nrows=None,
            compression=compression_infered,
        )

        partition_ids, index_ids, dtypes_ids = cls._launch_tasks(
            splits, callback=custom_parser, **partition_kwargs
        )

        new_query_compiler = cls._get_new_qc(
            partition_ids=partition_ids,
            index_ids=index_ids,
            dtypes_ids=dtypes_ids,
            index_col=None,
            index_name=index_name,
            column_widths=column_widths,
            column_names=columns,
            nrows=kwargs["nrows"],
        )
        return new_query_compiler
