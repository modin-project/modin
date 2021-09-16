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

"""Module houses `CustomTextExperimentalDispatcher` class, that is used for reading custom text files."""

from modin.core.io.file_dispatcher import OpenFile
from modin.core.io.text.text_file_dispatcher import TextFileDispatcher
import pandas
import numpy as np

from modin.config import NPartitions


class CustomTextExperimentalDispatcher(TextFileDispatcher):
    """Class handles utils for reading custom text files."""

    @classmethod
    def _read(cls, path_or_buf, **kwargs):
        """
        Read data from `path_or_buf` according to the passed `read_custom_text` `kwargs` parameters.

        Parameters
        ----------
        path_or_buf : str, path object or file-like object
            `path_or_buf` parameter of `read_custom_text` function.
        **kwargs : dict
            Parameters of `read_custom_text` function.

        Returns
        -------
        BaseQueryCompiler
            Query compiler with imported data for further processing.
        """
        if not isinstance(path_or_buf, str):
            raise ValueError("Support only str type of path")

        columns = kwargs.pop("columns", None)
        assert columns is not None and not isinstance(columns, pandas.Index)

        if callable(columns):
            with OpenFile(path_or_buf, "rb", kwargs.get("compression", "infer")) as f:
                columns = columns(f, **kwargs)

        columns = pandas.Index(columns)
        kwargs["columns"] = columns
        empty_pd_df = pandas.DataFrame(columns=columns)

        with OpenFile(path_or_buf, "rb", kwargs.get("compression", "infer")) as f:
            partition_ids = []
            index_ids = []
            dtypes_ids = []

            column_widths, num_splits = cls._define_metadata(empty_pd_df, columns)

            args = {"fname": path_or_buf, "num_splits": num_splits, **kwargs}

            splits = cls.partitioned_file(
                f,
                num_partitions=NPartitions.get(),
                is_quoting=kwargs.pop("is_quoting"),
                nrows=kwargs["nrows"],
            )
            for start, end in splits:
                args.update({"start": start, "end": end})
                partition_id = cls.deploy(cls.parse, num_splits + 3, args)
                partition_ids.append(partition_id[:-3])
                index_ids.append(partition_id[-3])
                dtypes_ids.append(partition_id[-2])

        row_lengths = cls.materialize(index_ids)
        new_index = pandas.RangeIndex(sum(row_lengths))

        dtypes = cls.get_dtypes(dtypes_ids)
        partition_ids = cls.build_partition(partition_ids, row_lengths, column_widths)

        if isinstance(dtypes, pandas.Series):
            dtypes.index = columns
        else:
            dtypes = pandas.Series(dtypes, index=columns)

        new_frame = cls.frame_cls(
            np.array(partition_ids),
            new_index,
            columns,
            row_lengths,
            column_widths,
            dtypes=dtypes,
        )
        new_frame.synchronize_labels(axis=0)
        return cls.query_compiler_cls(new_frame)
