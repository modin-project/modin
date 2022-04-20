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

"""Module houses `JSONDispatcher` class, that is used for reading `.json` files."""

from modin.core.io.file_dispatcher import OpenFile
from modin.core.io.text.text_file_dispatcher import TextFileDispatcher
from io import BytesIO
import pandas
import numpy as np

from modin.config import NPartitions


class JSONDispatcher(TextFileDispatcher):
    """Class handles utils for reading `.json` files."""

    @classmethod
    def _read(cls, path_or_buf, **kwargs):
        """
        Read data from `path_or_buf` according to the passed `read_json` `kwargs` parameters.

        Parameters
        ----------
        path_or_buf : str, path object or file-like object
            `path_or_buf` parameter of `read_json` function.
        **kwargs : dict
            Parameters of `read_json` function.

        Returns
        -------
        BaseQueryCompiler
            Query compiler with imported data for further processing.
        """
        path_or_buf = cls.get_path_or_buffer(path_or_buf)
        if isinstance(path_or_buf, str):
            if not cls.file_exists(path_or_buf):
                return cls.single_worker_read(path_or_buf, **kwargs)
            path_or_buf = cls.get_path(path_or_buf)
        elif not cls.pathlib_or_pypath(path_or_buf):
            return cls.single_worker_read(path_or_buf, **kwargs)
        if not kwargs.get("lines", False):
            return cls.single_worker_read(path_or_buf, **kwargs)
        with OpenFile(path_or_buf, "rb") as f:
            columns = pandas.read_json(BytesIO(b"" + f.readline()), lines=True).columns
        kwargs["columns"] = columns
        empty_pd_df = pandas.DataFrame(columns=columns)

        with OpenFile(path_or_buf, "rb", kwargs.get("compression", "infer")) as f:
            column_widths, num_splits = cls._define_metadata(empty_pd_df, columns)
            args = {"fname": path_or_buf, "num_splits": num_splits, **kwargs}
            splits = cls.partitioned_file(
                f,
                num_partitions=NPartitions.get(),
            )
            partition_ids = [None] * len(splits)
            index_ids = [None] * len(splits)
            dtypes_ids = [None] * len(splits)
            for idx, (start, end) in enumerate(splits):
                args.update({"start": start, "end": end})
                *partition_ids[idx], index_ids[idx], dtypes_ids[idx], _ = cls.deploy(
                    cls.parse, num_returns=num_splits + 3, **args
                )
        # partition_id[-1] contains the columns for each partition, which will be useful
        # for implementing when `lines=False`.
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
