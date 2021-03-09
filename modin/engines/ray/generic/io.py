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

import io
import pandas

from modin.engines.base.io import BaseIO
from ray.util.queue import Queue
from ray import wait


class RayIO(BaseIO):
    @classmethod
    def to_sql(cls, qc, **kwargs):
        """Write records stored in a DataFrame to a SQL database.
        Args:
            qc: the query compiler of the DF that we want to run to_sql on
            kwargs: parameters for pandas.to_sql(**kwargs)
        """
        # we first insert an empty DF in order to create the full table in the database
        # This also helps to validate the input against pandas
        # we would like to_sql() to complete only when all rows have been inserted into the database
        # since the mapping operation is non-blocking, each partition will return an empty DF
        # so at the end, the blocking operation will be this empty DF to_pandas

        empty_df = qc.getitem_row_array([0]).to_pandas().head(0)
        empty_df.to_sql(**kwargs)
        # so each partition will append its respective DF
        kwargs["if_exists"] = "append"
        columns = qc.columns

        def func(df):
            df.columns = columns
            df.to_sql(**kwargs)
            return pandas.DataFrame()

        result = qc._modin_frame._apply_full_axis(1, func, new_index=[], new_columns=[])
        # blocking operation
        result.to_pandas()

    @staticmethod
    def _to_csv_check_support(kwargs):
        path_or_buf = kwargs["path_or_buf"]
        compression = kwargs["compression"]
        if not isinstance(path_or_buf, str):
            return False
        # case when the pointer is placed at the beginning of the file.
        if "r" in kwargs["mode"] and "+" in kwargs["mode"]:
            return False
        # encodings with BOM don't support;
        # instead of one mark in result bytes we will have them by the number of partitions
        # so we should fallback in pandas for `utf-16`, `utf-32` with all aliases, in instance
        # (`utf_32_be`, `utf_16_le` and so on)
        if kwargs["encoding"] is not None:
            encoding = kwargs["encoding"].lower()
            if "u" in encoding or "utf" in encoding:
                if "16" in encoding or "32" in encoding:
                    return False
        if compression is None or not compression == "infer":
            return False
        if any((path_or_buf.endswith(ext) for ext in [".gz", ".bz2", ".zip", ".xz"])):
            return False
        return True

    @classmethod
    def to_csv(cls, qc, **kwargs):
        if not cls._to_csv_check_support(kwargs):
            return BaseIO.to_csv(qc, **kwargs)

        # The partition id will be added to the queue, for which the moment
        # of writing to the file has come
        queue = Queue(maxsize=1)

        def func(df, **kw):
            if kw["partition_idx"] != 0:
                # we need to create a new file only for first recording
                # all the rest should be recorded in appending mode
                if "w" in kwargs["mode"]:
                    kwargs["mode"] = kwargs["mode"].replace("w", "a")
                # It is enough to write the header for the first partition
                kwargs["header"] = False

            # for parallelization purposes, each partition is written to an intermediate buffer
            path_or_buf = kwargs["path_or_buf"]
            is_binary = "b" in kwargs["mode"]
            if is_binary:
                kwargs["path_or_buf"] = io.BytesIO()
            else:
                kwargs["path_or_buf"] = io.StringIO()
            df.to_csv(**kwargs)
            content = kwargs["path_or_buf"].getvalue()

            # each process waits for its turn to write to a file;
            # in case of violation of the order of receiving messages from the queue,
            # the message is placed back
            while True:
                get_value = queue.get(block=True)
                if get_value == kw["partition_idx"]:
                    break
                queue.put(get_value)

            # preparing to write data from the buffer to a file
            open_kwargs = {"mode": kwargs["mode"]}
            if not is_binary:
                # in the buffer, newline symbols have already been translated
                # for the current operating system;
                # in the process of writing the buffer to the file,
                # newline symbols must be left unchanged
                open_kwargs["newline"] = ""
            with open(path_or_buf, **open_kwargs) as _f:
                _f.write(content)
            # signal that the next process can start writing to the file
            queue.put(get_value + 1)

            # used for synchronization purposes
            return 0

        # signaling that the partition with id==0 can be written to the file
        queue.put(0)
        result = qc._modin_frame._frame_mgr_cls.map_axis_partitions(
            axis=1,
            partitions=qc._modin_frame._partitions,
            map_func=func,
            keep_partitioning=True,
            lengths=None,
            enumerate_partitions=True,
        )

        # pending completion
        for rows in result:
            for partition in rows:
                wait([partition.oid])
