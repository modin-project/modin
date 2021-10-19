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

"""The module holds base class implementing required I/O over Ray."""

import io
import pandas

from modin.core.io import BaseIO
from ray.util.queue import Queue
from ray import wait


class RayIO(BaseIO):
    """Base class for doing I/O operations over Ray."""

    @classmethod
    def to_sql(cls, qc, **kwargs):
        """
        Write records stored in the `qc` to a SQL database.

        Parameters
        ----------
        qc : BaseQueryCompiler
            The query compiler of the Modin dataframe that we want to run ``to_sql`` on.
        **kwargs : dict
            Parameters for ``pandas.to_sql(**kwargs)``.
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
            """
            Override column names in the wrapped dataframe and convert it to SQL.

            Notes
            -----
            This function returns an empty ``pandas.DataFrame`` because ``apply_full_axis``
            expects a Frame object as a result of operation (and ``to_sql`` has no dataframe result).
            """
            df.columns = columns
            df.to_sql(**kwargs)
            return pandas.DataFrame()

        result = qc._modin_frame.apply_full_axis(1, func, new_index=[], new_columns=[])
        # FIXME: we should be waiting for completion less expensievely, maybe use _modin_frame.materialize()?
        result.to_pandas()  # blocking operation

    @staticmethod
    def _to_csv_check_support(kwargs):
        """
        Check if parallel version of ``to_csv`` could be used.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed to ``.to_csv()``.

        Returns
        -------
        bool
            Whether parallel version of ``to_csv`` is applicable.
        """
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
        """
        Write records stored in the `qc` to a CSV file.

        Parameters
        ----------
        qc : BaseQueryCompiler
            The query compiler of the Modin dataframe that we want to run ``to_csv`` on.
        **kwargs : dict
            Parameters for ``pandas.to_csv(**kwargs)``.
        """
        if not cls._to_csv_check_support(kwargs):
            return BaseIO.to_csv(qc, **kwargs)

        # The partition id will be added to the queue, for which the moment
        # of writing to the file has come
        queue = Queue(maxsize=1)

        def func(df, **kw):
            """
            Dump a chunk of rows as csv, then save them to target maintaining order.

            Parameters
            ----------
            df : pandas.DataFrame
                A chunk of rows to write to a CSV file.
            **kw : dict
                Arguments to pass to ``pandas.to_csv(**kw)`` plus an extra argument
                `partition_idx` serving as chunk index to maintain rows order.
            """
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
            kwargs["path_or_buf"].close()

            # each process waits for its turn to write to a file;
            # in case of violation of the order of receiving messages from the queue,
            # the message is placed back
            while True:
                get_value = queue.get(block=True)
                if get_value == kw["partition_idx"]:
                    break
                queue.put(get_value)

            # preparing to write data from the buffer to a file
            with pandas.io.common.get_handle(
                path_or_buf,
                # in case when using URL in implicit text mode
                # pandas try to open `path_or_buf` in binary mode
                kwargs["mode"] if is_binary else kwargs["mode"] + "t",
                encoding=kwargs["encoding"],
                errors=kwargs["errors"],
                compression=kwargs["compression"],
                storage_options=kwargs["storage_options"],
                is_text=False,
            ) as handles:
                handles.handle.write(content)

            # signal that the next process can start writing to the file
            queue.put(get_value + 1)

            # used for synchronization purposes
            return pandas.DataFrame()

        # signaling that the partition with id==0 can be written to the file
        queue.put(0)
        result = qc._modin_frame._partition_mgr_cls.map_axis_partitions(
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

    @staticmethod
    def _to_parquet_check_support(kwargs):
        """
        Check if parallel version of ``to_parquet`` could be used.

        Parameters
        ----------
        kwargs : dict
            Keyword arguments passed to ``.to_parquet()``.

        Returns
        -------
        bool
            Whether parallel version of ``to_parquet`` is applicable.
        """
        path = kwargs["path"]
        engine = kwargs["engine"]
        compression = kwargs["compression"]
        if not isinstance(path, str):
            return False
        if any((path.endswith(ext) for ext in [".gz", ".bz2", ".zip", ".xz"])):
            return False
        if engine != "auto":
            return False
        if compression is None or not compression == "snappy":
            return False
        return True

    @classmethod
    def to_parquet(cls, qc, **kwargs):
        """
        Write a ``DataFrame`` to the binary parquet format.

        Parameters
        ----------
        qc : BaseQueryCompiler
            The query compiler of the Modin dataframe that we want to run ``to_parquet`` on.
        **kwargs : dict
            Parameters for ``pandas.to_parquet(**kwargs)``.
        """
        if not cls._to_parquet_check_support(kwargs):
            return BaseIO.to_parquet(qc, **kwargs)

        def func(df, **kw):
            """
            Dump a chunk of rows as parquet, then save them to target maintaining order.

            Parameters
            ----------
            df : pandas.DataFrame
                A chunk of rows to write to a parquet file.
            **kw : dict
                Arguments to pass to ``pandas.to_parquet(**kw)`` plus an extra argument
                `partition_idx` serving as chunk index to maintain rows order.
            """
            from modin.distributed.dataframe.pandas import unwrap_partitions
            partitions = unwrap_partitions(df, axis=0)
            for i, _ in enumerate(partitions):
                output_path = kwargs["path"]
                kwargs["path"] = f"{output_path}/part-{i:04d}.snappy.parquet"
                df.to_parquet(**kwargs)

        # signaling that the partition with id==0 can be written to the file
        qc._modin_frame._partition_mgr_cls.map_axis_partitions(
            axis=0,
            partitions=qc._modin_frame._partitions,
            map_func=func,
            keep_partitioning=True,
            lengths=None,
            enumerate_partitions=True,
        )
