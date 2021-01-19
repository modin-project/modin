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
Module houses experimental IO classes and parser functions needed for these classes.

Any function or class can be considered experimental API if it is not strictly replicating existent
backend API, even if it is only extending the API.
"""

import os
import numpy as np
import pandas
import warnings

from modin.backends.pandas.parsers import _split_result_for_readers, PandasCSVGlobParser
from modin.backends.pandas.query_compiler import PandasQueryCompiler
from modin.engines.ray.pandas_on_ray.io import PandasOnRayIO
from modin.engines.base.io import CSVGlobDispatcher
from modin.engines.ray.pandas_on_ray.frame.data import PandasOnRayFrame
from modin.engines.ray.pandas_on_ray.frame.partition import PandasOnRayFramePartition
from modin.engines.ray.task_wrapper import RayTask
from modin.config import NPartitions

import ray


# Ray functions are not detected by codecov (thus pragma: no cover)
@ray.remote
def _read_parquet_columns(path, columns, num_splits, kwargs):  # pragma: no cover
    """
    Read columns from Parquet file into a ``pandas.DataFrame`` using Ray task.

    Parameters
    ----------
    path : str or List[str]
        The path of the Parquet file.
    columns : List[str]
        The list of column names to read.
    num_splits : int
        The number of partitions to split the column into.
    kwargs : dict
        Keyward arguments to pass into ``pyarrow.parquet.read`` function.

    Returns
    -------
    list
        A list containing the splitted ``pandas.DataFrame``-s and the Index as the last
        element.

    Notes
    -----
    ``pyarrow.parquet.read`` is used internally as the parse function.
    """
    import pyarrow.parquet as pq

    df = (
        pq.ParquetDataset(path, **kwargs)
        .read(columns=columns, use_pandas_metadata=True)
        .to_pandas()
    )
    df = df[columns]
    # Append the length of the index here to build it externally
    return _split_result_for_readers(0, num_splits, df) + [len(df.index)]


class ExperimentalPandasOnRayIO(PandasOnRayIO):
    """
    Class for handling experimental IO functionality with pandas backend and Ray engine.

    ``ExperimentalPandasOnRayIO`` inherits some util functions and unmodified IO functions
    from ``PandasOnRayIO`` class.
    """

    build_args = dict(
        frame_partition_cls=PandasOnRayFramePartition,
        query_compiler_cls=PandasQueryCompiler,
        frame_cls=PandasOnRayFrame,
    )
    read_csv_glob = type(
        "", (RayTask, PandasCSVGlobParser, CSVGlobDispatcher), build_args
    )._read
    read_parquet_remote_task = _read_parquet_columns
    format_modin_pickle_files = 0.1

    @classmethod
    def read_sql(
        cls,
        sql,
        con,
        index_col=None,
        coerce_float=True,
        params=None,
        parse_dates=None,
        columns=None,
        chunksize=None,
        partition_column=None,
        lower_bound=None,
        upper_bound=None,
        max_sessions=None,
    ):
        """
        Read SQL query or database table into a DataFrame.

        Parameters
        ----------
        sql : str or SQLAlchemy Selectable (select or text object)
            SQL query to be executed or a table name.
        con : SQLAlchemy connectable or str
             Connection to database (sqlite3 connections are not supported).
        index_col : str or list of str, optional
            Column(s) to set as index(MultiIndex).
        coerce_float : bool, default: True
            Attempts to convert values of non-string, non-numeric objects
            (like decimal.Decimal) to floating point, useful for SQL result sets.
        params : list, tuple or dict, optional
            List of parameters to pass to ``execute`` method. The syntax used
            to pass parameters is database driver dependent. Check your
            database driver documentation for which of the five syntax styles,
            described in PEP 249's paramstyle, is supported.
        parse_dates : list or dict, optional
            The behavior is as follows:

            - List of column names to parse as dates.
            - Dict of `{column_name: format string}` where format string is
              strftime compatible in case of parsing string times, or is one of
              (D, s, ns, ms, us) in case of parsing integer timestamps.
            - Dict of `{column_name: arg dict}`, where the arg dict corresponds
              to the keyword arguments of ``pandas.to_datetime``.
              Especially useful with databases without native Datetime support,
              such as SQLite.
        columns : list, optional
            List of column names to select from SQL table (only used when reading a
            table).
        chunksize : int, optional
            If specified, return an iterator where `chunksize` is the number of rows
            to include in each chunk.
        partition_column : str, optional
            Column name used for data partitioning between the workers
            (MUST be an INTEGER column).
        lower_bound : int, optional
            The minimum value to be requested from the `partition_column`.
        upper_bound : int, optional
            The maximum value to be requested from the `partition_column`.
        max_sessions : int, optional
            The maximum number of simultaneous connections allowed to use.

        Returns
        -------
        BaseQueryCompiler
            A new query compiler with imported data for further processing.
        """
        from .sql import is_distributed, get_query_info

        if not is_distributed(partition_column, lower_bound, upper_bound):
            warnings.warn("Defaulting to Modin core implementation")
            return PandasOnRayIO.read_sql(
                sql,
                con,
                index_col,
                coerce_float=coerce_float,
                params=params,
                parse_dates=parse_dates,
                columns=columns,
                chunksize=chunksize,
            )
        #  starts the distributed alternative
        cols_names, query = get_query_info(sql, con, partition_column)
        num_parts = min(NPartitions.get(), max_sessions if max_sessions else 1)
        num_splits = min(len(cols_names), num_parts)
        diff = (upper_bound - lower_bound) + 1
        min_size = diff // num_parts
        rest = diff % num_parts
        partition_ids = []
        index_ids = []
        end = lower_bound - 1
        for part in range(num_parts):
            if rest:
                size = min_size + 1
                rest -= 1
            else:
                size = min_size
            start = end + 1
            end = start + size - 1
            partition_id = _read_sql_with_offset_pandas_on_ray.options(
                num_returns=num_splits + 1
            ).remote(
                partition_column,
                start,
                end,
                num_splits,
                query,
                con,
                index_col,
                coerce_float,
                params,
                parse_dates,
                columns,
                chunksize,
            )
            partition_ids.append(
                [PandasOnRayFramePartition(obj) for obj in partition_id[:-1]]
            )
            index_ids.append(partition_id[-1])
        new_index = pandas.RangeIndex(sum(ray.get(index_ids)))
        new_query_compiler = cls.query_compiler_cls(
            cls.frame_cls(np.array(partition_ids), new_index, cols_names)
        )
        new_query_compiler._modin_frame.synchronize_labels(axis=0)
        return new_query_compiler

    @classmethod
    def create_header_pattern(cls, num_partitions):
        locations_pattern = " ".join(["{:12}"] * num_partitions * 2)
        # format: - "modin format of pickle files XXX: num_partitions: XXX,
        # locations: position + lengths for all partitions
        header_pattern = "modin_format_of_pickle_files {:3} {:3} " + locations_pattern
        return header_pattern

    @classmethod
    def get_header_size(cls, num_partitions):
        # starts with `modin_format_of_pickle_files` - 28 bytes in utf-8 encoding
        # format - 3 bytes, num_splits - 3 bytes
        # for partition: position - 12 bytes, lengths - 12 bytes
        # all numbers splits by whitespace symbol
        count_whitespaces = (3 + num_partitions * 2) - 1
        return 28 + 6 + 24 * num_partitions + count_whitespaces

    @classmethod
    def to_pickle(cls, qc, **kwargs):
        num_partitions = cls.frame_cls._frame_mgr_cls._compute_num_partitions()
        header_size = cls.get_header_size(num_partitions)

        def func(df, **kw):
            partition_idx = kw["partition_idx"]
            if partition_idx == 0:
                with open(kwargs["path"], mode="wb") as dst:
                    # dummy header
                    dst.write(b"X" * header_size)
                    kwargs["path"] = dst
                    df.to_pickle(**kwargs)
            else:
                kwargs["path"] = kwargs["path"] + str(partition_idx)
                df.to_pickle(**kwargs)
            return pandas.DataFrame()

        result = qc._modin_frame._apply_full_axis(
            1, func, new_index=[], new_columns=[], enumerate_partitions=True
        )

        import shutil

        # import pdb;pdb.set_trace()
        header_pattern = cls.create_header_pattern(num_partitions)
        # blocking operation
        result.to_pandas()

        locations = []
        with open(kwargs["path"], mode="ab+") as dst:
            # take into account first partition
            locations.append(header_size)
            locations.append(dst.tell() - header_size)
            for idx in range(1, num_partitions):
                cur_pos = dst.tell()
                with open(kwargs["path"] + str(idx), mode="rb") as src:
                    shutil.copyfileobj(src, dst)
                os.remove(kwargs["path"] + str(idx))
                locations.append(cur_pos)
                locations.append(dst.tell() - cur_pos)

        header = header_pattern.format(
            cls.format_modin_pickle_files, num_partitions, *locations
        )
        with open(kwargs["path"], mode="rb+") as dst:
            dst.write(header.encode())

    @classmethod
    def read_pickle(cls, filepath_or_buffer, compression="infer"):
        if not isinstance(filepath_or_buffer, str):
            warnings.warn("Defaulting to Modin core implementation")
            return PandasOnRayIO.read_pickle(
                filepath_or_buffer,
                compression=compression,
            )

        partition_ids = []
        lengths_ids = []
        widths_ids = []

        import re

        locations = []
        header_size = cls.get_header_size(num_partitions=0)
        with open(filepath_or_buffer, mode="rb") as src:
            header = re.split(b"\s+", src.read(header_size))
            num_partitions = int(header[2])
            full_header_size = cls.get_header_size(num_partitions)
            locations = re.split(
                b"\s+", src.read(full_header_size - header_size).strip(b" ")
            )

        locations = [int(x) for x in locations]
        for idx_file in range(num_partitions):
            partition_id = _read_pickle_files_in_folder._remote(
                args=(
                    filepath_or_buffer,
                    compression,
                    locations[idx_file * 2],
                    locations[1 + idx_file * 2],
                ),
                num_returns=3,
            )
            partition_ids.append(partition_id[:-2])
            lengths_ids.append(partition_id[-2])
            widths_ids.append(partition_id[-1])

        lengths = ray.get(lengths_ids)
        widths = ray.get(widths_ids)
        # while num_splits is 1, need only one value
        partition_ids = build_partition(partition_ids, lengths, [widths[0]])

        new_index = cls.frame_cls._frame_mgr_cls.get_indices(
            0, partition_ids, lambda df: df.axes[0]
        )
        new_columns = cls.frame_cls._frame_mgr_cls.get_indices(
            1, partition_ids, lambda df: df.axes[1]
        )

        return cls.query_compiler_cls(
            cls.frame_cls(partition_ids, new_index, new_columns)
        )


def build_partition(partition_ids, row_lengths, column_widths):
    return np.array(
        [
            [
                PandasOnRayFramePartition(
                    partition_ids[i][j],
                    length=row_lengths[i],
                    width=column_widths[j],
                )
                for j in range(len(partition_ids[i]))
            ]
            for i in range(len(partition_ids))
        ]
    )


@ray.remote
def _read_pickle_files_in_folder(
    filepath: str,
    compression: str,
    position: int,
    length: int,
):  # pragma: no cover
    """
    Use a Ray task to read a pickle file under filepath folder.

    TODO: add parameters descriptors
    """
    with open(filepath, mode="rb") as src:
        src.seek(position)
        # read_pickle can read several pickled objects from file
        # so we can work with file instead of BytesIO
        df = pandas.read_pickle(src, compression)

    length = len(df)
    width = len(df.columns)
    num_splits = 1
    return _split_result_for_readers(1, num_splits, df) + [length, width]


# Ray functions are not detected by codecov (thus pragma: no cover)
@ray.remote
def _read_sql_with_offset_pandas_on_ray(
    partition_column,
    start,
    end,
    num_splits,
    sql,
    con,
    index_col=None,
    coerce_float=True,
    params=None,
    parse_dates=None,
    columns=None,
    chunksize=None,
):  # pragma: no cover
    """
    Read a chunk of SQL query or table into a pandas DataFrame using Ray task.

    Parameters
    ----------
    partition_column : str
        Column name used for data partitioning between the workers.
    start : int
        Lowest value to request from the `partition_column`.
    end : int
        Highest value to request from the `partition_column`.
    num_splits : int
        The number of partitions to split the column into.
    sql : str or SQLAlchemy Selectable (select or text object)
        SQL query to be executed or a table name.
    con : SQLAlchemy connectable or str
        Connection to database (sqlite3 connections are not supported).
    index_col : str or list of str, optional
        Column(s) to set as index(MultiIndex).
    coerce_float : bool, default: True
        Attempts to convert values of non-string, non-numeric objects
        (like decimal.Decimal) to floating point, useful for SQL result sets.
    params : list, tuple or dict, optional
        List of parameters to pass to ``execute`` method. The syntax used
        to pass parameters is database driver dependent. Check your
        database driver documentation for which of the five syntax styles,
        described in PEP 249's paramstyle, is supported.
    parse_dates : list or dict, optional
        The behavior is as follows:

        - List of column names to parse as dates.
        - Dict of `{column_name: format string}` where format string is
          strftime compatible in case of parsing string times, or is one of
          (D, s, ns, ms, us) in case of parsing integer timestamps.
        - Dict of `{column_name: arg dict}`, where the arg dict corresponds
          to the keyword arguments of ``pandas.to_datetime``
          Especially useful with databases without native Datetime support,
          such as SQLite.
    columns : list, optional
        List of column names to select from SQL table (only used when reading a
        table).
    chunksize : int, optional
        If specified, return an iterator where `chunksize` is the number of rows
        to include in each chunk.

    Returns
    -------
    list
        List with splitted read results and it's metadata (index, dtypes, etc.).
    """
    from .sql import query_put_bounders

    query_with_bounders = query_put_bounders(sql, partition_column, start, end)
    pandas_df = pandas.read_sql(
        query_with_bounders,
        con,
        index_col=index_col,
        coerce_float=coerce_float,
        params=params,
        parse_dates=parse_dates,
        columns=columns,
        chunksize=chunksize,
    )
    index = len(pandas_df)
    return _split_result_for_readers(1, num_splits, pandas_df) + [index]
