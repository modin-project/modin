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
            partition_id = _read_sql_with_offset_pandas_on_ray._remote(
                args=(
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
                ),
                num_returns=num_splits + 1,
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
