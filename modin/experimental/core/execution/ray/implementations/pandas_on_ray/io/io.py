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
Query Compiler API, even if it is only extending the API.
"""

import numpy as np
import pandas
import warnings

from modin.core.storage_formats.pandas.parsers import (
    _split_result_for_readers,
    PandasCSVGlobParser,
    PandasPickleExperimentalParser,
    CustomTextExperimentalParser,
)
from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.core.execution.ray.implementations.pandas_on_ray.io import PandasOnRayIO
from modin.core.io import (
    CSVGlobDispatcher,
    PickleExperimentalDispatcher,
    CustomTextExperimentalDispatcher,
)
from modin.core.execution.ray.implementations.pandas_on_ray.dataframe import (
    PandasOnRayDataframe,
)
from modin.core.execution.ray.implementations.pandas_on_ray.partitioning import (
    PandasOnRayDataframePartition,
)
from modin.core.execution.ray.common import RayTask
from modin.config import NPartitions

import ray


class ExperimentalPandasOnRayIO(PandasOnRayIO):
    """
    Class for handling experimental IO functionality with pandas storage format and Ray engine.

    ``ExperimentalPandasOnRayIO`` inherits some util functions and unmodified IO functions
    from ``PandasOnRayIO`` class.
    """

    build_args = dict(
        frame_partition_cls=PandasOnRayDataframePartition,
        query_compiler_cls=PandasQueryCompiler,
        frame_cls=PandasOnRayDataframe,
    )
    read_csv_glob = type(
        "", (RayTask, PandasCSVGlobParser, CSVGlobDispatcher), build_args
    )._read
    read_pickle_distributed = type(
        "",
        (RayTask, PandasPickleExperimentalParser, PickleExperimentalDispatcher),
        build_args,
    )._read

    read_custom_text = type(
        "",
        (RayTask, CustomTextExperimentalParser, CustomTextExperimentalDispatcher),
        build_args,
    )._read

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

        The function extended with `Spark-like parameters <https://spark.apache.org/docs/2.0.0/api/R/read.jdbc.html>`_
        such as ``partition_column``, ``lower_bound`` and ``upper_bound``. With these
        parameters, the user will be able to specify how to partition the imported data.

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
                [PandasOnRayDataframePartition(obj) for obj in partition_id[:-1]]
            )
            index_ids.append(partition_id[-1])
        new_index = pandas.RangeIndex(sum(ray.get(index_ids)))
        new_query_compiler = cls.query_compiler_cls(
            cls.frame_cls(np.array(partition_ids), new_index, cols_names)
        )
        new_query_compiler._modin_frame.synchronize_labels(axis=0)
        return new_query_compiler

    @classmethod
    def to_pickle_distributed(cls, qc, **kwargs):
        """
        When `*` in the filename all partitions are written to their own separate file.

        The filenames is determined as follows:
        - if `*` in the filename then it will be replaced by the increasing sequence 0, 1, 2, …
        - if `*` is not the filename, then will be used default implementation.

        Examples #1: 4 partitions and input filename="partition*.pkl.gz", then filenames will be:
        `partition0.pkl.gz`, `partition1.pkl.gz`, `partition2.pkl.gz`, `partition3.pkl.gz`.

        Parameters
        ----------
        qc : BaseQueryCompiler
            The query compiler of the Modin dataframe that we want
            to run ``to_pickle_distributed`` on.
        **kwargs : dict
            Parameters for ``pandas.to_pickle(**kwargs)``.
        """
        if not (
            isinstance(kwargs["filepath_or_buffer"], str)
            and "*" in kwargs["filepath_or_buffer"]
        ) or not isinstance(qc, PandasQueryCompiler):
            warnings.warn("Defaulting to Modin core implementation")
            return PandasOnRayIO.to_pickle(qc, **kwargs)

        def func(df, **kw):
            idx = str(kw["partition_idx"])
            kwargs["path"] = kwargs.pop("filepath_or_buffer").replace("*", idx)
            df.to_pickle(**kwargs)
            return pandas.DataFrame()

        result = qc._modin_frame.broadcast_apply_full_axis(
            1, func, other=None, new_index=[], new_columns=[], enumerate_partitions=True
        )
        result.to_pandas()


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
        List with split read results and it's metadata (index, dtypes, etc.).
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
