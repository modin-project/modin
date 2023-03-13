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

"""Module houses `SQLExperimentalDispatcher` class."""

import warnings

import pandas
import numpy as np

from modin.core.io import SQLDispatcher
from modin.core.execution.dask.implementations.pandas_on_dask.io import PandasOnDaskIO
from modin.config import NPartitions


class SQLExperimentalDispatcher(SQLDispatcher):
    """Class handles experimental utils for reading SQL queries or database tables."""

    __read_sql_with_offset = None

    @classmethod
    def preprocess_func(cls):  # noqa: RT01
        """Prepare a function for transmission to remote workers."""
        if cls.__read_sql_with_offset is None:
            from modin.experimental.core.io.sql.utils import read_sql_with_offset

            # cls.__read_sql_with_offset = cls.put(read_sql_with_offset)
            cls.__read_sql_with_offset = read_sql_with_offset
        return cls.__read_sql_with_offset

    @classmethod
    def _read(
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
        from modin.experimental.core.io.sql.utils import (
            is_distributed,
            get_query_info,
        )

        if not is_distributed(partition_column, lower_bound, upper_bound):
            warnings.warn("Defaulting to Modin core implementation")
            return cls.base_io.read_sql(
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
        func = cls.preprocess_func()
        for part in range(num_parts):
            if rest:
                size = min_size + 1
                rest -= 1
            else:
                size = min_size
            start = end + 1
            end = start + size - 1
            partition_id = cls.deploy(
                func,
                f_args=(
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
                pure=False,
            )
            partition_ids.append(
                [cls.frame_partition_cls(obj) for obj in partition_id[:-1]]
            )
            index_ids.append(partition_id[-1])
        new_index = pandas.RangeIndex(sum(cls.materialize(index_ids)))
        new_query_compiler = cls.query_compiler_cls(
            cls.frame_cls(np.array(partition_ids), new_index, cols_names)
        )
        new_query_compiler._modin_frame.synchronize_labels(axis=0)
        return new_query_compiler
