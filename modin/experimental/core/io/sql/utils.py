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

"""Utilities for experimental SQL format type IO functions implementations."""

import pandas
import pandas._libs.lib as lib
from sqlalchemy import MetaData, Table, create_engine, inspect, text

from modin.core.storage_formats.pandas.parsers import _split_result_for_readers


def is_distributed(partition_column, lower_bound, upper_bound):
    """
    Check if is possible to distribute a query with the given args.

    Parameters
    ----------
    partition_column : str
        Column name used for data partitioning between the workers.
    lower_bound : int
        The minimum value to be requested from the `partition_column`.
    upper_bound : int
        The maximum value to be requested from the `partition_column`.

    Returns
    -------
    bool
        Whether the given query is distributable or not.
    """
    if (
        (partition_column is not None)
        and (lower_bound is not None)
        and (upper_bound is not None)
    ):
        if upper_bound > lower_bound:
            return True
        raise InvalidArguments("upper_bound must be greater than lower_bound.")
    elif (partition_column is None) and (lower_bound is None) and (upper_bound is None):
        return False
    else:
        raise InvalidArguments(
            "Invalid combination of partition_column, lower_bound, upper_bound."
            + "All these arguments should be passed (distributed) or none of them (standard pandas)."
        )


def is_table(engine, sql):
    """
    Check if given `sql` parameter is a table name.

    Parameters
    ----------
    engine : sqlalchemy.engine.base.Engine
        SQLAlchemy connection engine.
    sql : str
        SQL query to be executed or a table name.

    Returns
    -------
    bool
        Whether `sql` a table name or not.
    """
    return inspect(engine).has_table(sql)


def get_table_metadata(engine, table):
    """
    Extract all useful data from the given table.

    Parameters
    ----------
    engine : sqlalchemy.engine.base.Engine
        SQLAlchemy connection engine.
    table : str
        Table name.

    Returns
    -------
    sqlalchemy.sql.schema.Table
        Extracted metadata.
    """
    metadata = MetaData()
    metadata.reflect(bind=engine, only=[table])
    table_metadata = Table(table, metadata, autoload=True)
    return table_metadata


def get_table_columns(metadata):
    """
    Extract columns names and python types from the `metadata`.

    Parameters
    ----------
    metadata : sqlalchemy.sql.schema.Table
        Table metadata.

    Returns
    -------
    dict
        Dictionary with columns names and python types.
    """
    cols = dict()
    for col in metadata.c:
        name = str(col).rpartition(".")[2]
        cols[name] = col.type.python_type.__name__
    return cols


def build_query_from_table(name):
    """
    Create a query from the given table name.

    Parameters
    ----------
    name : str
        Table name.

    Returns
    -------
    str
        Query string.
    """
    return "SELECT * FROM {0}".format(name)


def check_query(query):
    """
    Check query sanity.

    Parameters
    ----------
    query : str
        Query string.
    """
    q = query.lower()
    if "select " not in q:
        raise InvalidQuery("SELECT word not found in the query: {0}".format(query))
    if " from " not in q:
        raise InvalidQuery("FROM word not found in the query: {0}".format(query))


def get_query_columns(engine, query):
    """
    Extract columns names and python types from the `query`.

    Parameters
    ----------
    engine : sqlalchemy.engine.base.Engine
        SQLAlchemy connection engine.
    query : str
        SQL query.

    Returns
    -------
    dict
        Dictionary with columns names and python types.
    """
    con = engine.connect()
    result = con.execute(text(query))
    cols_names = list(result.keys())
    values = list(result.first())
    cols = dict()
    for i in range(len(cols_names)):
        cols[cols_names[i]] = type(values[i]).__name__
    return cols


def check_partition_column(partition_column, cols):
    """
    Check `partition_column` existence and it's type.

    Parameters
    ----------
    partition_column : str
        Column name used for data partitioning between the workers.
    cols : dict
        Dictionary with columns names and python types.
    """
    for k, v in cols.items():
        if k == partition_column:
            if v == "int":
                return
            raise InvalidPartitionColumn(f"partition_column must be int, and not {v}")
    raise InvalidPartitionColumn(
        f"partition_column {partition_column} not found in the query"
    )


def get_query_info(sql, con, partition_column):
    """
    Compute metadata needed for query distribution.

    Parameters
    ----------
    sql : str
        SQL query to be executed or a table name.
    con : SQLAlchemy connectable or str
        Database connection or url string.
    partition_column : str
        Column name used for data partitioning between the workers.

    Returns
    -------
    list
        Columns names list.
    str
        Query string.
    """
    engine = create_engine(con)
    if is_table(engine, sql):
        table_metadata = get_table_metadata(engine, sql)
        query = build_query_from_table(sql)
        cols = get_table_columns(table_metadata)
    else:
        check_query(sql)
        query = sql.replace(";", "")
        cols = get_query_columns(engine, query)
    # TODO allow validation that takes into account edge cases of pandas e.g. "[index]"
    # check_partition_column(partition_column, cols)
    # TODO partition_column isn't used; we need to use it;
    return list(cols.keys()), query


def query_put_bounders(query, partition_column, start, end):  # pragma: no cover
    """
    Put partition boundaries into the query.

    Parameters
    ----------
    query : str
        SQL query string.
    partition_column : str
        Column name used for data partitioning between the workers.
    start : int
        Lowest value to request from the `partition_column`.
    end : int
        Highest value to request from the `partition_column`.

    Returns
    -------
    str
        Query string with boundaries.
    """
    where = " WHERE TMP_TABLE.{0} >= {1} AND TMP_TABLE.{0} <= {2}".format(
        partition_column, start, end
    )
    query_with_bounders = "SELECT * FROM ({0}) AS TMP_TABLE {1}".format(query, where)
    return query_with_bounders


class InvalidArguments(Exception):
    """Exception that should be raised if invalid arguments combination was found."""


class InvalidQuery(Exception):
    """Exception that should be raised if invalid query statement was found."""


class InvalidPartitionColumn(Exception):
    """Exception that should be raised if `partition_column` doesn't satisfy predefined requirements."""


def read_sql_with_offset(
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
    dtype_backend=lib.no_default,
    dtype=None,
):  # pragma: no cover
    """
    Read a chunk of SQL query or table into a pandas DataFrame.

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
    dtype_backend : {"numpy_nullable", "pyarrow"}, default: NumPy backed DataFrames
        Which dtype_backend to use, e.g. whether a DataFrame should have NumPy arrays,
        nullable dtypes are used for all dtypes that have a nullable implementation when
        "numpy_nullable" is set, PyArrow is used for all dtypes if "pyarrow" is set.
        The dtype_backends are still experimential.
    dtype : Type name or dict of columns, optional
        Data type for data or columns. E.g. np.float64 or {'a': np.float64, 'b': np.int32, 'c': 'Int64'}. The argument is ignored if a table is passed instead of a query.

    Returns
    -------
    list
        List with split read results and it's metadata (index, dtypes, etc.).
    """
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
        dtype_backend=dtype_backend,
        dtype=dtype,
    )
    index = len(pandas_df)
    return _split_result_for_readers(1, num_splits, pandas_df) + [index]
