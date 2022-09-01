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

"""Module houses util functions for handling experimental SQL IO functions."""

from collections import OrderedDict
from sqlalchemy import MetaData, Table, create_engine, inspect


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
        else:
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
    if inspect(engine).has_table(sql):
        return True
    return False


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
    OrderedDict
        Dictionary with columns names and python types.
    """
    cols = OrderedDict()
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
    OrderedDict
        Dictionary with columns names and python types.
    """
    con = engine.connect()
    result = con.execute(query).fetchone()
    values = list(result)
    cols_names = list(result.keys())
    cols = OrderedDict()
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
    cols : OrderedDict/dict
        Dictionary with columns names and python types.
    """
    for k, v in cols.items():
        if k == partition_column:
            if v == "int":
                return
            else:
                raise InvalidPartitionColumn(
                    "partition_column must be int, and not {0}".format(v)
                )
    raise InvalidPartitionColumn(
        "partition_column {0} not found in the query".format(partition_column)
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
    cols_names = list(cols.keys())
    return cols_names, query


def query_put_bounders(query, partition_column, start, end):
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
