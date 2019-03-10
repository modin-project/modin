from collections import OrderedDict
from sqlalchemy import MetaData, Table, create_engine


def is_distributed(partition_column, lower_bound, upper_bound):
    """ Check if is possible distribute a query given that args

    Args:
        partition_column: column used to share the data between the workers
        lower_bound: the minimum value to be requested from the partition_column
        upper_bound: the maximum value to be requested from the partition_column

    Returns:
        True for distributed or False if not
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
            "All these arguments should be passed (distributed) or none of them (standard pandas)."
        )


def is_table(engine, sql):
    """ Check with the given sql arg is query or table

    Args:
        engine: SQLAlchemy connection engine
        sql: SQL query or table name

    Returns:
        True for table or False if not
    """
    if engine.dialect.has_table(engine, sql):
        return True
    return False


def get_table_metadata(engine, table):
    """ Extract all useful infos from the given table

    Args:
        engine: SQLAlchemy connection engine
        table: table name

    Returns:
        Dictionary of infos
    """
    metadata = MetaData()
    metadata.reflect(bind=engine, only=[table])
    table_metadata = Table(table, metadata, autoload=True)
    return table_metadata


def get_table_columns(metadata):
    """ Extract columns names and python typos from metadata

    Args:
        metadata: Table metadata

    Returns:
        dict with columns names and python types
    """
    cols = OrderedDict()
    for col in metadata.c:
        name = str(col).rpartition(".")[2]
        cols[name] = col.type.python_type.__name__
    return cols


def build_query_from_table(name):
    """ Create a query given the table name

    Args:
        name: Table name

    Returns:
        query string
    """
    return "SELECT * FROM {0}".format(name)


def check_query(query):
    """ Check query sanity

    Args:
        query: query string

    Returns:
        None
    """
    q = query.lower()
    if "select " not in q:
        raise InvalidQuery("SELECT word not found in the query: {0}".format(query))
    if " from " not in q:
        raise InvalidQuery("FROM word not found in the query: {0}".format(query))


def get_query_columns(engine, query):
    """ Extract columns names and python typos from query

    Args:
        engine: SQLAlchemy connection engine
        query: SQL query

    Returns:
        dict with columns names and python types
    """
    con = engine.connect()
    result = con.execute(query).fetchone()
    values = list(result)
    cols_names = result.keys()
    cols = OrderedDict()
    for i in range(len(cols_names)):
        cols[cols_names[i]] = type(values[i]).__name__
    return cols


def check_partition_column(partition_column, cols):
    """ Check partition_column existence and type

    Args:
        partition_column: partition_column name
        cols: dict with columns names and python types

    Returns:
        None
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
    """ Return a columns name list and the query string

    Args:
        sql: SQL query or table name
        con: database connection or url string
        partition_column: column used to share the data between the workers

    Returns:
        Columns name list and query string
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
    cols_names = list(cols.keys())
    return cols_names, query


def query_put_bounders(query, partition_column, start, end):
    """ Put bounders in the query

    Args:
        query: SQL query string
        partition_column: partition_column name
        start: lower_bound
        end: upper_bound

    Returns:
        Query with bounders
    """
    where = " WHERE TMP_TABLE.{0} >= {1} AND TMP_TABLE.{0} <= {2}".format(
        partition_column, start, end
    )
    query_with_bounders = "SELECT * FROM ({0}) AS TMP_TABLE {1}".format(query, where)
    return query_with_bounders


class InvalidArguments(Exception):
    pass


class InvalidQuery(Exception):
    pass


class InvalidPartitionColumn(Exception):
    pass
