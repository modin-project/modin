from sqlalchemy import MetaData, Table, create_engine


def is_distributed(partition_column, lower_bound, upper_bound):
    """
    Check if is possible distribute a query given that args
    :param partition_column: column used to share the data between the workers
    :param lower_bound: the minimum value to be requested from the partition_column
    :param upper_bound: the maximum value to be requested from the partition_column
    :return: True for distributed or False if not
    """
    if (
        (partition_column != None) and (lower_bound != None) and (upper_bound != None)
    ):  # noqa: E711
        if upper_bound > lower_bound:
            return True
        else:
            raise InvalidArguments("upper_bound must be greater than lower_bound.")
    elif (
        (partition_column == None) and (lower_bound == None) and (upper_bound == None)
    ):  # noqa: E711
        return False
    else:
        raise InvalidArguments(
            "Invalid combination of partition_column, lower_bound, upper_bound."
            "All these arguments should be passed (distributed) or none of them (standard pandas)."
        )


def is_table(engine, sql):
    """
    Check with the given sql arg is query or table
    :param engine: SQLAlchemy connection engine
    :param sql: SQL query or table name
    :return: True for table or False if not
    """
    print(sql)
    if engine.dialect.has_table(engine, sql):
        return True
    return False


def get_table_metadata(engine, table):
    """
    Extract all useful infos from the given table
    :param engine: SQLAlchemy connection engine
    :param table: table name
    :return: Dictionary of infos
    """
    metadata = MetaData()
    metadata.reflect(bind=engine, only=[table])
    table_metadata = Table(table, metadata, autoload=True)
    return table_metadata


def get_table_columns(metadata):
    """
    Extract columns names and python typos from metadata
    :param metadata: Table metadata
    :return: dict with columns names and python types
    """
    cols = dict()
    for col in metadata.c:
        name = str(col).rpartition(".")[2]
        cols[name] = col.type.python_type.__name__
    return cols


def build_query_from_table(name):
    """
    Create a query given the table name
    :param name: Table name
    :return: query string
    """
    return "SELECT * FROM {0}".format(name)


def check_query(query):
    """
    Check query sanity
    :param query: query string
    :return: None
    """
    q = query.lower()
    if "select " not in q:
        raise InvalidQuery("SELECT word not found in the query: {0}".format(query))
    if " from " not in q:
        raise InvalidQuery("FROM word not found in the query: {0}".format(query))


def get_query_columns(engine, query):
    """
    Extract columns names and python typos from query
    :param engine: SQLAlchemy connection engine
    :param query: SQL query
    :return: dict with columns names and python types
    """
    con = engine.connect()
    result = con.execute(query).fetchone()
    values = list(result)
    cols_names = result.keys()
    cols = dict()
    for i in range(len(cols_names)):
        cols[cols_names[i]] = type(values[i]).__name__
    return cols


def check_partition_column(partition_column, cols):
    """
    Check partition_column existence and type
    :param partition_column: partition_column name
    :param cols: dict with columns names and python types
    :return: None
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
    Return a columns name list and the query string
    :param sql: SQL query or table name
    :param con: database connection or url string
    :param partition_column: column used to share the data between the workers
    :return: Columns name list and query string
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
    check_partition_column(partition_column, cols)
    cols_names = list(cols.keys())
    return cols_names, query


def query_put_bounders(query, partition_column, start, end):
    """
    Put bounders in the query
    :param query: SQL query string
    :param partition_column: partition_column name
    :param start: lower_bound
    :param end: upper_bound
    :return: QUery with bounders
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
