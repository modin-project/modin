import inspect

from . import DataFrame
from ..factories_exp import ExperimentalBaseFactory


def read_sql(
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
):
    """ Read SQL query or database table into a DataFrame.

    Args:
        sql: string or SQLAlchemy Selectable (select or text object) SQL query to be executed or a table name.
        con: SQLAlchemy connectable (engine/connection) or database string URI or DBAPI2 connection (fallback mode)
        index_col: Column(s) to set as index(MultiIndex).
        coerce_float: Attempts to convert values of non-string, non-numeric objects (like decimal.Decimal) to
                      floating point, useful for SQL result sets.
        params: List of parameters to pass to execute method. The syntax used
                to pass parameters is database driver dependent. Check your
                database driver documentation for which of the five syntax styles,
                described in PEP 249's paramstyle, is supported.
        parse_dates:
                     - List of column names to parse as dates.
                     - Dict of ``{column_name: format string}`` where format string is
                       strftime compatible in case of parsing string times, or is one of
                       (D, s, ns, ms, us) in case of parsing integer timestamps.
                     - Dict of ``{column_name: arg dict}``, where the arg dict corresponds
                       to the keyword arguments of :func:`pandas.to_datetime`
                       Especially useful with databases without native Datetime support,
                       such as SQLite.
        columns: List of column names to select from SQL table (only used when reading a table).
        chunksize: If specified, return an iterator where `chunksize` is the number of rows to include in each chunk.
        partition_column: column used to share the data between the workers (MUST be a INTEGER column)
        lower_bound: the minimum value to be requested from the partition_column
        upper_bound: the maximum value to be requested from the partition_column

    Returns:
        Pandas Dataframe
    """
    _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())
    return DataFrame(query_compiler=ExperimentalBaseFactory.read_sql(**kwargs))
