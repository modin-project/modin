import numpy as np
import pandas
import ray

from modin.engines.ray.pandas_on_ray.io import PandasOnRayIO, _split_result_for_readers
from modin.engines.ray.pandas_on_ray.remote_partition import PandasOnRayRemotePartition
from .sql import is_distributed, get_query_info, query_put_bounders


class ExperimentalPandasOnRayIO(PandasOnRayIO):
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
        if not is_distributed(partition_column, lower_bound, upper_bound):
            # Change this so that when `PandasOnRayIO` has a parallel `read_sql` we can
            # still use it.
            return PandasOnRayIO.read_sql(
                sql,
                con,
                index_col,
                coerce_float,
                params,
                parse_dates,
                columns,
                chunksize,
            )
        #  starts the distributed alternative
        cols_names, query = get_query_info(sql, con, partition_column)
        num_parts = cls.block_partitions_cls._compute_num_partitions()
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
                num_return_vals=num_splits + 1,
            )
            partition_ids.append(
                [PandasOnRayRemotePartition(obj) for obj in partition_id[:-1]]
            )
            index_ids.append(partition_id[-1])
        new_index = pandas.RangeIndex(sum(ray.get(index_ids)))
        new_query_compiler = cls.query_compiler_cls(
            cls.block_partitions_cls(np.array(partition_ids)), new_index, cols_names
        )
        return new_query_compiler


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
    """Use a Ray task to read a chunk of SQL source.

    Note: Ray functions are not detected by codecov (thus pragma: no cover)
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
    )
    index = len(pandas_df)
    return _split_result_for_readers(1, num_splits, pandas_df) + [index]
