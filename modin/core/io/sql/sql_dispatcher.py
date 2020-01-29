import math
import numpy as np
import pandas
import warnings
from .utils import SqlQuery

from modin.engines.base.io.file_reader import FileReader


class SQLReader(FileReader):
    @classmethod
    def read(cls, sql, con, index_col=None, **kwargs):
        """Reads a SQL query or database table into a DataFrame.

        Args:
            sql: string or SQLAlchemy Selectable (select or text object) SQL query to be
                executed or a table name.
            con: SQLAlchemy connectable (engine/connection) or database string URI or
                DBAPI2 connection (fallback mode)
            index_col: Column(s) to set as index(MultiIndex).
            kwargs: Pass into pandas.read_sql function.

        Returns
        -------
        BaseQueryCompiler
            Query compiler with imported data for further processing.
        """
        is_modin_db_connection = isinstance(con, ModinDatabaseConnection)
        if not (is_modin_db_connection or isinstance(con, str)):
            warnings.warn(
                "To use parallel implementation of `read_sql`, pass either "
                + "the SQL connection string or a ModinDatabaseConnection "
                + "with the arguments required to make a connection, instead "
                + f"of {type(con)}. For documentation of ModinDatabaseConnection, see "
                + "https://modin.readthedocs.io/en/latest/supported_apis/io_supported.html#connecting-to-a-database-for-read-sql"
            )
            return cls.single_worker_read(sql, con=con, index_col=index_col, **kwargs)

        sa_engine = sa.create_engine(con)
        sql_query_engine = SqlQuery(sa_driver=sa_engine.driver)
        row_cnt_query = sql_query_engine.row_cnt(sql=sql)
        cols_names_query = sql_query_engine.empty(sql=sql)

        row_cnt = pandas.read_sql(row_cnt_query, con).squeeze()
        cols_names_df = pandas.read_sql(cols_names_query, con, index_col=index_col)
        cols_names = cols_names_df.columns
        from modin.pandas import DEFAULT_NPARTITIONS

        num_partitions = DEFAULT_NPARTITIONS
        partition_ids = []
        index_ids = []
        dtype_ids = []
        limit = math.ceil(row_cnt / num_partitions)
        for part in range(num_partitions):
            offset = part * limit
            query = sql_query_engine.partitioned(sql=sql, limit=limit, offset=offset)
            partition_id = cls.deploy(
                cls.parse,
                num_partitions + 2,
                dict(
                    num_splits=num_partitions,
                    sql=query,
                    con=con,
                    index_col=index_col,
                    **kwargs,
                ),
            )
            partition_ids.append(
                [cls.frame_partition_cls(obj) for obj in partition_id[:-2]]
            )
            index_ids.append(partition_id[-2])
            dtype_ids.append(partition_ids[-1])
        if index_col is None:  # sum all lens returned from partitions
            index_lens = cls.materialize(index_ids)
            new_index = pandas.RangeIndex(sum(index_lens))
        else:  # concat index returned from partitions
            index_lst = [
                x for part_index in cls.materialize(index_ids) for x in part_index
            ]
            new_index = pandas.Index(index_lst).set_names(index_col)
        new_frame = cls.frame_cls(np.array(partition_ids), new_index, cols_names)
        new_frame._apply_index_objs(axis=0)
        return cls.query_compiler_cls(new_frame)
