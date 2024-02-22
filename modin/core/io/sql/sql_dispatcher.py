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
Module houses `SQLDispatcher` class.

`SQLDispatcher` contains utils for handling SQL queries or database tables,
inherits util functions for handling files from `FileDispatcher` class and can be
used as base class for dipatchers of SQL queries.
"""

import math

import numpy as np
import pandas

from modin.config import NPartitions, ReadSqlEngine
from modin.core.io.file_dispatcher import FileDispatcher
from modin.db_conn import ModinDatabaseConnection


class SQLDispatcher(FileDispatcher):
    """Class handles utils for reading SQL queries or database tables."""

    @classmethod
    def _is_supported_sqlalchemy_object(cls, obj):  # noqa: GL08
        supported = None
        try:
            import sqlalchemy as sa

            supported = isinstance(obj, (sa.engine.Engine, sa.engine.Connection))
        except ImportError:
            supported = False
        return supported

    @classmethod
    def _read(cls, sql, con, index_col=None, **kwargs):
        """
        Read a SQL query or database table into a query compiler.

        Parameters
        ----------
        sql : str or SQLAlchemy Selectable (select or text object)
            SQL query to be executed or a table name.
        con : SQLAlchemy connectable, str, sqlite3 connection, or ModinDatabaseConnection
            Connection object to database.
        index_col : str or list of str, optional
            Column(s) to set as index(MultiIndex).
        **kwargs : dict
            Parameters to pass into `pandas.read_sql` function.

        Returns
        -------
        BaseQueryCompiler
            Query compiler with imported data for further processing.
        """
        if isinstance(con, str):
            con = ModinDatabaseConnection("sqlalchemy", con)

        if cls._is_supported_sqlalchemy_object(con):
            con = ModinDatabaseConnection(
                "sqlalchemy", con.engine.url.render_as_string(hide_password=False)
            )

        if not isinstance(con, ModinDatabaseConnection):
            return cls.single_worker_read(
                sql,
                con=con,
                index_col=index_col,
                read_sql_engine=ReadSqlEngine.get(),
                reason="To use the parallel implementation of `read_sql`, pass either "
                + "a SQLAlchemy connectable, the SQL connection string, or a ModinDatabaseConnection "
                + "with the arguments required to make a connection, instead "
                + f"of {type(con)}. For documentation on the ModinDatabaseConnection, see "
                + "https://modin.readthedocs.io/en/latest/supported_apis/io_supported.html#connecting-to-a-database-for-read-sql",
                **kwargs,
            )
        row_count_query = con.row_count_query(sql)
        connection_for_pandas = con.get_connection()
        colum_names_query = con.column_names_query(sql)
        row_cnt = pandas.read_sql(row_count_query, connection_for_pandas).squeeze()
        cols_names_df = pandas.read_sql(
            colum_names_query, connection_for_pandas, index_col=index_col
        )
        cols_names = cols_names_df.columns
        num_partitions = NPartitions.get()
        partition_ids = [None] * num_partitions
        index_ids = [None] * num_partitions
        dtypes_ids = [None] * num_partitions
        limit = math.ceil(row_cnt / num_partitions)
        for part in range(num_partitions):
            offset = part * limit
            query = con.partition_query(sql, limit, offset)
            *partition_ids[part], index_ids[part], dtypes_ids[part] = cls.deploy(
                func=cls.parse,
                f_kwargs={
                    "num_splits": num_partitions,
                    "sql": query,
                    "con": con,
                    "index_col": index_col,
                    "read_sql_engine": ReadSqlEngine.get(),
                    **kwargs,
                },
                num_returns=num_partitions + 2,
            )
            partition_ids[part] = [
                cls.frame_partition_cls(obj) for obj in partition_ids[part]
            ]
        if index_col is None:  # sum all lens returned from partitions
            index_lens = cls.materialize(index_ids)
            new_index = pandas.RangeIndex(sum(index_lens))
        else:  # concat index returned from partitions
            index_lst = [
                x for part_index in cls.materialize(index_ids) for x in part_index
            ]
            new_index = pandas.Index(index_lst).set_names(index_col)
        new_frame = cls.frame_cls(np.array(partition_ids), new_index, cols_names)
        new_frame.synchronize_labels(axis=0)
        return cls.query_compiler_cls(new_frame)

    @classmethod
    def write(cls, qc, **kwargs):
        """
        Write records stored in the `qc` to a SQL database.

        Parameters
        ----------
        qc : BaseQueryCompiler
            The query compiler of the Modin dataframe that we want to run ``to_sql`` on.
        **kwargs : dict
            Parameters for ``pandas.to_sql(**kwargs)``.
        """
        # we first insert an empty DF in order to create the full table in the database
        # This also helps to validate the input against pandas
        # we would like to_sql() to complete only when all rows have been inserted into the database
        # since the mapping operation is non-blocking, each partition will return an empty DF
        # so at the end, the blocking operation will be this empty DF to_pandas

        if not isinstance(
            kwargs["con"], str
        ) and not cls._is_supported_sqlalchemy_object(kwargs["con"]):
            return cls.base_io.to_sql(qc, **kwargs)

        # In the case that we are given a SQLAlchemy Connection or Engine, the objects
        # are not pickleable. We have to convert it to the URL string and connect from
        # each of the workers.
        if cls._is_supported_sqlalchemy_object(kwargs["con"]):
            kwargs["con"] = kwargs["con"].engine.url.render_as_string(
                hide_password=False
            )

        empty_df = qc.getitem_row_array([0]).to_pandas().head(0)
        empty_df.to_sql(**kwargs)
        # so each partition will append its respective DF
        kwargs["if_exists"] = "append"
        columns = qc.columns

        def func(df):  # pragma: no cover
            """
            Override column names in the wrapped dataframe and convert it to SQL.

            Notes
            -----
            This function returns an empty ``pandas.DataFrame`` because ``apply_full_axis``
            expects a Frame object as a result of operation (and ``to_sql`` has no dataframe result).
            """
            df.columns = columns
            df.to_sql(**kwargs)
            return pandas.DataFrame()

        # Ensure that the metadata is synchronized
        qc._modin_frame._propagate_index_objs(axis=None)
        result = qc._modin_frame.apply_full_axis(1, func, new_index=[], new_columns=[])
        cls.materialize(
            [part.list_of_blocks[0] for row in result._partitions for part in row]
        )
