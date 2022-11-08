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

"""The module holds the factory which performs I/O using pandas on a Client."""

from .query_compiler import ClientQueryCompiler
from modin.core.io.io import BaseIO
import os
import pandas


class ClientIO(BaseIO):
    """Factory providing methods for performing I/O operations using a given Client as the execution engine."""

    _server_conn = None
    _data_conn = None
    query_compiler_cls = ClientQueryCompiler

    @classmethod
    def set_server_connection(cls, conn):
        """
        Set the server connection for the I/O object.

        Parameters
        ----------
        conn : Any
            Connection object that implements various methods.
        """
        cls._server_conn = conn

    @classmethod
    def set_data_connection(cls, conn):
        """
        Set the data connection for the I/O object.

        Parameters
        ----------
        conn : Any
            Connection object that is implementation specific.
        """
        cls._data_conn = conn

    @classmethod
    def read_csv(cls, filepath_or_buffer, **kwargs):
        """
        Read CSV data from given filepath or buffer.

        Parameters
        ----------
        filepath_or_buffer : str, path object or file-like object
            `filepath_or_buffer` parameter of read functions.
        **kwargs : dict
            Parameters of ``read_csv`` function.

        Returns
        -------
        self.query_compiler_cls
            Query compiler with CSV data read in.
        """
        if cls._server_conn is None:
            raise ConnectionError(
                "Missing server connection, did you initialize the connection?"
            )
        if not isinstance(filepath_or_buffer, str):
            raise NotImplementedError("Only filepaths are supported for read_csv")
        if os.path.exists(filepath_or_buffer):
            # In case this is a local path, we should use the absolute path
            # because the service might be running in a different directory
            # on the same machine.
            filepath_or_buffer = os.path.abspath(filepath_or_buffer)
        server_result = cls._server_conn.read_csv(
            cls._data_conn, filepath_or_buffer, **kwargs
        )
        # This happens when `read_csv` returns a TextFileReader object for
        # iterating through, e.g. because iterator=True
        if isinstance(server_result, pandas.io.parsers.TextFileReader):
            return server_result
        return cls.query_compiler_cls(server_result)

    @classmethod
    def read_sql(cls, sql, con, **kwargs):
        """
        Read data from a SQL connection.

        Parameters
        ----------
        sql : str or SQLAlchemy Selectable (select or text object)
            SQL query to be executed or a table name.
        con : SQLAlchemy connectable, str, or sqlite3 connection
            Using SQLAlchemy makes it possible to use any DB supported by that
            library. If a DBAPI2 object, only sqlite3 is supported. The user is responsible
            for engine disposal and connection closure for the SQLAlchemy
            connectable; str connections are closed automatically. See
            `here <https://docs.sqlalchemy.org/en/13/core/connections.html>`_.
        **kwargs : dict
            Parameters of ``read_sql`` function.

        Returns
        -------
        self.query_compiler_cls
            Query compiler with data read in from SQL connection.
        """
        if isinstance(con, str) and con.lower() == "auto" and cls._data_conn is None:
            raise ConnectionError(
                "Cannot connect with parameter 'auto' because connection is not set. Did you initialize it?"
            )
        if cls._data_conn is None:
            cls._data_conn = con
        if cls._server_conn is None:
            raise ConnectionError(
                "Missing server connection, did you initialize the connection?"
            )
        return cls.query_compiler_cls(
            cls._server_conn.read_sql(sql, cls._data_conn, **kwargs)
        )

    @classmethod
    def to_sql(cls, qc, **kwargs) -> None:
        """
        Write records stored in a DataFrame to a SQL database.

        Databases supported by SQLAlchemy [1]_ are supported. Tables can be
        newly created, appended to, or overwritten.

        Parameters
        ----------
        qc : self.query_compiler_cls
            Query compiler with data to write to SQL.
        **kwargs : dict
            Parameters of ``read_sql`` function.
        """
        cls._server_conn.to_sql(qc._id, **kwargs)
