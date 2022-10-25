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

from modin.core.io.io import BaseIO
import fsspec
from .query_compiler import ClientQueryCompiler


class ClientIO(BaseIO):
    """Factory providing methods for performing I/O operations using a given Client as the execution engine."""

    _server_conn = None
    _data_conn = None

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
        ClientQueryCompiler
            Query compiler with CSV data read in.
        """
        if isinstance(filepath_or_buffer, str):
            filepath_or_buffer = fsspec.open(filepath_or_buffer).full_name
            if filepath_or_buffer.startswith("file://"):
                # We will do this so that the backend can know whether this
                # is a path or a URL.
                filepath_or_buffer = filepath_or_buffer[7:]
        else:
            raise NotImplementedError("Only filepaths are supported for read_csv")
        if cls._server_conn is None:
            raise ConnectionError(
                "Missing server connection, did you initialize the connection?"
            )
        return ClientQueryCompiler(
            cls._server_conn.read_csv(cls._data_conn, filepath_or_buffer, **kwargs)
        )

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
        ClientQueryCompiler
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
        return ClientQueryCompiler(
            cls._server_conn.read_sql(sql, cls._data_conn, **kwargs)
        )

    @classmethod
    def to_sql(cls, qc, **kwargs):
        cls._server_conn.to_sql(qc._id, **kwargs)
