from modin.core.io.io import BaseIO
import os
from .query_compiler import ClientQueryCompiler


class ClientIO(BaseIO):
    _server_conn = None
    _data_conn = None

    @classmethod
    def set_server_connection(cls, conn):
        cls._server_conn = conn

    @classmethod
    def set_data_connection(cls, conn):
        cls._data_conn = conn

    @classmethod
    def read_csv(cls, filepath_or_buffer, **kwargs):
        if isinstance(filepath_or_buffer, str):
            filepath_or_buffer = os.path.abspath(filepath_or_buffer)
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
