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
Module houses `ModinDatabaseConnection` class.

`ModinDatabaseConnection` lets a single process make its own connection to a
database to read from it. Whereas it's possible in pandas to pass an open
connection directly to `read_sql`, the open connection is not pickleable
in Modin, so each worker must open its own connection.
`ModinDatabaseConnection` saves the arguments that would normally be used to
make a db connection. It can make and provide a connection whenever the Modin
driver or a worker wants one.
"""

_PSYCOPG_LIB_NAME = "psycopg2"
_SQLALCHEMY_LIB_NAME = "sqlalchemy"


class UnsupportedDatabaseException(Exception):
    pass


class ModinDatabaseConnection:
    """
    Creates a SQL database connection.

    Parameters
    ----------
    lib : str
        The library for the SQL connection.
    *args : iterable
        Positional arguments to pass when creating the connection.
    **kwargs : dict
        Keyword arguments to pass when creating the connection.
    """

    def __init__(self, lib, *args, **kwargs):
        lib = lib.lower()
        if lib not in (_PSYCOPG_LIB_NAME, _SQLALCHEMY_LIB_NAME):
            raise UnsupportedDatabaseException(f"Unsupported database library {lib}")
        self.lib = lib
        self.args = args
        self.kwargs = kwargs

    def get_connection(self):
        """
        Make the database connection and get it.

        For psycopg2, pass all arguments to psycopg2.connect() and return the
        result of psycopg2.connect(). For sqlalchemy, pass all arguments to
        sqlalchemy.create_engine() and return the result of calling connect()
        on the engine.

        Returns
        -------
        Any
            The open database connection.
        """
        if self.lib == _PSYCOPG_LIB_NAME:
            import psycopg2

            return psycopg2.connect(*self.args, **self.kwargs)
        if self.lib == _SQLALCHEMY_LIB_NAME:
            from sqlalchemy import create_engine

            return create_engine(*self.args, **self.kwargs).connect()

        raise Exception("Unsupported database library")
