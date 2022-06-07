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

"""Module provides ``OmnisciServer`` class."""

import uuid
import os

import pyarrow as pa
import numpy as np

from .utils import PyDbEngine

from modin.config import OmnisciFragmentSize, OmnisciLaunchParameters
from modin.error_message import ErrorMessage


class OmnisciServer:
    """Wrapper class for OmniSci storage format."""

    _server = None

    @classmethod
    def start_server(cls):
        """
        Initialize OmniSci server.

        Do nothing if it is initialized already.
        """
        if cls._server is None:
            cls._server = PyDbEngine(**OmnisciLaunchParameters.get())

    @classmethod
    def stop_server(cls):
        """Destroy OmniSci server if any."""
        if cls._server is not None:
            cls._server.reset()
            cls._server = None

    def __init__(self):
        """Initialize OmniSci storage format."""
        self.start_server()

    @classmethod
    def executeDDL(cls, query):
        """
        Execute DDL SQL query.

        Parameters
        ----------
        query : str
            SQL query.
        """
        cls._server.executeDDL(query)

    @classmethod
    def executeDML(cls, query):
        """
        Execute DML SQL query.

        Parameters
        ----------
        query : str
            SQL query.

        Returns
        -------
        pyarrow.Table
            Execution result.
        """
        r = cls._server.executeDML(query)
        # todo: assert r
        return r

    @classmethod
    def executeRA(cls, query):
        """
        Execute calcite query.

        Parameters
        ----------
        query : str
            Serialized calcite query.

        Returns
        -------
        pyarrow.Table
            Execution result.
        """
        r = cls._server.executeRA(query)
        # todo: assert r
        return r

    @classmethod
    def _genName(cls, name):
        """
        Generate or mangle a table name.

        Parameters
        ----------
        name : str or None
            Table name to mangle or None to generate a unique
            table name.

        Returns
        -------
        str
            Table name.
        """
        if not name:
            name = "frame_" + str(uuid.uuid4()).replace("-", "")
        # TODO: reword name in case of caller's mistake
        return name

    @staticmethod
    def cast_to_compatible_types(table):
        """
        Cast PyArrow table to be fully compatible with OmniSci.

        Parameters
        ----------
        table : pyarrow.Table
            Source table.

        Returns
        -------
        pyarrow.Table
            Table with fully compatible types with OmniSci.
        """
        schema = table.schema
        new_schema = schema
        need_cast = False
        uint_to_int_cast = False
        new_cols = {}
        uint_to_int_map = {
            pa.uint8(): pa.int16(),
            pa.uint16(): pa.int32(),
            pa.uint32(): pa.int64(),
            pa.uint64(): pa.int64(),  # May cause overflow
        }
        for i, field in enumerate(schema):
            # Currently OmniSci doesn't support Arrow table import with
            # dictionary columns. Here we cast dictionaries until support
            # is in place.
            # https://github.com/modin-project/modin/issues/1738
            if pa.types.is_dictionary(field.type):
                # Conversion for dictionary of null type to string is not supported
                # in Arrow. Build new column for this case for now.
                if pa.types.is_null(field.type.value_type):
                    mask = np.full(table.num_rows, True, dtype=bool)
                    new_col_data = np.empty(table.num_rows, dtype=str)
                    new_col = pa.array(new_col_data, pa.string(), mask)
                    new_cols[i] = new_col
                else:
                    need_cast = True
                new_field = pa.field(
                    field.name, pa.string(), field.nullable, field.metadata
                )
                new_schema = new_schema.set(i, new_field)
            # OmniSci doesn't support importing Arrow's date type:
            # https://github.com/omnisci/omniscidb/issues/678
            elif pa.types.is_date(field.type):
                # Arrow's date is the number of days since the UNIX-epoch, so we can convert it
                # to a timestamp[s] (number of seconds since the UNIX-epoch) without losing precision
                new_field = pa.field(
                    field.name, pa.timestamp("s"), field.nullable, field.metadata
                )
                new_schema = new_schema.set(i, new_field)
                need_cast = True
            # OmniSci doesn't support unsigned types
            elif pa.types.is_unsigned_integer(field.type):
                new_field = pa.field(
                    field.name,
                    uint_to_int_map[field.type],
                    field.nullable,
                    field.metadata,
                )
                new_schema = new_schema.set(i, new_field)
                need_cast = True
                uint_to_int_cast = True

        # Such cast may affect the data, so we have to raise a warning about it
        if uint_to_int_cast:
            ErrorMessage.single_warning(
                "OmniSci does not support unsigned integer types, such types will be rounded up to the signed equivalent."
            )

        for i, col in new_cols.items():
            table = table.set_column(i, new_schema[i], col)

        if need_cast:
            try:
                table = table.cast(new_schema)
            except pa.lib.ArrowInvalid as e:
                raise (OverflowError if uint_to_int_cast else RuntimeError)(
                    "An error occurred when trying to convert unsupported by OmniSci 'dtypes' "
                    + f"to the supported ones, the schema to cast was: \n{new_schema}."
                ) from e

        return table

    @classmethod
    def put_arrow_to_omnisci(cls, table, name=None):
        """
        Import Arrow table to OmniSci engine.

        Parameters
        ----------
        table : pyarrow.Table
            A table to import.
        name : str, optional
            A table name to use. None to generate a unique name.

        Returns
        -------
        str
            Imported table name.
        """
        name = cls._genName(name)

        table = cls.cast_to_compatible_types(table)
        fragment_size = OmnisciFragmentSize.get()
        if fragment_size is None:
            cpu_count = os.cpu_count()
            if cpu_count is not None:
                fragment_size = table.num_rows // cpu_count
                fragment_size = min(fragment_size, 2**25)
                fragment_size = max(fragment_size, 2**18)
            else:
                fragment_size = 0
        else:
            fragment_size = int(fragment_size)

        cls._server.importArrowTable(name, table, fragment_size=fragment_size)

        return name

    @classmethod
    def put_pandas_to_omnisci(cls, df, name=None):
        """
        Import ``pandas.DataFrame`` to OmniSci storage format.

        Parameters
        ----------
        df : pandas.DataFrame
            A frame to import.
        name : str, optional
            A table name to use. None to generate a unique name.

        Returns
        -------
        str
            Imported table name.
        """
        return cls.put_arrow_to_omnisci(pa.Table.from_pandas(df))
