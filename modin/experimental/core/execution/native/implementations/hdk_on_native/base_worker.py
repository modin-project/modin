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

"""Module provides ``BaseDbWorker`` class."""

import abc
import uuid
from typing import List, Tuple

import numpy as np
import pyarrow as pa

from modin.error_message import ErrorMessage

_UINT_TO_INT_MAP = {
    pa.uint8(): pa.int16(),
    pa.uint16(): pa.int32(),
    pa.uint32(): pa.int64(),
    pa.uint64(): pa.int64(),  # May cause overflow
}


class DbTable(abc.ABC):
    """
    Base class, representing a table in the HDK database.

    Attributes
    ----------
    name : str
        Table name.
    """

    @property
    @abc.abstractmethod
    def shape(self) -> Tuple[int, int]:
        """
        Return a tuple with the number of rows and columns.

        Returns
        -------
        tuple of int
        """
        pass

    @property
    @abc.abstractmethod
    def column_names(self) -> List[str]:
        """
        Return a list of the table column names.

        Returns
        -------
        tuple of str
        """
        pass

    @abc.abstractmethod
    def to_arrow(self) -> pa.Table:
        """
        Convert this table to arrow.

        Returns
        -------
        pyarrow.Table
        """
        pass

    def __len__(self):
        """
        Return the number of rows in the table.

        Returns
        -------
        int
        """
        return self.shape[0]


class BaseDbWorker(abc.ABC):
    """Base class for HDK storage format based execution engine ."""

    @classmethod
    @abc.abstractmethod
    def dropTable(cls, name):
        """
        Drops table with the specified name.

        Parameters
        ----------
        name : str
            A table to drop.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def executeDML(cls, query):
        """
        Execute DML SQL query.

        Parameters
        ----------
        query : str
            SQL query.

        Returns
        -------
        DbTable
            Execution result.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def executeRA(cls, query):
        """
        Execute calcite query.

        Parameters
        ----------
        query : str
            Serialized calcite query.

        Returns
        -------
        DbTable
            Execution result.
        """
        pass

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

    @classmethod
    def cast_to_compatible_types(cls, table, cast_dict):
        """
        Cast PyArrow table to be fully compatible with HDK.

        Parameters
        ----------
        table : pyarrow.Table
            Source table.
        cast_dict : bool
            Cast dictionary columns to string.

        Returns
        -------
        pyarrow.Table
            Table with fully compatible types with HDK.
        """
        schema = table.schema
        new_schema = schema
        need_cast = False
        uint_to_int_cast = False

        for i, field in enumerate(schema):
            if pa.types.is_dictionary(field.type):
                value_type = field.type.value_type
                # Conversion for dictionary of null type to string is not supported
                # in Arrow. Build new column for this case for now.
                if pa.types.is_null(value_type):
                    mask = np.full(table.num_rows, True, dtype=bool)
                    new_col_data = np.empty(table.num_rows, dtype=str)
                    new_col = pa.array(new_col_data, pa.string(), mask)
                    new_field = pa.field(
                        field.name, pa.string(), field.nullable, field.metadata
                    )
                    table = table.set_column(i, new_field, new_col)
                elif pa.types.is_string(value_type):
                    if cast_dict:
                        need_cast = True
                        new_field = pa.field(
                            field.name, pa.string(), field.nullable, field.metadata
                        )
                    else:
                        new_field = field
                else:
                    new_field, int_cast = cls._convert_field(field, value_type)
                    need_cast = True
                    uint_to_int_cast = uint_to_int_cast or int_cast
                    if new_field == field:
                        new_field = pa.field(
                            field.name,
                            value_type,
                            field.nullable,
                            field.metadata,
                        )
                new_schema = new_schema.set(i, new_field)
            else:
                new_field, int_cast = cls._convert_field(field, field.type)
                need_cast = need_cast or new_field is not field
                uint_to_int_cast = uint_to_int_cast or int_cast
                new_schema = new_schema.set(i, new_field)

        # Such cast may affect the data, so we have to raise a warning about it
        if uint_to_int_cast:
            ErrorMessage.single_warning(
                "HDK does not support unsigned integer types, such types will be rounded up to the signed equivalent."
            )

        if need_cast:
            try:
                table = table.cast(new_schema)
            except pa.lib.ArrowInvalid as err:
                raise (OverflowError if uint_to_int_cast else RuntimeError)(
                    "An error occurred when trying to convert unsupported by HDK 'dtypes' "
                    + f"to the supported ones, the schema to cast was: \n{new_schema}."
                ) from err

        return table

    @staticmethod
    def _convert_field(field, field_type):
        """
        Convert the specified arrow field, if required.

        Parameters
        ----------
        field : pyarrow.Field
        field_type : pyarrow.DataType

        Returns
        -------
        Tuple[pyarrow.Field, boolean]
            A tuple, containing (new_field, uint_to_int_cast)
        """
        if pa.types.is_date(field_type):
            # Arrow's date is the number of days since the UNIX-epoch, so we can convert it
            # to a timestamp[s] (number of seconds since the UNIX-epoch) without losing precision
            return (
                pa.field(field.name, pa.timestamp("s"), field.nullable, field.metadata),
                False,
            )
        elif pa.types.is_unsigned_integer(field_type):
            # HDK doesn't support unsigned types
            return (
                pa.field(
                    field.name,
                    _UINT_TO_INT_MAP[field_type],
                    field.nullable,
                    field.metadata,
                ),
                True,
            )
        return field, False

    @classmethod
    @abc.abstractmethod
    def import_arrow_table(cls, table, name=None):
        """
        Import Arrow table to the worker.

        Parameters
        ----------
        table : pyarrow.Table
            A table to import.
        name : str, optional
            A table name to use. None to generate a unique name.

        Returns
        -------
        DbTable
            Imported table.
        """
        pass

    @classmethod
    def import_pandas_dataframe(cls, df, name=None):
        """
        Import ``pandas.DataFrame`` to the worker.

        Parameters
        ----------
        df : pandas.DataFrame
            A frame to import.
        name : str, optional
            A table name to use. None to generate a unique name.

        Returns
        -------
        DbTable
            Imported table.
        """
        return cls.import_arrow_table(pa.Table.from_pandas(df), name=name)
