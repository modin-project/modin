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

import random
import string

import dbms_dialect


class mssql_dialect(dbms_dialect):

    type_map = {
        "bool": "BIT",
        "datetime": "DATETIME2",
        "datetime64": "DATETIME2",
        "float64": "FLOAT",
        "int64": "BIGINT",
        "object": "VARCHAR(MAX)",
    }

    def quote_name(name):
        """
        Generate the "escaped" name given parameters.  The escaped name allows for creating tables, columns etc. with special characters in their name.
        Parameters
        ----------
        name : string
            name to be escaped.
        """
        return "[" + name + "]"

    def generate_qualified_table_name(
        self,
        table_name,
        schema_name,
        database_name,
        is_temp=False,
        is_global_temp=False,
    ):
        """
        Generate a qualified table name given parameters

        Parameters
        ----------
        table_name : string
            name of table to be created.
        schema_name : string
            schema in which table is to be created.
        database_name : string
            database in which table is to be created.
        is_temp : boolean
            are we creating a temp table ?
        is_global_temp: boolean
            if we're creating a temp table - is it a global temp table ?
        """

        if is_temp:
            physical_table_name = "#" + table_name
            if is_global_temp:
                physical_table_name = "#" + physical_table_name
            return self.quote_name(physical_table_name)

        qualified_table_name = self.quote_name(database_name)

        if schema_name and len(schema_name) > 0:
            qualified_table_name += "." + self.quote_name(schema_name)

        qualified_table_name += "." + self.quote_name(table_name)
        return qualified_table_name

    def dbms_type_from_pandas_type(self, dtype):
        """
        return a string representing the database column type corresponding to a pandas column type.

        Parameters
        ----------
        dtype : dataframe column type
        """

        if dtype in self.type_map:
            return self.type_map[dtype]
        return "VARCHAR(MAX)"

    def generate_create_table_statement(
        self, df, table_name, schema_name, database_name, is_temp, is_global_temp
    ):
        """
        return a string representing the SQL DDL statement to create a table.

        Parameters
        ----------
        df : dataframe to generate the create table statement from.
        qualified_table_name: fully qualified name of table.
        """
        create_statement = "CREATE TABLE "

        qualified_table_name = self.generate_qualified_table_name(
            table_name, schema_name, database_name, is_temp, is_global_temp
        )

        create_statement += qualified_table_name + " ( "

        dataframe_column_names = df.columns
        dataframe_column_types = df.dtypes
        num_columns = len(dataframe_column_names)

        for column_index in range(num_columns):
            column_name = self.quote_name(dataframe_column_names[column_index])
            column_type = self.dbms_type_from_pandas_type(
                dataframe_column_types[column_index]
            )
            create_statement = create_statement + column_name + " " + column_type
            if column_index < num_columns - 1:
                create_statement = create_statement + ", "
        create_statement = create_statement + ")"

    def generate_table_existence_check_statement(self, qualified_table_name):
        return "SELECT TOP 1 * FROM " + qualified_table_name

    def get_column_list_as_string(self, df):
        dataframe_column_names = df.columns
        column_list_as_string = " "
        column_index = 0
        for column_name in dataframe_column_names:
            column_list_as_string = column_list_as_string + self.dialect.quote_name(
                column_name
            )
            if column_index < len(dataframe_column_names) - 1:
                column_list_as_string += ", "
            column_index += 1
        column_list_as_string += " "
        return column_list_as_string

    def generate_move_records_statement(
        self, df, qualified_source_table_name, qualified_target_table_name
    ):
        data_movement_command = "INSERT INTO " + qualified_target_table_name
        column_list_as_string = self.get_column_list_as_string(df)
        data_movement_command += " (" + column_list_as_string
        data_movement_command += (
            ") SELECT " + column_list_as_string + " FROM " + qualified_source_table_name
        )
        return data_movement_command

