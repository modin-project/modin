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

from abc import ABCMeta


class dbms_dialect(metaclass=ABCMeta):
    def quote_name(self, name):
        return "`" + name + "`"

    def generate_qualified_table_name(
        self,
        table_name,
        schema_name,
        database_name,
        is_temp=False,
        is_global_temp=False,
    ):
        pass

    def dbms_type_from_pandas_type(self, dtype):
        """
        return a string representing the database column type corresponding to a pandas column type.

        Parameters
        ----------
        dtype : dataframe column type
        """
        pass

    def generate_create_table_statement(self, df, qualified_table_name):
        """
        return a string representing the SQL DDL statement to create a table.

        Parameters
        ----------
        df : dataframe to generate the create table statement from.
        qualified_table_name: fully qualified name of table.
        """
        pass

    def generate_drop_table_statement(self, qualified_table_name):
        return "DROP TABLE " + qualified_table_name

    def generate_insert_records_statement(self, df, qualified_table_name):
        insert_statement = "INSERT INTO " + qualified_table_name + " ("
        dataframe_column_names = df.columns
        column_index = 0
        for column_name in dataframe_column_names:
            insert_statement = insert_statement + self.quote_name(column_name)
            if column_index < len(dataframe_column_names) - 1:
                insert_statement += ", "
            column_index += 1
        insert_statement += ")"

        insert_statement += " VALUES ("

        for column_index in range(len(dataframe_column_names)):
            insert_statement += "?"
            if column_index < len(dataframe_column_names) - 1:
                insert_statement += ", "
        insert_statement += ")"

    def generate_table_existence_check_statement(self, qualified_table_name):
        return "SELECT * FROM " + qualified_table_name + " LIMIT 1"

    def generate_move_records_statement(
        self, df, qualified_source_table_name, qualified_target_table_name
    ):
        pass
        data_movement_command = "INSERT INTO " + qualified_target_table_name
        column_list_as_string = cls.get_column_list_as_string(qc)
        data_movement_command += " (" + column_list_as_string
        data_movement_command += (
            ") SELECT " + column_list_as_string + " FROM " + qualified_source_table_name
        )

        with pyodbc.connect(connection_string) as conn:
            with conn.cursor() as cursor:
                try:
                    cursor.execute(data_movement_command)
                    cursor.commit()
                except Exception as e:
                    cls.clean_up(qualified_source_table_name, conn)
                    raise e
