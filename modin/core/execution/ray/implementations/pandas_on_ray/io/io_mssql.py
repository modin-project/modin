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

import pandas
import pyodbc
import random
import string
import time


class PandasOnRayIOToMSSQL:
    """Factory providing methods for writing modin data frames to MSSQL on Ray as engine."""

    @classmethod
    def __generate_qualified_table_name(
        cls, table_name, schema_name, database_name, is_temp=False, is_global_temp=False
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
            return "[" + physical_table_name + "]"

        qualified_table_name = "[" + database_name + "]"

        if schema_name and len(schema_name) > 0:
            qualified_table_name += ".[" + schema_name + "]"

        qualified_table_name += ".[" + table_name + "]"
        return qualified_table_name

    @classmethod
    def __map_dataframe_columntypes_to_mssql_columntypes(cls, df):
        """
        Generate a map of column names to sql server data types given a data frame.

        Parameters
        ----------
        df : dataframe
            dataframe corresponding to which map is to be generated.
        """
        column_name_type_map = {}

        dataframe_column_names = df.columns
        dataframe_column_types = df.dtypes

        for column_name, column_type in zip(
            dataframe_column_names, dataframe_column_types
        ):
            if column_type == "int64":
                column_name_type_map[column_name] = "BIGINT"
            elif column_type == "float64":
                column_name_type_map[column_name] = "FLOAT"
            elif column_type == "object":
                column_name_type_map[column_name] = "VARCHAR(MAX)"
            elif column_type == "datetime64" or column_type == "datetime":
                column_name_type_map[column_name] = "DATETIME2"
        return column_name_type_map

    @classmethod
    def drop_table_if_exists(cls, qualified_table_name, conn):
        with conn.cursor() as cursor:
            drop_statement = "DROP TABLE " + qualified_table_name

            try:
                cursor.execute(drop_statement)
                cursor.commit()
            except Exception:
                pass

    @classmethod
    def table_exists(cls, qualified_table_name, conn):
        select_statement = "SELECT TOP 1 * FROM " + qualified_table_name

        with conn.cursor() as cursor:
            try:
                cursor.execute(select_statement)
            except Exception as e:
                if isinstance(e, pyodbc.ProgrammingError):
                    return False
                raise e
        return True

    @classmethod
    def create_table(cls, df, qualified_table_name, conn):

        cls.drop_table_if_exists(qualified_table_name, conn)
        column_name_type_map = cls.__map_dataframe_columntypes_to_mssql_columntypes(df)

        create_statement = "CREATE TABLE "

        create_statement += qualified_table_name + " ( "

        num_columns = len(column_name_type_map)

        column_index = 0

        for item in column_name_type_map:
            column_name = item
            column_type = column_name_type_map[column_name]
            create_statement = create_statement + "[" + column_name + "] " + column_type
            if column_index < num_columns - 1:
                create_statement = create_statement + ", "
            column_index += 1
        create_statement = create_statement + ")"

        with conn.cursor() as cursor:
            cursor.fast_executemany = True
            cursor.execute(create_statement)

    @classmethod
    def insert_records(cls, qc, qualified_table_name, connection_string):

        insert_statement = "INSERT INTO " + qualified_table_name + " ("
        dataframe_column_names = qc.columns
        column_index = 0
        for column_name in dataframe_column_names:
            insert_statement = insert_statement + "[" + column_name + "]"
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

        def func(df):

            conn = pyodbc.connect(connection_string)

            with conn.cursor() as cursor:
                cursor.fast_executemany = True

                start_time = time.time()
                num_records_per_batch = 10000
                num_records_inserted = 0

                while num_records_inserted < len(df):
                    last_index = num_records_inserted + num_records_per_batch
                    if last_index >= len(df):
                        last_index = len(df)
                    curr_df = df.iloc[num_records_inserted:last_index]
                    cursor.executemany(
                        insert_statement, curr_df.values.tolist(),
                    )
                    num_records_inserted = last_index

                print("time for executemany: ", time.time() - start_time)
                return pandas.DataFrame()

        result = qc.apply_full_axis(1, func, new_index=[], new_columns=[])
        # FIXME: we should be waiting for completion less expensively, maybe use _modin_frame.materialize()?
        result.to_pandas()  # blocking operation

    @classmethod
    def to_database(cls, **kwargs):
        """
        Write records stored in the `qc` to an MSSQL database

        Parameters
        ----------
        **kwargs : dict
            Parameters for ``pandas.to_sql(**kwargs)``.
        """
        # we first create a temp table so all partitions can write their into it.
        # Once all partitions have reported success we move data from the temp table into the destination table.
        # we would like to_sql() to complete only when all rows have been inserted into the database
        # since the mapping operation is non-blocking, each partition will return an empty DF
        # so at the end, the blocking operation will be this empty DF to_pandas
        # we first insert an empty DF in order to create the full table in the database
        # This also helps to validate the input against pandas
        # we would like to_sql() to complete only when all rows have been inserted into the database
        # since the mapping operation is non-blocking, each partition will return an empty DF
        # so at the end, the blocking operation will be this empty DF to_pandas

        # Ensure that the metadata is synchronized
        qc = kwargs["qc"]
        qc._modin_frame._propagate_index_objs(axis=None)

        temp_table_name = "".join(
            random.choices(string.ascii_uppercase + string.digits, k=12)
        )
        connection_params = kwargs["connection_params"]
        qualified_temp_table_name = cls.__generate_qualified_table_name(
            temp_table_name,
            connection_params["schema_name"],
            connection_params["database_name"],
            True,
            True,
        )
        df = qc._modin_frame
        qualified_final_table_name = cls.__generate_qualified_table_name(
            kwargs["name"],
            connection_params["schema_name"],
            connection_params["database_name"],
        )
        connection_string = connection_params["connection_string"]
        destination_table_name = kwargs["name"]
        if_exists_option = ""
        if "if_exists" in kwargs:
            if_exists_option = kwargs["if_exists"]
        with pyodbc.connect(connection_string) as conn:
            if cls.table_exists(qualified_final_table_name, conn):
                if len(if_exists_option) == 0:
                    raise ValueError(
                        "Table ",
                        destination_table_name,
                        " already exists and no if_exists argument was specified",
                    )
                if if_exists_option == "fail":
                    raise ValueError(
                        "Table ", destination_table_name, " already exists"
                    )
            cls.create_table(df, qualified_temp_table_name, conn)

            try:
                cls.insert_records(df, qualified_temp_table_name, connection_string)
            except Exception as e:
                cls.clean_up(qualified_temp_table_name, conn)
                raise e

            try:
                move_start_time = time.time()
                if if_exists_option == "replace":
                    cls.drop_table_if_exists(qualified_final_table_name, conn)
                    cls.create_table(df, qualified_final_table_name, conn)
                elif cls.table_exists(qualified_final_table_name) is False:
                    cls.create_table(df, qualified_final_table_name, conn)
                cls.move_records_from_temp_table_to_final(
                    qualified_temp_table_name,
                    qualified_final_table_name,
                    connection_string,
                    qc,
                )
                print(
                    "time to move data from temp table into final table: ",
                    time.time() - move_start_time,
                )
                return
            except Exception as e:
                cls.clean_up(qualified_temp_table_name, conn)
                raise e

    @classmethod
    def clean_up(cls, qualified_table_name, conn):
        with conn.cursor() as cursor:
            table_deletion_command = "DROP TABLE " + qualified_table_name + ";"
            cursor.execute(table_deletion_command)
            cursor.commit()

    @classmethod
    def get_column_list_as_string(cls, df):
        dataframe_column_names = df.columns
        column_list_as_string = " "
        column_index = 0
        for column_name in dataframe_column_names:
            column_list_as_string = column_list_as_string + "[" + column_name + "]"
            if column_index < len(dataframe_column_names) - 1:
                column_list_as_string += ", "
            column_index += 1
        column_list_as_string += " "
        return column_list_as_string

    @classmethod
    def move_records_from_temp_table_to_final(
        cls,
        qualified_source_table_name,
        qualified_target_table_name,
        connection_string,
        qc,
    ):
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
