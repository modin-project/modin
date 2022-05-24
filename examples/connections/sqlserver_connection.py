import modin.pandas as pd
import pandas as old
import pyodbc
import sqlalchemy as sa
import urllib
import csv
import time


def encode_value(val):
    return (
        str(val)
        if isinstance(val, int) or isinstance(val, float)
        else "'" + str(val) + "'"
    )


def add_records_pandas(df, alchemy_conn, database_name, schema_name, table_name):
    start = time.time()
    df.to_sql(table_name, alchemy_conn, index=False, chunksize=10000)
    print("time to insert records: ", time.time() - start, " seconds.")


def add_records_executemany(df, cursor, database_name, schema_name, table_name):

    qualified_table_name = (
        "[" + database_name + "]." + "[" + schema_name + "]." + "[" + table_name + "]"
    )
    insert_statement = "INSERT INTO " + qualified_table_name + " ("
    dataframe_column_names = df.columns
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

    cursor.fast_executemany = True

    start_time = time.time()

    cursor.executemany(insert_statement, df.values.tolist())

    print("time for executemany: ", time.time() - start_time)


def add_records_helper(df, database_name, schema_name, table_name):
    dataframe_column_names = df.columns
    qualified_table_name = (
        "[" + database_name + "]." + "[" + schema_name + "]." + "[" + table_name + "]"
    )
    insert_statement_start = "INSERT INTO " + qualified_table_name + " ("
    column_index = 0
    for column_name in dataframe_column_names:
        insert_statement_start = insert_statement_start + "[" + column_name + "]"
        if column_index < len(dataframe_column_names) - 1:
            insert_statement_start += ", "
        column_index += 1
    insert_statement_start += ")"
    insert_statement_start += " VALUES"
    return insert_statement_start


def add_records(
    insert_statement_start,
    df,
    cursor,
    database_name,
    schema_name,
    table_name,
    num_rows,
    start_row=0,
):

    print("reached add_records")
    start_time = time.time()
    insert_statement_elements = []
    insert_statement_elements.append(insert_statement_start)
    count_rows = 0
    cursor.fast_executemany = True

    curr_row = start_row
    row = df.iloc[curr_row]
    while not row.empty:
        insert_statement_elements.append("(")
        column_index = 0
        for val in row:
            insert_statement_elements.append(encode_value(val))
            if column_index < len(row) - 1:
                insert_statement_elements.append(", ")
            column_index += 1
        insert_statement_elements.append(")")
        count_rows += 1
        if count_rows >= num_rows:
            break
        else:
            insert_statement_elements.append(",")
        row = df.iloc[count_rows]

    insert_statement_elements.append(";")
    curr_time = time.time()
    print("time taken to create list of entries: ", curr_time - start_time, "seconds.")
    insert_statement = " ".join(insert_statement_elements)
    print("time taken to assemble batch: ", time.time() - curr_time, " seconds.")

    insert_time = time.time()
    print("invoking cursor.execute")
    try:
        cursor.execute(insert_statement)
    except Exception as inst:
        print(inst)
        print(insert_statement[0:100])
        return
    print("Number of rows inserted in batch: ", count_rows)
    print("time taken for inserting batch = ", time.time() - insert_time, " seconds.")

    return count_rows


flag = 3


def create_database_table(df, table_name):

    dataframe_column_names = df.columns
    dataframe_column_types = df.dtypes

    column_name_type_map = {}

    for column_name, column_type in zip(dataframe_column_names, dataframe_column_types):
        if column_type == "int64":
            column_name_type_map[column_name] = "bigint"
        elif column_type == "float64":
            column_name_type_map[column_name] = "float"
        elif column_type == "object":
            column_name_type_map[column_name] = "nvarchar(1024)"
        else:
            continue

    create_statement = "CREATE TABLE [Ponder].[dbo].[" + table_name + "] ("

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

    print(create_statement)

    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};SERVER=LAPTOP-K2JGSQTK;Trusted_Connection=yes"
    )

    cursor = conn.cursor()
    cursor.fast_executemany = True

    try:
        cursor.execute(create_statement)
    except:
        conn.close()
        return None

    count = len(df)
    print("number of records to insert: ", count)

    if flag == 0:
        count_inserted = 0

        batch_size = 1000

        insert_statement_start = add_records_helper(df, "Ponder", "dbo", table_name)
        print(insert_statement_start)

        try:
            while count_inserted < count:
                rows_inserted = add_records(
                    insert_statement_start,
                    df,
                    cursor,
                    "Ponder",
                    "dbo",
                    table_name,
                    batch_size,
                )
                count_inserted += rows_inserted
        except:
            conn.close()
            return None
    elif flag == 1:
        add_records_executemany(df, cursor, "Ponder", "dbo", table_name)
    elif flag == 2:
        params = urllib.parse.quote_plus(
            "DRIVER={ODBC Driver 17 for SQL Server};"
            "SERVER=LAPTOP-K2JGSQTK;"
            "DATABASE=Ponder;"
            "Trusted_Connection=yes;"
        )
        alchemy_engine_string = "mssql+pyodbc:///?odbc_connect={}".format(params)
        add_records_pandas(df, alchemy_engine_string, "Ponder", "dbo", "to_sql")
    else:
        params = urllib.parse.quote_plus(
            "DRIVER={ODBC Driver 17 for SQL Server};"
            "SERVER=LAPTOP-K2JGSQTK;"
            "DATABASE=Ponder;"
            "Trusted_Connection=yes;"
        )
        alchemy_engine = sa.create_engine(
            "mssql+pyodbc:///?odbc_connect={}".format(params), fast_executemany=True
        )
        pandas_df = df._to_pandas()
        add_records_pandas(pandas_df, alchemy_engine, "Ponder", "dbo", "to_sql")

    cursor.commit()
    conn.close()

    return cursor


modin_dataframe = pd.read_csv(
    "C:\\Ponder\\DataSets\\NYCityBikeShare\\2013-07 - Citi Bike trip data.csv"
)

create_database_table(modin_dataframe, "test_write")
