import modin.pandas as pd
import pandas as old_pd
import csv
import time
import urllib
import sqlalchemy as sa
import ray
import sys


# if len(sys.argv) <= 1:
#    print("Usage: ", sys.argv[0], "<one of 0, 1 or 2>")
#    sys.exit()


flag = 0

if flag == 0:

    ray.init()

    modin_dataframe = pd.read_csv(
        "C:\\Ponder\\DataSets\\NYCityBikeShare\\2013-07 - Citi Bike trip data.csv"
    )

    print("finished reading ", len(modin_dataframe), " records.")
    connection_params = {}

    connection_params["database_class"] = "mssql"
    connection_params["database_name"] = "Ponder"
    connection_params["schema_name"] = "dbo"
    connection_params[
        "connection_string"
    ] = "DRIVER={ODBC Driver 17 for SQL Server};SERVER=LAPTOP-K2JGSQTK;Trusted_Connection=yes"

    start_time = time.time()
    print("starting to write modin dataframe to sql server using to_database")
    modin_dataframe.to_database(
        connection_params=connection_params, name="TestModin", if_exists="replace"
    )

    print(
        "Modin with to_database: time to write ",
        len(modin_dataframe),
        " records to sql server: ",
        time.time() - start_time,
    )

    del modin_dataframe

    sys.exit(0)

elif flag == 1:

    pandas_dataframe = old_pd.read_csv(
        "C:\\Ponder\\DataSets\\NYCityBikeShare\\2013-07 - Citi Bike trip data.csv"
    )
    print("finished reading ", len(pandas_dataframe), " records.")
    params = urllib.parse.quote_plus(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=LAPTOP-K2JGSQTK;"
        "DATABASE=Ponder;"
        "Trusted_Connection=yes;"
    )
    alchemy_engine = sa.create_engine(
        "mssql+pyodbc:///?odbc_connect={}".format(params), fast_executemany=True
    )
    print("starting to write pandas dataframe to sql server using to_sql")
    start_time = time.time()
    pandas_dataframe.to_sql(
        "TestPandas", alchemy_engine, index=False, chunksize=10000, if_exists="replace"
    )

    print(
        "Pandas: time to write ",
        len(pandas_dataframe),
        " records to sql server: ",
        time.time() - start_time,
    )

    del pandas_dataframe

    sys.exit()

else:
    modin_dataframe = pd.read_csv(
        "C:\\Ponder\\DataSets\\NYCityBikeShare\\2013-07 - Citi Bike trip data.csv"
    )
    print("finished reading ", len(modin_dataframe), " records.")
    params = urllib.parse.quote_plus(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=LAPTOP-K2JGSQTK;"
        "DATABASE=Ponder;"
        "Trusted_Connection=yes;"
    )
    alchemy_engine_string = "mssql+pyodbc:///?odbc_connect={}".format(params)
    start_time = time.time()
    print("starting to write modin dataframe to sql server using to_sql")
    modin_dataframe.to_sql(
        "ModinToSql",
        alchemy_engine_string,
        index=False,
        chunksize=10000,
        if_exists="replace",
    )
    print(
        "Modin with to_sql: time to write ",
        len(modin_dataframe),
        " records to sql server: ",
        time.time() - start_time,
    )

    del modin_dataframe

    sys.exit(0)

