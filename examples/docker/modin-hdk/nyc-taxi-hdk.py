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

import sys

from utils import measure

import modin.pandas as pd
from modin.experimental.sql import query
from modin.pandas.test.utils import df_equals


def read(filename):
    columns_names = [
        "trip_id",
        "vendor_id",
        "pickup_datetime",
        "dropoff_datetime",
        "store_and_fwd_flag",
        "rate_code_id",
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
        "passenger_count",
        "trip_distance",
        "fare_amount",
        "extra",
        "mta_tax",
        "tip_amount",
        "tolls_amount",
        "ehail_fee",
        "improvement_surcharge",
        "total_amount",
        "payment_type",
        "trip_type",
        "pickup",
        "dropoff",
        "cab_type",
        "precipitation",
        "snow_depth",
        "snowfall",
        "max_temperature",
        "min_temperature",
        "average_wind_speed",
        "pickup_nyct2010_gid",
        "pickup_ctlabel",
        "pickup_borocode",
        "pickup_boroname",
        "pickup_ct2010",
        "pickup_boroct2010",
        "pickup_cdeligibil",
        "pickup_ntacode",
        "pickup_ntaname",
        "pickup_puma",
        "dropoff_nyct2010_gid",
        "dropoff_ctlabel",
        "dropoff_borocode",
        "dropoff_boroname",
        "dropoff_ct2010",
        "dropoff_boroct2010",
        "dropoff_cdeligibil",
        "dropoff_ntacode",
        "dropoff_ntaname",
        "dropoff_puma",
    ]
    # use string instead of category
    columns_types = [
        "int64",
        "string",
        "timestamp",
        "timestamp",
        "string",
        "int64",
        "float64",
        "float64",
        "float64",
        "float64",
        "int64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "string",
        "float64",
        "string",
        "string",
        "string",
        "float64",
        "int64",
        "float64",
        "int64",
        "int64",
        "float64",
        "float64",
        "float64",
        "float64",
        "string",
        "float64",
        "float64",
        "string",
        "string",
        "string",
        "float64",
        "float64",
        "float64",
        "float64",
        "string",
        "float64",
        "float64",
        "string",
        "string",
        "string",
        "float64",
    ]

    dtypes = {columns_names[i]: columns_types[i] for i in range(len(columns_names))}
    all_but_dates = {
        col: valtype
        for (col, valtype) in dtypes.items()
        if valtype not in ["timestamp"]
    }
    dates_only = [col for (col, valtype) in dtypes.items() if valtype in ["timestamp"]]

    df = pd.read_csv(
        filename,
        names=columns_names,
        dtype=all_but_dates,
        parse_dates=dates_only,
    )

    # to trigger real execution and table import
    df._query_compiler._modin_frame.force_import()
    return df


def q1_hdk(df):
    q1_pandas_output = df.groupby("cab_type").size()
    q1_pandas_output.shape  # to trigger real execution
    return q1_pandas_output


def q1_sql(df):
    sql = """
    SELECT
        cab_type,
        COUNT(*) AS 'count'
    FROM trips
    GROUP BY
        cab_type
    """
    return query(sql, trips=df)


def q2_hdk(df):
    q2_pandas_output = df.groupby("passenger_count").agg({"total_amount": "mean"})
    q2_pandas_output.shape  # to trigger real execution
    return q2_pandas_output


def q2_sql(df):
    sql = """
    SELECT
        passenger_count,
        AVG(total_amount) AS 'total_amount'
    FROM trips
    GROUP BY
        passenger_count
    """
    return query(sql, trips=df)


def q3_hdk(df):
    df["pickup_datetime"] = df["pickup_datetime"].dt.year
    q3_pandas_output = df.groupby(["passenger_count", "pickup_datetime"]).size()
    q3_pandas_output.shape  # to trigger real execution
    return q3_pandas_output


def q3_sql(df):
    sql = """
    SELECT
        passenger_count,
        pickup_datetime,
        COUNT(*) AS 'count'
    FROM trips
    GROUP BY
        passenger_count,
        pickup_datetime
    """
    df["pickup_datetime"] = df["pickup_datetime"].dt.year
    return query(sql, trips=df)


def q4_hdk(df):
    df["pickup_datetime"] = df["pickup_datetime"].dt.year
    df["trip_distance"] = df["trip_distance"].astype("int64")
    q4_pandas_output = (
        df.groupby(["passenger_count", "pickup_datetime", "trip_distance"], sort=False)
        .size()
        .reset_index()
        .sort_values(
            by=["pickup_datetime", 0], ignore_index=True, ascending=[True, False]
        )
    )
    q4_pandas_output.shape  # to trigger real execution
    return q4_pandas_output


def q4_sql(df):
    sql = """
    SELECT
        passenger_count,
        pickup_datetime,
        CAST(trip_distance AS int) AS trip_distance,
        COUNT(*) AS the_count
    FROM trips
    GROUP BY
        passenger_count,
        pickup_datetime,
        trip_distance
    ORDER BY
        pickup_datetime,
        the_count desc
    """
    df["pickup_datetime"] = df["pickup_datetime"].dt.year
    df["trip_distance"] = df["trip_distance"].astype("int64")
    return query(sql, trips=df)


def validate(df, hdk_func, sql_func, copy_df=False, reset_index=True, sort_by=None):
    hdk_result = hdk_func(df.copy() if copy_df else df)
    sql_result = sql_func(df.copy() if copy_df else df)
    if reset_index:
        hdk_result = hdk_result.reset_index()
    hdk_result.columns = sql_result.columns
    if sort_by is not None:
        hdk_result = hdk_result.sort_values(by=sort_by)
        sql_result = hdk_result.sort_values(by=sort_by)
    df_equals(hdk_result, sql_result)


def main():
    if len(sys.argv) != 2:
        print(
            f"USAGE: docker run --rm -v /path/to/dataset:/dataset python nyc-taxi-hdk.py <data file name starting with /dataset>"
        )
        return
    df = measure("Reading", read, sys.argv[1])
    measure("Q1H", q1_hdk, df)
    measure("Q1S", q1_sql, df)
    measure("Q2H", q2_hdk, df)
    measure("Q2S", q2_sql, df)
    # The data frame is modified by some tests, therefore a copy should be used for these tests.
    measure("Q3H", q3_hdk, df.copy())
    measure("Q3S", q3_sql, df.copy())
    measure("Q4H", q4_hdk, df.copy())
    measure("Q4S", q4_sql, df.copy())

    validate(df, q1_hdk, q1_sql)
    validate(df, q2_hdk, q2_sql)
    validate(df, q3_hdk, q3_sql, copy_df=True)
    # Additional sorting is required here to make the results identical
    validate(
        df, q4_hdk, q4_sql, copy_df=True, reset_index=False, sort_by=["trip_distance"]
    )


if __name__ == "__main__":
    main()
