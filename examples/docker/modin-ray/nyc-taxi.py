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
import time

import modin.pandas as pd


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
    parse_dates = ["pickup_datetime", "dropoff_datetime"]
    return pd.read_csv(
        filename, names=columns_names, header=None, parse_dates=parse_dates
    )


def q1(df):
    return df.groupby("cab_type")["cab_type"].count()


def q2(df):
    return df.groupby("passenger_count", as_index=False).mean()[
        ["passenger_count", "total_amount"]
    ]


def q3(df):
    transformed = pd.DataFrame(
        {
            "pickup_datetime": df["pickup_datetime"].dt.year,
            "passenger_count": df["passenger_count"],
        }
    )
    return transformed.groupby(
        ["pickup_datetime", "passenger_count"], as_index=False
    ).size()


def q4(df):
    transformed = pd.DataFrame(
        {
            "passenger_count": df["passenger_count"],
            "pickup_datetime": df["pickup_datetime"].dt.year,
            "trip_distance": df["trip_distance"].astype("int64"),
        }
    )
    return (
        transformed.groupby(
            ["passenger_count", "pickup_datetime", "trip_distance"], as_index=False
        )
        .size()
        .sort_values(by=["pickup_datetime", "size"], ascending=[True, False])
    )


def measure(name, func, *args, **kw):
    t0 = time.time()
    res = func(*args, **kw)
    t1 = time.time()
    print(f"{name}: {t1 - t0} sec")
    return res


def main():
    if len(sys.argv) != 2:
        print(
            f"USAGE: docker run --rm -v /path/to/dataset:/dataset python nyc-taxi.py <data file name starting with /dataset>"
        )
        return
    df = measure("Reading", read, sys.argv[1])
    measure("Q1", q1, df)
    measure("Q2", q2, df)
    measure("Q3", q3, df)
    measure("Q4", q4, df)


if __name__ == "__main__":
    main()
