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

import sklearnex
from sklearn import config_context

import modin.pandas as pd

sklearnex.patch_sklearn()
import numpy as np
import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split


def read(filename):
    columns_names = [
        "YEAR0",
        "DATANUM",
        "SERIAL",
        "CBSERIAL",
        "HHWT",
        "CPI99",
        "GQ",
        "QGQ",
        "PERNUM",
        "PERWT",
        "SEX",
        "AGE",
        "EDUC",
        "EDUCD",
        "INCTOT",
        "SEX_HEAD",
        "SEX_MOM",
        "SEX_POP",
        "SEX_SP",
        "SEX_MOM2",
        "SEX_POP2",
        "AGE_HEAD",
        "AGE_MOM",
        "AGE_POP",
        "AGE_SP",
        "AGE_MOM2",
        "AGE_POP2",
        "EDUC_HEAD",
        "EDUC_MOM",
        "EDUC_POP",
        "EDUC_SP",
        "EDUC_MOM2",
        "EDUC_POP2",
        "EDUCD_HEAD",
        "EDUCD_MOM",
        "EDUCD_POP",
        "EDUCD_SP",
        "EDUCD_MOM2",
        "EDUCD_POP2",
        "INCTOT_HEAD",
        "INCTOT_MOM",
        "INCTOT_POP",
        "INCTOT_SP",
        "INCTOT_MOM2",
        "INCTOT_POP2",
    ]
    columns_types = [
        "int64",
        "int64",
        "int64",
        "float64",
        "int64",
        "float64",
        "int64",
        "float64",
        "int64",
        "int64",
        "int64",
        "int64",
        "int64",
        "int64",
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
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
    ]
    dtypes = {columns_names[i]: columns_types[i] for i in range(len(columns_names))}

    df = pd.read_csv(
        filename,
        names=columns_names,
        dtype=dtypes,
        skiprows=1,
    )

    return df


def etl(df):
    keep_cols = [
        "YEAR0",
        "DATANUM",
        "SERIAL",
        "CBSERIAL",
        "HHWT",
        "CPI99",
        "GQ",
        "PERNUM",
        "SEX",
        "AGE",
        "INCTOT",
        "EDUC",
        "EDUCD",
        "EDUC_HEAD",
        "EDUC_POP",
        "EDUC_MOM",
        "EDUCD_MOM2",
        "EDUCD_POP2",
        "INCTOT_MOM",
        "INCTOT_POP",
        "INCTOT_MOM2",
        "INCTOT_POP2",
        "INCTOT_HEAD",
        "SEX_HEAD",
    ]
    df = df[keep_cols]

    df = df[df["INCTOT"] != 9999999]
    df = df[df["EDUC"] != -1]
    df = df[df["EDUCD"] != -1]

    df["INCTOT"] = df["INCTOT"] * df["CPI99"]

    for column in keep_cols:
        df[column] = df[column].fillna(-1)

        df[column] = df[column].astype("float64")

    y = df["EDUC"]
    X = df.drop(columns=["EDUC", "CPI99"])

    return (df, X, y)


def mse(y_test, y_pred):
    return ((y_test - y_pred) ** 2).mean()


def cod(y_test, y_pred):
    y_bar = y_test.mean()
    total = ((y_test - y_bar) ** 2).sum()
    residuals = ((y_test - y_pred) ** 2).sum()
    return 1 - (residuals / total)


def ml(X, y, random_state, n_runs, test_size):
    clf = lm.Ridge()

    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)

    mse_values, cod_values = [], []
    ml_scores = {}

    print("ML runs: ", n_runs)
    for i in range(n_runs):
        (X_train, X_test, y_train, y_test) = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        random_state += 777

        with config_context(assume_finite=True):
            model = clf.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse_values.append(mse(y_test, y_pred))
        cod_values.append(cod(y_test, y_pred))

    ml_scores["mse_mean"] = sum(mse_values) / len(mse_values)
    ml_scores["cod_mean"] = sum(cod_values) / len(cod_values)
    ml_scores["mse_dev"] = pow(
        sum([(mse_value - ml_scores["mse_mean"]) ** 2 for mse_value in mse_values])
        / (len(mse_values) - 1),
        0.5,
    )
    ml_scores["cod_dev"] = pow(
        sum([(cod_value - ml_scores["cod_mean"]) ** 2 for cod_value in cod_values])
        / (len(cod_values) - 1),
        0.5,
    )

    return ml_scores


def measure(name, func, *args, **kw):
    t0 = time.time()
    res = func(*args, **kw)
    t1 = time.time()
    print(f"{name}: {t1 - t0} sec")
    return res


def main():
    if len(sys.argv) != 2:
        print(
            f"USAGE: docker run --rm -v /path/to/dataset:/dataset python census.py <data file name starting with /dataset>"
        )
        return
    # ML specific
    N_RUNS = 50
    TEST_SIZE = 0.1
    RANDOM_STATE = 777

    df = measure("Reading", read, sys.argv[1])
    _, X, y = measure("ETL", etl, df)
    measure(
        "ML", ml, X, y, random_state=RANDOM_STATE, n_runs=N_RUNS, test_size=TEST_SIZE
    )


if __name__ == "__main__":
    main()
