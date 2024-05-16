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
from functools import partial

import numpy as np
import sklearnex
import xgboost as xgb

import modin.pandas as pd

sklearnex.patch_sklearn()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


################ helper functions ###############################
def create_dtypes():
    dtypes = dict(
        [
            ("object_id", "int32"),
            ("mjd", "float32"),
            ("passband", "int32"),
            ("flux", "float32"),
            ("flux_err", "float32"),
            ("detected", "int32"),
        ]
    )

    # load metadata
    columns_names = [
        "object_id",
        "ra",
        "decl",
        "gal_l",
        "gal_b",
        "ddf",
        "hostgal_specz",
        "hostgal_photoz",
        "hostgal_photoz_err",
        "distmod",
        "mwebv",
        "target",
    ]
    meta_dtypes = ["int32"] + ["float32"] * 4 + ["int32"] + ["float32"] * 5 + ["int32"]
    meta_dtypes = dict(
        [(columns_names[i], meta_dtypes[i]) for i in range(len(meta_dtypes))]
    )
    return dtypes, meta_dtypes


def ravel_column_names(cols):
    d0 = cols.get_level_values(0)
    d1 = cols.get_level_values(1)
    return ["%s_%s" % (i, j) for i, j in zip(d0, d1)]


def measure(name, func, *args, **kw):
    t0 = time.time()
    res = func(*args, **kw)
    t1 = time.time()
    print(f"{name}: {t1 - t0} sec")
    return res


def all_etl(train, train_meta, test, test_meta):
    train_final = etl(train, train_meta)
    test_final = etl(test, test_meta)
    return (train_final, test_final)


def split_step(train_final, test_final):
    X = train_final.drop(["object_id", "target"], axis=1).values
    Xt = test_final.drop(["object_id"], axis=1).values

    y = train_final["target"]
    assert X.shape[1] == Xt.shape[1]
    classes = sorted(y.unique())

    class_weights = {c: 1 for c in classes}
    class_weights.update({c: 2 for c in [64, 15]})

    lbl = LabelEncoder()
    y = lbl.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=126
    )

    return X_train, y_train, X_test, y_test, Xt, classes, class_weights


def multi_weighted_logloss(y_true, y_preds, classes, class_weights):
    """
    refactor from
    @author olivier https://www.kaggle.com/ogrellier
    multi logloss for PLAsTiCC challenge
    """
    y_p = y_preds.reshape(y_true.shape[0], len(classes), order="F")
    y_ohe = pd.get_dummies(y_true)
    y_p = np.clip(a=y_p, a_min=1e-15, a_max=1 - 1e-15)
    y_p_log = np.log(y_p)
    y_log_ones = np.sum(y_ohe.values * y_p_log, axis=0)
    nb_pos = y_ohe.sum(axis=0).values.astype(float)
    class_arr = np.array([class_weights[k] for k in sorted(class_weights.keys())])
    y_w = y_log_ones * class_arr / nb_pos

    loss = -np.sum(y_w) / np.sum(class_arr)
    return loss


def xgb_multi_weighted_logloss(y_predicted, y_true, classes, class_weights):
    loss = multi_weighted_logloss(
        y_true.get_label(), y_predicted, classes, class_weights
    )
    return "wloss", loss


################ helper functions ###############################


def read(
    training_set_filename,
    test_set_filename,
    training_set_metadata_filename,
    test_set_metadata_filename,
    dtypes,
    meta_dtypes,
):
    train = pd.read_csv(training_set_filename, dtype=dtypes)
    test = pd.read_csv(
        test_set_filename,
        names=list(dtypes.keys()),
        dtype=dtypes,
        header=0,
    )

    train_meta = pd.read_csv(training_set_metadata_filename, dtype=meta_dtypes)
    target = meta_dtypes.pop("target")
    test_meta = pd.read_csv(test_set_metadata_filename, dtype=meta_dtypes)
    meta_dtypes["target"] = target

    dfs = (train, train_meta, test, test_meta)
    return dfs


def etl(df, df_meta):
    # workaround for Modin_on_ray. Eventually this should be fixed
    df["flux_ratio_sq"] = (df["flux"] / df["flux_err"]) * (
        df["flux"] / df["flux_err"]
    )  # np.power(df["flux"] / df["flux_err"], 2.0)
    df["flux_by_flux_ratio_sq"] = df["flux"] * df["flux_ratio_sq"]

    aggs = {
        "passband": ["mean"],
        "flux": ["min", "max", "mean", "skew"],
        "flux_err": ["min", "max", "mean"],
        "detected": ["mean"],
        "mjd": ["max", "min"],
        "flux_ratio_sq": ["sum"],
        "flux_by_flux_ratio_sq": ["sum"],
    }
    agg_df = df.groupby("object_id", sort=False).agg(aggs)

    agg_df.columns = ravel_column_names(agg_df.columns)

    agg_df["flux_diff"] = agg_df["flux_max"] - agg_df["flux_min"]
    agg_df["flux_dif2"] = agg_df["flux_diff"] / agg_df["flux_mean"]
    agg_df["flux_w_mean"] = (
        agg_df["flux_by_flux_ratio_sq_sum"] / agg_df["flux_ratio_sq_sum"]
    )
    agg_df["flux_dif3"] = agg_df["flux_diff"] / agg_df["flux_w_mean"]
    agg_df["mjd_diff"] = agg_df["mjd_max"] - agg_df["mjd_min"]

    agg_df = agg_df.drop(["mjd_max", "mjd_min"], axis=1)

    agg_df = agg_df.reset_index()

    df_meta = df_meta.drop(["ra", "decl", "gal_l", "gal_b"], axis=1)

    df_meta = df_meta.merge(agg_df, on="object_id", how="left")

    return df_meta


def ml(train_final, test_final):
    X_train, y_train, X_test, y_test, Xt, classes, class_weights = split_step(
        train_final, test_final
    )

    cpu_params = {
        "objective": "multi:softprob",
        "eval_metric": "merror",
        "tree_method": "hist",
        "nthread": 16,
        "num_class": 14,
        "max_depth": 7,
        "verbosity": 1,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
    }

    func_loss = partial(
        xgb_multi_weighted_logloss, classes=classes, class_weights=class_weights
    )

    dtrain = xgb.DMatrix(data=X_train, label=y_train)
    dvalid = xgb.DMatrix(data=X_test, label=y_test)
    dtest = xgb.DMatrix(data=Xt)

    watchlist = [(dvalid, "eval"), (dtrain, "train")]

    clf = xgb.train(
        cpu_params,
        dtrain=dtrain,
        num_boost_round=60,
        evals=watchlist,
        feval=func_loss,
        early_stopping_rounds=10,
        verbose_eval=None,
    )

    yp = clf.predict(dvalid)
    cpu_loss = multi_weighted_logloss(y_test, yp, classes, class_weights)
    ysub = clf.predict(dtest)  # noqa: F841 (unused variable)

    return cpu_loss


def main():
    if len(sys.argv) != 5:
        print(
            f"USAGE: docker run --rm -v /path/to/dataset:/dataset python plasticc.py <training set file name startin with /dataset> <test set file name starting with /dataset> <training set metadata file name starting with /dataset> <test set metadata file name starting with /dataset>"
        )
        return

    dtypes, meta_dtypes = create_dtypes()

    train, train_meta, test, test_meta = measure(
        "Reading",
        read,
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        sys.argv[4],
        dtypes,
        meta_dtypes,
    )
    train_final, test_final = measure(
        "ETL", all_etl, train, train_meta, test, test_meta
    )
    cpu_loss = measure("ML", ml, train_final, test_final)

    print("validation cpu_loss:", cpu_loss)


if __name__ == "__main__":
    main()
