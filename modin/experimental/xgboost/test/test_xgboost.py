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


import multiprocessing as mp

import numpy as np
import pytest
import ray
import xgboost
from sklearn.datasets import (
    load_breast_cancer,
    load_diabetes,
    load_digits,
    load_iris,
    load_wine,
)
from sklearn.metrics import accuracy_score, mean_squared_error

import modin
import modin.experimental.xgboost as xgb
import modin.pandas as pd
from modin.config import Engine
from modin.experimental.sklearn.model_selection.train_test_split import train_test_split

if Engine.get() != "Ray":
    pytest.skip("Implemented only for Ray engine.", allow_module_level=True)

ray.init(log_to_driver=False)

num_cpus = mp.cpu_count()


@pytest.mark.parametrize(
    "modin_type_y",
    [pd.DataFrame, pd.Series],
)
@pytest.mark.parametrize(
    "num_actors",
    [1, num_cpus, None, modin.config.NPartitions.get() + 1],
)
@pytest.mark.parametrize(
    "data",
    [
        (
            load_breast_cancer(),
            {"objective": "binary:logistic", "eval_metric": ["logloss", "error"]},
        ),
    ],
    ids=["load_breast_cancer"],
)
def test_xgb_with_binary_classification_datasets(data, num_actors, modin_type_y):
    dataset, param = data
    num_round = 10

    X = dataset.data
    y = dataset.target
    xgb_dmatrix = xgboost.DMatrix(X, label=y)

    modin_X = pd.DataFrame(X)
    modin_y = modin_type_y(y)
    mxgb_dmatrix = xgb.DMatrix(modin_X, label=modin_y)

    evals_result_xgb = {}
    evals_result_mxgb = {}
    verbose_eval = False
    bst = xgboost.train(
        param,
        xgb_dmatrix,
        num_round,
        evals_result=evals_result_xgb,
        evals=[(xgb_dmatrix, "train")],
        verbose_eval=verbose_eval,
    )
    modin_bst = xgb.train(
        param,
        mxgb_dmatrix,
        num_round,
        evals_result=evals_result_mxgb,
        evals=[(mxgb_dmatrix, "train")],
        num_actors=num_actors,
        verbose_eval=verbose_eval,
    )

    for par in param["eval_metric"]:
        assert len(evals_result_xgb["train"][par]) == len(
            evals_result_xgb["train"][par]
        )
        for i in range(len(evals_result_xgb["train"][par])):
            np.testing.assert_allclose(
                evals_result_xgb["train"][par][i],
                evals_result_mxgb["train"][par][i],
                atol=0.011,
            )

    predictions = bst.predict(xgb_dmatrix)
    modin_predictions = modin_bst.predict(mxgb_dmatrix)

    preds = pd.DataFrame(predictions).apply(round)
    modin_preds = modin_predictions.apply(round)

    val = accuracy_score(y, preds)
    modin_val = accuracy_score(modin_y, modin_preds)

    np.testing.assert_allclose(val, modin_val, atol=0.002, rtol=0.002)


@pytest.mark.parametrize(
    "modin_type_y",
    [pd.DataFrame, pd.Series],
)
@pytest.mark.parametrize(
    "num_actors",
    [1, num_cpus, None, modin.config.NPartitions.get() + 1],
)
@pytest.mark.parametrize(
    "data",
    [
        (
            load_iris(),
            {"num_class": 3},
        ),
        (
            load_digits(),
            {"num_class": 10},
        ),
        (
            load_wine(),
            {"num_class": 3},
        ),
    ],
    ids=["load_iris", "load_digits", "load_wine"],
)
def test_xgb_with_multiclass_classification_datasets(data, num_actors, modin_type_y):
    dataset, param_ = data
    num_round = 10
    part_param = {"objective": "multi:softprob", "eval_metric": "mlogloss"}
    param = {**param_, **part_param}

    X = dataset.data
    y = dataset.target
    xgb_dmatrix = xgboost.DMatrix(X, label=y)

    modin_X = pd.DataFrame(X)
    modin_y = modin_type_y(y)
    mxgb_dmatrix = xgb.DMatrix(modin_X, label=modin_y)

    evals_result_xgb = {}
    evals_result_mxgb = {}
    verbose_eval = False
    bst = xgboost.train(
        param,
        xgb_dmatrix,
        num_round,
        evals_result=evals_result_xgb,
        evals=[(xgb_dmatrix, "train")],
        verbose_eval=verbose_eval,
    )
    modin_bst = xgb.train(
        param,
        mxgb_dmatrix,
        num_round,
        evals_result=evals_result_mxgb,
        evals=[(mxgb_dmatrix, "train")],
        num_actors=num_actors,
        verbose_eval=verbose_eval,
    )

    assert len(evals_result_xgb["train"]["mlogloss"]) == len(
        evals_result_mxgb["train"]["mlogloss"]
    )
    for i in range(len(evals_result_xgb["train"]["mlogloss"])):
        np.testing.assert_allclose(
            evals_result_xgb["train"]["mlogloss"][i],
            evals_result_mxgb["train"]["mlogloss"][i],
            atol=0.009,
        )

    predictions = bst.predict(xgb_dmatrix)
    modin_predictions = modin_bst.predict(mxgb_dmatrix)

    array_preds = np.asarray([np.argmax(line) for line in predictions])
    modin_array_preds = np.asarray(
        [np.argmax(line) for line in modin_predictions.to_numpy()]
    )

    val = accuracy_score(y, array_preds)
    modin_val = accuracy_score(modin_y, modin_array_preds)

    np.testing.assert_allclose(val, modin_val)


@pytest.mark.parametrize(
    "modin_type_y",
    [pd.DataFrame, pd.Series],
)
@pytest.mark.parametrize(
    "num_actors",
    [1, num_cpus, None, modin.config.NPartitions.get() + 1],
)
@pytest.mark.parametrize(
    "data",
    [(load_diabetes(), {"eta": 0.01})],
    ids=["load_diabetes"],
)
def test_xgb_with_regression_datasets(data, num_actors, modin_type_y):
    dataset, param = data
    num_round = 10

    X_df = pd.DataFrame(dataset.data)
    y_df = modin_type_y(dataset.target)
    X_train, X_test = train_test_split(X_df)
    y_train, y_test = train_test_split(y_df)

    train_xgb_dmatrix = xgboost.DMatrix(X_train, label=y_train)
    test_xgb_dmatrix = xgboost.DMatrix(X_test, label=y_test)

    train_mxgb_dmatrix = xgb.DMatrix(X_train, label=y_train)
    test_mxgb_dmatrix = xgb.DMatrix(X_test, label=y_test)

    evals_result_xgb = {}
    evals_result_mxgb = {}
    verbose_eval = False
    bst = xgboost.train(
        param,
        train_xgb_dmatrix,
        num_round,
        evals_result=evals_result_xgb,
        evals=[(train_xgb_dmatrix, "train"), (test_xgb_dmatrix, "test")],
        verbose_eval=verbose_eval,
    )
    modin_bst = xgb.train(
        param,
        train_mxgb_dmatrix,
        num_round,
        evals_result=evals_result_mxgb,
        evals=[(train_mxgb_dmatrix, "train"), (test_mxgb_dmatrix, "test")],
        num_actors=num_actors,
        verbose_eval=verbose_eval,
    )

    for param in ["train", "test"]:
        assert len(evals_result_xgb[param]["rmse"]) == len(
            evals_result_mxgb[param]["rmse"]
        )
        for i in range(len(evals_result_xgb[param]["rmse"])):
            np.testing.assert_allclose(
                evals_result_xgb[param]["rmse"][i],
                evals_result_mxgb[param]["rmse"][i],
                rtol=0.0007,
            )

    predictions = bst.predict(train_xgb_dmatrix)
    modin_predictions = modin_bst.predict(train_mxgb_dmatrix)

    val = mean_squared_error(y_train, predictions)
    modin_val = mean_squared_error(y_train, modin_predictions)

    np.testing.assert_allclose(val, modin_val, rtol=1.25e-05)


def test_invalid_input():
    list_df = [[1, 2.0, True], [2, 3.0, False]]
    with pytest.raises(AssertionError):
        # Check that DMatrix uses only DataFrame
        xgb.DMatrix(list_df, label=pd.Series([1, 2]))

    param = {}
    num_round = 2
    with pytest.raises(AssertionError):
        # Check that train uses only DMatrix
        xgb.train(param, list_df, num_round)

    df = pd.DataFrame([[1, 2.0, True], [2, 3.0, False]], columns=["a", "b", "c"])
    modin_dtrain = xgb.DMatrix(df, label=pd.Series([1, 2]))

    modin_bst = xgb.train(param, modin_dtrain, num_round)

    dt = [[1, 2.0, 3.3], [2, 3.0, 4.4]]

    with pytest.raises(AssertionError):
        # Check that predict uses only DMatrix
        modin_bst.predict(dt)
