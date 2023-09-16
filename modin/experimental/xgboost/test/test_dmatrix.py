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

import numpy as np
import pandas
import pytest
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

import modin.experimental.xgboost as mxgb
import modin.pandas as pd
from modin.config import Engine
from modin.utils import try_cast_to_pandas

if Engine.get() != "Ray":
    pytest.skip(
        "Modin' xgboost extension works only with Ray engine.",
        allow_module_level=True,
    )


rng = np.random.RandomState(1994)


def check_dmatrix(data, label=None, **kwargs):
    modin_data = pd.DataFrame(data)
    modin_label = label if label is None else pd.Series(label)
    try:
        dm = xgb.DMatrix(data, label=label, **kwargs)
    except Exception as xgb_exception:
        with pytest.raises(Exception) as mxgb_exception:
            mxgb.DMatrix(modin_data, label=modin_label, **kwargs)
        # Thrown exceptions are `XGBoostError`, which is a descendant of `ValueError`, and `ValueError`
        # for XGBoost and Modin, respectively,  so we intentionally use `xgb_exception`
        # as a first parameter of `isinstance` to pass the assertion
        assert isinstance(
            xgb_exception, type(mxgb_exception.value)
        ), "Got Modin Exception type {}, but xgboost Exception type {} was expected".format(
            type(mxgb_exception.value), type(xgb_exception)
        )
    else:
        md_dm = mxgb.DMatrix(modin_data, label=modin_label, **kwargs)
        assert md_dm.num_row() == dm.num_row()
        assert md_dm.num_col() == dm.num_col()
        assert md_dm.feature_names == dm.feature_names
        assert md_dm.feature_types == dm.feature_types


@pytest.mark.parametrize(
    "data",
    [
        np.random.randn(5, 5),
        np.array([[1, 2], [3, 4]]),
        np.array([["a", "b"], ["c", "d"]]),
        [[1, 2], [3, 4]],
        [["a", "b"], ["c", "d"]],
    ],
)
@pytest.mark.parametrize(
    "feature_names",
    [
        list("abcdef"),
        ["a", "b", "c", "d", "d"],
        ["a", "b", "c", "d", "e<1"],
        list("abcde"),
    ],
)
@pytest.mark.parametrize(
    "feature_types",
    [None, "q", list("qiqiq")],
)
def test_dmatrix_feature_names_and_feature_types(data, feature_names, feature_types):
    check_dmatrix(data, feature_names=feature_names, feature_types=feature_types)


@pytest.mark.skipif(
    Engine.get() != "Ray",
    reason="implemented only for Ray engine.",
)
def test_feature_names():
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target
    feature_names = [f"feat{i}" for i in range(X.shape[1])]

    check_dmatrix(
        X,
        y,
        feature_names=feature_names,
    )

    dmatrix = xgb.DMatrix(X, label=y, feature_names=feature_names)
    md_dmatrix = mxgb.DMatrix(
        pd.DataFrame(X), label=pd.Series(y), feature_names=feature_names
    )

    params = {
        "objective": "binary:logistic",
        "eval_metric": "mlogloss",
    }

    booster = xgb.train(params, dmatrix, num_boost_round=10)
    md_booster = mxgb.train(params, md_dmatrix, num_boost_round=10)

    predictions = booster.predict(dmatrix)
    modin_predictions = md_booster.predict(md_dmatrix)

    preds = pandas.DataFrame(predictions).apply(np.round, axis=0)
    modin_preds = modin_predictions.apply(np.round, axis=0)

    accuracy = accuracy_score(y, preds)
    md_accuracy = accuracy_score(y, modin_preds)

    np.testing.assert_allclose(accuracy, md_accuracy, atol=0.005, rtol=0.002)

    # Different feature_names (default) must raise error in this case
    dm = xgb.DMatrix(X)
    md_dm = mxgb.DMatrix(pd.DataFrame(X))
    with pytest.raises(ValueError):
        booster.predict(dm)
    with pytest.raises(ValueError):
        try_cast_to_pandas(md_booster.predict(md_dm))  # force materialization


def test_feature_weights():
    n_rows = 10
    n_cols = 50
    fw = rng.uniform(size=n_cols)
    X = rng.randn(n_rows, n_cols)
    dm = xgb.DMatrix(X)
    md_dm = mxgb.DMatrix(pd.DataFrame(X))
    dm.set_info(feature_weights=fw)
    md_dm.set_info(feature_weights=fw)
    np.testing.assert_allclose(
        dm.get_float_info("feature_weights"), md_dm.get_float_info("feature_weights")
    )
    # Handle empty
    dm.set_info(feature_weights=np.empty((0,)))
    md_dm.set_info(feature_weights=np.empty((0,)))

    assert (
        dm.get_float_info("feature_weights").shape[0]
        == md_dm.get_float_info("feature_weights").shape[0]
        == 0
    )
