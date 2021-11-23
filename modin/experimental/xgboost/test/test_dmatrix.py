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
import pytest
from sklearn.metrics import accuracy_score
import xgboost as xgb

import modin.pandas as pd
import modin.experimental.xgboost as mxgb


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


@pytest.mark.parametrize(
    "feature_names",
    [
        ["Feature1", "Feature2", "Feature3", "Feature4", "Feature5"],
        [u"??1", u"??2", u"??3", u"??4", u"??5"],
    ],
)
def test_feature_names(feature_names):
    data = np.random.randn(100, 5)
    label = np.array([0, 1] * 50)

    check_dmatrix(
        data,
        label,
        feature_names=feature_names,
    )

    dm = xgb.DMatrix(data, label=label, feature_names=feature_names)
    md_dm = mxgb.DMatrix(
        pd.DataFrame(data), label=pd.Series(label), feature_names=feature_names
    )

    params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "eta": 0.3,
        "num_class": 3,
    }

    bst = xgb.train(params, dm, num_boost_round=10)
    md_bst = mxgb.train(params, md_dm, num_boost_round=10)

    predictions = bst.predict(dm)
    modin_predictions = md_bst.predict(md_dm)

    preds = np.asarray([np.argmax(line) for line in predictions])
    md_preds = np.asarray([np.argmax(line) for line in modin_predictions.to_numpy()])

    val = accuracy_score(label, preds)
    md_val = accuracy_score(label, md_preds)

    np.testing.assert_allclose(val, md_val, atol=0.02, rtol=0.01)

    dummy = np.random.randn(5, 5)
    dm = xgb.DMatrix(dummy, feature_names=feature_names)
    md_dm = mxgb.DMatrix(pd.DataFrame(dummy), feature_names=feature_names)
    predictions = bst.predict(dm)
    modin_predictions = md_bst.predict(md_dm)

    preds = np.asarray([np.argmax(line) for line in predictions])
    md_preds = np.asarray([np.argmax(line) for line in modin_predictions.to_numpy()])

    assert preds.all() == md_preds.all()

    # different feature names must raises error
    dm = xgb.DMatrix(dummy, feature_names=list("abcde"))
    md_dm = mxgb.DMatrix(pd.DataFrame(dummy), feature_names=list("abcde"))
    with pytest.raises(ValueError):
        bst.predict(dm)
    with pytest.raises(ValueError):
        md_bst.predict(md_dm)


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
    dm.set_info(feature_weights=np.empty((0, 0)))
    md_dm.set_info(feature_weights=np.empty((0, 0)))

    assert (
        dm.get_float_info("feature_weights").shape[0]
        == md_dm.get_float_info("feature_weights").shape[0]
        == 0
    )
