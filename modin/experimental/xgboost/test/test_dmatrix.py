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


import pytest

import modin.experimental.xgboost as mxgb
import modin.pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score

import multiprocessing as mp
import ray

import xgboost as xgb


ray.init(log_to_driver=False)

num_cpus = mp.cpu_count()


rng = np.random.RandomState(1994)


@pytest.mark.parametrize(
    "missing",
    [None, 2.8, 0.0, -1.2],
)
@pytest.mark.parametrize(
    "base_margin_val",
    [None, "val"],
)
@pytest.mark.parametrize(
    "weight_val",
    [None, "val"],
)
@pytest.mark.parametrize(
    "num_actors",
    [num_cpus, None],
)
def test_dmatrix(num_actors, weight_val, base_margin_val, missing):
    dataset = load_breast_cancer()
    param = {"objective": "binary:logistic", "eval_metric": ["logloss", "error"]}
    num_round = 10

    X = dataset.data
    y = dataset.target
    w = np.random.rand(X.shape[0], 1) if weight_val == "val" else None
    base_margin = np.random.rand(X.shape[0], 1) if base_margin_val == "val" else None
    dmatrix_xgb = xgb.DMatrix(
        X, label=y, weight=w, base_margin=base_margin, missing=missing
    )

    modin_X = pd.DataFrame(X)
    modin_y = pd.DataFrame(y)
    modin_w = pd.DataFrame(w) if weight_val == "val" else None
    modin_base_margin = pd.DataFrame(base_margin) if base_margin_val == "val" else None
    dmatrix_mxgb = mxgb.DMatrix(
        modin_X,
        label=modin_y,
        weight=modin_w,
        base_margin=modin_base_margin,
        missing=missing,
    )

    evals_result_xgb = {}
    evals_result_mxgb = {}
    verbose_eval = False
    bst = xgb.train(
        param,
        dmatrix_xgb,
        num_round,
        evals_result=evals_result_xgb,
        evals=[(dmatrix_xgb, "train")],
        verbose_eval=verbose_eval,
    )
    modin_bst = mxgb.train(
        param,
        dmatrix_mxgb,
        num_round,
        evals_result=evals_result_mxgb,
        evals=[(dmatrix_mxgb, "train")],
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
                atol=0.2,
            )

    predictions = bst.predict(dmatrix_xgb)
    modin_predictions = modin_bst.predict(dmatrix_mxgb, num_actors=num_actors)

    preds = pd.DataFrame(predictions).apply(lambda x: round(x))
    modin_preds = modin_predictions.apply(lambda x: round(x))

    val = accuracy_score(y, preds)
    modin_val = accuracy_score(modin_y, modin_preds)

    np.testing.assert_allclose(val, modin_val, atol=0.023)


@pytest.mark.parametrize(
    "nthread",
    [None, 2, 0, -1],
)
@pytest.mark.parametrize(
    "silent",
    [None, False, True],
)
@pytest.mark.parametrize(
    "num_actors",
    [num_cpus, None],
)
def test_dmatrix2(num_actors, silent, nthread):
    dataset = load_breast_cancer()
    param = {"objective": "binary:logistic", "eval_metric": ["logloss", "error"]}
    num_round = 10

    X = dataset.data
    y = dataset.target
    dmatrix_xgb = xgb.DMatrix(X, label=y, silent=silent, nthread=nthread)

    modin_X = pd.DataFrame(X)
    modin_y = pd.DataFrame(y)
    dmatrix_mxgb = mxgb.DMatrix(modin_X, label=modin_y, silent=silent, nthread=nthread)

    evals_result_xgb = {}
    evals_result_mxgb = {}
    verbose_eval = False
    bst = xgb.train(
        param,
        dmatrix_xgb,
        num_round,
        evals_result=evals_result_xgb,
        evals=[(dmatrix_xgb, "train")],
        verbose_eval=verbose_eval,
    )
    modin_bst = mxgb.train(
        param,
        dmatrix_mxgb,
        num_round,
        evals_result=evals_result_mxgb,
        evals=[(dmatrix_mxgb, "train")],
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
                atol=0.02,
            )

    predictions = bst.predict(dmatrix_xgb)
    modin_predictions = modin_bst.predict(dmatrix_mxgb, num_actors=num_actors)

    preds = pd.DataFrame(predictions).apply(lambda x: round(x))
    modin_preds = modin_predictions.apply(lambda x: round(x))

    val = accuracy_score(y, preds)
    modin_val = accuracy_score(modin_y, modin_preds)

    np.testing.assert_allclose(val, modin_val, atol=0.004)


class TestDMatrix:
    def test_dmatrix_numpy_init(self):
        data = np.random.randn(5, 5)
        dm = xgb.DMatrix(data)
        md_data = pd.DataFrame(data)
        md_dm = mxgb.DMatrix(md_data)
        assert md_dm.num_row() == dm.num_row() == 5
        assert md_dm.num_col() == dm.num_col() == 5

        data = np.array([[1, 2], [3, 4]])
        dm = xgb.DMatrix(data)
        md_data = pd.DataFrame(data)
        md_dm = mxgb.DMatrix(md_data)
        assert md_dm.num_row() == dm.num_row() == 2
        assert md_dm.num_col() == dm.num_col() == 2

        # 0d array
        with pytest.raises(ValueError):
            xgb.DMatrix(np.array(1))
        with pytest.raises(ValueError):
            mxgb.DMatrix(pd.DataFrame(np.array(1)))
        # 3d array
        data = np.random.randn(5, 5, 5)
        with pytest.raises(ValueError):
            xgb.DMatrix(data)
        with pytest.raises(ValueError):
            mxgb.DMatrix(pd.DataFrame(data))
        # object dtype
        data = np.array([["a", "b"], ["c", "d"]])
        with pytest.raises(ValueError):
            xgb.DMatrix(data)
        with pytest.raises(ValueError):
            mxgb.DMatrix(pd.DataFrame(data))

    def test_np_view(self):
        # Sliced Float32 array
        y = np.array([12, 34, 56], np.float32)[::2]
        md_y = pd.Series(np.array([12, 34, 56], np.float32)[::2])
        from_view = xgb.DMatrix(np.array([[]]), label=y).get_label()
        md_from_view = mxgb.DMatrix(
            pd.DataFrame(np.array([[]])), label=md_y
        ).get_label()
        from_array = xgb.DMatrix(np.array([[]]), label=y + 0).get_label()
        md_from_array = mxgb.DMatrix(
            pd.DataFrame(np.array([[]])), label=md_y + 0
        ).get_label()
        assert (
            md_from_view.shape
            == md_from_array.shape
            == from_view.shape
            == from_array.shape
        )
        assert (from_view == from_array).all()

    def test_slice(self):
        X = rng.randn(100, 100)
        md_X = pd.DataFrame(X)
        y = rng.randint(low=0, high=3, size=100).astype(np.float32)
        md_y = pd.Series(y)
        d = xgb.DMatrix(X, y)
        md_d = mxgb.DMatrix(md_X, md_y)

        np.testing.assert_equal(d.get_label().size, md_d.get_label().size)
        for i in range(d.get_label().size):
            np.testing.assert_equal(d.get_label()[i], md_d.get_label()[i])

        fw = rng.uniform(size=100).astype(np.float32)
        md_fw = pd.DataFrame(fw)
        d.set_info(feature_weights=fw)
        md_d.set_info(feature_weights=md_fw)

        # base margin is per-class in multi-class classifier
        base_margin = rng.randn(100, 3).astype(np.float32)
        d.set_base_margin(base_margin.flatten())
        md_d.set_base_margin(pd.DataFrame(base_margin.flatten()))
