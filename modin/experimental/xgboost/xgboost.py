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


import logging
from typing import Dict, Optional
from multiprocessing import cpu_count

import xgboost as xgb

from modin.config import Engine
import modin.pandas as pd

LOGGER = logging.getLogger("[modin.xgboost]")


class DMatrix(xgb.DMatrix):
    """
    DMatrix holding on references to DataFrame.

    Parameters
    ----------
    data : DataFrame
        Data source of DMatrix.
    label : DataFrame
        Labels used for training.

    Notes
    -----
    Currently DMatrix supports only `data` and `label` parameters.
    """

    def __init__(self, data, label):
        assert isinstance(
            data, pd.DataFrame
        ), f"Type of `data` is {type(data)}, but expected {pd.DataFrame}."
        assert isinstance(
            label, (pd.DataFrame, pd.Series)
        ), f"Type of `data` is {type(label)}, but expected {pd.DataFrame} or {pd.Series}."

        self.data = data
        self.label = label

    def __iter__(self):
        yield self.data
        yield self.label


class Booster(xgb.Booster):
    """
    A Modin Booster of XGBoost.

    Booster is the model of xgboost, that contains low level routines for
    training, prediction and evaluation.

    Parameters
    ----------
    params : dict. Default is None
        Parameters for boosters.
    cache : list
        List of cache items.
    model_file : string/os.PathLike/Booster/bytearray
        Path to the model file if it's string or PathLike.
    """

    def __init__(self, params=None, cache=(), model_file=None):
        super(Booster, self).__init__(params=params, cache=cache, model_file=model_file)

    def predict(
        self,
        data: DMatrix,
        nthread: Optional[int] = cpu_count(),
        **kwargs,
    ):
        """
        Run prediction with a trained booster.

        Parameters
        ----------
        data : DMatrix
            Input data used for prediction.
        nthread : int. Default is number of threads on master node
            Number of threads for using in each node.
        \\*\\*kwargs :
            Other parameters are the same as `xgboost.Booster.predict`.

        Returns
        -------
        ``modin.pandas.DataFrame``
            Modin DataFrame with prediction results.
        """
        LOGGER.info("Prediction started")

        if Engine.get() == "Ray":
            from .xgboost_ray import _predict
        else:
            raise ValueError("Current version supports only Ray engine.")

        assert isinstance(
            data, DMatrix
        ), f"Type of `data` is {type(data)}, but expected {DMatrix}."

        result = _predict(self.copy(), data, nthread, **kwargs)
        LOGGER.info("Prediction finished")

        return result


def train(
    params: Dict,
    dtrain: DMatrix,
    *args,
    evals=(),
    nthread: Optional[int] = cpu_count(),
    evals_result: Optional[Dict] = None,
    **kwargs,
):
    """
    Train XGBoost model.

    Parameters
    ----------
    params : dict
        Booster params.
    dtrain : DMatrix
        Data to be trained against.
    evals: list of pairs (DMatrix, string)
        List of validation sets for which metrics will evaluated during training.
        Validation metrics will help us track the performance of the model.
    nthread : int. Default is number of threads on master node
        Number of threads for using in each node.
    evals_result : dict. Default is None
        Dict to store evaluation results in.
    \\*\\*kwargs :
        Other parameters are the same as `xgboost.train`.

    Returns
    -------
    ``modin.experimental.xgboost.Booster``
        A trained booster.
    """
    LOGGER.info("Training started")

    if Engine.get() == "Ray":
        from .xgboost_ray import _train
    else:
        raise ValueError("Current version supports only Ray engine.")

    assert isinstance(
        dtrain, DMatrix
    ), f"Type of `dtrain` is {type(dtrain)}, but expected {DMatrix}."
    result = _train(dtrain, nthread, params, *args, evals=evals, **kwargs)
    if isinstance(evals_result, dict):
        evals_result.update(result["history"])

    LOGGER.info("Training finished")
    return Booster(model_file=result["booster"])
