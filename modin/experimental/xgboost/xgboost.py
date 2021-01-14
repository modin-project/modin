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

if Engine.get() == "Ray":
    from .xgboost_ray import _train, _predict
else:
    raise ValueError("Current version supports only Ray engine as MODIN_ENGINE.")

LOGGER = logging.getLogger("[modin.xgboost]")


class ModinDMatrix(xgb.DMatrix):
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
    Currently ModinDMatrix supports only `data` and `label` parameters.
    """

    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __iter__(self):
        yield self.data
        yield self.label


def train(
    params: Dict,
    dtrain: ModinDMatrix,
    *args,
    evals=(),
    nthread: Optional[int] = cpu_count(),
    evenly_data_distribution: Optional[bool] = True,
    **kwargs,
):
    """
    Train XGBoost model.

    Parameters
    ----------
    params : dict
        Booster params.
    dtrain : ModinDMatrix
        Data to be trained against.
    evals: list of pairs (ModinDMatrix, string)
        List of validation sets for which metrics will evaluated during training.
        Validation metrics will help us track the performance of the model.
    nthread : int
        Number of threads for using in each node. By default it is equal to
        number of threads on master node.
    evenly_data_distribution : boolean, default True
        Whether make evenly distribution of partitions between nodes or not.
        In case `False` minimal datatransfer between nodes will be provided
        but the data may not be evenly distributed.
    \\*\\*kwargs :
        Other parameters are the same as `xgboost.train` except for
        `evals_result`, which is returned as part of function return value
        instead of argument.

    Returns
    -------
    dict
        A dictionary containing trained booster and evaluation history.
        `history` field is the same as `eval_result` from `xgboost.train`.

        .. code-block:: python

            {'booster': xgboost.Booster,
             'history': {'train': {'logloss': ['0.48253', '0.35953']},
                         'eval': {'logloss': ['0.480385', '0.357756']}}}
    """
    LOGGER.info("Training started")
    result = _train(
        dtrain, nthread, evenly_data_distribution, params, *args, evals=evals, **kwargs
    )
    LOGGER.info("Training finished")
    return result


def predict(
    model,
    data: ModinDMatrix,
    nthread: Optional[int] = cpu_count(),
    evenly_data_distribution: Optional[bool] = True,
    **kwargs,
):
    """
    Run prediction with a trained booster.

    Parameters
    ----------
    model : A Booster or a dictionary returned by `modin.experimental.xgboost.train`.
        The trained model.
    data : ModinDMatrix.
        Input data used for prediction.
    nthread : int
        Number of threads for using in each node. By default it is equal to
        number of threads on master node.
    evenly_data_distribution : boolean, default True
        Whether make evenly distribution of partitions between nodes or not.
        In case `False` minimal datatransfer between nodes will be provided
        but the data may not be evenly distributed.

    Returns
    -------
    numpy.array
        Array with prediction results.
    """
    LOGGER.info("Prediction started")

    if isinstance(model, xgb.Booster):
        booster = model
    elif isinstance(model, dict):
        booster = model["booster"]
    else:
        raise TypeError(
            f"Expected types for `model` xgb.Booster or dict, but presented type is {type(model)}"
        )
    result = _predict(booster, data, nthread, evenly_data_distribution, **kwargs)
    LOGGER.info("Prediction finished")

    return result
