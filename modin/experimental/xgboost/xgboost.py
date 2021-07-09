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

"""Module holds public interfaces for work Modin XGBoost."""

import logging
from typing import Dict, Optional

import xgboost as xgb

from modin.config import Engine
from modin.distributed.dataframe.pandas import unwrap_partitions
import modin.pandas as pd

LOGGER = logging.getLogger("[modin.xgboost]")


class DMatrix(xgb.DMatrix):
    """
    DMatrix holds references to partitions of Modin DataFrame.

    On init stage unwrapping partitions of Modin DataFrame is started.

    Parameters
    ----------
    data : modin.pandas.DataFrame
        Data source of DMatrix.
    label : modin.pandas.DataFrame or modin.pandas.Series
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

        self.data = unwrap_partitions(data, axis=0, get_ip=True)
        self.label = unwrap_partitions(label, axis=0)

        self.data_metainfo = (
            data.index,
            data.columns,
            data._query_compiler._modin_frame._row_lengths,
        )

    def __iter__(self):
        """
        Return unwrapped `self.data` and `self.label`.

        Yields
        ------
        list
            List of `self.data` with pairs of references to IP of row partition
            and row partition [(IP_ref0, partition_ref0), ..].
        list
            List of `self.label` with references to row partitions
            [partition_ref0, ..].
        """
        yield self.data
        yield self.label


class Booster(xgb.Booster):
    """
    A Modin Booster of XGBoost.

    Booster is the model of XGBoost, that contains low level routines for
    training, prediction and evaluation.

    Parameters
    ----------
    params : dict, optional
        Parameters for boosters.
    cache : list, default: empty
        List of cache items.
    model_file : string/os.PathLike/xgb.Booster/bytearray, optional
        Path to the model file if it's string or PathLike or xgb.Booster.
    """

    def __init__(self, params=None, cache=(), model_file=None):  # noqa: MD01
        super(Booster, self).__init__(params=params, cache=cache, model_file=model_file)

    def predict(
        self,
        data: DMatrix,
        **kwargs,
    ):
        """
        Run distributed prediction with a trained booster.

        During work it runs xgb.predict on each worker for row partition of `data`
        and creates Modin DataFrame with prediction results.

        Parameters
        ----------
        data : modin.experimental.xgboost.DMatrix
            Input data used for prediction.
        **kwargs : dict
            Other parameters are the same as `xgboost.Booster.predict`.

        Returns
        -------
        modin.pandas.DataFrame
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

        result = _predict(self.copy(), data, **kwargs)
        LOGGER.info("Prediction finished")

        return result


def train(
    params: Dict,
    dtrain: DMatrix,
    *args,
    evals=(),
    num_actors: Optional[int] = None,
    evals_result: Optional[Dict] = None,
    **kwargs,
):
    """
    Run distributed training of XGBoost model.

    During work it evenly distributes `dtrain` between workers according
    to IP addresses partitions (in case of not even distribution of `dtrain`
    over nodes, some partitions will be re-distributed between nodes),
    runs xgb.train on each worker for subset of `dtrain` and reduces training results
    of each worker using Rabit Context.

    Parameters
    ----------
    params : dict
        Booster params.
    dtrain : modin.experimental.xgboost.DMatrix
        Data to be trained against.
    *args : iterable
        Other parameters for `xgboost.train`.
    evals : list of pairs (modin.experimental.xgboost.DMatrix, str), default: empty
        List of validation sets for which metrics will evaluated during training.
        Validation metrics will help us track the performance of the model.
    num_actors : int, optional
        Number of actors for training. If unspecified, this value will be
        computed automatically.
    evals_result : dict, optional
        Dict to store evaluation results in.
    **kwargs : dict
        Other parameters are the same as `xgboost.train`.

    Returns
    -------
    modin.experimental.xgboost.Booster
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
    result = _train(dtrain, num_actors, params, *args, evals=evals, **kwargs)
    if isinstance(evals_result, dict):
        evals_result.update(result["history"])

    LOGGER.info("Training finished")
    return Booster(model_file=result["booster"])
