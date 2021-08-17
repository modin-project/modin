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
import numpy as np

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

    def __init__(
        self,
        data,
        label=None,
        *args,
        weight=None,
        base_margin=None,
        missing: Optional[float] = None,
        silent=False,
        feature_names=None,
        feature_types=None,
        nthread: Optional[int] = None,
        group=None,
        qid=None,
        label_lower_bound=None,
        label_upper_bound=None,
        feature_weights=None,
        enable_categorical: bool = False,
        **kwargs,
    ):
        assert isinstance(
            data, pd.DataFrame
        ), f"Type of `data` is {type(data)}, but expected {pd.DataFrame}."
        if label is not None:
            assert isinstance(
                label, (pd.DataFrame, pd.Series)
            ), f"Type of `label` is {type(label)}, but expected {pd.DataFrame} or {pd.Series}."
        self.label = unwrap_partitions(label, axis=0) if label is not None else label
        self.label_ = label
        if weight is not None:
            assert isinstance(
                weight, (pd.DataFrame, pd.Series)
            ), f"Type of `weight` is {type(weight)}, but expected {pd.DataFrame} or {pd.Series}."
            self.weight = unwrap_partitions(weight, axis=0)

        if len(data.shape) != 2:
            raise ValueError("Expecting 2 dimensional, got: ", data.shape)

        self.rows = data.shape[0]
        self.cols = data.shape[1]

        for i in range(self.cols):
            if data.dtypes[i] == "object":
                raise ValueError("Cannot work with object dtype")

        self.data = unwrap_partitions(data, axis=0, get_ip=True)
        self.data_ = data

        self.missing = missing if missing is not None else np.nan
        self.nthread = nthread if nthread is not None else -1
        self.handle = None

        self.silent = silent

        if feature_names is not None:
            self.feature_names = feature_names
        if feature_types is not None:
            self.feature_types = feature_types

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

    def num_row(self):
        """Get the number of rows in the DMatrix.
        Returns
        -------
        number of rows : int
        """
        return self.rows

    def num_col(self):
        """Get the number of columns (features) in the DMatrix.
        Returns
        -------
        number of columns : int
        """
        return self.cols

    def get_label(self):
        """Get the label of the DMatrix.
        Returns
        -------
        label : array
        """
        return self.label_

    def set_base_margin(self, base_margin):
        """Set base margin of booster to start from."""
        self.base_margin = base_margin

    def slice(self, rindex):
        res_list = self.data_[rindex]
        res = DMatrix(pd.DataFrame([float(i) for i in res_list]))
        return res

    def set_info(
        self,
        *,
        label=None,
        weight=None,
        base_margin=None,
        group=None,
        qid=None,
        label_lower_bound=None,
        label_upper_bound=None,
        feature_names=None,
        feature_types=None,
        feature_weights=None,
    ):

        if label is not None:
            self.label = label
        if weight is not None:
            self.weight
        if base_margin is not None:
            self.set_base_margin(base_margin)
        if group is not None:
            self.group = group
        if qid is not None:
            self.qid = qid
        if label_lower_bound is not None:
            self.label_lower_bound = label_lower_bound
        if label_upper_bound is not None:
            self.label_upper_bound = label_upper_bound
        if feature_names is not None:
            self.feature_names = feature_names
        if feature_types is not None:
            self.feature_types = feature_types
        if feature_weights is not None:
            feature_weights = feature_weights


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
        num_actors: Optional[int] = None,
        **kwargs,
    ):
        """
        Run distributed prediction with a trained booster.

        During work it evenly distributes `data` between workers,
        runs xgb.predict on each worker for subset of `data` and creates
        Modin DataFrame with prediction results.

        Parameters
        ----------
        data : modin.experimental.xgboost.DMatrix
            Input data used for prediction.
        num_actors : int, optional
            Number of actors for prediction. If unspecified, this value will be
            computed automatically.
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

        result = _predict(self.copy(), data, num_actors, **kwargs)
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
