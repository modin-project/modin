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

import modin.pandas as pd
from modin.config import Engine
from modin.distributed.dataframe.pandas import unwrap_partitions

LOGGER = logging.getLogger("[modin.xgboost]")


class DMatrix:
    """
    DMatrix holds references to partitions of Modin DataFrame.

    On init stage unwrapping partitions of Modin DataFrame is started.

    Parameters
    ----------
    data : modin.pandas.DataFrame
        Data source of DMatrix.
    label : modin.pandas.DataFrame or modin.pandas.Series, optional
        Labels used for training.
    missing : float, optional
        Value in the input data which needs to be present as a missing
        value. If ``None``, defaults to ``np.nan``.
    silent : boolean, optional
        Whether to print messages during construction or not.
    feature_names : list, optional
        Set names for features.
    feature_types : list, optional
        Set types for features.
    feature_weights : array_like, optional
        Set feature weights for column sampling.
    enable_categorical : boolean, optional
        Experimental support of specializing for categorical features.

    Notes
    -----
    Currently DMatrix doesn't support `weight`, `base_margin`, `nthread`,
    `group`, `qid`, `label_lower_bound`, `label_upper_bound` parameters.
    """

    def __init__(
        self,
        data,
        label=None,
        missing=None,
        silent=False,
        feature_names=None,
        feature_types=None,
        feature_weights=None,
        enable_categorical=None,
    ):
        assert isinstance(
            data, pd.DataFrame
        ), f"Type of `data` is {type(data)}, but expected {pd.DataFrame}."

        if label is not None:
            assert isinstance(
                label, (pd.DataFrame, pd.Series)
            ), f"Type of `data` is {type(label)}, but expected {pd.DataFrame} or {pd.Series}."
            self.label = unwrap_partitions(label, axis=0)
        else:
            self.label = None

        self.data = unwrap_partitions(data, axis=0, get_ip=True)

        self._n_rows = data.shape[0]
        self._n_cols = data.shape[1]

        for i, dtype in enumerate(data.dtypes):
            if dtype == "object":
                raise ValueError(f"Column {i} has unsupported data type {dtype}.")

        self.feature_names = feature_names
        self.feature_types = feature_types

        self.missing = missing
        self.silent = silent
        self.feature_weights = feature_weights
        self.enable_categorical = enable_categorical

        self.metadata = (
            data.index,
            data.columns,
            data._query_compiler._modin_frame.row_lengths,
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

    def get_dmatrix_params(self):
        """
        Get dict of DMatrix parameters excluding `self.data`/`self.label`.

        Returns
        -------
        dict
        """
        dmatrix_params = {
            "feature_names": self.feature_names,
            "feature_types": self.feature_types,
            "missing": self.missing,
            "silent": self.silent,
            "feature_weights": self.feature_weights,
            "enable_categorical": self.enable_categorical,
        }
        return dmatrix_params

    @property
    def feature_names(self):
        """
        Get column labels.

        Returns
        -------
        Column labels.
        """
        return self._feature_names

    @feature_names.setter
    def feature_names(self, feature_names):
        """
        Set column labels.

        Parameters
        ----------
        feature_names : list or None
            Labels for columns. In the case of ``None``, existing feature names will be reset.
        """
        if feature_names is not None:
            feature_names = (
                list(feature_names)
                if not isinstance(feature_names, str)
                else [feature_names]
            )

            if len(feature_names) != len(set(feature_names)):
                raise ValueError("Items in `feature_names` must be unique.")
            if len(feature_names) != self.num_col() and self.num_col() != 0:
                raise ValueError(
                    "`feature_names` must have the same width as `self.data`."
                )
            if not all(
                isinstance(f, str) and not any(x in f for x in set(("[", "]", "<")))
                for f in feature_names
            ):
                raise ValueError(
                    "Items of `feature_names` must be string and must not contain [, ] or <."
                )
        else:
            feature_names = None
        self._feature_names = feature_names

    @property
    def feature_types(self):
        """
        Get column types.

        Returns
        -------
        Column types.
        """
        return self._feature_types

    @feature_types.setter
    def feature_types(self, feature_types):
        """
        Set column types.

        Parameters
        ----------
        feature_types : list or None
            Labels for columns. In case None, existing feature names will be reset.
        """
        if feature_types is not None:
            if not isinstance(feature_types, (list, str)):
                raise TypeError("feature_types must be string or list of strings")
            if isinstance(feature_types, str):
                feature_types = [feature_types] * self.num_col()
                feature_types = (
                    list(feature_types)
                    if not isinstance(feature_types, str)
                    else [feature_types]
                )
        else:
            feature_types = None
        self._feature_types = feature_types

    def num_row(self):
        """
        Get number of rows.

        Returns
        -------
        int
        """
        return self._n_rows

    def num_col(self):
        """
        Get number of columns.

        Returns
        -------
        int
        """
        return self._n_cols

    def get_float_info(self, name):
        """
        Get float property from the DMatrix.

        Parameters
        ----------
        name : str
            The field name of the information.

        Returns
        -------
        A NumPy array of float information of the data.
        """
        return getattr(self, name)

    def set_info(
        self,
        *,
        label=None,
        feature_names=None,
        feature_types=None,
        feature_weights=None,
    ) -> None:
        """
        Set meta info for DMatrix.

        Parameters
        ----------
        label : modin.pandas.DataFrame or modin.pandas.Series, optional
            Labels used for training.
        feature_names : list, optional
            Set names for features.
        feature_types : list, optional
            Set types for features.
        feature_weights : array_like, optional
            Set feature weights for column sampling.
        """
        if label is not None:
            self.label = label
        if feature_names is not None:
            self.feature_names = feature_names
        if feature_types is not None:
            self.feature_types = feature_types
        if feature_weights is not None:
            self.feature_weights = feature_weights


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

        During execution it runs ``xgb.predict`` on each worker for subset of `data`
        and creates Modin DataFrame with prediction results.

        Parameters
        ----------
        data : modin.experimental.xgboost.DMatrix
            Input data used for prediction.
        **kwargs : dict
            Other parameters are the same as for ``xgboost.Booster.predict``.

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

        if (
            self.feature_names is not None
            and data.feature_names is not None
            and self.feature_names != data.feature_names
        ):
            data_missing = set(self.feature_names) - set(data.feature_names)
            self_missing = set(data.feature_names) - set(self.feature_names)

            msg = "feature_names mismatch: {0} {1}"

            if data_missing:
                msg += (
                    "\nexpected "
                    + ", ".join(str(s) for s in data_missing)
                    + " in input data"
                )

            if self_missing:
                msg += (
                    "\ntraining data did not have the following fields: "
                    + ", ".join(str(s) for s in self_missing)
                )

            raise ValueError(msg.format(self.feature_names, data.feature_names))

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
    result = _train(dtrain, params, *args, num_actors=num_actors, evals=evals, **kwargs)
    if isinstance(evals_result, dict):
        evals_result.update(result["history"])

    LOGGER.info("Training finished")
    return Booster(model_file=result["booster"])
