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


import time
import logging
from threading import Thread
from typing import Dict, Optional
from multiprocessing import cpu_count

import xgboost as xgb
import numpy as np
import pandas

from modin.config import Engine
from modin.developer import unwrap_row_partitions

if Engine.get() == "Ray":
    import ray
    from ray.services import get_node_ip_address
else:
    raise ValueError("Current version supports only Ray engine as MODIN_ENGINE.")

LOGGER = logging.getLogger("[modin.xgboost]")


def _start_rabit_tracker(num_workers: int):
    """Start Rabit tracker. The workers connect to this tracker to share
    their results."""
    host = get_node_ip_address()

    env = {"DMLC_NUM_WORKER": num_workers}
    rabit_tracker = xgb.RabitTracker(hostIP=host, nslave=num_workers)

    # Get tracker Host + IP
    env.update(rabit_tracker.slave_envs())
    rabit_tracker.start(num_workers)

    # Wait until context completion
    thread = Thread(target=rabit_tracker.join)
    thread.daemon = True
    thread.start()

    return env


class RabitContext:
    """Context to connect a worker to a rabit tracker"""

    def __init__(self, actor_id, args):
        self.args = args
        self.args.append(("DMLC_TASK_ID=[modin.xgboost]:" + actor_id).encode())

    def __enter__(self):
        xgb.rabit.init(self.args)
        LOGGER.info("-------------- rabit started ------------------")

    def __exit__(self, *args):
        xgb.rabit.finalize()
        LOGGER.info("-------------- rabit finished ------------------")


@ray.remote
class ModinXGBoostActor:
    def __init__(self, ip, nthread=cpu_count()):
        self._evals = []
        self._dpredict = []
        self._ip = ip
        self._nthreads = nthread

        LOGGER.info(f"Actor <{self._ip}>, nthread = {self._nthreads} was initialized.")

    def get_actor_ip(self):
        return self._ip

    def _get_dmatrix(self, *X_y):
        s = time.time()
        X = X_y[: len(X_y) // 2]
        y = X_y[len(X_y) // 2 :]

        assert (
            len(X) == len(y) and len(X) > 0
        ), "X and y should have the equal length more than 0"

        X = pandas.concat(X, axis=0)
        y = pandas.concat(y, axis=0)
        LOGGER.info(f"Concat time: {time.time() - s} s")

        return xgb.DMatrix(X, y)

    def set_train_data(self, *X_y, add_as_eval_method=None):
        self._dtrain = self._get_dmatrix(*X_y)

        if add_as_eval_method is not None:
            self._evals.append((self._dtrain, add_as_eval_method))

    def set_predict_data(
        self,
        *X,
    ):
        for x in X:
            self._dpredict.append(xgb.DMatrix(x, None))

    def add_eval_data(self, *X_y, eval_method):
        self._evals.append((self._get_dmatrix(*X_y), eval_method))

    def train(self, rabit_args, params, *args, **kwargs):
        local_params = params.copy()
        local_dtrain = self._dtrain
        local_evals = self._evals

        local_params["nthread"] = self._nthreads

        evals_result = dict()

        s = time.time()
        with RabitContext(str(id(self)), rabit_args):
            bst = xgb.train(
                local_params,
                local_dtrain,
                *args,
                evals=local_evals,
                evals_result=evals_result,
                **kwargs,
            )
            LOGGER.info(f"Local training time: {time.time() - s} s")
            return {"booster": bst, "history": evals_result}

    def predict(self, booster: xgb.Booster, *args, **kwargs):
        local_dpredict = self._dpredict
        booster.set_param({"nthread": self._nthreads})

        s = time.time()
        predictions = [booster.predict(X, *args, **kwargs) for X in local_dpredict]
        LOGGER.info(f"Local prediction time: {time.time() - s} s")
        return np.concatenate(predictions)


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


def create_actors(num_cpus=1, nthread=cpu_count()):
    num_nodes = len(ray.nodes())

    # Create remote actors
    actors = [
        ModinXGBoostActor.options(num_cpus=num_cpus, resources={node_info: 1.0}).remote(
            node_info.split("node:")[-1], nthread=nthread
        )
        for node_info in ray.cluster_resources()
        if "node" in node_info
    ]

    assert num_nodes == len(
        actors
    ), f"Number of nodes {num_nodes} is not equal to number of actors {len(actors)}."

    return actors


def _split_data_across_actors(
    actors, set_func, X_parts, y_parts=None, evenly_data_distribution=True
):
    X_parts_by_actors = _set_row_partitions_to_actors(
        actors, X_parts, evenly_data_distribution=evenly_data_distribution
    )
    if y_parts is not None:
        y_parts_by_actors = _set_row_partitions_to_actors(
            actors,
            y_parts,
            X_parts_by_actors,
            evenly_data_distribution=evenly_data_distribution,
        )

    # Need to add assert on order of actors in case evenly_data_distribution=False
    for i, actor in enumerate(actors):

        X_parts = (
            X_parts_by_actors[i]
            if evenly_data_distribution
            else X_parts_by_actors[i][2]
        )
        if y_parts is None:
            set_func(actor, *X_parts)
        else:
            y_parts = (
                y_parts_by_actors[i]
                if evenly_data_distribution
                else y_parts_by_actors[i][2]
            )
            set_func(actor, *(X_parts + y_parts))


def _set_row_partitions_to_actors(
    actors, row_partitions, data_for_aligning=None, evenly_data_distribution=True
):

    row_partitions_to_actors = []
    if evenly_data_distribution:
        num_actors = len(actors)
        row_parts_last_idx = (
            len(row_partitions) // num_actors
            if len(row_partitions) % num_actors == 0
            else len(row_partitions) // num_actors + 1
        )
        start_idx = 0
        for actor in actors:
            idx_slice = (
                slice(start_idx, start_idx + row_parts_last_idx)
                if start_idx + row_parts_last_idx < len(row_partitions)
                else slice(start_idx, len(row_partitions))
            )
            row_partitions_to_actors.append(row_partitions[idx_slice])
            start_idx += row_parts_last_idx
    else:
        actors_ips = [ray.get(actor.get_actor_ip.remote()) for actor in actors]
        partitions_ips = [ray.get(row[0]) for row in row_partitions]

        # If one of the actors doesn't contain partitions we make evenly
        # distribution of data by nodes.
        if (
            len(np.unique(actors_ips)) != len(np.unique(partitions_ips))
            and data_for_aligning is None
        ):
            return _get_evenly_distr_of_partitions_by_ips(
                actors, actors_ips, row_partitions
            )

        if data_for_aligning is None:
            for actor_ip in actors_ips:
                actor_row_partitions = []
                order_of_indexes = []
                for i, row in enumerate(row_partitions):
                    if partitions_ips[i] in actor_ip:
                        actor_row_partitions.append(row[1])
                        order_of_indexes.append(i)

                row_partitions_to_actors.append(
                    (actor_ip, order_of_indexes, actor_row_partitions)
                )
        else:
            for (actor_ip, order_of_indexes, _) in data_for_aligning:
                actor_row_partitions = []
                for row_idx in order_of_indexes:
                    actor_row_partitions.append(row_partitions[row_idx][1])

                row_partitions_to_actors.append(
                    (actor_ip, order_of_indexes, actor_row_partitions)
                )

    return row_partitions_to_actors


def _get_evenly_distr_of_partitions_by_ips(actors, actors_ips, row_partitions):
    num_actors = len(actors)
    row_parts_last_idx = (
        len(row_partitions) // num_actors
        if len(row_partitions) % num_actors == 0
        else len(row_partitions) // num_actors + 1
    )
    row_partitions_to_actors = []
    start_idx = 0
    for actor, ip in zip(actors, actors_ips):
        last = (
            (start_idx + row_parts_last_idx)
            if (start_idx + row_parts_last_idx < len(row_partitions))
            else len(row_partitions)
        )

        actor_row_partitions = []
        order_of_indexes = []
        for first in range(start_idx, last):
            actor_row_partitions.append(row_partitions[first][1])
            order_of_indexes.append(first)
        start_idx += row_parts_last_idx

        row_partitions_to_actors.append((ip, order_of_indexes, actor_row_partitions))

    return row_partitions_to_actors


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

    s = time.time()

    X, y = dtrain
    assert len(X) == len(y)
    X_row_parts = unwrap_row_partitions(X, bind_ip=not evenly_data_distribution)
    y_row_parts = unwrap_row_partitions(y, bind_ip=not evenly_data_distribution)
    assert len(X_row_parts) == len(y_row_parts), "Unaligned train data"

    # Create remote actors
    actors = create_actors(nthread=nthread)

    add_as_eval_method = None
    if evals:
        for (eval_data, method) in evals[:]:
            if eval_data is dtrain:
                add_as_eval_method = method
                evals.remove((eval_data, method))

        for ((eval_X, eval_y), eval_method) in evals:
            # Split data across workers
            _split_data_across_actors(
                actors,
                lambda actor, *X_y: actor.add_eval_data.remote(
                    *X_y, eval_method=eval_method
                ),
                unwrap_row_partitions(eval_X, bind_ip=not evenly_data_distribution),
                unwrap_row_partitions(eval_y, bind_ip=not evenly_data_distribution),
                evenly_data_distribution=evenly_data_distribution,
            )

    # Split data across workers
    _split_data_across_actors(
        actors,
        lambda actor, *X_y: actor.set_train_data.remote(
            *X_y, add_as_eval_method=add_as_eval_method
        ),
        X_row_parts,
        y_row_parts,
        evenly_data_distribution=evenly_data_distribution,
    )
    LOGGER.info(f"Data preparation time: {time.time() - s} s")

    s = time.time()
    # Start Rabit tracker
    env = _start_rabit_tracker(len(actors))
    rabit_args = [("%s=%s" % item).encode() for item in env.items()]

    # Train
    fut = [actor.train.remote(rabit_args, params, *args, **kwargs) for actor in actors]

    # All results should be the same because of Rabit tracking. So we just
    # return the first one.
    result = ray.get(fut[0])
    LOGGER.info(f"Training time: {time.time() - s} s")
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
    s = time.time()

    if isinstance(model, xgb.Booster):
        booster = model
    elif isinstance(model, dict):
        booster = model["booster"]
    else:
        raise TypeError(
            f"Expected types for `model` xgb.Booster or dict, but presented type is {type(model)}"
        )

    X, _ = data
    X_row_parts = unwrap_row_partitions(X, bind_ip=not evenly_data_distribution)

    # Create remote actors
    actors = create_actors(nthread=nthread)

    # Split data across workers
    _split_data_across_actors(
        actors,
        lambda actor, *X: actor.set_predict_data.remote(*X),
        X_row_parts,
        evenly_data_distribution=evenly_data_distribution,
    )

    LOGGER.info(f"Data preparation time: {time.time() - s} s")
    s = time.time()

    # Predict
    predictions = [actor.predict.remote(booster, **kwargs) for actor in actors]
    result = ray.get(predictions)
    LOGGER.info(f"Prediction time: {time.time() - s} s")
    LOGGER.info("Prediction finished")

    return result[0]
