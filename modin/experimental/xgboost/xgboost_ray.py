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
from typing import Dict, Optional
from multiprocessing import cpu_count

import xgboost as xgb
import ray
from ray.services import get_node_ip_address
import numpy as np
import pandas

from modin.api import unwrap_partitions
from .utils import RabitContext, RabitContextManager

LOGGER = logging.getLogger("[modin.xgboost]")


@ray.remote
class ModinXGBoostActor:
    def __init__(self, ip, nthread=cpu_count()):
        self._evals = []
        self._dpredict = []
        self._ip = ip
        self._nthreads = nthread

        LOGGER.info(f"Actor <{self._ip}>, nthread = {self._nthreads} was initialized.")

    def _get_dmatrix(self, X_y):
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
        self._dtrain = self._get_dmatrix(X_y)

        if add_as_eval_method is not None:
            self._evals.append((self._dtrain, add_as_eval_method))

    def set_predict_data(
        self,
        *X,
    ):
        for x in X:
            self._dpredict.append(xgb.DMatrix(x, None))

    def add_eval_data(self, *X_y, eval_method):
        self._evals.append((self._get_dmatrix(X_y), eval_method))

    def train(self, rabit_args, params, *args, **kwargs):
        local_params = params.copy()
        local_dtrain = self._dtrain
        local_evals = self._evals

        local_params["nthread"] = self._nthreads

        evals_result = dict()

        s = time.time()
        with RabitContext(self._ip, rabit_args):
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


def create_actors(num_cpus=1, nthread=cpu_count()):
    num_nodes = len(ray.nodes())

    # Create remote actors
    actors = {
        node_info.split("node:")[-1]: ModinXGBoostActor.options(
            num_cpus=num_cpus, resources={node_info: 1.0}
        ).remote(node_info.split("node:")[-1], nthread=nthread)
        for node_info in ray.cluster_resources()
        if "node" in node_info
    }

    assert num_nodes == len(
        actors
    ), f"Number of nodes {num_nodes} is not equal to number of actors {len(actors)}."

    return actors


def _split_data_across_actors(
    actors: Dict, set_func, X_parts, y_parts=None, evenly_data_distribution=True
):
    """
    Split row partitions of data between actors.

    Parameters
    ----------
    actors : dict
        Dictionary of used actors.
    set_func : callable
        The function for setting data in actor.
    X_parts : list
        Row partitions of X data.
    y_parts : list, default None
        Row partitions of y data.
    evenly_data_distribution : boolean, default True
        Whether make evenly distribution of partitions between nodes or not.
        In case `False` minimal datatransfer between nodes will be provided
        but the data may not be evenly distributed.
    """
    X_parts_by_actors = _assign_row_partitions_to_actors(
        actors, X_parts, evenly_data_distribution=evenly_data_distribution
    )

    if y_parts is not None:
        y_parts_by_actors = _assign_row_partitions_to_actors(
            actors,
            y_parts,
            X_parts_by_actors,
            evenly_data_distribution=evenly_data_distribution,
        )

    for ip, actor in actors.items():
        X_parts = X_parts_by_actors[ip][0]
        if y_parts is None:
            set_func(actor, *X_parts)
        else:
            y_parts = y_parts_by_actors[ip][0]
            set_func(actor, *(X_parts + y_parts))


def _assign_row_partitions_to_actors(
    actors: Dict, row_partitions, data_for_aligning=None, evenly_data_distribution=True
):
    """
    Assign row_partitions to actors.

    Parameters
    ----------
    actors : dict
        Dictionary of used actors.
    row_partitions : list
        Row partitions of data to assign.
    data_for_aligning : dict, default None
        Data according to the order of which should be
        distributed row_partitions. Used to align y with X.
    evenly_data_distribution : boolean, default True
        Whether make evenly distribution of partitions between nodes or not.
        In case `False` minimal datatransfer between nodes will be provided
        but the data may not be evenly distributed.

    Returns
    -------
    dict
        Dictionary of assigned to actors partitions
        as {ip: (partitions, order)}.
    """
    row_partitions_by_actors = {ip: ([], []) for ip in actors}
    if evenly_data_distribution:
        _assign_partitions_evenly(
            actors,
            row_partitions,
            False,
            row_partitions_by_actors,
        )
    else:
        if data_for_aligning is None:
            actors_ips = list(actors.keys())
            partitions_ips = [ray.get(row_part[0]) for row_part in row_partitions]
            unique_partitions_ips = set(partitions_ips)
            empty_actor_ips = []
            for ip in actors_ips:
                if ip not in unique_partitions_ips:
                    empty_actor_ips.append(ip)

            # In case portion of nodes without data is less than 10%,
            # no data redistribution between nodes will be performed.
            if len(empty_actor_ips) / len(actors_ips) < 0.1:
                import warnings

                for ip in empty_actor_ips:
                    actors.pop(ip)
                    row_partitions_by_actors.pop(ip)
                    warnings.warn(
                        f"Node {ip} isn't used as it doesn't contain any data."
                    )
                for i, row_part in enumerate(row_partitions):
                    row_partitions_by_actors[partitions_ips[i]][0].append(row_part[1])
                    row_partitions_by_actors[partitions_ips[i]][1].append(i)
            else:
                _assign_partitions_evenly(
                    actors,
                    row_partitions,
                    True,
                    row_partitions_by_actors,
                )
        else:
            for ip, (_, order_of_indexes) in data_for_aligning.items():
                row_partitions_by_actors[ip][1].extend(order_of_indexes)
                for row_idx in order_of_indexes:
                    row_partitions_by_actors[ip][0].append(row_partitions[row_idx][1])

    return row_partitions_by_actors


def _assign_partitions_evenly(
    actors: Dict,
    row_partitions,
    is_partitions_have_ip,
    row_partitions_by_actors: Dict,
):
    """
    Make evenly assigning of row_partitions to actors.

    Parameters
    ----------
    actors : dict
        Dictionary of used actors.
    row_partitions : list
        Row partitions of data to assign.
    is_partitions_have_ip : boolean
        Whether each value of row_partitions is (ip, partition).
    row_partitions_by_actors : dict
        Dictionary of assigned to actors partitions
        as {ip: (partitions, order)}. Output parameter.
    """
    num_actors = len(actors)
    row_parts_last_idx = (
        len(row_partitions) // num_actors
        if len(row_partitions) % num_actors == 0
        else len(row_partitions) // num_actors + 1
    )

    start_idx = 0
    for ip, actor in actors.items():
        if is_partitions_have_ip:
            last_idx = (
                (start_idx + row_parts_last_idx)
                if (start_idx + row_parts_last_idx < len(row_partitions))
                else len(row_partitions)
            )
            row_partitions_by_actors[ip][1].extend(list(range(start_idx, last_idx)))
            for idx in range(start_idx, last_idx):
                row_partitions_by_actors[ip][0].append(row_partitions[idx][1])
        else:
            idx_slice = (
                slice(start_idx, start_idx + row_parts_last_idx)
                if start_idx + row_parts_last_idx < len(row_partitions)
                else slice(start_idx, len(row_partitions))
            )
            row_partitions_by_actors[ip][0].extend(row_partitions[idx_slice])
        start_idx += row_parts_last_idx


def _train(
    dtrain,
    nthread,
    evenly_data_distribution,
    params: Dict,
    *args,
    evals=(),
    **kwargs,
):
    s = time.time()

    X, y = dtrain
    assert len(X) == len(y)

    X_row_parts = unwrap_partitions(X, axis=0, bind_ip=not evenly_data_distribution)
    y_row_parts = unwrap_partitions(y, axis=0, bind_ip=not evenly_data_distribution)
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
                unwrap_partitions(eval_X, axis=0, bind_ip=not evenly_data_distribution),
                unwrap_partitions(eval_y, axis=0, bind_ip=not evenly_data_distribution),
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
    with RabitContextManager(len(actors), get_node_ip_address()) as env:
        rabit_args = [("%s=%s" % item).encode() for item in env.items()]

        # Train
        fut = [
            actor.train.remote(rabit_args, params, *args, **kwargs)
            for _, actor in actors.items()
        ]

        # All results should be the same because of Rabit tracking. So we just
        # return the first one.
        result = ray.get(fut[0])
        LOGGER.info(f"Training time: {time.time() - s} s")
        return result


def _predict(
    booster,
    data,
    nthread: Optional[int] = cpu_count(),
    evenly_data_distribution: Optional[bool] = True,
    **kwargs,
):
    s = time.time()

    X, _ = data
    X_row_parts = unwrap_partitions(X, axis=0, bind_ip=not evenly_data_distribution)

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
    predictions = [
        actor.predict.remote(booster, **kwargs) for _, actor in actors.items()
    ]
    result = ray.get(predictions)
    LOGGER.info(f"Prediction time: {time.time() - s} s")

    return result[0]
