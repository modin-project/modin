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
from collections import OrderedDict

import xgboost as xgb
import ray
from ray.services import get_node_ip_address
import pandas

from modin.distributed.dataframe.pandas import unwrap_partitions, from_partitions
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

        return xgb.DMatrix(X, y, nthread=self._nthreads)

    def set_train_data(self, *X_y, add_as_eval_method=None):
        self._dtrain = self._get_dmatrix(X_y)

        if add_as_eval_method is not None:
            self._evals.append((self._dtrain, add_as_eval_method))

    def set_predict_data(
        self,
        *X,
    ):
        for x in X:
            self._dpredict.append(xgb.DMatrix(x, nthread=self._nthreads))

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

    def predict(self, booster: xgb.Booster, **kwargs):
        local_dpredict = self._dpredict
        booster.set_param({"nthread": self._nthreads})

        s = time.time()
        predictions = [
            pandas.DataFrame(booster.predict(X, **kwargs)) for X in local_dpredict
        ]
        LOGGER.info(f"Local prediction time: {time.time() - s} s")
        return predictions if len(predictions) > 1 else predictions[0]

    def exit_actor(self):
        ray.actor.exit_actor()


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
    actors: Dict,
    set_func,
    X_parts,
    y_parts=None,
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
    y_parts : list. Default is None
        Row partitions of y data.

    Returns
    -------
    dict
        Dictionary with orders of partitions by IP's as {ip: order}.
    """
    X_parts_by_actors = _assign_row_partitions_to_actors(
        actors,
        X_parts,
    )

    if y_parts is not None:
        y_parts_by_actors = _assign_row_partitions_to_actors(
            actors,
            y_parts,
            data_for_aligning=X_parts_by_actors,
        )

    for ip, actor in actors.items():
        X_parts = X_parts_by_actors[ip][0]
        if y_parts is None:
            set_func(actor, *X_parts)
        else:
            y_parts = y_parts_by_actors[ip][0]
            set_func(actor, *(X_parts + y_parts))

    order_of_parts = {ip: order for ip, (_, order) in X_parts_by_actors.items()}

    return order_of_parts


def _assign_row_partitions_to_actors(
    actors: Dict,
    row_partitions,
    data_for_aligning=None,
):
    """
    Assign row_partitions to actors.

    Parameters
    ----------
    actors : dict
        Dictionary of used actors.
    row_partitions : list
        Row partitions of data to assign.
    data_for_aligning : dict. Default is None
        Data according to the order of which should be
        distributed row_partitions. Used to align y with X.

    Returns
    -------
    dict
        Dictionary of assigned to actors partitions
        as {ip: (partitions, order)}.
    """
    if data_for_aligning is None:
        partitions_ips = [ray.get(row_part[0]) for row_part in row_partitions]

        partitions_distribution = {ip: partitions_ips.count(ip) for ip in actors}

        parts_distribution_sorted = dict()

        num_actors = len(actors)
        parts_per_actor = (
            len(row_partitions) // num_actors
            if len(row_partitions) % num_actors < num_actors // 2 + 1
            else len(row_partitions) // num_actors + 1
        )
        parts_per_last_actor = len(row_partitions) - parts_per_actor * (num_actors - 1)

        for idx, (ip, _) in enumerate(
            sorted(
                partitions_distribution.items(), key=lambda item: item[1], reverse=True
            )
        ):
            if idx == num_actors - 1:
                parts_per_actor = parts_per_last_actor

            parts_distribution_sorted[ip] = parts_per_actor

        row_partitions_by_actors = OrderedDict()
        for ip in parts_distribution_sorted:
            row_partitions_by_actors[ip] = ([], [])

        # Get initial distribution
        for i, row_part in enumerate(row_partitions):
            row_partitions_by_actors[partitions_ips[i]][0].append(row_part[1])
            row_partitions_by_actors[partitions_ips[i]][1].append(i)

        # Iterating over all actors except last
        for idx, ip in enumerate(list(row_partitions_by_actors)[:-1]):
            if len(row_partitions_by_actors[ip][0]) == parts_distribution_sorted[ip]:
                continue
            else:
                num_extra_parts = (
                    len(row_partitions_by_actors[ip][0]) - parts_distribution_sorted[ip]
                )
                extra_parts = (
                    row_partitions_by_actors[ip][0][:num_extra_parts],
                    row_partitions_by_actors[ip][1][:num_extra_parts],
                )

                sliced_parts = (
                    row_partitions_by_actors[ip][0][num_extra_parts:],
                    row_partitions_by_actors[ip][1][num_extra_parts:],
                )

                # Save only slice for original partitions
                row_partitions_by_actors[ip] = sliced_parts

                # Move extra partitions to the next actor
                row_partitions_by_actors[list(row_partitions_by_actors)[idx + 1]][
                    0
                ].extend(extra_parts[0])
                row_partitions_by_actors[list(row_partitions_by_actors)[idx + 1]][
                    1
                ].extend(extra_parts[1])

        # Check correctness of distribution
        for ip, (parts, _) in row_partitions_by_actors.items():
            assert (
                len(parts) == parts_distribution_sorted[ip]
            ), f"Distribution of partitions is incorrect. {ip} contains {len(parts)} but {parts_distribution_sorted[ip]} expected."

    else:
        row_partitions_by_actors = {ip: ([], []) for ip in actors}

        for ip, (_, order_of_indexes) in data_for_aligning.items():
            row_partitions_by_actors[ip][1].extend(order_of_indexes)
            for row_idx in order_of_indexes:
                row_partitions_by_actors[ip][0].append(row_partitions[row_idx])

    return dict(row_partitions_by_actors)


def _train(
    dtrain,
    nthread,
    params: Dict,
    *args,
    evals=(),
    **kwargs,
):
    s = time.time()

    X, y = dtrain
    assert len(X) == len(y)

    X_row_parts = unwrap_partitions(X, axis=0, get_ip=True)
    y_row_parts = unwrap_partitions(y, axis=0)
    assert len(X_row_parts) == len(y_row_parts), "Unaligned train data"

    # Create remote actors
    actors = create_actors(nthread=nthread)

    assert len(actors) <= len(
        X_row_parts
    ), f"{len(X_row_parts)} row partitions couldn't be distributed between {len(actors)} nodes."

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
                unwrap_partitions(eval_X, axis=0, get_ip=True),
                y_parts=unwrap_partitions(eval_y, axis=0),
            )

    # Split data across workers
    _split_data_across_actors(
        actors,
        lambda actor, *X_y: actor.set_train_data.remote(
            *X_y, add_as_eval_method=add_as_eval_method
        ),
        X_row_parts,
        y_parts=y_row_parts,
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
    **kwargs,
):
    s = time.time()

    X, _ = data
    X_row_parts = unwrap_partitions(X, axis=0, get_ip=True)

    # Create remote actors
    actors = create_actors(nthread=nthread)

    assert len(actors) <= len(
        X_row_parts
    ), f"{len(X_row_parts)} row partitions couldn't be distributed between {len(actors)} nodes."

    # Split data across workers
    order_of_parts = _split_data_across_actors(
        actors,
        lambda actor, *X: actor.set_predict_data.remote(*X),
        X_row_parts,
    )

    LOGGER.info(f"Data preparation time: {time.time() - s} s")
    s = time.time()

    # Predict
    predictions = [
        actor.predict._remote(
            args=(booster,), kwargs=kwargs, num_returns=len(order_of_parts[ip])
        )
        if len(order_of_parts[ip]) > 1
        else [
            actor.predict._remote(
                args=(booster,), kwargs=kwargs, num_returns=len(order_of_parts[ip])
            )
        ]
        for ip, actor in actors.items()
    ]

    results_to_sort = list()
    for ip, part_res in zip(actors, predictions):
        results_to_sort.extend(list(zip(part_res, order_of_parts[ip])))

    results = sorted(results_to_sort, key=lambda l: l[1])
    results = [part_res for part_res, _ in results]

    result = from_partitions(results, 0).reset_index(drop=True)
    LOGGER.info(f"Prediction time: {time.time() - s} s")

    return result
