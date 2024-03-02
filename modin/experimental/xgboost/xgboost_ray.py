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

"""
Module holds internal entities for Modin XGBoost on Ray engine.

Class ModinXGBoostActor provides interfaces to run XGBoost operations
on remote workers. Other functions create Ray actors, distribute data between them, etc.
"""

import logging
import math
import time
import warnings
from collections import defaultdict
from typing import Dict, List

import numpy as np
import pandas
import ray
import xgboost as xgb
from ray.util import get_node_ip_address

from modin.core.execution.ray.common import RayWrapper
from modin.distributed.dataframe.pandas import from_partitions

from .utils import RabitContext, RabitContextManager

LOGGER = logging.getLogger("[modin.xgboost]")


@ray.remote(num_cpus=0)
class ModinXGBoostActor:
    """
    Ray actor-class runs training on the remote worker.

    Parameters
    ----------
    rank : int
        Rank of this actor.
    nthread : int
        Number of threads used by XGBoost in this actor.
    """

    def __init__(self, rank, nthread):
        self._evals = []
        self._rank = rank
        self._nthreads = nthread

        LOGGER.info(
            f"Actor <{self._rank}>, nthread = {self._nthreads} was initialized."
        )

    def _get_dmatrix(self, X_y, **dmatrix_kwargs):
        """
        Create xgboost.DMatrix from sequence of pandas.DataFrame objects.

        First half of `X_y` should contains objects for `X`, second for `y`.

        Parameters
        ----------
        X_y : list
            List of pandas.DataFrame objects.
        **dmatrix_kwargs : dict
            Keyword parameters for ``xgb.DMatrix``.

        Returns
        -------
        xgb.DMatrix
            A XGBoost DMatrix.
        """
        s = time.time()
        X = X_y[: len(X_y) // 2]
        y = X_y[len(X_y) // 2 :]

        assert (
            len(X) == len(y) and len(X) > 0
        ), "X and y should have the equal length more than 0"

        X = pandas.concat(X, axis=0)
        y = pandas.concat(y, axis=0)
        LOGGER.info(f"Concat time: {time.time() - s} s")

        return xgb.DMatrix(X, y, nthread=self._nthreads, **dmatrix_kwargs)

    def set_train_data(self, *X_y, add_as_eval_method=None, **dmatrix_kwargs):
        """
        Set train data for actor.

        Parameters
        ----------
        *X_y : iterable
            Sequence of ray.ObjectRef objects. First half of sequence is for
            `X` data, second for `y`. When it is passed in actor, auto-materialization
            of ray.ObjectRef -> pandas.DataFrame happens.
        add_as_eval_method : str, optional
            Name of eval data. Used in case when train data also used for evaluation.
        **dmatrix_kwargs : dict
            Keyword parameters for ``xgb.DMatrix``.
        """
        self._dtrain = self._get_dmatrix(X_y, **dmatrix_kwargs)

        if add_as_eval_method is not None:
            self._evals.append((self._dtrain, add_as_eval_method))

    def add_eval_data(self, *X_y, eval_method, **dmatrix_kwargs):
        """
        Add evaluation data for actor.

        Parameters
        ----------
        *X_y : iterable
            Sequence of ray.ObjectRef objects. First half of sequence is for
            `X` data, second for `y`. When it is passed in actor, auto-materialization
            of ray.ObjectRef -> pandas.DataFrame happens.
        eval_method : str
            Name of eval data.
        **dmatrix_kwargs : dict
            Keyword parameters for ``xgb.DMatrix``.
        """
        self._evals.append((self._get_dmatrix(X_y, **dmatrix_kwargs), eval_method))

    def train(self, rabit_args, params, *args, **kwargs):
        """
        Run local XGBoost training.

        Connects to Rabit Tracker environment to share training data between
        actors and trains XGBoost booster using `self._dtrain`.

        Parameters
        ----------
        rabit_args : list
            List with environment variables for Rabit Tracker.
        params : dict
            Booster params.
        *args : iterable
            Other parameters for `xgboost.train`.
        **kwargs : dict
            Other parameters for `xgboost.train`.

        Returns
        -------
        dict
            A dictionary with trained booster and dict of
            evaluation results
            as {"booster": xgb.Booster, "history": dict}.
        """
        local_params = params.copy()
        local_dtrain = self._dtrain
        local_evals = self._evals

        local_params["nthread"] = self._nthreads

        evals_result = dict()

        s = time.time()
        with RabitContext(self._rank, rabit_args):
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


def _get_cluster_cpus():
    """
    Get number of CPUs available on Ray cluster.

    Returns
    -------
    int
        Number of CPUs available on cluster.
    """
    return ray.cluster_resources().get("CPU", 1)


def _get_min_cpus_per_node():
    """
    Get min number of node CPUs available on cluster nodes.

    Returns
    -------
    int
        Min number of CPUs per node.
    """
    # TODO: max_node_cpus -> min_node_cpus
    max_node_cpus = min(
        node.get("Resources", {}).get("CPU", 0.0) for node in ray.nodes()
    )
    return max_node_cpus if max_node_cpus > 0.0 else _get_cluster_cpus()


def _get_cpus_per_actor(num_actors):
    """
    Get number of CPUs to use by each actor.

    Parameters
    ----------
    num_actors : int
        Number of Ray actors.

    Returns
    -------
    int
        Number of CPUs per actor.
    """
    cluster_cpus = _get_cluster_cpus()
    cpus_per_actor = max(
        1, min(int(_get_min_cpus_per_node() or 1), int(cluster_cpus // num_actors))
    )
    return cpus_per_actor


def _get_num_actors(num_actors=None):
    """
    Get number of actors to create.

    Parameters
    ----------
    num_actors : int, optional
        Desired number of actors. If is None, integer number of actors
        will be computed by condition 2 CPUs per 1 actor.

    Returns
    -------
    int
        Number of actors to create.
    """
    min_cpus_per_node = _get_min_cpus_per_node()
    if num_actors is None:
        num_actors_per_node = max(1, int(min_cpus_per_node // 2))
        return num_actors_per_node * len(ray.nodes())
    elif isinstance(num_actors, int):
        assert (
            num_actors % len(ray.nodes()) == 0
        ), "`num_actors` must be a multiple to number of nodes in Ray cluster."
        return num_actors
    else:
        RuntimeError("`num_actors` must be int or None")


def create_actors(num_actors):
    """
    Create ModinXGBoostActors.

    Parameters
    ----------
    num_actors : int
        Number of actors to create.

    Returns
    -------
    list
        List of pairs (ip, actor).
    """
    num_cpus_per_actor = _get_cpus_per_actor(num_actors)
    # starting from ray 2.6 there is a new field: 'node:__internal_head__'
    # example:
    # >>> ray.cluster_resources()
    # {'object_store_memory': 1036438732.0, 'memory': 2072877467.0, 'node:127.0.0.1': 1.0, 'CPU': 8.0, 'node:__internal_head__': 1.0}
    node_ips = [
        key
        for key in ray.cluster_resources().keys()
        if key.startswith("node:") and "__internal_head__" not in key
    ]

    num_actors_per_node = max(num_actors // len(node_ips), 1)
    actors_ips = [ip for ip in node_ips for _ in range(num_actors_per_node)]

    actors = [
        (
            node_ip.split("node:")[-1],
            ModinXGBoostActor.options(resources={node_ip: 0.01}).remote(
                i, nthread=num_cpus_per_actor
            ),
        )
        for i, node_ip in enumerate(actors_ips)
    ]
    return actors


def _split_data_across_actors(
    actors: List,
    set_func,
    X_parts,
    y_parts,
):
    """
    Split row partitions of data between actors.

    Parameters
    ----------
    actors : list
        List of used actors.
    set_func : callable
        The function for setting data in actor.
    X_parts : list
        Row partitions of X data.
    y_parts : list
        Row partitions of y data.
    """
    X_parts_by_actors = _assign_row_partitions_to_actors(
        actors,
        X_parts,
    )

    y_parts_by_actors = _assign_row_partitions_to_actors(
        actors,
        y_parts,
        data_for_aligning=X_parts_by_actors,
    )

    for rank, (_, actor) in enumerate(actors):
        set_func(actor, *(X_parts_by_actors[rank][0] + y_parts_by_actors[rank][0]))


def _assign_row_partitions_to_actors(
    actors: List,
    row_partitions,
    data_for_aligning=None,
):
    """
    Assign row_partitions to actors.

    `row_partitions` will be assigned to actors according to their IPs.
    If distribution isn't even, partitions will be moved from actor
    with excess partitions to actor with lack of them.

    Parameters
    ----------
    actors : list
        List of used actors.
    row_partitions : list
        Row partitions of data to assign.
    data_for_aligning : dict, optional
        Data according to the order of which should be
        distributed `row_partitions`. Used to align y with X.

    Returns
    -------
    dict
        Dictionary of assigned to actors partitions
        as {actor_rank: (partitions, order)}.
    """
    num_actors = len(actors)
    if data_for_aligning is None:
        parts_ips_ref, parts_ref = zip(*row_partitions)

        # Group actors which are one the same ip
        actor_ips = defaultdict(list)
        for rank, (ip, _) in enumerate(actors):
            actor_ips[ip].append(rank)

        # Get distribution of parts between nodes ({ip:[(part, position),..],..})
        init_parts_distribution = defaultdict(list)
        for idx, (ip, part_ref) in enumerate(
            zip(RayWrapper.materialize(list(parts_ips_ref)), parts_ref)
        ):
            init_parts_distribution[ip].append((part_ref, idx))

        num_parts = len(parts_ref)
        min_parts_per_actor = math.floor(num_parts / num_actors)
        max_parts_per_actor = math.ceil(num_parts / num_actors)
        num_actors_with_max_parts = num_parts % num_actors

        row_partitions_by_actors = defaultdict(list)
        # Fill actors without movement parts between ips
        for actor_ip, ranks in actor_ips.items():
            # Loop across actors which are placed on actor_ip
            for rank in ranks:
                num_parts_on_ip = len(init_parts_distribution[actor_ip])

                # Check that have something to distribute on this ip
                if num_parts_on_ip == 0:
                    break
                # Check that node with `actor_ip` has enough parts for minimal
                # filling actor with `rank`
                if num_parts_on_ip >= min_parts_per_actor:
                    # Check that node has enough parts for max filling
                    # actor with `rank`
                    if (
                        num_parts_on_ip >= max_parts_per_actor
                        and num_actors_with_max_parts > 0
                    ):
                        pop_slice = slice(0, max_parts_per_actor)
                        num_actors_with_max_parts -= 1
                    else:
                        pop_slice = slice(0, min_parts_per_actor)

                    row_partitions_by_actors[rank].extend(
                        init_parts_distribution[actor_ip][pop_slice]
                    )
                    # Delete parts which we already assign
                    del init_parts_distribution[actor_ip][pop_slice]
                else:
                    row_partitions_by_actors[rank].extend(
                        init_parts_distribution[actor_ip]
                    )
                    init_parts_distribution[actor_ip] = []

        # Remove empty IPs
        for ip in list(init_parts_distribution):
            if len(init_parts_distribution[ip]) == 0:
                init_parts_distribution.pop(ip)

        # IP's aren't necessary now
        init_parts_distribution = [
            pair for pairs in init_parts_distribution.values() for pair in pairs
        ]

        # Fill the actors with extra parts (movements data between nodes)
        for rank in range(len(actors)):
            num_parts_on_rank = len(row_partitions_by_actors[rank])

            if num_parts_on_rank == max_parts_per_actor or (
                num_parts_on_rank == min_parts_per_actor
                and num_actors_with_max_parts == 0
            ):
                continue

            if num_actors_with_max_parts > 0:
                pop_slice = slice(0, max_parts_per_actor - num_parts_on_rank)
                num_actors_with_max_parts -= 1
            else:
                pop_slice = slice(0, min_parts_per_actor - num_parts_on_rank)

            row_partitions_by_actors[rank].extend(init_parts_distribution[pop_slice])
            del init_parts_distribution[pop_slice]

        if len(init_parts_distribution) != 0:
            raise RuntimeError(
                f"Not all partitions were ditributed between actors: {len(init_parts_distribution)} left."
            )

        row_parts_by_ranks = dict()
        for rank, pairs_part_pos in dict(row_partitions_by_actors).items():
            parts, order = zip(*pairs_part_pos)
            row_parts_by_ranks[rank] = (list(parts), list(order))
    else:
        row_parts_by_ranks = {rank: ([], []) for rank in range(len(actors))}

        for rank, (_, order_of_indexes) in data_for_aligning.items():
            row_parts_by_ranks[rank][1].extend(order_of_indexes)
            for row_idx in order_of_indexes:
                row_parts_by_ranks[rank][0].append(row_partitions[row_idx])

    return row_parts_by_ranks


def _train(
    dtrain,
    params: Dict,
    *args,
    num_actors=None,
    evals=(),
    **kwargs,
):
    """
    Run distributed training of XGBoost model on Ray engine.

    During work it evenly distributes `dtrain` between workers according
    to IP addresses partitions (in case of not even distribution of `dtrain`
    by nodes, part of partitions will be re-distributed between nodes),
    runs xgb.train on each worker for subset of `dtrain` and reduces training results
    of each worker using Rabit Context.

    Parameters
    ----------
    dtrain : modin.experimental.DMatrix
        Data to be trained against.
    params : dict
        Booster params.
    *args : iterable
        Other parameters for `xgboost.train`.
    num_actors : int, optional
        Number of actors for training. If unspecified, this value will be
        computed automatically.
    evals : list of pairs (modin.experimental.xgboost.DMatrix, str), default: empty
        List of validation sets for which metrics will be evaluated during training.
        Validation metrics will help us track the performance of the model.
    **kwargs : dict
        Other parameters are the same as `xgboost.train`.

    Returns
    -------
    dict
        A dictionary with trained booster and dict of
        evaluation results
        as {"booster": xgboost.Booster, "history": dict}.
    """
    s = time.time()

    X_row_parts, y_row_parts = dtrain
    dmatrix_kwargs = dtrain.get_dmatrix_params()

    assert len(X_row_parts) == len(y_row_parts), "Unaligned train data"

    num_actors = _get_num_actors(num_actors)

    if num_actors > len(X_row_parts):
        num_actors = len(X_row_parts)

    if evals:
        min_num_parts = num_actors
        for (eval_X, _), eval_method in evals:
            if len(eval_X) < min_num_parts:
                min_num_parts = len(eval_X)
                method_name = eval_method

        if num_actors != min_num_parts:
            num_actors = min_num_parts
            warnings.warn(
                f"`num_actors` is set to {num_actors}, because `evals` data with name `{method_name}` has only {num_actors} partition(s)."
            )

    actors = create_actors(num_actors)

    add_as_eval_method = None
    if evals:
        for eval_data, method in evals[:]:
            if eval_data is dtrain:
                add_as_eval_method = method
                evals.remove((eval_data, method))

        for (eval_X, eval_y), eval_method in evals:
            # Split data across workers
            _split_data_across_actors(
                actors,
                lambda actor, *X_y: actor.add_eval_data.remote(
                    *X_y, eval_method=eval_method, **dmatrix_kwargs
                ),
                eval_X,
                eval_y,
            )

    # Split data across workers
    _split_data_across_actors(
        actors,
        lambda actor, *X_y: actor.set_train_data.remote(
            *X_y, add_as_eval_method=add_as_eval_method, **dmatrix_kwargs
        ),
        X_row_parts,
        y_row_parts,
    )
    LOGGER.info(f"Data preparation time: {time.time() - s} s")

    s = time.time()
    with RabitContextManager(len(actors), get_node_ip_address()) as env:
        rabit_args = [("%s=%s" % item).encode() for item in env.items()]

        # Train
        fut = [
            actor.train.remote(rabit_args, params, *args, **kwargs)
            for _, actor in actors
        ]
        # All results should be the same because of Rabit tracking. So we just
        # return the first one.
        result = RayWrapper.materialize(fut[0])
        LOGGER.info(f"Training time: {time.time() - s} s")
        return result


@ray.remote
def _map_predict(booster, part, columns, dmatrix_kwargs={}, **kwargs):
    """
    Run prediction on a remote worker.

    Parameters
    ----------
    booster : xgboost.Booster or ray.ObjectRef
        A trained booster.
    part : pandas.DataFrame or ray.ObjectRef
        Partition of full data used for local prediction.
    columns : list or ray.ObjectRef
        Columns for the result.
    dmatrix_kwargs : dict, optional
        Keyword parameters for ``xgb.DMatrix``.
    **kwargs : dict
        Other parameters are the same as for ``xgboost.Booster.predict``.

    Returns
    -------
    ray.ObjectRef
        ``ray.ObjectRef`` with partial prediction.
    """
    dmatrix = xgb.DMatrix(part, **dmatrix_kwargs)
    prediction = pandas.DataFrame(
        booster.predict(dmatrix, **kwargs),
        index=part.index,
        columns=columns,
    )
    return prediction


def _predict(
    booster,
    data,
    **kwargs,
):
    """
    Run distributed prediction with a trained booster on Ray engine.

    During execution it runs ``xgb.predict`` on each worker for subset of `data`
    and creates Modin DataFrame with prediction results.

    Parameters
    ----------
    booster : xgboost.Booster
        A trained booster.
    data : modin.experimental.xgboost.DMatrix
        Input data used for prediction.
    **kwargs : dict
        Other parameters are the same as for ``xgboost.Booster.predict``.

    Returns
    -------
    modin.pandas.DataFrame
        Modin DataFrame with prediction results.
    """
    s = time.time()
    dmatrix_kwargs = data.get_dmatrix_params()

    # Get metadata from DMatrix
    input_index, input_columns, row_lengths = data.metadata

    # Infer columns of result
    def _get_num_columns(booster, n_features, **kwargs):
        rng = np.random.RandomState(777)
        test_data = rng.randn(1, n_features)
        test_predictions = booster.predict(
            xgb.DMatrix(test_data), validate_features=False, **kwargs
        )
        num_columns = (
            test_predictions.shape[1] if len(test_predictions.shape) > 1 else 1
        )
        return num_columns

    result_num_columns = _get_num_columns(booster, len(input_columns), **kwargs)
    new_columns = list(range(result_num_columns))

    # Put common data in object store
    booster = RayWrapper.put(booster)
    new_columns_ref = RayWrapper.put(new_columns)

    prediction_refs = [
        _map_predict.remote(booster, part, new_columns_ref, dmatrix_kwargs, **kwargs)
        for _, part in data.data
    ]
    predictions = from_partitions(
        prediction_refs,
        0,
        index=input_index,
        columns=new_columns,
        row_lengths=row_lengths,
        column_widths=[len(new_columns)],
    )
    LOGGER.info(f"Prediction time: {time.time() - s} s")
    return predictions
