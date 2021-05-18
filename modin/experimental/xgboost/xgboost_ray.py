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
Module holds internal entities for Modin XGBoost on Ray backend.

Class ModinXGBoostActor provides interfaces to run XGBoost operations
on remote workers. Other functions create Ray actors, distribute data between them, etc.
"""

import time
import logging
from typing import Dict, List
import math
from collections import defaultdict

import xgboost as xgb
import ray
from ray.services import get_node_ip_address
from ray.util.placement_group import placement_group, remove_placement_group
import pandas

from modin.distributed.dataframe.pandas import from_partitions
from .utils import RabitContext, RabitContextManager

LOGGER = logging.getLogger("[modin.xgboost]")


@ray.remote
class ModinXGBoostActor:
    """
    Ray actor-class runs training/prediction on remote workers.

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

    def _get_dmatrix(self, X_y):
        """
        Create xgboost.DMatrix from sequence of pandas.DataFrame objects.

        First half of `X_y` should contains objects for `X`, second for `y`.

        Parameters
        ----------
        X_y : list
            List of pandas.DataFrame objects.

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

        return xgb.DMatrix(X, y, nthread=self._nthreads)

    def get_ip(self):
        """
        Get IP address of actor.

        Returns
        -------
        str
            An IP address of node used by actor.
        """
        return get_node_ip_address()

    def set_train_data(self, *X_y, add_as_eval_method=None):
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
        """
        self._dtrain = self._get_dmatrix(X_y)

        if add_as_eval_method is not None:
            self._evals.append((self._dtrain, add_as_eval_method))

    def set_predict_data(
        self,
        *X,
    ):
        """
        Set prediction data for actor.

        Parameters
        ----------
        *X : iterable
            Sequence of ray.ObjectRef objects. When it is passed in actor, auto-materialization
            of ray.ObjectRef -> pandas.DataFrame happens.
        """
        X = pandas.concat(X, axis=0)
        self._dpredict = {
            "dmatrix": xgb.DMatrix(X, nthread=self._nthreads),
            "index": X.index,
        }

    def add_eval_data(self, *X_y, eval_method):
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
        """
        self._evals.append((self._get_dmatrix(X_y), eval_method))

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

    def predict(self, booster: xgb.Booster, **kwargs):
        """
        Run local XGBoost prediction.

        Parameters
        ----------
        booster : xgboost.Booster
            A trained booster.
        **kwargs : dict
            Other parameters for `xgboost.Booster.predict`.

        Returns
        -------
        tuple
            Pair of IP address of caller and pandas.DataFrame
            with partial prediction result.
        """
        local_dpredict = self._dpredict
        booster.set_param({"nthread": self._nthreads})

        s = time.time()

        predictions = pandas.DataFrame(
            booster.predict(local_dpredict["dmatrix"], **kwargs),
            index=local_dpredict["index"],
        )
        LOGGER.info(f"Local prediction time: {time.time() - s} s")

        return get_node_ip_address(), predictions


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


def _get_num_actors(num_actors):
    """
    Get number on actors to create.

    In case `num_actors` is 'default_train' string,
    integer number of actors will be computed  by condition
    2 CPUs per 1 actor, in case 'default_predict' 8 CPUs per 1 actor.

    Parameters
    ----------
    num_actors : int or {'default_train', 'default_predict'}
        Number of actors to create or str to set defaults.

    Returns
    -------
    int
        Number of actors to create.
    """
    min_cpus_per_node = _get_min_cpus_per_node()
    if num_actors == "default_train":
        num_actors_per_node = max(1, int(min_cpus_per_node // 2))
        return num_actors_per_node * len(ray.nodes())
    elif num_actors == "default_predict":
        num_actors_per_node = max(1, int(min_cpus_per_node // 8))
        return num_actors_per_node * len(ray.nodes())
    elif isinstance(num_actors, int):
        return num_actors
    else:
        RuntimeError("`num_actors` must be int or None")


def _create_placement_group(num_cpus_per_actor, num_actors):
    """
    Create Ray placement group to grab resources.

    Parameters
    ----------
    num_cpus_per_actor : int
        Number of CPUs per actor.
    num_actors : int
        Number of actors.

    Returns
    -------
    ray.util.PlacementGroup
        Placement group with grabbed resources.
    """
    cpu_bundle = {"CPU": num_cpus_per_actor}
    bundles = [cpu_bundle for _ in range(num_actors)]

    pg = placement_group(bundles, strategy="SPREAD")

    ready, _ = ray.wait([pg.ready()], timeout=100)

    if ready is None:
        raise TimeoutError("Placement group creation timeout.")

    return pg


def create_actors(num_actors):
    """
    Create ModinXGBoostActors.

    Parameters
    ----------
    num_actors : int
        Number of actors to create.

    Returns
    -------
    tuple
        Pair of actors list and placement group of actors.
    """
    num_cpus_per_actor = _get_cpus_per_actor(num_actors)
    pg = _create_placement_group(num_cpus_per_actor, num_actors)
    actors = [
        ModinXGBoostActor.options(
            num_cpus=num_cpus_per_actor, placement_group=pg
        ).remote(i, nthread=num_cpus_per_actor)
        for i in range(num_actors)
    ]
    return actors, pg


def _split_data_across_actors(
    actors: List, set_func, X_parts, y_parts=None, is_predict=False
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
    y_parts : list, optional
        Row partitions of y data.
    is_predict : bool, default: False
        Is split data for predict or not.
    """
    X_parts_by_actors = _assign_row_partitions_to_actors(
        actors,
        X_parts,
        is_predict=is_predict,
    )

    if y_parts is not None:
        y_parts_by_actors = _assign_row_partitions_to_actors(
            actors,
            y_parts,
            data_for_aligning=X_parts_by_actors,
        )

    for rank, actor in enumerate(actors):
        X_parts = X_parts_by_actors[rank] if is_predict else X_parts_by_actors[rank][0]
        if y_parts is None:
            set_func(actor, *X_parts)
        else:
            y_parts = y_parts_by_actors[rank][0]
            set_func(actor, *(X_parts + y_parts))


def _assign_row_partitions_to_actors(
    actors: List,
    row_partitions,
    data_for_aligning=None,
    is_predict=False,
):
    """
    Assign row_partitions to actors.

    In case of `is_predict` == False, `row_partitions` will be
    assigned to actors according to their IPs. If distribution isn't
    even, partitions will be moved from actor with excess partitions
    to actor with lack of them.

    In case of `is_predict` == True, `row_partitions` will be
    assigned evenly to actors in order.

    Parameters
    ----------
    actors : list
        List of used actors.
    row_partitions : list
        Row partitions of data to assign.
    data_for_aligning : dict, optional
        Data according to the order of which should be
        distributed `row_partitions`. Used to align y with X.
    is_predict : bool, default: False
        Is split data for predict or not.

    Returns
    -------
    dict
        Dictionary of assigned to actors partitions
        as {actor_rank: (partitions, order)}.
    """
    num_actors = len(actors)
    if not is_predict:
        if data_for_aligning is None:
            parts_ips_ref, parts_ref = zip(*row_partitions)

            # Group actors which are one the same ip
            actor_ips = defaultdict(list)

            for rank, ip in enumerate(
                ray.get([actor.get_ip.remote() for actor in actors])
            ):
                actor_ips[ip].append(rank)

            # Get distribution of parts between nodes ({ip:[(part, position),..],..})
            init_parts_distribution = defaultdict(list)
            for idx, (ip, part_ref) in enumerate(
                zip(ray.get(list(parts_ips_ref)), parts_ref)
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

                row_partitions_by_actors[rank].extend(
                    init_parts_distribution[pop_slice]
                )
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
    else:
        row_parts_by_ranks = defaultdict(list)
        _, parts_ref = zip(*row_partitions)

        num_parts = len(parts_ref)
        min_parts_per_actor = math.floor(num_parts / num_actors)
        max_parts_per_actor = math.ceil(num_parts / num_actors)
        num_actors_with_max_parts = num_parts % num_actors

        start_idx = 0
        for rank, actor in enumerate(actors):
            if num_actors_with_max_parts > 0:
                num_actor_parts = max_parts_per_actor
                num_actors_with_max_parts -= 1
            else:
                num_actor_parts = min_parts_per_actor

            idx_slice = slice(start_idx, start_idx + num_actor_parts)
            row_parts_by_ranks[rank].extend(parts_ref[idx_slice])
            start_idx += num_actor_parts

    return row_parts_by_ranks


def _train(
    dtrain,
    num_actors,
    params: Dict,
    *args,
    evals=(),
    **kwargs,
):
    """
    Run distributed training of XGBoost model on Ray backend.

    During work it evenly distributes `dtrain` between workers according
    to IP addresses partitions (in case of not even distribution of `dtrain`
    by nodes, part of partitions will be re-distributed between nodes),
    runs xgb.train on each worker for subset of `dtrain` and reduces training results
    of each worker using Rabit Context.

    Parameters
    ----------
    dtrain : modin.experimental.DMatrix
        Data to be trained against.
    num_actors : int, optional
        Number of actors for training. If unspecified, this value will be
        computed automatically.
    params : dict
        Booster params.
    *args : iterable
        Other parameters for `xgboost.train`.
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

    assert len(X_row_parts) == len(y_row_parts), "Unaligned train data"

    num_actors = _get_num_actors(
        num_actors if isinstance(num_actors, int) else "default_train"
    )

    if num_actors > len(X_row_parts):
        num_actors = len(X_row_parts)

    actors, pg = create_actors(num_actors)

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
                eval_X,
                y_parts=eval_y,
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
            actor.train.remote(rabit_args, params, *args, **kwargs) for actor in actors
        ]
        # All results should be the same because of Rabit tracking. So we just
        # return the first one.
        result = ray.get(fut[0])
        remove_placement_group(pg)
        LOGGER.info(f"Training time: {time.time() - s} s")
        return result


def _predict(
    booster,
    data,
    num_actors,
    **kwargs,
):
    """
    Run distributed prediction with a trained booster on Ray backend.

    During work it evenly distributes `data` between workers,
    runs xgb.predict on each worker for subset of `data` and creates
    Modin DataFrame with prediction results.

    Parameters
    ----------
    booster : xgboost.Booster
        A trained booster.
    data : modin.experimental.xgboost.DMatrix
        Input data used for prediction.
    num_actors : int, optional
        Number of actors for prediction. If unspecified, this value will be
        computed automatically.
    **kwargs : dist
        Other parameters are the same as `xgboost.Booster.predict`.

    Returns
    -------
    modin.pandas.DataFrame
        Modin DataFrame with prediction results.
    """
    s = time.time()

    X_row_parts, _ = data

    num_actors = _get_num_actors(
        num_actors if isinstance(num_actors, int) else "default_predict"
    )

    if num_actors > len(X_row_parts):
        num_actors = len(X_row_parts)

    # Create remote actors
    actors, pg = create_actors(num_actors)

    # Split data across workers
    _split_data_across_actors(
        actors,
        lambda actor, *X: actor.set_predict_data.remote(*X),
        X_row_parts,
        is_predict=True,
    )

    LOGGER.info(f"Data preparation time: {time.time() - s} s")
    s = time.time()

    booster = ray.put(booster)

    predictions = [
        tuple(actor.predict._remote(args=(booster,), kwargs=kwargs, num_returns=2))
        for actor in actors
    ]

    ray.wait([part for _, part in predictions], num_returns=len(predictions))
    remove_placement_group(pg)

    result = from_partitions(predictions, 0)
    LOGGER.info(f"Prediction time: {time.time() - s} s")

    return result
