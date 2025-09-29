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
Module contains ``BackendCostCalculator`` class.

``BackendCostCalculator`` is used to determine the casting cost
between a set of different backends. It aggregates the cost across
all query compilers to determine the best query compiler to use.
"""

import random
from types import MappingProxyType
from typing import Any, Optional

from modin.config import Backend, BackendJoinConsiderAllBackends
from modin.core.storage_formats.base.query_compiler import (
    BaseQueryCompiler,
    QCCoercionCost,
)
from modin.logging import get_logger
from modin.logging.metrics import emit_metric


def all_switchable_backends() -> list[str]:
    """
    Return a list of all currently active backends that are candidates for switching.

    Returns
    -------
    list
        A list of valid backends.
    """
    return list(
        filter(
            # Disable automatically switching to these engines for now, because
            # 1) _get_prepared_factory_for_backend() currently calls
            # _initialize_engine(), which starts up the ray/dask/unidist
            #  processes
            # 2) we can't decide to switch to unidist in the middle of execution.
            lambda backend: backend not in ("Ray", "Unidist", "Dask"),
            Backend.get_active_backends(),
        )
    )


class AggregatedBackendData:
    """
    Contains information on Backends considered for computation.

    Parameters
    ----------
    backend : str
        String representing the backend name.
    qc_cls : type[QueryCompiler]
        The query compiler sub-class for this backend.
    """

    def __init__(self, backend: str, qc_cls: type[BaseQueryCompiler]):
        self.backend = backend
        self.qc_cls = qc_cls
        self.cost = 0
        self.max_cost = qc_cls.max_cost()


class BackendCostCalculator:
    """
    Calculate which Backend should be used for an operation.

    Given a set of QueryCompilers containing various data, determine
    which query compiler's backend would minimize the cost of casting
    or coercion. Use the aggregate sum of coercion to determine overall
    cost.

    Parameters
    ----------
    operation_arguments : MappingProxyType[str, Any]
        Mapping from operation argument names to their values.
    api_cls_name : str or None
        Representing the class name of the function being called.
    operation : str representing the operation being performed
    query_compilers : list of query compiler arguments
    preop_switch : bool
        True if the operation is a pre-operation switch point.
    """

    def __init__(
        self,
        *,
        operation_arguments: MappingProxyType[str, Any],
        api_cls_name: Optional[str],
        operation: str,
        query_compilers: list[BaseQueryCompiler],
        preop_switch: bool,
    ):
        from modin.core.execution.dispatching.factories.dispatcher import (
            FactoryDispatcher,
        )

        self._qc_list: list[BaseQueryCompiler] = []
        self._result_backend = None
        self._api_cls_name = api_cls_name
        self._op = operation
        self._operation_arguments = operation_arguments
        self._backend_data = {}
        self._qc_list = query_compilers[:]
        for query_compiler in query_compilers:
            # If a QC's backend was not configured as active, we need to create an entry for it here.
            backend = query_compiler.get_backend()
            if backend not in self._backend_data:
                self._backend_data[backend] = AggregatedBackendData(
                    backend,
                    FactoryDispatcher._get_prepared_factory_for_backend(
                        backend=backend
                    ).io_cls.query_compiler_cls,
                )
        if preop_switch and BackendJoinConsiderAllBackends.get():
            # Initialize backend data for any backends not found among query compiler arguments.
            # Because we default to the first query compiler's backend if no cost information is available,
            # this initialization must occur after iterating over query compiler arguments to ensure
            # correct ordering in dictionary arguments.
            for backend in all_switchable_backends():
                if backend not in self._backend_data:
                    self._backend_data[backend] = AggregatedBackendData(
                        backend,
                        FactoryDispatcher._get_prepared_factory_for_backend(
                            backend=backend
                        ).io_cls.query_compiler_cls,
                    )

    def calculate(self) -> str:
        """
        Calculate which query compiler we should cast to.

        Switching calculation is performed as follows:
        - For every registered query compiler in qc_list, with backend `backend_from`, compute
          `self_cost = qc_from.stay_cost(...)` and add it to the total cost for `backend_from`.
          - For every valid target `backend_to`, compute `qc_from.move_to_cost(qc_cls_to, ...)`. If it
            returns None, instead compute `qc_cls_to.move_to_me_cost(qc_from, ...)`. Add the result
            to the cost for `backend_to`.
        At a high level, the cost for choosing a particular backend is the sum of
            (all stay costs for data already on that backend)
            + (cost of moving all other query compilers to this backend)

        If the operation is a registered pre-operation switch point, then the list of target backends
        is ALL active backends. Otherwise, only backends found among the arguments are considered.
        Post-operation switch points are not yet supported.

        If the arguments contain no query compilers for a particular backend, then there are no stay
        costs. In this scenario, we expect the move_to cost for this backend to outweigh the corresponding
        stay costs for each query compiler's original backend.

        If no argument QCs have cost information for each other (that is, move_to_cost and move_to_me_cost
        returns None), then we attempt to move all data to the backend of the first QC.

        We considered a few alternative algorithms for switching calculation:

        1. Instead of considering all active backends, consider only backends found among input QCs.
        This was used in the calculator's original implementation, as we figured transfer cost to
        unrelated backends would outweigh any possible gains in computation speed. However, certain
        pathological cases that significantly changed the size of input or output data (e.g. cross join)
        would create situations where transferring data after the computation became prohibitively
        expensive, so we chose to allow switching to unrelated backends.
        Additionally, the original implementation had a bug where stay_cost was only computed for the
        _first_ query compiler of each backend, thus under-reporting the cost of computation for any
        backend with multiple QCs present. In practice this very rarely affected the chosen result.
        2. Compute stay/move costs only once for each backend pair, but force QCs to consider other
        arguments when calculating.
        This approach is the most robust and accurate for cases like cross join, where a product of
        transfer costs between backends is more reflective of cost than size. This approach requires
        more work in the query compiler, as each QC must be aware of when multiple QC arguments are
        passed and adjust the cost computation accordingly. It is also unclear how often this would
        make a meaningful difference compared to the summation approach.

        Returns
        -------
        str
            A string representing a backend.

        Raises
        ------
        ValueError
            Raises ValueError when the reported transfer cost for every backend exceeds its maximum cost.
        """
        if self._result_backend is not None:
            return self._result_backend
        if len(self._qc_list) == 1:
            return self._qc_list[0].get_backend()
        if len(self._qc_list) == 0:
            raise ValueError("No query compilers registered")
        # See docstring for explanation of switching decision algorithm.
        for qc_from in self._qc_list:
            # Add self cost for the current query compiler
            self_cost = qc_from.stay_cost(
                self._api_cls_name, self._op, self._operation_arguments
            )
            backend_from = qc_from.get_backend()
            if self_cost is not None:
                self._add_cost_data(backend_from, self_cost)

            for backend_to, agg_data_to in self._backend_data.items():
                if backend_to == backend_from:
                    continue
                qc_cls_to = agg_data_to.qc_cls
                cost = qc_from.move_to_cost(
                    qc_cls_to,
                    self._api_cls_name,
                    self._op,
                    self._operation_arguments,
                )
                if cost is not None:
                    self._add_cost_data(backend_to, cost)
                else:
                    # We have some information asymmetry in query compilers,
                    # qc_from does not know about qc_to types so we instead
                    # ask the same question but of qc_to.
                    cost = qc_cls_to.move_to_me_cost(
                        qc_from,
                        self._api_cls_name,
                        self._op,
                        self._operation_arguments,
                    )
                    if cost is not None:
                        self._add_cost_data(backend_to, cost)

        min_value = None
        for k, v in self._backend_data.items():
            if v.cost > v.max_cost:
                continue
            if min_value is None or min_value > v.cost:
                min_value = v.cost
                self._result_backend = k

        if len(self._backend_data) > 1:
            get_logger().info(
                f"BackendCostCalculator results for {'pd' if self._api_cls_name is None else self._api_cls_name}.{self._op}: {self._calc_result_log(self._result_backend)}"
            )
            # Does not need to be secure, should not use system entropy
            metrics_group = "%04x" % random.randrange(16**4)
            for qc in self._qc_list:
                max_shape = qc._max_shape()
                backend = qc.get_backend()
                emit_metric(
                    f"hybrid.merge.candidate.{backend}.group.{metrics_group}.rows",
                    max_shape[0],
                )
                emit_metric(
                    f"hybrid.merge.candidate.{backend}.group.{metrics_group}.cols",
                    max_shape[1],
                )
            for k, v in self._backend_data.items():
                emit_metric(
                    f"hybrid.merge.candidate.{k}.group.{metrics_group}.cost", v.cost
                )
            emit_metric(
                f"hybrid.merge.decision.{self._result_backend}.group.{metrics_group}",
                1,
            )

        if self._result_backend is None:
            raise ValueError(
                f"Cannot cast to any of the available backends, as the estimated cost is too high. Tried these backends: [{', '.join(self._backend_data.keys())}]"
            )
        return self._result_backend

    def _add_cost_data(self, backend, cost):
        """
        Add the cost data to the calculator.

        Parameters
        ----------
        backend : str
            String representing the backend for this engine.
        cost : dict
            Dictionary of query compiler classes to costs.
        """
        # We can assume that if we call this method, backend
        # exists in the backend_data map
        QCCoercionCost.validate_coersion_cost(cost)
        self._backend_data[backend].cost += cost

    def _calc_result_log(self, selected_backend: str) -> str:
        """
        Create a string summary of the backend costs.

        The format is
            [*|][backend name]:[cost]/[max_cost],...
        where '*' indicates this was the selected backend
        and [cost]/[max_cost] represents the aggregated
        cost of moving to that backend over the maximum
        cost allowed on that backend.

        Parameters
        ----------
        selected_backend : str
            String representing the backend selected by
            the calculator.

        Returns
        -------
        str
            String representation of calculator state.
        """
        return ", ".join(
            f"{'*'+k if k is selected_backend else k}:{v.cost}/{v.max_cost}"
            for k, v in self._backend_data.items()
        )
