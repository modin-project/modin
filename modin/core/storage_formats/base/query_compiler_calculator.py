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

from modin.core.storage_formats.base.query_compiler import (
    BaseQueryCompiler,
    QCCoercionCost,
)
from modin.logging import get_logger
from modin.logging.metrics import emit_metric


class AggregatedBackendData:
    """
    Contains information on Backends considered for computation.

    Parameters
    ----------
    backend : str
        String representing the backend name.
    query_compiler : QueryCompiler
    """

    def __init__(self, backend: str, query_compiler: BaseQueryCompiler):
        self.backend = backend
        self.qc_cls = type(query_compiler)
        self.cost = 0
        self.max_cost = query_compiler.max_cost()


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
    """

    def __init__(
        self,
        operation_arguments: MappingProxyType[str, Any],
        api_cls_name: Optional[str],
        operation: str,
    ):
        self._backend_data: dict[str, AggregatedBackendData] = {}
        self._qc_list: list[BaseQueryCompiler] = []
        self._result_backend = None
        self._api_cls_name = api_cls_name
        self._op = operation
        self._operation_arguments = operation_arguments

    def add_query_compiler(self, query_compiler: BaseQueryCompiler):
        """
        Add a query compiler to be considered for casting.

        Parameters
        ----------
        query_compiler : QueryCompiler
        """
        self._qc_list.append(query_compiler)
        backend = query_compiler.get_backend()
        backend_data = AggregatedBackendData(backend, query_compiler)
        self._backend_data[backend] = backend_data

    def calculate(self) -> str:
        """
        Calculate which query compiler we should cast to.

        Returns
        -------
        str
            A string representing a backend.
        """
        if self._result_backend is not None:
            return self._result_backend
        if len(self._qc_list) == 1:
            return self._qc_list[0].get_backend()
        if len(self._qc_list) == 0:
            raise ValueError("No query compilers registered")
        qc_from_cls_costed = set()
        # instance selection
        for qc_from in self._qc_list:

            # Add self cost for the current query compiler
            if type(qc_from) not in qc_from_cls_costed:
                self_cost = qc_from.stay_cost(
                    self._api_cls_name, self._op, self._operation_arguments
                )
                backend_from = qc_from.get_backend()
                if self_cost is not None:
                    self._add_cost_data(backend_from, self_cost)
                qc_from_cls_costed.add(type(qc_from))

            qc_to_cls_costed = set()
            for qc_to in self._qc_list:
                qc_cls_to = type(qc_to)
                if qc_cls_to not in qc_to_cls_costed:
                    qc_to_cls_costed.add(qc_cls_to)
                    backend_to = qc_to.get_backend()
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
                f"BackendCostCalculator Results: {self._calc_result_log(self._result_backend)}"
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
                f"Cannot cast to any of the available backends, as the estimated cost is too high. Tried these backends: [{','.join(self._backend_data.keys())}]"
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
        return ",".join(
            f"{'*'+k if k is selected_backend else k}:{v.cost}/{v.max_cost}"
            for k, v in self._backend_data.items()
        )
