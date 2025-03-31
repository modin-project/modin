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

from modin.core.storage_formats.base.query_compiler import (
    BaseQueryCompiler,
    QCCoercionCost,
)


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
        self.max_cost = query_compiler.qc_engine_switch_max_cost()


class BackendCostCalculator:
    """
    Calculate which Backend should be used for an operation.

    Given a set of QueryCompilers containing various data, determine
    which query compiler's backend would minimize the cost of casting
    or coercion. Use the aggregate sum of coercion to determine overall
    cost.
    """

    def __init__(self):
        self._backend_data = {}
        self._qc_list = []
        self._result_backend = None

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
        if len(self._qc_list) == 0:
            raise ValueError("No query compilers registered")

        # instance selection
        for qc_from in self._qc_list:
            qc_to_cls_costed = set()
            for qc_to in self._qc_list:
                qc_cls_to = type(qc_to)
                if qc_cls_to not in qc_to_cls_costed:
                    qc_to_cls_costed.add(qc_cls_to)
                    backend_to = qc_to.get_backend()
                    cost = qc_from.qc_engine_switch_cost(qc_cls_to)
                    if cost is not None:
                        self._add_cost_data(backend_to, cost)

        min_value = None
        for k, v in self._backend_data.items():
            if v.cost > v.max_cost:
                continue
            if min_value is None or min_value > v.cost:
                min_value = v.cost
                self._result_backend = k

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
