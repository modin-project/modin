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

from modin.core.storage_formats.base.query_compiler import QCCoercionCost


class QueryCompilerCostCalculator:
    """
    Calculate which QueryCompiler should be used for an operation.

    Given a set of QueryCompilers containing various data, determine
    which query compiler's backend would minimize the cost of casting
    or coercion. Use the aggregate sum of coercion to determine overall
    cost.
    """

    def __init__(self):
        self._compiler_class_to_cost = {}
        self._compiler_class_to_data_class = {}
        self._qc_list = []
        self._qc_cls_set = set()
        self._result_type = None

    def add_query_compiler(self, query_compiler):
        """
        Add a query compiler to be considered for casting.

        Parameters
        ----------
        query_compiler : QueryCompiler
        """
        if isinstance(query_compiler, type):
            # class
            qc_type = query_compiler
        else:
            # instance
            qc_type = type(query_compiler)
            self._qc_list.append(query_compiler)
            self._compiler_class_to_data_class[qc_type] = type(
                query_compiler._modin_frame
            )
        self._qc_cls_set.add(qc_type)

    def calculate(self):
        """
        Calculate which query compiler we should cast to.

        Returns
        -------
        type
            QueryCompiler class which should be used for the operation.
        """
        if self._result_type is not None:
            return self._result_type
        if len(self._qc_cls_set) == 1:
            return list(self._qc_cls_set)[0]
        if len(self._qc_cls_set) == 0:
            raise ValueError("No query compilers registered")

        for qc_from in self._qc_list:
            for qc_cls_to in self._qc_cls_set:
                cost = qc_from.qc_engine_switch_cost(qc_cls_to)
                if cost is not None:
                    self._add_cost_data({qc_cls_to: cost})
            self._add_cost_data({type(qc_from): QCCoercionCost.COST_ZERO})
        min_value = min(self._compiler_class_to_cost.values())
        for key, value in self._compiler_class_to_cost.items():
            if min_value == value:
                self._result_type = key
                break
        return self._result_type

    def _add_cost_data(self, costs: dict):
        """
        Add the cost data to the calculator.

        Parameters
        ----------
        costs : dict
            Dictionary of query compiler classes to costs.
        """
        for k, v in costs.items():
            # filter out any extranious query compilers not in this operation
            if k in self._qc_cls_set:
                QCCoercionCost.validate_coersion_cost(v)
                # Adds the costs associated with all coercions to a type, k
                self._compiler_class_to_cost[k] = (
                    v + self._compiler_class_to_cost[k]
                    if k in self._compiler_class_to_cost
                    else v
                )

    def result_data_cls(self):
        """
        Return the data frame associated with the calculated query compiler.

        Returns
        -------
        DataFrame object
            DataFrame object associated with the preferred query compiler.
        """
        qc_type = self.calculate()
        if qc_type in self._compiler_class_to_data_class:
            return self._compiler_class_to_data_class[qc_type]
        else:
            return None
