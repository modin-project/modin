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

import abc
from .expr import BaseExpr


class CalciteInputRefExpr(BaseExpr):
    """Caclcite version of input column reference.

    Calcite translation should replace all InputRefExpr with
    CalciteInputRefExpr. Calcite references columns by their
    indexes (positions in input table). If there are multiple
    input tables for Caclcite node, then index in concatenated
    list of all columns is used.
    """

    def __init__(self, idx):
        self.input = idx

    def copy(self):
        return CalciteInputRefExpr(self.input)

    def __repr__(self):
        return f"(input {self.input})"


class CalciteInputIdxExpr(BaseExpr):
    """Same as CalciteInputRefExpr but with different serialization"""

    def __init__(self, idx):
        self.input = idx

    def copy(self):
        return CalciteInputIdxExpr(self.input)

    def __repr__(self):
        return f"(input_idx {self.input})"


class CalciteBaseNode(abc.ABC):
    _next_id = [0]

    def __init__(self, relOp):
        self.id = str(type(self)._next_id[0])
        type(self)._next_id[0] += 1
        self.relOp = relOp

    @classmethod
    def reset_id(cls):
        cls._next_id[0] = 0


class CalciteScanNode(CalciteBaseNode):
    def __init__(self, modin_frame):
        assert modin_frame._partitions.size == 1
        assert modin_frame._partitions[0][0].frame_id is not None
        super(CalciteScanNode, self).__init__("EnumerableTableScan")
        self.table = ["omnisci", modin_frame._partitions[0][0].frame_id]
        self.fieldNames = [f"F_{col}" for col in modin_frame._table_cols] + ["rowid"]
        # OmniSci expects from scan node to have 'inputs' field
        # holding empty list
        self.inputs = []


class CalciteProjectionNode(CalciteBaseNode):
    def __init__(self, fields, exprs):
        super(CalciteProjectionNode, self).__init__("LogicalProject")
        self.fields = [f"F_{field}" for field in fields]
        self.exprs = exprs


class CalciteFilterNode(CalciteBaseNode):
    def __init__(self, condition):
        super(CalciteFilterNode, self).__init__("LogicalFilter")
        self.condition = condition


class CalciteAggregateNode(CalciteBaseNode):
    def __init__(self, fields, group, aggs):
        super(CalciteAggregateNode, self).__init__("LogicalAggregate")
        self.fields = [f"F_{field}" for field in fields]
        self.group = group
        self.aggs = aggs


class CalciteCollation:
    def __init__(self, field, dir="ASCENDING", nulls="LAST"):
        self.field = field
        self.direction = dir
        self.nulls = nulls


class CalciteSortNode(CalciteBaseNode):
    def __init__(self, collation):
        super(CalciteSortNode, self).__init__("LogicalSort")
        self.collation = collation


class CalciteJoinNode(CalciteBaseNode):
    def __init__(self, left_id, right_id, how, condition):
        super(CalciteJoinNode, self).__init__("LogicalJoin")
        self.inputs = [left_id, right_id]
        self.joinType = how
        self.condition = condition


class CalciteUnionNode(CalciteBaseNode):
    def __init__(self, inputs, all):
        super(CalciteUnionNode, self).__init__("LogicalUnion")
        self.inputs = inputs
        self.all = all
