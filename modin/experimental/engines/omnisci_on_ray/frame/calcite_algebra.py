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
from pandas.core.dtypes.common import is_list_like


class CalciteBaseExpr(abc.ABC):
    def eq(self, other):
        if not isinstance(other, CalciteBaseExpr):
            other = CalciteLiteralExpr(other)
        res_type = CalciteOpExprType("BOOLEAN", False)
        new_expr = CalciteOpExpr("=", [self, other], res_type)
        return new_expr


class CalciteInputRefExpr(CalciteBaseExpr):
    def __init__(self, input_idx):
        self.input = input_idx


class CalciteLiteralExpr(CalciteBaseExpr):
    def __init__(self, val):
        # For now only INTEGER is supported. Integers should be
        # passed as DECIMAL values with scale = 0
        # TODO: support various integer types and other types
        # (float, text, dates etc.)
        assert isinstance(val, int)
        self.literal = val
        self.type = "DECIMAL"
        self.target_type = "BIGINT"
        self.scale = 0
        self.precision = len(str(val))
        self.type_scale = 0
        self.type_precision = 19


class CalciteOpExprType(CalciteBaseExpr):
    def __init__(self, expr_type, nullable):
        self.type = expr_type
        self.nullable = nullable


class CalciteOpExpr(CalciteBaseExpr):
    def __init__(self, op, operands, res_type):
        self.op = op
        self.operands = operands
        self.type = res_type


class CalciteAggregateExpr(CalciteBaseExpr):
    def __init__(self, agg, operands, res_type, distinct):
        self.agg = agg.upper()
        self.operands = operands
        if not isinstance(self.operands, list):
            self.operands = [self.operands]
        self.type = res_type
        self.distinct = distinct


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
        self.table = ["modin_db", modin_frame._partitions[0][0].frame_id]
        self.fieldNames = modin_frame._table_cols + ["rowid"]
        # OmniSci expects from scan node to have 'inputs' field
        # holding empty list
        self.inputs = []


class CalciteProjectionNode(CalciteBaseNode):
    def __init__(self, fields, exprs):
        super(CalciteProjectionNode, self).__init__("LogicalProject")
        self.fields = fields
        self.exprs = exprs


class CalciteFilterNode(CalciteBaseNode):
    def __init__(self, condition):
        super(CalciteFilterNode, self).__init__("LogicalFilter")
        self.condition = condition


class CalciteAggregateNode(CalciteBaseNode):
    def __init__(self, fields, group, aggs):
        super(CalciteAggregateNode, self).__init__("LogicalAggregate")
        self.fields = fields
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


def build_calcite_row_idx_filter_expr(row_idx, row_col):
    """Build calcite expression to filter rows by rowid.

    Parameters
    ----------
    row_idx
        The row numeric indices to select
    row_col
        CalciteInputRefExpr referencing proper rowid column to filter by
    Returns
    -------
    CalciteBaseExpr
        A CalciteBaseExpr implementing filter condition
    """
    if not is_list_like(row_idx):
        return row_col.eq(row_idx)

    exprs = []
    for idx in row_idx:
        exprs.append(row_col.eq(idx))

    res_type = CalciteOpExprType("BOOLEAN", False)
    res = CalciteOpExpr("OR", exprs, res_type)

    return res
