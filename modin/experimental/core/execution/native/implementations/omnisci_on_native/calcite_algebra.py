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
Module provides classes for relational algebra expressions.

Provided classes reflect relational algebra format used by
OmniSci storage format.
"""

import abc
from .expr import BaseExpr


class CalciteInputRefExpr(BaseExpr):
    """
    Calcite version of input column reference.

    Calcite translation should replace all ``InputRefExpr``.

    Calcite references columns by their indexes (positions in input table).
    If there are multiple input tables for Calcite node, then a position
    in a concatenated list of all columns is used.

    Parameters
    ----------
    idx : int
        Input column index.

    Attributes
    ----------
    input : int
        Input column index.
    """

    def __init__(self, idx):
        self.input = idx

    def copy(self):
        """
        Make a shallow copy of the expression.

        Returns
        -------
        CalciteInputRefExpr
        """
        return CalciteInputRefExpr(self.input)

    def __repr__(self):
        """
        Return a string representation of the expression.

        Returns
        -------
        str
        """
        return f"(input {self.input})"


class CalciteInputIdxExpr(BaseExpr):
    """
    Basically the same as ``CalciteInputRefExpr`` but with a different serialization.

    Parameters
    ----------
    idx : int
        Input column index.

    Attributes
    ----------
    input : int
        Input column index.
    """

    def __init__(self, idx):
        self.input = idx

    def copy(self):
        """
        Make a shallow copy of the expression.

        Returns
        -------
        CalciteInputIdxExpr
        """
        return CalciteInputIdxExpr(self.input)

    def __repr__(self):
        """
        Return a string representation of the expression.

        Returns
        -------
        str
        """
        return f"(input_idx {self.input})"


class CalciteBaseNode(abc.ABC):
    """
    A base class for a Calcite computation sequence node.

    Calcite nodes are not combined into a tree but usually stored
    in a sequence which works similar to a stack machine: the result
    of the previous operation is an implicit operand of the current
    one. Input nodes also can be referenced directly via its unique
    ID number.

    Calcite nodes structure is based on a JSON representation used by
    OmniSci for parsed queries serialization/deserialization for
    interactions with a Calcite server. Currently, this format is
    internal and is not a part of public API. It's not documented
    and can be modified in an incompatible way in the future.

    Parameters
    ----------
    relOp : str
        An operation name.

    Attributes
    ----------
    id : int
        Id of the node. Should be unique within a single query.
    relOp : str
        Operation name.
    """

    _next_id = [0]

    def __init__(self, relOp):
        self.id = str(type(self)._next_id[0])
        type(self)._next_id[0] += 1
        self.relOp = relOp

    @classmethod
    def reset_id(cls):
        """
        Reset ID to be used for the next new node to 0.

        Can be used to have a zero-based numbering for each
        generated query.
        """
        cls._next_id[0] = 0


class CalciteScanNode(CalciteBaseNode):
    """
    A node to represent a scan operation.

    Scan operation can only be applied to physical tables.

    Parameters
    ----------
    modin_frame : OmnisciOnNativeDataframe
        A frame to scan. The frame should have a materialized table
        in OmniSci.

    Attributes
    ----------
    table : list of str
        A list holding a database name and a table name.
    fieldNames : list of str
        A list of columns to include into the scan.
    inputs : list
        An empty list existing for the sake of serialization
        simplicity. Has no meaning but is expected by OmniSci
        deserializer.
    """

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
    """
    A node to represent a projection operation.

    Parameters
    ----------
    fields : list of str
        Output column names.
    exprs : list of BaseExpr
        Output column expressions.

    Attributes
    ----------
    fields : list of str
        A list of output columns.
    exprs : list of BaseExpr
        A list of expressions describing how output columns are computed.
        Order of expression follows `fields` order.
    """

    def __init__(self, fields, exprs):
        super(CalciteProjectionNode, self).__init__("LogicalProject")
        self.fields = [f"F_{field}" for field in fields]
        self.exprs = exprs


class CalciteFilterNode(CalciteBaseNode):
    """
    A node to represent a filter operation.

    Parameters
    ----------
    condition : BaseExpr
        A filtering condition.

    Attributes
    ----------
    condition : BaseExpr
        A filter to apply.
    """

    def __init__(self, condition):
        super(CalciteFilterNode, self).__init__("LogicalFilter")
        self.condition = condition


class CalciteAggregateNode(CalciteBaseNode):
    """
    A node to represent an aggregate operation.

    Parameters
    ----------
    fields : list of str
        Output field names.
    group : list of CalciteInputIdxExpr
        Group key columns.
    aggs : list of BaseExpr
        Aggregates to compute.

    Attributes
    ----------
    fields : list of str
        Output field names.
    group : list of CalciteInputIdxExpr
        Group key columns.
    aggs : list of BaseExpr
        Aggregates to compute.
    """

    def __init__(self, fields, group, aggs):
        super(CalciteAggregateNode, self).__init__("LogicalAggregate")
        self.fields = [f"F_{field}" for field in fields]
        self.group = group
        self.aggs = aggs


class CalciteCollation:
    """
    A structure to describe sorting order.

    Parameters
    ----------
    field : CalciteInputIdxExpr
        A column to sort by.
    dir : {"ASCENDING", "DESCENDING"}, default: "ASCENDING"
        A sort order.
    nulls : {"LAST", "FIRST"}, default: "LAST"
        NULLs position after the sort.

    Attributes
    ----------
    field : CalciteInputIdxExpr
        A column to sort by.
    dir : {"ASCENDING", "DESCENDING"}
        A sort order.
    nulls : {"LAST", "FIRST"}
        NULLs position after the sort.
    """

    def __init__(self, field, dir="ASCENDING", nulls="LAST"):
        self.field = field
        self.direction = dir
        self.nulls = nulls


class CalciteSortNode(CalciteBaseNode):
    """
    A node to represent a sort operation.

    Parameters
    ----------
    collation : list of CalciteCollation
        Sort keys.

    Attributes
    ----------
    collation : list of CalciteCollation
        Sort keys.
    """

    def __init__(self, collation):
        super(CalciteSortNode, self).__init__("LogicalSort")
        self.collation = collation


class CalciteJoinNode(CalciteBaseNode):
    """
    A node to represent a join operation.

    Parameters
    ----------
    left_id : int
        ID of the left join operand.
    right_id : int
        ID of the right join operand.
    how : str
        Type of the join.
    condition : BaseExpr
        Join condition.

    Attributes
    ----------
    inputs : list of int
        IDs of the left and the right operands of the join.
    joinType : str
        Type of the join.
    condition : BaseExpr
        Join condition.
    """

    def __init__(self, left_id, right_id, how, condition):
        super(CalciteJoinNode, self).__init__("LogicalJoin")
        self.inputs = [left_id, right_id]
        self.joinType = how
        self.condition = condition


class CalciteUnionNode(CalciteBaseNode):
    """
    A node to represent a union operation.

    Parameters
    ----------
    inputs : list of int
        Input frame IDs.
    all : bool
        True for UNION ALL operation.

    Attributes
    ----------
    inputs : list of int
        Input frame IDs.
    all : bool
        True for UNION ALL operation.
    """

    def __init__(self, inputs, all):
        super(CalciteUnionNode, self).__init__("LogicalUnion")
        self.inputs = inputs
        self.all = all
