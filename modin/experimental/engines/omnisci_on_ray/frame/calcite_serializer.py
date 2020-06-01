from .expr import *
from .calcite_algebra import *
from .df_algebra import FrameNode

import json


class CalciteSerializer:
    type_strings = {
        int: "INTEGER",
        bool: "BOOLEAN",
    }

    def serialize(self, plan):
        return json.dumps({"rels": [self.serialize_item(node) for node in plan]})

    def expect_one_of(self, val, *types):
        for t in types:
            if isinstance(val, t):
                return
        raise TypeError("Can not serialize {}".format(type(val).__name__))

    def serialize_item(self, item):
        if isinstance(item, CalciteBaseNode):
            return self.serialize_node(item)
        elif isinstance(item, BaseExpr):
            return self.serialize_expr(item)
        elif isinstance(item, CalciteCollation):
            return self.serialize_obj(item)
        elif isinstance(item, list):
            return [self.serialize_item(v) for v in item]

        self.expect_one_of(item, str, int)
        return item

    def serialize_node(self, node):
        # We need to setup context for proper references
        # serialization
        if isinstance(
            node,
            (
                CalciteScanNode,
                CalciteProjectionNode,
                CalciteFilterNode,
                CalciteAggregateNode,
                CalciteSortNode,
                CalciteJoinNode,
                CalciteUnionNode,
            ),
        ):
            return self.serialize_obj(node)
        else:
            raise NotImplementedError(
                "Can not serialize {}".format(type(node).__name__)
            )

    def serialize_obj(self, obj):
        res = {}
        for k, v in obj.__dict__.items():
            if k[0] != "_":
                res[k] = self.serialize_item(v)
        return res

    def serialize_expr(self, expr):
        if isinstance(expr, LiteralExpr):
            return self.serialize_literal(expr)
        elif isinstance(expr, OpExprType):
            return self.serialize_type(expr)
        elif isinstance(expr, CalciteInputRefExpr):
            return self.serialize_obj(expr)
        elif isinstance(expr, CalciteInputIdxExpr):
            return self.serialize_input_idx(expr)
        elif isinstance(expr, OpExpr):
            return self.serialize_obj(expr)
        elif isinstance(expr, AggregateExpr):
            return self.serialize_obj(expr)
        else:
            raise NotImplementedError(
                "Can not serialize {}".format(type(expr).__name__)
            )

    def serialize_literal(self, literal):
        if literal.val is None:
            return {
                "literal": None,
                "type": "NULL",
                "target_type": "BIGINT",
                "scale": 0,
                "precision": 19,
                "type_scale": 0,
                "type_precision": 19,
            }
        self.expect_one_of(literal.val, int)
        return {
            "literal": literal.val,
            "type": "DECIMAL",
            "target_type": "BIGINT",
            "scale": 0,
            "precision": len(str(literal.val)),
            "type_scale": 0,
            "type_precision": 19,
        }

    def serialize_type(self, typ):
        assert typ.type in type(self).type_strings
        return {"type": type(self).type_strings[typ.type], "nullable": typ.nullable}

    def serialize_input_idx(self, expr):
        return expr.input
