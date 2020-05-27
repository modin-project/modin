from .expr import *
from .calcite_algebra import *

import json


class CalciteSerializer:
    def serialize(self, plan):
        return json.dumps({"rels": [self.serialize_obj(node) for node in plan]})

    def expect_one_of(self, val, *types):
        for t in types:
            if isinstance(val, t):
                return
        raise TypeError("Can not serialize {}".format(typ.__name__))

    def serialize_obj(self, obj):
        self.expect_one_of(obj, CalciteCollation, BaseExpr, CalciteBaseNode)
        res = {}
        for k, v in obj.__dict__.items():
            res[k] = self.serialize_item(v)
        return res

    def serialize_item(self, item):
        if isinstance(item, list):
            return [self.serialize_item(v) for v in item]
        if isinstance(item, BaseExpr):
            return self.serialize_expr(item)
        if isinstance(item, CalciteCollation):
            return self.serialize_obj(item)
        self.expect_one_of(item, str, int)
        return item

    def serialize_expr(self, expr):
        if isinstance(expr, LiteralExpr):
            return self.serialize_literal(expr)
        if isinstance(expr, OpExprType):
            return self.serialize_type(expr)
        return self.serialize_obj(expr)

    def serialize_literal(self, literal):
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
        type_strings = {
            int: "INTEGER",
            bool: "BOOLEAN",
        }
        assert typ.type in type_strings
        return {"type": type_strings[typ.type], "nullable": typ.nullable}