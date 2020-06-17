from .expr import *
from .calcite_algebra import *
from .df_algebra import FrameNode
from pandas.core.dtypes.common import is_integer_dtype
import json


class CalciteSerializer:
    dtype_strings = {"int64": "INTEGER", "bool": "BOOLEAN", "float64": "DOUBLE"}

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

    def serialize_typed_obj(self, obj):
        res = self.serialize_obj(obj)
        force_decimal = self.force_decimal_type(obj)
        res["type"] = self.serialize_dtype(obj._dtype, force_decimal)
        return res

    def serialize_expr(self, expr):
        if isinstance(expr, LiteralExpr):
            return self.serialize_literal(expr)
        elif isinstance(expr, CalciteInputRefExpr):
            return self.serialize_obj(expr)
        elif isinstance(expr, CalciteInputIdxExpr):
            return self.serialize_input_idx(expr)
        elif isinstance(expr, OpExpr):
            return self.serialize_typed_obj(expr)
        elif isinstance(expr, AggregateExpr):
            return self.serialize_typed_obj(expr)
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
        if type(literal.val) is str:
            return {
                "literal": literal.val,
                "type": "CHAR",
                "target_type": "CHAR",
                "scale": -2147483648,
                "precision": len(literal.val),
                "type_scale": -2147483648,
                "type_precision": len(literal.val),
            }
        if type(literal.val) is int:
            return {
                "literal": literal.val,
                "type": "DECIMAL",
                "target_type": "BIGINT",
                "scale": 0,
                "precision": len(str(literal.val)),
                "type_scale": 0,
                "type_precision": 19,
            }
        if type(literal.val) is float:
            str_val = f"{literal.val:f}"
            precision = len(str_val) - 1
            scale = precision - str_val.index(".")
            print("SERIALIZE", literal.val, "AS", int(str_val.replace(".", "")))
            return {
                "literal": int(str_val.replace(".", "")),
                "type": "DECIMAL",
                "target_type": "DECIMAL",
                "scale": scale,
                "precision": precision,
                "type_scale": scale,
                "type_precision": precision,
            }
        raise NotImplemented(f"Can not serialize {type(literal.val).__name__}")

    def force_decimal_type(self, obj):
        """In some cases calcite representation requieres DECIMAL type
           with 0 scale instead of an INTEGER type. Dectect such cases
           here."""
        return isinstance(obj, OpExpr) and obj.op == "FLOOR"

    def serialize_dtype(self, dtype, force_decimal):
        if is_integer_dtype(dtype) and force_decimal:
            return {"type": "DECIMAL", "nullable": True, "scale": 0}
        return {"type": type(self).dtype_strings[str(dtype)], "nullable": True}

    def serialize_input_idx(self, expr):
        return expr.input
