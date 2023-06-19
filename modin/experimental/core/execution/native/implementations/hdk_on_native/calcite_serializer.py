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

"""Module provides ``CalciteSerializer`` class."""

from pandas.core.dtypes.common import is_datetime64_dtype

from .expr import (
    BaseExpr,
    LiteralExpr,
    OpExpr,
    AggregateExpr,
)
from .calcite_algebra import (
    CalciteBaseNode,
    CalciteInputRefExpr,
    CalciteInputIdxExpr,
    CalciteScanNode,
    CalciteProjectionNode,
    CalciteFilterNode,
    CalciteAggregateNode,
    CalciteCollation,
    CalciteSortNode,
    CalciteJoinNode,
    CalciteUnionNode,
)
from modin.error_message import ErrorMessage
import json
import numpy as np


def _warn_if_unsigned(dtype):  # noqa: GL08
    if np.issubdtype(dtype, np.unsignedinteger):
        ErrorMessage.single_warning(
            "HDK does not support unsigned integer types, such types will be rounded up to the signed equivalent."
        )


class CalciteSerializer:
    """
    Serializer for calcite node sequence.

    ``CalciteSerializer`` is used to serialize a sequence of ``CalciteBaseNode``
    based nodes including nested ``BaseExpr`` based expression trees into
    a request in JSON format which can be fed to HDK.
    """

    _DTYPE_STRINGS = {
        "int8": "TINYINT",
        "int16": "SMALLINT",
        "int32": "INTEGER",
        "int64": "BIGINT",
        "uint8": "SMALLINT",
        "uint16": "INTEGER",
        "uint32": "BIGINT",
        "uint64": "BIGINT",
        "bool": "BOOLEAN",
        "float32": "FLOAT",
        "float64": "DOUBLE",
        "datetime64": "TIMESTAMP",
    }

    _INT_OPTS = {
        np.int8: ("TINYINT", 3),
        np.int16: ("SMALLINT", 5),
        np.int32: ("INTEGER", 10),
        np.int64: ("BIGINT", 19),
        np.uint8: ("SMALLINT", 5),
        np.uint16: ("INTEGER", 10),
        np.uint32: ("BIGINT", 19),
        np.uint64: ("BIGINT", 19),
        int: ("BIGINT", 19),
    }

    _TIMESTAMP_PRECISION = {
        "s": 0,
        "ms": 3,
        "us": 6,
        "ns": 9,
    }
    _DTYPE_STRINGS.update(
        {f"datetime64[{u}]": "TIMESTAMP" for u in _TIMESTAMP_PRECISION}
    )

    def serialize(self, plan):
        """
        Serialize a sequence of Calcite nodes into JSON format.

        Parameters
        ----------
        plan : list of CalciteBaseNode
            A sequence to serialize.

        Returns
        -------
        str
            A query in JSON format.
        """
        return json.dumps({"rels": [self.serialize_item(node) for node in plan]})

    def expect_one_of(self, val, *types):
        """
        Raise an error if values doesn't belong to any of specified types.

        Parameters
        ----------
        val : Any
            Value to check.
        *types : list of type
            Allowed value types.
        """
        for t in types:
            if isinstance(val, t):
                return
        raise TypeError("Can not serialize {}".format(type(val).__name__))

    def serialize_item(self, item):
        """
        Serialize a single expression item.

        Parameters
        ----------
        item : Any
            Item to serialize.

        Returns
        -------
        str, int, None, dict or list of dict
            Serialized item.
        """
        if isinstance(item, CalciteBaseNode):
            return self.serialize_node(item)
        elif isinstance(item, BaseExpr):
            return self.serialize_expr(item)
        elif isinstance(item, CalciteCollation):
            return self.serialize_obj(item)
        elif isinstance(item, list):
            return [self.serialize_item(v) for v in item]
        elif isinstance(item, dict):
            return {k: self.serialize_item(v) for k, v in item.items()}

        self.expect_one_of(item, str, int, type(None))
        return item

    def serialize_node(self, node):
        """
        Serialize a frame operation.

        Parameters
        ----------
        node : CalciteBaseNode
            A node to serialize.

        Returns
        -------
        dict
            Serialized object.
        """
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
        """
        Serialize an object into a dictionary.

        Add all non-hidden attributes (not starting with '_') of the object
        to the output dictionary.

        Parameters
        ----------
        obj : object
            An object to serialize.

        Returns
        -------
        dict
            Serialized object.
        """
        res = {}
        for k, v in obj.__dict__.items():
            if k[0] != "_":
                if k == "op" and isinstance(obj, OpExpr) and v == "//":
                    res[k] = "/"
                else:
                    res[k] = self.serialize_item(v)
        return res

    def serialize_typed_obj(self, obj):
        """
        Serialize an object and its dtype into a dictionary.

        Similar to `serialize_obj` but also include '_dtype' field
        of the object under 'type' key.

        Parameters
        ----------
        obj : object
            An object to serialize.

        Returns
        -------
        dict
            Serialized object.
        """
        res = self.serialize_obj(obj)
        res["type"] = self.serialize_dtype(obj._dtype)
        return res

    def serialize_expr(self, expr):
        """
        Serialize ``BaseExpr`` based expression into a dictionary.

        Parameters
        ----------
        expr : BaseExpr
            An expression to serialize.

        Returns
        -------
        dict
            Serialized expression.
        """
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
        """
        Serialize ``LiteralExpr`` into a dictionary.

        Parameters
        ----------
        literal : LiteralExpr
            A literal to serialize.

        Returns
        -------
        dict
            Serialized literal.
        """
        val = literal.val
        if val is None:
            return {
                "literal": None,
                "type": "BIGINT",
                "target_type": "BIGINT",
                "scale": 0,
                "precision": 19,
                "type_scale": 0,
                "type_precision": 19,
            }
        if type(val) is str:
            return {
                "literal": val,
                "type": "CHAR",
                "target_type": "CHAR",
                "scale": -2147483648,
                "precision": len(val),
                "type_scale": -2147483648,
                "type_precision": len(val),
            }
        if type(val) in self._INT_OPTS.keys():
            target_type, precision = self.opts_for_int_type(type(val))
            return {
                "literal": int(val),
                "type": "DECIMAL",
                "target_type": target_type,
                "scale": 0,
                "precision": len(str(val)),
                "type_scale": 0,
                "type_precision": precision,
            }
        if type(val) in (float, np.float64):
            if np.isnan(val):
                return {
                    "literal": None,
                    "type": "DOUBLE",
                    "target_type": "DOUBLE",
                    "scale": 0,
                    "precision": 19,
                    "type_scale": 0,
                    "type_precision": 19,
                }

            str_val = f"{val:f}"
            precision = len(str_val) - 1
            scale = precision - str_val.index(".")
            return {
                "literal": int(str_val.replace(".", "")),
                "type": "DECIMAL",
                "target_type": "DOUBLE",
                "scale": scale,
                "precision": precision,
                "type_scale": -2147483648,
                "type_precision": 15,
            }
        if type(val) is bool:
            return {
                "literal": val,
                "type": "BOOLEAN",
                "target_type": "BOOLEAN",
                "scale": -2147483648,
                "precision": 1,
                "type_scale": -2147483648,
                "type_precision": 1,
            }
        if isinstance(val, np.datetime64):
            unit = np.datetime_data(val)[0]
            precision = self._TIMESTAMP_PRECISION.get(unit, None)
            if precision is not None:
                return {
                    "literal": int(val.astype(np.int64)),
                    "type": "TIMESTAMP",
                    "target_type": "TIMESTAMP",
                    "scale": -2147483648,
                    "precision": precision,
                    "type_scale": -2147483648,
                    "type_precision": precision,
                }

        raise NotImplementedError(f"Can not serialize {type(val).__name__}")

    def opts_for_int_type(self, int_type):
        """
        Get serialization params for an integer type.

        Return a SQL type name and a number of meaningful decimal digits
        for an integer type.

        Parameters
        ----------
        int_type : type
            An integer type to describe.

        Returns
        -------
        tuple
        """
        try:
            _warn_if_unsigned(int_type)
            return self._INT_OPTS[int_type]
        except KeyError:
            raise NotImplementedError(f"Unsupported integer type {int_type.__name__}")

    def serialize_dtype(self, dtype):
        """
        Serialize data type to a dictionary.

        Parameters
        ----------
        dtype : dtype
            Data type to serialize.

        Returns
        -------
        dict
            Serialized data type.
        """
        _warn_if_unsigned(dtype)
        try:
            type_info = {"type": self._DTYPE_STRINGS[dtype.name], "nullable": True}
            if is_datetime64_dtype(dtype):
                unit = np.datetime_data(dtype)[0]
                type_info["precision"] = self._TIMESTAMP_PRECISION[unit]
            return type_info
        except KeyError:
            raise TypeError(f"Unsupported dtype: {dtype}")

    def serialize_input_idx(self, expr):
        """
        Serialize ``CalciteInputIdxExpr`` expression.

        Parameters
        ----------
        expr : CalciteInputIdxExpr
            An expression to serialize.

        Returns
        -------
        int
            Serialized expression.
        """
        return expr.input
