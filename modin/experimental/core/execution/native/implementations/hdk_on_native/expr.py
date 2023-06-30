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

"""Module provides classes for scalar expression trees."""

import abc
from typing import Union, Generator, Type

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

import pandas
from pandas.core.dtypes.common import (
    is_list_like,
    get_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_string_dtype,
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_bool_dtype,
    is_datetime64_dtype,
)

from modin.pandas.indexing import is_range_like
from modin.utils import _inherit_docstrings
from .dataframe.utils import ColNameCodec, to_arrow_type


def _get_common_dtype(lhs_dtype, rhs_dtype):
    """
    Get data type for a binary operation result.

    Parameters
    ----------
    lhs_dtype : dtype
        The type of the first operand.
    rhs_dtype : dtype
        The type of the second operand.

    Returns
    -------
    dtype
        The result data type.
    """
    if lhs_dtype == rhs_dtype:
        return lhs_dtype
    if is_float_dtype(lhs_dtype) and (
        is_float_dtype(rhs_dtype) or is_integer_dtype(rhs_dtype)
    ):
        return get_dtype(float)
    if is_float_dtype(rhs_dtype) and (
        is_float_dtype(lhs_dtype) or is_integer_dtype(lhs_dtype)
    ):
        return get_dtype(float)
    if is_integer_dtype(lhs_dtype) and is_integer_dtype(rhs_dtype):
        return get_dtype(int)
    if is_datetime64_dtype(lhs_dtype) and is_datetime64_dtype(rhs_dtype):
        return np.promote_types(lhs_dtype, rhs_dtype)
    raise NotImplementedError(
        f"Cannot perform operation on types: {lhs_dtype}, {rhs_dtype}"
    )


_aggs_preserving_numeric_type = {"sum", "min", "max"}
_aggs_with_int_result = {"count", "size"}
_aggs_with_float_result = {"mean", "median", "std", "skew"}


def _agg_dtype(agg, dtype):
    """
    Compute aggregate data type.

    Parameters
    ----------
    agg : str
        Aggregate name.
    dtype : dtype
        Operand data type.

    Returns
    -------
    dtype
        The aggregate data type.
    """
    if agg in _aggs_preserving_numeric_type:
        return dtype
    elif agg in _aggs_with_int_result:
        return get_dtype(int)
    elif agg in _aggs_with_float_result:
        return get_dtype(float)
    else:
        raise NotImplementedError(f"unsupported aggregate {agg}")


_cmp_ops = {"eq", "ge", "gt", "le", "lt", "ne"}


def is_cmp_op(op):
    """
    Check if operation is a comparison.

    Parameters
    ----------
    op : str
        Operation to check.

    Returns
    -------
    bool
        True for comparison operations and False otherwise.
    """
    return op in _cmp_ops


_logical_ops = {"and", "or"}


def is_logical_op(op):
    """
    Check if operation is a logical one.

    Parameters
    ----------
    op : str
        Operation to check.

    Returns
    -------
    bool
        True for logical operations and False otherwise.
    """
    return op in _logical_ops


class BaseExpr(abc.ABC):
    """
    An abstract base class for expression tree node.

    An expression tree is used to describe how a single column of a dataframe
    is computed.

    Each node can belong to multiple trees and therefore should be immutable
    until proven to have no parent nodes (e.g. by making a copy).

    Attributes
    ----------
    operands : list of BaseExpr, optional
        Holds child nodes. Leaf nodes shouldn't have `operands` attribute.
    """

    binary_operations = {
        "add": "+",
        "sub": "-",
        "mul": "*",
        "mod": "MOD",
        "floordiv": "//",
        "truediv": "/",
        "pow": "POWER",
        "eq": "=",
        "ge": ">=",
        "gt": ">",
        "le": "<=",
        "lt": "<",
        "ne": "<>",
        "and": "AND",
        "or": "OR",
    }

    preserve_dtype_math_ops = {"add", "sub", "mul", "mod", "floordiv", "pow"}
    promote_to_float_math_ops = {"truediv"}

    def eq(self, other):
        """
        Build an equality comparison of `self` with `other`.

        Parameters
        ----------
        other : BaseExpr or scalar
            An operand to compare with.

        Returns
        -------
        BaseExpr
            The resulting comparison expression.
        """
        return self.cmp("=", other)

    def le(self, other):
        """
        Build a less or equal comparison with `other`.

        Parameters
        ----------
        other : BaseExpr or scalar
            An operand to compare with.

        Returns
        -------
        BaseExpr
            The resulting comparison expression.
        """
        return self.cmp("<=", other)

    def ge(self, other):
        """
        Build a greater or equal comparison with `other`.

        Parameters
        ----------
        other : BaseExpr or scalar
            An operand to compare with.

        Returns
        -------
        BaseExpr
            The resulting comparison expression.
        """
        return self.cmp(">=", other)

    def cmp(self, op, other):
        """
        Build a comparison expression with `other`.

        Parameters
        ----------
        op : str
            A comparison operation.
        other : BaseExpr or scalar
            An operand to compare with.

        Returns
        -------
        BaseExpr
            The resulting comparison expression.
        """
        if not isinstance(other, BaseExpr):
            other = LiteralExpr(other)
        return OpExpr(op, [self, other], get_dtype(bool))

    def cast(self, res_type):
        """
        Build a cast expression.

        Parameters
        ----------
        res_type : dtype
            A data type to cast to.

        Returns
        -------
        BaseExpr
            The cast expression.
        """
        # From float to int cast we expect truncate behavior but CAST
        # operation would give us round behavior.
        if is_float_dtype(self._dtype) and is_integer_dtype(res_type):
            return self.floor()

        new_expr = OpExpr("CAST", [self], res_type)
        return new_expr

    def is_null(self):
        """
        Build a NULL check expression.

        Returns
        -------
        BaseExpr
            The NULL check expression.
        """
        new_expr = OpExpr("IS NULL", [self], get_dtype(bool))
        return new_expr

    def is_not_null(self):
        """
        Build a NOT NULL check expression.

        Returns
        -------
        BaseExpr
            The NOT NULL check expression.
        """
        new_expr = OpExpr("IS NOT NULL", [self], get_dtype(bool))
        return new_expr

    def bin_op(self, other, op_name):
        """
        Build a binary operation expression.

        Parameters
        ----------
        other : BaseExpr
            The second operand.
        op_name : str
            A binary operation name.

        Returns
        -------
        BaseExpr
            The resulting binary operation expression.
        """
        if op_name not in self.binary_operations:
            raise NotImplementedError(f"unsupported binary operation {op_name}")

        if is_cmp_op(op_name):
            return self._cmp_op(other, op_name)

        # True division may require prior cast to float to avoid integer division
        if op_name == "truediv":
            if is_integer_dtype(self._dtype) and is_integer_dtype(other._dtype):
                other = other.cast(get_dtype(float))
        res_type = self._get_bin_op_res_type(op_name, self._dtype, other._dtype)
        new_expr = OpExpr(self.binary_operations[op_name], [self, other], res_type)
        # Floor division may require additional FLOOR expr.
        if op_name == "floordiv" and not is_integer_dtype(res_type):
            return new_expr.floor()
        return new_expr

    def add(self, other):
        """
        Build an add expression.

        Parameters
        ----------
        other : BaseExpr
            The second operand.

        Returns
        -------
        BaseExpr
            The resulting add expression.
        """
        return self.bin_op(other, "add")

    def sub(self, other):
        """
        Build a sub expression.

        Parameters
        ----------
        other : BaseExpr
            The second operand.

        Returns
        -------
        BaseExpr
            The resulting sub expression.
        """
        return self.bin_op(other, "sub")

    def mul(self, other):
        """
        Build a mul expression.

        Parameters
        ----------
        other : BaseExpr
            The second operand.

        Returns
        -------
        BaseExpr
            The resulting mul expression.
        """
        return self.bin_op(other, "mul")

    def mod(self, other):
        """
        Build a mod expression.

        Parameters
        ----------
        other : BaseExpr
            The second operand.

        Returns
        -------
        BaseExpr
            The resulting mod expression.
        """
        return self.bin_op(other, "mod")

    def truediv(self, other):
        """
        Build a truediv expression.

        The result always has float data type.

        Parameters
        ----------
        other : BaseExpr
            The second operand.

        Returns
        -------
        BaseExpr
            The resulting truediv expression.
        """
        return self.bin_op(other, "truediv")

    def floordiv(self, other):
        """
        Build a floordiv expression.

        The result always has an integer data type.

        Parameters
        ----------
        other : BaseExpr
            The second operand.

        Returns
        -------
        BaseExpr
            The resulting floordiv expression.
        """
        return self.bin_op(other, "floordiv")

    def pow(self, other):
        """
        Build a power expression.

        Parameters
        ----------
        other : BaseExpr
            The power operand.

        Returns
        -------
        BaseExpr
            The resulting power expression.
        """
        return self.bin_op(other, "pow")

    def floor(self):
        """
        Build a floor expression.

        Returns
        -------
        BaseExpr
            The resulting floor expression.
        """
        return OpExpr("FLOOR", [self], get_dtype(int))

    def invert(self) -> "OpExpr":
        """
        Build a bitwise inverse expression.

        Returns
        -------
        OpExpr
            The resulting bitwise inverse expression.
        """
        return OpExpr("~", [self], self._dtype)

    def _cmp_op(self, other, op_name):
        """
        Build a comparison expression.

        Parameters
        ----------
        other : BaseExpr
            A value to compare with.
        op_name : str
            The comparison operation name.

        Returns
        -------
        BaseExpr
            The resulting comparison expression.
        """
        lhs_dtype_class = self._get_dtype_cmp_class(self._dtype)
        rhs_dtype_class = self._get_dtype_cmp_class(other._dtype)
        res_dtype = get_dtype(bool)
        # In HDK comparison with NULL always results in NULL,
        # but in pandas it is True for 'ne' comparison and False
        # for others.
        # Also pandas allows 'eq' and 'ne' comparison for values
        # of incompatible types which doesn't work in HDK.
        if lhs_dtype_class != rhs_dtype_class:
            if op_name == "eq" or op_name == "ne":
                return LiteralExpr(op_name == "ne")
            else:
                raise TypeError(
                    f"Invalid comparison between {self._dtype} and {other._dtype}"
                )
        else:
            cmp = OpExpr(self.binary_operations[op_name], [self, other], res_dtype)
            return build_if_then_else(
                self.is_null(), LiteralExpr(op_name == "ne"), cmp, res_dtype
            )

    @staticmethod
    def _get_dtype_cmp_class(dtype):
        """
        Get a comparison class name for specified data type.

        Values of different comparison classes cannot be compared.

        Parameters
        ----------
        dtype : dtype
            A data type of a compared value.

        Returns
        -------
        str
            The comparison class name.
        """
        if is_numeric_dtype(dtype) or is_bool_dtype(dtype):
            return "numeric"
        if is_string_dtype(dtype) or is_categorical_dtype(dtype):
            return "string"
        if is_datetime64_any_dtype(dtype):
            return "datetime"
        return "other"

    def _get_bin_op_res_type(self, op_name, lhs_dtype, rhs_dtype):
        """
        Return the result data type for a binary operation.

        Parameters
        ----------
        op_name : str
            A binary operation name.
        lhs_dtype : dtype
            A left operand's type.
        rhs_dtype : dtype
            A right operand's type.

        Returns
        -------
        dtype
        """
        if op_name in self.preserve_dtype_math_ops:
            return _get_common_dtype(lhs_dtype, rhs_dtype)
        elif op_name in self.promote_to_float_math_ops:
            return get_dtype(float)
        elif is_cmp_op(op_name):
            return get_dtype(bool)
        elif is_logical_op(op_name):
            return get_dtype(bool)
        else:
            raise NotImplementedError(f"unsupported binary operation {op_name}")

    @abc.abstractmethod
    def copy(self):
        """
        Make a shallow copy of the expression.

        Returns
        -------
        BaseExpr
        """
        pass

    def nested_expressions(
        self,
    ) -> Generator[Type["BaseExpr"], Type["BaseExpr"], Type["BaseExpr"]]:
        """
        Return a generator that allows to iterate over and replace the nested expressions.

        If the generator receives a new expression, it creates a copy of `self` and
        replaces the expression in the copy. The copy is returned to the sender.

        Returns
        -------
        Generator
        """
        expr = self
        if operands := getattr(self, "operands", None):
            for i, op in enumerate(operands):
                new_op = yield op
                if new_op is not None:
                    if new_op is not op:
                        if expr is self:
                            expr = self.copy()
                        expr.operands[i] = new_op
                    yield expr
        return expr

    def collect_frames(self, frames):
        """
        Recursively collect all frames participating in the expression.

        Collected frames are put into the `frames` set. Default implementation
        collects frames from the operands of the expression. Derived classes
        directly holding frames should provide their own implementations.

        Parameters
        ----------
        frames : set
            Output set of collected frames.
        """
        for expr in self.nested_expressions():
            expr.collect_frames(frames)

    # currently we translate only exprs with a single input frame
    def translate_input(self, mapper):
        """
        Make a deep copy of the expression translating input nodes using `mapper`.

        The default implementation builds a copy and recursively run
        translation for all its operands. For leaf expressions
        `_translate_input` is called.

        Parameters
        ----------
        mapper : InputMapper
            A mapper to use for input columns translation.

        Returns
        -------
        BaseExpr
            The expression copy with translated input columns.
        """
        res = None
        gen = self.nested_expressions()
        for expr in gen:
            res = gen.send(expr.translate_input(mapper))
        return self._translate_input(mapper) if res is None else res

    def _translate_input(self, mapper):
        """
        Make a deep copy of the expression translating input nodes using `mapper`.

        Called by default translator for leaf nodes. Method should be overriden
        by derived classes holding input references.

        Parameters
        ----------
        mapper : InputMapper
            A mapper to use for input columns translation.

        Returns
        -------
        BaseExpr
            The expression copy with translated input columns.
        """
        return self

    def fold(self):
        """
        Fold the operands.

        This operation is used by `TransformNode` when translating to base.

        Returns
        -------
        BaseExpr
        """
        res = self
        gen = self.nested_expressions()
        for expr in gen:
            res = gen.send(expr.fold())
        return res

    def can_execute_hdk(self) -> bool:
        """
        Check for possibility of HDK execution.

        Check if the computation can be executed using an HDK query.

        Returns
        -------
        bool
        """
        return True

    def can_execute_arrow(self) -> bool:
        """
        Check for possibility of Arrow execution.

        Check if the computation can be executed using
        the Arrow API instead of HDK query.

        Returns
        -------
        bool
        """
        return False

    def execute_arrow(self, table: pa.Table) -> pa.ChunkedArray:
        """
        Compute the column data using the Arrow API.

        Parameters
        ----------
        table : pa.Table

        Returns
        -------
        pa.ChunkedArray
        """
        raise RuntimeError(f"Arrow execution is not supported by {type(self)}")


class InputRefExpr(BaseExpr):
    """
    An expression tree node to represent an input frame column.

    Parameters
    ----------
    frame : HdkOnNativeDataframe
        An input frame.
    col : str
        An input column name.
    dtype : dtype
        Input column data type.

    Attributes
    ----------
    modin_frame : HdkOnNativeDataframe
        An input frame.
    column : str
        An input column name.
    _dtype : dtype
        Input column data type.
    """

    def __init__(self, frame, col, dtype):
        self.modin_frame = frame
        self.column = col
        self._dtype = dtype

    def copy(self):
        """
        Make a shallow copy of the expression.

        Returns
        -------
        InputRefExpr
        """
        return InputRefExpr(self.modin_frame, self.column, self._dtype)

    def collect_frames(self, frames):
        """
        Add referenced frame to the `frames` set.

        Parameters
        ----------
        frames : set
            Output set of collected frames.
        """
        frames.add(self.modin_frame)

    def _translate_input(self, mapper):
        """
        Translate the referenced column using `mapper`.

        Parameters
        ----------
        mapper : InputMapper
            A mapper to use for input column translation.

        Returns
        -------
        BaseExpr
            The translated expression.
        """
        return mapper.translate(self)

    @_inherit_docstrings(BaseExpr.fold)
    def fold(self):
        return self

    @_inherit_docstrings(BaseExpr.can_execute_arrow)
    def can_execute_arrow(self) -> bool:
        return True

    @_inherit_docstrings(BaseExpr.execute_arrow)
    def execute_arrow(self, table: pa.Table) -> pa.ChunkedArray:
        if self.column == ColNameCodec.ROWID_COL_NAME:
            return pa.chunked_array([range(len(table))], pa.int64())
        return table.column(ColNameCodec.encode(self.column))

    def __repr__(self):
        """
        Return a string representation of the expression.

        Returns
        -------
        str
        """
        return f"{self.modin_frame.id_str()}.{self.column}[{self._dtype}]"


class LiteralExpr(BaseExpr):
    """
    An expression tree node to represent a literal value.

    Parameters
    ----------
    val : int, np.int, float, bool, str, np.datetime64 or None
        Literal value.
    dtype : None or dtype, default: None
        Value dtype.

    Attributes
    ----------
    val : int, np.int, float, bool, str, np.datetime64 or None
        Literal value.
    _dtype : dtype
        Literal data type.
    """

    def __init__(self, val, dtype=None):
        if val is not None and not isinstance(
            val,
            (
                int,
                float,
                bool,
                str,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
                np.datetime64,
            ),
        ):
            raise NotImplementedError(f"Literal value {val} of type {type(val)}")
        self.val = val
        if dtype is not None:
            self._dtype = dtype
        elif val is None:
            self._dtype = get_dtype(float)
        else:
            self._dtype = (
                val.dtype if isinstance(val, np.generic) else get_dtype(type(val))
            )

    def copy(self):
        """
        Make a shallow copy of the expression.

        Returns
        -------
        LiteralExpr
        """
        return LiteralExpr(self.val)

    @_inherit_docstrings(BaseExpr.fold)
    def fold(self):
        return self

    @_inherit_docstrings(BaseExpr.cast)
    def cast(self, res_type):
        val = self.val
        if val is not None:
            if isinstance(val, np.generic):
                val = val.astype(res_type)
            elif is_integer_dtype(res_type):
                val = int(val)
            elif is_float_dtype(res_type):
                val = float(val)
            elif is_bool_dtype(res_type):
                val = bool(val)
            elif is_string_dtype(res_type):
                val = str(val)
            else:
                raise TypeError(f"Cannot cast '{val}' to '{res_type}'")
        return LiteralExpr(val, res_type)

    @_inherit_docstrings(BaseExpr.is_null)
    def is_null(self):
        return LiteralExpr(pandas.isnull(self.val), np.dtype(bool))

    @_inherit_docstrings(BaseExpr.is_null)
    def is_not_null(self):
        return LiteralExpr(not pandas.isnull(self.val), np.dtype(bool))

    @_inherit_docstrings(BaseExpr.can_execute_arrow)
    def can_execute_arrow(self) -> bool:
        return True

    @_inherit_docstrings(BaseExpr.execute_arrow)
    def execute_arrow(self, table: pa.Table) -> pa.ChunkedArray:
        return pa.chunked_array([[self.val] * len(table)], to_arrow_type(self._dtype))

    def __repr__(self):
        """
        Return a string representation of the expression.

        Returns
        -------
        str
        """
        return f"{self.val}[{self._dtype}]"

    def __eq__(self, obj):
        """
        Check if `obj` is a `LiteralExpr` with an equal value.

        Parameters
        ----------
        obj : Any object

        Returns
        -------
        bool
        """
        return isinstance(obj, LiteralExpr) and self.val == obj.val


class OpExpr(BaseExpr):
    """
    A generic operation expression.

    Used for arithmetic, comparisons, conditional operations, etc.

    Parameters
    ----------
    op : str
        Operation name.
    operands : list of BaseExpr
        Operation operands.
    dtype : dtype
        Result data type.

    Attributes
    ----------
    op : str
        Operation name.
    operands : list of BaseExpr
        Operation operands.
    _dtype : dtype
        Result data type.
    partition_keys : list of BaseExpr, optional
        This attribute is used with window functions only and contains
        a list of column expressions to partition the result set.
    order_keys : list of dict, optional
        This attribute is used with window functions only and contains
        order clauses.
    lower_bound : dict, optional
        Lover bound for windowed aggregates.
    upper_bound : dict, optional
        Upper bound for windowed aggregates.
    """

    _FOLD_OPS = {
        "+": lambda self: self._fold_arithm("__add__"),
        "-": lambda self: self._fold_arithm("__sub__"),
        "*": lambda self: self._fold_arithm("__mul__"),
        "POWER": lambda self: self._fold_arithm("__pow__"),
        "/": lambda self: self._fold_arithm("__truediv__"),
        "//": lambda self: self._fold_arithm("__floordiv__"),
        "~": lambda self: self._fold_invert(),
        "CAST": lambda self: self._fold_literal("cast", self._dtype),
        "IS NULL": lambda self: self._fold_literal("is_null"),
        "IS NOT NULL": lambda self: self._fold_literal("is_not_null"),
    }

    _ARROW_EXEC = {
        "+": lambda self, table: self._pc("add", table),
        "-": lambda self, table: self._pc("subtract", table),
        "*": lambda self, table: self._pc("multiply", table),
        "POWER": lambda self, table: self._pc("power", table),
        "/": lambda self, table: self._pc("divide", table),
        "//": lambda self, table: self._pc("divide", table),
        "~": lambda self, table: self._invert(table),
        "CAST": lambda self, table: self._col(table).cast(to_arrow_type(self._dtype)),
        "IS NULL": lambda self, table: self._col(table).is_null(nan_is_null=True),
        "IS NOT NULL": lambda self, table: pc.invert(
            self._col(table).is_null(nan_is_null=True)
        ),
    }

    _UNSUPPORTED_HDK_OPS = {"~"}

    def __init__(self, op, operands, dtype):
        self.op = op
        self.operands = operands
        self._dtype = dtype

    def set_window_opts(self, partition_keys, order_keys, order_ascending, na_pos):
        """
        Set the window function options.

        Parameters
        ----------
        partition_keys : list of BaseExpr
        order_keys : list of BaseExpr
        order_ascending : list of bool
        na_pos : {"FIRST", "LAST"}
        """
        self.is_rows = True
        self.partition_keys = partition_keys
        self.order_keys = []
        for key, asc in zip(order_keys, order_ascending):
            key = {
                "field": key,
                "direction": "ASCENDING" if asc else "DESCENDING",
                "nulls": na_pos,
            }
            self.order_keys.append(key)
        self.lower_bound = {
            "unbounded": True,
            "preceding": True,
            "following": False,
            "is_current_row": False,
            "offset": None,
            "order_key": 0,
        }
        self.upper_bound = {
            "unbounded": False,
            "preceding": False,
            "following": False,
            "is_current_row": True,
            "offset": None,
            "order_key": 1,
        }

    def copy(self):
        """
        Make a shallow copy of the expression.

        Returns
        -------
        OpExpr
        """
        op = OpExpr(self.op, self.operands.copy(), self._dtype)
        if pk := getattr(self, "partition_keys", None):
            op.partition_keys = pk
            op.is_rows = self.is_rows
            op.order_keys = self.order_keys
            op.lower_bound = self.lower_bound
            op.upper_bound = self.upper_bound
        return op

    @_inherit_docstrings(BaseExpr.nested_expressions)
    def nested_expressions(
        self,
    ) -> Generator[Type["BaseExpr"], Type["BaseExpr"], Type["BaseExpr"]]:
        expr = yield from super().nested_expressions()
        if partition_keys := getattr(self, "partition_keys", None):
            for i, key in enumerate(partition_keys):
                new_key = yield key
                if new_key is not None:
                    if new_key is not key:
                        if expr is self:
                            expr = self.copy()
                        expr.partition_keys[i] = new_key
                    yield expr
            for i, key in enumerate(self.order_keys):
                field = key["field"]
                new_field = yield field
                if new_field is not None:
                    if new_field is not field:
                        if expr is self:
                            expr = self.copy()
                        expr.order_keys[i]["field"] = new_field
                    yield expr
        return expr

    @_inherit_docstrings(BaseExpr.fold)
    def fold(self):
        super().fold()
        return self if (op := self._FOLD_OPS.get(self.op, None)) is None else op(self)

    def _fold_arithm(self, op) -> Union["OpExpr", LiteralExpr]:
        """
        Fold arithmetic expressions.

        Parameters
        ----------
        op : str

        Returns
        -------
        OpExpr or LiteralExpr
        """
        operands = self.operands
        i = 0
        while i < len(operands):
            if isinstance((o := operands[i]), OpExpr):
                if self.op == o.op:
                    # Fold operands in case of the same operation
                    operands[i : i + 1] = o.operands
                else:
                    i += 1
                    continue
            if i == 0:
                i += 1
                continue
            if isinstance(o, LiteralExpr) and isinstance(operands[i - 1], LiteralExpr):
                # Fold two sequential literal expressions
                val = getattr(operands[i - 1].val, op)(o.val)
                operands[i - 1] = LiteralExpr(val).cast(o._dtype)
                del operands[i]
            else:
                i += 1
        return operands[0] if len(operands) == 1 else self

    def _fold_invert(self) -> Union["OpExpr", LiteralExpr]:
        """
        Fold invert expression.

        Returns
        -------
        OpExpr or LiteralExpr
        """
        assert len(self.operands) == 1
        op = self.operands[0]
        if isinstance(op, LiteralExpr):
            return LiteralExpr(~op.val, op._dtype)
        if isinstance(op, OpExpr):
            if op.op == "IS NULL":
                return OpExpr("IS NOT NULL", op.operands, op._dtype)
            if op.op == "IS NOT NULL":
                return OpExpr("IS NULL", op.operands, op._dtype)
        return self

    def _fold_literal(self, op, *args):
        """
        Fold literal expressions.

        Parameters
        ----------
        op : str

        *args : list

        Returns
        -------
        OpExpr or LiteralExpr
        """
        assert len(self.operands) == 1
        expr = self.operands[0]
        return getattr(expr, op)(*args) if isinstance(expr, LiteralExpr) else self

    @_inherit_docstrings(BaseExpr.can_execute_hdk)
    def can_execute_hdk(self) -> bool:
        return self.op not in self._UNSUPPORTED_HDK_OPS

    @_inherit_docstrings(BaseExpr.can_execute_arrow)
    def can_execute_arrow(self) -> bool:
        return self.op in self._ARROW_EXEC

    @_inherit_docstrings(BaseExpr.execute_arrow)
    def execute_arrow(self, table: pa.Table) -> pa.ChunkedArray:
        return self._ARROW_EXEC[self.op](self, table)

    def __repr__(self):
        """
        Return a string representation of the expression.

        Returns
        -------
        str
        """
        if pk := getattr(self, "partition_keys", None):
            return f"({self.op} {self.operands} {pk} {self.order_keys} [{self._dtype}])"
        return f"({self.op} {self.operands} [{self._dtype}])"

    def _col(self, table: pa.Table) -> pa.ChunkedArray:
        """
        Return the column referenced by the `InputRefExpr` operand.

        Parameters
        ----------
        table : pa.Table

        Returns
        -------
        pa.ChunkedArray
        """
        assert isinstance(self.operands[0], InputRefExpr)
        return self.operands[0].execute_arrow(table)

    def _pc(self, op: str, table: pa.Table) -> pa.ChunkedArray:
        """
        Perform the specified pyarrow.compute operation on the operands.

        Parameters
        ----------
        op : str
        table : pyarrow.Table

        Returns
        -------
        pyarrow.ChunkedArray
        """
        op = getattr(pc, op)
        val = self._op_value(0, table)
        for i in range(1, len(self.operands)):
            val = op(val, self._op_value(i, table))
        if not isinstance(val, pa.ChunkedArray):
            val = LiteralExpr(val).execute_arrow(table)
        if val.type != (at := to_arrow_type(self._dtype)):
            val = val.cast(at)
        return val

    def _op_value(self, op_idx: int, table: pa.Table):
        """
        Get the specified operand value.

        Parameters
        ----------
        op_idx : int
        table : pyarrow.Table

        Returns
        -------
        pyarrow.ChunkedArray or expr.val
        """
        expr = self.operands[op_idx]
        return expr.val if isinstance(expr, LiteralExpr) else expr.execute_arrow(table)

    def _invert(self, table: pa.Table) -> pa.ChunkedArray:
        """
        Bitwise inverse the column values.

        Parameters
        ----------
        table : pyarrow.Table

        Returns
        -------
        pyarrow.ChunkedArray
        """
        if is_bool_dtype(self._dtype):
            return pc.invert(self._col(table))

        try:
            return pc.bit_wise_not(self._col(table))
        except pa.ArrowNotImplementedError as err:
            raise TypeError(str(err))


class AggregateExpr(BaseExpr):
    """
    An aggregate operation expression.

    Parameters
    ----------
    agg : str
        Aggregate name.
    op : BaseExpr
        Aggregate operand.
    distinct : bool, default: False
        Distinct modifier for 'count' aggregate.
    dtype : dtype, optional
        Aggregate data type. Computed if not specified.

    Attributes
    ----------
    agg : str
        Aggregate name.
    operands : list of BaseExpr
        Aggregate operands. Always has a single operand.
    distinct : bool
        Distinct modifier for 'count' aggregate.
    _dtype : dtype
        Aggregate data type.
    """

    def __init__(self, agg, op, distinct=False, dtype=None):
        if agg == "nunique":
            self.agg = "count"
            self.distinct = True
        else:
            self.agg = agg
            self.distinct = distinct
        self.operands = [op]
        self._dtype = (
            dtype if dtype else _agg_dtype(self.agg, op._dtype if op else None)
        )
        assert self._dtype is not None

    def copy(self):
        """
        Make a shallow copy of the expression.

        Returns
        -------
        AggregateExpr
        """
        return AggregateExpr(self.agg, self.operands[0], self.distinct, self._dtype)

    def __repr__(self):
        """
        Return a string representation of the expression.

        Returns
        -------
        str
        """
        if len(self.operands) == 1:
            return f"{self.agg}({self.operands[0]})[{self._dtype}]"
        return f"{self.agg}({self.operands})[{self._dtype}]"


def build_row_idx_filter_expr(row_idx, row_col):
    """
    Build an expression to filter rows by rowid.

    Parameters
    ----------
    row_idx : int or list of int
        The row numeric indices to select.
    row_col : InputRefExpr
        The rowid column reference expression.

    Returns
    -------
    BaseExpr
        The resulting filtering expression.
    """
    if not is_list_like(row_idx):
        return row_col.eq(row_idx)

    if is_range_like(row_idx):
        start = row_idx[0]
        stop = row_idx[-1]
        step = row_idx.step
        if step < 0:
            start, stop = stop, start
            step = -step
        exprs = [row_col.ge(start), row_col.le(stop)]
        if step > 1:
            mod = OpExpr("MOD", [row_col, LiteralExpr(step)], get_dtype(int))
            exprs.append(mod.eq(0))
        return OpExpr("AND", exprs, get_dtype(bool))

    exprs = [row_col.eq(idx) for idx in row_idx]
    return OpExpr("OR", exprs, get_dtype(bool))


def build_if_then_else(cond, then_val, else_val, res_type):
    """
    Build a conditional operator expression.

    Parameters
    ----------
    cond : BaseExpr
        A condition to check.
    then_val : BaseExpr
        A value to use for passed condition.
    else_val : BaseExpr
        A value to use for failed condition.
    res_type : dtype
        The result data type.

    Returns
    -------
    BaseExpr
        The conditional operator expression.
    """
    if is_datetime64_dtype(res_type):
        if then_val._dtype != res_type:
            then_val = then_val.cast(res_type)
        if else_val._dtype != res_type:
            else_val = else_val.cast(res_type)
    return OpExpr("CASE", [cond, then_val, else_val], res_type)


def build_dt_expr(dt_operation, col_expr):
    """
    Build a datetime extraction expression.

    Parameters
    ----------
    dt_operation : str
        Datetime field to extract.
    col_expr : BaseExpr
        An expression to extract from.

    Returns
    -------
    BaseExpr
        The extract expression.
    """
    operation = LiteralExpr(dt_operation)

    res = OpExpr("PG_EXTRACT", [operation, col_expr], get_dtype("int32"))

    return res
