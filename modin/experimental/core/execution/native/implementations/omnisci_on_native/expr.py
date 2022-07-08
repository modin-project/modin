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
)
import numpy as np


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
    raise TypeError(f"Cannot perform operation on types: {lhs_dtype}, {rhs_dtype}")


_aggs_preserving_numeric_type = {"sum", "min", "max"}
_aggs_with_int_result = {"count", "size"}
_aggs_with_float_result = {"mean", "std", "skew"}


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
        raise NotImplementedError(f"unsupported aggreagte {agg}")


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
        "floordiv": "/",
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
        if not isinstance(other, BaseExpr):
            other = LiteralExpr(other)
        new_expr = OpExpr("=", [self, other], get_dtype(bool))
        return new_expr

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
        if not isinstance(other, BaseExpr):
            other = LiteralExpr(other)
        new_expr = OpExpr("<=", [self, other], get_dtype(bool))
        return new_expr

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
        # In OmniSci comparison with NULL always results in NULL,
        # but in pandas it is True for 'ne' comparison and False
        # for others.
        # Also pandas allows 'eq' and 'ne' comparison for values
        # of incompatible types which doesn't work in OmniSci.
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
        for op in getattr(self, "operands", []):
            op.collect_frames(frames)

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
        if hasattr(self, "operands"):
            res = self.copy()
            for i in range(0, len(self.operands)):
                res.operands[i] = res.operands[i].translate_input(mapper)
            return res
        return self._translate_input(mapper)

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


class InputRefExpr(BaseExpr):
    """
    An expression tree node to represent an input frame column.

    Parameters
    ----------
    frame : OmnisciOnNativeDataframe
        An input frame.
    col : str
        An input column name.
    dtype : dtype
        Input column data type.

    Attributes
    ----------
    modin_frame : OmnisciOnNativeDataframe
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
    val : int, np.int, float, bool, str or None
        Literal value.

    Attributes
    ----------
    val : int, np.int, float, bool, str or None
        Literal value.
    _dtype : dtype
        Literal data type.
    """

    def __init__(self, val):
        assert val is None or isinstance(
            val, (int, float, bool, str, np.int8, np.int16, np.int32, np.int64)
        ), f"unsupported literal value {val} of type {type(val)}"
        self.val = val
        if val is None:
            self._dtype = get_dtype(float)
        else:
            self._dtype = get_dtype(type(val))

    def copy(self):
        """
        Make a shallow copy of the expression.

        Returns
        -------
        LiteralExpr
        """
        return LiteralExpr(self.val)

    def __repr__(self):
        """
        Return a string representation of the expression.

        Returns
        -------
        str
        """
        return f"{self.val}[{self._dtype}]"


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
    """

    def __init__(self, op, operands, dtype):
        self.op = op
        self.operands = operands
        self._dtype = dtype

    def copy(self):
        """
        Make a shallow copy of the expression.

        Returns
        -------
        OpExpr
        """
        return OpExpr(self.op, self.operands.copy(), self._dtype)

    def __repr__(self):
        """
        Return a string representation of the expression.

        Returns
        -------
        str
        """
        if len(self.operands) == 1:
            return f"({self.op} {self.operands[0]} [{self._dtype}])"
        return f"({self.op} {self.operands} [{self._dtype}])"


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

    exprs = []
    for idx in row_idx:
        exprs.append(row_col.eq(idx))

    res = OpExpr("OR", exprs, get_dtype(bool))

    return res


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

    res = OpExpr("PG_EXTRACT", [operation, col_expr], get_dtype(int))

    return res
