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
from pandas.core.dtypes.common import (
    is_list_like,
    get_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_string_like_dtype,
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_bool_dtype,
)
import numpy as np


def _get_common_dtype(lhs_dtype, rhs_dtype):
    if lhs_dtype == rhs_dtype:
        return lhs_dtype
    if is_float_dtype(lhs_dtype) or is_float_dtype(rhs_dtype):
        return get_dtype(float)
    assert is_integer_dtype(lhs_dtype) and is_integer_dtype(rhs_dtype)
    return get_dtype(int)


_aggs_preserving_numeric_type = {"sum", "min", "max"}
_aggs_with_int_result = {"count", "size"}
_aggs_with_float_result = {"mean", "std", "skew"}


def _agg_dtype(agg, dtype):
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
    return op in _cmp_ops


class BaseExpr(abc.ABC):
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
    }

    preserve_dtype_math_ops = {"add", "sub", "mul", "mod", "floordiv", "pow"}
    promote_to_float_math_ops = {"truediv"}

    def eq(self, other):
        if not isinstance(other, BaseExpr):
            other = LiteralExpr(other)
        new_expr = OpExpr("=", [self, other], get_dtype(bool))
        return new_expr

    def le(self, other):
        if not isinstance(other, BaseExpr):
            other = LiteralExpr(other)
        new_expr = OpExpr("<=", [self, other], get_dtype(bool))
        return new_expr

    def cast(self, res_type):
        new_expr = OpExpr("CAST", [self], res_type)
        return new_expr

    def is_null(self):
        new_expr = OpExpr("IS NULL", [self], get_dtype(bool))
        return new_expr

    def is_not_null(self):
        new_expr = OpExpr("IS NOT NULL", [self], get_dtype(bool))
        return new_expr

    def bin_op(self, other, op_name):
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
        return self.bin_op(other, "add")

    def sub(self, other):
        return self.bin_op(other, "sub")

    def mul(self, other):
        return self.bin_op(other, "mul")

    def mod(self, other):
        return self.bin_op(other, "mod")

    def truediv(self, other):
        return self.bin_op(other, "truediv")

    def floordiv(self, other):
        return self.bin_op(other, "floordiv")

    def pow(self, other):
        return self.bin_op(other, "pow")

    def floor(self):
        return OpExpr("FLOOR", [self], get_dtype(int))

    def _cmp_op(self, other, op_name):
        lhs_dtype_class = self._get_dtype_cmp_class(self._dtype)
        rhs_dtype_class = self._get_dtype_cmp_class(other._dtype)
        res_dtype = get_dtype(bool)
        # In OmniSci comparison with NULL always results in NULL,
        # but in Pandas it is True for 'ne' comparison and False
        # for others.
        # Also Pandas allow 'eq' and 'ne' comparison for values
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
        if is_numeric_dtype(dtype) or is_bool_dtype(dtype):
            return "numeric"
        if is_string_like_dtype(dtype) or is_categorical_dtype(dtype):
            return "string"
        if is_datetime64_any_dtype(dtype):
            return "datetime"
        return "other"

    def _get_bin_op_res_type(self, op_name, lhs_dtype, rhs_dtype):
        if op_name in self.preserve_dtype_math_ops:
            return _get_common_dtype(lhs_dtype, rhs_dtype)
        elif op_name in self.promote_to_float_math_ops:
            return get_dtype(float)
        elif is_cmp_op(op_name):
            return get_dtype(bool)
        else:
            raise NotImplementedError(f"unsupported binary operation {op_name}")

    @abc.abstractmethod
    def copy(self):
        pass

    def collect_frames(self, frames):
        """Return the first modin frame referenced in expression."""
        for op in getattr(self, "operands", []):
            op.collect_frames(frames)

    # currently we translate only exprs with a single input frame
    def translate_input(self, mapper):
        if hasattr(self, "operands"):
            res = self.copy()
            for i in range(0, len(self.operands)):
                res.operands[i] = res.operands[i].translate_input(mapper)
            return res
        return self._translate_input(mapper)

    def _translate_input(self, mapper):
        return self


class InputRefExpr(BaseExpr):
    def __init__(self, frame, col, dtype):
        self.modin_frame = frame
        self.column = col
        self._dtype = dtype

    def copy(self):
        return InputRefExpr(self.modin_frame, self.column, self._dtype)

    def collect_frames(self, frames):
        frames.add(self.modin_frame)

    def _translate_input(self, mapper):
        return mapper.translate(self)

    def __repr__(self):
        return f"{self.modin_frame.id_str()}.{self.column}[{self._dtype}]"


class LiteralExpr(BaseExpr):
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
        return LiteralExpr(self.val)

    def __repr__(self):
        return f"{self.val}[{self._dtype}]"


class OpExpr(BaseExpr):
    def __init__(self, op, operands, dtype):
        self.op = op
        self.operands = operands
        self._dtype = dtype

    def copy(self):
        return OpExpr(self.op, self.operands.copy(), self._dtype)

    def __repr__(self):
        if len(self.operands) == 1:
            return f"({self.op} {self.operands[0]} [{self._dtype}])"
        return f"({self.op} {self.operands} [{self._dtype}])"


class AggregateExpr(BaseExpr):
    def __init__(self, agg, op, distinct=False, dtype=None):
        self.agg = agg
        self.operands = [op]
        self._dtype = dtype if dtype else _agg_dtype(agg, op._dtype if op else None)
        assert self._dtype is not None
        self.distinct = distinct

    def copy(self):
        return AggregateExpr(self.agg, self.operands[0], self.distinct, self._dtype)

    def __repr__(self):
        if len(self.operands) == 1:
            return f"{self.agg}({self.operands[0]})[{self._dtype}]"
        return f"{self.agg}({self.operands})[{self._dtype}]"


def build_row_idx_filter_expr(row_idx, row_col):
    """Build calcite expression to filter rows by rowid.

    Parameters
    ----------
    row_idx
        The row numeric indices to select
    row_col
        InputRefExpr referencing proper rowid column to filter by
    Returns
    -------
    CalciteBaseExpr
        A BaseExpr implementing filter condition
    """
    if not is_list_like(row_idx):
        return row_col.eq(row_idx)

    exprs = []
    for idx in row_idx:
        exprs.append(row_col.eq(idx))

    res = OpExpr("OR", exprs, get_dtype(bool))

    return res


def build_if_then_else(cond, then_val, else_val, res_type):
    return OpExpr("CASE", [cond, then_val, else_val], res_type)


def build_dt_expr(dt_operation, col_expr):
    operation = LiteralExpr(dt_operation)

    res = OpExpr("PG_EXTRACT", [operation, col_expr], get_dtype(int))

    return res
