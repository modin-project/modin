import abc
from pandas.core.dtypes.common import (
    is_list_like,
    _get_dtype,
    is_float_dtype,
    is_integer_dtype,
)
import numpy as np


def _get_common_dtype(lhs_dtype, rhs_dtype):
    if is_float_dtype(lhs_dtype) or is_float_dtype(rhs_dtype):
        return _get_dtype(float)
    assert is_integer_dtype(lhs_dtype) and is_integer_dtype(rhs_dtype)
    return _get_dtype(int)


_aggs_preserving_numeric_type = {"sum", "min", "max"}
_aggs_with_int_result = {"count", "size"}
_aggs_with_float_result = {"mean"}


def _agg_dtype(agg, dtype):
    if agg in _aggs_preserving_numeric_type:
        return dtype
    elif agg in _aggs_with_int_result:
        return _get_dtype(int)
    elif agg in _aggs_with_float_result:
        return _get_dtype(float)
    else:
        raise NotImplementedError(f"unsupported aggreagte {agg}")


class BaseExpr(abc.ABC):
    binary_operations = {"add": "+", "sub": "-", "mul": "*"}

    preserve_dtype_math_ops = {"add", "sub", "mul"}
    promote_to_float_math_ops = {"div"}
    cmp_ops = {}

    def eq(self, other):
        if not isinstance(other, BaseExpr):
            other = LiteralExpr(other)
        new_expr = OpExpr("=", [self, other], _get_dtype(bool))
        return new_expr

    def cast(self, res_type):
        new_expr = OpExpr("CAST", [self], res_type)
        return new_expr

    def is_null(self):
        new_expr = OpExpr("IS NULL", [self], _get_dtype(bool))
        return new_expr

    def bin_op(self, other, op_name):
        if op_name not in self.binary_operations:
            raise NotImplementedError(f"unsupported binary operation {op_name}")
        res_type = self._get_bin_op_res_type(op_name, self._dtype, other._dtype)
        new_expr = OpExpr(self.binary_operations[op_name], [self, other], res_type)
        return new_expr

    def _get_bin_op_res_type(self, op_name, lhs_dtype, rhs_dtype):
        if op_name in self.preserve_dtype_math_ops:
            return _get_common_dtype(lhs_dtype, rhs_dtype)
        elif op_name in self.promote_to_float_math_ops:
            return _get_dtype(float)
        elif op_name in self.cmp_ops:
            return _get_dtype(bool)
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
            val, (int, float, bool, str)
        ), f"unsupported literal value {val}"
        self.val = val
        if val is None:
            self._dtype = _get_dtype(float)
        else:
            self._dtype = _get_dtype(type(val))

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
            return f"({self.op} {self.operands[0]})"
        return f"({self.op} {self.operands} [{self._dtype}])"


class AggregateExpr(BaseExpr):
    def __init__(self, agg, operands, dtype, distinct):
        self.agg = agg.upper()
        self.operands = operands
        if not isinstance(self.operands, list):
            self.operands = [self.operands]
        self._dtype = dtype
        self.distinct = distinct

    def copy(self):
        return OpExpr(self.agg, self.operands.copy(), self._dtype, self.distinct)

    def __repr__(self):
        if len(self.operands) == 1:
            return f"{self.agg} {self.operands[0]} [{self._dtype}]"
        return f"{self.agg} {self.operands} [{self._dtype}]"


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

    res = OpExpr("OR", exprs, _get_dtype(bool))

    return res


def build_if_then_else(cond, then_val, else_val, res_type):
    return OpExpr("CASE", [cond, then_val, else_val], res_type)


def build_dt_expr(dt_operation, col_expr):
    operation = LiteralExpr(dt_operation)

    res = OpExpr("PG_EXTRACT", [operation, col_expr], _get_dtype(int))

    return res
