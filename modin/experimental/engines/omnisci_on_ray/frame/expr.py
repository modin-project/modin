import abc
from pandas.core.dtypes.common import is_list_like


class BaseExpr(abc.ABC):
    def eq(self, other):
        if not isinstance(other, BaseExpr):
            other = LiteralExpr(other)
        res_type = OpExprType(bool, False)
        new_expr = OpExpr("=", [self, other], res_type)
        return new_expr

    def is_null(self):
        res_type = OpExprType(bool, False)
        new_expr = OpExpr("IS NULL", [self], res_type)
        return new_expr


class InputRefExpr(BaseExpr):
    def __init__(self, input_idx):
        self.input = input_idx

    def __repr__(self):
        return "(INPUT {})".format(self.input)


class LiteralExpr(BaseExpr):
    def __init__(self, val):
        self.val = val

    def __repr__(self):
        return str(self.val)


class OpExprType(BaseExpr):
    def __init__(self, expr_type, nullable=False):
        self.type = expr_type
        self.nullable = nullable


class OpExpr(BaseExpr):
    def __init__(self, op, operands, res_type):
        self.op = op
        self.operands = operands
        self.type = res_type

    def __repr__(self):
        if len(self.operands) == 1:
            return "({} {})".format(self.op.upper(), self.operands[0])
        return "({} {})".format(self.op.upper(), self.operands)


class AggregateExpr(BaseExpr):
    def __init__(self, agg, operands, res_type, distinct):
        self.agg = agg.upper()
        self.operands = operands
        if not isinstance(self.operands, list):
            self.operands = [self.operands]
        self.type = res_type
        self.distinct = distinct


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

    res_type = OpExprType(bool, False)
    res = OpExpr("OR", exprs, res_type)

    return res


def build_if_then_else(cond, then_val, else_val, res_type):
    return OpExpr("CASE", [cond, then_val, else_val], res_type)
