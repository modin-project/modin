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

    @abc.abstractmethod
    def copy(self):
        pass


class InputRefExpr(BaseExpr):
    def __init__(self, frame, col):
        self.modin_frame = frame
        self.column = col

    def copy(self):
        return InputRefExpr(self.modin_frame, self.column)

    def __repr__(self):
        return f"{self.modin_frame.id_str()}.{self.column}"


class LiteralExpr(BaseExpr):
    def __init__(self, val):
        self.val = val

    def copy(self):
        return LiteralExpr(self.val)

    def __repr__(self):
        return str(self.val)


class OpExprType(BaseExpr):
    def __init__(self, expr_type, nullable=False):
        self.type = expr_type
        self.nullable = nullable

    def copy(self):
        return OpExprType(self.type, self.nullable)


class OpExpr(BaseExpr):
    def __init__(self, op, operands, res_type):
        self.op = op
        self.operands = operands
        self.type = res_type

    def copy(self):
        return OpExpr(self.op, self.operands.copy(), self.type)

    def __repr__(self):
        if len(self.operands) == 1:
            return f"({self.op} {self.operands[0]})"
        return f"({self.op} {self.operands})"


class AggregateExpr(BaseExpr):
    def __init__(self, agg, operands, res_type, distinct):
        self.agg = agg.upper()
        self.operands = operands
        if not isinstance(self.operands, list):
            self.operands = [self.operands]
        self.type = res_type
        self.distinct = distinct

    def copy(self):
        return OpExpr(self.agg, self.operands.copy(), self.type, self.distinct)

    def __repr__(self):
        if len(self.operands) == 1:
            return "{self.agg} {self.operands[0]}"
        return "{self.agg} {self.operands}"


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
