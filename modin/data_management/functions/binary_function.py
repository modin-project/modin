import numpy as np
import pandas

from .function import Function


class BinaryFunction(Function):
    @classmethod
    def call(cls, func, *call_args, **call_kwds):
        def caller(query_compiler, other, *args, **kwargs):
            axis = kwargs.get("axis", 0)
            if isinstance(other, type(query_compiler)):
                return query_compiler.__constructor__(
                    query_compiler._modin_frame._binary_op(
                        lambda x, y: func(x, y, *args, **kwargs), other._modin_frame,
                    )
                )
            else:
                if isinstance(other, (list, np.ndarray, pandas.Series)):
                    if axis == 1 and isinstance(other, pandas.Series):
                        new_columns = query_compiler.columns.join(
                            other.index, how="outer"
                        )
                    else:
                        new_columns = query_compiler.columns
                    new_modin_frame = query_compiler._modin_frame._apply_full_axis(
                        axis,
                        lambda df: func(df, other, *args, **kwargs),
                        new_index=query_compiler.index,
                        new_columns=new_columns,
                    )
                else:
                    new_modin_frame = query_compiler._modin_frame._map(
                        lambda df: func(df, other, *args, **kwargs)
                    )
                return query_compiler.__constructor__(new_modin_frame)

        return caller
