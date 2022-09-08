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

"""Module houses builder class for Binary operator."""

import numpy as np
import pandas

from .operator import Operator


class Binary(Operator):
    """Builder class for Binary operator."""

    @classmethod
    def register(cls, func, join_type="outer", labels="replace"):
        """
        Build template binary operator.

        Parameters
        ----------
        func : callable(pandas.DataFrame, [pandas.DataFrame, list-like, scalar]) -> pandas.DataFrame
            Binary function to execute. Have to be able to accept at least two arguments.
        join_type : {'left', 'right', 'outer', 'inner', None}, default: 'outer'
            Type of join that will be used if indices of operands are not aligned.
        labels : {"keep", "replace", "drop"}, default: "replace"
            Whether keep labels from left Modin DataFrame, replace them with labels
            from joined DataFrame or drop altogether to make them be computed lazily later.

        Returns
        -------
        callable
            Function that takes query compiler and executes binary operation.
        """

        def caller(
            query_compiler, other, broadcast=False, *args, dtypes=None, **kwargs
        ):
            """
            Apply binary `func` to passed operands.

            Parameters
            ----------
            query_compiler : QueryCompiler
                Left operand of `func`.
            other : QueryCompiler, list-like object or scalar
                Right operand of `func`.
            broadcast : bool, default: False
                If `other` is a one-column query compiler, indicates whether it is a Series or not.
                Frames and Series have to be processed differently, however we can't distinguish them
                at the query compiler level, so this parameter is a hint that passed from a high level API.
            *args : args,
                Arguments that will be passed to `func`.
            dtypes : "copy" or None, default: None
                Whether to keep old dtypes or infer new dtypes from data.
            **kwargs : kwargs,
                Arguments that will be passed to `func`.

            Returns
            -------
            QueryCompiler
                Result of binary function.
            """
            axis = kwargs.get("axis", 0)
            if isinstance(other, type(query_compiler)):
                if broadcast:
                    assert (
                        len(other.columns) == 1
                    ), "Invalid broadcast argument for `broadcast_apply`, too many columns: {}".format(
                        len(other.columns)
                    )
                    # Transpose on `axis=1` because we always represent an individual
                    # column or row as a single-column Modin DataFrame
                    if axis == 1:
                        other = other.transpose()
                    return query_compiler.__constructor__(
                        query_compiler._modin_frame.broadcast_apply(
                            axis,
                            lambda l, r: func(l, r.squeeze(), *args, **kwargs),
                            other._modin_frame,
                            join_type=join_type,
                            labels=labels,
                            dtypes=dtypes,
                        )
                    )
                else:
                    return query_compiler.__constructor__(
                        query_compiler._modin_frame.n_ary_op(
                            lambda x, y: func(x, y, *args, **kwargs),
                            [other._modin_frame],
                            join_type=join_type,
                        )
                    )
            else:
                if isinstance(other, (list, np.ndarray, pandas.Series)):
                    new_modin_frame = query_compiler._modin_frame.apply_full_axis(
                        axis,
                        lambda df: func(df, other, *args, **kwargs),
                        new_index=query_compiler.index,
                        new_columns=query_compiler.columns,
                        dtypes=dtypes,
                    )
                else:
                    new_modin_frame = query_compiler._modin_frame.map(
                        lambda df: func(df, other, *args, **kwargs),
                        dtypes=dtypes,
                    )
                return query_compiler.__constructor__(new_modin_frame)

        return caller
