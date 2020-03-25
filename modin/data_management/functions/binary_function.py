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
