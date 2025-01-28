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

"""Module houses builder class for Reduce operator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from .operator import Operator

if TYPE_CHECKING:
    import pandas

    from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler


class Reduce(Operator):
    """Builder class for Reduce operator."""

    @classmethod
    def register(
        cls,
        reduce_function: Callable[..., pandas.Series],
        axis: Optional[int] = None,
        shape_hint: Optional[str] = None,
    ) -> Callable[..., PandasQueryCompiler]:
        """
        Build Reduce operator that will be performed across rows/columns.

        It's used if `func` reduces the dimension of partitions in contrast to `Fold`.

        Parameters
        ----------
        reduce_function : callable(pandas.DataFrame, *args, **kwargs) -> pandas.Series
            Source function.
        axis : int, optional
            Axis to apply function along.
        shape_hint : {"row", "column", None}, default: None
            Shape hint for the results known to be a column or a row, otherwise None.

        Returns
        -------
        callable
            Function that takes query compiler and executes Reduce function.
        """

        def caller(
            query_compiler: PandasQueryCompiler, *args: tuple, **kwargs: dict
        ) -> PandasQueryCompiler:
            """Execute Reduce function against passed query compiler."""
            _axis = kwargs.get("axis") if axis is None else axis
            return query_compiler.__constructor__(
                query_compiler._modin_frame.reduce(
                    cls.validate_axis(_axis),
                    lambda x: reduce_function(x, *args, **kwargs),
                ),
                shape_hint=shape_hint,
            )

        return caller
