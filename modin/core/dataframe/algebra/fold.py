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

"""Module houses builder class for Fold operator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from .operator import Operator

if TYPE_CHECKING:
    import pandas

    from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler


class Fold(Operator):
    """Builder class for Fold functions."""

    @classmethod
    def register(
        cls, fold_function: Callable[..., pandas.DataFrame], shape_preserved=False
    ) -> Callable[..., PandasQueryCompiler]:
        """
        Build Fold operator that will be performed across rows/columns.

        Parameters
        ----------
        fold_function : callable(pandas.DataFrame, *args, **kwargs) -> pandas.DataFrame
            Function to apply across rows/columns.
        shape_preserved : bool, default: False
            Whether the shape of the dataframe is preserved or not
            after applying a function.

        Returns
        -------
        callable
            Function that takes query compiler and executes Fold function.
        """

        def caller(
            query_compiler: PandasQueryCompiler,
            fold_axis: Optional[int] = None,
            *args: tuple,
            new_index=None,
            new_columns=None,
            **kwargs: dict,
        ) -> PandasQueryCompiler:
            """
            Execute Fold function against passed query compiler.

            Parameters
            ----------
            query_compiler : PandasQueryCompiler
                The query compiler to execute the function on.
            fold_axis : int, optional
                0 or None means apply across full column partitions. 1 means
                apply across full row partitions.
            *args : tuple
                Additional arguments passed to `fold_function`.
            new_index : list-like, optional
                The index of the result.
            new_columns : list-like, optional
                The columns of the result.
            **kwargs: dict
                Additional keyword arguments passed to `fold_function`.

            Returns
            -------
            PandasQueryCompiler
                A new query compiler representing the result of executing the
                function.
            """
            return query_compiler.__constructor__(
                query_compiler._modin_frame.fold(
                    cls.validate_axis(fold_axis),
                    lambda x: fold_function(x, *args, **kwargs),
                    new_index=new_index,
                    new_columns=new_columns,
                    shape_preserved=shape_preserved,
                )
            )

        return caller
