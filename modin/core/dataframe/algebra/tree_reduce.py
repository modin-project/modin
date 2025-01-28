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

"""Module houses builder class for TreeReduce operator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from .operator import Operator

if TYPE_CHECKING:
    import pandas
    from pandas._typing import DtypeObj

    from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler


class TreeReduce(Operator):
    """Builder class for TreeReduce operator."""

    @classmethod
    def register(
        cls,
        map_function: Optional[Callable[..., pandas.DataFrame]],
        reduce_function: Optional[Callable[..., pandas.Series]] = None,
        axis: Optional[int] = None,
        compute_dtypes: Optional[Callable[..., DtypeObj]] = None,
    ) -> Callable[..., PandasQueryCompiler]:
        """
        Build TreeReduce operator.

        Parameters
        ----------
        map_function : callable(pandas.DataFrame, *args, **kwargs) -> pandas.DataFrame
            Source map function.
        reduce_function : callable(pandas.DataFrame, *args, **kwargs) -> pandas.Series, optional
            Source reduce function.
        axis : int, optional
            Specifies axis to apply function along.
        compute_dtypes : callable(pandas.Series, *func_args, **func_kwargs) -> DtypeObj, optional
            Callable for computing dtypes.

        Returns
        -------
        callable
            Function that takes query compiler and executes passed functions
            with TreeReduce algorithm.
        """
        if reduce_function is None:
            reduce_function = map_function

        def caller(
            query_compiler: PandasQueryCompiler, *args: tuple, **kwargs: dict
        ) -> PandasQueryCompiler:
            """Execute TreeReduce function against passed query compiler."""
            _axis = kwargs.get("axis") if axis is None else axis

            new_dtypes = None
            if compute_dtypes and query_compiler.frame_has_materialized_dtypes:
                new_dtypes = str(compute_dtypes(query_compiler.dtypes, *args, **kwargs))

            return query_compiler.__constructor__(
                query_compiler._modin_frame.tree_reduce(
                    cls.validate_axis(_axis),
                    lambda x: map_function(x, *args, **kwargs),
                    lambda y: reduce_function(y, *args, **kwargs),
                    dtypes=new_dtypes,
                )
            )

        return caller
