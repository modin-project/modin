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

from .operator import Operator


class Fold(Operator):
    """Builder class for Fold functions."""

    @classmethod
    def register(cls, fold_function):
        """
        Build Fold operator that will be performed across rows/columns.

        Parameters
        ----------
        fold_function : callable(pandas.DataFrame) -> pandas.DataFrame
            Function to apply across rows/columns.

        Returns
        -------
        callable
            Function that takes query compiler and executes Fold function.
        """

        def caller(query_compiler, fold_axis=None, *args, **kwargs):
            """
            Execute Fold function against passed query compiler.

            Parameters
            ----------
            query_compiler : BaseQueryCompiler
                The query compiler to execute the function on.
            fold_axis : int, optional
                0 or None means apply across full column partitions. 1 means
                apply across full row partitions.
            *args : iterable
                Additional arguments passed to fold_function.
            **kwargs: dict
                Additional keyword arguments passed to fold_function.

            Returns
            -------
            BaseQueryCompiler
                A new query compiler representing the result of executing the
                function.
            """
            return query_compiler.__constructor__(
                query_compiler._modin_frame.fold(
                    cls.validate_axis(fold_axis),
                    lambda x: fold_function(x, *args, **kwargs),
                )
            )

        return caller

    @classmethod
    def apply(cls, df, func, fold_axis=0, func_args=None, func_kwargs=None):
        """
        Apply a Fold (full-axis) function to the dataframe.

        Parameters
        ----------
        df : modin.pandas.DataFrame or modin.pandas.Series
            DataFrame object to apply the operator against.
        func : callable(pandas.DataFrame[NxM], *args, **kwargs) -> pandas.DataFrame[NxM]
            A function to apply to every partition. Note that the function shouldn't change
            the shape of the dataframe.
        fold_axis : int, default: 0
            Whether to apply the function across rows (``axis=0``) or across columns (``axis=1``).
        func_args : tuple, optional
            Positional arguments to pass to the funcs.
        func_kwargs : dict, optional
            Keyword arguments to pass to the funcs.

        Returns
        -------
        the same type as `df`
        """
        func_args = tuple() if func_args is None else func_args
        func_kwargs = dict() if func_kwargs is None else func_kwargs
        func_args = (fold_axis,) + func_args

        return super().apply(df, func, func_args, func_kwargs)
