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

from .operator import Operator


class TreeReduce(Operator):
    """Builder class for TreeReduce operator."""

    @classmethod
    def register(cls, map_function, reduce_function=None, axis=None):
        """
        Build TreeReduce operator.

        Parameters
        ----------
        map_function : callable(pandas.DataFrame) -> pandas.DataFrame
            Source map function.
        reduce_function : callable(pandas.DataFrame) -> pandas.Series, optional
            Source reduce function.
        axis : int, optional
            Specifies axis to apply function along.

        Returns
        -------
        callable
            Function that takes query compiler and executes passed functions
            with TreeReduce algorithm.
        """
        if reduce_function is None:
            reduce_function = map_function

        def caller(query_compiler, *args, **kwargs):
            """Execute TreeReduce function against passed query compiler."""
            _axis = kwargs.get("axis") if axis is None else axis
            return query_compiler.__constructor__(
                query_compiler._modin_frame.tree_reduce(
                    cls.validate_axis(_axis),
                    lambda x: map_function(x, *args, **kwargs),
                    lambda y: reduce_function(y, *args, **kwargs),
                )
            )

        return caller

    @classmethod
    def apply(
        cls, df, map_function, reduce_function, axis=0, func_args=None, func_kwargs=None
    ):
        r"""
        Apply a map-reduce function to the dataframe.

        Parameters
        ----------
        df : modin.pandas.DataFrame or modin.pandas.Series
            DataFrame object to apply the operator against.
        map_function : callable(pandas.DataFrame, \*args, \*\*kwargs) -> pandas.DataFrame
            A map function to apply to every partition.
        reduce_function : callable(pandas.DataFrame, \*args, \*\*kwargs) -> Union[pandas.Series, pandas.DataFrame[1xN]]
            A reduction function to apply to the results of the map functions.
        axis : int, default: 0
            Whether to apply the reduce function across rows (``axis=0``) or across columns (``axis=1``).
        func_args : tuple, optional
            Positional arguments to pass to the funcs.
        func_kwargs : dict, optional
            Keyword arguments to pass to the funcs.

        Returns
        -------
        modin.pandas.Series
        """
        from modin.pandas import Series

        return super().apply(
            df,
            map_function,
            func_args,
            func_kwargs,
            reduce_function=reduce_function,
            axis=axis,
            _return_type=Series,
        )
