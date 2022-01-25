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
    def call(cls, map_function, reduce_function, axis=None):
        """
        Build TreeReduce operator.

        Parameters
        ----------
        map_function : callable(pandas.DataFrame) -> pandas.DataFrame
            Source map function.
        reduce_function : callable(pandas.DataFrame) -> pandas.Series
            Source reduce function.
        axis : int, optional
            Specifies axis to apply function along.

        Returns
        -------
        callable
            Function that takes query compiler and executes passed functions
            with TreeReduce algorithm.
        """

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
    # FIXME: `register` is an alias for `call` method. One of them should be removed.
    def register(cls, map_function, reduce_function=None, **kwargs):
        """
        Build TreeReduce function.

        Parameters
        ----------
        map_function : callable(pandas.DataFrame) -> [pandas.DataFrame, pandas.Series]
            Source map function.
        reduce_function : callable(pandas.DataFrame) -> pandas.Series, optional
            Source reduce function. If not specified `map_function` will be used.
        **kwargs : dict
            Additional parameters to pass to the builder function.

        Returns
        -------
        callable
            Function that takes query compiler and executes passed functions
            with TreeReduce algorithm.
        """
        if reduce_function is None:
            reduce_function = map_function
        return cls.call(map_function, reduce_function, **kwargs)
