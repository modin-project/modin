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

"""Module houses MapReduce functions builder class."""

from .function import Function


class MapReduceFunction(Function):
    """Builder class for MapReduce functions."""

    @classmethod
    # FIXME: spread `**call_kwds` into an actual function arguments.
    def call(cls, map_function, reduce_function, **call_kwds):  # noqa: PR02
        """
        Build MapReduce function.

        Parameters
        ----------
        map_function : callable(pandas.DataFrame) -> pandas.DataFrame
            Source map function.
        reduce_function : callable(pandas.DataFrame) -> scalar
            Source reduce function.
        axis : int, optional
            Specifies axis to apply function along.
            (If specified, have to be passed via `kwargs`).
        **call_kwds : kwargs
            Additional parameters in the glory of compatibility. Does not affect the result.

        Returns
        -------
        callable
            Function that takes query compiler and executes passed functions
            with MapReduce algorithm.
        """

        def caller(query_compiler, *args, **kwargs):
            axis = call_kwds.get("axis", kwargs.get("axis"))
            return query_compiler.__constructor__(
                query_compiler._modin_frame._map_reduce(
                    cls.validate_axis(axis),
                    lambda x: map_function(x, *args, **kwargs),
                    lambda y: reduce_function(y, *args, **kwargs),
                )
            )

        return caller

    @classmethod
    # FIXME: `register` is an alias for `call` method. One of them should be removed.
    def register(cls, map_function, reduce_function=None, **kwargs):  # noqa: PR02
        """
        Build MapReduce function.

        Parameters
        ----------
        map_function : callable
            Source map function.
        reduce_function : callable, optional
            Source reduce function. If not specified `map_function` will be used.
        axis : int, optional
            Specifies axis to apply function along.
            (If specified, have to be passed via `kwargs`).
        **kwargs : kwargs
            Additional parameters in the glory of compatibility. Does not affect the result.

        Returns
        -------
        callable
            Function that takes query compiler and executes passed functions
            with MapReduce algorithm.
        """
        if reduce_function is None:
            reduce_function = map_function
        return cls.call(map_function, reduce_function, **kwargs)
