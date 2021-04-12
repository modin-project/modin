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

from typing import Callable
from .function import Function


class MapReduceFunction(Function):
    """Builder class for MapReduce functions."""

    @classmethod
    def register(
        cls,
        map_func: Callable,
        reduce_func: Callable = None,
        axis=None,
    ):
        """
        Build MapReduce function.

        Parameters
        ----------
        map_func : callable
            Source map function.
        reduce_func : callable
            Source reduce function.
        axis : int, optional
            Specifies axis to apply function along.

        Returns
        -------
        callable
            Function that takes query compiler and executes passed functions
            with MapReduce algorithm.
        """

        if reduce_func is None:
            reduce_func = map_func

        def map_reduce(query_compiler, *args, **kwargs):
            _axis = axis if axis is not None else kwargs.get("axis")
            return query_compiler.__constructor__(
                query_compiler._modin_frame._map_reduce(
                    cls.validate_axis(_axis),
                    lambda x: map_func(x, *args, **kwargs),
                    lambda y: reduce_func(y, *args, **kwargs),
                )
            )

        return map_reduce
