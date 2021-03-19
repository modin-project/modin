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


class MapFunction(Function):
    @classmethod
    def register(cls, func: Callable, *reg_args, **reg_kwargs):
        """
        Build Map function that perform across each partition.

        Parameters
        ----------
        func: callable
            source function
        *reg_args: args,
            Args that will be passed to the returned function.
        **reg_kwargs: kwargs,
            Kwargs that will be passed to the returned function.

        Returns
        -------
        callable
            map function
        """

        def map_function(query_compiler, *args, **kwargs):
            return query_compiler.__constructor__(
                query_compiler._modin_frame._map(
                    lambda x: func(x, *args, **kwargs), *reg_args, **reg_kwargs
                )
            )

        return map_function
