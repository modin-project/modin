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

"""Module houses builder class for Map operator."""

from .operator import Operator


class Map(Operator):
    """Builder class for Map operator."""

    @classmethod
    def call(cls, function, *call_args, **call_kwds):
        """
        Build Map operator that will be performed across each partition.

        Parameters
        ----------
        function : callable(pandas.DataFrame) -> pandas.DataFrame
            Function that will be applied to the each partition.
            Function takes `pandas.DataFrame` and returns `pandas.DataFrame`
            of the same shape.
        *call_args : args
            Args that will be passed to the returned function.
        **call_kwds : kwargs
            Kwargs that will be passed to the returned function.

        Returns
        -------
        callable
            Function that takes query compiler and executes map function.
        """

        def caller(query_compiler, *args, **kwargs):
            """Execute Map function against passed query compiler."""
            return query_compiler.__constructor__(
                query_compiler._modin_frame.map(
                    lambda x: function(x, *args, **kwargs), *call_args, **call_kwds
                )
            )

        return caller
