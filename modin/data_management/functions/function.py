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

from typing import Callable, Optional


class Function(object):
    """
    Interface for building functions that can execute in parallel across partitions.
    """

    def __init__(self):
        raise ValueError(
            "Please use {}.register instead of the constructor".format(
                type(self).__name__
            )
        )

    @classmethod
    def register(cls, func: Callable, *reg_args, **reg_kwargs):
        """
        Build function that apply source function across the entire dataset.

        Parameters
        ----------
        func: callable
            source function
        *reg_args: args,
            Args that will be used for building.
        **reg_kwargs: kwargs,
            Kwargs that will be used for building.

        Returns
        -------
        callable
        """
        raise NotImplementedError("Please implement in child class")

    @classmethod
    def validate_axis(cls, axis: Optional[int]) -> int:
        return 0 if axis is None else axis
