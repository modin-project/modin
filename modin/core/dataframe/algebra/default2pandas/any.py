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

# FIXME: This whole module is duplicating the logic of `default.py` and should be removed.

"""Module houses default functions builder class."""

from .default import DefaultMethod


class ObjTypeDeterminer:
    """
    Class that routes work to the frame.

    Provides an instance which forwards all of the `__getattribute__` calls
    to an object under which `key` function is applied.
    """

    def __getattr__(self, key):
        """
        Build function that executes `key` function over passed frame.

        Parameters
        ----------
        key : str

        Returns
        -------
        callable
            Function that takes DataFrame and executes `key` function on it.
        """

        def func(df, *args, **kwargs):
            """Access specified attribute of the passed object and call it if it's callable."""
            prop = getattr(df, key)
            if callable(prop):
                return prop(*args, **kwargs)
            else:
                return prop

        return func


class AnyDefault(DefaultMethod):
    """Builder for default-to-pandas methods which can be executed under any type of object."""

    @classmethod
    def register(cls, func, obj_type=None, **kwargs):
        """
        Build function that do fallback to default pandas implementation for passed `func`.

        Parameters
        ----------
        func : callable or str,
            Function to apply to the casted to pandas frame.
        obj_type : object, optional
            If `func` is a string with a function name then `obj_type` provides an
            object to search function in. If not specified `ObjTypeDeterminer` will be used.
        **kwargs : kwargs
            Additional parameters that will be used for building.

        Returns
        -------
        callable
            Function that takes query compiler, does fallback to pandas and applies `func`
            to the casted to pandas frame.
        """
        if obj_type is None:
            obj_type = ObjTypeDeterminer()

        return super().register(func, obj_type=obj_type, **kwargs)
