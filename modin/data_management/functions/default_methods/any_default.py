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

from .default import DefaultMethod
from modin.utils import _inherit_docstrings


class ObjTypeDeterminer:
    """
    This class provides an instances which forwards all of the `__getattribute__` calls
    to an object under which `key` function is going to be applied.
    """

    def __getattr__(self, key):
        def func(df, *args, **kwargs):
            prop = getattr(df, key)
            if callable(prop):
                return prop(*args, **kwargs)
            else:
                return prop

        return func


@_inherit_docstrings(DefaultMethod, exclude=[DefaultMethod])
class AnyDefault(DefaultMethod):
    """Build default-to-pandas methods which can be executed under any type of object"""

    @classmethod
    def register(cls, func, obj_type=None, inplace=False, fn_name=None):
        if obj_type is None:
            obj_type = ObjTypeDeterminer()

        return super().register(
            func, obj_type=obj_type, inplace=inplace, fn_name=fn_name
        )
