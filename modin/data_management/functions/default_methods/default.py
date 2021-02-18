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

from modin.data_management.functions.function import Function
from modin.utils import try_cast_to_pandas

from pandas.core.dtypes.common import is_list_like
import pandas


class DefaultMethod(Function):
    OBJECT_TYPE = "DataFrame"

    @classmethod
    def call(cls, func, **call_kwds):
        obj = call_kwds.get("obj_type", pandas.DataFrame)
        force_inplace = call_kwds.get("inplace")
        fn_name = call_kwds.get("fn_name", getattr(func, "__name__", str(func)))

        if isinstance(func, str):
            fn = getattr(obj, func)
        else:
            fn = func

        if type(fn) == property:
            fn = cls.build_property_wrapper(fn)

        def applyier(df, *args, **kwargs):
            df = cls.frame_wrapper(df)
            result = fn(df, *args, **kwargs)

            if (
                not isinstance(result, pandas.Series)
                and not isinstance(result, pandas.DataFrame)
                and func != "to_numpy"
                and func != pandas.DataFrame.to_numpy
            ):
                result = (
                    pandas.DataFrame(result)
                    if is_list_like(result)
                    else pandas.DataFrame([result])
                )
            if isinstance(result, pandas.Series):
                if result.name is None:
                    result.name = "__reduced__"
                result = result.to_frame()

            inplace = kwargs.get("inplace", False)
            if force_inplace is not None:
                inplace = force_inplace
            return result if not inplace else df

        return cls.build_wrapper(applyier, fn_name)

    @classmethod
    def register(cls, func, **kwargs):
        return cls.call(func, **kwargs)

    @classmethod
    def build_wrapper(cls, fn, fn_name):
        wrapper = cls.build_default_to_pandas(fn, fn_name)

        def args_cast(self, *args, **kwargs):
            args = try_cast_to_pandas(args)
            kwargs = try_cast_to_pandas(kwargs)
            return wrapper(self, *args, **kwargs)

        return args_cast

    @classmethod
    def build_property_wrapper(cls, prop):
        def property_wrapper(df):
            return prop.fget(df)

        return property_wrapper

    @classmethod
    def build_default_to_pandas(cls, fn, fn_name):
        fn.__name__ = f"<function {cls.OBJECT_TYPE}.{fn_name}>"

        def wrapper(self, *args, **kwargs):
            return self.default_to_pandas(fn, *args, **kwargs)

        return wrapper

    @classmethod
    def frame_wrapper(cls, df):
        return df
