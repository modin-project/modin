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

"""Module houses default functions builder class."""

from modin.core.dataframe.algebra import Operator
from modin.utils import try_cast_to_pandas

from pandas.core.dtypes.common import is_list_like
import pandas


class DefaultMethod(Operator):
    """
    Builder for default-to-pandas methods.

    Attributes
    ----------
    OBJECT_TYPE : str
        Object type name that will be shown in default-to-pandas warning message.
    """

    OBJECT_TYPE = "DataFrame"

    @classmethod
    def call(cls, func, obj_type=pandas.DataFrame, inplace=None, fn_name=None):
        """
        Build function that do fallback to default pandas implementation for passed `func`.

        Parameters
        ----------
        func : callable or str,
            Function to apply to the casted to pandas frame or its property accesed
            by ``cls.frame_wrapper``.
        obj_type : object, default: pandas.DataFrame
            If `func` is a string with a function name then `obj_type` provides an
            object to search function in.
        inplace : bool, optional
            If True return an object to which `func` was applied, otherwise return
            the result of `func`.
        fn_name : str, optional
            Function name which will be shown in default-to-pandas warning message.
            If not specified, name will be deducted from `func`.

        Returns
        -------
        callable
            Function that takes query compiler, does fallback to pandas and applies `func`
            to the casted to pandas frame or its property accesed by ``cls.frame_wrapper``.
        """
        fn_name = getattr(func, "__name__", str(func)) if fn_name is None else fn_name

        if isinstance(func, str):
            fn = getattr(obj_type, func)
        else:
            fn = func

        if type(fn) == property:
            fn = cls.build_property_wrapper(fn)

        def applyier(df, *args, **kwargs):
            """
            Apply target function to the casted to pandas frame.

            This function is directly applied to the casted to pandas frame, executes target
            function under it and processes result so it is possible to create a valid
            query compiler from it.
            """
            df = cls.frame_wrapper(df)
            result = fn(df, *args, **kwargs)

            if (
                not isinstance(result, pandas.Series)
                and not isinstance(result, pandas.DataFrame)
                and func != "to_numpy"
                and func != pandas.DataFrame.to_numpy
            ):
                # When applying a DatetimeProperties function, if we don't
                # specify the dtype for the DataFrame, the frame might get the
                # wrong dtype, e.g. for to_pydatetime in
                # https://github.com/modin-project/modin/issues/4436
                astype_kwargs = {}
                dtype = getattr(result, "dtype", None)
                if dtype and isinstance(
                    df, pandas.core.indexes.accessors.DatetimeProperties
                ):
                    astype_kwargs["dtype"] = dtype
                result = (
                    pandas.DataFrame(result, **astype_kwargs)
                    if is_list_like(result)
                    else pandas.DataFrame([result], **astype_kwargs)
                )
            if isinstance(result, pandas.Series):
                if result.name is None:
                    result.name = "__reduced__"
                result = result.to_frame()

            inplace_method = kwargs.get("inplace", False)
            if inplace is not None:
                inplace_method = inplace
            return result if not inplace_method else df

        return cls.build_wrapper(applyier, fn_name)

    @classmethod
    # FIXME: `register` is an alias for `call` method. One of them should be removed.
    def register(cls, func, **kwargs):
        """
        Build function that do fallback to default pandas implementation for passed `func`.

        Parameters
        ----------
        func : callable or str,
            Function to apply to the casted to pandas frame or its property accesed
            by ``cls.frame_wrapper``.
        **kwargs : kwargs
            Additional parameters that will be used for building.

        Returns
        -------
        callable
            Function that takes query compiler, does fallback to pandas and applies `func`
            to the casted to pandas frame or its property accesed by ``cls.frame_wrapper``.
        """
        return cls.call(func, **kwargs)

    @classmethod
    # FIXME: this method is almost a duplicate of `cls.build_default_to_pandas`.
    # Those two methods should be merged into a single one.
    def build_wrapper(cls, fn, fn_name):
        """
        Build function that do fallback to pandas for passed `fn`.

        In comparison with ``cls.build_default_to_pandas`` this method also
        casts function arguments to pandas before doing fallback.

        Parameters
        ----------
        fn : callable
            Function to apply to the defaulted frame.
        fn_name : str
            Function name which will be shown in default-to-pandas warning message.

        Returns
        -------
        callable
            Method that does fallback to pandas and applies `fn` to the pandas frame.
        """
        wrapper = cls.build_default_to_pandas(fn, fn_name)

        def args_cast(self, *args, **kwargs):
            """
            Preprocess `default_to_pandas` function arguments and apply default function.

            Cast all Modin objects that function arguments contain to its pandas representation.
            """
            args = try_cast_to_pandas(args)
            kwargs = try_cast_to_pandas(kwargs)
            return wrapper(self, *args, **kwargs)

        return args_cast

    @classmethod
    def build_property_wrapper(cls, prop):
        """
        Build function that accesses specified property of the frame.

        Parameters
        ----------
        prop : str
            Property name to access.

        Returns
        -------
        callable
            Function that takes DataFrame and returns its value of `prop` property.
        """

        def property_wrapper(df):
            """Get specified property of the passed object."""
            return prop.fget(df)

        return property_wrapper

    @classmethod
    def build_default_to_pandas(cls, fn, fn_name):
        """
        Build function that do fallback to pandas for passed `fn`.

        Parameters
        ----------
        fn : callable
            Function to apply to the defaulted frame.
        fn_name : str
            Function name which will be shown in default-to-pandas warning message.

        Returns
        -------
        callable
            Method that does fallback to pandas and applies `fn` to the pandas frame.
        """
        fn.__name__ = f"<function {cls.OBJECT_TYPE}.{fn_name}>"

        def wrapper(self, *args, **kwargs):
            """Do fallback to pandas for the specified function."""
            return self.default_to_pandas(fn, *args, **kwargs)

        return wrapper

    @classmethod
    def frame_wrapper(cls, df):
        """
        Extract frame property to apply function on.

        This method is executed under casted to pandas frame right before applying
        a function passed to `register`, which gives an ability to transform frame somehow
        or access its properties, by overriding this method in a child class.

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        -------
        pandas.DataFrame

        Notes
        -----
        Being a base implementation, this particular method does nothing with passed frame.
        """
        return df
