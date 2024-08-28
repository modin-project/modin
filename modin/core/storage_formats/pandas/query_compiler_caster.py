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

"""
Module contains ``QueryCompilerCaster`` class.

``QueryCompilerCaster`` is used for automatically casting query compiler
arguments to the type of the current query compiler for query compiler class functions.
This ensures compatibility between different query compiler classes.
"""

import functools
import inspect
from types import FunctionType, MethodType
from typing import Any, Dict, Tuple, TypeVar

from pandas.core.indexes.frozen import FrozenList

from modin.core.storage_formats.base.query_compiler import BaseQueryCompiler

Fn = TypeVar("Fn", bound=Any)


class QueryCompilerCaster:
    """Cast all query compiler arguments of the member function to current query compiler."""

    @classmethod
    def __init_subclass__(
        cls,
        **kwargs: Dict,
    ) -> None:
        """
        Apply type casting to all children of ``QueryCompilerCaster``.

        This method is called automatically when a class inherits from
        ``QueryCompilerCaster``. It ensures that all member functions within the
        subclass have their arguments automatically casted to the current query
        compiler type.

        Parameters
        ----------
        **kwargs : Additional keyword arguments
        """
        super().__init_subclass__(**kwargs)
        apply_argument_cast(cls)


def cast_nested_args_to_current_qc_type(arguments, current_qc):
    """
    Cast all arguments in nested fashion to current query compiler.

    Parameters
    ----------
    arguments : tuple or dict
    current_qc : BaseQueryCompiler

    Returns
    -------
    tuple or dict
        Returns args and kwargs with all query compilers casted to current_qc.
    """

    def cast_arg_to_current_qc(arg):
        current_qc_type = type(current_qc)
        if isinstance(arg, BaseQueryCompiler) and not isinstance(arg, current_qc_type):
            data_cls = current_qc._modin_frame
            return current_qc_type.from_pandas(arg.to_pandas(), data_cls)
        else:
            return arg

    imutable_types = (FrozenList, tuple)
    if isinstance(arguments, imutable_types):
        args_type = type(arguments)
        arguments = list(arguments)
        arguments = cast_nested_args_to_current_qc_type(arguments, current_qc)

        return args_type(arguments)
    if isinstance(arguments, list):
        for i in range(len(arguments)):
            if isinstance(arguments[i], (list, dict)):
                cast_nested_args_to_current_qc_type(arguments[i], current_qc)
            else:
                arguments[i] = cast_arg_to_current_qc(arguments[i])
    elif isinstance(arguments, dict):
        for key in arguments:
            if isinstance(arguments[key], (list, dict)):
                cast_nested_args_to_current_qc_type(arguments[key], current_qc)
            else:
                arguments[key] = cast_arg_to_current_qc(arguments[key])
    return arguments


def apply_argument_cast(obj: Fn) -> Fn:
    """
    Cast all arguments that are query compilers to the current query compiler.

    Parameters
    ----------
    obj : function

    Returns
    -------
    function
        Returns decorated function which does argument casting.
    """
    if isinstance(obj, type):
        all_attrs = dict(inspect.getmembers(obj))
        all_attrs.pop("__abstractmethods__")

        # This is required because inspect converts class methods to member functions
        current_class_attrs = vars(obj)
        for key in current_class_attrs:
            all_attrs[key] = current_class_attrs[key]

        for attr_name, attr_value in all_attrs.items():
            if isinstance(
                attr_value, (FunctionType, MethodType, classmethod, staticmethod)
            ):
                wrapped = apply_argument_cast(attr_value)
                setattr(obj, attr_name, wrapped)
        return obj  # type: ignore [return-value]
    elif isinstance(obj, classmethod):
        return classmethod(apply_argument_cast(obj.__func__))  # type: ignore [return-value, arg-type]
    elif isinstance(obj, staticmethod):
        return staticmethod(apply_argument_cast(obj.__func__))

    @functools.wraps(obj)
    def cast_args(*args: Tuple, **kwargs: Dict) -> Any:
        """
        Add casting for query compiler arguments.

        Parameters
        ----------
        *args : tuple
            The function arguments.
        **kwargs : dict
            The function keyword arguments.

        Returns
        -------
        Any
        """
        current_qc = args[0]
        if isinstance(current_qc, BaseQueryCompiler):
            kwargs = cast_nested_args_to_current_qc_type(kwargs, current_qc)
            args = cast_nested_args_to_current_qc_type(args, current_qc)
        return obj(*args, **kwargs)

    return cast_args
