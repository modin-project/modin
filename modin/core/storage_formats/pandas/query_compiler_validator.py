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
Module contains ``QueryCompilerTypeCaster`` class.

``QueryCompilerTypeCaster`` is used for va.
"""

from types import FunctionType, MethodType
from typing import Any, Dict, Tuple, TypeVar

from modin.core.storage_formats.base.query_compiler import BaseQueryCompiler

Fn = TypeVar("Fn", bound=Any)


class QueryCompilerTypeCaster:
    """
    Cast all query compiler arguments of a the member function to current query compiler.

    """

    @classmethod
    def __init_subclass__(
        cls,
        **kwargs: Dict,
    ) -> None:
        """
        Apply type casting to all children of ``QueryCompilerTypeCaster``.

        """
        super().__init_subclass__(**kwargs)
        apply_argument_casting()(cls)


def cast_nested_args(arguments, type_to_cast):
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    if isinstance(arguments, tuple):
        arguments = list(arguments)
        arguments = cast_nested_args(arguments, type_to_cast)
        return tuple(arguments)
    if isinstance(arguments, list):
        for i in range(len(arguments)):
            if isinstance(arguments[i], (list, dict)):
                cast_nested_args(arguments[i], type_to_cast)
            else:
                if isinstance(arguments[i], BaseQueryCompiler) and not isinstance(
                    arguments[i], type_to_cast
                ):
                    arguments[i] = FactoryDispatcher.from_pandas(
                        arguments[i].to_pandas()
                    )
    elif isinstance(arguments, dict):
        for key in arguments:
            if isinstance(arguments[key], (list, dict)):
                cast_nested_args(arguments[key], type_to_cast)
            else:
                if isinstance(arguments[key], BaseQueryCompiler) and not isinstance(
                    arguments[key], type_to_cast
                ):
                    arguments[key] = FactoryDispatcher.from_pandas(
                        arguments[key].to_pandas()
                    )
    return arguments


def apply_argument_casting():
    """
    Cast args of all functions that are members of query compilers.
    Returns
    -------
    func
        A decorator function.
    """

    def decorator(obj: Fn) -> Fn:
        """Decorate function or class to cast all arguments that are query comilers to the current query compiler"""
        if isinstance(obj, type):
            seen: Dict[Any, Any] = {}
            for attr_name, attr_value in vars(obj).items():
                if isinstance(
                    attr_value, (FunctionType, MethodType, classmethod, staticmethod)
                ):
                    try:
                        wrapped = seen[attr_value]
                    except KeyError:
                        wrapped = seen[attr_value] = apply_argument_casting()(
                            attr_value
                        )

                    setattr(obj, attr_name, wrapped)
            return obj  # type: ignore [return-value]
        elif isinstance(obj, classmethod):
            return classmethod(decorator(obj.__func__))  # type: ignore [return-value, arg-type]
        elif isinstance(obj, staticmethod):
            return staticmethod(decorator(obj.__func__))

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
            current_qc_type = None
            if isinstance(args[0], BaseQueryCompiler):
                current_qc_type = type(args[0])
            elif issubclass(args[0], BaseQueryCompiler):
                current_qc_type = type(args[0])
            if current_qc_type:
                args = cast_nested_args(args, current_qc_type)
            return obj(*args, **kwargs)

        return cast_args

    return decorator
