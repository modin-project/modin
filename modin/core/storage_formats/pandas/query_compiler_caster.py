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

from modin.core.storage_formats.base.query_compiler import (
    BaseQueryCompiler,
)
from modin.core.storage_formats.base.query_compiler_calculator import (
    BackendCostCalculator,
)

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


def visit_nested_args(arguments, fn: callable):
    """
    Visit each argument recursively, calling fn on each one.

    Parameters
    ----------
    arguments : tuple or dict
    fn : Callable to apply to matching arguments

    Returns
    -------
    tuple or dict
        Returns args and kwargs with all query compilers casted to current_qc.
    """
    imutable_types = (FrozenList, tuple)
    if isinstance(arguments, imutable_types):
        args_type = type(arguments)
        arguments = list(arguments)
        arguments = visit_nested_args(arguments, fn)

        return args_type(arguments)
    if isinstance(arguments, list):
        for i in range(len(arguments)):
            if isinstance(arguments[i], (list, dict)):
                visit_nested_args(arguments[i], fn)
            else:
                arguments[i] = fn(arguments[i])
    elif isinstance(arguments, dict):
        for key in arguments:
            if isinstance(arguments[key], (list, dict)):
                visit_nested_args(arguments[key], fn)
            else:
                arguments[key] = fn(arguments[key])
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

        # This is required because inspect converts class methods to member functions
        current_class_attrs = vars(obj)
        for key in current_class_attrs:
            all_attrs[key] = current_class_attrs[key]
        all_attrs.pop("__abstractmethods__")
        all_attrs.pop("get_backend")
        all_attrs.pop("__init__")
        all_attrs.pop("qc_engine_switch_cost")
        all_attrs.pop("from_pandas")
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
        if len(args) == 0 and len(kwargs) == 0:
            return
        
        from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher
        
        current_qc = args[0]
        calculator = BackendCostCalculator()
        calculator.add_query_compiler(current_qc)

        def arg_needs_casting(arg):
            current_qc_type = type(current_qc)
            return isinstance(arg, BaseQueryCompiler) and not isinstance(
                arg, current_qc_type
            )

        def register_query_compilers(arg):
            if not arg_needs_casting(arg):
                return arg
            calculator.add_query_compiler(arg)
            return arg

        def cast_to_qc(arg):
            if not arg_needs_casting(arg):
                return arg
            qc_type = calculator.calculate()
            if qc_type is None or qc_type is type(arg):
                return arg
            # TODO: Should use the factory dispatcher here to switch backends
            # TODO: handle the non-backend string approach
            return FactoryDispatcher.from_pandas(arg.to_pandas(), calculator.calculate())
        if isinstance(current_qc, BaseQueryCompiler):
            visit_nested_args(kwargs, register_query_compilers)
            visit_nested_args(args, register_query_compilers)

            args = visit_nested_args(args, cast_to_qc)
            kwargs = visit_nested_args(kwargs, cast_to_qc)

        result_backend = calculator.calculate()
        current_backend = args[0].get_backend()

        if result_backend == current_backend:
            return obj(*args, **kwargs)
        # TODO: Should use the factory dispatcher here to switch backends
        # TODO: handle the non-backend string approach

        new_qc = FactoryDispatcher.from_pandas(current_qc.to_pandas(), calculator.calculate())
        obj_new = getattr(new_qc, obj.__name__)
        return obj_new(*args[1:], **kwargs)

    return cast_args
