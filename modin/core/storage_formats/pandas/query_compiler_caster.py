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
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from types import FunctionType, MethodType
from typing import Any, Dict, Optional, Tuple, TypeVar, Union, ValuesView

from pandas.core.indexes.frozen import FrozenList
from typing_extensions import Self

from modin.config import Backend
from modin.core.storage_formats.base.query_compiler import BaseQueryCompiler
from modin.core.storage_formats.base.query_compiler_calculator import (
    BackendCostCalculator,
)

Fn = TypeVar("Fn", bound=Any)


# This type describes a defaultdict that maps backend name (or `None` for
# method implementation and not bound to any one extension) to the dictionary of
# extensions for that backend. The keys of the inner dictionary are the names of
# the extensions, and the values are the extensions themselves.
EXTENSION_DICT_TYPE = defaultdict[Optional[str], dict[str, Any]]


_NON_EXTENDABLE_ATTRIBUTES = {
    # we use these attributes to implement casting and backend dispatching, so
    # we can't allow extensions to override them.
    "__getattribute__",
    "__setattr__",
    "__delattr__",
    "__getattr__",
    "_getattribute__from_extension_impl",
    "_getattr__from_extension_impl",
    "get_backend",
    "move_to",
    "_update_inplace",
    "set_backend",
    "_get_extension",
    "_query_compiler",
    "_get_query_compiler",
    "_copy_into",
}


class QueryCompilerCaster(ABC):
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
        apply_argument_cast_to_class(cls)

    @abstractmethod
    def _get_query_compiler(self) -> Optional[BaseQueryCompiler]:
        """
        Get the query compiler storing data for this object.

        Returns
        -------
        Optional[BaseQueryCompiler]
            The query compiler storing data for this object, if it exists.
            Otherwise, None.
        """
        pass

    @abstractmethod
    def get_backend(self) -> str:
        """
        Get the backend of this object.

        Returns
        -------
        str
            The backend of this object. The backend name must be title-cased.
        """
        pass

    @abstractmethod
    def set_backend(self, backend: str) -> Self:
        """
        Set the backend of this object.

        Parameters
        ----------
        backend : str
            The new backend.
        """
        pass

    @abstractmethod
    def _copy_into(self, other: Self) -> None:
        """
        Copy the data from this object into another object of the same type.

        Parameters
        ----------
        other : Self
            The object to copy data into.
        """
        pass


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
    immutable_types = (FrozenList, tuple, ValuesView)
    if isinstance(arguments, immutable_types):
        args_type = type(arguments)
        return (
            # ValuesView, which we might get from dict.values(), is immutable,
            # but not constructable, so we convert it to a tuple. Otherwise,
            # we return an object of the same type as the input.
            tuple
            if issubclass(args_type, ValuesView)
            else args_type
        )(visit_nested_args(list(arguments), fn))
    types_to_recursively_visit = (list, dict, *immutable_types)
    if isinstance(
        arguments,
        list,
    ):
        for i in range(len(arguments)):
            if isinstance(arguments[i], types_to_recursively_visit):
                visit_nested_args(arguments[i], fn)
            else:
                arguments[i] = fn(arguments[i])
    elif isinstance(arguments, dict):
        for key in arguments:
            if isinstance(arguments[key], types_to_recursively_visit):
                visit_nested_args(arguments[key], fn)
            else:
                arguments[key] = fn(arguments[key])
    return arguments


def _assert_casting_functions_wrap_same_implementation(
    m1: callable, m2: callable
) -> None:
    """
    Assert that two casting wrappers wrap the same implementation.

    Parameters
    ----------
    m1 : callable
        The first casting wrapper.
    m2 : callable
        The second casting wrapper.

    Raises
    ------
    AssertionError
        If the two casting wrappers wrap different implementations.
    """
    assert (
        # For cases like (m1=Series.agg, m2=Series.aggregate), where Series
        # defines its own method and aliases it, the two wrapped methods
        # are the same.
        m2._wrapped_method_for_casting is m1._wrapped_method_for_casting
        # For cases like (m1=Series.kurt, m2=Series.kurtosis), where Series
        # inherits both kurt and kurtosis from BasePandasDataset but does
        # not define its own implementation of either,
        # Series.kurt._wrapped_method_for_casting points to
        # BasePandasDataset.kurt, which is not the same as
        # BasePandasDataset.kurtosis. In that case, we need to go one level
        # deeper to compare the wrapped methods of the two aliases of
        # BasePandasDataset.
        or m2._wrapped_method_for_casting._wrapped_method_for_casting
        is m1._wrapped_method_for_casting._wrapped_method_for_casting
    )


def apply_argument_cast_to_class(klass: type) -> type:
    """
    Apply argument casting to all functions in a class.

    Parameters
    ----------
    klass : type
        The class to apply argument casting to.

    Returns
    -------
    type
        The class with argument casting applied to all functions.
    """
    all_attrs = dict(inspect.getmembers(klass))
    # This is required because inspect converts class methods to member functions
    current_class_attrs = vars(klass)
    for key in current_class_attrs:
        all_attrs[key] = current_class_attrs[key]

    for attr_name, attr_value in all_attrs.items():
        if attr_name in _NON_EXTENDABLE_ATTRIBUTES or not isinstance(
            attr_value, (FunctionType, classmethod, staticmethod)
        ):
            continue

        implementation_function = (
            attr_value.__func__
            if isinstance(attr_value, (classmethod, staticmethod))
            else attr_value
        )
        if attr_name not in klass._extensions[None]:
            # Register the original implementation as the default
            # extension. We fall back to this implementation if the
            # object's backend does not have an implementation for this
            # method.
            klass._extensions[None][attr_name] = implementation_function

        casting_implementation = wrap_function_in_argument_caster(
            f=implementation_function,
            wrapping_function_type=(
                classmethod
                if isinstance(attr_value, classmethod)
                else (
                    staticmethod if isinstance(attr_value, staticmethod) else MethodType
                )
            ),
            extensions=klass._extensions,
            name=attr_name,
        )
        wrapped = (
            classmethod(casting_implementation)
            if isinstance(attr_value, classmethod)
            else (
                staticmethod(casting_implementation)
                if isinstance(attr_value, staticmethod)
                else casting_implementation
            )
        )
        if attr_name not in klass.__dict__:
            # If this class's method comes from a superclass (i.e.
            # it's not in klass.__dict__), mark it so that
            # modin.utils._inherit_docstrings knows that the method
            # must get its docstrings from its superclass.
            wrapped._wrapped_superclass_method = attr_value
        setattr(klass, attr_name, wrapped)

    return klass


def wrap_function_in_argument_caster(
    f: callable,
    name: str,
    wrapping_function_type: Optional[
        Union[type[classmethod], type[staticmethod], type[MethodType]]
    ],
    extensions: EXTENSION_DICT_TYPE,
) -> callable:
    """
    Wrap a function so that it casts all castable arguments to a consistent query compiler, and uses the correct extension implementation for methods.

    Parameters
    ----------
    f : callable
        The function to wrap.
    name : str
        The name of the function.
    wrapping_function_type : Optional[Union[type[classmethod], type[staticmethod], type[MethodType]]
        The type of the original function that `f` implements.
        - `None` means we are wrapping a free function, e.g. pd.concat()
        - `classmethod` means we are wrapping a classmethod.
        - `staticmethod` means we are wrapping a staticmethod.
        - `MethodType` means we are wrapping a regular method of a class.
    extensions : EXTENSION_DICT_TYPE
        The class of the function we are wrapping. This should be None if
        and only if `wrapping_function_type` is None.

    Returns
    -------
    callable
        The wrapped function.
    """

    @functools.wraps(f)
    def f_with_argument_casting(*args: Tuple, **kwargs: Dict) -> Any:
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
        if wrapping_function_type in (classmethod, staticmethod):
            # TODO: currently we don't support any kind of casting or extension
            # for classmethod or staticmethod.
            return f(*args, **kwargs)

        # f() may make in-place updates to some of its arguments. If we cast
        # an argument and then f() updates it in place, the updates will not
        # be reflected in the original object. As a fix, we keep track of all
        # the in-place updates that f() makes, and once f() is finished, we
        # copy the updates back into the original objects. The query compiler
        # interface is mostly immutable (the only exceptions being the mutable
        # index and column properties), so to check for an in-place update, we
        # check whether an input's query compiler has changed its identity.
        InplaceUpdateTracker = namedtuple(
            "InplaceUpdateTracker",
            ["input_castable", "original_query_compiler", "new_castable"],
        )
        inplace_update_trackers: list[InplaceUpdateTracker] = []
        calculator: BackendCostCalculator = BackendCostCalculator()

        def register_query_compilers(arg):
            if (
                isinstance(arg, QueryCompilerCaster)
                and (qc := arg._get_query_compiler()) is not None
            ):
                calculator.add_query_compiler(qc)
            elif isinstance(arg, BaseQueryCompiler):
                # We might get query compiler arguments in __init__()
                calculator.add_query_compiler(arg)
            return arg

        visit_nested_args(args, register_query_compilers)
        visit_nested_args(kwargs, register_query_compilers)

        # Sometimes the function takes no query compilers as inputs,
        # e.g. pd.to_datetime(0), or some cases of DataFrame.__init__()
        if len(calculator._qc_list) > 0:
            result_backend = calculator.calculate()

            def cast_to_qc(arg):
                if not (
                    isinstance(arg, QueryCompilerCaster)
                    and arg._get_query_compiler() is not None
                    and arg.get_backend() != result_backend
                ):
                    return arg
                cast = arg.set_backend(result_backend)
                inplace_update_trackers.append(
                    InplaceUpdateTracker(
                        input_castable=arg,
                        original_query_compiler=cast._get_query_compiler(),
                        new_castable=cast,
                    )
                )
                return cast

            args = visit_nested_args(args, cast_to_qc)
            kwargs = visit_nested_args(kwargs, cast_to_qc)
        else:
            result_backend = Backend.get()
        if name in extensions[result_backend]:
            f_to_apply = extensions[result_backend][name]
        else:
            if name not in extensions[None]:
                raise AttributeError(
                    (
                        # When python invokes a method on an object, it passes the object as
                        # the first positional argument.
                        (
                            f"{(type(args[0]).__name__)} object"
                            if wrapping_function_type is MethodType
                            else "module 'modin.pandas'"
                        )
                        + f" has no attribute {name}"
                    )
                )
            f_to_apply = extensions[None][name]
        result = f_to_apply(*args, **kwargs)
        for (
            original_castable,
            original_qc,
            new_castable,
        ) in inplace_update_trackers:
            new_qc = new_castable._get_query_compiler()
            if original_qc is not new_qc:
                new_castable._copy_into(original_castable)
        return result

    f_with_argument_casting._wrapped_method_for_casting = f
    return f_with_argument_casting


_GENERAL_EXTENSIONS: EXTENSION_DICT_TYPE = defaultdict(dict)


def wrap_free_function_in_argument_caster(name: str) -> callable:
    """
    Get a wrapper for a free function that casts all castable arguments to a consistent query compiler.

    Parameters
    ----------
    name : str
        The name of the function.

    Returns
    -------
    callable
        A wrapper for a free function that casts all castable arguments to a consistent query compiler.
    """

    def wrapper(f):
        if name not in _GENERAL_EXTENSIONS[None]:
            _GENERAL_EXTENSIONS[None][name] = f

        return wrap_function_in_argument_caster(
            f=f,
            wrapping_function_type=None,
            extensions=_GENERAL_EXTENSIONS,
            name=name,
        )

    return wrapper
