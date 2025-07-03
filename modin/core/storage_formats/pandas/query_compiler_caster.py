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
import random
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from types import FunctionType, MappingProxyType, MethodType
from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union, ValuesView

import pandas
from pandas.core.indexes.frozen import FrozenList
from typing_extensions import Self

from modin.config import AutoSwitchBackend, Backend
from modin.config import context as config_context
from modin.core.storage_formats.base.query_compiler import (
    BaseQueryCompiler,
    QCCoercionCost,
)
from modin.core.storage_formats.base.query_compiler_calculator import (
    BackendCostCalculator,
)
from modin.error_message import ErrorMessage
from modin.logging import disable_logging, get_logger
from modin.logging.metrics import emit_metric
from modin.utils import _inherit_docstrings, sentinel

Fn = TypeVar("Fn", bound=Any)

# Constant for the default class name when class_of_wrapped_fn is None
# (represents functions in the modin.pandas module)
MODIN_PANDAS_MODULE_NAME = "modin.pandas"


def _normalize_class_name(class_of_wrapped_fn: Optional[str]) -> str:
    """
    Normalize class name for logging and operation tracking.

    Parameters
    ----------
    class_of_wrapped_fn : Optional[str]
        The name of the class that the function belongs to. `None` for functions
        in the modin.pandas module.

    Returns
    -------
    str
        The normalized class name. Returns "modin.pandas" if input is None.
    """
    return (
        class_of_wrapped_fn
        if class_of_wrapped_fn is not None
        else MODIN_PANDAS_MODULE_NAME
    )


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
    "is_backend_pinned",
    "_set_backend_pinned",
    "pin_backend",
    "unpin_backend",
    "__dict__",
}


# Do not look up these attributes when searching for extensions. We use them
# to implement the extension lookup itself.
EXTENSION_NO_LOOKUP = {
    "_get_extension",
    "_query_compiler",
    "get_backend",
    "_getattribute__from_extension_impl",
    "_getattr__from_extension_impl",
    "_get_query_compiler",
    "set_backend",
    "_pinned",
    "is_backend_pinned",
    "_set_backend_pinned",
    "pin_backend",
    "unpin_backend",
}


BackendAndClassName = namedtuple("BackendAndClassName", ["backend", "class_name"])

_AUTO_SWITCH_CLASS = defaultdict[BackendAndClassName, set[str]]

_CLASS_AND_BACKEND_TO_POST_OP_SWITCH_METHODS: _AUTO_SWITCH_CLASS = _AUTO_SWITCH_CLASS(
    set
)

_CLASS_AND_BACKEND_TO_PRE_OP_SWITCH_METHODS: _AUTO_SWITCH_CLASS = _AUTO_SWITCH_CLASS(
    set
)


def _get_empty_qc_for_default_backend() -> BaseQueryCompiler:
    """
    Get an empty query compiler for the default backend.

    Returns
    -------
    BaseQueryCompiler
        An empty query compiler for the default backend.
    """
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    return FactoryDispatcher.get_factory().io_cls.from_pandas(pandas.DataFrame())


_BACKEND_TO_EMPTY_QC: defaultdict[str, BaseQueryCompiler] = defaultdict(
    _get_empty_qc_for_default_backend
)


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
    def is_backend_pinned(self) -> bool:
        """
        Get whether this object's data is pinned to a particular backend.

        Returns
        -------
        bool
            True if the data is pinned.
        """
        pass

    @abstractmethod
    def _set_backend_pinned(self, pinned: bool, inplace: bool) -> Optional[Self]:
        """
        Update whether this object's data is pinned to a particular backend.

        Parameters
        ----------
        pinned : bool
            Whether the data is pinned.

        inplace : bool, default: False
            Whether to update the object in place.

        Returns
        -------
        Optional[Self]
            The object with the new pin state, if `inplace` is False. Otherwise, None.
        """
        pass

    def pin_backend(self, inplace: bool = False) -> Optional[Self]:
        """
        Pin the object's underlying data, preventing Modin from automatically moving it to another backend.

        Parameters
        ----------
        inplace : bool, default: False
            Whether to update the object in place.

        Returns
        -------
        Optional[Self]
            The newly-pinned object, if `inplace` is False. Otherwise, None.
        """
        return self._set_backend_pinned(True, inplace)

    def unpin_backend(self, inplace: bool = False) -> Optional[Self]:
        """
        Unpin the object's underlying data, allowing Modin to automatically move it to another backend.

        Parameters
        ----------
        inplace : bool, default: False
            Whether to update the object in place.

        Returns
        -------
        Optional[Self]
            The newly-unpinned object, if `inplace` is False. Otherwise, None.
        """
        return self._set_backend_pinned(False, inplace)

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
    def set_backend(
        self,
        backend: str,
        inplace: bool = False,
        *,
        switch_operation: Optional[str] = None,
    ) -> Optional[Self]:
        """
        Set the backend of this object.

        Parameters
        ----------
        backend : str
            The new backend.

        inplace : bool, default: False
            Whether to update the object in place.

        switch_operation : Optional[str], default: None
            The name of the operation that triggered the set_backend call.
            Internal argument used for displaying progress bar information.

        Returns
        -------
        Optional[Self]
            The object with the new backend, if `inplace` is False. Otherwise, None.
        """
        pass

    @_inherit_docstrings(set_backend)
    def move_to(
        self,
        backend: str,
        inplace: bool = False,
        *,
        switch_operation: Optional[str] = None,
    ) -> Optional[Self]:
        return self.set_backend(
            backend=backend, inplace=inplace, switch_operation=switch_operation
        )

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

    @disable_logging
    def _get_extension(self, name: str, extensions: EXTENSION_DICT_TYPE) -> Any:
        """
        Get an extension with the given name from the given set of extensions.

        Parameters
        ----------
        name : str
            The name of the extension.
        extensions : EXTENSION_DICT_TYPE
            The set of extensions.

        Returns
        -------
        Any
            The extension with the given name, or `sentinel` if the extension is not found.
        """
        if self._get_query_compiler() is not None:
            extensions_for_backend = extensions[self.get_backend()]
            if name in extensions_for_backend:
                return extensions_for_backend[name]
            if name in extensions[None]:
                return extensions[None][name]
        return sentinel

    @disable_logging
    def _getattribute__from_extension_impl(
        self, item: str, extensions: EXTENSION_DICT_TYPE
    ):
        """
        __getatttribute__() an extension with the given name from the given set of extensions.

        Implement __getattribute__() for extensions. Python calls
        __getattribute_() every time you access an attribute of an object.

        Parameters
        ----------
        item : str
            The name of the attribute to get.
        extensions : EXTENSION_DICT_TYPE
            The set of extensions.

        Returns
        -------
        Any
            The attribute from the extension, or `sentinel` if the attribute is
            not found.
        """
        # An extension property is only accessible if the backend supports it.
        extension = self._get_extension(item, extensions)
        if (
            extension is not sentinel
            # We should implement callable extensions by wrapping them in
            # methods that dispatch to the corrrect backend. We should get the
            # wrapped method with the usual object.__getattribute__() method
            # lookup rather than by getting a particular extension when we call
            # __getattribute__(). For example, if we've extended sort_values(),
            # then __getattribute__('sort_values') should return a wrapper that
            # calls the correct extension once it's invoked.
            and not callable(extension)
        ):
            return (
                extension.__get__(self) if hasattr(extension, "__get__") else extension
            )
        return sentinel

    @disable_logging
    def _getattr__from_extension_impl(
        self,
        key: str,
        default_behavior_attributes: set[str],
        extensions: EXTENSION_DICT_TYPE,
    ) -> Any:
        """
        Implement __getattr__, which the python interpreter falls back to if __getattribute__ raises AttributeError.

        We override this method to make sure we try to get the extension
        attribute for `key`, even if this class has a different
        attribute for `key`.

        Parameters
        ----------
        key : str
            Attribute name.
        default_behavior_attributes : set[str]
            The set of attributes for which we should follow the default
            __getattr__ behavior and not try to get the extension.
        extensions : EXTENSION_DICT_TYPE
            The set of extensions.

        Returns
        -------
        The value of the attribute.
        """
        if key not in default_behavior_attributes:
            # If this class has a an extension for `key`, but __getattribute__()
            # for the extension raises an AttributeError, we end up in this
            # method, which should try getting the extension again (and
            # probably raise the AttributeError that
            # _getattribute__from_extension_impl() originally raised), rather
            # than following back to object.__getattribute__().
            extensions_result = self._getattribute__from_extension_impl(key, extensions)
            # If extensions_result is not `sentinel`, __getattribute__() should have
            # returned it first.
            ErrorMessage.catch_bugs_and_request_email(
                failure_condition=extensions_result is not sentinel,
                extra_log=(
                    "This object should return extensions via "
                    + "__getattribute__ rather than __getattr__"
                ),
            )
        return object.__getattribute__(self, key)


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
    if isinstance(arguments, pandas.NamedAgg):
        # NamedAgg needs special treatment because it's an immutable subclass
        # of tuple that can't be constructed from another tuple.
        return pandas.NamedAgg(
            column=fn(arguments.column), aggfunc=fn(arguments.aggfunc)
        )
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
            klass=klass,
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


def _maybe_switch_backend_pre_op(
    function_name: str,
    input_qc: BaseQueryCompiler,
    class_of_wrapped_fn: Optional[str],
    arguments: MappingProxyType[str, Any],
) -> tuple[str, Callable[[Any], Any]]:
    """
    Possibly switch backend before a function.

    Parameters
    ----------
    function_name : str
        The name of the function.
    input_qc : BaseQueryCompiler
        The input query compiler.
    class_of_wrapped_fn : Optional[str]
        The name of the class that the function belongs to. `None` for functions
        in the modin.pandas module.
    arguments : MappingProxyType[str, Any]
        Mapping from operation argument names to their values.

    Returns
    -------
    Tuple[str, callable]
        A tuple of the new backend and a function that casts all castable arguments
        to the new query compiler type.
    """
    input_backend = input_qc.get_backend()
    if (
        function_name
        in _CLASS_AND_BACKEND_TO_PRE_OP_SWITCH_METHODS[
            BackendAndClassName(
                backend=input_qc.get_backend(), class_name=class_of_wrapped_fn
            )
        ]
    ):
        result_backend = _get_backend_for_auto_switch(
            input_qc=input_qc,
            class_of_wrapped_fn=class_of_wrapped_fn,
            function_name=function_name,
            arguments=arguments,
        )
    else:
        result_backend = input_backend

    def cast_to_qc(arg: Any) -> Any:
        if not (
            isinstance(arg, QueryCompilerCaster)
            and arg._get_query_compiler() is not None
            and arg.get_backend() != result_backend
        ):
            return arg
        arg.set_backend(
            result_backend,
            inplace=True,
            switch_operation=f"{_normalize_class_name(class_of_wrapped_fn)}.{function_name}",
        )
        return arg

    return result_backend, cast_to_qc


def _maybe_switch_backend_post_op(
    result: Any,
    function_name: str,
    qc_list: list[BaseQueryCompiler],
    starting_backend: str,
    class_of_wrapped_fn: Optional[str],
    pin_backend: bool,
    arguments: MappingProxyType[str, Any],
) -> Any:
    """
    Possibly switch the backend of the result of a function.

    Use cost-based optimization to determine whether to switch the backend of the
    result of a function. If the function returned a QueryCompilerCaster and the
    cost of switching is less than the cost of staying on the current backend,
    we switch. If there are multiple backends we can switch to, we choose the
    one that minimizes cost_to_move - cost_to_stay.

    Parameters
    ----------
    result : Any
        The result of the function.
    function_name : str
        The name of the function.
    qc_list : list[BaseQueryCompiler]
        The list of query compilers that were arguments to the function.
    starting_backend : str
        The backend used to run the function.
    class_of_wrapped_fn : Optional[str]
        The name of the class that the function belongs to. `None` for functions
        in the modin.pandas module.
    pin_backend : bool
        Whether the result should have its backend pinned, and therefore not moved.
    arguments : MappingProxyType[str, Any]
        Mapping from operation argument names to their values.

    Returns
    -------
    Any
        The result of the function, possibly with its backend switched.
    """
    # If any input QC was pinned, then the output should be as well.
    if pin_backend:
        if isinstance(result, QueryCompilerCaster):
            result.pin_backend(inplace=True)
        return result
    if (
        # only apply post-operation switch to nullary and unary methods
        len(qc_list) in (0, 1)
        and function_name
        in _CLASS_AND_BACKEND_TO_POST_OP_SWITCH_METHODS[
            BackendAndClassName(
                backend=(
                    qc_list[0].get_backend() if len(qc_list) == 1 else starting_backend
                ),
                class_name=class_of_wrapped_fn,
            )
        ]
        # if the operation did not return a query compiler, we can't switch the
        # backend of the result.
        and isinstance(result, QueryCompilerCaster)
        and (input_qc := result._get_query_compiler()) is not None
    ):
        return result.move_to(
            _get_backend_for_auto_switch(
                input_qc=input_qc,
                class_of_wrapped_fn=class_of_wrapped_fn,
                function_name=function_name,
                arguments=arguments,
            ),
            switch_operation=f"{_normalize_class_name(class_of_wrapped_fn)}.{function_name}",
        )
    return result


def _get_backend_for_auto_switch(
    input_qc: BaseQueryCompiler,
    class_of_wrapped_fn: str,
    function_name: str,
    arguments: MappingProxyType[str, Any],
) -> str:
    """
    Get the best backend to switch to.

    Use cost-based optimization to determine whether to switch the backend of the
    arguments to a function. If the cost of switching is less than the cost of
    staying on the current backend, we switch. If there are multiple backends we
    can switch to, we choose the one that minimizes cost_to_move - cost_to_stay.

    Parameters
    ----------
    input_qc : BaseQueryCompiler
        The query compiler representing the starting backend.
    class_of_wrapped_fn : Optional[str]
        The name of the class that the function belongs to. `None` for functions
        in the modin.pandas module.
    function_name : str
        The name of the function.
    arguments : MappingProxyType[str, Any]
        Mapping from operation argument names to their values.

    Returns
    -------
    str
        The name of the best backend to switch to.
    """
    # TODO(https://github.com/modin-project/modin/issues/7503): Make costing
    # methods take backend instead of query compiler type so that we don't
    # have to use the dispatcher to figure out the appropriate type for each
    # backend.
    from modin.core.execution.dispatching.factories.dispatcher import FactoryDispatcher

    # Does not need to be secure, should not use system entropy
    metrics_group = "%04x" % random.randrange(16**4)
    starting_backend = input_qc.get_backend()

    min_move_stay_delta = None
    best_backend = starting_backend

    stay_cost = input_qc.stay_cost(
        api_cls_name=class_of_wrapped_fn,
        operation=function_name,
        arguments=arguments,
    )
    data_max_shape = input_qc._max_shape()
    emit_metric(
        f"hybrid.auto.api.{class_of_wrapped_fn}.{function_name}.group.{metrics_group}",
        1,
    )
    emit_metric(
        f"hybrid.auto.current.{starting_backend}.group.{metrics_group}.stay_cost",
        stay_cost,
    )
    emit_metric(
        f"hybrid.auto.current.{starting_backend}.group.{metrics_group}.rows",
        data_max_shape[0],
    )
    emit_metric(
        f"hybrid.auto.current.{starting_backend}.group.{metrics_group}.cols",
        data_max_shape[1],
    )
    for backend in Backend.get_active_backends():
        if backend in ("Ray", "Unidist", "Dask"):
            # Disable automatically switching to these engines for now, because
            # 1) _get_prepared_factory_for_backend() currently calls
            # _initialize_engine(), which starts up the ray/dask/unidist
            #  processes
            # 2) we can't decide to switch to unidist in the middle of execution.
            continue
        if backend == starting_backend:
            continue
        move_to_class = FactoryDispatcher._get_prepared_factory_for_backend(
            backend=backend
        ).io_cls.query_compiler_cls
        move_to_cost = input_qc.move_to_cost(
            move_to_class,
            api_cls_name=class_of_wrapped_fn,
            operation=function_name,
            arguments=arguments,
        )
        other_execute_cost = move_to_class.move_to_me_cost(
            input_qc,
            api_cls_name=class_of_wrapped_fn,
            operation=function_name,
            arguments=arguments,
        )
        if (
            move_to_cost is not None
            and stay_cost is not None
            and other_execute_cost is not None
        ):
            if stay_cost >= QCCoercionCost.COST_IMPOSSIBLE:
                # We cannot execute the workload on the current engine
                # disregard the move_to_cost and just consider whether
                # the other engine can execute the workload
                move_stay_delta = other_execute_cost - stay_cost
            else:
                # We can execute this workload if we need to, consider
                # move_to_cost/transfer time in our decision
                move_stay_delta = (move_to_cost + other_execute_cost) - stay_cost
            if move_stay_delta < 0 and (
                min_move_stay_delta is None or move_stay_delta < min_move_stay_delta
            ):
                min_move_stay_delta = move_stay_delta
                best_backend = backend
            emit_metric(
                f"hybrid.auto.candidate.{backend}.group.{metrics_group}.move_to_cost",
                move_to_cost,
            )
            emit_metric(
                f"hybrid.auto.candidate.{backend}.group.{metrics_group}.other_execute_cost",
                other_execute_cost,
            )
            emit_metric(
                f"hybrid.auto.candidate.{backend}.group.{metrics_group}.delta",
                move_stay_delta,
            )

            get_logger().info(
                f"After {_normalize_class_name(class_of_wrapped_fn)} function {function_name}, "
                + f"considered moving to backend {backend} with "
                + f"(transfer_cost {move_to_cost} + other_execution_cost {other_execute_cost}) "
                + f", stay_cost {stay_cost}, and move-stay delta "
                + f"{move_stay_delta}"
            )

    if best_backend == starting_backend:
        emit_metric(f"hybrid.auto.decision.{best_backend}.group.{metrics_group}", 0)
        get_logger().info(
            f"Chose not to switch backends after operation {function_name}"
        )
    else:
        emit_metric(f"hybrid.auto.decision.{best_backend}.group.{metrics_group}", 1)
        get_logger().info(f"Chose to move to backend {best_backend}")
    return best_backend


def _get_extension_for_method(
    name: str,
    extensions: EXTENSION_DICT_TYPE,
    backend: str,
    args: tuple,
    wrapping_function_type: Optional[
        Union[type[classmethod], type[staticmethod], type[MethodType]]
    ],
) -> callable:
    """
    Get the extension implementation for a method.

    Parameters
    ----------
    name : str
        The name of the method.
    extensions : EXTENSION_DICT_TYPE
        The extension dictionary for the modin-API-level object (e.g. class
        DataFrame or module modin.pandas) that the method belongs to.
    backend : str
        The backend to use for this method call.
    args : tuple
        The arguments to the method.
    wrapping_function_type : Union[type[classmethod], type[staticmethod], type[MethodType]]
        The type of the original function that `f` implements.
        - `None` means we are wrapping a free function, e.g. pd.concat()
        - `classmethod` means we are wrapping a classmethod.
        - `staticmethod` means we are wrapping a staticmethod.
        - `MethodType` means we are wrapping a regular method of a class.

    Returns
    -------
    callable
        The implementation of the method for the given backend.
    """
    if name in extensions[backend]:
        f_to_apply = extensions[backend][name]
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
    return f_to_apply


def wrap_function_in_argument_caster(
    klass: Optional[type],
    f: callable,
    name: str,
    wrapping_function_type: Optional[
        Union[type[classmethod], type[staticmethod], type[MethodType]]
    ],
    extensions: EXTENSION_DICT_TYPE,
) -> callable:
    """
    Wrap a function so that it casts all castable arguments to a consistent query compiler, and uses the correct extension implementation for methods.

    Also propagates pin behavior across operations.

    Parameters
    ----------
    klass : Optional[type]
        Class of the function being wrapped.
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
        # The function name and class name of the function are passed to the calculator as strings
        class_of_wrapped_fn = klass.__name__ if klass is not None else None

        input_query_compilers: list[BaseQueryCompiler] = []

        pin_target_backend = None

        def register_query_compilers(arg):
            nonlocal pin_target_backend
            if (
                isinstance(arg, QueryCompilerCaster)
                and (qc := arg._get_query_compiler()) is not None
            ):
                arg_backend = arg.get_backend()
                if pin_target_backend is not None:
                    if arg.is_backend_pinned() and arg_backend != pin_target_backend:
                        raise ValueError(
                            f"Cannot combine arguments that are pinned to conflicting backends ({pin_target_backend}, {arg_backend})"
                        )
                elif arg.is_backend_pinned():
                    pin_target_backend = arg_backend
                input_query_compilers.append(qc)
            elif isinstance(arg, BaseQueryCompiler):
                # We might get query compiler arguments in __init__()
                input_query_compilers.append(arg)
            return arg

        visit_nested_args(args, register_query_compilers)
        visit_nested_args(kwargs, register_query_compilers)

        # Before determining any automatic switches, we perform the following checks:
        # 1. If the global AutoSwitchBackend configuration variable is set to False, do not switch.
        # 2. If there's only one query compiler and it's pinned, do not switch.
        # 3. If there are multiple query compilers, and at least one is pinned to a particular
        #    backend, then switch to that backend.
        # 4. If there are multiple query compilers, at least two of which are pinned to distinct
        #    backends, raise a ValueError.

        if len(input_query_compilers) == 0:
            input_backend = Backend.get()
            # For nullary functions, we need to create a dummy query compiler
            # to calculate the cost of switching backends. We should only
            # create the dummy query compiler once per backend.
            input_qc_for_pre_op_switch = _BACKEND_TO_EMPTY_QC[input_backend]
        else:
            input_qc_for_pre_op_switch = input_query_compilers[0]
            input_backend = input_qc_for_pre_op_switch.get_backend()

        inputs_pinned = (
            len(input_query_compilers) < 2 and pin_target_backend is not None
        )
        if not AutoSwitchBackend.get() or inputs_pinned:
            f_to_apply = _get_extension_for_method(
                name=name,
                extensions=extensions,
                backend=(
                    pin_target_backend
                    if pin_target_backend is not None
                    else input_backend
                ),
                args=args,
                wrapping_function_type=wrapping_function_type,
            )
            result = f_to_apply(*args, **kwargs)
            if isinstance(result, QueryCompilerCaster) and inputs_pinned:
                result._set_backend_pinned(True, inplace=True)
            return result

        # Bind the arguments using the function implementation for the input
        # backend. TODO(https://github.com/modin-project/modin/issues/7525):
        # Ideally every implementation would have the same signature.
        bound_arguments = inspect.signature(
            _get_extension_for_method(
                name=name,
                extensions=extensions,
                backend=input_backend,
                args=args,
                wrapping_function_type=wrapping_function_type,
            ),
        ).bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        args_dict = MappingProxyType(bound_arguments.arguments)

        if len(input_query_compilers) < 2:
            # No need to check should_pin_result() again, since we have already done so above.
            result_backend, cast_to_qc = _maybe_switch_backend_pre_op(
                name,
                input_qc=input_qc_for_pre_op_switch,
                class_of_wrapped_fn=class_of_wrapped_fn,
                arguments=args_dict,
            )
        else:
            calculator: BackendCostCalculator = BackendCostCalculator(
                operation_arguments=args_dict,
                api_cls_name=class_of_wrapped_fn,
                operation=name,
            )

            for qc in input_query_compilers:
                calculator.add_query_compiler(qc)

            if pin_target_backend is None:
                result_backend = calculator.calculate()
            else:
                result_backend = pin_target_backend

            def cast_to_qc(arg):
                if not (
                    isinstance(arg, QueryCompilerCaster)
                    and arg._get_query_compiler() is not None
                    and arg.get_backend() != result_backend
                ):
                    return arg
                cast = arg.set_backend(
                    result_backend,
                    switch_operation=f"{_normalize_class_name(class_of_wrapped_fn)}.{name}",
                )
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

        # `result_backend` may be different from `input_backend`, so we have to
        # look up the correct implementation based on `result_backend`.
        f_to_apply = _get_extension_for_method(
            name=name,
            extensions=extensions,
            backend=result_backend,
            args=args,
            wrapping_function_type=wrapping_function_type,
        )

        # We have to set the global Backend correctly for I/O methods like
        # read_json() to use the correct backend.
        with config_context(Backend=result_backend):
            result = f_to_apply(*args, **kwargs)
        for (
            original_castable,
            original_qc,
            new_castable,
        ) in inplace_update_trackers:
            new_qc = new_castable._get_query_compiler()
            if original_qc is not new_qc:
                new_castable._copy_into(original_castable)

        return _maybe_switch_backend_post_op(
            result,
            function_name=name,
            qc_list=input_query_compilers,
            starting_backend=result_backend,
            class_of_wrapped_fn=class_of_wrapped_fn,
            pin_backend=pin_target_backend is not None,
            arguments=args_dict,
        )

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
            klass=None,
            f=f,
            wrapping_function_type=None,
            extensions=_GENERAL_EXTENSIONS,
            name=name,
        )

    return wrapper


def register_function_for_post_op_switch(
    class_name: Optional[str], backend: str, method: str
) -> None:
    """
    Register a function for post-operation backend switch.

    Parameters
    ----------
    class_name : Optional[str]
        The name of the class that the function belongs to. `None` for functions
        in the modin.pandas module.
    backend : str
        Only consider switching when the starting backend is this one.
    method : str
        The name of the method to register.
    """
    _CLASS_AND_BACKEND_TO_POST_OP_SWITCH_METHODS[
        BackendAndClassName(backend=backend, class_name=class_name)
    ].add(method)


def register_function_for_pre_op_switch(
    class_name: Optional[str], backend: str, method: str
) -> None:
    """
    Register a function for pre-operation backend switch.

    Parameters
    ----------
    class_name : Optional[str]
        The name of the class that the function belongs to. `None` for functions
        in the modin.pandas module.
    backend : str
        Only consider switching when the starting backend is this one.
    method : str
        The name of the method to register.
    """
    _CLASS_AND_BACKEND_TO_PRE_OP_SWITCH_METHODS[
        BackendAndClassName(backend=backend, class_name=class_name)
    ].add(method)
