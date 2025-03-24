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

import inspect
from collections import defaultdict
from functools import wraps
from typing import Any, Callable, Optional

import modin.pandas as pd
from modin.config import Backend

# This type describes a defaultdict that maps backend name (or `None` for
# method implementation and not bound to any one extension) to the dictionary of
# extensions for that backend. The keys of the inner dictionary are the names of
# the extensions, and the values are the extensions themselves.
EXTENSION_DICT_TYPE = defaultdict[Optional[str], dict[str, Any]]

_attrs_to_delete_on_test = defaultdict(list)

_NON_EXTENDABLE_ATTRIBUTES = (
    # we use these attributes to implement the extension system, so we can't
    # allow extensions to override them.
    "__getattribute__",
    "__setattr__",
    "__delattr__",
    "get_backend",
    "set_backend",
    "__getattr__",
    "_get_extension",
    "_getattribute__from_extension_impl",
    "_getattr__from_extension_impl",
    "_query_compiler",
)


def _set_attribute_on_obj(
    name: str, extensions: dict, backend: Optional[str], obj: type
):
    """
    Create a new or override existing attribute on obj.

    Parameters
    ----------
    name : str
        The name of the attribute to assign to `obj`.
    extensions : dict
        The dictionary mapping extension name to `new_attr` (assigned below).
    backend : Optional[str]
        The backend to which the accessor applies. If `None`, this accessor
        will become the default for all backends.
    obj : DataFrame, Series, or modin.pandas
        The object we are assigning the new attribute to.

    Returns
    -------
    decorator
        Returns the decorator function.
    """
    if name in _NON_EXTENDABLE_ATTRIBUTES:
        raise ValueError(f"Cannot register an extension with the reserved name {name}.")

    def decorator(new_attr: Any):
        """
        Decorate a function or class to be assigned to the given name.

        Parameters
        ----------
        new_attr : Any
            The new attribute to assign to name.

        Returns
        -------
        new_attr
            Unmodified new_attr is return from the decorator.
        """
        extensions[None if backend is None else Backend.normalize(backend)][
            name
        ] = new_attr
        if callable(new_attr) and name not in dir(obj):
            # For callable extensions, we add a method to the class that
            # dispatches to the correct implementation.
            setattr(
                obj,
                name,
                wrap_method_in_backend_dispatcher(name, new_attr, extensions),
            )
            _attrs_to_delete_on_test[obj].append(name)
        return new_attr

    return decorator


def register_dataframe_accessor(name: str, *, backend: Optional[str] = None):
    """
    Registers a dataframe attribute with the name provided.

    This is a decorator that assigns a new attribute to DataFrame. It can be used
    with the following syntax:

    ```
    @register_dataframe_accessor("new_method")
    def my_new_dataframe_method(*args, **kwargs):
        # logic goes here
        return
    ```

    The new attribute can then be accessed with the name provided:

    ```
    df.new_method(*my_args, **my_kwargs)
    ```

    Parameters
    ----------
    name : str
        The name of the attribute to assign to DataFrame.

    Returns
    -------
    decorator
        Returns the decorator function.
    backend : Optional[str]
        The backend to which the accessor applies. If ``None``, this accessor
        will become the default for all backends.
    """
    return _set_attribute_on_obj(
        name, pd.dataframe._DATAFRAME_EXTENSIONS_, backend, pd.dataframe.DataFrame
    )


def register_series_accessor(name: str, *, backend: Optional[str] = None):
    """
    Registers a series attribute with the name provided.

    This is a decorator that assigns a new attribute to Series. It can be used
    with the following syntax:

    ```
    @register_series_accessor("new_method")
    def my_new_series_method(*args, **kwargs):
        # logic goes here
        return
    ```

    The new attribute can then be accessed with the name provided:

    ```
    s.new_method(*my_args, **my_kwargs)
    ```

    Parameters
    ----------
    name : str
        The name of the attribute to assign to Series.
    backend : Optional[str]
        The backend to which the accessor applies. If ``None``, this accessor
        will become the default for all backends.

    Returns
    -------
    decorator
        Returns the decorator function.
    """
    return _set_attribute_on_obj(
        name, pd.series._SERIES_EXTENSIONS_, backend=backend, obj=pd.series.Series
    )


def register_base_accessor(name: str, *, backend: Optional[str] = None):
    """
    Register a base attribute with the name provided.

    This is a decorator that assigns a new attribute to BasePandasDataset. It can be used
    with the following syntax:

    ```
    @register_base_accessor("new_method")
    def register_base_accessor(*args, **kwargs):
        # logic goes here
        return
    ```

    The new attribute can then be accessed with the name provided:

    ```
    s.new_method(*my_args, **my_kwargs)
    ```

    Parameters
    ----------
    name : str
        The name of the attribute to assign to BasePandasDataset.
    backend : Optional[str]
        The backend to which the accessor applies. If ``None``, this accessor
        will become the default for all backends.

    Returns
    -------
    decorator
        Returns the decorator function.
    """
    import modin.pandas.base

    return _set_attribute_on_obj(
        name,
        modin.pandas.base._BASE_EXTENSIONS,
        backend=backend,
        obj=modin.pandas.base.BasePandasDataset,
    )


def register_pd_accessor(name: str):
    """
    Registers a pd namespace attribute with the name provided.

    This is a decorator that assigns a new attribute to modin.pandas. It can be used
    with the following syntax:

    ```
    @register_pd_accessor("new_function")
    def my_new_pd_function(*args, **kwargs):
        # logic goes here
        return
    ```

    The new attribute can then be accessed with the name provided:

    ```
    import modin.pandas as pd

    pd.new_method(*my_args, **my_kwargs)
    ```


    Parameters
    ----------
    name : str
        The name of the attribute to assign to modin.pandas.

    Returns
    -------
    decorator
        Returns the decorator function.
    """

    def decorator(new_attr: Any):
        """
        The decorator for a function or class to be assigned to name

        Parameters
        ----------
        new_attr : Any
            The new attribute to assign to name.

        Returns
        -------
        new_attr
            Unmodified new_attr is return from the decorator.
        """
        pd._PD_EXTENSIONS_[name] = new_attr
        setattr(pd, name, new_attr)
        return new_attr

    return decorator


def wrap_method_in_backend_dispatcher(
    name: str, method: Callable, extensions: EXTENSION_DICT_TYPE
) -> Callable:
    """
    Wraps a method to dispatch to the correct backend implementation.

    This function is a wrapper that is used to dispatch to the correct backend
    implementation of a method.

    Parameters
    ----------
    name : str
        The name of the method being wrapped.
    method : Callable
        The method being wrapped.
    extensions : EXTENSION_DICT_TYPE
        The extensions dictionary for the class this method is defined on.

    Returns
    -------
    Callable
        Returns the wrapped function.
    """

    @wraps(method)
    def method_dispatcher(*args, **kwargs):
        if len(args) == 0 and len(kwargs) == 0:
            # Handle some cases like __init__()
            return method(*args, **kwargs)
        # TODO(https://github.com/modin-project/modin/issues/7470): this
        # method may take dataframes and series backed by different backends
        # as input, e.g. if we are here because of a call like
        # `pd.DataFrame().set_backend('python_test').merge(pd.DataFrame().set_backend('pandas'))`.
        # In that case, we should determine which backend to cast to, cast all
        # arguments to that backend, and then choose the appropriate extension
        # method, if it exists.

        # Assume that `self` is the first argument.
        self = args[0]
        remaining_args = args[1:]
        if (
            hasattr(self, "_query_compiler")
            and self.get_backend() in extensions
            and name in extensions[self.get_backend()]
        ):
            # If `self` is using a query compiler whose backend has an
            # extension for this method, use that extension.
            return extensions[self.get_backend()][name](self, *remaining_args, **kwargs)
        else:
            # Otherwise, use the default implementation.
            if name not in extensions[None]:
                raise AttributeError(
                    f"{type(self).__name__} object has no attribute {name}"
                )
            return extensions[None][name](self, *remaining_args, **kwargs)

    return method_dispatcher


def wrap_class_methods_in_backend_dispatcher(
    extensions: EXTENSION_DICT_TYPE,
) -> Callable:
    """
    Get a function that can wrap a class's instance methods so that they dispatch to the correct backend.

    Parameters
    ----------
    extensions : EXTENSION_DICT_TYPE
        The extension dictionary for the class.

    Returns
    -------
    Callable
        The class wrapper.
    """

    def wrap_methods(cls: type):
        # We want to avoid wrapping synonyms like __add__() and add() with
        # different wrappers, so keep a dict mapping methods we've wrapped
        # to their wrapped versions.
        already_seen_to_wrapped: dict[Callable, Callable] = {}
        for method_name, method_value in inspect.getmembers(
            cls, predicate=inspect.isfunction
        ):
            if method_value in already_seen_to_wrapped:
                setattr(cls, method_name, already_seen_to_wrapped[method_value])
                continue
            elif method_name not in _NON_EXTENDABLE_ATTRIBUTES:
                extensions[None][method_name] = method_value
                wrapped = wrap_method_in_backend_dispatcher(
                    method_name, method_value, extensions
                )
                if method_name not in cls.__dict__:
                    # If this class's method comes from a superclass (i.e.
                    # it's not in cls.__dict__), mark it so that
                    # modin.utils._inherit_docstrings knows that the method
                    # must get its docstrings from its superclass.
                    wrapped._wrapped_superclass_method = method_value
                setattr(cls, method_name, wrapped)
                already_seen_to_wrapped[method_value] = getattr(cls, method_name)
        return cls

    return wrap_methods
