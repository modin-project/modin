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

from collections import defaultdict
from types import MethodType, ModuleType
from typing import Any, Optional, Union

import modin.pandas as pd
from modin.config import Backend
from modin.core.storage_formats.pandas.query_compiler_caster import (
    _GENERAL_EXTENSIONS,
    _NON_EXTENDABLE_ATTRIBUTES,
    EXTENSION_DICT_TYPE,
    wrap_function_in_argument_caster,
)

_attrs_to_delete_on_test = defaultdict(list)


def _set_attribute_on_obj(
    name: str,
    extensions: EXTENSION_DICT_TYPE,
    backend: Optional[str],
    obj: Union[type, ModuleType],
):
    """
    Create a new or override existing attribute on obj.

    Parameters
    ----------
    name : str
        The name of the attribute to assign to `obj`.
    extensions : EXTENSION_DICT_TYPE
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
            # For callable extensions, we add a method to `obj`'s namespace that
            # dispatches to the correct implementation.
            setattr(
                obj,
                name,
                wrap_function_in_argument_caster(
                    klass=obj if isinstance(obj, type) else None,
                    f=new_attr,
                    wrapping_function_type=(
                        MethodType if isinstance(obj, type) else None
                    ),
                    extensions=extensions,
                    name=name,
                ),
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
        name, pd.dataframe.DataFrame._extensions, backend, pd.dataframe.DataFrame
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
        name, pd.series.Series._extensions, backend=backend, obj=pd.series.Series
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
    from modin.pandas.base import BasePandasDataset

    return _set_attribute_on_obj(
        name,
        BasePandasDataset._extensions,
        backend=backend,
        obj=BasePandasDataset,
    )


def register_pd_accessor(name: str, *, backend: Optional[str] = None):
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
    backend : Optional[str]
        The backend to which the accessor applies. If ``None``, this accessor
        will become the default for all backends.

    Returns
    -------
    decorator
        Returns the decorator function.
    """
    return _set_attribute_on_obj(
        name=name, extensions=_GENERAL_EXTENSIONS, backend=backend, obj=pd
    )


def __getattr___impl(name: str):
    """
    Override __getatttr__ on the modin.pandas module to enable extensions.

    Note that python only falls back to this function if the attribute is not
    found in this module's namespace.

    Parameters
    ----------
    name : str
        The name of the attribute being retrieved.

    Returns
    -------
    Attribute
        Returns the extension attribute, if it exists, otherwise returns the attribute
        imported in this file.
    """

    from modin.config import Backend

    backend = Backend.get()
    if name in _GENERAL_EXTENSIONS[backend]:
        return _GENERAL_EXTENSIONS[backend][name]
    elif name in _GENERAL_EXTENSIONS[None]:
        return _GENERAL_EXTENSIONS[None][name]
    else:
        raise AttributeError(f"module 'modin.pandas' has no attribute '{name}'")
