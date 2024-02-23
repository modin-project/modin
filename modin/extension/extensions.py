from typing import Any, Union

import modin.pandas
from modin.pandas import _PD_EXTENSIONS_
from modin.pandas.dataframe import _DATAFRAME_EXTENSIONS_, DataFrame
from modin.pandas.series import _SERIES_EXTENSIONS_, Series


def _set_attribute_on_obj(
    name: str, extensions_dict: dict, obj: Union[DataFrame, Series, modin.pandas]
):
    """
    Create a new or override existing attribute on obj.

    Parameters
    ----------
    name : str
        The name of the attribute to assign to `obj`.
    extensions_dict : dict
        The dictionary mapping extension name to `new_attr` (assigned below).
    obj : DataFrame, Series, or modin.pandas
        The object we are assigning the new attribute to.

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
        extensions_dict[name] = new_attr
        setattr(obj, name, new_attr)
        return new_attr

    return decorator


def register_dataframe_accessor(name: str):
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
    """
    return _set_attribute_on_obj(name, _DATAFRAME_EXTENSIONS_, DataFrame)


def register_series_accessor(name: str):
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

    Returns
    -------
    decorator
        Returns the decorator function.
    """
    return _set_attribute_on_obj(name, _SERIES_EXTENSIONS_, Series)


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
    return _set_attribute_on_obj(name, _PD_EXTENSIONS_, modin.pandas)
