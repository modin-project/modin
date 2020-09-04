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

import pandas


def from_non_pandas(df, index, columns, dtype):
    from modin.data_management.dispatcher import EngineDispatcher

    new_qc = EngineDispatcher.from_non_pandas(df, index, columns, dtype)
    if new_qc is not None:
        from .dataframe import DataFrame

        return DataFrame(query_compiler=new_qc)
    return new_qc


def from_pandas(df):
    """Converts a pandas DataFrame to a Modin DataFrame.
    Args:
        df (pandas.DataFrame): The pandas DataFrame to convert.

    Returns:
        A new Modin DataFrame object.
    """
    from modin.data_management.dispatcher import EngineDispatcher
    from .dataframe import DataFrame

    return DataFrame(query_compiler=EngineDispatcher.from_pandas(df))


def from_arrow(at):
    """Converts an Arrow Table to a Modin DataFrame.

    Parameters
    ----------
        at : Arrow Table
            The Arrow Table to convert from.

    Returns
    -------
    DataFrame
        A new Modin DataFrame object.
    """
    from modin.data_management.dispatcher import EngineDispatcher
    from .dataframe import DataFrame

    return DataFrame(query_compiler=EngineDispatcher.from_arrow(at))


def to_pandas(modin_obj):
    """Converts a Modin DataFrame/Series to a pandas DataFrame/Series.

    Args:
        obj {modin.DataFrame, modin.Series}: The Modin DataFrame/Series to convert.

    Returns:
        A new pandas DataFrame or Series.
    """
    return modin_obj._to_pandas()


def _inherit_docstrings(parent, excluded=[]):
    """Creates a decorator which overwrites a decorated class' __doc__
    attribute with parent's __doc__ attribute. Also overwrites __doc__ of
    methods and properties defined in the class with the __doc__ of matching
    methods and properties in parent.

    Args:
        parent (object): Class from which the decorated class inherits __doc__.
        excluded (list): List of parent objects from which the class does not
            inherit docstrings.

    Returns:
        function: decorator which replaces the decorated class' documentation
            parent's documentation.
    """

    def decorator(cls):
        if parent not in excluded:
            cls.__doc__ = parent.__doc__
        for attr, obj in cls.__dict__.items():
            parent_obj = getattr(parent, attr, None)
            if parent_obj in excluded or (
                not callable(parent_obj) and not isinstance(parent_obj, property)
            ):
                continue
            if callable(obj):
                obj.__doc__ = parent_obj.__doc__
            elif isinstance(obj, property) and obj.fget is not None:
                p = property(obj.fget, obj.fset, obj.fdel, parent_obj.__doc__)
                setattr(cls, attr, p)
        return cls

    return decorator


def try_cast_to_pandas(obj):
    """
    Converts obj and all nested objects from modin to pandas if it is possible,
    otherwise returns obj

    Parameters
    ----------
        obj : object,
            object to convert from modin to pandas

    Returns
    -------
        Converted object
    """
    if hasattr(obj, "_to_pandas"):
        return obj._to_pandas()
    if isinstance(obj, (list, tuple)):
        return type(obj)([try_cast_to_pandas(o) for o in obj])
    if isinstance(obj, dict):
        return {k: try_cast_to_pandas(v) for k, v in obj.items()}
    if callable(obj):
        module_hierarchy = getattr(obj, "__module__", "").split(".")
        fn_name = getattr(obj, "__name__", None)
        if fn_name and module_hierarchy[0] == "modin":
            return (
                getattr(pandas.DataFrame, fn_name, obj)
                if module_hierarchy[-1] == "dataframe"
                else getattr(pandas.Series, fn_name, obj)
            )
    return obj


def wrap_udf_function(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        # if user accidently returns modin DataFrame or Series
        # casting it back to pandas to properly process
        return try_cast_to_pandas(result)

    wrapper.__name__ = func.__name__
    return wrapper


def hashable(obj):
    """Return whether the object is hashable."""
    try:
        hash(obj)
    except TypeError:
        return False
    return True


def is_scalar(obj):
    """
    Return True if given object is scalar.

    This method wrks the same as is_scalar method from Pandas but
    it is optimized for Modin frames. For BasePandasDataset objects
    Pandas version of is_scalar tries to access missing attribute
    causing index scan. This tiggers execution for lazy frames and
    we avoid it by handling BasePandasDataset objects separately.

    Parameters
    ----------
    val : object
        Object to check.

    Returns
    -------
    bool
        True if given object is scalar and False otherwise.
    """

    from pandas.api.types import is_scalar as pandas_is_scalar
    from .base import BasePandasDataset

    return not isinstance(obj, BasePandasDataset) and pandas_is_scalar(obj)
