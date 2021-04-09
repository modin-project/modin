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
import numpy as np

from modin.config import Engine, Backend, IsExperimental


def _inherit_func_docstring(source_func):
    """Define `func` docstring from `source_func`."""

    def decorator(func):
        func.__doc__ = source_func.__doc__
        return func

    return decorator


def _inherit_docstrings(parent, excluded=[], overwrite_existing=False, apilink=None):
    """Creates a decorator which overwrites a decorated class' __doc__
    attribute with parent's __doc__ attribute. Also overwrites __doc__ of
    methods and properties defined in the class with the __doc__ of matching
    methods and properties in parent.

    Parameters
    ----------
        parent : object
            Class from which the decorated class inherits __doc__.
        excluded : list
            List of parent objects from which the class does not
            inherit docstrings.
        overwrite_existing : bool, default: False
            Allow overwriting docstrings that already exist in
            the decorated class
        apilink : str, default: None
            If non-empty, insert the link to Pandas API documentation.
            Could be magic value `'auto'` in which case the link would be auto-generated
            based on parent class using some heuristics.

    Returns
    -------
    callable
        decorator which replaces the decorated class' documentation with parent's documentation
    """

    def _make_doc(pandas_obj):
        """Makes docstring from a parent, Pandas object.

        Adds API link if required, can generate it in 'auto' mode"""
        return pandas_obj.__doc__

    def decorator(cls):
        if parent not in excluded:
            if overwrite_existing or not getattr(cls, "__doc__", ""):
                cls.__doc__ = _make_doc(parent)
        for attr, obj in cls.__dict__.items():
            parent_obj = getattr(parent, attr, None)
            if parent_obj in excluded or (
                not callable(parent_obj) and not isinstance(parent_obj, property)
            ):
                continue
            if not overwrite_existing and getattr(obj, "__doc__", ""):
                # do not overwrite existing docstring unless allowed
                continue
            if callable(obj):
                obj.__doc__ = _make_doc(parent_obj)
            elif isinstance(obj, property) and obj.fget is not None:
                p = property(obj.fget, obj.fset, obj.fdel, _make_doc(parent_obj))
                setattr(cls, attr, p)
        return cls

    return decorator


def to_pandas(modin_obj):
    """Converts a Modin DataFrame/Series to a pandas DataFrame/Series.

    Args:
        obj {modin.DataFrame, modin.Series}: The Modin DataFrame/Series to convert.

    Returns:
        A new pandas DataFrame or Series.
    """
    return modin_obj._to_pandas()


def hashable(obj):
    """Return whether the object is hashable."""
    try:
        hash(obj)
    except TypeError:
        return False
    return True


def try_cast_to_pandas(obj, squeeze=False):
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
        result = obj._to_pandas()
        if squeeze:
            result = result.squeeze(axis=1)
        return result
    if hasattr(obj, "to_pandas"):
        result = obj.to_pandas()
        if squeeze:
            result = result.squeeze(axis=1)
        # Query compiler case, it doesn't have logic about convertion to Series
        if (
            isinstance(getattr(result, "name", None), str)
            and result.name == "__reduced__"
        ):
            result.name = None
        return result
    if isinstance(obj, (list, tuple)):
        return type(obj)([try_cast_to_pandas(o, squeeze=squeeze) for o in obj])
    if isinstance(obj, dict):
        return {k: try_cast_to_pandas(v, squeeze=squeeze) for k, v in obj.items()}
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


def wrap_into_list(*args, skipna=True):
    """
    Creates a list of passed values, if some value is a list it appends its values
    to the result one by one instead inserting the whole list object.

    Parameters
    ----------
    skipna: boolean,
        Whether or not to skip nan or None values.

    Returns
    -------
    List of passed values.
    """

    def isnan(o):
        return o is None or (isinstance(o, float) and np.isnan(o))

    res = []
    for o in args:
        if skipna and isnan(o):
            continue
        if isinstance(o, list):
            res.extend(o)
        else:
            res.append(o)
    return res


def wrap_udf_function(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        # if user accidently returns modin DataFrame or Series
        # casting it back to pandas to properly process
        return try_cast_to_pandas(result)

    wrapper.__name__ = func.__name__
    return wrapper


def get_current_backend():
    return f"{'Experimental' if IsExperimental.get() else ''}{Backend.get()}On{Engine.get()}"
