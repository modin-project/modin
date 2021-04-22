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

"""Collection of general utility functions, mostly for internal use."""

import types

import pandas
import numpy as np

from modin.config import Engine, Backend, IsExperimental

PANDAS_API_URL_TEMPLATE = f"https://pandas.pydata.org/pandas-docs/version/{pandas.__version__}/reference/api/{{}}.html"


def _make_api_url(token):
    """
    Generate the link to pandas documentation.

    Parameters
    ----------
    token : str
        Part of URL to use for generation.

    Returns
    -------
    str
        URL to pandas doc.

    Notes
    -----
    This function is extracted for better testability.
    """
    return PANDAS_API_URL_TEMPLATE.format(token)


def _get_indent(doc: str) -> int:
    """
    Compute indentation in docstring.

    Parameters
    ----------
    doc : str
        The docstring to compute indentation for.

    Returns
    -------
    int
        Minimal indent (excluding empty lines).
    """
    indents = []
    for line in doc.splitlines():
        if not line.strip():
            continue
        for pos, ch in enumerate(line):
            if ch != " ":
                break
        indents.append(pos)
    return min(indents) if indents else 0


def _replace_doc(
    source_obj, target_obj, overwrite, apilink, parent_cls=None, attr_name=None
):
    """
    Replace docstring in `target_obj`, possibly taking from `source_obj` and augmenting.

    Can append the link to pandas API online documentation.

    Parameters
    ----------
    source_obj : object
        Any object from which to take docstring from.
    target_obj : object
        The object which docstring to replace.
    overwrite : bool
        Forces replacing the docstring with the one from `source_obj` even
        if `target_obj` has its own non-empty docstring.
    apilink : str
        If non-empty, insert the link to pandas API documentation.
        Should be the prefix part in the URL template, e.g. "pandas.DataFrame".
    parent_cls : class, optional
        If `target_obj` is an attribute of a class, `parent_cls` should be that class.
        This is used for generating the API URL as well as for handling special cases
        like `target_obj` being a property.
    attr_name : str, optional
        Gives the name to `target_obj` if it's an attribute of `parent_cls`.
        Needed to handle some special cases and in most cases could be determined automatically.
    """
    if isinstance(target_obj, (staticmethod, classmethod)):
        # we cannot replace docs on decorated objects, we must replace them
        # on original functions instead
        target_obj = target_obj.__func__

    source_doc = source_obj.__doc__ or ""
    target_doc = target_obj.__doc__ or ""
    overwrite = overwrite or not target_doc
    doc = source_doc if overwrite else target_doc

    if parent_cls and not attr_name:
        if isinstance(target_obj, property):
            attr_name = target_obj.fget.__name__
        elif isinstance(target_obj, (staticmethod, classmethod)):
            attr_name = target_obj.__func__.__name__
        else:
            attr_name = target_obj.__name__

    if (
        source_doc.strip()
        and apilink
        and "`pandas API documentation for " not in target_doc
        and (not (attr_name or "").startswith("_"))
    ):
        if attr_name:
            token = f"{apilink}.{attr_name}"
        else:
            token = apilink
        url = _make_api_url(token)
        doc += f"\n{' ' * _get_indent(doc)}See `pandas API documentation for {token} <{url}>`_ for more.\n"

    if parent_cls and isinstance(target_obj, property):
        if overwrite:
            target_obj.fget.__doc_inherited__ = True
        setattr(
            parent_cls,
            attr_name,
            property(target_obj.fget, target_obj.fset, target_obj.fdel, doc),
        )
    else:
        if overwrite:
            target_obj.__doc_inherited__ = True
        target_obj.__doc__ = doc


def _inherit_docstrings(parent, excluded=[], overwrite_existing=False, apilink=None):
    """
    Create a decorator which overwrites decorated object docstring(s).

    It takes `parent` __doc__ attribute. Also overwrites __doc__ of
    methods and properties defined in the target if it's a class with the __doc__ of
    matching methods and properties from the `parent`.

    Parameters
    ----------
    parent : object
        Parent object from which the decorated object inherits __doc__.
    excluded : list, optional
        List of parent objects from which the class does not
        inherit docstrings.
    overwrite_existing : bool, default: False
        Allow overwriting docstrings that already exist in
        the decorated class.
    apilink : str, default: None
        If non-empty, insert the link to pandas API documentation.
        Should be the prefix part in the URL template, e.g. "pandas.DataFrame".

    Returns
    -------
    callable
        Decorator which replaces the decorated object's documentation with `parent` documentation.
    """

    def _documentable_obj(obj):
        """Check if `obj` docstring could be patched."""
        return (
            callable(obj)
            or (isinstance(obj, property) and obj.fget)
            or (isinstance(obj, (staticmethod, classmethod)) and obj.__func__)
        )

    def decorator(cls_or_func):
        if parent not in excluded:
            _replace_doc(parent, cls_or_func, overwrite_existing, apilink)

        if not isinstance(cls_or_func, types.FunctionType):
            for base in cls_or_func.__mro__:
                if base is object:
                    continue
                for attr, obj in base.__dict__.items():
                    parent_obj = getattr(parent, attr, None)
                    if (
                        parent_obj in excluded
                        or not _documentable_obj(parent_obj)
                        or not _documentable_obj(obj)
                    ):
                        continue

                    _replace_doc(
                        parent_obj,
                        obj,
                        overwrite_existing,
                        apilink,
                        parent_cls=cls_or_func,
                        attr_name=attr,
                    )

        return cls_or_func

    return decorator


def to_pandas(modin_obj):
    """
    Convert a Modin DataFrame/Series to a pandas DataFrame/Series.

    Parameters
    ----------
    modin_obj : modin.DataFrame, modin.Series
        The Modin DataFrame/Series to convert.

    Returns
    -------
    pandas.DataFrame or pandas.Series
        Converted object with type depending on input.
    """
    return modin_obj._to_pandas()


def hashable(obj):
    """
    Return whether the `obj` is hashable.

    Parameters
    ----------
    obj : object
        The object to check.

    Returns
    -------
    bool
    """
    try:
        hash(obj)
    except TypeError:
        return False
    return True


def try_cast_to_pandas(obj, squeeze=False):
    """
    Convert `obj` and all nested objects from Modin to pandas if it is possible.

    If no convertion possible return `obj`.

    Parameters
    ----------
    obj : object
        Object to convert from Modin to pandas.
    squeeze : bool, default: False
        Squeeze the converted object(s) before returning them.

    Returns
    -------
    object
        Converted object.
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
    Wrap a sequence of passed values in a flattened list.

    If some value is a list by itself the function appends its values
    to the result one by one instead inserting the whole list object.

    Parameters
    ----------
    *args : tuple
        Objects to wrap into a list.
    skipna : bool, default: True
        Whether or not to skip nan or None values.

    Returns
    -------
    list
        Passed values wrapped in a list.
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
    """
    Create a decorator that makes `func` return pandas objects instead of Modin.

    Parameters
    ----------
    func : callable
        Function to wrap.

    Returns
    -------
    callable
    """

    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        # if user accidently returns modin DataFrame or Series
        # casting it back to pandas to properly process
        return try_cast_to_pandas(result)

    wrapper.__name__ = func.__name__
    return wrapper


def get_current_backend():
    """
    Compose current backend and factory names as a string.

    Returns
    -------
    str
        Returns <Backend>On<Engine>-like string.
    """
    return f"{'Experimental' if IsExperimental.get() else ''}{Backend.get()}On{Engine.get()}"
