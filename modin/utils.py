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

import types

import pandas
import numpy as np

from modin.config import Engine, Backend, IsExperimental, DocstringUrlTestMode

PANDAS_API_URL_TEMPLATE = f"https://pandas.pydata.org/pandas-docs/version/{pandas.__version__}/reference/api/{{}}.html"

if DocstringUrlTestMode.get():
    # a list which contains all generated urls so they could be checked for being valid
    _GENERATED_URLS = []


def _replace_doc(
    source_obj, target_obj, overwrite, apilink, parent_cls=None, attr_name=None
):
    """Replaces docstring in `target_obj`, possibly taking from `source_obj` and augmenting.

    Can append the link to Pandas API online documentation.

    Parameters
    ----------
    source_obj : object
        Any object from which to take docstring from.
    target_obj : object
        The object which docstring to replace
    overwrite : bool
        Forces replacing the docstring with the one from `source_obj` even
        if `target_obj` has its own non-empty docstring
    apilink : str
        If non-empty, insert the link to Pandas API documentation.
        Should be the prefix part in the URL template, e.g. "pandas.DataFrame".
    parent_cls : class, optional
        If `target_obj` is an attribute of a class, `parent_cls` should be that class.
        This is used for generating the API URL as well as for handling special cases
        like `target_obj` being a property.
    attr_name : str, optional
        Gives the name to `target_obj` if it's an attribute of `parent_cls`.
        Needed to handle some special cases and in most cases could be determined automatically.
    """
    source_doc = source_obj.__doc__ or ""
    target_doc = target_obj.__doc__ or ""
    doc = source_doc if overwrite or not target_doc else target_doc

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
        and "Pandas API documentation <" not in target_doc
        and (not (attr_name or "").startswith("_"))
    ):
        if attr_name:
            token = f"{apilink}.{attr_name}"
        else:
            token = apilink
        url = PANDAS_API_URL_TEMPLATE.format(token)
        if DocstringUrlTestMode.get():
            _GENERATED_URLS.append(url)
        doc += f"\n\nSee `Pandas API documentation <{url}>`_ for more."

    if parent_cls and isinstance(target_obj, property):
        setattr(
            parent_cls,
            target_obj.fget.__name__,
            property(target_obj.fget, target_obj.fset, target_obj.fdel, doc),
        )
    else:
        target_obj.__doc__ = doc


def _inherit_docstrings(parent, excluded=[], overwrite_existing=False, apilink=None):
    """Creates a decorator which overwrites a decorated object __doc__
    attribute with parent's __doc__ attribute. Also overwrites __doc__ of
    methods and properties defined in the target if it's a class with the __doc__ of
    matching methods and properties in parent.

    Parameters
    ----------
        parent : object
            Parent object from which the decorated object inherits __doc__.
        excluded : list
            List of parent objects from which the class does not
            inherit docstrings.
        overwrite_existing : bool, default: False
            Allow overwriting docstrings that already exist in
            the decorated class
        apilink : str, default: None
            If non-empty, insert the link to Pandas API documentation.
            Should be the prefix part in the URL template, e.g. "pandas.DataFrame".

    Returns
    -------
    callable
        decorator which replaces the decorated object's documentation with parent's documentation.
    """

    def _documentable_obj(obj):
        """Check if `obj` docstring could be patched."""
        return callable(obj) or (isinstance(obj, property) and obj.fget)

    def decorator(cls_or_func):
        if parent not in excluded:
            _replace_doc(parent, cls_or_func, overwrite_existing, apilink)

        if not isinstance(cls_or_func, types.FunctionType):
            for base in cls_or_func.__bases__:
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
