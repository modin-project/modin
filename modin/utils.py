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

import codecs
import functools
import importlib
import inspect
import json
import os
import re
import sys
import types
import warnings
from pathlib import Path
from textwrap import dedent, indent
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    TypeVar,
    Union,
    runtime_checkable,
)

import numpy as np
import pandas
from packaging import version
from pandas._typing import JSONSerializable
from pandas.util._decorators import Appender  # type: ignore
from pandas.util._print_versions import (  # type: ignore[attr-defined]
    _get_dependency_info,
    _get_sys_info,
)

from modin._version import get_versions
from modin.config import DocModule, Engine, StorageFormat

T = TypeVar("T")
"""Generic type parameter"""

Fn = TypeVar("Fn", bound=Callable)
"""Function type parameter (used in decorators that don't change a function's signature)"""


@runtime_checkable
class SupportsPublicToPandas(Protocol):  # noqa: PR01
    """Structural type for objects with a ``to_pandas`` method (without a leading underscore)."""

    def to_pandas(self) -> Any:  # noqa: GL08
        pass


@runtime_checkable
class SupportsPublicToNumPy(Protocol):  # noqa: PR01
    """Structural type for objects with a ``to_numpy`` method (without a leading underscore)."""

    def to_numpy(self) -> Any:  # noqa: GL08
        pass


@runtime_checkable
class SupportsPrivateToNumPy(Protocol):  # noqa: PR01
    """Structural type for objects with a ``_to_numpy`` method (note the leading underscore)."""

    def _to_numpy(self) -> Any:  # noqa: GL08
        pass


MIN_RAY_VERSION = version.parse("2.1.0")
MIN_DASK_VERSION = version.parse("2.22.0")
MIN_UNIDIST_VERSION = version.parse("0.2.1")

PANDAS_API_URL_TEMPLATE = f"https://pandas.pydata.org/pandas-docs/version/{pandas.__version__}/reference/api/{{}}.html"

# The '__reduced__' name is used internally by the query compiler as a column name to
# represent pandas Series objects that are not explicitly assigned a name, so as to
# distinguish between an N-element series and 1xN dataframe.
MODIN_UNNAMED_SERIES_LABEL = "__reduced__"


def _make_api_url(token: str) -> str:
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
    indents = _get_indents(doc)
    return min(indents) if indents else 0


def _get_indents(source: Union[list, str]) -> list:
    """
    Compute indentation for each line of the source string.

    Parameters
    ----------
    source : str or list of str
        String to compute indents for. Passed list considered
        as a list of lines of the source string.

    Returns
    -------
    list of ints
        List containing computed indents for each line.
    """
    indents = []

    if not isinstance(source, list):
        source = source.splitlines()

    for line in source:
        if not line.strip():
            continue
        for pos, ch in enumerate(line):
            if ch != " ":
                break
        indents.append(pos)
    return indents


def format_string(template: str, **kwargs: str) -> str:
    """
    Insert passed values at the corresponding placeholders of the specified template.

    In contrast with the regular ``str.format()`` this function computes proper
    indents for the placeholder values.

    Parameters
    ----------
    template : str
        Template to substitute values in.
    **kwargs : dict
        Dictionary that maps placeholder names with values.

    Returns
    -------
    str
        Formated string.
    """
    # We want to change indentation only for those values which placeholders are located
    # at the start of the line, in that case the placeholder sets an indentation
    # that the filling value has to obey.
    # RegExp determining placeholders located at the beginning of the line.
    regex = r"^( *)\{(\w+)\}"
    for line in template.splitlines():
        if line.strip() == "":
            continue
        match = re.search(regex, line)
        if match is None:
            continue
        nspaces = len(match.group(1))
        key = match.group(2)

        value = kwargs.get(key)
        if not value:
            continue
        value = dedent(value)

        # Since placeholder is located at the beginning of a new line,
        # it already has '\n' before it, so to avoid double new lines
        # we want to discard the first leading '\n' at the value line,
        # the others leading '\n' are considered as being put on purpose
        if value[0] == "\n":
            value = value[1:]
        # `.splitlines()` doesn't preserve last empty line,
        # so we have to restore it further
        value_lines = value.splitlines()
        # We're not indenting the first line of the value, since it's already indented
        # properly because of the placeholder indentation.
        indented_lines = [
            indent(line, " " * nspaces) if line != "\n" else line
            for line in value_lines[1:]
        ]
        # If necessary, restoring the last line dropped by `.splitlines()`
        if value[-1] == "\n":
            indented_lines += [" " * nspaces]

        indented_value = "\n".join([value_lines[0], *indented_lines])
        kwargs[key] = indented_value

    return template.format(**kwargs)


def align_indents(source: str, target: str) -> str:
    """
    Align indents of two strings.

    Parameters
    ----------
    source : str
        Source string to align indents with.
    target : str
        Target string to align indents.

    Returns
    -------
    str
        Target string with indents aligned with the source.
    """
    source_indent = _get_indent(source)
    target = dedent(target)
    return indent(target, " " * source_indent)


def append_to_docstring(message: str) -> Callable[[Fn], Fn]:
    """
    Create a decorator which appends passed message to the function's docstring.

    Parameters
    ----------
    message : str
        Message to append.

    Returns
    -------
    callable
    """

    def decorator(func: Fn) -> Fn:
        to_append = align_indents(func.__doc__ or "", message)
        return Appender(to_append)(func)

    return decorator


def _replace_doc(
    source_obj: object,
    target_obj: object,
    overwrite: bool,
    apilink: Optional[Union[str, List[str]]],
    parent_cls: Optional[Fn] = None,
    attr_name: Optional[str] = None,
) -> None:
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
    apilink : str | List[str], optional
        If non-empty, insert the link(s) to pandas API documentation.
        Should be the prefix part in the URL template, e.g. "pandas.DataFrame".
    parent_cls : class, optional
        If `target_obj` is an attribute of a class, `parent_cls` should be that class.
        This is used for generating the API URL as well as for handling special cases
        like `target_obj` being a property or a cached_property.
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
    if doc == "":
        # Empty docstrings do not need to be inherited
        return

    if parent_cls and not attr_name:
        if isinstance(target_obj, property):
            attr_name = target_obj.fget.__name__  # type: ignore[union-attr]
        elif isinstance(target_obj, functools.cached_property):
            attr_name = target_obj.func.__name__
        elif isinstance(target_obj, (staticmethod, classmethod)):
            attr_name = target_obj.__func__.__name__
        else:
            attr_name = target_obj.__name__  # type: ignore[attr-defined]

    if (
        source_doc.strip()
        and apilink
        and "pandas API documentation for " not in target_doc
        and (not (attr_name or "").startswith("_"))
    ):
        apilink_l = [apilink] if not isinstance(apilink, list) and apilink else apilink
        links = []
        for link in apilink_l:
            if attr_name:
                token = f"{link}.{attr_name}"
            else:
                token = link
            url = _make_api_url(token)
            links.append(f"`{token} <{url}>`_")

        indent_line = " " * _get_indent(doc)
        notes_section = f"\n{indent_line}Notes\n{indent_line}-----\n"

        url_line = f"{indent_line}See pandas API documentation for {', '.join(links)} for more.\n"
        notes_section_with_url = notes_section + url_line

        if notes_section in doc:
            doc = doc.replace(notes_section, notes_section_with_url)
        else:
            doc += notes_section_with_url

    if parent_cls and isinstance(target_obj, property):
        if overwrite:
            target_obj.fget.__doc_inherited__ = True  # type: ignore[union-attr]
        assert attr_name is not None
        setattr(
            parent_cls,
            attr_name,
            property(target_obj.fget, target_obj.fset, target_obj.fdel, doc),
        )
    elif parent_cls and isinstance(target_obj, functools.cached_property):
        if overwrite:
            target_obj.func.__doc_inherited__ = True  # type: ignore[attr-defined]
        assert attr_name is not None
        target_obj.func.__doc__ = doc
        setattr(
            parent_cls,
            attr_name,
            functools.cached_property(target_obj.func),
        )
        # otherwise: `TypeError: Cannot use cached_property instance without calling __set_name__ on it.`
        getattr(parent_cls, attr_name).__set_name__(parent_cls, attr_name)
    else:
        if overwrite:
            target_obj.__doc_inherited__ = True  # type: ignore[attr-defined]
        target_obj.__doc__ = doc


# This is a map from objects whose docstrings we are overriding to functions that
# take a DocModule string and override the docstring according to the
# DocModule. When we update DocModule, we can use this map to update all
# inherited docstrings.
_docstring_inheritance_calls: list[Callable[[str], None]] = []

# This is a set of (class, attribute_name) pairs whose docstrings we have
# already replaced since we last updated DocModule. Note that we don't store
# the attributes themselves since we replace property attributes instead of
# modifying them in place:
# https://github.com/modin-project/modin/blob/e9dbcc127913db77473a83936e8b6bb94ef84f0d/modin/utils.py#L353
_attributes_with_docstrings_replaced: set[tuple[type, str]] = set()


def _documentable_obj(obj: object) -> bool:
    """
    Check whether we can replace the docstring of `obj`.

    Parameters
    ----------
    obj : object
        Object whose docstring we want to replace.

    Returns
    -------
    bool
        Whether we can replace the docstring.
    """
    return bool(
        callable(obj)
        and not inspect.isclass(obj)
        or (isinstance(obj, property) and obj.fget)
        or (isinstance(obj, functools.cached_property))
        or (isinstance(obj, (staticmethod, classmethod)) and obj.__func__)
    )


def _update_inherited_docstrings(doc_module: DocModule) -> None:
    """
    Update all inherited docstrings.

    Parameters
    ----------
    doc_module : DocModule
        The current DocModule.
    """
    _attributes_with_docstrings_replaced.clear()
    _doc_module = doc_module.get()
    for doc_inheritance_call in _docstring_inheritance_calls:
        doc_inheritance_call(doc_module=_doc_module)  # type: ignore[call-arg]


def _inherit_docstrings_in_place(
    cls_or_func: Fn,
    doc_module: str,
    parent: object,
    excluded: List[object],
    overwrite_existing: bool = False,
    apilink: Optional[Union[str, List[str]]] = None,
) -> None:
    """
    Replace `cls_or_func` docstrings with `parent` docstrings in place.

    Parameters
    ----------
    cls_or_func : Fn
        The class or function whose docstrings we need to update.
    doc_module : str
        The docs module.
    parent : object
        Parent object from which the decorated object inherits __doc__.
    excluded : list, default: []
        List of parent objects from which the class does not
        inherit docstrings.
    overwrite_existing : bool, default: False
        Allow overwriting docstrings that already exist in
        the decorated class.
    apilink : str | List[str], optional
        If non-empty, insert the link(s) to pandas API documentation.
        Should be the prefix part in the URL template, e.g. "pandas.DataFrame".
    """
    # Import the docs module and get the class (e.g. `DataFrame`).
    imported_doc_module = importlib.import_module(doc_module)
    # Set the default parent so we can use it in case some docs are missing from
    # parent module.
    default_parent = parent
    # Try to get the parent object from the doc module, and if it isn't there,
    # get it from parent instead. We only do this if we are overriding pandas
    # documentation. We don't touch other docs.
    if doc_module != DocModule.default and "pandas" in str(
        getattr(parent, "__module__", "")
    ):
        parent_name = (
            # DocModule should use the class BasePandasDataset to override the
            # docstrings of BasePandasDataset, even if BasePandasDataset
            # normally inherits docstrings from a different `parent`.
            "BasePandasDataset"
            if getattr(cls_or_func, "__name__", "") == "BasePandasDataset"
            # For other classes, override docstrings with the class that has the
            # same name as the `parent` class, e.g. DataFrame inherits
            # docstrings from doc_module.DataFrame.
            else getattr(parent, "__name__", "")
        )
        parent = getattr(imported_doc_module, parent_name, parent)
    if parent != default_parent:
        # Reset API link in case the docs are overridden.
        apilink = None
        overwrite_existing = True

    if parent not in excluded:
        _replace_doc(parent, cls_or_func, overwrite_existing, apilink)

    if not isinstance(cls_or_func, types.FunctionType):
        seen = set()
        for base in cls_or_func.__mro__:  # type: ignore[attr-defined]
            if base is object:
                continue
            for attr, obj in base.__dict__.items():
                # only replace docstrings once to prevent https://github.com/modin-project/modin/issues/7113
                if attr in seen or (base, attr) in _attributes_with_docstrings_replaced:
                    continue
                seen.add(attr)
                # Try to get the attribute from the docs class first, then
                # from the default parent (pandas), and if it's not in either,
                # set `parent_obj` to `None`.
                parent_obj = getattr(parent, attr, getattr(default_parent, attr, None))
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
                    parent_cls=base,
                    attr_name=attr,
                )

                _attributes_with_docstrings_replaced.add((base, attr))


def _inherit_docstrings(
    parent: object,
    excluded: List[object] = [],
    overwrite_existing: bool = False,
    apilink: Optional[Union[str, List[str]]] = None,
) -> Callable[[Fn], Fn]:
    """
    Create a decorator which overwrites decorated object docstring(s).

    It takes `parent` __doc__ attribute. Also overwrites __doc__ of
    methods and properties defined in the target or its ancestors if it's a class
    with the __doc__ of matching methods and properties from the `parent`.

    Parameters
    ----------
    parent : object
        Parent object from which the decorated object inherits __doc__.
    excluded : list, default: []
        List of parent objects from which the class does not
        inherit docstrings.
    overwrite_existing : bool, default: False
        Allow overwriting docstrings that already exist in
        the decorated class.
    apilink : str | List[str], optional
        If non-empty, insert the link(s) to pandas API documentation.
        Should be the prefix part in the URL template, e.g. "pandas.DataFrame".

    Returns
    -------
    callable
        Decorator which replaces the decorated object's documentation with `parent` documentation.

    Notes
    -----
    Keep in mind that the function will override docstrings even for attributes which
    are not defined in target class (but are defined in the ancestor class),
    which means that ancestor class attribute docstrings could also change.
    """

    def decorator(cls_or_func: Fn) -> Fn:
        inherit_docstring_in_place = functools.partial(
            _inherit_docstrings_in_place,
            cls_or_func=cls_or_func,
            parent=parent,
            excluded=excluded,
            overwrite_existing=overwrite_existing,
            apilink=apilink,
        )
        inherit_docstring_in_place(doc_module=DocModule.get())
        _docstring_inheritance_calls.append(inherit_docstring_in_place)
        return cls_or_func

    return decorator


DocModule.subscribe(_update_inherited_docstrings)


def expanduser_path_arg(argname: str) -> Callable[[Fn], Fn]:
    """
    Decorate a function replacing its path argument with "user-expanded" value.

    Parameters
    ----------
    argname : str
        Name of the argument which is containing a path to be expanded.

    Returns
    -------
    callable
        Decorator which performs the replacement.
    """

    def decorator(func: Fn) -> Fn:
        signature = inspect.signature(func)
        assert (
            getattr(signature.parameters.get(argname), "name", None) == argname
        ), f"Function {func} does not take '{argname}' as argument"

        @functools.wraps(func)
        def wrapped(*args: tuple, **kw: dict) -> Any:
            params = signature.bind(*args, **kw)
            if patharg := params.arguments.get(argname, None):
                if isinstance(patharg, str) and patharg.startswith("~"):
                    params.arguments[argname] = os.path.expanduser(patharg)
                elif isinstance(patharg, Path):
                    params.arguments[argname] = patharg.expanduser()
                return func(*params.args, **params.kwargs)
            return func(*args, **kw)

        return wrapped  # type: ignore[return-value]

    return decorator


def func_from_deprecated_location(
    func_name: str, module: str, deprecation_message: str
) -> Callable:
    """
    Create a function that decorates a function ``module.func_name`` with a ``FutureWarning``.

    Parameters
    ----------
    func_name : str
        Function name to decorate.
    module : str
        Module where the function is located.
    deprecation_message : str
        Message to print in a future warning.

    Returns
    -------
    callable
    """

    def deprecated_func(*args: tuple[Any], **kwargs: dict[Any, Any]) -> Any:
        """Call deprecated function."""
        func = getattr(importlib.import_module(module), func_name)
        # using 'FutureWarning' as 'DeprecationWarnings' are filtered out by default
        warnings.warn(deprecation_message, FutureWarning)
        return func(*args, **kwargs)

    return deprecated_func


def hashable(obj: bool) -> bool:
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
    # Happy path: if there's no __hash__ method, the object definitely isn't hashable
    if not hasattr(obj, "__hash__"):
        return False
    # Otherwise, we may still need to check for type errors, as in the case of `hash(([],))`.
    # (e.g. an unhashable object inside a tuple)
    try:
        hash(obj)
    except TypeError:
        return False
    return True


def try_cast_to_pandas(obj: Any, squeeze: bool = False) -> Any:
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
    if isinstance(obj, SupportsPublicToPandas) or hasattr(obj, "modin"):
        result = obj.modin.to_pandas() if hasattr(obj, "modin") else obj.to_pandas()
        if squeeze:
            result = result.squeeze(axis=1)

        # QueryCompiler/low-level ModinFrame case, it doesn't have logic about convertion to Series
        if (
            isinstance(getattr(result, "name", None), str)
            and result.name == MODIN_UNNAMED_SERIES_LABEL
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


def execute(*objs: Iterable[Any]) -> None:
    """
    Trigger the lazy computations for each obj in `objs`, if any, and wait for them to complete.

    Parameters
    ----------
    *objs : Iterable[Any]
        A collection of objects to trigger lazy computations.
    """
    for obj in objs:
        if not hasattr(obj, "_query_compiler"):
            continue
        query_compiler = obj._query_compiler
        query_compiler.execute()


def wrap_into_list(*args: Any, skipna: bool = True) -> List[Any]:
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

    def isnan(o: Any) -> bool:
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


def wrap_udf_function(func: Callable) -> Callable:
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

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = func(*args, **kwargs)
        # if user accidently returns modin DataFrame or Series
        # casting it back to pandas to properly process
        return try_cast_to_pandas(result)

    wrapper.__name__ = func.__name__
    return wrapper


def get_current_execution() -> str:
    """
    Return current execution name as a string.

    Returns
    -------
    str
        Returns <StorageFormat>On<Engine>-like string.
    """
    return f"{StorageFormat.get()}On{Engine.get()}"


def instancer(_class: Callable[[], T]) -> T:
    """
    Create a dummy instance each time this is imported.

    This serves the purpose of allowing us to use all of pandas plotting methods
    without aliasing and writing each of them ourselves.

    Parameters
    ----------
    _class : object

    Returns
    -------
    object
        Instance of `_class`.
    """
    return _class()


def import_optional_dependency(name: str, message: str) -> types.ModuleType:
    """
    Import an optional dependecy.

    Parameters
    ----------
    name : str
        The module name.
    message : str
        Additional text to include in the ImportError message.

    Returns
    -------
    module : ModuleType
        The imported module.
    """
    try:
        return importlib.import_module(name)
    except ImportError:
        raise ImportError(
            f"Missing optional dependency '{name}'. {message} "
            + f"Use pip or conda to install {name}."
        ) from None


def _get_modin_deps_info() -> Mapping[str, Optional[JSONSerializable]]:
    """
    Return Modin-specific dependencies information as a JSON serializable dictionary.

    Returns
    -------
    Mapping[str, Optional[pandas.JSONSerializable]]
        The dictionary of Modin dependencies and their versions.
    """
    import modin  # delayed import so modin.__init__ is fully initialized

    result = {"modin": modin.__version__}

    for pkg_name, pkg_version in [
        ("ray", MIN_RAY_VERSION),
        ("dask", MIN_DASK_VERSION),
        ("distributed", MIN_DASK_VERSION),
    ]:
        try:
            pkg = importlib.import_module(pkg_name)
        except ImportError:
            result[pkg_name] = None
        else:
            result[pkg_name] = pkg.__version__ + (
                f" (outdated; >={pkg_version} required)"
                if version.parse(pkg.__version__) < pkg_version
                else ""
            )
    return result


def show_versions(as_json: Union[str, bool] = False) -> None:
    """
    Provide useful information, important for bug reports.

    It comprises info about hosting operation system, pandas version,
    and versions of other installed relative packages.

    Parameters
    ----------
    as_json : str or bool, default: False
        * If False, outputs info in a human readable form to the console.
        * If str, it will be considered as a path to a file.
          Info will be written to that file in JSON format.
        * If True, outputs info in JSON format to the console.

    Notes
    -----
    This is mostly a copy of pandas.show_versions() but adds separate listing
    of Modin-specific dependencies.
    """
    sys_info = _get_sys_info()
    sys_info["commit"] = get_versions()["full-revisionid"]
    modin_deps = _get_modin_deps_info()
    deps = _get_dependency_info()

    if as_json:
        j = {
            "system": sys_info,
            "modin dependencies": modin_deps,
            "dependencies": deps,
        }

        if as_json is True:
            sys.stdout.writelines(json.dumps(j, indent=2))
        else:
            assert isinstance(as_json, str)  # needed for mypy
            with codecs.open(as_json, "wb", encoding="utf8") as f:
                json.dump(j, f, indent=2)

    else:
        assert isinstance(sys_info["LOCALE"], dict)  # needed for mypy
        language_code = sys_info["LOCALE"]["language-code"]
        encoding = sys_info["LOCALE"]["encoding"]
        sys_info["LOCALE"] = f"{language_code}.{encoding}"

        maxlen = max(max(len(x) for x in d) for d in (deps, modin_deps))
        print("\nINSTALLED VERSIONS\n------------------")  # noqa: T201
        for k, v in sys_info.items():
            print(f"{k:<{maxlen}}: {v}")  # noqa: T201
        for name, d in (("Modin", modin_deps), ("pandas", deps)):
            print(f"\n{name} dependencies\n{'-' * (len(name) + 13)}")  # noqa: T201
            for k, v in d.items():
                print(f"{k:<{maxlen}}: {v}")  # noqa: T201


class ModinAssumptionError(Exception):
    """An exception that allows us defaults to pandas if any assumption fails."""

    pass


class classproperty:
    """
    Decorator that allows creating read-only class properties.

    Parameters
    ----------
    func : method

    Examples
    --------
    >>> class A:
    ...     field = 10
    ...     @classproperty
    ...     def field_x2(cls):
    ...             return cls.field * 2
    ...
    >>> print(A.field_x2)
    20
    """

    def __init__(self, func: Any):
        self.fget = func

    def __get__(self, instance: Any, owner: Any) -> Any:  # noqa: GL08
        return self.fget(owner)


def reload_modin() -> None:
    """
    Reload all previously imported Modin modules.

    The call to this function is required
    if an execution engine has been shut down and
    is going to be started up once again.
    """
    modules = sys.modules.copy()
    for name, module in modules.items():
        if name.startswith("modin"):
            importlib.reload(module)
