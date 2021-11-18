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
Validate docstrings using pydocstyle and numpydoc.

Example usage:
python scripts/doc_checker.py asv_bench/benchmarks/utils.py modin/pandas
"""

import argparse
import pathlib
import subprocess
import os
import re
import ast
from typing import List
import sys
import inspect
import shutil
import logging
import functools
from numpydoc.validate import Docstring
from numpydoc.docscrape import NumpyDocString

import types

# fake cuDF-related modules if they're missing
for mod_name in ("cudf", "cupy"):
    try:
        __import__(mod_name)
    except ImportError:
        sys.modules[mod_name] = types.ModuleType(
            mod_name, f"fake {mod_name} for checking docstrings"
        )
if not hasattr(sys.modules["cudf"], "DataFrame"):
    sys.modules["cudf"].DataFrame = type("DataFrame", (object,), {})

logging.basicConfig(
    stream=sys.stdout, format="%(levelname)s:%(message)s", level=logging.INFO
)

MODIN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, MODIN_PATH)

# error codes that pandas test in CI
# https://numpydoc.readthedocs.io/en/latest/validation.html#built-in-validation-checks
NUMPYDOC_BASE_ERROR_CODES = {
    *("GL01", "GL02", "GL03", "GL05", "GL06", "GL07", "GL08", "GL09", "GL10"),
    *("SS02", "SS03", "SS04", "SS05", "PR01", "PR02", "PR03", "PR04", "PR05"),
    *("PR08", "PR09", "PR10", "RT01", "RT04", "RT05", "SA02", "SA03"),
}

MODIN_ERROR_CODES = {
    "MD01": "'{parameter}' description should be '[type], default: [value]', found: '{found}'",
    "MD02": "Spelling error in line: {line}, found: '{word}', reference: '{reference}'",
    "MD03": "Section contents is over-indented (in section '{section}')",
}


def get_optional_args(doc: Docstring) -> dict:
    """
    Get optional parameters for the object for which the docstring is checked.

    Parameters
    ----------
    doc : numpydoc.validate.Docstring
        Docstring handler.

    Returns
    -------
    dict
        Dict with default argument names and its values.
    """
    obj = doc.obj
    if not callable(obj) or inspect.isclass(obj):
        return {}
    signature = inspect.signature(obj)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def check_optional_args(doc: Docstring) -> list:
    """
    Check type description of optional arguments.

    Parameters
    ----------
    doc : numpydoc.validate.Docstring

    Returns
    -------
    list
        List of tuples with Modin error code and its description.
    """
    if not doc.doc_parameters:
        return []
    optional_args = get_optional_args(doc)
    if not optional_args:
        return []

    errors = []
    for parameter in optional_args:
        # case when not all parameters are listed in "Parameters" section;
        # it's handled by numpydoc itself
        if parameter not in doc.doc_parameters:
            continue
        type_line = doc.doc_parameters[parameter][0]
        has_default = "default: " in type_line
        has_optional = "optional" in type_line
        if not (has_default ^ has_optional):
            errors.append(
                (
                    "MD01",
                    MODIN_ERROR_CODES["MD01"].format(
                        parameter=parameter,
                        found=type_line,
                    ),
                )
            )
    return errors


def check_spelling_words(doc: Docstring) -> list:
    """
    Check spelling of chosen words in doc.

    Parameters
    ----------
    doc : numpydoc.validate.Docstring
        Docstring handler.

    Returns
    -------
    list
        List of tuples with Modin error code and its description.

    Notes
    -----
    Any special words enclosed in apostrophes(") are treated as python string
    constants and are not checked for spelling.
    """
    if not doc.raw_doc:
        return []
    components = set(
        ["Modin", "pandas", "NumPy", "Ray", "Dask"]
        + ["PyArrow", "OmniSci", "XGBoost", "Plasma"]
    )
    check_words = "|".join(x.lower() for x in components)

    # comments work only with re.VERBOSE
    pattern = r"""
    (?:                     # non-capturing group
        [^-\\\w\/]          # any symbol except: '-', '\', '/' and any from [a-zA-Z0-9_]
        | ^                 # or line start
    )
    ({check_words})         # words to check, example - "modin|pandas|numpy"
    (?:                     # non-capturing group
        [^-"\.\/\w\\]       # any symbol except: '-', '"', '.', '\', '/' and any from [a-zA-Z0-9_]
        | \.\s              # or '.' and any whitespace
        | \.$               # or '.' and line end
        | $                 # or line end
    )
    """.format(
        check_words=check_words
    )
    results = [
        set(re.findall(pattern, line, re.I | re.VERBOSE)) - components
        for line in doc.raw_doc.splitlines()
    ]

    docstring_start_line = None
    for idx, line in enumerate(inspect.getsourcelines(doc.code_obj)[0]):
        if '"""' in line or "'''" in line:
            docstring_start_line = doc.source_file_def_line + idx
            break

    errors = []
    for line_idx, words_in_line in enumerate(results):
        for word in words_in_line:
            reference = [x for x in components if x.lower() == word.lower()][0]
            errors.append(
                (
                    "MD02",
                    MODIN_ERROR_CODES["MD02"].format(
                        line=docstring_start_line + line_idx,
                        word=word,
                        reference=reference,
                    ),
                )
            )
    return errors


def check_docstring_indention(doc: Docstring) -> list:
    """
    Check indention of docstring since numpydoc reports weird results.

    Parameters
    ----------
    doc : numpydoc.validate.Docstring
        Docstring handler.

    Returns
    -------
    list
        List of tuples with Modin error code and its description.
    """
    from modin.utils import _get_indent

    numpy_docstring = NumpyDocString(doc.clean_doc)
    numpy_docstring._doc.reset()
    numpy_docstring._parse_summary()
    sections = list(numpy_docstring._read_sections())
    errors = []
    for section in sections:
        description = "\n".join(section[1])
        if _get_indent(description) != 0:
            errors.append(
                ("MD03", MODIN_ERROR_CODES["MD03"].format(section=section[0]))
            )
    return errors


def validate_modin_error(doc: Docstring, results: dict) -> list:
    """
    Validate custom Modin errors.

    Parameters
    ----------
    doc : numpydoc.validate.Docstring
        Docstring handler.
    results : dict
        Dictionary that numpydoc.validate.validate return.

    Returns
    -------
    dict
        Updated dict with Modin custom errors.
    """
    errors = check_optional_args(doc)
    errors += check_spelling_words(doc)
    errors += check_docstring_indention(doc)
    results["errors"].extend(errors)
    return results


def skip_check_if_noqa(doc: Docstring, err_code: str, noqa_checks: list) -> bool:
    """
    Skip the check that matches `err_code` if `err_code` found in noqa string.

    Parameters
    ----------
    doc : numpydoc.validate.Docstring
        Docstring handler.
    err_code : str
        Error code found by numpydoc.
    noqa_checks : list
        Found noqa checks.

    Returns
    -------
    bool
        Return True if 'noqa' found.
    """
    if noqa_checks == ["all"]:
        return True

    # GL08 - missing docstring in an arbitary object; numpydoc code
    if err_code == "GL08":
        name = doc.name.split(".")[-1]
        # Numpydoc recommends to add docstrings of __init__ method in class docstring.
        # So there is no error if docstring is missing in __init__
        if name == "__init__":
            return True
    return err_code in noqa_checks


def get_noqa_checks(doc: Docstring) -> list:
    """
    Get codes after `# noqa`.

    Parameters
    ----------
    doc : numpydoc.validate.Docstring
        Docstring handler.

    Returns
    -------
    list
        List with codes.

    Notes
    -----
    If noqa doesn't have any codes - returns ["all"].
    """
    source = doc.method_source
    if not source:
        return []

    noqa_str = ""
    if not inspect.ismodule(doc.obj):
        # find last line of obj definition
        for line in source.split("\n"):
            if ")" in line and ":" in line.split(")", 1)[1]:
                noqa_str = line
                break
    else:
        # noqa string is defined as the first line before the docstring
        if not doc.raw_doc:
            # noqa string is meaningless if there is no docstring in module
            return []
        lines = source.split("\n")
        for idx, line in enumerate(lines):
            if '"""' in line or "'''" in line:
                noqa_str = lines[idx - 1]
                break

    if "# noqa:" in noqa_str:
        noqa_checks = noqa_str.split("# noqa:", 1)[1].split(",")
    elif "# noqa" in noqa_str:
        noqa_checks = ["all"]
    else:
        noqa_checks = []
    return [check.strip() for check in noqa_checks]


# code snippet from numpydoc
def validate_object(import_path: str) -> list:
    """
    Check docstrings of an entity that can be imported.

    Parameters
    ----------
    import_path : str
        Python-like import path.

    Returns
    -------
    errors : list
        List with string representations of errors.
    """
    from numpydoc.validate import validate

    errors = []
    doc = Docstring(import_path)
    if getattr(doc.obj, "__doc_inherited__", False) or (
        isinstance(doc.obj, property)
        and getattr(doc.obj.fget, "__doc_inherited__", False)
    ):
        # do not check inherited docstrings
        return errors
    results = validate(import_path)
    results = validate_modin_error(doc, results)
    noqa_checks = get_noqa_checks(doc)
    for err_code, err_desc in results["errors"]:
        if (
            err_code not in NUMPYDOC_BASE_ERROR_CODES
            and err_code not in MODIN_ERROR_CODES
        ) or skip_check_if_noqa(doc, err_code, noqa_checks):
            continue
        errors.append(
            ":".join([import_path, str(results["file_line"]), err_code, err_desc])
        )
    return errors


def numpydoc_validate(path: pathlib.Path) -> bool:
    """
    Perform numpydoc checks.

    Parameters
    ----------
    path : pathlib.Path
        Filename or directory path for check.

    Returns
    -------
    is_successfull : bool
        Return True if all checks are successful.
    """
    is_successfull = True

    if path.is_file():
        walker = ((str(path.parent), [], [path.name]),)
    else:
        walker = os.walk(path)

    for root, _, files in walker:
        if "__pycache__" in root:
            continue
        for _file in files:
            if not _file.endswith(".py"):
                continue

            current_path = os.path.join(root, _file)
            # get importable name
            module_name = current_path.replace("/", ".").replace("\\", ".")
            # remove ".py"
            module_name = os.path.splitext(module_name)[0]

            with open(current_path) as fd:
                file_contents = fd.read()

            # using static parsing for collecting module, functions, classes and their methods
            module = ast.parse(file_contents)

            def is_public_func(node):
                return isinstance(node, ast.FunctionDef) and (
                    not node.name.startswith("__") or node.name.endswith("__")
                )

            functions = [node for node in module.body if is_public_func(node)]
            classes = [node for node in module.body if isinstance(node, ast.ClassDef)]
            methods = [
                f"{module_name}.{_class.name}.{node.name}"
                for _class in classes
                for node in _class.body
                if is_public_func(node)
            ]

            # numpydoc docstrings validation
            # docstrings are taken dynamically
            to_validate = (
                [module_name]
                + [f"{module_name}.{x.name}" for x in (functions + classes)]
                + methods
            )
            results = list(map(validate_object, to_validate))
            is_successfull_file = not any(results)
            if not is_successfull_file:
                logging.info(f"NUMPYDOC OUTPUT FOR {current_path}")
            [logging.error(error) for errors in results for error in errors]
            is_successfull &= is_successfull_file
    return is_successfull


def pydocstyle_validate(
    path: pathlib.Path, add_ignore: List[str], use_numpydoc: bool
) -> int:
    """
    Perform pydocstyle checks.

    Parameters
    ----------
    path : pathlib.Path
        Filename or directory path for check.
    add_ignore : List[int]
        `pydocstyle` error codes which are not verified.
    use_numpydoc : bool
        Disable duplicate `pydocstyle` checks if `numpydoc` is in use.

    Returns
    -------
    bool
        Return True if all pydocstyle checks are successful.
    """
    pydocstyle = "pydocstyle"
    if not shutil.which(pydocstyle):
        raise ValueError(f"{pydocstyle} not found in PATH")
    # These check can be done with numpydoc tool, so disable them for pydocstyle.
    if use_numpydoc:
        add_ignore.extend(["D100", "D101", "D102", "D103", "D104", "D105"])
    result = subprocess.run(
        [
            pydocstyle,
            "--convention",
            "numpy",
            "--add-ignore",
            ",".join(add_ignore),
            str(path),
        ],
        text=True,
        capture_output=True,
    )
    if result.returncode:
        logging.info(f"PYDOCSTYLE OUTPUT FOR {path}")
        logging.error(result.stdout)
    return True if result.returncode == 0 else False


def monkeypatching():
    """Monkeypatch not installed modules and decorators which change __doc__ attribute."""
    import ray
    import modin.utils
    from unittest.mock import Mock

    def monkeypatch(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            # This is the case where the decorator is just @ray.remote without parameters.
            return args[0]
        return lambda cls_or_func: cls_or_func

    ray.remote = monkeypatch

    # We are mocking packages we don't need for docs checking in order to avoid import errors
    sys.modules["pyarrow.gandiva"] = Mock()
    sys.modules["sqlalchemy"] = Mock()

    modin.utils.instancer = functools.wraps(modin.utils.instancer)(lambda cls: cls)

    # monkey-patch numpydoc for working correctly with properties
    def load_obj(name, old_load_obj=Docstring._load_obj):
        obj = old_load_obj(name)
        if isinstance(obj, property):
            obj = obj.fget
        return obj

    Docstring._load_obj = staticmethod(load_obj)

    # for testing omnisci-engine docs without `dbe` installation
    sys.modules["dbe"] = Mock()
    # enable docs testing on windows
    sys.getdlopenflags = Mock()
    sys.setdlopenflags = Mock()


def validate(
    paths: List[pathlib.Path], add_ignore: List[str], use_numpydoc: bool
) -> bool:
    """
    Perform pydocstyle and numpydoc checks.

    Parameters
    ----------
    paths : List[pathlib.Path]
        Filenames of directories for check.
    add_ignore : List[str]
        `pydocstyle` error codes which are not verified.
    use_numpydoc : bool
        Determine if numpydoc checks are needed.

    Returns
    -------
    is_successfull : bool
        Return True if all checks are successful.
    """
    is_successfull = True
    for path in paths:
        if not pydocstyle_validate(path, add_ignore, use_numpydoc):
            is_successfull = False
        if use_numpydoc:
            if not numpydoc_validate(path):
                is_successfull = False
    return is_successfull


def check_args(args: argparse.Namespace):
    """
    Check the obtained values for correctness.

    Parameters
    ----------
    args : argparse.Namespace
        Parser arguments.

    Raises
    ------
    ValueError
        Occurs in case of non-existent files or directories.
    """
    for path in args.paths:
        if not path.exists():
            raise ValueError(f"{path} does not exist")
        abs_path = os.path.abspath(path)
        if not abs_path.startswith(MODIN_PATH):
            raise ValueError(
                f"it is unsupported to use this script on files from another "
                f"repository; script' repo '{MODIN_PATH}', "
                f"input path '{abs_path}'"
            )


def get_args() -> argparse.Namespace:
    """
    Get args from cli with validation.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Check docstrings by using pydocstyle and numpydoc"
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=pathlib.Path,
        help="Filenames or directories; in case of direstories perform recursive check",
    )
    parser.add_argument(
        "--add-ignore",
        nargs="*",
        default=[],
        help="Pydocstyle error codes; for example: D100,D100,D102",
    )
    parser.add_argument(
        "--disable-numpydoc",
        default=False,
        action="store_true",
        help="Determine if numpydoc checks are not needed",
    )
    args = parser.parse_args()
    check_args(args)
    return args


if __name__ == "__main__":
    args = get_args()
    monkeypatching()
    if not validate(args.paths, args.add_ignore, not args.disable_numpydoc):
        logging.error("INVALID DOCUMENTATION FOUND")
        exit(1)
    logging.info("SUCCESSFUL CHECK")
