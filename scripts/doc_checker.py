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
from numpydoc.validate import Docstring

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
        type_line = doc.doc_parameters[parameter][0]
        if "default: " not in type_line or "optional" in type_line:
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
    Check spelling of chosen words: "Modin", "pandas", "NumPy" in doc.

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
    "modin" word is treated as the constant.
    """
    if not doc.raw_doc:
        return []
    components = set(["Modin", "pandas", "NumPy"])
    check_words = "|".join(x.lower() for x in components)

    # comments work only with re.VERBOSE
    pattern = r"""
    (?:\W|^)                # non-capturing group of either non-word symbol or line start
    ({check_words})         # words to check, example - "modin|pandas|numpy"
    (?:                     # non-capturing group
        [^"\.\/\w\\]        # any symbol except: '"', '.', '\' and any from [a-zA-Z0-9_]
        | \.\s              # or '.' and any whitespace
        | \.$               # or '.' and end
        | $                 # or end
    )
    """.format(
        check_words=check_words
    )
    results = [
        set(re.findall(pattern, line, re.I | re.VERBOSE)) - components
        for line in doc.raw_doc.splitlines()
    ]

    errors = []
    for line_idx, words_in_line in enumerate(results):
        for word in words_in_line:
            reference = [x for x in components if x.lower() == word.lower()][0]
            errors.append(
                (
                    "MD02",
                    MODIN_ERROR_CODES["MD02"].format(
                        line=line_idx,
                        word=word,
                        reference=reference,
                    ),
                )
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
    for error in errors:
        results["errors"].append(error)
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

    # Documentation checking is performed by two tools: pydocstyle and numpydoc,
    # the checks of which are sometimes similar, but have different codes. In the
    # case when we want to disable one of the checks, we would like not to indicate
    # the codes of the two tools, because this is a duplication in meaning. To do
    # this, it is necessary to map the codes of the two tools, which is further
    # done for check of missing docstrings.

    # GL08 - missing docstring in an arbitary object; numpydoc code
    if err_code == "GL08":
        name = doc.name.split(".")[-1]
        magic = name.startswith("__") and name.endswith("__")
        # Codes of missing docstring for different types of objects; pydocstyle codes
        if inspect.isclass(doc.obj):
            # D101 - missing docstring in a class
            err_code = "D101"
        elif inspect.ismethod(doc.obj):
            # D102 - missing docstring in a method
            err_code = "D102"
        elif inspect.isfunction(doc.obj) and not magic:
            # D103 - missing docstring in a function
            err_code = "D103"
        elif inspect.isfunction(doc.obj) and magic:
            # D105 - missing docstring in a magic method
            err_code = "D105"

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
    noqa_str = None
    if not inspect.ismodule(doc.obj):
        # find last line of obj definition
        for line in source.split("\n"):
            if ")" in line and ":" in line.split(")", 1)[1]:
                noqa_str = line
                break
    else:
        # pydocstyle only check first line; aling with it
        noqa_str = source.split("\n", 1)[0]

    if "# noqa:" in noqa_str:
        noqa_checks = noqa_str.split("# noqa:", 1)[1].split(",")
    elif "# noqa" in noqa_str:
        noqa_checks = ["all"]
    else:
        noqa_checks = []
    return [check.strip() for check in noqa_checks]


# code snippet from numpydoc
def validate_object(import_path: str) -> bool:
    """
    Check docstrings of an entity that can be imported.

    Parameters
    ----------
    import_path : str
        Python-like import path.

    Returns
    -------
    is_successfull : bool
        Return True if all checks are successful.
    """
    from numpydoc.validate import validate

    is_successfull = True
    doc = Docstring(import_path)
    results = validate(import_path)
    results = validate_modin_error(doc, results)
    noqa_checks = get_noqa_checks(doc)
    for err_code, err_desc in results["errors"]:
        if (
            err_code not in NUMPYDOC_BASE_ERROR_CODES
            and err_code not in MODIN_ERROR_CODES
        ) or skip_check_if_noqa(doc, err_code, noqa_checks):
            continue
        is_successfull = False
        print(":".join([import_path, str(results["file_line"]), err_code, err_desc]))
    return is_successfull


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
        for f in files:
            if not f.endswith(".py"):
                continue

            current_path = os.path.join(root, f)
            # get importable name
            module_name = current_path.replace("/", ".").replace("\\", ".")
            module_name = module_name.replace(".py", "")

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

            print(f"NUMPYDOC OUTPUT FOR {current_path} - CAN BE EMPTY")
            # numpydoc docstrings validation
            # docstrings are taken dynamically
            to_validate = (
                [module_name]
                + [f"{module_name}.{x.name}" for x in (functions + classes)]
                + methods
            )
            if not all(list(map(validate_object, to_validate))):
                is_successfull = False
    return is_successfull


def pydocstyle_validate(path: pathlib.Path, add_ignore: List[str]) -> int:
    """
    Perform pydocstyle checks.

    Parameters
    ----------
    path : pathlib.Path
        Filename or directory path for check.
    add_ignore : List[int]
        `pydocstyle` error codes which are not verified.

    Returns
    -------
    bool
        Return True if all pydocstyle checks are successful.
    """
    result = subprocess.run(
        [
            "pydocstyle",
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
        print(f"PYDOCSTYLE OUTPUT FOR {path}\n", result.stdout)
    return True if result.returncode == 0 else False


def monkeypatching():
    """Monkeypatch decorators which change __doc__ attribute."""
    import ray
    import modin.utils

    ray.remote = lambda *args, **kwargs: args[0]

    modin.utils._inherit_docstrings = lambda *args, **kwargs: lambda cls: cls
    modin.utils._inherit_func_docstring = lambda *args, **kwargs: lambda func: func


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
        if not pydocstyle_validate(path, add_ignore):
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
        print("NOT SUCCESSFUL CHECK")
        exit(1)
    print("SUCCESSFUL CHECK")
