"""
Validate docstrings using pydocstyle and numpydoc.

Example usage:
python scripts/doc-checker.py asv_bench/benchmarks/utils.py modin/pandas

Notes
-----
    * the script can take some paths to files or to directories
"""

import argparse
import pathlib
import subprocess
import os
import ast
from typing import List

# error codes that pandas test in CI
# https://numpydoc.readthedocs.io/en/latest/validation.html#built-in-validation-checks
NUMPYDOC_BASE_ERR_CODE = {
    *("GL01", "GL02", "GL03", "GL04", "GL05", "GL06", "GL07", "GL09", "GL10"),
    *("SS02", "SS04", "SS05", "PR03", "PR04", "PR05", "PR10", "EX04", "RT01"),
    *("RT04", "RT05", "SA02", "SA03"),
}


# code snippet from numpydoc
def validate_object(import_path: str) -> int:
    """
    Check docstrings of an entity that can be imported.

    Parameters
    ----------
    import_path : str
        python-like import path

    Returns
    -------
    exit_status : int
    """
    from numpydoc.validate import validate

    exit_status = 0
    results = validate(import_path)
    for err_code, err_desc in results["errors"]:
        if err_code not in NUMPYDOC_BASE_ERR_CODE:
            # filter
            continue
        exit_status = 1
        print(":".join([import_path, str(results["file_line"]), err_code, err_desc]))
    return exit_status


def numpydoc_validate(path: pathlib.Path) -> int:
    """
    Perform numpydoc checks.

    Parameters
    ----------
    path : pathlib.Path

    Returns
    -------
    exit_status : int
    """
    exit_status = 0

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

            # using static parsing for collecting module, functions, classes and its methods
            module = ast.parse(file_contents)

            def is_public_func(node):
                return isinstance(node, ast.FunctionDef) and (
                    not node.name.startswith("__") or node.name.endswith("__")
                )

            functions = [node for node in module.body if is_public_func(node)]
            classes = [node for node in module.body if isinstance(node, ast.ClassDef)]
            methods = []
            for _class in classes:
                for node in _class.body:
                    if is_public_func(node):
                        methods.append(f"{module_name}.{_class.name}.{node.name}")

            print(f"NUMPYDOC OUTPUT FOR {current_path} - CAN BE EMPTY")
            # numpydoc docstrings validation
            # docstrings are taken dynamically
            to_validate = (
                [module_name]
                + [f"{module_name}.{x.name}" for x in (functions + classes)]
                + methods
            )
            if any(list(map(validate_object, to_validate))):
                exit_status = 1
    return exit_status


def pydocstyle_validate(path: pathlib.Path, add_ignore: List[str]) -> int:
    """
    Perform pydocstyle checks.

    Parameters
    ----------
    path : pathlib.Path
    add_ignore : List[int]

    Returns
    -------
    int
        Return code of pydocstyle subprocess.
    """
    result = subprocess.run(
        [
            "pydocstyle",
            "--convention",
            "numpy",
            "--add-ignore",
            ",".join(add_ignore),
            path,
        ],
        text=True,
        capture_output=True,
    )
    if result.returncode:
        print(f"PYDOCSTYLE OUTPUT FOR {path}\n", result.stdout)
    return result.returncode


def monkeypatching():
    """Monkeypatch decorators which incorrectly define __doc__ attribute."""
    import ray

    ray.remote = lambda *args, **kwargs: args[0]


def validate(
    paths: List[pathlib.Path], add_ignore: List[str], use_numpydoc: bool
) -> int:
    """
    Perform pydocstyle and numpydoc checks.

    Parameters
    ----------
    paths : List[pathlib.Path]
        filenames of directories for check
    add_ignore : List[str]
        pydocstyle error codes which are not verified
    use_numpydoc : bool
        check docstrings by numpydoc or no

    Returns
    -------
    exit_status : int
    """
    exit_status = 0
    for path in paths:
        if pydocstyle_validate(path, add_ignore):
            exit_status = 1
        if use_numpydoc:
            monkeypatching()
            if numpydoc_validate(path):
                exit_status = 1
    return exit_status


def check_args(args: argparse.Namespace):
    """
    Check the obtained values for correctness.

    Raises
    ------
    ValueError
        Occurs in case of non-existent files or directories.
    """
    for path in args.paths:
        if not path.exists():
            raise ValueError(f"{path} is not exist")


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
        "--use-numpydoc",
        type=bool,
        default=True,
        help="Check docstrings by numpydoc or no",
    )
    args = parser.parse_args()
    check_args(args)
    return args


if __name__ == "__main__":
    args = get_args()
    exit_status = validate(args.paths, args.add_ignore, args.use_numpydoc)

    if exit_status:
        print("NOT SUCCESSFUL CHECK")
        exit(exit_status)
    print("SUCCESSFUL CHECK")
