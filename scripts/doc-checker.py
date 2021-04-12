"""
Example usage:
python scripts/doc-checker.py asv_bench/benchmarks/utils.py modin/pandas

Notes
-----
    * the script can take some paths to files or to directories
"""

import click
import subprocess
import os
import ast
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# error codes that pandas test in CI
# https://numpydoc.readthedocs.io/en/latest/validation.html#built-in-validation-checks
NUMPYDOC_BASE_ERR_CODE = {
    *("GL01", "GL02", "GL03", "GL04", "GL05", "GL06", "GL07", "GL09", "GL10"),
    *("SS02", "SS04", "SS05", "PR03", "PR04", "PR05", "PR10", "EX04", "RT01"),
    *("RT04", "RT05", "SA02", "SA03"),
}


# code snippet from numpydoc
def validate_object(import_path):
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


def numpydoc_validate(path):
    exit_status = 0
    for root, dirs, files in os.walk(path):
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


def pydocstyle_validate(path, add_ignore):
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


@click.command()
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--add-ignore", multiple=True, help="Pydocstyle errors")
@click.option("--use-numpydoc", type=bool, default=True)
def validate(paths, add_ignore, use_numpydoc):
    exit_status = 0
    for path in paths:
        if pydocstyle_validate(path, add_ignore):
            exit_status = 1
        if use_numpydoc:
            if numpydoc_validate(path):
                exit_status = 1
    if exit_status:
        exit(exit_status)
    print("SUCCESSFUL CHECK")


if __name__ == "__main__":
    validate()
