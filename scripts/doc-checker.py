"""
Example usage:
python scripts/doc-checker.py asv_bench/benchmarks/utils.py modin/pandas

Notes
-----
    * the script can take some pathes to files or to directories
"""

import click
import subprocess


def numpydoc_validate(path):
    import os
    import ast
    from numpydoc.validate import validate

    # error codes that pandas test in CI
    # https://numpydoc.readthedocs.io/en/latest/validation.html#built-in-validation-checks
    BASE_ERR_CODE = {
        *("GL01", "GL02", "GL03", "GL04", "GL05", "GL06", "GL07", "GL09", "GL10"),
        *("SS02", "SS04", "SS05", "PR03", "PR04", "PR05", "PR10", "EX04", "RT01"),
        *("RT04", "RT05", "SA02", "SA03"),
    }

    # Recursive handle directories
    if os.path.isdir(path):
        for f in os.listdir(path):
            if f.endswith(".py"):
                numpydoc_validate(os.path.join(path, f))
        return

    # get importable name
    module_name = path.replace("/", ".").replace("\\", ".")
    module_name = module_name.replace(".py", "")

    # code snippet from numpydoc
    def validate_object(import_path):
        exit_status = 0
        results = validate(import_path)
        for err_code, err_desc in results["errors"]:
            if err_code not in BASE_ERR_CODE:
                # filter
                continue
            exit_status += 1
            print(
                ":".join([import_path, str(results["file_line"]), err_code, err_desc])
            )
        return exit_status

    with open(path) as fd:
        file_contents = fd.read()

    # static parsing
    module = ast.parse(file_contents)

    print("NUMPYDOC OUTPUT - CAN BE EMPTY")
    # Validate module
    validate_object(module_name)

    # Validate functions
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    [
        validate_object(f"{module_name}.{x.name}")
        for x in functions
        if not x.name.startswith("__")
    ]

    # Validate classes
    classes = [node for node in module.body if isinstance(node, ast.ClassDef)]
    [validate_object(f"{module_name}.{x.name}") for x in classes]
    if classes == []:
        return

    # Validate class' methods
    methods = []
    for _class in classes:
        for method in [
            f"{module_name}.{_class.name}.{node.name}"
            for node in _class.body
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("__")
        ]:
            methods.append(method)
    [validate_object(x) for x in methods]


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
        print("PYDOCSTYLE OUTPUT\n", result.stdout)
    return result.returncode


@click.command()
@click.argument("pathes", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--add-ignore", multiple=True, help="Pydocstyle errors")
@click.option("--use-numpydoc", type=bool, default=True)
def validate(pathes, add_ignore, use_numpydoc):
    for path in pathes:
        pydocstyle_validate(path, add_ignore)
        if use_numpydoc:
            numpydoc_validate(path)


if __name__ == "__main__":
    validate()
