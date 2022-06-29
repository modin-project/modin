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
Get supported APIs list by parsing docstrings in the Modin source files.

Example usage:
python scripts/supported_apis.py
"""

import pathlib
import os
import sys

import ast
from numpydoc.validate import Docstring
import pandas

MODIN_PATH = pathlib.Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(MODIN_PATH))

dataframe_supported_csv_path = (
    MODIN_PATH / "docs/supported_apis/dataframe_supported.csv"
)
series_supported_csv_path = MODIN_PATH / "docs/supported_apis/series_supported.csv"
io_supported_csv_path = MODIN_PATH / "docs/supported_apis/io_supported.csv"
utilities_supported_csv_path = (
    MODIN_PATH / "docs/supported_apis/utilities_supported.csv"
)
dataframe_path = pathlib.Path("modin/pandas/dataframe.py")
series_path = pathlib.Path("modin/pandas/series.py")
base_path = pathlib.Path("modin/pandas/base.py")
io_path = pathlib.Path("modin/pandas/io.py")
utilities_path = pathlib.Path("modin/pandas/general.py")

csv_columns = [
    "Method Name",
    "Parameter",
    "PandasOnRay",
    "PandasOnDask",
    "OmniSci",
    "Notes",
]

# these utilities just imported from pandas in `modin.pandas.__init__`
pandas_utilities = [
    "cut",
    "eval",
    "factorize",
    "test",
    "qcut",
    "date_range",
    "period_range",
    "bdate_range",
    "to_timedelta",
    "set_eng_float_format",
    "set_option",
    "array",
    "timedelta_range",
    "infer_freq",
    "interval_range",
]

pandas_utilities_parameters = ["pure pandas", "pure pandas", "pure pandas", ""]


def get_methods(path: pathlib.Path) -> list:
    """
    Get all functions and methods from the file passed via `path`.

    Parameters
    ----------
    path : pathlib.Path
        Path to the file to get functions and methods from.

    Returns
    -------
    list
        List with functions and methods from the passed file.
    """
    # get importable name
    module_name = str(path).replace("/", ".").replace("\\", ".")
    # remove ".py"
    module_name = os.path.splitext(module_name)[0]

    with open(MODIN_PATH / path) as fd:
        file_contents = fd.read()

    # using static parsing for collecting module, functions, classes and their methods
    module = ast.parse(file_contents)

    def is_public_func(node):
        return isinstance(node, ast.FunctionDef) and not node.name.startswith("_")

    functions = [
        f"{module_name}.{node.name}" for node in module.body if is_public_func(node)
    ]
    classes = [node for node in module.body if isinstance(node, ast.ClassDef)]
    methods = [
        f"{module_name}.{_class.name}.{node.name}"
        for _class in classes
        for node in _class.body
        if is_public_func(node)
    ]

    return methods + functions


def get_supported_params(path: pathlib.Path) -> dict:
    """
    Get supported parameters of functions and methods from the file passed via `path`.

    Parameters
    ----------
    path : pathlib.Path
        Path to the file to get functions and methods and information
        about their supportred parameters from.

    Returns
    -------
    apis : dict
        Dictionary with methods and supported parameters information.

    Notes
    -----
    Information about supported parameters should be put in the table with
    predefined format in the docstring `Extended Summary` section. Final
    docstring should be structured in accordance with the next template:
    ```
    <method/function summary>

    Parameters Support Status:

    +-----------------+-----------------+----------------+----------------+----------------+
    | Parameters      | PandasOnRay     | PandasOnDask   | OmniSci        | Notes          |
    +=================+=================+================+================+================+
    | All parameters  |                 |                |                |                |
    +-----------------+-----------------+----------------+----------------+----------------+
    |<parameter name> |                 |                |                |                |
    +-----------------+-----------------+----------------+----------------+----------------+
    ```
    """
    apis = {}
    methods = get_methods(path)
    for path in methods:
        method_name = path.split(".")[-1]
        supported_apis = []
        doc = Docstring(path).doc["Extended Summary"]
        contain_supported_info = len(doc) and any(
            "Parameters Support Status:" in line for line in doc
        )
        if contain_supported_info:
            table_begin_idx = doc.index("Parameters Support Status:") + 5
            for line_num in range(table_begin_idx, len(doc)):
                line = doc[line_num]
                if line.startswith("+-"):
                    break
                else:
                    data = list(map(lambda x: x.strip(), line.split("|")))[1:-1]
                    # handling the case when field splitted into
                    # several lines
                    if not data[0]:
                        for field_idx, field in enumerate(data):
                            if field:
                                # assuming notes on multiple lines as plain text
                                # without splits
                                if field_idx == 4:
                                    supported_apis[4] += " " + data[4]
                                # combine parameter execution properties with commas
                                else:
                                    supported_apis[field_idx] += ", " + data[field_idx]
                    else:
                        supported_apis = data

            if supported_apis[0] != "All parameters":
                raise ValueError(
                    "The first line of Parameters Notes table should contain summary information "
                    + f"about all parameters, but actually parameter described: {supported_apis[0]}"
                )
            apis[method_name] = supported_apis[1:]
        else:
            # use some default values until information about supported
            # parameters is not merged into master
            apis[method_name] = ["Partial", "Partial", "Partial", ""]

    return apis


def add_methods_links(apis: dict, method_prefix: str = None) -> dict:
    """
    Add links to the method names in the provided `apis` dictionary.

    Parameters
    ----------
    apis : dict
        Dictionary with methods and supported parameters information.
    method_prefix : str, optional
        Method prefix to put before method name in the method link.

    Returns
    -------
    apis_with_links : dict
        Dictionary with methods with links and supported parameters information.
    """
    apis_with_links = {}
    for method_name, params_info in apis.items():
        method_abs_path = (
            f"modin.pandas.{method_prefix}.{method_name}"
            if method_prefix
            else f"modin.pandas.{method_name}"
        )
        method_link = f":doc:`{method_name} </flow/modin/pandas/api/{method_abs_path}>`"
        apis_with_links[method_link] = params_info

    return apis_with_links


def generate_csv(csv_path: pathlib.Path, data: dict):
    """
    Generate csv with provided `data`.

    Parameters
    ----------
    csv_path : pathlib.Path
        CSV file path to generate.
    data : dict
        Data to put into the generated CSV file.
    """
    csv_data = []
    for method, params in data.items():
        param_data = {
            "Method Name": method,
            "PandasOnRay": params[0],
            "PandasOnDask": params[1],
            "OmniSci": params[2],
            "Notes": params[3],
        }
        csv_data.append(param_data)

    pandas.DataFrame(csv_data).to_csv(csv_path, index=False)


def generate_df_series_supportred_apis():
    """Generate CSV file with the supported parameters for DataFrame and Series."""

    def merge_df_series_base_params(base, other):
        """Add inherited methods from `base` to `other`."""
        other_api = other.copy()
        base_inherited = {
            method: params for method, params in base.items() if method not in other
        }

        other_api.update(base_inherited)

        return other_api

    base_pure_api = get_supported_params(base_path)
    dataframe_pure_api = get_supported_params(dataframe_path)
    series_pure_api = get_supported_params(series_path)

    dataframe_api = merge_df_series_base_params(base_pure_api, dataframe_pure_api)
    series_api = merge_df_series_base_params(base_pure_api, series_pure_api)

    dataframe_api = add_methods_links(dataframe_api, "DataFrame")
    series_api = add_methods_links(series_api, "Series")

    generate_csv(dataframe_supported_csv_path, dataframe_api)
    generate_csv(series_supported_csv_path, series_api)


def generate_io_supportred_apis():
    """Generate CSV file with the supported parameters for IO functions."""
    io_api = add_methods_links(get_supported_params(io_path))
    generate_csv(io_supported_csv_path, io_api)


def generate_utilities_supportred_apis():
    """Generate CSV file with the supported parameters for utility functions."""
    utilities_api = add_methods_links(get_supported_params(utilities_path))
    pandas_utilities_api = {
        method: pandas_utilities_parameters for method in pandas_utilities
    }
    utilities_api.update(pandas_utilities_api)
    generate_csv(utilities_supported_csv_path, utilities_api)


def generate_supported_apis_csvs():
    """Generate CSV files with the supported parameters for all needed functions and classes."""
    generate_df_series_supportred_apis()
    generate_io_supportred_apis()
    generate_utilities_supportred_apis()


if __name__ == "__main__":
    generate_supported_apis_csvs()
