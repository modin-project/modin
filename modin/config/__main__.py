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
Content of this file should be executed if module `modin.config` is called.

If module is called (using `python -m modin.config`) configs help will be printed.
Using `-export_path` option configs description can be exported to the external CSV file
provided with this flag.
"""

from textwrap import dedent

from . import *  # noqa: F403, F401
from .pubsub import Parameter
import pandas
import argparse


def print_config_help():
    """Print configs help messages."""
    for objname in sorted(globals()):
        obj = globals()[objname]
        if isinstance(obj, type) and issubclass(obj, Parameter) and not obj.is_abstract:
            print(f"{obj.get_help()}\n\tCurrent value: {obj.get()}")  # noqa: T201


def export_config_help(filename: str):
    """
    Export all configs help messages to the CSV file.

    Parameters
    ----------
    filename : str
        Name of the file to export configs data.
    """
    configs_data = []
    for objname in sorted(globals()):
        obj = globals()[objname]
        if isinstance(obj, type) and issubclass(obj, Parameter) and not obj.is_abstract:
            data = {
                "Config Name": obj.__name__,
                "Env. Variable Name": getattr(
                    obj, "varname", "not backed by environment"
                ),
                "Default Value": obj._get_default()
                if obj.__name__ != "RayRedisPassword"
                else "random string",
                # `Notes` `-` underlining can't be correctly parsed inside csv table by sphinx
                "Description": dedent(obj.__doc__).replace("Notes\n-----", "Notes:\n"),
                "Options": obj.choices,
            }
            configs_data.append(data)

    pandas.DataFrame(
        configs_data,
        columns=[
            "Config Name",
            "Env. Variable Name",
            "Default Value",
            "Description",
            "Options",
        ],
    ).to_csv(filename, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--export-path",
        dest="export_path",
        type=str,
        required=False,
        default=None,
        help="File path to export configs data.",
    )
    export_path = parser.parse_args().export_path
    if export_path:
        export_config_help(export_path)
    else:
        print_config_help()
